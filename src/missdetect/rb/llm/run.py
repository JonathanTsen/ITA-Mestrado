"""Etapa `llm`: o painel de LLM de um run (passos 42 e 43; pré-registro §4, H4; emendas P20–P22; 30/10/2026).

Fluxo, por modelo e braço:
1. **Painel** (`panel`): as instâncias do gerador principal nos 6 contextos de cada família do painel, mais a
   subamostra de 10% perguntada uma segunda vez.
2. **Perguntas** (`render`): A1 = tabela cega; A2 = os 28 descritores. Cada pergunta é gravada no caderno de respostas
   (`<diretório dos runs>/_llm_cache/<nome da config>/`), compartilhado entre runs da mesma config: a mesma pergunta nunca é
   paga duas vezes.
3. **Orçamento** (`budget`, P22): gasto do caderno + 1,5 × projeção ≤ US$ 300, antes de qualquer chamada.
4. **Chamadas** (`client`): síncronas ou em lote, na ordem sorteada das famílias; a repetição vem por último.
5. **Saídas** em `<run>/llm/<modelo>/<braço>/`, criadas de uma vez (diretório temporário renomeado no fim):
   - `predictions.parquet`: a resposta válida de cada instância, ou falha (probabilidades uniformes, classe −1).
     Empate no máximo é desfeito por sorteio com semente da instância;
   - `per_family.parquet`: o LLM e a configuração primária restrita às mesmas instâncias (média das 3 repetições da
     CV principal), que é a entrada do `stats-llm` (H4). Sem o `eval-symmetric`, a etapa faz as chamadas e para antes
     das saídas; repetida depois do `evaluate`, monta tudo sem pagar de novo;
   - `repeat.parquet` e `repeat_agreement.csv`: a segunda consulta e a concordância (κ de Cohen e diferença de
     macro-recall), descritivas;
   - `usage.csv` e `summary.json`: tokens, custo, planos, versões do modelo, janela das chamadas e o SHA-256 do caderno.

O log do run (`run.llm-<modelo>-<braço>.json`) é gravado por último e marca a etapa como concluída.
"""

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from missdetect.rb.config import RunConfig
from missdetect.rb.evaluate import metrics, summarize_by_family
from missdetect.rb.ids import child_rng, child_seed
from missdetect.rb.llm import budget, probes
from missdetect.rb.llm.cache import CLASSES, ResponseCache
from missdetect.rb.llm.client import (
    FREE,
    PAID,
    Backend,
    GeminiBackend,
    Runner,
    build_request,
    load_api_keys,
    require_authorization,
)
from missdetect.rb.llm.panel import family_order, panel_contexts, panel_families, panel_instances, repeat_subsample
from missdetect.rb.llm.render import (
    A1_SCHEMA,
    A1_SYSTEM,
    A2_SCHEMA,
    A2_SYSTEM,
    A3_SYSTEM,
    RENDERER_VERSION,
    neutral_description,
    render_a1,
    render_a2,
    render_a3,
)
from missdetect.rb.runs import append_run_log, read_observed, repo_root
from missdetect.rb.stats import PRIMARY

ARMS = {"A1": (A1_SYSTEM, A1_SCHEMA), "A2": (A2_SYSTEM, A2_SCHEMA), "A3": (A3_SYSTEM, A1_SCHEMA)}
THINKING_LEVEL = "medium"
MAX_OUTPUT_TOKENS = 32_768  # teto de segurança generoso; o adendo fixa o valor a partir do piloto (P20)
MODES = ("sync", "batch")
OPENML_METADATA = Path("data_rb/openml_metadata_20261022.csv")
OPENML_DESCRIPTIONS = Path("data_rb/openml_descriptions_20261030.csv")  # texto cru congelado para o A3
ALIASES = Path("data_rb/llm_dataset_aliases.csv")  # apelidos da sonda de nome (lista fechada, antes das sondas)


def cache_root(run_dir: Path, config: RunConfig) -> Path:
    """Ao lado dos runs (`<diretório dos runs>/_llm_cache/<config>`), qualquer que seja o diretório de trabalho."""
    return Path(run_dir).resolve().parent / "_llm_cache" / config.name


def method_name(arm: str) -> str:
    return f"llm_{arm.lower()}"


def build_panel(run_dir: Path, config: RunConfig) -> pd.DataFrame:
    """Instâncias do painel com a repetição (replicate = 1 na subamostra de 10%), na ordem de execução."""
    contexts = pd.read_parquet(run_dir / "contexts.parquet")
    families = panel_families(contexts.family_id.unique().tolist(), config.root_seed)
    chosen = panel_contexts(contexts, pd.read_parquet(run_dir / "bases.parquet"), families)
    panel = panel_instances(
        pd.read_parquet(run_dir / "instances.parquet"), pd.read_parquet(run_dir / "oracle.parquet"), chosen
    )
    order = {f: i for i, f in enumerate(family_order(families, config.root_seed))}
    panel = panel.assign(family_rank=panel.family_id.map(order), replicate=0)
    repeat = panel[panel.sample_id.isin(repeat_subsample(panel.sample_id, config.root_seed))].assign(replicate=1)
    out = pd.concat([panel, repeat], ignore_index=True)
    return out.sort_values(["replicate", "family_rank", "context_id", "sample_id"]).reset_index(drop=True)


def _descriptor_frame(run_dir: Path) -> pd.DataFrame:
    names = [f["name"] for f in json.loads((run_dir / "feature_schema.json").read_text())["features"]]
    return pd.read_parquet(run_dir / "features.parquet", columns=["sample_id", *names]).set_index("sample_id")


def neutral_descriptions(descriptions: Path, openml_ids: list[int]) -> dict[int, str]:
    """Descrição neutra do A3 por tabela-fonte, a partir do texto cru congelado; falta de descrição é erro."""
    texts = pd.read_csv(descriptions).set_index("openml_id")
    missing = [i for i in openml_ids if i not in texts.index or texts.loc[i, "status"] != "ok"]
    if missing:
        raise ValueError(f"A3 sem descrição do OpenML para as tabelas {missing} ({descriptions})")
    return {i: neutral_description(str(texts.loc[i, "description"]), str(texts.loc[i, "name"])) for i in openml_ids}


def register_requests(
    run_dir: Path, config: RunConfig, panel: pd.DataFrame, model: str, arm: str, descriptions: Path | None = None
) -> pd.DataFrame:  # fmt: skip
    """Renderiza e grava cada pergunta no caderno; devolve o painel com a chave de cada pergunta."""
    system, schema = ARMS[arm]
    cache = ResponseCache(cache_root(run_dir, config))
    descriptors = _descriptor_frame(run_dir) if arm == "A2" else None
    contexts = pd.read_parquet(run_dir / "contexts.parquet").set_index("context_id")
    neutral: dict[int, str] = {}
    if arm == "A3":
        ids = sorted({int(i) for i in contexts.loc[panel.context_id.unique(), "openml_id"]})
        neutral = neutral_descriptions(descriptions or repo_root() / OPENML_DESCRIPTIONS, ids)
    keys = []
    for row in panel.itertuples():
        seed = child_seed(config.root_seed, "llm_render", row.context_id)
        if arm == "A1":
            user = render_a1(read_observed(run_dir, row.sample_id), seed)
        elif arm == "A3":
            context = contexts.loc[row.context_id]
            user = render_a3(
                read_observed(run_dir, row.sample_id), seed, str(context.target), neutral[int(context.openml_id)]
            )
        else:
            user = render_a2(descriptors.loc[row.sample_id].to_dict())  # type: ignore[union-attr]
        request = build_request(
            arm=arm, model=model, system=system, user=user, schema=schema, thinking_level=THINKING_LEVEL,
            max_output_tokens=MAX_OUTPUT_TOKENS, renderer_version=RENDERER_VERSION, replicate=int(row.replicate),
        )  # fmt: skip
        keys.append(cache.register(request))
    return panel.assign(key=keys)


def _choose(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(np.isclose(probabilities, probabilities.max()))
    return int(best[0] if len(best) == 1 else rng.choice(best))


def predictions_from_cache(cache: ResponseCache, panel: pd.DataFrame, arm: str, root_seed: int) -> pd.DataFrame:
    rows = []
    for row in panel.itertuples():
        attempts = cache.attempts(row.key)
        record = next((a for a in attempts if a.get("valid")), None)
        base = {
            "method": method_name(arm), "repeat": 0, "fold": -1, "sample_id": row.sample_id,
            "family_id": int(row.family_id), "replicate": int(row.replicate),
            "y_true": CLASSES.index(row.mechanism), "attempts": len(attempts), "seconds": np.nan,
        }  # fmt: skip
        if record is None:
            p = np.full(3, 1 / 3)
            rows.append(
                {**base, "y_pred": -1, "status": "failed", **dict(zip([f"p_{c}" for c in CLASSES], p, strict=True))}
            )
            continue
        p = np.asarray(record["parsed"], dtype=float)
        rng = child_rng(root_seed, "llm_tie", row.sample_id, int(row.replicate))
        rows.append(
            {**base, "y_pred": _choose(p, rng), "status": "ok", **dict(zip([f"p_{c}" for c in CLASSES], p, strict=True)),
             "tier": record["tier"], "mode": record["mode"], "model_version": record["model_version"],
             "response_id": record["response_id"]}
        )  # fmt: skip
    return pd.DataFrame(rows)


def usage_table(cache: ResponseCache, panel: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for row in panel.itertuples():
        for a in cache.attempts(row.key):
            usage = a.get("usage") or {}
            rows.append(
                {"sample_id": row.sample_id, "replicate": int(row.replicate), "attempt": a["attempt"],
                 "valid": a["valid"], "invalid_reason": a.get("invalid_reason"), "tier": a["tier"], "mode": a["mode"],
                 "model_version": a["model_version"], "finish_reason": a.get("finish_reason"),
                 "requested_at": a["requested_at"], "received_at": a["received_at"],
                 "prompt_tokens": usage.get("prompt_token_count"), "output_tokens": usage.get("candidates_token_count"),
                 "thoughts_tokens": usage.get("thoughts_token_count"), "cost_usd": budget.response_cost(model, a)}
            )  # fmt: skip
    return pd.DataFrame(rows)


def repeat_agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    first = predictions[predictions.replicate == 0].set_index("sample_id")
    second = predictions[predictions.replicate == 1].set_index("sample_id")
    if second.empty:
        return pd.DataFrame()
    a, b = first.loc[second.index], second

    def macro(frame: pd.DataFrame) -> float:
        proba = frame[[f"p_{c}" for c in CLASSES]].to_numpy()
        return metrics(frame.y_true.to_numpy(), proba, frame.y_pred.to_numpy())["macro_recall"]

    with np.errstate(divide="ignore", invalid="ignore"):  # κ indefinido (NaN) se as duas consultas dão uma só classe
        kappa = float(cohen_kappa_score(a.y_pred, b.y_pred, labels=[-1, 0, 1, 2]))
    return pd.DataFrame(
        [{"n": len(b), "agreement": float(np.mean(a.y_pred.to_numpy() == b.y_pred.to_numpy())),
          "cohen_kappa": kappa,
          "macro_recall_first": macro(a), "macro_recall_second": macro(b),
          "macro_recall_diff": macro(b) - macro(a)}]
    )  # fmt: skip


def h4_per_family(run_dir: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    """LLM × configuração primária nas mesmas instâncias (a primária: média das 3 repetições da CV principal)."""
    llm = predictions[predictions.replicate == 0]
    primary = pd.read_parquet(run_dir / "eval-symmetric" / "predictions.parquet")
    primary = primary[(primary.method == PRIMARY) & primary.sample_id.isin(llm.sample_id)]
    columns = ["method", "repeat", "family_id", "sample_id", "y_true", "y_pred", *[f"p_{c}" for c in CLASSES]]
    per_family, _ = summarize_by_family(pd.concat([llm[columns], primary[columns]], ignore_index=True))
    return per_family


def _request_text(request: dict) -> str:
    user = request["user"] if isinstance(request["user"], str) else "\n\n".join(m["content"] for m in request["user"])
    return request["system"] + "\n\n" + user


def budget_gate(
    runner: Runner, run_dir: Path, config: RunConfig, keys: list[str], arm: str, mode: str, work: Path,
    today: date | None,
) -> Path:  # fmt: skip
    """Trava do orçamento (P22), antes de qualquer geração: conta os tokens de entrada do que falta perguntar
    (`countTokens` do Flash), projeta com a média de saída já observada e recusa se gasto + 1,5 × projeção > teto.
    Cada conferência é acrescentada a `budget.jsonl`."""
    root = cache_root(run_dir, config)
    backend, cache, model = runner.backend, runner.cache, runner.model
    counter = budget.TokenCounter(backend, root / "token_counts.json", PAID if PAID in backend.tiers() else FREE)
    counts = [counter.count(k, _request_text(cache.request(k))) for k in dict.fromkeys(runner.pending(keys))]
    mean_out = budget.observed_output_tokens(root, model, arm) or budget.DEFAULT_OUTPUT_TOKENS
    when = today or datetime.now(UTC).date()
    projected = budget.projection_usd(model, when, counts, mean_out, batch=mode == "batch")
    spent = budget.spent_usd(root)
    check = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"), "model": model, "arm": arm, "mode": mode,
        "n_pending": len(counts), "mean_input_tokens": float(np.mean(counts)) if counts else 0.0,
        "mean_output_tokens_assumed": mean_out, "projected_usd": projected, "spent_usd": spent,
        "margin": budget.MARGIN, "cap_usd": budget.CAP_USD, "prices": budget.PRICES[model],
    }  # fmt: skip
    path = work / "budget.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(check, default=str) + "\n")
    budget.check_budget(spent, projected)
    return path


def limited_summary(
    cache: ResponseCache, keys: list[str], model: str, arm: str, mode: str, limit: int, authorization: str | None,
    work: Path,
) -> Path:  # fmt: skip
    """Resumo de uso de uma execução parcial (`--limit`): validade, tokens, custo e versões, sem predições."""
    attempts = [a for k in keys for a in cache.attempts(k)]
    usage = [a.get("usage") or {} for a in attempts]

    def mean(field: str) -> float | None:
        values = [u.get(field) or 0 for u in usage]
        return float(np.mean(values)) if values else None

    summary = {
        "model": model, "arm": arm, "mode": mode, "limit": limit, "n_questions": len(keys),
        "n_valid": sum(cache.first_valid(k) is not None for k in keys), "n_calls": len(attempts),
        "n_invalid_attempts": sum(not a.get("valid") for a in attempts),
        "cost_usd": float(sum(budget.response_cost(model, a) for a in attempts)),
        "mean_prompt_tokens": mean("prompt_token_count"), "mean_thoughts_tokens": mean("thoughts_token_count"),
        "max_thoughts_tokens": max((u.get("thoughts_token_count") or 0 for u in usage), default=None),
        "mean_output_tokens": mean("candidates_token_count"),
        "model_versions": sorted({str(a["model_version"]) for a in attempts if a.get("model_version")}),
        "authorization": authorization, "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    path = work / f"limit-{limit}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_llm(
    run_dir: Path, config: RunConfig, *, model: str, arm: str, mode: str, authorization: str | None = None,
    env_file: Path | None = None, max_workers: int = 4, budget_only: bool = False, backend: Backend | None = None,
    sleep: Callable[[float], None] = time.sleep, today: date | None = None, limit: int | None = None,
    descriptions: Path | None = None,
) -> Path:  # fmt: skip
    """Com `limit`, pergunta só as N primeiras perguntas distintas, na ordem de execução, e grava só um resumo de uso
    em `<run>/llm/<modelo>/_work/<braço>/limit-<N>.json`, sem as saídas finais (o teste de ≈ 30 chamadas do Argon)."""
    if arm not in ARMS or mode not in MODES:
        raise ValueError(f"braço {arm} ou modo {mode} desconhecido")
    require_authorization(model, authorization)  # antes de abrir o cliente: o Argon nunca chama sem autorização
    run_dir = Path(run_dir)
    out_dir = run_dir / "llm" / model / arm
    log = run_dir / f"run.llm-{model}-{arm}.json"
    if out_dir.exists():  # etapa concluída; se a queda foi entre a renomeação e o log, o log é refeito do resumo
        if not log.exists():
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            append_run_log(
                run_dir, config, f"llm-{model}-{arm}", {k: v for k, v in summary.items() if k != "model_versions"}
            )
        return out_dir
    work = run_dir / "llm" / model / "_work" / arm
    work.mkdir(parents=True, exist_ok=True)
    panel = register_requests(run_dir, config, build_panel(run_dir, config), model, arm, descriptions)
    cache = ResponseCache(cache_root(run_dir, config))
    if backend is None:
        backend = GeminiBackend(load_api_keys(env_file or repo_root() / ".env"))
    runner = Runner(
        backend, cache, model, authorization=authorization, max_workers=max_workers, events=work / "events.jsonl",
        sleep=sleep,
    )  # fmt: skip
    keys = list(dict.fromkeys(panel.key))
    if limit is not None:
        keys = keys[:limit]
    gate = budget_gate(runner, run_dir, config, keys, arm, mode, work, today)
    if budget_only:
        return gate
    if mode == "sync":
        runner.run_sync(keys)
    else:
        runner.run_batch(keys, work / "jobs.jsonl", f"rb-{config.name}-{arm}")
    if limit is not None:
        return limited_summary(cache, keys, model, arm, mode, limit, authorization, work)
    if not (run_dir / "eval-symmetric" / "predictions.parquet").exists():
        note = work / "aguardando-eval-symmetric.txt"
        note.write_text(
            "As respostas estão no caderno. A tabela da H4 precisa das predições da CV principal: rode o evaluate e "
            "repita este comando (nenhuma pergunta é paga de novo).\n",
            encoding="utf-8",
        )
        return note
    predictions = predictions_from_cache(cache, panel, arm, config.root_seed)
    usage = usage_table(cache, panel, model)
    main = predictions[predictions.replicate == 0]
    summary = {
        "model": model, "arm": arm, "mode": mode, "thinking_level": THINKING_LEVEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "renderer_version": RENDERER_VERSION,
        "n_instances": len(main), "n_families": int(main.family_id.nunique()),
        "n_valid": int((main.status == "ok").sum()), "n_failed": int((main.status == "failed").sum()),
        "n_repeat": int((predictions.replicate == 1).sum()), "n_calls": len(usage),
        "n_invalid_attempts": int((~usage.valid).sum()) if len(usage) else 0,
        "tiers": usage.tier.value_counts().to_dict() if len(usage) else {},
        "model_versions": sorted(usage.model_version.dropna().unique().tolist()) if len(usage) else [],
        "first_request": usage.requested_at.min() if len(usage) else None,
        "last_response": usage.received_at.max() if len(usage) else None,
        "cost_usd": float(usage.cost_usd.sum()) if len(usage) else 0.0,
        "mean_prompt_tokens": float(usage.prompt_tokens.mean()) if len(usage) else None,
        "mean_thoughts_tokens": float(usage.thoughts_tokens.mean()) if len(usage) else None,
        "mean_output_tokens": float(usage.output_tokens.mean()) if len(usage) else None,
        "cache_digest": cache.digest(), "authorization": authorization,
    }  # fmt: skip
    for stale in (run_dir / "llm" / model).glob(f".tmp-{arm}-*"):  # sobras de uma execução interrompida
        shutil.rmtree(stale)
    tmp = run_dir / "llm" / model / f".tmp-{arm}-{os.getpid()}"
    tmp.mkdir(parents=True)
    main.to_parquet(tmp / "predictions.parquet", index=False)
    predictions[predictions.replicate == 1].to_parquet(tmp / "repeat.parquet", index=False)
    repeat_agreement(predictions).to_csv(tmp / "repeat_agreement.csv", index=False)
    usage.to_csv(tmp / "usage.csv", index=False)
    h4_per_family(run_dir, predictions).to_parquet(tmp / "per_family.parquet", index=False)
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.rename(out_dir)
    append_run_log(run_dir, config, f"llm-{model}-{arm}", {k: v for k, v in summary.items() if k != "model_versions"})
    return out_dir


def run_probes(
    run_dir: Path, config: RunConfig, *, model: str, mode: str, authorization: str | None = None,
    env_file: Path | None = None, max_workers: int = 4, budget_only: bool = False, backend: Backend | None = None,
    sleep: Callable[[float], None] = time.sleep, today: date | None = None, metadata: Path | None = None,
    aliases: Path | None = None,
) -> Path:  # fmt: skip
    """Sondas de memorização (P21) nas famílias do painel: tabela crua (covariável) e renderização cega do A1 (critério
    de exclusão da H4). Saídas em `<run>/llm/<modelo>/probes/`: `results.csv`, `excluded_families.csv` e `summary.json`.
    """
    if mode not in MODES:
        raise ValueError(f"modo {mode} desconhecido")
    require_authorization(model, authorization)
    run_dir = Path(run_dir)
    out_dir = run_dir / "llm" / model / "probes"
    if out_dir.exists():
        return out_dir
    work = run_dir / "llm" / model / "_work" / "probes"
    work.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(cache_root(run_dir, config))
    tables = probes.probe_tables(
        run_dir, build_panel(run_dir, config), Path(config.sources.csv_dir),
        pd.read_csv(metadata or repo_root() / OPENML_METADATA), work, config.root_seed,
        pd.read_csv(alias_file) if (alias_file := aliases or repo_root() / ALIASES).exists() else None,
    )  # fmt: skip

    def battery(llm: probes.ProbeLLM) -> list[dict]:
        rows = []
        with (work / "tabmemcheck.log").open("a", encoding="utf-8") as log:
            for row in tables.itertuples():
                seed = child_seed(config.root_seed, "llm_probe", row.family_id, row.table) % 2**31
                result = probes.run_battery(row.path, llm, seed, list(row.names), log)
                rows.append({"family_id": row.family_id, "openml_id": row.openml_id, "name": row.name,
                             "table": row.table, **result})  # fmt: skip
        return rows

    recorder = probes.ProbeLLM(cache, model)
    battery(recorder)  # passada 1: só grava as perguntas
    keys = list(dict.fromkeys(recorder.keys))
    if backend is None:
        backend = GeminiBackend(load_api_keys(env_file or repo_root() / ".env"))
    runner = Runner(
        backend, cache, model, authorization=authorization, validator=probes.text_validator, max_workers=max_workers,
        events=work / "events.jsonl", sleep=sleep,
    )  # fmt: skip
    gate = budget_gate(runner, run_dir, config, keys, "probe", mode, work, today)
    if budget_only:
        return gate
    if mode == "sync":
        runner.run_sync(keys)
    else:
        runner.run_batch(keys, work / "jobs.jsonl", f"rb-{config.name}-probes")
    reader = probes.ProbeLLM(cache, model)
    results = pd.DataFrame(battery(reader))  # passada 2: lê as respostas
    if list(dict.fromkeys(reader.keys)) != keys:
        raise RuntimeError("as perguntas das sondas mudaram entre as passadas; as sementes não fixaram o tabmemcheck")
    failed = [k for k in keys if cache.first_valid(k) is None]
    excluded = results[(results.table == "blind") & results.recognized][["family_id", "openml_id", "name"]]
    usage = [a for k in keys for a in cache.attempts(k)]
    summary = {
        "model": model, "mode": mode, "thinking_level": probes.THINKING_LEVEL, "probes_version": probes.PROBES_VERSION,
        "thresholds": probes.THRESHOLDS, "aliases": str(aliases or repo_root() / ALIASES),
        "n_tables": len(results), "n_questions": len(keys),
        "n_failed_questions": len(failed), "n_calls": len(usage),
        "cost_usd": float(sum(budget.response_cost(model, a) for a in usage)),
        "recognized_raw": int(results[results.table == "raw"].recognized.sum()),
        "recognized_blind": len(excluded), "excluded_families": excluded.family_id.tolist(),
        "excluded_sha256": hashlib.sha256(excluded.to_csv(index=False).encode()).hexdigest(),
        "cache_digest": cache.digest(), "authorization": authorization,
    }  # fmt: skip
    for stale in (run_dir / "llm" / model).glob(".tmp-probes-*"):
        shutil.rmtree(stale)
    tmp = run_dir / "llm" / model / f".tmp-probes-{os.getpid()}"
    tmp.mkdir(parents=True)
    results.to_csv(tmp / "results.csv", index=False)
    excluded.to_csv(tmp / "excluded_families.csv", index=False)
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.rename(out_dir)
    append_run_log(run_dir, config, f"llm-probes-{model}", summary)
    return out_dir
