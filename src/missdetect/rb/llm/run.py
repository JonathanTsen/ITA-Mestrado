"""Etapa `llm`: o painel de LLM de um run (passos 42 e 43; pré-registro §4, H4; emendas P20–P22; 30/10/2026).

Fluxo, por modelo e braço:
1. **Painel** (`panel`): as instâncias do gerador principal nos 6 contextos de cada família do painel, mais a
   subamostra de 10% perguntada uma segunda vez.
2. **Perguntas** (`render`): A1 = tabela cega; A2 = os 28 descritores. Cada pergunta é gravada no caderno de respostas
   (`<output_root>/_llm_cache/<nome da config>/`), compartilhado entre runs da mesma config: a mesma pergunta nunca é
   paga duas vezes.
3. **Orçamento** (`budget`, P22): gasto do caderno + 1,5 × projeção ≤ US$ 300, antes de qualquer chamada.
4. **Chamadas** (`client`): síncronas ou em lote, na ordem sorteada das famílias; a repetição vem por último.
5. **Saídas** em `<run>/llm/<modelo>/<braço>/`, criadas de uma vez (diretório temporário renomeado no fim):
   - `predictions.parquet`: a resposta válida de cada instância, ou falha (probabilidades uniformes, classe −1).
     Empate no máximo é desfeito por sorteio com semente da instância;
   - `per_family.parquet`: o LLM e a configuração primária restrita às mesmas instâncias (média das 3 repetições da
     CV principal), que é a entrada do `stats-llm` (H4). Só sai se o `eval-symmetric` já existir;
   - `repeat.parquet` e `repeat_agreement.csv`: a segunda consulta e a concordância (κ de Cohen e diferença de
     macro-recall), descritivas;
   - `usage.csv` e `summary.json`: tokens, custo, planos, versões do modelo, janela das chamadas e o SHA-256 do caderno.

O log do run (`run.llm-<modelo>-<braço>.json`) é gravado por último e marca a etapa como concluída.
"""

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
from missdetect.rb.llm import budget
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
from missdetect.rb.llm.render import A1_SCHEMA, A1_SYSTEM, A2_SCHEMA, A2_SYSTEM, RENDERER_VERSION, render_a1, render_a2
from missdetect.rb.runs import append_run_log, read_observed, repo_root
from missdetect.rb.stats import PRIMARY

ARMS = {"A1": (A1_SYSTEM, A1_SCHEMA), "A2": (A2_SYSTEM, A2_SCHEMA)}
THINKING_LEVEL = "medium"
MAX_OUTPUT_TOKENS = 32_768  # teto de segurança generoso; o adendo fixa o valor a partir do piloto (P20)
MODES = ("sync", "batch")


def cache_root(config: RunConfig) -> Path:
    return Path(config.output_root) / "_llm_cache" / config.name


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


def register_requests(run_dir: Path, config: RunConfig, panel: pd.DataFrame, model: str, arm: str) -> pd.DataFrame:
    """Renderiza e grava cada pergunta no caderno; devolve o painel com a chave de cada pergunta."""
    system, schema = ARMS[arm]
    cache = ResponseCache(cache_root(config))
    descriptors = _descriptor_frame(run_dir) if arm == "A2" else None
    keys = []
    for row in panel.itertuples():
        if arm == "A1":
            seed = child_seed(config.root_seed, "llm_render", row.context_id)
            user = render_a1(read_observed(run_dir, row.sample_id), seed)
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


def h4_per_family(run_dir: Path, predictions: pd.DataFrame) -> pd.DataFrame | None:
    """LLM × configuração primária nas mesmas instâncias (a primária: média das 3 repetições da CV principal)."""
    path = run_dir / "eval-symmetric" / "predictions.parquet"
    if not path.exists():
        return None
    llm = predictions[predictions.replicate == 0]
    primary = pd.read_parquet(path)
    primary = primary[(primary.method == PRIMARY) & primary.sample_id.isin(llm.sample_id)]
    columns = ["method", "repeat", "family_id", "sample_id", "y_true", "y_pred", *[f"p_{c}" for c in CLASSES]]
    per_family, _ = summarize_by_family(pd.concat([llm[columns], primary[columns]], ignore_index=True))
    return per_family


def run_llm(
    run_dir: Path, config: RunConfig, *, model: str, arm: str, mode: str, authorization: str | None = None,
    env_file: Path | None = None, max_workers: int = 4, budget_only: bool = False, backend: Backend | None = None,
    sleep: Callable[[float], None] = time.sleep, today: date | None = None,
) -> Path:  # fmt: skip
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
    panel = register_requests(run_dir, config, build_panel(run_dir, config), model, arm)
    cache = ResponseCache(cache_root(config))
    if backend is None:
        backend = GeminiBackend(load_api_keys(env_file or repo_root() / ".env"))
    runner = Runner(
        backend, cache, model, authorization=authorization, max_workers=max_workers, events=work / "events.jsonl",
        sleep=sleep,
    )  # fmt: skip
    pending = runner.pending(panel.key)
    counter = budget.TokenCounter(
        backend, cache_root(config) / "token_counts.json", PAID if PAID in backend.tiers() else FREE
    )
    system = ARMS[arm][0]
    counts = [counter.count(k, system + "\n\n" + cache.request(k)["user"]) for k in dict.fromkeys(pending)]
    mean_out = budget.observed_output_tokens(cache_root(config), model, arm) or budget.DEFAULT_OUTPUT_TOKENS
    when = today or datetime.now(UTC).date()
    projected = budget.projection_usd(model, when, counts, mean_out, batch=mode == "batch")
    spent = budget.spent_usd(cache_root(config))
    check = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"), "model": model, "arm": arm, "mode": mode,
        "n_pending": len(counts), "mean_input_tokens": float(np.mean(counts)) if counts else 0.0,
        "mean_output_tokens_assumed": mean_out, "projected_usd": projected, "spent_usd": spent,
        "margin": budget.MARGIN, "cap_usd": budget.CAP_USD, "prices": budget.PRICES[model],
    }  # fmt: skip
    with (work / "budget.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(check, default=str) + "\n")
    if budget_only:
        return work / "budget.jsonl"
    budget.check_budget(spent, projected)
    keys = list(dict.fromkeys(panel.key))
    if mode == "sync":
        runner.run_sync(keys)
    else:
        runner.run_batch(keys, work / "jobs.jsonl", f"rb-{config.name}-{arm}")
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
    per_family = h4_per_family(run_dir, predictions)
    if per_family is not None:
        per_family.to_parquet(tmp / "per_family.parquet", index=False)
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.rename(out_dir)
    append_run_log(run_dir, config, f"llm-{model}-{arm}", {k: v for k, v in summary.items() if k != "model_versions"})
    return out_dir
