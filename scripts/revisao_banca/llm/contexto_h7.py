"""H7 confirmatória: o conjunto de informações (valores originais + contexto com faixas típicas aproximadas, S7) muda o
acerto do LLM em relação à tabela cega (S0)? Adendo LLM, §10; inclusão decidida pelo autor em 03/11/2026.

Desenho congelado no adendo: 30 bases inventadas novas (`contexto_h7_dominios.py`, IDs 990201–990230, nenhuma das 16
de desenvolvimento), o apagamento do benchmark (18 instâncias por base, 6 de cada mecanismo), os braços S0 e S7 de
`contexto_sintetico.prompts` nas mesmas linhas e colunas, um único contraste S7 × S0 (sem H7b), família própria com
α = 0,05 fora do Holm de H1–H6. Análise por base: d = macro-recall(S7) − macro-recall(S0); Wilcoxon-Pratt bilateral,
Hodges–Lehmann com IC por bootstrap de bases; McNemar e recalls por classe só descritivos.

Travas: com o papel `confirm` (`configs/rb/contexto-h7.yaml`), o `inject` só roda no commit da tag `prereg-v1` e o
`perguntar` só no da `prereg-v1-llm` (`runs.confirmatory_guard`); modelos além do Flash exigem a autorização do autor;
a trava do orçamento (P22) roda antes de qualquer chamada. O `gerar` roda antes das tags, e o manifesto com os SHA-256
das bases e dos contextos vai para o adendo.

Uso (no ITA-Mestrado):
  uv run --frozen python scripts/revisao_banca/llm/contexto_h7.py gerar data_rb/contexto_h7
  uv run --frozen python -m missdetect.rb.cli inject --config configs/rb/contexto-h7.yaml        # worktree prereg-v1
  uv run --frozen python scripts/revisao_banca/llm/contexto_h7.py perguntar <run> --model <m> [--authorization ...]
  uv run --frozen python scripts/revisao_banca/llm/contexto_h7.py analisar <run> <saída> --model <m>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from contexto_h7_dominios import DOMAINS_H7
from contexto_sintetico import _mcnemar, _panel_with_keys, gerar_bases

from missdetect.rb.config import load_config
from missdetect.rb.evaluate import metrics, summarize_by_family
from missdetect.rb.llm.cache import CLASSES, ResponseCache
from missdetect.rb.llm.client import GeminiBackend, Runner, load_api_keys, require_authorization
from missdetect.rb.llm.run import budget_gate, cache_root, predictions_from_cache
from missdetect.rb.runs import confirmatory_guard, repo_root
from missdetect.rb.stats import bootstrap_hl_ci, clean_differences, hodges_lehmann, rank_biserial, wilcoxon_p

ARMS = ("S0", "S7")
ALPHA = 0.05
CONFIG = Path("configs/rb/contexto-h7.yaml")
DOMAINS_FILE = Path(__file__).with_name("contexto_h7_dominios.py")


def gerar(out: Path) -> None:
    gerar_bases(out, DOMAINS_H7, "h7", "confirm", DOMAINS_FILE)


def _guard(config) -> None:
    if "confirm" in config.sources.roles:
        confirmatory_guard(repo_root(), "llm")


def perguntar(run_dir: Path, config_path: Path, model: str, authorization: str | None, budget_only: bool) -> None:
    config = load_config(config_path)
    _guard(config)
    require_authorization(model, authorization)
    panels = _panel_with_keys(run_dir, config, ARMS, model)
    cache = ResponseCache(cache_root(run_dir, config))
    backend = GeminiBackend(load_api_keys(repo_root() / ".env"))
    work = run_dir / "h7" / model / "_work"
    work.mkdir(parents=True, exist_ok=True)
    runner = Runner(backend, cache, model, authorization=authorization, events=work / "events.jsonl")
    keys = list(dict.fromkeys(k for arm in ARMS for k in panels[arm].key))
    gate = budget_gate(runner, run_dir, config, keys, "ctx", "batch", work, None)
    print("projeção:", gate.read_text().splitlines()[-1])
    if budget_only:
        return
    runner.run_batch(keys, work / "jobs.jsonl", f"rb-{config.name}-{model}")
    print("respostas no caderno:", sum(cache.first_valid(k) is not None for k in keys), "de", len(keys))


def analisar_h7(s0: pd.DataFrame, s7: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """O teste confirmatório da H7 a partir das predições dos dois braços (uma linha por instância, mesmas instâncias).

    Devolve o resumo e a tabela por base (macro-recall de cada braço e d = S7 − S0)."""
    if sorted(s0.sample_id) != sorted(s7.sample_id):
        raise ValueError("S0 e S7 precisam ter as mesmas instâncias")
    per = {}
    for arm, pred in (("S0", s0), ("S7", s7)):
        per_family, _ = summarize_by_family(pred)
        per[arm] = per_family.set_index("family_id").macro_recall
    by_family = pd.DataFrame({"macro_recall_S0": per["S0"], "macro_recall_S7": per["S7"]})
    by_family["d"] = clean_differences(by_family.macro_recall_S7 - by_family.macro_recall_S0)  # desvio D4
    d = by_family.d.to_numpy()
    low, high = bootstrap_hl_ci(d)
    p_value = wilcoxon_p(d)
    order = s0.sort_values("sample_id"), s7.sort_values("sample_id")
    correct = [(p.y_pred.to_numpy() == p.y_true.to_numpy()) for p in order]
    pooled = {}
    for arm, pred in (("S0", s0), ("S7", s7)):
        m = metrics(pred.y_true.to_numpy(), pred[[f"p_{c}" for c in CLASSES]].to_numpy(), pred.y_pred.to_numpy())
        pooled |= {f"{k}_{arm}": float(m[k]) for k in ("macro_recall", "recall_MCAR", "recall_MAR", "recall_MNAR")}
        pooled[f"falhas_{arm}"] = int((pred.status == "failed").sum())
    if p_value < ALPHA:
        leitura = "S7 acima de S0" if np.median(d) > 0 else "S7 abaixo de S0"
    else:
        leitura = "evidência insuficiente de diferença (não é equivalência)"
    summary = {
        "n_bases": len(d), "n_instancias": len(s0), "media_S0": float(by_family.macro_recall_S0.mean()),
        "media_S7": float(by_family.macro_recall_S7.mean()), "dif_media": float(d.mean()),
        "hodges_lehmann": hodges_lehmann(d), "ic95_baixo": low, "ic95_alto": high,
        "p_wilcoxon_pratt": p_value, "rank_biserial": rank_biserial(d), "alpha": ALPHA,
        "bases_S7_melhor": int((d > 0).sum()), "bases_S7_pior": int((d < 0).sum()), "bases_empate": int((d == 0).sum()),
        "leitura": leitura, **pooled, **{f"descritivo_{k}": v for k, v in _mcnemar(*correct).items()},
    }  # fmt: skip
    return summary, by_family.reset_index()


def analisar(run_dir: Path, config_path: Path, model: str, out: Path) -> None:
    config = load_config(config_path)
    panels = _panel_with_keys(run_dir, config, ARMS, model)
    cache = ResponseCache(cache_root(run_dir, config))
    preds = {arm: predictions_from_cache(cache, panels[arm], arm, config.root_seed) for arm in ARMS}
    summary, by_family = analisar_h7(preds["S0"], preds["S7"])
    summary |= {"modelo": model, "config": config.name, "run": str(run_dir)}
    out.mkdir(parents=True, exist_ok=False)
    for arm, pred in preds.items():
        pred.to_parquet(out / f"predicoes_{arm}.parquet", index=False)
    by_family.to_csv(out / "h7_por_base.csv", index=False)
    (out / "h7_resultado.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gerar")
    p.add_argument("out", type=Path)
    for name in ("perguntar", "analisar"):
        p = sub.add_parser(name)
        p.add_argument("run", type=Path)
        if name == "analisar":
            p.add_argument("out", type=Path)
        p.add_argument("--config", type=Path, default=CONFIG)
        p.add_argument("--model", required=True, help="alias do modelo (o confirmatório pela regra do adendo, §1)")
        if name == "perguntar":
            p.add_argument(
                "--authorization", default=None, help="autorização do autor (obrigatória para modelos ≠ Flash)"
            )
            p.add_argument("--budget-only", action="store_true", help="só conta tokens e projeta o custo")
    args = parser.parse_args()
    if args.command == "gerar":
        gerar(args.out)
    elif args.command == "perguntar":
        perguntar(args.run, args.config, args.model, args.authorization, args.budget_only)
    else:
        analisar(args.run, args.config, args.model, args.out)


if __name__ == "__main__":
    main()
