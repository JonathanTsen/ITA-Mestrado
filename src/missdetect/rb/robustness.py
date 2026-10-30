"""Ensaios de robustez do passo 40 (exploratórios; pré-registro §6), a partir das predições gravadas (30/10/2026).

**Prior-shift** (item 5 do passo 40; ANA-080, ORA-16/22). O benchmark é balanceado; na prática, as prevalências não são.
- `expected_accuracy`: a acurácia esperada sob uma prevalência π é Σ_c π_c · recall_c. Os recalls são os do teste
  balanceado, por família (média das repetições); falha conta como erro.
- `saerens_adjust`: correção de priors de Saerens, Latinne e Decaestecker (2002) com a prevalência conhecida,
  p'(c | x) ∝ p(c | x) · π'_c / π_c, seguida da classe de maior probabilidade. Mostra quanto se recupera ajustando a
  decisão à prevalência.
- `prior_shift`: as duas decisões (`argmax` e `saerens`) em cenários nomeados e numa grade do simplex, com a média
  entre famílias e IC95% por bootstrap de famílias.
- `rare_mcar`: o L1 quando o MCAR é raro. Recall do MCAR e especificidade (não-MCAR chamado de não-MCAR) vêm do teste
  balanceado; a precisão do MCAR sob a prevalência π é π·rec / (π·rec + (1 − π)·(1 − esp)).

Nenhum p-valor: tudo aqui é descritivo, e nenhuma frase de "limite" ou "teto" (passo 40, aceite).
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from missdetect.rb.config import RunConfig
from missdetect.rb.models import CLASSES
from missdetect.rb.runs import append_run_log, write_tables_atomically
from missdetect.rb.stats import PRIMARY, bootstrap_mean_ci

TRAIN_PRIOR = (1 / 3, 1 / 3, 1 / 3)
SCENARIOS: dict[str, tuple[float, float, float]] = {
    "balanceada": TRAIN_PRIOR,
    "realista (MCAR 10%, MAR 50%, MNAR 40%)": (0.10, 0.50, 0.40),
    "MCAR raro (5%)": (0.05, 0.475, 0.475),
    "MNAR raro (10%)": (0.45, 0.45, 0.10),
    "MAR dominante (60%)": (0.20, 0.60, 0.20),
}
MCAR_PREVALENCES = (0.05, 0.10, 0.20, 1 / 3)
METHODS = ("direct_hgb", "hier_hgb_fullprob", "logit_simple", "mechdetect_obs")


def _mean_ci(values: np.ndarray) -> dict:
    low, high = bootstrap_mean_ci(values)
    return {"mean": float(np.mean(values)), "ci_low": low, "ci_high": high}


def simplex_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    n = round(1 / step)
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


def _recalls(pred: pd.DataFrame) -> pd.DataFrame:
    """Recall por classe, por método, família e repetição (falha, y_pred = −1, conta como erro)."""
    hit = pred.y_pred.to_numpy() == pred.y_true.to_numpy()
    frame = pred.assign(hit=hit)
    table = frame.groupby(["method", "family_id", "repeat", "y_true"]).hit.mean().unstack("y_true")
    table.columns = [f"recall_{CLASSES[int(c)]}" for c in table.columns]
    return table.reset_index()


def expected_accuracy(pred: pd.DataFrame, prior: tuple[float, float, float]) -> pd.DataFrame:
    """Acurácia esperada sob a prevalência `prior`, por método e família (média das repetições)."""
    rec = _recalls(pred)
    weights = np.asarray(prior, dtype=float)
    rec["accuracy"] = rec[[f"recall_{c}" for c in CLASSES]].to_numpy() @ weights
    return rec.groupby(["method", "family_id"], as_index=False).accuracy.mean()


def saerens_adjust(proba: np.ndarray, train_prior: Iterable[float], new_prior: Iterable[float]) -> np.ndarray:
    ratio = np.asarray(tuple(new_prior), dtype=float) / np.asarray(tuple(train_prior), dtype=float)
    adjusted = np.asarray(proba, dtype=float) * ratio
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _saerens_pred(pred: pd.DataFrame, prior: tuple[float, float, float]) -> pd.DataFrame:
    proba = pred[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=float)
    adjusted = saerens_adjust(proba, TRAIN_PRIOR, prior).argmax(axis=1)
    return pred.assign(y_pred=np.where(pred.y_pred.to_numpy() < 0, -1, adjusted))  # falha continua falha


def prior_shift(
    pred: pd.DataFrame, methods: Iterable[str], scenarios: dict[str, tuple[float, float, float]]
) -> pd.DataFrame:  # fmt: skip
    pred = pred[pred.method.isin(tuple(methods))]
    rows = []
    for scenario, prior in scenarios.items():
        for decision, frame in (("argmax", pred), ("saerens", _saerens_pred(pred, prior))):
            acc = expected_accuracy(frame, prior)
            for method, g in acc.groupby("method"):
                rows.append({"method": method, "scenario": scenario, "prior_MCAR": prior[0], "prior_MAR": prior[1],
                             "prior_MNAR": prior[2], "decision": decision, "n_families": len(g),
                             **_mean_ci(g.accuracy.to_numpy())})  # fmt: skip
    return pd.DataFrame(rows)


def rare_mcar(pred: pd.DataFrame, methods: Iterable[str], prevalences: Iterable[float]) -> pd.DataFrame:
    """L1 com MCAR raro: recall e especificidade do teste balanceado (média por família) e a precisão sob π(MCAR)."""
    pred = pred[pred.method.isin(tuple(methods))]
    is_mcar = pred.y_true.to_numpy() == 0
    said_mcar = pred.y_pred.to_numpy() == 0
    frame = pred.assign(sens=np.where(is_mcar, said_mcar, np.nan), spec=np.where(~is_mcar, ~said_mcar, np.nan))
    per = frame.groupby(["method", "family_id", "repeat"])[["sens", "spec"]].mean().groupby(level=[0, 1]).mean()
    rows = []
    for method, g in per.groupby(level=0):
        sens, spec = float(g.sens.mean()), float(g.spec.mean())
        for pi in prevalences:
            denominator = pi * sens + (1 - pi) * (1 - spec)
            rows.append({"method": method, "mcar_prevalence": pi, "n_families": len(g), "recall_mcar": sens,
                         "specificity": spec, "precision_mcar": pi * sens / denominator if denominator else np.nan,
                         **{f"recall_mcar_{k}": v for k, v in _mean_ci(g.sens.to_numpy()).items() if k != "mean"}})  # fmt: skip
    return pd.DataFrame(rows)


def run_robustness(run_dir: Path, config: RunConfig) -> Path:
    """Etapa `robustness` (passo 40, exploratória): prior-shift e L1 com MCAR raro, a partir do `eval-symmetric`.

    Saídas em `<run>/robustness/`, gravadas juntas: `prior_shift.csv` (cenários nomeados), `prior_shift_grid.csv`
    (simplex com passo 0,1) e `rare_mcar.csv` (o marcador de conclusão)."""
    pred = pd.read_parquet(Path(run_dir) / "eval-symmetric" / "predictions.parquet")
    methods = tuple(m for m in METHODS if m in set(pred.method))
    grid = {f"grade {a:.1f}/{b:.1f}/{c:.1f}": (a, b, c) for a, b, c in simplex_grid(0.1)}
    out_dir = Path(run_dir) / "robustness"
    out_dir.mkdir(exist_ok=True)
    tables = {
        "prior_shift.csv": prior_shift(pred, methods, SCENARIOS),
        "prior_shift_grid.csv": prior_shift(pred, (PRIMARY,), grid),
        "rare_mcar.csv": rare_mcar(pred, methods, MCAR_PREVALENCES),
    }
    write_tables_atomically(out_dir, tables)
    append_run_log(Path(run_dir), config, "robustness", {"methods": list(methods), "outputs": list(tables)})
    return out_dir
