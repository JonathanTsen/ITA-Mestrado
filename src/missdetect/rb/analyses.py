"""Análises secundárias e exploratórias do pré-registro (itens 6–7), a partir dos artefatos de um run avaliado.

Nenhuma função daqui ajusta modelo: todas leem predições, baselines e o oráculo já gravados. As análises que ajustam
modelo (ablação, leave-one-generator-out, transferência) ficam em `pipeline.py`.

- `l1_secondary`: L1 do hierárquico × Little, teste 1 do MechDetect-O e PKLM (secundária, Holm m = 3, acurácia
  balanceada por família; falha conta como erro).
- `per_mechanism`: recall por classe para um contraste (exploratória, Holm m = 3).
- `l2_evaluation`: L2 diagnóstico (nas instâncias verdadeiramente não-MCAR) e operacional (nas roteadas ao L2).
- `descriptive_classifiers`: Friedman + Nemenyi e diagrama CD entre os classificadores descritivos (só descritivo).
- `boundary_sensitivity`: escore por família sem os contextos cujo Y completo tem massa ≥ 10% num valor de fronteira.
- `generator_family_tests`: uma comparação por gerador (simétrico, mdatagen, tab_err), com Holm m = 3.
- `mar_mnar_auc`: AUC MAR × MNAR por família nas instâncias verdadeiramente não-MCAR (H6, confirmatória).
- `bootstrap_mean_ci` (de `stats`): IC percentil da média entre famílias, para as análises exploratórias.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from matplotlib.figure import Figure
from scipy import stats
from sklearn.metrics import roc_auc_score

from missdetect.rb.models import CLASSES
from missdetect.rb.stats import _adjust, _raw_comparison, bootstrap_mean_ci

L1_ALPHA = 0.05
L1_METHOD = "hier_hgb_fullprob"
BOUNDARY_MASS = 0.10
DESCRIPTIVE = ("direct_hgb", "direct_logreg", "direct_rf", "direct_svm", "direct_knn", "direct_mlp", "direct_nb")


def _l1_rows(frame: pd.DataFrame, method: str, reject: np.ndarray, failed: np.ndarray, repeat: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": method,
            "repeat": repeat,
            "sample_id": frame.sample_id.to_numpy(),
            "family_id": frame.family_id.to_numpy(),
            "non_mcar": (frame.y_true.to_numpy() != 0) if "y_true" in frame else frame.non_mcar.to_numpy(),
            "reject": reject & ~failed,
            "failed": failed,
        }
    )


GENERATORS = ("symmetric", "mdatagen", "tab_err")


def generator_family_tests(run_dir: Path, method: str, comparator: str, prefix: str) -> pd.DataFrame | None:
    """Método × comparador em cada gerador, cada um com a sua CV (`eval-<gerador>/`), com Holm m = 3. Um gerador sem
    avaliação ou sem os dois métodos entra no Holm com p = 1. Devolve None se nenhum gerador tiver os dois."""
    raw, missing = [], {}
    for generator in GENERATORS:
        name = f"{prefix}_{generator}"
        path = run_dir / f"eval-{generator}" / "per_family.parquet"
        per_family = pd.read_parquet(path) if path.exists() else None
        if per_family is None or not {method, comparator} <= set(per_family.method):
            missing[name] = 1.0
        else:
            raw.append(_raw_comparison(per_family, name, method, comparator, "macro_recall"))
    return _adjust(raw, missing) if raw else None


def mar_mnar_auc(pred: pd.DataFrame, method: str) -> pd.DataFrame:
    """H6: por família, AUC de P(MNAR) / (P(MAR) + P(MNAR)) nas instâncias verdadeiramente não-MCAR (positivo =
    MNAR), média das repetições. Uma falha conta como erro: recebe o pior escore para a sua classe verdadeira (MNAR → 0,
    MAR → 1)."""
    g = pred[(pred.method == method) & pred.y_true.isin([1, 2])]
    denom = (g.p_MAR + g.p_MNAR).to_numpy()
    score = np.divide(g.p_MNAR.to_numpy(), denom, out=np.full(len(g), 0.5), where=denom > 0)
    failed = (~g.status.eq("ok")).to_numpy() if "status" in g else np.zeros(len(g), bool)
    score = np.where(failed, (g.y_true != 2).astype(float).to_numpy(), score)
    frame = g.assign(score=score, target=(g.y_true == 2).astype(int))
    rows = []
    for (family, repeat), h in frame.groupby(["family_id", "repeat"]):
        auc = roc_auc_score(h.target, h.score) if h.target.nunique() == 2 else np.nan
        rows.append({"family_id": family, "repeat": repeat, "auc": float(auc)})
    return pd.DataFrame(rows).groupby("family_id", as_index=False).auc.mean()


def l1_decisions(run_dir: Path, eval_dir: str = "eval-symmetric", l1_method: str = L1_METHOD) -> pd.DataFrame:
    """Decisão L1 (rejeita MCAR?) por instância e método, nas instâncias da avaliação primária."""
    pred = pd.read_parquet(run_dir / eval_dir / "predictions.parquet")
    hier = pred[pred.method == l1_method]
    parts = [
        _l1_rows(g, "l1_hier", (g.p_MCAR < 0.5).to_numpy(), ~g.status.eq("ok").to_numpy(), int(rep))
        for rep, g in hier.groupby("repeat")
    ]
    base = hier[hier.repeat == hier.repeat.min()][["sample_id", "family_id", "y_true"]]
    baselines = run_dir / "baselines.parquet"
    if baselines.exists():
        b = base.merge(pd.read_parquet(baselines), on="sample_id", how="left")
        for name, col, level in (("little", "little_p", L1_ALPHA), ("mdo_test1", "mdo_p_shuffled", L1_ALPHA / 2)):
            p = b[col].to_numpy(float)
            parts.append(_l1_rows(b, name, p < level, np.isnan(p)))
    pklm_path = run_dir / "pklm.parquet"
    if pklm_path.exists():
        k = base.merge(pd.read_parquet(pklm_path)[["sample_id", "pklm_p"]], on="sample_id")
        p = k.pklm_p.to_numpy(float)
        parts.append(_l1_rows(k, "pklm", p < L1_ALPHA, np.isnan(p)))
        subset = set(k.sample_id)
        for rep, g in hier.groupby("repeat"):
            g = g[g.sample_id.isin(subset)]
            parts.append(
                _l1_rows(g, "l1_hier_pklm_subset", (g.p_MCAR < 0.5).to_numpy(), ~g.status.eq("ok").to_numpy(), int(rep))
            )
    return pd.concat(parts, ignore_index=True)


def l1_per_family(decisions: pd.DataFrame) -> pd.DataFrame:
    """Sensibilidade, especificidade e acurácia balanceada por família (média das repetições quando houver)."""
    d = decisions.assign(hit=np.where(decisions.non_mcar, decisions.reject, ~decisions.reject & ~decisions.failed))
    rows = []
    for (method, rep, family), g in d.groupby(["method", "repeat", "family_id"]):
        sens = g.loc[g.non_mcar, "hit"].mean()
        spec = g.loc[~g.non_mcar, "hit"].mean()
        rows.append({"method": method, "repeat": rep, "family_id": family, "sensitivity": sens, "specificity": spec,
                     "balanced_accuracy": (sens + spec) / 2})  # fmt: skip
    return (
        pd.DataFrame(rows)
        .groupby(["method", "family_id"], as_index=False)[["sensitivity", "specificity", "balanced_accuracy"]]
        .mean()
    )


L1_HYPOTHESES = {
    "L1_vs_Little": ("l1_hier", "little"),
    "L1_vs_MechDetectO_teste1": ("l1_hier", "mdo_test1"),
    "L1_vs_PKLM": ("l1_hier_pklm_subset", "pklm"),
}


def l1_secondary(run_dir: Path, l1_method: str = L1_METHOD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """L1 secundária com Holm m = 3 fixo: uma comparação que não pôde rodar (ex.: sem PKLM) entra com p = 1."""
    per_family = l1_per_family(l1_decisions(run_dir, l1_method=l1_method))
    present = set(per_family.method)
    raw, missing = [], {}
    for name, (m, c) in L1_HYPOTHESES.items():
        if {m, c} <= present:
            raw.append(_raw_comparison(per_family, name, m, c, "balanced_accuracy"))
        else:
            missing[name] = 1.0
    return per_family, _adjust(raw, missing)


def per_mechanism(per_family: pd.DataFrame, name: str, method: str, comparator: str) -> pd.DataFrame:
    """Recall por classe para um contraste, com Holm sobre as 3 classes (exploratória)."""
    raw = [_raw_comparison(per_family, f"{name}_{c}", method, comparator, f"recall_{c}") for c in CLASSES]
    return _adjust(raw)


def l2_evaluation(pred: pd.DataFrame, method: str = L1_METHOD) -> pd.DataFrame:
    """Macro-recall MAR × MNAR do L2, diagnóstico e operacional, por família (média das repetições).

    Com o fullprob, P(MAR | não-MCAR) = p_MAR / (p_MAR + p_MNAR). Diagnóstico: instâncias verdadeiramente não-MCAR.
    Operacional: as verdadeiramente não-MCAR que o L1 roteou ao L2 (P(MCAR) < 0,5); as MCAR roteadas (falsos
    positivos do L1) entram como fração à parte. Instâncias com falha contam como erro e não são roteadas.
    """
    g = pred[pred.method == method].copy()
    ok = g.status.eq("ok").to_numpy()
    g["p2"] = g.p_MAR / (g.p_MAR + g.p_MNAR).clip(lower=1e-12)
    g["l2_pred"] = np.where(ok, np.where(g.p2 >= 0.5, 1, 2), -1)  # falha conta como erro (classe −1)
    g["routed"] = (g.p_MCAR < 0.5) & ok
    rows = []
    for (rep, family), h in g.groupby(["repeat", "family_id"]):
        nm = h[h.y_true != 0]
        op = nm[nm.routed]

        def mr(x: pd.DataFrame) -> float:
            recalls = [np.mean(x.l2_pred[x.y_true == c] == c) for c in (1, 2) if np.any(x.y_true == c)]
            return float(np.mean(recalls)) if recalls else np.nan

        rows.append({"repeat": rep, "family_id": family, "l2_diagnostic_macro_recall": mr(nm),
                     "l2_operational_macro_recall": mr(op), "non_mcar_routed_fraction": float(nm.routed.mean()),
                     "mcar_routed_fraction": float(h[h.y_true == 0].routed.mean())})  # fmt: skip
    return pd.DataFrame(rows).groupby("family_id", as_index=False).mean().drop(columns="repeat")


def descriptive_classifiers(
    per_family: pd.DataFrame, out_png: Path | None = None, methods: tuple[str, ...] = DESCRIPTIVE
) -> dict:
    """Friedman + Nemenyi e diagrama CD entre os classificadores descritivos (macro-recall por família)."""
    wide = per_family.pivot(index="family_id", columns="method", values="macro_recall")
    cols = [c for c in methods if c in wide]
    wide = wide[cols].dropna()
    friedman = stats.friedmanchisquare(*[wide[c].to_numpy() for c in cols])
    nemenyi = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    nemenyi.index = nemenyi.columns = cols
    ranks = wide.rank(axis=1, ascending=False).mean()
    if out_png is not None:
        fig = Figure(figsize=(7, 2.6))
        ax = fig.add_subplot()
        sp.critical_difference_diagram(ranks, nemenyi, ax=ax)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
    return {"friedman_statistic": float(friedman.statistic), "friedman_p": float(friedman.pvalue),
            "n_families": len(wide), "mean_ranks": ranks.round(3).to_dict(), "nemenyi_p": nemenyi.round(4)}  # fmt: skip


def boundary_flags(run_dir: Path, threshold: float = BOUNDARY_MASS) -> pd.DataFrame:
    """Por contexto: o Y completo tem massa ≥ `threshold` em 0 ou no seu mínimo? (lê o oráculo)."""
    rows = []
    for path in sorted((run_dir / "oracle_complete").glob("*.parquet")):
        y = pd.read_parquet(path).y_complete.to_numpy(float)
        mass = max(np.mean(y == 0), np.mean(y == np.min(y)))
        rows.append({"context_id": path.stem, "boundary_mass": float(mass), "flagged": bool(mass >= threshold)})
    return pd.DataFrame(rows)


def boundary_sensitivity(run_dir: Path, pred: pd.DataFrame, method: str) -> pd.DataFrame:
    """Macro-recall por família com todos os contextos × sem os sinalizados (exploratória), pareado nas famílias que
    sobram nos dois conjuntos; as famílias com todos os contextos sinalizados são listadas à parte."""
    instances = pd.read_parquet(run_dir / "instances.parquet")[["sample_id", "context_id"]]
    flags = boundary_flags(run_dir)
    p = pred[pred.method == method].merge(instances, on="sample_id").merge(flags, on="context_id")
    rows = []
    for subset, data in (("todos", p), ("sem_sinalizados", p[~p.flagged])):
        for (rep, family), g in data.groupby(["repeat", "family_id"]):
            recalls = [np.mean(g.y_pred[g.y_true == c] == c) for c in range(3) if np.any(g.y_true == c)]
            rows.append({"subset": subset, "repeat": rep, "family_id": family, "macro_recall": float(np.mean(recalls)),
                         "n": len(g)})  # fmt: skip
    per_family = pd.DataFrame(rows).groupby(["subset", "family_id"], as_index=False)[["macro_recall", "n"]].mean()
    wide = per_family.pivot(index="family_id", columns="subset", values="macro_recall")
    common = wide.dropna()
    removed = sorted(set(wide.index) - set(common.index))
    diff = (common["sem_sinalizados"] - common["todos"]).to_numpy()
    low, high = bootstrap_mean_ci(diff) if len(diff) else (np.nan, np.nan)
    return pd.DataFrame([{
        "n_families_total": len(wide), "n_families_common": len(common), "families_fully_flagged": ";".join(map(str, removed)),
        "macro_recall_all_common": float(common["todos"].mean()), "macro_recall_unflagged_common": float(common["sem_sinalizados"].mean()),
        "paired_mean_diff": float(np.mean(diff)) if len(diff) else np.nan, "ci_low": low, "ci_high": high,
        "contexts_flagged": int(flags.flagged.sum()), "contexts_total": len(flags),
    }])  # fmt: skip
