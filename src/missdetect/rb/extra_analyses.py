"""Análises das emendas de 29/10/2026 ao pré-registro (§0.4), a partir de um run já avaliado.

Nenhuma função daqui ajusta modelo: todas leem predições, baselines, contextos e o oráculo gravados. São descritivas,
exceto a P12, que é uma secundária com Holm m = 2. A etapa `secondary-extra` (em `pipeline.py`) grava as saídas em
`secondary-extra/`.

- P6, `recall_by_regime` e `h6_auc_by_regime`: recall por classe e AUC MAR × MNAR por β e, com β = ∞, por direção.
  É o que mostra se a censura determinística de Y é reconhecida como MNAR.
- P7, `h6_outside_generator` e `trio_concordance`: a diferença de AUC da H6 nos geradores externos e a concordância
  dentro do trio (a instância MNAR recebe escore maior que a MAR do mesmo contexto?).
- P8, `class_decomposition`, `generator_effect` e `specialization`: a diferença da H5 por classe; primária no externo
  − primária no recorte β = ∞ do simétrico; treinada no externo − treinada no simétrico.
- P11, `by_repeat`: a família principal refeita em cada repetição, separadamente.
- P12, `decision_errors` e `decision_error_tests`: falso MCAR e MNAR → MAR, por família, primária × logística simples.
- P13, `l1_auc`: AUC do L1 sem limiar contra o teste de Little, o PKLM e o teste 1 do MechDetect-O.
- P15, `nominal_contexts`: a lista fechada de bases com Y nominal, para a sensibilidade.

Convenção de falhas (a mesma das hipóteses): uma predição com falha (`y_pred = −1`) conta como erro; nos escores de
AUC, uma falha recebe o pior escore para a sua classe verdadeira.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from missdetect.rb.analyses import mar_mnar_auc
from missdetect.rb.models import CLASSES
from missdetect.rb.stats import (
    PRIMARY,
    _adjust,
    _raw_comparison,
    bootstrap_hl_ci,
    bootstrap_mean_ci,
    confirmatory,
    hodges_lehmann,
)

SIMPLE = "logit_simple"
CROSS_PRIMARY = "cross_direct_hgb"
CROSS_SIMPLE = "cross_logit_simple"
INGEN = "ingen_direct_hgb"
RECALLS = [f"recall_{c}" for c in CLASSES]


def _mean_ci(values: Iterable[float]) -> dict:
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"n_families": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    low, high = bootstrap_mean_ci(v)
    return {"n_families": len(v), "mean": float(v.mean()), "ci_low": low, "ci_high": high}


def _hl_ci(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"n_families": 0, "hodges_lehmann": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    low, high = bootstrap_hl_ci(d)
    return {"n_families": len(d), "hodges_lehmann": hodges_lehmann(d), "ci_low": low, "ci_high": high}


def recalls_by_family(pred: pd.DataFrame, keys: tuple[str, ...] = ()) -> pd.DataFrame:
    """Recall por classe e macro-recall por (método, família, repetição, *keys). Falha (`y_pred = −1`) conta como erro."""
    group = ["method", "family_id", "repeat", *keys]
    rows = []
    for key, g in pred.groupby(group):
        recall = {
            f"recall_{c}": float(np.mean(g.y_pred[g.y_true == k] == k)) if np.any(g.y_true == k) else np.nan
            for k, c in enumerate(CLASSES)
        }
        present = [v for v in recall.values() if not np.isnan(v)]
        rows.append({**dict(zip(group, key, strict=True)), **recall,
                     "macro_recall": float(np.mean(present)) if present else np.nan})  # fmt: skip
    return pd.DataFrame(rows, columns=[*group, *RECALLS, "macro_recall"])


def _family_mean(table: pd.DataFrame, keys: tuple[str, ...] = ()) -> pd.DataFrame:
    """Média das repetições por (método, família, *keys)."""
    return table.drop(columns="repeat").groupby(["method", "family_id", *keys], as_index=False).mean()


# ---------------------------------------------------------------------------------------------------------------------
# P6: por regime (β e, com β = ∞, direção)
# ---------------------------------------------------------------------------------------------------------------------
def _with_regime(pred: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Duplica as instâncias com β = ∞ para que apareçam também no regime `inf/<direção>`."""
    p = pred.merge(regimes[["sample_id", "beta", "direction"]], on="sample_id")
    by_beta = p.assign(regime=np.where(np.isinf(p.beta), "inf", p.beta.astype(str)))
    inf = p[np.isinf(p.beta)]
    by_dir = inf.assign(regime="inf/" + inf.direction.astype(str))
    return pd.concat([by_beta, by_dir], ignore_index=True)


def recall_by_regime(pred: pd.DataFrame, regimes: pd.DataFrame, method: str = PRIMARY) -> pd.DataFrame:
    """P6: recall por classe e macro-recall por regime, média entre famílias com IC95% por bootstrap."""
    p = _with_regime(pred[pred.method == method], regimes)
    per_family = _family_mean(recalls_by_family(p, ("regime",)), ("regime",))
    rows = []
    for regime, g in per_family.groupby("regime"):
        for metric in (*RECALLS, "macro_recall"):
            rows.append({"method": method, "regime": regime, "metric": metric, **_mean_ci(g[metric])})
    return pd.DataFrame(rows)


def h6_auc_by_regime(pred: pd.DataFrame, regimes: pd.DataFrame, method: str = PRIMARY) -> pd.DataFrame:
    """P6: AUC MAR × MNAR da H6 por regime, média entre famílias com IC95%."""
    p = _with_regime(pred[pred.method == method], regimes)
    rows = []
    for regime, g in p.groupby("regime"):
        auc = mar_mnar_auc(g.drop(columns=["beta", "direction", "regime"]), method)
        rows.append({"method": method, "regime": regime, "metric": "auc_mar_mnar", **_mean_ci(auc.auc)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# P7: H6 fora do gerador e concordância dentro do trio
# ---------------------------------------------------------------------------------------------------------------------
def h6_outside_generator(cross_pred: pd.DataFrame) -> pd.DataFrame:
    """P7: AUC MAR × MNAR por família nos geradores externos, com os modelos da H5, e a diferença (HL com IC)."""
    auc = {m: mar_mnar_auc(cross_pred, m).set_index("family_id").auc for m in (CROSS_PRIMARY, CROSS_SIMPLE)}
    wide = pd.DataFrame(auc).dropna()
    rows = [{"quantity": f"auc_{m}", **_mean_ci(wide[m])} for m in (CROSS_PRIMARY, CROSS_SIMPLE)]
    rows.append({"quantity": f"diff_{CROSS_PRIMARY}_minus_{CROSS_SIMPLE}",
                 **_hl_ci((wide[CROSS_PRIMARY] - wide[CROSS_SIMPLE]).to_numpy())})  # fmt: skip
    return pd.DataFrame(rows)


def _mnar_share(g: pd.DataFrame) -> np.ndarray:
    """P(MNAR) / (P(MAR) + P(MNAR)); falha recebe o pior escore para a classe verdadeira (MNAR → 0, MAR → 1)."""
    denom = (g.p_MAR + g.p_MNAR).to_numpy()
    score = np.divide(g.p_MNAR.to_numpy(), denom, out=np.full(len(g), 0.5), where=denom > 0)
    failed = (~g.status.eq("ok")).to_numpy() if "status" in g else np.zeros(len(g), bool)
    return np.where(failed, (g.y_true != 2).astype(float).to_numpy(), score)


def trio_concordance(pred: pd.DataFrame, contexts: pd.DataFrame, method: str) -> pd.DataFrame:
    """P7: por família, a fração dos contextos em que a instância MNAR tem escore maior que a MAR do mesmo contexto
    (empate = 1/2), média das repetições."""
    g = pred[(pred.method == method) & pred.y_true.isin([1, 2])].merge(contexts[["sample_id", "context_id"]])
    g = g.assign(score=_mnar_share(g))
    wide = g.pivot_table(index=["family_id", "repeat", "context_id"], columns="y_true", values="score").dropna()
    hit = (wide[2] > wide[1]).astype(float) + 0.5 * (wide[2] == wide[1])
    by_repeat = hit.groupby(level=["family_id", "repeat"]).mean()
    out = by_repeat.groupby(level="family_id").mean().rename("concordance").reset_index()
    return out.assign(method=method)


def trio_concordance_summary(conc: pd.DataFrame) -> pd.DataFrame:
    """P7: nível de cada configuração e a diferença primária − logística simples (HL com IC)."""
    wide = conc.pivot(index="family_id", columns="method", values="concordance").dropna()
    rows = [{"quantity": f"concordance_{m}", **_mean_ci(wide[m])} for m in wide.columns]
    if {PRIMARY, SIMPLE} <= set(wide.columns):
        rows.append({"quantity": f"diff_{PRIMARY}_minus_{SIMPLE}", **_hl_ci((wide[PRIMARY] - wide[SIMPLE]).to_numpy())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# P8: decomposições da H5
# ---------------------------------------------------------------------------------------------------------------------
def class_decomposition(per_family: pd.DataFrame, method: str, comparator: str, label: str) -> pd.DataFrame:
    """P8: a diferença método − comparador por classe (recall) e no macro-recall, HL com IC por bootstrap."""
    rows = []
    for metric in (*RECALLS, "macro_recall"):
        wide = per_family.pivot(index="family_id", columns="method", values=metric)
        if {method, comparator} <= set(wide.columns):
            d = (wide[method] - wide[comparator]).dropna().to_numpy()
            rows.append({"contrast": label, "method": method, "comparator": comparator, "metric": metric, **_hl_ci(d)})
    return pd.DataFrame(rows)


def _macro_by_family(pred: pd.DataFrame, method: str) -> pd.Series:
    table = recalls_by_family(pred[pred.method == method])
    return _family_mean(table).set_index("family_id").macro_recall


def generator_effect(
    cross_pred: pd.DataFrame, sym_pred: pd.DataFrame, regimes: pd.DataFrame, generators: pd.DataFrame
) -> pd.DataFrame:
    """P8: por gerador externo, primária no externo − primária no recorte β = ∞ do simétrico, nas mesmas famílias."""
    inf_ids = set(regimes.loc[np.isinf(regimes.beta), "sample_id"])
    sym_inf = _macro_by_family(sym_pred[sym_pred.sample_id.isin(inf_ids)], PRIMARY)
    cross = cross_pred.merge(generators[["sample_id", "generator"]], on="sample_id")
    rows = []
    for generator, g in cross.groupby("generator"):
        ext = _macro_by_family(g, CROSS_PRIMARY)
        d = (ext - sym_inf).dropna().to_numpy()
        rows.append({"generator": generator, "quantity": "effect_external_minus_symmetric_beta_inf", **_hl_ci(d)})
    return pd.DataFrame(rows)


def specialization(cross_pred: pd.DataFrame, generators: pd.DataFrame) -> pd.DataFrame:
    """P8: por gerador externo, treinada no próprio externo − treinada no simétrico (HL com IC)."""
    cross = cross_pred.merge(generators[["sample_id", "generator"]], on="sample_id")
    rows = []
    for generator, g in cross.groupby("generator"):
        d = (_macro_by_family(g, INGEN) - _macro_by_family(g, CROSS_PRIMARY)).dropna().to_numpy()
        rows.append({"generator": generator, "quantity": "specialization_ingen_minus_cross", **_hl_ci(d)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# P11: sensibilidade à partição
# ---------------------------------------------------------------------------------------------------------------------
def by_repeat(sym_pred: pd.DataFrame, cross_pred: pd.DataFrame | None) -> pd.DataFrame:
    """P11: a família principal (H1–H6, Holm m = 6) refeita com os escores de cada repetição, separadamente."""
    parts = []
    for repeat in sorted(sym_pred.repeat.unique()):
        sym_r = sym_pred[sym_pred.repeat == repeat]
        main = recalls_by_family(sym_r).drop(columns="repeat")
        cross = None
        if cross_pred is not None and not cross_pred.empty:
            cross = recalls_by_family(cross_pred[cross_pred.repeat == repeat]).drop(columns="repeat")
        auc = pd.concat([mar_mnar_auc(sym_r, m).assign(method=m) for m in (PRIMARY, SIMPLE)], ignore_index=True)
        parts.append(confirmatory(main, cross, auc).assign(repeat=int(repeat)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------
# P12: erros que mudam a decisão do usuário
# ---------------------------------------------------------------------------------------------------------------------
def decision_errors(pred: pd.DataFrame, methods: tuple[str, ...] = (PRIMARY, SIMPLE)) -> pd.DataFrame:
    """P12: por família (média das repetições), a taxa de falso MCAR (MAR ou MNAR previstos como MCAR) e a taxa
    MNAR → MAR. As colunas `avoid_*` = 1 − taxa entram no teste, para que "superior" continue querendo dizer melhor."""
    rows = []
    for (method, family, repeat), g in pred[pred.method.isin(methods)].groupby(["method", "family_id", "repeat"]):
        non_mcar, mnar = g[g.y_true != 0], g[g.y_true == 2]
        rows.append({
            "method": method, "family_id": family, "repeat": repeat,
            "false_mcar": float(np.mean(non_mcar.y_pred == 0)) if len(non_mcar) else np.nan,
            "mnar_as_mar": float(np.mean(mnar.y_pred == 1)) if len(mnar) else np.nan,
        })  # fmt: skip
    table = _family_mean(pd.DataFrame(rows))
    return table.assign(avoid_false_mcar=1 - table.false_mcar, avoid_mnar_as_mar=1 - table.mnar_as_mar)


def decision_error_tests(per_family: pd.DataFrame) -> pd.DataFrame:
    """P12: secundária (Holm m = 2), primária × logística simples em 1 − taxa de cada erro."""
    raw = [
        _raw_comparison(per_family, "P12_falso_MCAR", PRIMARY, SIMPLE, "avoid_false_mcar"),
        _raw_comparison(per_family, "P12_MNAR_como_MAR", PRIMARY, SIMPLE, "avoid_mnar_as_mar"),
    ]
    return _adjust(raw)


# ---------------------------------------------------------------------------------------------------------------------
# P13: L1 sem limiar
# ---------------------------------------------------------------------------------------------------------------------
def _auc_rows(frame: pd.DataFrame, method: str, score: np.ndarray, failed: np.ndarray) -> list[dict]:
    non_mcar = (frame.y_true.to_numpy() != 0).astype(int)
    score = np.where(failed, 1.0 - non_mcar, score)  # falha: pior escore para a classe verdadeira
    data = frame.assign(_score=score, _target=non_mcar)
    rows = []
    for (family, repeat), g in data.groupby(["family_id", "repeat"]):
        auc = roc_auc_score(g._target, g._score) if g._target.nunique() == 2 else np.nan
        rows.append({"method": method, "family_id": family, "repeat": repeat, "auc": float(auc)})
    return rows


def l1_auc(
    sym_pred: pd.DataFrame, baselines: pd.DataFrame | None, pklm: pd.DataFrame | None, l1_method: str
) -> pd.DataFrame:
    """P13: AUC por família de P(não-MCAR) do L1 (1 − p_MCAR) e de 1 − p dos testes, nas mesmas instâncias."""
    hier = sym_pred[sym_pred.method == l1_method]
    rows = _auc_rows(hier, "l1_hier", 1 - hier.p_MCAR.to_numpy(), ~hier.status.eq("ok").to_numpy())
    base = hier[hier.repeat == hier.repeat.min()][["sample_id", "family_id", "y_true"]].assign(repeat=0)
    if baselines is not None:
        b = base.merge(baselines, on="sample_id", how="left")
        for name, col in (("little", "little_p"), ("mdo_test1", "mdo_p_shuffled")):
            if col in b:
                p = b[col].to_numpy(float)
                rows += _auc_rows(b, name, 1 - np.nan_to_num(p, nan=0.5), np.isnan(p))
    if pklm is not None:
        k = base.merge(pklm[["sample_id", "pklm_p"]], on="sample_id")
        p = k.pklm_p.to_numpy(float)
        rows += _auc_rows(k, "pklm", 1 - np.nan_to_num(p, nan=0.5), np.isnan(p))
        sub = hier[hier.sample_id.isin(set(k.sample_id))]
        rows += _auc_rows(sub, "l1_hier_pklm_subset", 1 - sub.p_MCAR.to_numpy(), ~sub.status.eq("ok").to_numpy())
    table = pd.DataFrame(rows)
    return table.groupby(["method", "family_id"], as_index=False).auc.mean()


L1_AUC_CONTRASTS = {
    "L1_vs_Little": ("l1_hier", "little"),
    "L1_vs_MechDetectO_teste1": ("l1_hier", "mdo_test1"),
    "L1_vs_PKLM": ("l1_hier_pklm_subset", "pklm"),
}


def l1_auc_summary(table: pd.DataFrame) -> pd.DataFrame:
    """P13: nível de cada método e a diferença L1 − teste por família (HL com IC), sem p."""
    wide = table.pivot(index="family_id", columns="method", values="auc")
    rows = [{"quantity": f"auc_{m}", **_mean_ci(wide[m])} for m in wide.columns]
    for name, (m, c) in L1_AUC_CONTRASTS.items():
        if {m, c} <= set(wide.columns):
            rows.append({"quantity": name, **_hl_ci((wide[m] - wide[c]).to_numpy())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# P15: bases com Y nominal
# ---------------------------------------------------------------------------------------------------------------------
def nominal_contexts(contexts: pd.DataFrame, nominal: pd.DataFrame) -> set[str]:
    """Contextos da lista fechada de Y nominal: `target == "*"` marca a família inteira."""
    keys = {(int(f), str(t)) for f, t in zip(nominal.family_id, nominal.target, strict=True)}
    whole = {f for f, t in keys if t == "*"}
    hit = [
        (int(f) in whole) or ((int(f), str(t)) in keys)
        for f, t in zip(contexts.family_id, contexts.target, strict=True)
    ]
    return set(contexts.loc[hit, "context_id"])


def without_contexts(
    sym_pred: pd.DataFrame, cross_pred: pd.DataFrame | None, instances: pd.DataFrame, excluded: set[str]
) -> pd.DataFrame:
    """P15: a família principal (H1–H6) refeita sem as instâncias dos contextos excluídos (Y nominal)."""
    keep = set(instances.loc[~instances.context_id.isin(excluded), "sample_id"])
    sym = sym_pred[sym_pred.sample_id.isin(keep)]
    main = _family_mean(recalls_by_family(sym))
    cross = None
    if cross_pred is not None and not cross_pred.empty:
        cross = _family_mean(recalls_by_family(cross_pred[cross_pred.sample_id.isin(keep)]))
    auc = pd.concat([mar_mnar_auc(sym, m).assign(method=m) for m in (PRIMARY, SIMPLE)], ignore_index=True)
    return confirmatory(main, cross, auc).assign(n_contexts_excluded=len(excluded))


def l1_false_mcar(l1_per_family: pd.DataFrame) -> pd.DataFrame:
    """P12 (descritivo): taxa de falso MCAR dos métodos do L1 = 1 − sensibilidade, média entre famílias com IC."""
    rows = [
        {"method": method, "metric": "false_mcar", **_mean_ci(1 - g.sensitivity.to_numpy(float))}
        for method, g in l1_per_family.groupby("method")
    ]
    return pd.DataFrame(rows)
