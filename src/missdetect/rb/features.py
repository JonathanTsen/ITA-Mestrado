"""Descritores observáveis, versão 34b (congelada com a tag `freeze-v1`, passo 37 do plano).

Histórico: 34a-1 (primeira versão); 34a-2 (Ŷ cross-fit sem vazar a máscara); 34b (sem os três agregados de
|U/(n₁n₀) − 0,5| do Mann-Whitney, que são idênticos aos de |AUC − 0,5| univariado; decisão D4 de 21/10/2026).

Regras (Cap3, `sec:m_descriptors`):
1. só informação observável: Y observado, Z e M; os valores de Y nas linhas com M = 1 nunca são usados, nem
   imputados (um valor imputado num ponto fixo revela a máscara em qualquer mecanismo);
2. invariância à ordem das linhas e das colunas: as linhas e as auxiliares são postas numa ordem canônica, derivada
   só dos valores observados, antes de qualquer cálculo; folds internos e sementes são fixos;
3. agregação independente de k: estatísticas por auxiliar são resumidas por máximo, média e segundo maior valor;
4. valor indefinido → NaN, com a contagem em `status_undefined`;
5. nomes pelo que medem.

Grupos: G0 taxa; G1 forma de Y observado; G2 máscara × auxiliares; G3 Y dado Z (mediação); G4 análogo observável
do 1º teste do MechDetect.

NaN nas auxiliares (só no ensaio do passo 40, item 6; 30/10/2026): o HGB trata NaN nativamente; a logística recebe a
mediana do treino; AUC, KS e informação mútua usam os pares completos de cada auxiliar. Esses desvios só rodam quando Z
tem NaN: com Z completo, o cálculo é o original, idêntico ao congelado.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from missdetect.rb.inject import ObservedInstance

FEATURE_SCHEMA_VERSION = "34b"
N_FOLDS = 5
N_FOLDS_G4 = 10
SEED = 0


def canonicalize(y: np.ndarray, z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ordem canônica das colunas (pela sua soma e variância) e das linhas (lexicográfica nos valores observados)."""
    zv = z.to_numpy(dtype=float)
    if np.isnan(zv).any():  # passo 40, item 6 (NaN em Z): estatísticas sem os NaN; Z completo segue o caminho original
        col_keys = np.lexsort((np.round(np.nanvar(zv, axis=0), 10), np.round(np.nansum(zv, axis=0), 8)))
    else:
        col_keys = np.lexsort((np.round(zv.var(axis=0), 10), np.round(zv.sum(axis=0), 8)))
    zv = zv[:, col_keys]
    mask = np.isnan(y)
    y_key = np.where(mask, np.inf, y)
    row_keys = np.lexsort((*(np.round(zv[:, j], 10) for j in range(zv.shape[1] - 1, -1, -1)), y_key))
    return y[row_keys], zv[row_keys]


def _agg(prefix: str, values: list[float]) -> dict:
    vals = np.sort(np.asarray([v for v in values if np.isfinite(v)], dtype=float))[::-1]
    if len(vals) == 0:
        return {f"{prefix}_max": np.nan, f"{prefix}_mean": np.nan, f"{prefix}_second": np.nan}
    second = vals[1] if len(vals) > 1 else vals[0]
    return {f"{prefix}_max": float(vals[0]), f"{prefix}_mean": float(vals.mean()), f"{prefix}_second": float(second)}


def _auc(target: np.ndarray, score: np.ndarray) -> float:
    if target.min() == target.max():
        return np.nan
    return float(roc_auc_score(target, score))


def _cv_auc(x: np.ndarray, m: np.ndarray, model: str, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """AUCs por fold de M predito de x (folds estratificados fixos sobre a ordem canônica)."""
    if min(m.sum(), (~m).sum()) < n_folds:
        return np.full(n_folds, np.nan)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for train, test in folds.split(x, m):
        if model == "logistic" and np.isnan(x).any():  # NaN em Z (passo 40): mediana do treino antes da logística
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
        elif model == "logistic":
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        else:
            clf = HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=seed)
        clf.fit(x[train], m[train])
        scores.append(_auc(m[test], clf.predict_proba(x[test])[:, 1]))
    return np.asarray(scores)


def crossfit_prediction(z_obs: np.ndarray, y_obs: np.ndarray, z_all: np.ndarray, obs_idx: np.ndarray):
    """Ŷ(Z) ajustado nas linhas observadas, com o mesmo procedimento para todas as linhas.

    Cada linha recebe a predição de um único modelo de dobra que não a viu: as observadas, o modelo da sua dobra
    (fora-da-dobra); as ausentes, o modelo de uma dobra atribuída de forma determinística (posição na ordem recebida,
    em rodízio). Assim, sob MCAR, Ŷ tem a mesma distribuição nas linhas com M = 1 e M = 0. A versão 34a-1 usava a
    média das dobras nas ausentes, o que reduz a variância só nessas linhas e revela a máscara (achado do piloto).
    """
    pred_all = np.full(len(z_all), np.nan)
    pred_obs = np.full(len(y_obs), np.nan)
    missing_idx = np.setdiff1d(np.arange(len(z_all)), obs_idx)
    missing_fold = np.arange(len(missing_idx)) % N_FOLDS
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(z_obs)
    for fold, (train, test) in enumerate(folds):
        reg = HistGradientBoostingRegressor(max_iter=100, early_stopping=False, random_state=SEED)
        reg.fit(z_obs[train], y_obs[train])
        pred_obs[test] = reg.predict(z_obs[test])
        assigned = missing_idx[missing_fold == fold]
        if len(assigned):
            pred_all[assigned] = reg.predict(z_all[assigned])
    pred_all[obs_idx] = pred_obs
    return pred_all, pred_obs


def _bimodality(x: np.ndarray) -> float:
    n = len(x)
    if n < 4 or np.std(x) == 0:
        return np.nan
    g = stats.skew(x)
    k = stats.kurtosis(x)
    return float((g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def compute(instance: ObservedInstance) -> dict:
    """Vetor de descritores de uma instância observada. Não recebe rótulo nem valores removidos."""
    y, z = canonicalize(np.asarray(instance.y, dtype=float), instance.z)
    m = np.isnan(y)
    obs_idx = np.flatnonzero(~m)
    y_obs = y[obs_idx]
    out: dict = {"sample_id": instance.sample_id, "schema": FEATURE_SCHEMA_VERSION}

    # G0
    out["g0_missing_rate"] = float(m.mean())

    # G1: forma de Y observado
    sd = float(np.std(y_obs)) if len(y_obs) > 1 else np.nan
    iqr = float(np.subtract(*np.percentile(y_obs, [75, 25]))) if len(y_obs) else np.nan
    rng_obs = float(y_obs.max() - y_obs.min()) if len(y_obs) else np.nan
    out["g1_skewness"] = float(stats.skew(y_obs)) if sd and sd > 0 else np.nan
    out["g1_excess_kurtosis"] = float(stats.kurtosis(y_obs)) if sd and sd > 0 else np.nan
    out["g1_iqr_over_sd"] = iqr / sd if sd and sd > 0 else np.nan
    out["g1_repeated_fraction"] = 1.0 - len(np.unique(y_obs)) / len(y_obs) if len(y_obs) else np.nan
    out["g1_median_position"] = (float(np.median(y_obs)) - y_obs.min()) / rng_obs if rng_obs and rng_obs > 0 else np.nan
    out["g1_bimodality"] = _bimodality(y_obs)

    # G2: máscara × auxiliares
    out["g2_cv_auc_logistic"] = float(np.nanmean(_cv_auc(z, m, "logistic")))
    out["g2_cv_auc_boosting"] = float(np.nanmean(_cv_auc(z, m, "boosting")))
    z_has_nan = bool(np.isnan(z).any())
    uni_auc, ks = [], []
    for j in range(z.shape[1]):
        zj, mj = z[:, j], m
        if z_has_nan:  # NaN em Z (passo 40): pares completos da auxiliar
            ok = ~np.isnan(zj)
            zj, mj = zj[ok], m[ok]
        a = _auc(mj.astype(int), zj) if len(zj) else np.nan
        uni_auc.append(abs(a - 0.5) if np.isfinite(a) else np.nan)
        ks.append(stats.ks_2samp(zj[mj], zj[~mj]).statistic if mj.any() and (~mj).any() else np.nan)
    out.update(_agg("g2_univariate_auc_dev", uni_auc))
    out.update(_agg("g2_ks", ks))
    if not (m.any() and (~m).any()):
        mi = np.full(z.shape[1], np.nan)
    elif z_has_nan:  # NaN em Z: uma auxiliar por vez, nos pares completos (Z completo usa a chamada conjunta original)
        mi = np.array([
            mutual_info_classif(z[~np.isnan(z[:, j]), j : j + 1], m[~np.isnan(z[:, j])], random_state=SEED)[0]
            if (~np.isnan(z[:, j])).sum() > 3 else np.nan
            for j in range(z.shape[1])
        ])  # fmt: skip
    else:
        mi = mutual_info_classif(z, m, random_state=SEED)
    out.update(_agg("g2_mutual_information", list(mi)))

    # G3: Y dado Z (mediação), só com as linhas observadas para ajustar
    if len(y_obs) >= 2 * N_FOLDS and m.any():
        y_hat, y_hat_obs = crossfit_prediction(z[obs_idx], y_obs, z, obs_idx)
        resid = y_obs - y_hat_obs
        ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
        out["g3_r2_observed"] = float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan
        out["g3_auc_mask_from_yhat_dev"] = abs(_auc(m.astype(int), y_hat) - 0.5)
        out["g3_auc_yhat_minus_z"] = abs(_auc(m.astype(int), y_hat) - 0.5) - (out["g2_cv_auc_boosting"] - 0.5)
        sd_hat = float(np.std(y_hat))
        out["g3_yhat_std_mean_diff"] = float((y_hat[m].mean() - y_hat[~m].mean()) / sd_hat) if sd_hat > 0 else np.nan
        out["g3_yhat_ks"] = float(stats.ks_2samp(y_hat[m], y_hat[~m]).statistic)
        out["g3_residual_skewness"] = float(stats.skew(resid)) if np.std(resid) > 0 else np.nan
        out["g3_residual_excess_kurtosis"] = float(stats.kurtosis(resid)) if np.std(resid) > 0 else np.nan
        out["g3_residual_heteroscedasticity"] = (
            float(stats.spearmanr(np.abs(resid), y_hat_obs).statistic) if np.std(resid) > 0 else np.nan
        )
    else:
        for key in (
            "g3_r2_observed",
            "g3_auc_mask_from_yhat_dev",
            "g3_auc_yhat_minus_z",
            "g3_yhat_std_mean_diff",
            "g3_yhat_ks",
            "g3_residual_skewness",
            "g3_residual_excess_kurtosis",
            "g3_residual_heteroscedasticity",
        ):
            out[key] = np.nan

    # G4: análogo observável do 1º teste do MechDetect (máscara verdadeira × permutada)
    true_aucs = _cv_auc(z, m, "boosting", n_folds=N_FOLDS_G4)
    perm = np.random.default_rng(SEED).permutation(m)
    perm_aucs = _cv_auc(z, perm, "boosting", n_folds=N_FOLDS_G4)
    if np.isfinite(true_aucs).all() and np.isfinite(perm_aucs).all():
        out["g4_auc_true_minus_permuted"] = float(true_aucs.mean() - perm_aucs.mean())
        out["g4_mann_whitney_p"] = float(stats.mannwhitneyu(true_aucs, perm_aucs, alternative="greater").pvalue)
    else:
        out["g4_auc_true_minus_permuted"] = np.nan
        out["g4_mann_whitney_p"] = np.nan

    values = [v for key, v in out.items() if key.startswith("g")]
    out["status_undefined"] = int(sum(1 for v in values if not np.isfinite(v)))
    return out


def schema() -> list[dict]:
    """Nome, grupo e descrição curta de cada descritor (base da tabela de definições do Cap3)."""
    groups = {"g0": "G0", "g1": "G1", "g2": "G2", "g3": "G3", "g4": "G4"}
    rng = np.random.default_rng(1)
    demo = ObservedInstance(
        sample_id="schema",
        y=np.where(rng.random(200) < 0.2, np.nan, rng.normal(size=200)),
        z=pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"]),
    )
    names = [k for k in compute(demo) if k[:2] in groups]
    return [{"name": n, "group": groups[n[:2]], "version": FEATURE_SCHEMA_VERSION} for n in names]
