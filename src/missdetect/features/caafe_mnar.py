"""
Features CAAFE-inspired focadas na confusão MCAR vs MNAR.

Baseado em CAAFE (Hollmann et al., NeurIPS 2023) — features geradas
para resolver o gargalo principal: MNAR que se disfarça de MCAR.

MNAR depende de X0, mas X0 está faltante onde precisamos medi-lo.
Estas features atacam esse problema circular com medidas indiretas:

1. auc_self_delta     — delta AUC ao adicionar X0_imputado para prever a própria máscara
2. kl_density         — KL divergence entre distribuição de X0_imp onde está ausente vs presente
3. X0_kurtosis_excess — excesso de curtose em X0 observado
4. conditional_entropy_X0_mask — entropia condicional mask|X0 discretizado
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# =========================================
# Helpers internos (evitar import circular com validar_rotulos_v2)
# =========================================

def _cv_auc_simple(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> float:
    """AUC via StratifiedKFold sem permutações — rápido, para uso interno."""
    n_splits = min(n_splits, max(2, int(np.min(np.bincount(y)))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=seed, n_jobs=1
        )
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs)) if aucs else 0.5


def _auc_self_delta(df: pd.DataFrame, missing_col: str = "X0", random_state: int = 42) -> float:
    """Delta AUC ao incluir X0_imputado como feature para prever a própria máscara.

    MNAR: X0 prediz sua própria ausência → delta > 0.
    MCAR: X0 não agrega informação → delta ≈ 0.
    """
    mask = df[missing_col].isna().astype(int).values
    if mask.sum() < 5 or (1 - mask).sum() < 5:
        return 0.0
    X0_imp = df[missing_col].fillna(df[missing_col].median()).values.reshape(-1, 1)
    other_cols = [c for c in df.columns if c != missing_col]
    X_obs = SimpleImputer(strategy="median").fit_transform(df[other_cols].values)
    try:
        auc_without = _cv_auc_simple(X_obs, mask, random_state)
        auc_with = _cv_auc_simple(np.hstack([X_obs, X0_imp]), mask, random_state + 1)
        return float(max(0.0, auc_with - auc_without))
    except Exception:
        return 0.0


def _kl_density_score(df: pd.DataFrame, missing_col: str = "X0") -> float:
    """KL divergence entre distribuição de X0_imputado onde X0 está ausente vs presente.

    MNAR: valores específicos de X0 tendem a faltar → distribuições diferem → KL > 0.
    MCAR: distribuições similares → KL ≈ 0.
    """
    mask = df[missing_col].isna().astype(int).values
    if mask.sum() < 5 or (1 - mask).sum() < 5:
        return 0.0
    X0_imp = df[missing_col].fillna(df[missing_col].median()).values
    x_miss = X0_imp[mask == 1]
    x_obs_vals = X0_imp[mask == 0]
    x_min, x_max = X0_imp.min(), X0_imp.max()
    if x_max - x_min < 1e-10:
        return 0.0
    bins = np.linspace(x_min, x_max + 1e-10, 21)
    eps = 1e-10
    h_miss = np.histogram(x_miss, bins=bins)[0].astype(float) + eps
    h_obs = np.histogram(x_obs_vals, bins=bins)[0].astype(float) + eps
    h_miss /= h_miss.sum()
    h_obs /= h_obs.sum()
    return float(np.sum(h_obs * np.log(h_obs / h_miss)))


# =========================================
# Função pública principal
# =========================================

def extract_caafe_mnar_features(df: pd.DataFrame) -> dict:
    """Extrai 4 features focadas em separar MCAR de MNAR.

    Args:
        df: DataFrame com X0 (missing), X1-X4 (observados)

    Returns:
        Dict com 4 features CAAFE-MNAR
    """
    mask = df["X0"].isna().astype(int).values
    n_missing = mask.sum()
    n_total = len(mask)

    if n_missing == 0 or n_missing == n_total:
        return _default_caafe_features()

    X0_obs = df["X0"].dropna().values
    X0_imputed = df["X0"].fillna(df["X0"].median()).values

    feats: dict = {}

    # =========================================
    # 1. Delta AUC ao incluir X0_imputado (substitui missing_rate_by_quantile)
    # =========================================
    feats["caafe_auc_self_delta"] = _auc_self_delta(df, missing_col="X0", random_state=42)

    # =========================================
    # 2. KL divergence entre grupos de ausência (substitui tail_asymmetry)
    # =========================================
    feats["caafe_kl_density"] = _kl_density_score(df, missing_col="X0")

    # =========================================
    # 3. Excesso de curtose em X0 observado
    # =========================================
    # MNAR muda a forma da distribuição de X0 (truncar valores remove caudas)
    try:
        if len(X0_obs) > 10:
            feats["caafe_kurtosis_excess"] = float(sp_stats.kurtosis(X0_obs, fisher=True))
        else:
            feats["caafe_kurtosis_excess"] = 0.0
    except Exception:
        feats["caafe_kurtosis_excess"] = 0.0

    # =========================================
    # 4. Entropia condicional mask | X0 discretizado
    # =========================================
    # Se mask e X0 são independentes (MCAR): H(mask|X0) ≈ H(mask)
    # Se dependentes (MNAR): H(mask|X0) < H(mask) → mutual info > 0
    try:
        if len(X0_imputed) > 10:
            n_bins = min(5, len(np.unique(X0_imputed)))
            _, bin_edges = np.histogram(X0_imputed, bins=n_bins)
            X0_disc = np.digitize(X0_imputed, bin_edges[1:-1])

            p_miss = n_missing / n_total
            h_mask = -p_miss * np.log2(max(p_miss, 1e-10)) - (1 - p_miss) * np.log2(max(1 - p_miss, 1e-10))

            h_cond = 0.0
            for b in np.unique(X0_disc):
                bin_mask = X0_disc == b
                n_bin = bin_mask.sum()
                if n_bin > 0:
                    p_miss_bin = mask[bin_mask].mean()
                    p_bin = n_bin / n_total
                    if 0 < p_miss_bin < 1:
                        h_bin = -p_miss_bin * np.log2(p_miss_bin) - (1 - p_miss_bin) * np.log2(1 - p_miss_bin)
                    else:
                        h_bin = 0.0
                    h_cond += p_bin * h_bin

            if h_mask > 1e-10:
                feats["caafe_cond_entropy_X0_mask"] = float((h_mask - h_cond) / h_mask)
            else:
                feats["caafe_cond_entropy_X0_mask"] = 0.0
        else:
            feats["caafe_cond_entropy_X0_mask"] = 0.0
    except Exception:
        feats["caafe_cond_entropy_X0_mask"] = 0.0

    return feats


def _default_caafe_features() -> dict:
    """Retorna features padrão quando não há missing suficiente."""
    return {
        "caafe_auc_self_delta": 0.0,
        "caafe_kl_density": 0.0,
        "caafe_kurtosis_excess": 0.0,
        "caafe_cond_entropy_X0_mask": 0.0,
    }
