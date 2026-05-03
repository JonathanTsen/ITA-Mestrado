"""
Features avançadas para o Nível 2 (MAR vs MNAR) — STEP 03.

Três famílias:
1. Divergência de imputação (3 features) — se MNAR, imputações divergem
2. Independência condicional (2 features) — testa definição de MAR
3. KDE density ratio (2 features) — captura shifts não-lineares

Total: 7 features novas, usadas apenas no L2.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, spearmanr, wasserstein_distance
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression


def extract_advanced_l2_features(df: pd.DataFrame) -> dict:
    """Extrai 7 features avançadas focadas em separar MAR de MNAR."""
    feats = {}
    feats.update(_compute_imputation_divergence(df))
    feats.update(_compute_conditional_independence(df))
    feats.update(_compute_density_ratio(df))
    return feats


# =========================================================================
# Família 1: Divergência entre Métodos de Imputação (3 features)
# =========================================================================
def _compute_imputation_divergence(df: pd.DataFrame) -> dict:
    """Compara 3 métodos de imputação para detectar MNAR.

    MAR: imputações concordam (X1-X4 explicam X0).
    MNAR: imputações divergem (nenhum método recupera a dependência X0→mask).
    """
    defaults = {
        "adv_imputation_divergence_ks": 0.0,
        "adv_imputation_divergence_wasserstein": 0.0,
        "adv_imputation_cv": 0.0,
    }

    X0 = df["X0"].values
    mask = np.isnan(X0)
    n_missing = mask.sum()

    if n_missing < 5 or (~mask).sum() < 5:
        return defaults

    try:
        X_full = df[["X0", "X1", "X2", "X3", "X4"]].values

        # Método 1: Median
        imp_median = SimpleImputer(strategy="median").fit_transform(X_full)[:, 0]

        # Método 2: KNN (usa X1-X4 como vizinhos)
        imp_knn = KNNImputer(n_neighbors=min(5, (~mask).sum() - 1)).fit_transform(X_full)[:, 0]

        # Método 3: MICE/Iterative
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imp_mice = IterativeImputer(max_iter=10, random_state=42, sample_posterior=False).fit_transform(X_full)[:, 0]

        # Extrair apenas valores imputados (onde era NaN)
        vals = {
            "median": imp_median[mask],
            "knn": imp_knn[mask],
            "mice": imp_mice[mask],
        }

        # KS entre pares
        pairs = [("median", "knn"), ("median", "mice"), ("knn", "mice")]
        ks_stats = []
        ws_dists = []
        for a, b in pairs:
            if len(vals[a]) >= 2 and len(vals[b]) >= 2:
                ks_stats.append(ks_2samp(vals[a], vals[b]).statistic)
                ws_dists.append(wasserstein_distance(vals[a], vals[b]))

        if not ks_stats:
            return defaults

        # CV das médias imputadas
        means = [v.mean() for v in vals.values()]
        cv = np.std(means) / max(np.mean(np.abs(means)), 1e-10)

        return {
            "adv_imputation_divergence_ks": float(np.mean(ks_stats)),
            "adv_imputation_divergence_wasserstein": float(np.max(ws_dists)),
            "adv_imputation_cv": float(min(cv, 10.0)),  # cap
        }
    except Exception:
        return defaults


# =========================================================================
# Família 2: Teste de Independência Condicional (2 features)
# =========================================================================
def _compute_conditional_independence(df: pd.DataFrame) -> dict:
    """Testa mask ⊥ X0 | X_obs usando resíduos parciados.

    MAR: dado X_obs, mask é independente de X0 → resíduos não correlacionam.
    MNAR: dado X_obs, mask depende de X0 → resíduos correlacionam.
    """
    defaults = {
        "adv_partial_dcor_X0_mask": 0.0,
        "adv_residual_spearman_X0_mask": 0.0,
    }

    mask_vec = df["X0"].isna().astype(float).values
    X0_imputed = df["X0"].fillna(df["X0"].median()).values
    X_obs = df[["X1", "X2", "X3", "X4"]].values

    if len(X0_imputed) < 20 or mask_vec.sum() < 5:
        return defaults

    try:
        # Residualizar X0 e mask contra X_obs
        lr_x0 = LinearRegression().fit(X_obs, X0_imputed)
        residual_x0 = X0_imputed - lr_x0.predict(X_obs)

        lr_mask = LinearRegression().fit(X_obs, mask_vec)
        residual_mask = mask_vec - lr_mask.predict(X_obs)

        # Distance correlation (zero iff independent)
        try:
            import dcor

            partial_dcor = dcor.distance_correlation(residual_x0, residual_mask)
        except (ImportError, Exception):
            # Fallback: usar correlação absoluta se dcor não disponível
            partial_dcor = abs(np.corrcoef(residual_x0, residual_mask)[0, 1])

        # Spearman parcial
        rho, _ = spearmanr(residual_x0, residual_mask)

        return {
            "adv_partial_dcor_X0_mask": float(partial_dcor) if np.isfinite(partial_dcor) else 0.0,
            "adv_residual_spearman_X0_mask": float(abs(rho)) if np.isfinite(rho) else 0.0,
        }
    except Exception:
        return defaults


# =========================================================================
# Família 3: KDE Density Ratio (2 features)
# =========================================================================
def _compute_density_ratio(df: pd.DataFrame) -> dict:
    """Estima o ratio de densidades p(X0|obs)/p(X0|all).

    MCAR: ratio ≈ 1 em todo lugar (densidades iguais).
    MNAR: ratio varia (caudas truncadas mudam a densidade).
    """
    defaults = {
        "adv_density_ratio_range": 0.0,
        "adv_density_ratio_tail_asym": 1.0,
    }

    X0 = df["X0"].values
    mask = np.isnan(X0)
    X0_obs = X0[~mask]

    if mask.sum() < 10 or (~mask).sum() < 10:
        return defaults

    # Verificar variância suficiente
    if np.std(X0_obs) < 1e-10:
        return defaults

    try:
        X0_full = np.where(mask, np.median(X0_obs), X0)

        kde_obs = gaussian_kde(X0_obs)
        kde_all = gaussian_kde(X0_full)

        # Avaliar em quantis da distribuição observada
        eval_points = np.percentile(X0_obs, np.linspace(5, 95, 10))
        # Garantir pontos únicos
        if len(np.unique(eval_points)) < 3:
            return defaults

        d_obs = kde_obs(eval_points)
        d_all = kde_all(eval_points)

        ratio = d_obs / np.maximum(d_all, 1e-10)

        ratio_range = float(ratio.max() - ratio.min())
        ratio_tail = float(ratio[-1] / max(ratio[0], 1e-10))

        return {
            "adv_density_ratio_range": min(ratio_range, 10.0),  # cap
            "adv_density_ratio_tail_asym": min(max(ratio_tail, 0.01), 100.0),  # cap
        }
    except (np.linalg.LinAlgError, ValueError, Exception):
        return defaults
