"""
Features CAAFE-inspired focadas na confusão MCAR vs MNAR.

Baseado em CAAFE (Hollmann et al., NeurIPS 2023) — features geradas
para resolver o gargalo principal: MNAR que se disfarça de MCAR.

MNAR depende de X0, mas X0 está faltante onde precisamos medi-lo.
Estas features atacam esse problema circular com medidas indiretas:

1. missing_rate_by_X0_quantile — taxa de missing por faixa de X0 imputado
2. X0_obs_tail_asymmetry — assimetria nas caudas de X0 observado
3. X0_kurtosis_excess — excesso de curtose em X0 observado
4. conditional_entropy_X0_mask — entropia condicional mask|X0 discretizado
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def extract_caafe_mnar_features(df: pd.DataFrame) -> dict:
    """
    Extrai 4 features focadas em separar MCAR de MNAR.

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

    feats = {}

    # =========================================
    # 1. Missing rate por quantil de X0 imputado
    # =========================================
    # MNAR: taxa de missing varia por faixa de X0 (valores altos/baixos faltam mais)
    # MCAR: taxa de missing uniforme em todas as faixas
    try:
        quantiles = np.percentile(X0_imputed, [25, 50, 75])
        bins = [-np.inf, quantiles[0], quantiles[1], quantiles[2], np.inf]
        bin_indices = np.digitize(X0_imputed, bins[1:-1])  # 0,1,2,3

        rates = []
        for b in range(4):
            bin_mask = bin_indices == b
            if bin_mask.sum() > 0:
                rates.append(mask[bin_mask].mean())

        if len(rates) >= 2:
            # Razão entre maior e menor taxa — MNAR terá alta variância
            rates = np.array(rates)
            feats["caafe_missing_rate_by_quantile"] = float(
                rates.max() / max(rates.min(), 1e-10)
            )
            # Cap para evitar explosão quando min rate ≈ 0
            feats["caafe_missing_rate_by_quantile"] = min(
                feats["caafe_missing_rate_by_quantile"], 20.0
            )
        else:
            feats["caafe_missing_rate_by_quantile"] = 1.0
    except Exception:
        feats["caafe_missing_rate_by_quantile"] = 1.0

    # =========================================
    # 2. Assimetria nas caudas de X0 observado
    # =========================================
    # MNAR trunca uma cauda (valores extremos são removidos)
    # Medimos a diferença de densidade entre cauda inferior e superior
    try:
        if len(X0_obs) > 10:
            p10, p50, p90 = np.percentile(X0_obs, [10, 50, 90])

            # Proporção de obs na cauda inferior vs superior
            n_lower = np.sum(X0_obs < p10)
            n_upper = np.sum(X0_obs > p90)

            if n_lower + n_upper > 0:
                # 0 = simétrico, >0 ou <0 = assimétrico
                feats["caafe_tail_asymmetry"] = float(
                    abs(n_upper - n_lower) / (n_lower + n_upper)
                )
            else:
                feats["caafe_tail_asymmetry"] = 0.0
        else:
            feats["caafe_tail_asymmetry"] = 0.0
    except Exception:
        feats["caafe_tail_asymmetry"] = 0.0

    # =========================================
    # 3. Excesso de curtose em X0 observado
    # =========================================
    # MNAR muda a forma da distribuição de X0 (truncar valores remove caudas)
    # Curtose < 0 (platicúrtica) sugere truncamento; > 0 sugere concentração
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
            # Discretiza X0 em 5 bins
            n_bins = min(5, len(np.unique(X0_imputed)))
            _, bin_edges = np.histogram(X0_imputed, bins=n_bins)
            X0_disc = np.digitize(X0_imputed, bin_edges[1:-1])

            # H(mask) — entropia marginal
            p_miss = n_missing / n_total
            h_mask = -p_miss * np.log2(max(p_miss, 1e-10)) - (1 - p_miss) * np.log2(
                max(1 - p_miss, 1e-10)
            )

            # H(mask|X0) — entropia condicional
            h_cond = 0.0
            for b in np.unique(X0_disc):
                bin_mask = X0_disc == b
                n_bin = bin_mask.sum()
                if n_bin > 0:
                    p_miss_bin = mask[bin_mask].mean()
                    p_bin = n_bin / n_total
                    if 0 < p_miss_bin < 1:
                        h_bin = -p_miss_bin * np.log2(p_miss_bin) - (
                            1 - p_miss_bin
                        ) * np.log2(1 - p_miss_bin)
                    else:
                        h_bin = 0.0
                    h_cond += p_bin * h_bin

            # Informação mútua normalizada: 0 = independente, 1 = totalmente dependente
            if h_mask > 1e-10:
                feats["caafe_cond_entropy_X0_mask"] = float(
                    (h_mask - h_cond) / h_mask
                )
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
        "caafe_missing_rate_by_quantile": 1.0,
        "caafe_tail_asymmetry": 0.0,
        "caafe_kurtosis_excess": 0.0,
        "caafe_cond_entropy_X0_mask": 0.0,
    }
