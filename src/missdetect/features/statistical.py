"""
Features estatísticas invariantes para classificação de mecanismos de missing data.

Features invariantes ao dataset: medem propriedades relativas (ratios, diffs)
em vez de valores absolutos, evitando fingerprints do dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def extract_statistical_features(df: pd.DataFrame) -> dict:
    """
    Extrai features estatísticas invariantes de X0 (variável com missing).

    Features:
    - X0_missing_rate: proporção de NaN em X0
    - X0_obs_vs_full_ratio: média observados / média imputados (mediana)
    - X0_iqr_ratio: IQR observados / IQR imputados
    - X0_obs_skew_diff: skew(observados) - skew(imputados)

    Args:
        df: DataFrame com colunas X0, X1, X2, X3, X4 (X0 tem missing)

    Returns:
        Dict com 4 features estatísticas invariantes
    """
    feats = {}

    total = len(df)
    n_missing = df["X0"].isna().sum()

    feats["X0_missing_rate"] = n_missing / total if total > 0 else 0.0

    X0_obs = df["X0"].dropna().values

    if len(X0_obs) < 2 or n_missing == 0:
        feats["X0_obs_vs_full_ratio"] = 1.0
        feats["X0_iqr_ratio"] = 1.0
        feats["X0_obs_skew_diff"] = 0.0
        return feats

    # Imputação simples com mediana para obter distribuição "full"
    X0_full = df["X0"].fillna(df["X0"].median()).values

    # Ratio de médias: observados vs full (imputado)
    mean_full = np.mean(X0_full)
    mean_obs = np.mean(X0_obs)
    if abs(mean_full) > 1e-10:
        feats["X0_obs_vs_full_ratio"] = mean_obs / mean_full
    else:
        feats["X0_obs_vs_full_ratio"] = 1.0

    # Ratio de IQR: observados vs full
    iqr_obs = np.percentile(X0_obs, 75) - np.percentile(X0_obs, 25)
    iqr_full = np.percentile(X0_full, 75) - np.percentile(X0_full, 25)
    if abs(iqr_full) > 1e-10:
        feats["X0_iqr_ratio"] = iqr_obs / iqr_full
    else:
        feats["X0_iqr_ratio"] = 1.0

    # Diferença de skewness
    skew_obs = sp_stats.skew(X0_obs)
    skew_full = sp_stats.skew(X0_full)
    feats["X0_obs_skew_diff"] = float(skew_obs - skew_full)

    return feats
