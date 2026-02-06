"""
Features estatísticas robustas para classificação de mecanismos de missing data.

Este módulo implementa features estatísticas que capturam características
discriminativas entre MCAR, MAR e MNAR.

NOTA: Versão otimizada com apenas 4 features relevantes identificadas
pela análise de relevância (analyze_feature_relevance.py).
"""
import numpy as np
import pandas as pd


def extract_statistical_features(df: pd.DataFrame) -> dict:
    """
    Extrai features estatísticas de X0 (variável com missing).
    
    Features mantidas após análise de relevância:
    - X0_mean: média dos valores observados
    - X0_q25: percentil 25
    - X0_q50: mediana
    - X0_q75: percentil 75
    
    Args:
        df: DataFrame com colunas X0, X1, X2, X3, X4 (X0 tem missing)
    
    Returns:
        Dict com 4 features estatísticas
    """
    feats = {}
    
    # Valores observados de X0
    X0_obs = df["X0"].dropna().values
    
    # Features de distribuição de X0 (apenas as 4 relevantes)
    if len(X0_obs) > 1:
        feats["X0_mean"] = np.mean(X0_obs)
        feats["X0_q25"] = np.percentile(X0_obs, 25)
        feats["X0_q50"] = np.percentile(X0_obs, 50)
        feats["X0_q75"] = np.percentile(X0_obs, 75)
    else:
        feats["X0_mean"] = 0.5
        feats["X0_q25"] = 0.5
        feats["X0_q50"] = 0.5
        feats["X0_q75"] = 0.5
    
    return feats
