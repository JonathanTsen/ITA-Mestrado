"""
Features discriminativas específicas para distinguir MCAR, MAR e MNAR.

Estas features são baseadas nas definições teóricas dos mecanismos:
- MCAR: P(missing|X_obs, X_miss) = P(missing) - completamente aleatório
- MAR: P(missing|X_obs, X_miss) = P(missing|X_obs) - depende apenas de observados
- MNAR: P(missing|X_obs, X_miss) ≠ P(missing|X_obs) - depende do próprio valor faltante

NOTA: Versão otimizada com apenas 6 features relevantes identificadas
pela análise de relevância (analyze_feature_relevance.py).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def extract_discriminative_features(df: pd.DataFrame) -> dict:
    """
    Extrai features discriminativas entre MCAR, MAR e MNAR.
    
    Features mantidas após análise de relevância:
    - auc_mask_from_Xobs: AUC de prever missing usando X1-X4
    - coef_X1_abs: coeficiente absoluto de X1 no modelo logístico
    - log_pval_X1_mask: -log10 do p-valor da correlação X1 vs mask
    - X1_mean_diff: diferença de média de X1 entre grupos missing/observed
    - X1_mannwhitney_pval: p-valor do teste Mann-Whitney
    - little_proxy_score: proxy do teste de Little (média KS stats)
    
    Args:
        df: DataFrame com X0 (missing), X1, X2, X3, X4 (observados)
    
    Returns:
        Dict com 6 features discriminativas
    """
    feats = {}
    
    mask = df["X0"].isna().astype(int).values
    n_missing = mask.sum()
    n_total = len(mask)
    
    if n_missing == 0 or n_missing == n_total:
        return _default_discriminative_features()
    
    # =========================================
    # 1. MODELO PREDITIVO: AUC e coef_X1_abs
    # =========================================
    X_predictors = df[["X1", "X2", "X3", "X4"]].values
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_predictors)
        
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_scaled, mask)
        
        proba = clf.predict_proba(X_scaled)[:, 1]
        feats["auc_mask_from_Xobs"] = roc_auc_score(mask, proba)
        feats["coef_X1_abs"] = np.abs(clf.coef_.ravel()[0])
        
    except Exception:
        feats["auc_mask_from_Xobs"] = 0.5
        feats["coef_X1_abs"] = 0.0
    
    # =========================================
    # 2. TESTE MAR: X1 vs mask
    # =========================================
    X1 = df["X1"].values
    
    # log p-valor da correlação
    if np.std(X1) > 0:
        _, pval_X1_mask = stats.pointbiserialr(mask, X1)
        feats["log_pval_X1_mask"] = -np.log10(max(pval_X1_mask, 1e-100))
    else:
        feats["log_pval_X1_mask"] = 0.0
    
    # Diferença de média e Mann-Whitney
    X1_when_missing = X1[mask == 1]
    X1_when_observed = X1[mask == 0]
    
    if len(X1_when_missing) > 0 and len(X1_when_observed) > 0:
        feats["X1_mean_diff"] = np.mean(X1_when_missing) - np.mean(X1_when_observed)
        
        try:
            _, u_pval = stats.mannwhitneyu(X1_when_missing, X1_when_observed, alternative="two-sided")
            feats["X1_mannwhitney_pval"] = u_pval
        except Exception:
            feats["X1_mannwhitney_pval"] = 1.0
    else:
        feats["X1_mean_diff"] = 0.0
        feats["X1_mannwhitney_pval"] = 1.0
    
    # =========================================
    # 3. LITTLE'S MCAR TEST PROXY
    # =========================================
    try:
        chi2_sum = 0.0
        for col in ["X1", "X2", "X3", "X4"]:
            xi = df[col].values
            xi_when_missing = xi[mask == 1]
            xi_when_observed = xi[mask == 0]
            
            if len(xi_when_missing) > 0 and len(xi_when_observed) > 0:
                ks_stat, _ = stats.ks_2samp(xi_when_missing, xi_when_observed)
                chi2_sum += ks_stat
        
        feats["little_proxy_score"] = chi2_sum / 4
    except Exception:
        feats["little_proxy_score"] = 0.0
    
    return feats


def _default_discriminative_features():
    """Retorna features padrão quando não há missing suficiente."""
    return {
        "auc_mask_from_Xobs": 0.5,
        "coef_X1_abs": 0.0,
        "log_pval_X1_mask": 0.0,
        "X1_mean_diff": 0.0,
        "X1_mannwhitney_pval": 1.0,
        "little_proxy_score": 0.0,
    }
