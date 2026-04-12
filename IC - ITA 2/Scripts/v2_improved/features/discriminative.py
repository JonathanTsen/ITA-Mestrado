"""
Features discriminativas específicas para distinguir MCAR, MAR e MNAR.

Estas features são baseadas nas definições teóricas dos mecanismos:
- MCAR: P(missing|X_obs, X_miss) = P(missing) - completamente aleatório
- MAR: P(missing|X_obs, X_miss) = P(missing|X_obs) - depende apenas de observados
- MNAR: P(missing|X_obs, X_miss) ≠ P(missing|X_obs) - depende do próprio valor faltante

6 features originais + 5 novas features para detecção de MNAR.
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

    6 features originais + 5 novas para detecção de MNAR:
    - X0_ks_obs_vs_imputed: KS entre X0 observado e X0 imputado
    - X0_tail_missing_ratio: taxa de missing na cauda vs centro
    - mask_entropy: entropia de Shannon dos runs de missing
    - X0_censoring_score: correlação Spearman rank(X0) vs máscara
    - X0_mean_shift_X1_to_X4: shift médio de X1-X4 por grupo missing/obs

    Args:
        df: DataFrame com X0 (missing), X1, X2, X3, X4 (observados)

    Returns:
        Dict com 11 features discriminativas
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

    # =========================================
    # 4. FEATURES MNAR: X0 observado vs imputado
    # =========================================
    X0_obs = df["X0"].dropna().values
    X0_full = df["X0"].fillna(df["X0"].median()).values

    # KS entre X0 observado e X0 imputado (mediana)
    if len(X0_obs) > 1 and n_missing > 0:
        ks_stat, _ = stats.ks_2samp(X0_obs, X0_full)
        feats["X0_ks_obs_vs_imputed"] = ks_stat
    else:
        feats["X0_ks_obs_vs_imputed"] = 0.0

    # Taxa de missing na cauda (Q4) vs centro (Q2)
    try:
        quartiles = np.percentile(X0_full, [25, 50, 75])
        q2_mask = (X0_full >= quartiles[0]) & (X0_full < quartiles[1])
        q4_mask = X0_full >= quartiles[2]

        n_q2 = q2_mask.sum()
        n_q4 = q4_mask.sum()

        if n_q2 > 0 and n_q4 > 0:
            missing_rate_q4 = mask[q4_mask].mean()
            missing_rate_q2 = mask[q2_mask].mean()
            if missing_rate_q2 > 1e-10:
                feats["X0_tail_missing_ratio"] = missing_rate_q4 / missing_rate_q2
            else:
                feats["X0_tail_missing_ratio"] = missing_rate_q4 * 10 if missing_rate_q4 > 0 else 1.0
        else:
            feats["X0_tail_missing_ratio"] = 1.0
    except Exception:
        feats["X0_tail_missing_ratio"] = 1.0

    # Entropia de Shannon dos comprimentos de runs consecutivos
    try:
        runs = []
        current_len = 1
        for i in range(1, len(mask)):
            if mask[i] == mask[i - 1]:
                current_len += 1
            else:
                runs.append(current_len)
                current_len = 1
        runs.append(current_len)

        runs = np.array(runs, dtype=float)
        runs_prob = runs / runs.sum()
        feats["mask_entropy"] = float(-np.sum(runs_prob * np.log2(runs_prob + 1e-10)))
    except Exception:
        feats["mask_entropy"] = 0.0

    # Correlação de Spearman: rank(X0_imputado) vs máscara
    try:
        corr, _ = stats.spearmanr(X0_full, mask)
        feats["X0_censoring_score"] = abs(float(corr)) if not np.isnan(corr) else 0.0
    except Exception:
        feats["X0_censoring_score"] = 0.0

    # Mean shift de X1-X4 entre grupos missing/observed
    try:
        shifts = []
        for col in ["X1", "X2", "X3", "X4"]:
            xi = df[col].values
            xi_miss = xi[mask == 1]
            xi_obs = xi[mask == 0]
            if len(xi_miss) > 0 and len(xi_obs) > 0:
                shifts.append(abs(np.mean(xi_miss) - np.mean(xi_obs)))
        feats["X0_mean_shift_X1_to_X4"] = float(np.mean(shifts)) if shifts else 0.0
    except Exception:
        feats["X0_mean_shift_X1_to_X4"] = 0.0

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
        "X0_ks_obs_vs_imputed": 0.0,
        "X0_tail_missing_ratio": 1.0,
        "mask_entropy": 0.0,
        "X0_censoring_score": 0.0,
        "X0_mean_shift_X1_to_X4": 0.0,
    }
