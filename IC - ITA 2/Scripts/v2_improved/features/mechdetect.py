"""
Features MechDetect para classificação de mecanismos de missing data.

Baseado em: Jung et al. (2024) - MechDetect
Abordagem de 3 tarefas AUC-ROC que medem a capacidade preditiva do missing:
- Complete: usa X0 observado + X1-X4 para prever máscara
- Shuffled: mesma tarefa com máscara permutada (baseline ~0.5)
- Excluded: usa apenas X1-X4 para prever máscara

Usa LogisticRegression (rápido) — o sinal discriminativo está nos deltas
entre tarefas, não no AUC absoluto.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats


def extract_mechdetect_features(df: pd.DataFrame) -> dict:
    """
    Extrai 6 features MechDetect baseadas em 3 tarefas AUC-ROC.

    Args:
        df: DataFrame com colunas X0 (com missing), X1, X2, X3, X4

    Returns:
        Dict com 6 features MechDetect
    """
    mask = df["X0"].isna().astype(int).values
    n_missing = mask.sum()
    n_total = len(mask)

    if n_missing < 5 or (n_total - n_missing) < 5:
        return _default_mechdetect_features()

    X0_imputed = df["X0"].fillna(df["X0"].median()).values
    X_other = df[["X1", "X2", "X3", "X4"]].values

    X_complete = np.column_stack([X0_imputed, X_other])
    X_excluded = X_other

    min_class = min(n_missing, n_total - n_missing)
    n_folds = min(3, max(2, min_class))

    rng = np.random.RandomState(42)
    mask_shuffled = mask.copy()
    rng.shuffle(mask_shuffled)

    fold_aucs_c, fold_aucs_s, fold_aucs_e = _compute_all_tasks(
        X_complete, X_excluded, mask, mask_shuffled, n_folds
    )

    auc_complete = float(np.mean(fold_aucs_c)) if fold_aucs_c else 0.5
    auc_shuffled = float(np.mean(fold_aucs_s)) if fold_aucs_s else 0.5
    auc_excluded = float(np.mean(fold_aucs_e)) if fold_aucs_e else 0.5

    mwu_pvalue = 1.0
    if len(fold_aucs_c) >= 2 and len(fold_aucs_s) >= 2:
        try:
            _, mwu_pvalue = stats.mannwhitneyu(
                fold_aucs_c, fold_aucs_s, alternative="two-sided"
            )
            mwu_pvalue = float(mwu_pvalue)
        except Exception:
            pass

    return {
        "mechdetect_auc_complete": auc_complete,
        "mechdetect_auc_shuffled": auc_shuffled,
        "mechdetect_auc_excluded": auc_excluded,
        "mechdetect_delta_complete_shuffled": auc_complete - auc_shuffled,
        "mechdetect_delta_complete_excluded": auc_complete - auc_excluded,
        "mechdetect_mwu_pvalue": mwu_pvalue,
    }


def _fit_and_auc(X_train, y_train, X_test, y_test):
    """Treina LogisticRegression e retorna AUC-ROC."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_train)
    proba = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_test, proba)


def _compute_all_tasks(X_complete, X_excluded, mask, mask_shuffled, n_folds):
    """Treina Complete, Shuffled e Excluded em uma única passada de CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    aucs_c, aucs_s, aucs_e = [], [], []

    try:
        for train_idx, test_idx in skf.split(X_complete, mask):
            if len(np.unique(mask[test_idx])) < 2:
                continue

            aucs_c.append(_fit_and_auc(
                X_complete[train_idx], mask[train_idx],
                X_complete[test_idx], mask[test_idx]
            ))
            aucs_s.append(_fit_and_auc(
                X_complete[train_idx], mask_shuffled[train_idx],
                X_complete[test_idx], mask[test_idx]
            ))
            aucs_e.append(_fit_and_auc(
                X_excluded[train_idx], mask[train_idx],
                X_excluded[test_idx], mask[test_idx]
            ))
    except Exception:
        pass

    return aucs_c, aucs_s, aucs_e


def _default_mechdetect_features():
    """Retorna features padrão quando não há missing suficiente."""
    return {
        "mechdetect_auc_complete": 0.5,
        "mechdetect_auc_shuffled": 0.5,
        "mechdetect_auc_excluded": 0.5,
        "mechdetect_delta_complete_shuffled": 0.0,
        "mechdetect_delta_complete_excluded": 0.0,
        "mechdetect_mwu_pvalue": 1.0,
    }
