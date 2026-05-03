"""
Classificação Hierárquica V3+ — Plano 3.

Evolução de train_hierarchical_variants.py com:
- STEP 04: Roteamento probabilístico L1→L2 (hard, threshold, soft3zone, fullprob)
- STEP 01: Integração com Cleanlab (--clean-labels)
- STEP 02: CatBoost + XGBoost + Optuna tuning (--optimize)
- Calibração de probabilidades com CalibratedClassifierCV

Mantém compatibilidade: roda V3 original como baseline para comparação.

Uso:
    # Baseline V3 (reproduz resultado original)
    python train_hierarchical_v3plus.py --data real --experiment step05_v3plus --llm-model gemini-3.1-pro-preview

    # V3+ com roteamento probabilístico
    python train_hierarchical_v3plus.py --data real --experiment step05_v3plus --routing fullprob

    # V3+ com labels limpos (rodar clean_labels.py antes)
    python train_hierarchical_v3plus.py --data real --experiment step05_v3plus --routing fullprob --clean-labels weight

    # Testar todas as estratégias de routing
    python train_hierarchical_v3plus.py --data real --experiment step05_v3plus --routing all

    # Com CatBoost/XGBoost + Optuna tuning
    python train_hierarchical_v3plus.py --data real --experiment step05_v3plus --routing fullprob --optimize --n-trials 100
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import OUTPUT_BASE, get_output_dir

warnings.filterwarnings("ignore")

# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(description="Hierárquica V3+ (Plano 3)")
parser.add_argument("--data", choices=["sintetico", "real"], required=True)
parser.add_argument("--experiment", required=True)
parser.add_argument(
    "--llm-model", default="gemini-3.1-pro-preview", help="Nome do modelo LLM cujas features serão usadas"
)
parser.add_argument(
    "--routing",
    choices=["hard", "threshold", "soft3zone", "fullprob", "all"],
    default="all",
    help="Estratégia de roteamento L1→L2 (default: all = testa todas)",
)
parser.add_argument(
    "--clean-labels",
    choices=["none", "weight", "prune", "relabel"],
    default="none",
    help="Modo de limpeza de labels (requer clean_labels.py rodado antes)",
)
parser.add_argument(
    "--calibrate", action="store_true", default=True, help="Calibrar probabilidades com Platt scaling (default: True)"
)
parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
parser.add_argument(
    "--balancing",
    choices=["smote", "smote_enn", "borderline", "smote_tomek", "none"],
    default="smote",
    help="Método de balanceamento de classes (default: smote)",
)
parser.add_argument(
    "--optimize", action="store_true", help="Usar Optuna para otimizar hiperparametros de CatBoost/XGBoost (~10min)"
)
parser.add_argument("--n-trials", type=int, default=100, help="Numero de trials Optuna por nivel (default: 100)")
args = parser.parse_args()

DATA_TYPE = args.data
EXPERIMENT = args.experiment
LLM_MODEL = args.llm_model
ROUTING_MODE = args.routing
CLEAN_MODE = args.clean_labels
CALIBRATE = args.calibrate
BALANCING = args.balancing
OPTIMIZE = args.optimize
N_TRIALS = args.n_trials

# Silenciar logs verbosos do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROUTING_STRATEGIES = ["hard", "threshold", "soft3zone", "fullprob"] if ROUTING_MODE == "all" else [ROUTING_MODE]

# ==============================================================================
# PATHS
# ==============================================================================
BASELINE_DIR = get_output_dir(DATA_TYPE, "none", EXPERIMENT)
LLM_DIR = get_output_dir(DATA_TYPE, LLM_MODEL, EXPERIMENT)
HIER_DIR = os.path.join(OUTPUT_BASE, EXPERIMENT, DATA_TYPE, "hierarquico_v3plus")
os.makedirs(HIER_DIR, exist_ok=True)

for label, d in [("Baseline", BASELINE_DIR), ("LLM", LLM_DIR)]:
    fpath = os.path.join(d, "X_features.csv")
    if not os.path.exists(fpath):
        print(f"[ERRO] {label} features nao encontradas: {fpath}")
        sys.exit(1)

print("=" * 70)
print("CLASSIFICACAO HIERARQUICA V3+ (Plano 3)")
print("=" * 70)
print(f"  Dados: {DATA_TYPE}")
print(f"  LLM: {LLM_MODEL}")
print(f"  Routing: {ROUTING_STRATEGIES}")
print(f"  Clean labels: {CLEAN_MODE}")
print(f"  Calibrate: {CALIBRATE}")
print(f"  Balancing: {BALANCING}")
print(f"  Optimize: {OPTIMIZE} (n_trials={N_TRIALS})")
print(f"  Baseline: {BASELINE_DIR}")
print(f"  Output:   {HIER_DIR}")
print("=" * 70)


# ==============================================================================
# CARREGA DADOS
# ==============================================================================
X_baseline = pd.read_csv(os.path.join(BASELINE_DIR, "X_features.csv"))
X_llm_full = pd.read_csv(os.path.join(LLM_DIR, "X_features.csv"))
y = pd.read_csv(os.path.join(BASELINE_DIR, "y_labels.csv")).squeeze("columns")

groups = None
groups_path = os.path.join(BASELINE_DIR, "groups.csv")
if os.path.exists(groups_path):
    groups = pd.read_csv(groups_path).squeeze("columns")

sample_weights = None

# Aplicar limpeza de labels se solicitado
if CLEAN_MODE != "none":
    analysis_dir = os.path.join(OUTPUT_BASE, EXPERIMENT, DATA_TYPE, "label_analysis")
    if CLEAN_MODE == "weight":
        wp = os.path.join(analysis_dir, "sample_weights.csv")
        if os.path.exists(wp):
            sample_weights = pd.read_csv(wp)["sample_weight"].values
            print(f"\n  Labels: pesos carregados (mean={sample_weights.mean():.3f})")
        else:
            print("\n  [AVISO] sample_weights.csv nao encontrado. Rode clean_labels.py --action weight primeiro.")
    elif CLEAN_MODE == "prune":
        xp = os.path.join(analysis_dir, "X_features_clean.csv")
        yp = os.path.join(analysis_dir, "y_labels_clean.csv")
        if os.path.exists(xp) and os.path.exists(yp):
            X_baseline_clean = pd.read_csv(xp)
            y_clean = pd.read_csv(yp).squeeze("columns")
            # Precisamos alinhar com X_llm_full — usar índices
            # Cleanlab remove linhas por índice, precisamos do mapeamento
            gp = os.path.join(analysis_dir, "groups_clean.csv")
            if os.path.exists(gp):
                groups = pd.read_csv(gp).squeeze("columns")
            # Reconstruir X_llm_full com as mesmas linhas
            keep_mask = np.isin(
                np.arange(len(y)), np.where(np.isin(X_baseline.values[:, 0], X_baseline_clean.values[:, 0]))[0]
            )
            if keep_mask.sum() != len(y_clean):
                # Fallback: usar quality scores para identificar as linhas mantidas
                scores_path = os.path.join(analysis_dir, "label_quality_scores.csv")
                if os.path.exists(scores_path):
                    df_scores = pd.read_csv(scores_path)
                    keep_indices = df_scores[~df_scores["is_issue"]].index.values
                    # Limitar ao prune_pct
                    if len(keep_indices) > len(y_clean):
                        keep_indices = keep_indices[: len(y_clean)]
                    keep_mask = np.isin(np.arange(len(y)), keep_indices)

            X_baseline = X_baseline[keep_mask].reset_index(drop=True)
            X_llm_full = X_llm_full[keep_mask].reset_index(drop=True)
            y = y_clean
            print(f"\n  Labels: pruned {keep_mask.sum()}/{len(keep_mask)} amostras mantidas")
        else:
            print("\n  [AVISO] Dados limpos nao encontrados. Rode clean_labels.py --action prune primeiro.")
    elif CLEAN_MODE == "relabel":
        rp = os.path.join(analysis_dir, "y_labels_relabeled.csv")
        if os.path.exists(rp):
            y = pd.read_csv(rp).squeeze("columns")
            print("\n  Labels: relabeled carregados")
        else:
            print("\n  [AVISO] y_labels_relabeled.csv nao encontrado. Rode clean_labels.py --action relabel primeiro.")

# Identificar colunas por tipo
STAT_COLS = list(X_baseline.columns)
CAAFE_COLS = [c for c in X_llm_full.columns if c.startswith("caafe_")]
LLM_COLS = [c for c in X_llm_full.columns if c.startswith("llm_")]
ADV_L2_COLS = [c for c in X_llm_full.columns if c.startswith("adv_")]

FEAT_STAT = STAT_COLS
FEAT_STAT_CAAFE = STAT_COLS + CAAFE_COLS
FEAT_STAT_CAAFE_ADV = STAT_COLS + CAAFE_COLS + ADV_L2_COLS

print(f"\n  Dados: {len(y)} amostras")
print(f"  Stat: {len(STAT_COLS)}f, CAAFE: {len(CAAFE_COLS)}f, ADV L2: {len(ADV_L2_COLS)}f")
print(f"  Classes: {dict(y.value_counts().sort_index())}")

CLASS_NAMES = {0: "MCAR", 1: "MAR", 2: "MNAR"}


# ==============================================================================
# MODELOS
# ==============================================================================
def get_modelos(n_samples: int, optuna_params: dict | None = None) -> dict:
    if n_samples < 100:
        modelos = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            ),
            "LogisticRegression": Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=0.5, random_state=42))]
            ),
            "SVM_RBF": Pipeline(
                [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1, random_state=42, probability=True))]
            ),
            "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=3))]),
            "MLP": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)),
                ]
            ),
            "NaiveBayes": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        }
    else:
        modelos = {
            "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
            "LogisticRegression": Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=42))]
            ),
            "SVM_RBF": Pipeline(
                [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=3, random_state=42, probability=True))]
            ),
            "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
            "MLP": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=42)),
                ]
            ),
            "NaiveBayes": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        }

    # STEP 02: XGBoost e CatBoost
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }

    cat_params = {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "auto_class_weights": "Balanced",
    }

    # Substituir por params otimizados se disponíveis
    if optuna_params:
        if "XGBoost" in optuna_params:
            xgb_params.update(optuna_params["XGBoost"])
        if "CatBoost" in optuna_params:
            cat_params.update(optuna_params["CatBoost"])

    modelos["XGBoost"] = XGBClassifier(**xgb_params)
    modelos["CatBoost"] = CatBoostClassifier(**cat_params)

    return modelos


# ==============================================================================
# OPTUNA — Otimização de hiperparâmetros por nível
# ==============================================================================
def _optuna_xgb_objective(trial, X_train, y_train, groups_train):
    """Objective function para XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    clf = XGBClassifier(**params, random_state=42, eval_metric="mlogloss", verbosity=0)

    from sklearn.model_selection import cross_val_score

    if groups_train is not None and len(np.unique(groups_train)) >= 5:
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
        scores = cross_val_score(clf, X_train, y_train, cv=cv, groups=groups_train, scoring="accuracy")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")

    return scores.mean()


def _optuna_cat_objective(trial, X_train, y_train, groups_train):
    """Objective function para CatBoost."""
    params = {
        "iterations": trial.suggest_int("iterations", 50, 500),
        "depth": trial.suggest_int("depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
    }

    clf = CatBoostClassifier(**params, random_seed=42, verbose=0, auto_class_weights="Balanced")

    from sklearn.model_selection import cross_val_score

    if groups_train is not None and len(np.unique(groups_train)) >= 5:
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
        scores = cross_val_score(clf, X_train, y_train, cv=cv, groups=groups_train, scoring="accuracy")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")

    return scores.mean()


def run_optuna_optimization(X_train, y_train, groups_train, level_name, n_trials=100):
    """Otimiza XGBoost e CatBoost para um nível da hierarquia."""
    results = {}

    for clf_name, objective_fn in [
        ("XGBoost", _optuna_xgb_objective),
        ("CatBoost", _optuna_cat_objective),
    ]:
        print(f"    Otimizando {clf_name} para {level_name} ({n_trials} trials)...")
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{clf_name}_{level_name}",
        )
        study.optimize(
            lambda trial, _fn=objective_fn: _fn(trial, X_train, y_train, groups_train),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        results[clf_name] = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "history": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials],
        }
        print(f"      Melhor accuracy CV: {study.best_value:.4f}")
        print(f"      Params: {study.best_params}")

    return results


def apply_balancing(X_in, y_in, method="smote"):
    """Aplica balanceamento de classes.

    method: "smote", "smote_enn", "borderline", "smote_tomek", "none"
    """
    if method == "none":
        return X_in, y_in

    try:
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    except ImportError:
        return X_in, y_in

    vc = y_in.value_counts() if hasattr(y_in, "value_counts") else pd.Series(y_in).value_counts()
    min_count = vc.min()
    if min_count < 2:
        return X_in, y_in

    k = min(3, min_count - 1)

    if method == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=k)
    elif method == "smote_enn":
        sampler = SMOTEENN(
            smote=SMOTE(random_state=42, k_neighbors=k),
            random_state=42,
        )
    elif method == "borderline":
        if min_count >= 5:
            sampler = BorderlineSMOTE(random_state=42, k_neighbors=k)
        else:
            sampler = SMOTE(random_state=42, k_neighbors=k)
    elif method == "smote_tomek":
        sampler = SMOTETomek(
            smote=SMOTE(random_state=42, k_neighbors=k),
            random_state=42,
        )
    else:
        return X_in, y_in

    return sampler.fit_resample(X_in, y_in)


def _fit_with_weights(model, X, y, weights=None):
    """Treina modelo com sample_weight se suportado."""
    if weights is None:
        model.fit(X, y)
        return

    if hasattr(model, "steps"):
        # Pipeline: passar sample_weight para o step "clf"
        step_name = model.steps[-1][0]  # "clf" nos nossos pipelines
        try:
            model.fit(X, y, **{f"{step_name}__sample_weight": weights})
        except (TypeError, ValueError):
            # Classificador não suporta sample_weight (e.g., KNN, GaussianNB)
            model.fit(X, y)
    else:
        # Modelo direto (RF, GBT)
        try:
            model.fit(X, y, sample_weight=weights)
        except TypeError:
            model.fit(X, y)


def _calibrate_model(model, X_train, y_train, method="sigmoid"):
    """Aplica calibração Platt (sigmoid) ao modelo."""
    n_classes = len(np.unique(y_train))
    n_samples = len(y_train)
    cv_folds = min(5, min(n_samples // max(n_classes, 1), 10))
    if cv_folds < 2:
        return model
    cal = CalibratedClassifierCV(model, method=method, cv=cv_folds)
    cal.fit(X_train, y_train)
    return cal


# ==============================================================================
# SPLIT
# ==============================================================================
if groups is not None and groups.nunique() > 1:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X_baseline, y, groups))
else:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X_baseline))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, stratify=y, random_state=42)

y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_weights = sample_weights[train_idx] if sample_weights is not None else None

print(f"\n  Split: train={len(y_train)}, test={len(y_test)}")
print(f"  Test dist: {dict(y_test.value_counts().sort_index())}")


# ==============================================================================
# OPTUNA — Otimização de hiperparâmetros (STEP 02)
# ==============================================================================
OPTUNA_PARAMS = None

if OPTIMIZE:
    print(f"\n{'='*70}")
    print(f"OPTUNA: Otimizando XGBoost + CatBoost ({N_TRIALS} trials/nivel)")
    print(f"{'='*70}")

    # Preparar dados para L1 (MCAR vs nao-MCAR)
    y_train_l1_opt = (y_train != 0).astype(int)
    X_train_l1_opt = X_llm_full[FEAT_STAT].iloc[train_idx]
    groups_train_opt = groups.iloc[train_idx] if groups is not None else None

    print("\n  Nivel 1 (MCAR vs nao-MCAR):")
    optuna_l1 = run_optuna_optimization(X_train_l1_opt, y_train_l1_opt, groups_train_opt, "L1", N_TRIALS)

    # Preparar dados para L2 (MAR vs MNAR)
    mask_nm_train = y_train != 0
    X_train_l2_opt = X_llm_full[FEAT_STAT_CAAFE].iloc[train_idx][mask_nm_train.values]
    y_train_l2_opt = (y_train[mask_nm_train] == 2).astype(int)
    groups_train_l2_opt = groups.iloc[train_idx][mask_nm_train.values] if groups is not None else None

    print("\n  Nivel 2 (MAR vs MNAR):")
    optuna_l2 = run_optuna_optimization(X_train_l2_opt, y_train_l2_opt, groups_train_l2_opt, "L2", N_TRIALS)

    # Salvar resultados Optuna
    OPTUNA_PARAMS = {
        "L1": {k: v["best_params"] for k, v in optuna_l1.items()},
        "L2": {k: v["best_params"] for k, v in optuna_l2.items()},
    }

    optuna_output = {
        "L1": {k: {"best_params": v["best_params"], "best_cv_accuracy": v["best_value"]} for k, v in optuna_l1.items()},
        "L2": {k: {"best_params": v["best_params"], "best_cv_accuracy": v["best_value"]} for k, v in optuna_l2.items()},
    }
    with open(os.path.join(HIER_DIR, "optuna_params.json"), "w") as f:
        json.dump(optuna_output, f, indent=2)

    # Salvar historico de trials
    for level_name, level_results in [("l1", optuna_l1), ("l2", optuna_l2)]:
        for clf_name, clf_data in level_results.items():
            hist_rows = []
            for trial in clf_data["history"]:
                row = {"trial": trial["number"], "accuracy_cv": trial["value"]}
                row.update(trial["params"])
                hist_rows.append(row)
            pd.DataFrame(hist_rows).to_csv(
                os.path.join(HIER_DIR, f"optuna_history_{level_name}_{clf_name.lower()}.csv"), index=False
            )

    print(f"\n  Params salvos em: {HIER_DIR}/optuna_params.json")


# ==============================================================================
# FUNCOES DE CLASSIFICACAO — V3+ com roteamento probabilístico
# ==============================================================================
def find_optimal_threshold(m_l1, m_l2, X_val_l1, X_val_l2, y_val, metric="accuracy"):
    """Encontra threshold ótimo no L1 avaliando o pipeline completo."""
    probs_l1 = m_l1.predict_proba(X_val_l1)[:, 1]
    best_thr, best_score = 0.5, 0.0

    for thr in np.arange(0.30, 0.71, 0.02):
        pred_l1 = (probs_l1 >= thr).astype(int)
        y_pred = np.zeros(len(y_val), dtype=int)
        mask_nm = pred_l1 == 1
        y_pred[~mask_nm] = 0
        if mask_nm.any():
            pred_l2 = m_l2.predict(X_val_l2[mask_nm])
            y_pred[mask_nm] = np.where(pred_l2 == 0, 1, 2)

        score = accuracy_score(y_val, y_pred)
        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr


def _get_optuna_params_for_level(optuna_params, level, modelo_nome):
    """Extrai params Optuna para um modelo/nível específico."""
    if optuna_params is None:
        return None
    level_params = optuna_params.get(level, {})
    if modelo_nome in level_params:
        return {modelo_nome: level_params[modelo_nome]}
    return None


def run_hierarchical_v3plus(
    X_full,
    y,
    train_idx,
    test_idx,
    feat_l1,
    feat_l2,
    modelo_nome,
    routing="hard",
    calibrate=True,
    sample_weights_train=None,
    optuna_params=None,
):
    """Classificação hierárquica com roteamento probabilístico."""
    X_l1 = X_full[feat_l1]
    X_l2 = X_full[feat_l2]

    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # --- Nível 1: MCAR vs NAO-MCAR ---
    y_tr_l1 = (y_tr != 0).astype(int)
    X_tr_l1 = X_l1.iloc[train_idx]
    X_l1.iloc[test_idx]

    if sample_weights_train is not None:
        X_tr_l1_fit, y_tr_l1_fit = X_tr_l1, y_tr_l1
        w_l1 = sample_weights_train
    else:
        X_tr_l1_fit, y_tr_l1_fit = apply_balancing(X_tr_l1, y_tr_l1, method=BALANCING)
        w_l1 = None

    params_l1 = _get_optuna_params_for_level(optuna_params, "L1", modelo_nome)
    m_l1 = get_modelos(len(X_tr_l1_fit), params_l1)[modelo_nome]
    _fit_with_weights(m_l1, X_tr_l1_fit, y_tr_l1_fit, w_l1)

    if calibrate and routing != "hard":
        m_l1 = _calibrate_model(m_l1, X_tr_l1_fit, y_tr_l1_fit)

    # --- Nível 2: MAR vs MNAR ---
    mask_tr_nm = y_tr != 0
    X_tr_l2 = X_l2.iloc[train_idx][mask_tr_nm.values]
    y_tr_l2 = (y_tr[mask_tr_nm] == 2).astype(int)

    if sample_weights_train is not None:
        # Filtrar pesos para amostras não-MCAR
        w_l2 = sample_weights_train[mask_tr_nm.values]
        X_tr_l2_fit, y_tr_l2_fit = X_tr_l2, y_tr_l2
    else:
        X_tr_l2_fit, y_tr_l2_fit = apply_balancing(X_tr_l2, y_tr_l2, method=BALANCING)
        w_l2 = None

    params_l2 = _get_optuna_params_for_level(optuna_params, "L2", modelo_nome)
    m_l2 = get_modelos(len(X_tr_l2_fit), params_l2)[modelo_nome]
    _fit_with_weights(m_l2, X_tr_l2_fit, y_tr_l2_fit, w_l2)

    if calibrate and routing in ("soft3zone", "fullprob"):
        m_l2 = _calibrate_model(m_l2, X_tr_l2_fit, y_tr_l2_fit)

    # --- Predição com roteamento ---
    X_te_l1_df = X_l1.iloc[test_idx]
    X_te_l2_df = X_l2.iloc[test_idx]
    y_pred = np.zeros(len(y_te), dtype=int)

    if routing == "hard":
        # Original: predição hard
        pred_l1 = m_l1.predict(X_te_l1_df)
        mask_notmcar = pred_l1 == 1
        y_pred[~mask_notmcar] = 0
        if mask_notmcar.any():
            pred_l2 = m_l2.predict(X_te_l2_df[mask_notmcar])
            y_pred[mask_notmcar] = np.where(pred_l2 == 0, 1, 2)

    elif routing == "threshold":
        # Threshold otimizado via validação interna
        probs_l1 = m_l1.predict_proba(X_te_l1_df)[:, 1]

        # Usar treino com CV interna para encontrar threshold ótimo
        # Simplificação: usar threshold fixo otimizado a priori
        # Para otimização real, precisaríamos de val set separado
        optimal_thr = find_optimal_threshold(
            m_l1,
            m_l2,
            X_tr_l1.values if hasattr(X_tr_l1, "values") else X_tr_l1,
            X_tr_l2.reindex(X_l2.iloc[train_idx].index).fillna(0) if len(X_tr_l2) > 0 else X_l2.iloc[train_idx],
            y_tr,
        )
        # Aplicar no teste
        pred_l1 = (probs_l1 >= optimal_thr).astype(int)
        mask_notmcar = pred_l1 == 1
        y_pred[~mask_notmcar] = 0
        if mask_notmcar.any():
            pred_l2 = m_l2.predict(X_te_l2_df[mask_notmcar])
            y_pred[mask_notmcar] = np.where(pred_l2 == 0, 1, 2)

    elif routing == "soft3zone":
        # 3 zonas: MCAR confiante, incerto, não-MCAR confiante
        probs_l1 = m_l1.predict_proba(X_te_l1_df)[:, 1]  # P(não-MCAR)
        thr_low, thr_high = 0.35, 0.65

        zone_mcar = probs_l1 < thr_low
        zone_notmcar = probs_l1 > thr_high
        zone_uncertain = ~zone_mcar & ~zone_notmcar

        y_pred[zone_mcar] = 0

        if zone_notmcar.any():
            pred_l2 = m_l2.predict(X_te_l2_df[zone_notmcar])
            y_pred[zone_notmcar] = np.where(pred_l2 == 0, 1, 2)

        if zone_uncertain.any():
            # Combinar probabilidades de ambos os níveis
            prob_mcar = 1 - probs_l1[zone_uncertain]
            prob_l2 = m_l2.predict_proba(X_te_l2_df[zone_uncertain])
            prob_mar = probs_l1[zone_uncertain] * prob_l2[:, 0]
            prob_mnar = probs_l1[zone_uncertain] * prob_l2[:, 1]
            all_probs = np.column_stack([prob_mcar, prob_mar, prob_mnar])
            y_pred[zone_uncertain] = all_probs.argmax(axis=1)

    elif routing == "fullprob":
        # Sempre combinar probabilidades de ambos os níveis
        probs_l1 = m_l1.predict_proba(X_te_l1_df)[:, 1]  # P(não-MCAR)
        prob_l2 = m_l2.predict_proba(X_te_l2_df)  # [P(MAR|nM), P(MNAR|nM)]

        prob_mcar = 1 - probs_l1
        prob_mar = probs_l1 * prob_l2[:, 0]
        prob_mnar = probs_l1 * prob_l2[:, 1]

        all_probs = np.column_stack([prob_mcar, prob_mar, prob_mnar])
        y_pred = all_probs.argmax(axis=1)

    # --- Métricas por nível ---
    if routing == "hard":
        pred_l1_binary = pred_l1
    else:
        pred_l1_binary = (m_l1.predict_proba(X_te_l1_df)[:, 1] >= 0.5).astype(int)

    acc_l1 = accuracy_score((y_te != 0).astype(int), pred_l1_binary)

    mask_true_nm = y_te != 0
    mask_pred_nm = y_pred != 0
    mask_both = mask_true_nm.values & mask_pred_nm
    if mask_both.any():
        acc_l2 = accuracy_score(y_te.values[mask_both], y_pred[mask_both])
    else:
        acc_l2 = float("nan")

    return y_pred, acc_l1, acc_l2


def run_direct(X_full, y, train_idx, test_idx, feat_cols, modelo_nome):
    """Classificação direta 3-way (baseline)."""
    X_sel = X_full[feat_cols]
    X_tr, X_te = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
    y_tr, _y_te = y.iloc[train_idx], y.iloc[test_idx]
    X_tr_sm, y_tr_sm = apply_balancing(X_tr, y_tr, method=BALANCING)
    modelo = get_modelos(len(X_tr_sm))[modelo_nome]
    modelo.fit(X_tr_sm, y_tr_sm)
    return modelo.predict(X_te)


# ==============================================================================
# DEFINIÇÃO DAS VARIANTES
# ==============================================================================
# V1: baseline direto, V3: hierárquico hard (reproduz original)
# V3+: hierárquico com routing probabilístico (novo)
VARIANTES = {}

# V1 baseline direto
VARIANTES["V1_direto_stat"] = {
    "tipo": "direto",
    "features": FEAT_STAT,
    "desc": "Direto 3-way (baseline)",
}

# V3 hard (reproduz original)
VARIANTES["V3_hier_hard"] = {
    "tipo": "hierarquico",
    "features_l1": FEAT_STAT,
    "features_l2": FEAT_STAT_CAAFE,
    "routing": "hard",
    "desc": "V3 Hier CAAFE N2 (hard, original)",
}

# V3+ para cada estratégia de routing
for routing in ROUTING_STRATEGIES:
    if routing == "hard":
        continue  # Já coberto por V3_hier_hard
    VARIANTES[f"V3plus_{routing}"] = {
        "tipo": "hierarquico",
        "features_l1": FEAT_STAT,
        "features_l2": FEAT_STAT_CAAFE,
        "routing": routing,
        "desc": f"V3+ Hier CAAFE N2 ({routing})",
    }

# V3+adv: CAAFE + Advanced L2 features no Nível 2 (STEP 03)
if ADV_L2_COLS:
    VARIANTES["V3adv_hier_hard"] = {
        "tipo": "hierarquico",
        "features_l1": FEAT_STAT,
        "features_l2": FEAT_STAT_CAAFE_ADV,
        "routing": "hard",
        "desc": f"V3+adv Hier CAAFE+ADV N2 (hard, {len(ADV_L2_COLS)}f extra)",
    }
    for routing in ROUTING_STRATEGIES:
        if routing == "hard":
            continue
        VARIANTES[f"V3adv_{routing}"] = {
            "tipo": "hierarquico",
            "features_l1": FEAT_STAT,
            "features_l2": FEAT_STAT_CAAFE_ADV,
            "routing": routing,
            "desc": f"V3+adv CAAFE+ADV N2 ({routing}, {len(ADV_L2_COLS)}f extra)",
        }


# ==============================================================================
# RODAR TODAS AS VARIANTES x TODOS OS MODELOS
# ==============================================================================
model_names = list(get_modelos(1000).keys())
all_results = {}

n_variants = len(VARIANTES)
n_models = len(model_names)
print(f"\nRodando {n_variants} variantes x {n_models} modelos...")

for var_name, var_cfg in VARIANTES.items():
    print(f"\n  {var_name}: {var_cfg['desc']}")
    all_results[var_name] = {}

    for modelo_nome in tqdm(model_names, desc=f"  {var_name}", leave=False):
        if var_cfg["tipo"] == "direto":
            y_pred = run_direct(X_llm_full, y, train_idx, test_idx, var_cfg["features"], modelo_nome)
            acc_l1, acc_l2 = float("nan"), float("nan")
        else:
            routing = var_cfg.get("routing", "hard")
            y_pred, acc_l1, acc_l2 = run_hierarchical_v3plus(
                X_llm_full,
                y,
                train_idx,
                test_idx,
                var_cfg["features_l1"],
                var_cfg["features_l2"],
                modelo_nome,
                routing=routing,
                calibrate=CALIBRATE,
                sample_weights_train=train_weights,
                optuna_params=OPTUNA_PARAMS,
            )

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        all_results[var_name][modelo_nome] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "report": report,
            "confusion": cm,
            "y_pred": y_pred.copy(),
            "acc_level1": acc_l1,
            "acc_level2": acc_l2,
            "recall_MCAR": report.get("0", {}).get("recall", 0),
            "recall_MAR": report.get("1", {}).get("recall", 0),
            "recall_MNAR": report.get("2", {}).get("recall", 0),
            "f1_MCAR": report.get("0", {}).get("f1-score", 0),
            "f1_MAR": report.get("1", {}).get("f1-score", 0),
            "f1_MNAR": report.get("2", {}).get("f1-score", 0),
        }


# ==============================================================================
# TABELA COMPARATIVA
# ==============================================================================
print(f"\n{'='*70}")
print("TABELA COMPARATIVA V3+")
print(f"{'='*70}")

rows = []
for var_name in VARIANTES:
    for modelo_nome in model_names:
        r = all_results[var_name][modelo_nome]
        rows.append(
            {
                "variante": var_name,
                "modelo": modelo_nome,
                "accuracy": r["accuracy"],
                "f1_macro": r["f1_macro"],
                "recall_MCAR": r["recall_MCAR"],
                "recall_MAR": r["recall_MAR"],
                "recall_MNAR": r["recall_MNAR"],
                "f1_MCAR": r["f1_MCAR"],
                "f1_MAR": r["f1_MAR"],
                "f1_MNAR": r["f1_MNAR"],
                "acc_level1": r["acc_level1"],
                "acc_level2": r["acc_level2"],
            }
        )

df_all = pd.DataFrame(rows)
df_all.to_csv(os.path.join(HIER_DIR, "todas_variantes_v3plus.csv"), index=False)

# Resumo
df_summary = (
    df_all.groupby("variante")
    .agg(
        {
            "accuracy": ["mean", "std", "max"],
            "f1_macro": ["mean", "std", "max"],
            "recall_MNAR": ["mean", "std", "max"],
            "acc_level1": "mean",
            "acc_level2": "mean",
        }
    )
    .round(4)
)
df_summary.to_csv(os.path.join(HIER_DIR, "resumo_v3plus.csv"))

# Print ranking
print("\nRanking por accuracy maxima:")
for var_name in VARIANTES:
    subset = df_all[df_all["variante"] == var_name]
    acc_max = subset["accuracy"].max()
    best_row = subset.loc[subset["accuracy"].idxmax()]
    best_model = best_row["modelo"]
    mnar_recall = best_row["recall_MNAR"]
    f1 = best_row["f1_macro"]
    print(f"  {var_name:25s}: acc={acc_max:.3f} f1={f1:.3f} " f"MNAR_recall={mnar_recall:.3f} ({best_model})")


# ==============================================================================
# COMPARACAO V3 HARD vs V3+ POR MODELO
# ==============================================================================
if "V3_hier_hard" in all_results:
    print(f"\n{'='*70}")
    print("COMPARACAO: V3 (hard) vs V3+ por modelo")
    print(f"{'='*70}")

    for routing in ROUTING_STRATEGIES:
        if routing == "hard":
            continue
        v3plus_name = f"V3plus_{routing}"
        if v3plus_name not in all_results:
            continue

        print(f"\n  --- V3 hard vs V3+ {routing} ---")
        for modelo_nome in model_names:
            r_hard = all_results["V3_hier_hard"][modelo_nome]
            r_plus = all_results[v3plus_name][modelo_nome]
            delta_acc = r_plus["accuracy"] - r_hard["accuracy"]
            delta_mnar = r_plus["recall_MNAR"] - r_hard["recall_MNAR"]
            sym = "+" if delta_acc > 0 else "-" if delta_acc < 0 else "="
            print(
                f"    {sym} {modelo_nome:20s}: hard={r_hard['accuracy']:.3f} "
                f"{routing}={r_plus['accuracy']:.3f} "
                f"(acc {delta_acc:+.3f}, MNAR {delta_mnar:+.3f})"
            )


# ==============================================================================
# TESTES DE SIGNIFICANCIA — McNemar V3 hard vs V3+ melhor
# ==============================================================================
print(f"\n{'='*70}")
print("TESTES DE SIGNIFICANCIA (McNemar)")
print(f"{'='*70}")

from scipy.stats import chi2

sig_rows = []
for routing in ROUTING_STRATEGIES:
    if routing == "hard":
        continue
    v3plus_name = f"V3plus_{routing}"
    if v3plus_name not in all_results:
        continue

    print(f"\n  V3_hard vs V3+_{routing}:")
    for modelo_nome in model_names:
        y_pred_hard = all_results["V3_hier_hard"][modelo_nome]["y_pred"]
        y_pred_plus = all_results[v3plus_name][modelo_nome]["y_pred"]

        correct_hard = y_pred_hard == y_test.values
        correct_plus = y_pred_plus == y_test.values
        b = np.sum(correct_hard & ~correct_plus)
        c = np.sum(~correct_hard & correct_plus)

        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
            mcnemar_p = 1 - chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat, mcnemar_p = 0.0, 1.0

        sig_star = "***" if mcnemar_p < 0.001 else "**" if mcnemar_p < 0.01 else "*" if mcnemar_p < 0.05 else ""
        print(f"    {modelo_nome:20s}: b={b:3d} c={c:3d} chi2={mcnemar_stat:.2f} p={mcnemar_p:.4f} {sig_star}")

        sig_rows.append(
            {
                "comparison": f"V3hard_vs_V3plus_{routing}",
                "modelo": modelo_nome,
                "hard_correct_plus_wrong": int(b),
                "hard_wrong_plus_correct": int(c),
                "mcnemar_chi2": mcnemar_stat,
                "mcnemar_p": mcnemar_p,
                "significant_005": mcnemar_p < 0.05,
            }
        )

if sig_rows:
    pd.DataFrame(sig_rows).to_csv(os.path.join(HIER_DIR, "significancia_v3plus.csv"), index=False)


# ==============================================================================
# LOGO CROSS-VALIDATION
# ==============================================================================
if groups is not None and groups.nunique() > 2:
    print(f"\n{'='*70}")
    print("LEAVE-ONE-GROUP-OUT CV")
    print(f"{'='*70}")

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(X_baseline, y, groups)
    print(f"  {n_folds} folds")

    cv_variantes = ["V3_hier_hard"]
    for routing in ROUTING_STRATEGIES:
        if routing != "hard":
            vname = f"V3plus_{routing}"
            if vname in VARIANTES:
                cv_variantes.append(vname)

    cv_results = {v: {} for v in cv_variantes}

    for modelo_nome in tqdm(model_names, desc="LOGO CV"):
        for var_name in cv_variantes:
            var_cfg = VARIANTES[var_name]
            fold_accs = []

            for tr_i, te_i in logo.split(X_baseline, y, groups):
                routing = var_cfg.get("routing", "hard")

                y_tr_fold, y_te_fold = y.iloc[tr_i], y.iloc[te_i]
                feat_l1 = var_cfg["features_l1"]
                feat_l2 = var_cfg["features_l2"]
                X_l1_f = X_llm_full[feat_l1]
                X_l2_f = X_llm_full[feat_l2]

                # L1
                y_tr_l1 = (y_tr_fold != 0).astype(int)
                X_tr_l1_sm, y_tr_l1_sm = apply_balancing(X_l1_f.iloc[tr_i], y_tr_l1, method=BALANCING)
                m_l1 = get_modelos(len(X_tr_l1_sm))[modelo_nome]
                m_l1.fit(X_tr_l1_sm, y_tr_l1_sm)
                if CALIBRATE and routing != "hard":
                    m_l1 = _calibrate_model(m_l1, X_tr_l1_sm, y_tr_l1_sm)

                # L2
                mask_nm = y_tr_fold != 0
                X_tr_l2 = X_l2_f.iloc[tr_i][mask_nm.values]
                y_tr_l2 = (y_tr_fold[mask_nm] == 2).astype(int)

                if len(y_tr_l2) >= 2 and y_tr_l2.nunique() >= 2:
                    X_tr_l2_sm, y_tr_l2_sm = apply_balancing(X_tr_l2, y_tr_l2, method=BALANCING)
                    m_l2 = get_modelos(len(X_tr_l2_sm))[modelo_nome]
                    m_l2.fit(X_tr_l2_sm, y_tr_l2_sm)
                    if CALIBRATE and routing in ("soft3zone", "fullprob"):
                        m_l2 = _calibrate_model(m_l2, X_tr_l2_sm, y_tr_l2_sm)

                    X_te_l1_f = X_l1_f.iloc[te_i]
                    X_te_l2_f = X_l2_f.iloc[te_i]

                    if routing == "hard":
                        pred_l1 = m_l1.predict(X_te_l1_f)
                        y_pred_fold = np.zeros(len(y_te_fold), dtype=int)
                        mask_nm_pred = pred_l1 == 1
                        if mask_nm_pred.any():
                            p_l2 = m_l2.predict(X_te_l2_f[mask_nm_pred])
                            y_pred_fold[mask_nm_pred] = np.where(p_l2 == 0, 1, 2)

                    elif routing == "fullprob":
                        probs_l1 = m_l1.predict_proba(X_te_l1_f)[:, 1]
                        prob_l2 = m_l2.predict_proba(X_te_l2_f)
                        prob_mcar = 1 - probs_l1
                        prob_mar = probs_l1 * prob_l2[:, 0]
                        prob_mnar = probs_l1 * prob_l2[:, 1]
                        all_probs = np.column_stack([prob_mcar, prob_mar, prob_mnar])
                        y_pred_fold = all_probs.argmax(axis=1)

                    elif routing == "soft3zone":
                        probs_l1 = m_l1.predict_proba(X_te_l1_f)[:, 1]
                        y_pred_fold = np.zeros(len(y_te_fold), dtype=int)
                        zone_mcar = probs_l1 < 0.35
                        zone_notmcar = probs_l1 > 0.65
                        zone_unc = ~zone_mcar & ~zone_notmcar
                        y_pred_fold[zone_mcar] = 0
                        if zone_notmcar.any():
                            p_l2 = m_l2.predict(X_te_l2_f[zone_notmcar])
                            y_pred_fold[zone_notmcar] = np.where(p_l2 == 0, 1, 2)
                        if zone_unc.any():
                            pc = 1 - probs_l1[zone_unc]
                            pl2 = m_l2.predict_proba(X_te_l2_f[zone_unc])
                            pm = probs_l1[zone_unc] * pl2[:, 0]
                            pn = probs_l1[zone_unc] * pl2[:, 1]
                            ap = np.column_stack([pc, pm, pn])
                            y_pred_fold[zone_unc] = ap.argmax(axis=1)

                    elif routing == "threshold":
                        probs_l1 = m_l1.predict_proba(X_te_l1_f)[:, 1]
                        # Usar threshold fixo 0.45 (conservador) para LOGO
                        pred_l1 = (probs_l1 >= 0.45).astype(int)
                        y_pred_fold = np.zeros(len(y_te_fold), dtype=int)
                        mask_nm_pred = pred_l1 == 1
                        if mask_nm_pred.any():
                            p_l2 = m_l2.predict(X_te_l2_f[mask_nm_pred])
                            y_pred_fold[mask_nm_pred] = np.where(p_l2 == 0, 1, 2)
                else:
                    y_pred_fold = np.where(m_l1.predict(X_l1_f.iloc[te_i]) == 0, 0, 1)

                fold_accs.append(accuracy_score(y_te_fold, y_pred_fold))

            cv_results[var_name][modelo_nome] = {
                "mean": np.mean(fold_accs),
                "std": np.std(fold_accs),
            }

    # Print
    cv_rows = []
    for modelo_nome in model_names:
        row = {"modelo": modelo_nome}
        for var_name in cv_variantes:
            r = cv_results[var_name][modelo_nome]
            row[f"{var_name}_cv_mean"] = r["mean"]
            row[f"{var_name}_cv_std"] = r["std"]
        cv_rows.append(row)

        hard_cv = cv_results["V3_hier_hard"][modelo_nome]["mean"]
        best_plus_name = None
        best_plus_cv = hard_cv
        for vn in cv_variantes:
            if vn == "V3_hier_hard":
                continue
            cv_val = cv_results[vn][modelo_nome]["mean"]
            if cv_val > best_plus_cv:
                best_plus_cv = cv_val
                best_plus_name = vn

        delta = best_plus_cv - hard_cv
        sym = "+" if delta > 0 else "-" if delta < 0 else "="
        plus_str = f"{best_plus_name}={best_plus_cv:.3f}" if best_plus_name else "nenhum melhor"
        print(f"  {sym} {modelo_nome:20s}: hard={hard_cv:.3f} best_plus={plus_str} (delta={delta:+.3f})")

    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(os.path.join(HIER_DIR, "cv_logo_v3plus.csv"), index=False)


# ==============================================================================
# GRAFICOS
# ==============================================================================

# Barras comparativas: V3 hard vs V3+ routing strategies
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
var_names_plot = list(VARIANTES)
n_vars = len(var_names_plot)
width = 0.8 / n_vars

for ax_idx, (metric, title) in enumerate([("accuracy", "Accuracy"), ("recall_MNAR", "MNAR Recall")]):
    ax = axes[ax_idx]
    x = np.arange(len(model_names))

    for i, var_name in enumerate(var_names_plot):
        vals = [all_results[var_name][m][metric] for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=var_name, alpha=0.8)

    ax.set_ylabel(title)
    ax.set_title(f"{title} por Modelo e Variante")
    ax.set_xticks(x + width * (n_vars - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.3)

plt.suptitle(f"V3+ Routing Comparison — {DATA_TYPE.upper()}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(HIER_DIR, "v3plus_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# Heatmap
var_names_list = list(VARIANTES.keys())
fig, ax = plt.subplots(figsize=(12, max(4, len(var_names_list) * 0.8)))
heat_data = np.zeros((len(var_names_list), len(model_names)))
for i, v in enumerate(var_names_list):
    for j, m in enumerate(model_names):
        heat_data[i, j] = all_results[v][m]["accuracy"]

im = ax.imshow(heat_data, cmap="RdYlGn", aspect="auto", vmin=0.2, vmax=0.7)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(var_names_list)))
ax.set_yticklabels(var_names_list, fontsize=8)
ax.set_title(f"Accuracy Heatmap V3+ — {DATA_TYPE.upper()}")
plt.colorbar(im, ax=ax, label="Accuracy")
for i in range(len(var_names_list)):
    for j in range(len(model_names)):
        ax.text(
            j,
            i,
            f"{heat_data[i, j]:.1%}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if heat_data[i, j] < 0.45 else "black",
        )
plt.tight_layout()
plt.savefig(os.path.join(HIER_DIR, "heatmap_v3plus.png"), dpi=300, bbox_inches="tight")
plt.close()


# ==============================================================================
# SUMMARY
# ==============================================================================
summary = {
    "timestamp": datetime.now().isoformat(),
    "data_type": DATA_TYPE,
    "experiment": EXPERIMENT,
    "routing_strategies": ROUTING_STRATEGIES,
    "clean_labels": CLEAN_MODE,
    "calibrate": CALIBRATE,
    "balancing": BALANCING,
    "n_samples": int(len(y)),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "variantes": {},
}

for var_name in VARIANTES:
    best_m = max(all_results[var_name].items(), key=lambda x: x[1]["accuracy"])
    summary["variantes"][var_name] = {
        "desc": VARIANTES[var_name]["desc"],
        "best_model": best_m[0],
        "best_accuracy": float(best_m[1]["accuracy"]),
        "best_f1_macro": float(best_m[1]["f1_macro"]),
        "best_recall_MNAR": float(best_m[1]["recall_MNAR"]),
    }

with open(os.path.join(HIER_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


# ==============================================================================
# RESUMO FINAL
# ==============================================================================
print(f"\n{'='*70}")
print("CONCLUIDO!")
print(f"{'='*70}")

print("\nRanking final (accuracy maxima):")
ranking = sorted(
    [(v, max(all_results[v].items(), key=lambda x: x[1]["accuracy"])) for v in VARIANTES],
    key=lambda x: x[1][1]["accuracy"],
    reverse=True,
)

for i, (var, (modelo, metrics)) in enumerate(ranking, 1):
    print(
        f"  {i}. {var:25s} {metrics['accuracy']:.3f} ({modelo}) "
        f"MNAR={metrics['recall_MNAR']:.3f} F1={metrics['f1_macro']:.3f}"
    )

print(f"\nSalvos em: {HIER_DIR}")
