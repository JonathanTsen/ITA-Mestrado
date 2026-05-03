"""
MechDetect Original — Reimplementação fiel de Jung et al. (2024).

Diferenças vs nossa implementação em features/mechdetect.py:
  - Usa HistGradientBoosting (como no paper) em vez de LogisticRegression
  - 10-fold CV para AUC (como no paper) em vez de 3-fold
  - Regra de decisão por thresholds (como no paper) em vez de features para ML

Uso:
    python -m baselines.mechdetect_original --data sintetico --experiment step05_pro
    python -m baselines.mechdetect_original --data real --experiment step05_pro
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
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contextlib

from utils.paths import DATASET_PATHS, OUTPUT_BASE

warnings.filterwarnings("ignore")

# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(description="MechDetect Original Baseline")
parser.add_argument("--data", choices=["sintetico", "real"], required=True)
parser.add_argument("--experiment", required=True)
args = parser.parse_args()

DATA_TYPE = args.data
EXPERIMENT = args.experiment
OUT_DIR = os.path.join(OUTPUT_BASE, EXPERIMENT, DATA_TYPE, "baselines", "mechdetect_original")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("🔬 MECHDETECT ORIGINAL — Jung et al. (2024)")
print("=" * 70)
print(f"📊 Dados: {DATA_TYPE}")
print(f"📂 Output: {OUT_DIR}")
print("=" * 70)


# ==============================================================================
# FUNÇÕES MECHDETECT
# ==============================================================================
def compute_auc_task(X, y, n_folds=5):
    """Computa AUC-ROC com HistGradientBoosting e k-fold CV."""
    min_class = min(np.sum(y == 0), np.sum(y == 1))
    n_folds = min(n_folds, max(2, min_class))

    if min_class < 2:
        return 0.5, []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aucs = []

    for train_idx, test_idx in skf.split(X, y):
        if len(np.unique(y[test_idx])) < 2:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=50, max_depth=3, learning_rate=0.1, min_samples_leaf=20, random_state=42
        )
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        try:
            auc = roc_auc_score(y[test_idx], proba)
            fold_aucs.append(auc)
        except ValueError:
            continue

    if fold_aucs:
        return float(np.mean(fold_aucs)), fold_aucs
    return 0.5, []


def mechdetect_classify_dataset(df):
    """Classifica um dataset como MCAR/MAR/MNAR usando regras MechDetect."""
    mask = df["X0"].isna().astype(int).values
    n_missing = mask.sum()
    n_total = len(mask)

    if n_missing < 5 or (n_total - n_missing) < 5:
        return "MCAR", {"auc_complete": 0.5, "auc_shuffled": 0.5, "auc_excluded": 0.5}

    X0_imputed = df["X0"].fillna(df["X0"].median()).values
    X_other = df[["X1", "X2", "X3", "X4"]].values
    X_complete = np.column_stack([X0_imputed, X_other])
    X_excluded = X_other

    rng = np.random.RandomState(42)
    mask_shuffled = mask.copy()
    rng.shuffle(mask_shuffled)

    auc_complete, folds_c = compute_auc_task(X_complete, mask, n_folds=10)
    auc_shuffled, folds_s = compute_auc_task(X_complete, mask_shuffled, n_folds=10)
    auc_excluded, folds_e = compute_auc_task(X_excluded, mask, n_folds=10)

    delta_cs = auc_complete - auc_shuffled
    delta_ce = auc_complete - auc_excluded

    # MWU test: complete vs shuffled
    mwu_p = 1.0
    if len(folds_c) >= 2 and len(folds_s) >= 2:
        with contextlib.suppress(Exception):
            _, mwu_p = stats.mannwhitneyu(folds_c, folds_s, alternative="greater")

    # Regra de decisão MechDetect (baseada nos thresholds do paper)
    # Threshold principal: se AUC_complete não é significativamente > AUC_shuffled → MCAR
    # Se AUC_excluded ≈ AUC_complete → MAR (X1-X4 explicam tudo)
    # Se AUC_excluded < AUC_complete → MNAR (X0 contribui)

    if delta_cs < 0.05 or mwu_p > 0.05:
        prediction = "MCAR"
    elif delta_ce < 0.03:
        prediction = "MAR"
    else:
        prediction = "MNAR"

    features = {
        "auc_complete": auc_complete,
        "auc_shuffled": auc_shuffled,
        "auc_excluded": auc_excluded,
        "delta_complete_shuffled": delta_cs,
        "delta_complete_excluded": delta_ce,
        "mwu_pvalue": mwu_p,
    }
    return prediction, features


# ==============================================================================
# CARREGAR E CLASSIFICAR TODOS OS DATASETS
# ==============================================================================
CLASS_MAP = {"MCAR": 0, "MAR": 1, "MNAR": 2}
results = []

for class_name in ["MCAR", "MAR", "MNAR"]:
    folder = DATASET_PATHS[DATA_TYPE][class_name]
    if not os.path.exists(folder):
        print(f"⚠️ Pasta não encontrada: {folder}")
        continue

    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    print(f"\n📂 {class_name}: {len(files)} arquivos em {folder}")

    for fname in tqdm(files, desc=f"  {class_name}", leave=False):
        fpath = os.path.join(folder, fname)
        try:
            df = pd.read_csv(fpath, sep="\t")
            if "X0" not in df.columns:
                df.columns = [f"X{i}" for i in range(df.shape[1])]
            prediction, features = mechdetect_classify_dataset(df)
            results.append(
                {
                    "file": fname,
                    "true_label": class_name,
                    "true_code": CLASS_MAP[class_name],
                    "pred_label": prediction,
                    "pred_code": CLASS_MAP[prediction],
                    **features,
                }
            )
        except Exception as e:
            print(f"    ⚠️ Erro em {fname}: {e}")
            results.append(
                {
                    "file": fname,
                    "true_label": class_name,
                    "true_code": CLASS_MAP[class_name],
                    "pred_label": "MCAR",
                    "pred_code": 0,
                    "auc_complete": 0.5,
                    "auc_shuffled": 0.5,
                    "auc_excluded": 0.5,
                    "delta_complete_shuffled": 0.0,
                    "delta_complete_excluded": 0.0,
                    "mwu_pvalue": 1.0,
                }
            )

df_results = pd.DataFrame(results)

# ==============================================================================
# MÉTRICAS
# ==============================================================================
y_true = df_results["true_code"].values
y_pred = df_results["pred_code"].values

acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
report = classification_report(y_true, y_pred, target_names=["MCAR", "MAR", "MNAR"], output_dict=True, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

print(f"\n{'='*70}")
print(f"📊 RESULTADOS MECHDETECT ORIGINAL — {DATA_TYPE.upper()}")
print(f"{'='*70}")
print(f"   Accuracy: {acc:.4f}")
print(f"   F1 Macro: {f1_macro:.4f}")
print("\n   Per class:")
for cls in ["MCAR", "MAR", "MNAR"]:
    r = report[cls]
    print(f"     {cls}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f} (n={int(r['support'])})")

print("\n   Confusion Matrix:")
print("          Pred MCAR  MAR  MNAR")
for i, cls in enumerate(["MCAR", "MAR", "MNAR"]):
    print(f"   {cls:5s}  {cm[i][0]:5d} {cm[i][1]:4d} {cm[i][2]:5d}")

# ==============================================================================
# SALVAR RESULTADOS
# ==============================================================================
df_results.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

# Métricas por classe
metrics_rows = []
for cls in ["MCAR", "MAR", "MNAR"]:
    r = report[cls]
    metrics_rows.append(
        {
            "modelo": "MechDetect_Original",
            "classe": cls,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1-score"],
            "support": int(r["support"]),
        }
    )
pd.DataFrame(metrics_rows).to_csv(os.path.join(OUT_DIR, "metrics_per_class.csv"), index=False)

# Distribuição das predições
print(f"\n   Distribuição predições: {dict(df_results['pred_label'].value_counts())}")
print(f"   Distribuição real:      {dict(df_results['true_label'].value_counts())}")

# AUC stats
print("\n   AUC stats (média por classe real):")
for cls in ["MCAR", "MAR", "MNAR"]:
    sub = df_results[df_results["true_label"] == cls]
    print(
        f"     {cls}: auc_c={sub['auc_complete'].mean():.3f} "
        f"auc_s={sub['auc_shuffled'].mean():.3f} "
        f"auc_e={sub['auc_excluded'].mean():.3f} "
        f"delta_cs={sub['delta_complete_shuffled'].mean():.3f} "
        f"delta_ce={sub['delta_complete_excluded'].mean():.3f}"
    )

# Gráfico: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
labels = ["MCAR", "MAR", "MNAR"]
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.set_title(f"MechDetect Original — {DATA_TYPE.upper()}\nAcc={acc:.1%}, F1_macro={f1_macro:.3f}")
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_yticks(range(3))
ax.set_yticklabels(labels)
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
plt.colorbar(im, ax=ax)
thresh = cm.max() / 2.0
for i in range(3):
    for j in range(3):
        ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# Gráfico: AUC distributions por classe
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric in zip(axes, ["auc_complete", "auc_excluded", "delta_complete_excluded"], strict=False):
    for cls, color in zip(["MCAR", "MAR", "MNAR"], ["#3498db", "#2ecc71", "#e74c3c"], strict=False):
        vals = df_results[df_results["true_label"] == cls][metric]
        ax.hist(vals, alpha=0.5, label=cls, color=color, bins=20)
    ax.set_title(metric)
    ax.legend()
    ax.set_xlabel("Value")
plt.suptitle(f"MechDetect AUC Distributions — {DATA_TYPE.upper()}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "auc_distributions.png"), dpi=300, bbox_inches="tight")
plt.close()

# Training summary
summary = {
    "timestamp": datetime.now().isoformat(),
    "method": "MechDetect_Original",
    "reference": "Jung et al. (2024)",
    "data_type": DATA_TYPE,
    "experiment": EXPERIMENT,
    "n_datasets": len(df_results),
    "classifier": "HistGradientBoostingClassifier",
    "cv_folds": 10,
    "accuracy": float(acc),
    "f1_macro": float(f1_macro),
    "per_class": {
        cls: {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }
        for cls in ["MCAR", "MAR", "MNAR"]
    },
    "confusion_matrix": cm.tolist(),
}
with open(os.path.join(OUT_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n💾 Salvos em: {OUT_DIR}")
print(f"{'='*70}")
