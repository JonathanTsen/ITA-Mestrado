"""
classificar_mnar.py — Classifica datasets MNAR como Focused vs Diffuse.

Focused MNAR: f(M | X_m) — missing depende SOMENTE do valor faltante.
Diffuse MNAR: f(M | X_m, X_o) — missing depende do valor faltante E de observados.

Método: compara AUC de predição de mask com e sem X1-X4.
  - AUC_complete (X0_imputed + X1-X4) vs AUC_excluded (apenas X0_imputed)
  - Se delta > threshold → Diffuse (X_o contribui significativamente)
  - Se delta <= threshold → Focused (apenas X_m importa)

Uso:
    cd "IC - ITA 2/Scripts/v2_improved"
    uv run python classificar_mnar.py [--data sintetico|real] [--experiment <name>]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.args import parse_common_args
from utils.paths import get_comparison_dir, get_dataset_paths

warnings.filterwarnings("ignore")

_, DATA_TYPE, _, EXPERIMENT = parse_common_args()

DELTA_THRESHOLD = 0.03  # Diferença mínima de AUC para classificar como Diffuse

OUTPUT_DIR = os.path.join(get_comparison_dir(DATA_TYPE, EXPERIMENT), "mnar_classification")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("CLASSIFICACAO MNAR: FOCUSED vs DIFFUSE")
print("=" * 60)
print(f"Dados: {DATA_TYPE}")
print(f"Delta threshold: {DELTA_THRESHOLD}")
print("=" * 60)


def compute_mask_auc(df: pd.DataFrame, use_predictors: bool) -> float:
    """Calcula AUC de predição de mask de missing."""
    mask = df["X0"].isna().astype(int).values

    if mask.sum() < 5 or (1 - mask).sum() < 5:
        return 0.5

    x0_imputed = df["X0"].fillna(df["X0"].median()).values.reshape(-1, 1)

    if use_predictors:
        X_pred = df[["X1", "X2", "X3", "X4"]].values
        X = np.hstack([x0_imputed, X_pred])
    else:
        X = x0_imputed

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    try:
        scores = cross_val_score(
            LogisticRegression(max_iter=500, random_state=42),
            X, mask, cv=min(5, mask.sum(), (1 - mask).sum()),
            scoring="roc_auc"
        )
        return float(np.mean(scores))
    except Exception:
        return 0.5


# ======================================================
# PROCESSA DATASETS MNAR
# ======================================================
dataset_paths = get_dataset_paths(DATA_TYPE)
mnar_dir = dataset_paths["MNAR"]

if not os.path.isdir(mnar_dir):
    print(f"Diretorio MNAR nao encontrado: {mnar_dir}")
    sys.exit(1)

files = sorted([f for f in os.listdir(mnar_dir) if f.endswith(".txt")])

# Para dados reais com bootstrap, agrupar por dataset original
if DATA_TYPE == "real":
    import re
    groups = {}
    for f in files:
        group = re.sub(r"_boot\d+\.txt$", "", f)
        groups.setdefault(group, []).append(f)
    file_groups = groups
else:
    # Sintéticos: processar amostra representativa
    file_groups = {f.replace(".txt", ""): [f] for f in files[:100]}

results = []

for group_name, group_files in file_groups.items():
    aucs_complete = []
    aucs_excluded = []

    sample_files = group_files[:10]  # Limitar para eficiência

    for fname in sample_files:
        filepath = os.path.join(mnar_dir, fname)
        df = pd.read_csv(filepath, sep="\t")

        auc_complete = compute_mask_auc(df, use_predictors=True)
        auc_excluded = compute_mask_auc(df, use_predictors=False)
        aucs_complete.append(auc_complete)
        aucs_excluded.append(auc_excluded)

    mean_complete = np.mean(aucs_complete)
    mean_excluded = np.mean(aucs_excluded)
    delta = mean_complete - mean_excluded

    classification = "Diffuse" if delta > DELTA_THRESHOLD else "Focused"

    print(f"\n  {group_name}:")
    print(f"    AUC_complete (X0+X1..X4): {mean_complete:.4f}")
    print(f"    AUC_excluded (X0 only):   {mean_excluded:.4f}")
    print(f"    Delta: {delta:.4f}")
    print(f"    -> {classification} MNAR")

    results.append({
        "dataset": group_name,
        "auc_complete": round(mean_complete, 4),
        "auc_excluded": round(mean_excluded, 4),
        "delta": round(delta, 4),
        "classification": classification,
        "n_samples": len(sample_files),
    })

# ======================================================
# SALVA RESULTADOS
# ======================================================
df_results = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "mnar_focused_vs_diffuse.csv")
df_results.to_csv(csv_path, index=False)

print(f"\n{'='*60}")
print(f"CLASSIFICACAO CONCLUIDA!")
print(f"{'='*60}")
print(f"Resultados: {csv_path}")

n_focused = sum(1 for r in results if r["classification"] == "Focused")
n_diffuse = sum(1 for r in results if r["classification"] == "Diffuse")
print(f"Focused: {n_focused}, Diffuse: {n_diffuse}")
