"""
Análise SHAP + Error Analysis — STEP 08.

Gera:
  1. SHAP beeswarm por nível hierárquico (L1: MCAR vs NAO-MCAR, L2: MAR vs MNAR)
  2. SHAP comparison L1 vs L2 (importance ranking side-by-side)
  3. SHAP para classificação direta 3-way (comparação)
  4. Error analysis por variante sintética / dataset real
  5. t-SNE/UMAP do espaço de features
  6. Tabela comparativa final

Uso:
    python analyze_shap.py --data sintetico --experiment step05_pro --llm-model gemini-3.1-pro-preview
    python analyze_shap.py --data real --experiment step05_pro --llm-model gemini-3.1-pro-preview
"""
import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import get_output_dir, OUTPUT_BASE

warnings.filterwarnings("ignore")

# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(description="SHAP + Error Analysis")
parser.add_argument("--data", choices=["sintetico", "real"], required=True)
parser.add_argument("--experiment", required=True)
parser.add_argument("--llm-model", default="gemini-3.1-pro-preview")
args = parser.parse_args()

DATA_TYPE = args.data
EXPERIMENT = args.experiment
LLM_MODEL = args.llm_model

BASELINE_DIR = get_output_dir(DATA_TYPE, "none", EXPERIMENT)
LLM_DIR = get_output_dir(DATA_TYPE, LLM_MODEL, EXPERIMENT)
OUT_DIR = os.path.join(OUTPUT_BASE, EXPERIMENT, DATA_TYPE, "shap_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("📊 SHAP + ERROR ANALYSIS — STEP 08")
print("=" * 70)
print(f"📊 Dados: {DATA_TYPE}")
print(f"📂 Output: {OUT_DIR}")

# ==============================================================================
# CARREGA DADOS
# ==============================================================================
X_baseline = pd.read_csv(os.path.join(BASELINE_DIR, "X_features.csv"))
X_full = pd.read_csv(os.path.join(LLM_DIR, "X_features.csv"))
y = pd.read_csv(os.path.join(BASELINE_DIR, "y_labels.csv")).squeeze("columns")

groups = None
gpath = os.path.join(BASELINE_DIR, "groups.csv")
if os.path.exists(gpath):
    groups = pd.read_csv(gpath).squeeze("columns")

# Feature groups
STAT_COLS = list(X_baseline.columns)
CAAFE_COLS = [c for c in X_full.columns if c.startswith("caafe_")]
LLM_COLS = [c for c in X_full.columns if c.startswith("llm_")]
FEAT_STAT_CAAFE = STAT_COLS + CAAFE_COLS

CLASS_NAMES = {0: "MCAR", 1: "MAR", 2: "MNAR"}
print(f"📊 {len(y)} amostras, {len(STAT_COLS)} stat + {len(CAAFE_COLS)} CAAFE + {len(LLM_COLS)} LLM features")


def normalize_shap(sv_raw):
    """Converte SHAP values para lista de arrays [class0, class1, ...]."""
    if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        return [sv_raw[:, :, i] for i in range(sv_raw.shape[2])]
    if isinstance(sv_raw, list):
        return sv_raw
    return [sv_raw]

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


def apply_smote(X_in, y_in):
    try:
        from imblearn.over_sampling import SMOTE
        mc = pd.Series(y_in).value_counts().min() if not hasattr(y_in, "value_counts") else y_in.value_counts().min()
        if mc >= 2:
            return SMOTE(random_state=42, k_neighbors=min(3, mc - 1)).fit_resample(X_in, y_in)
    except ImportError:
        pass
    return X_in, y_in


# ==============================================================================
# 1. SHAP: CLASSIFICACAO DIRETA 3-WAY (baseline 21 features)
# ==============================================================================
print("\n🔬 1. SHAP — Classificação direta 3-way (21 stat features)...")

X_tr_stat = X_baseline.iloc[train_idx]
X_te_stat = X_baseline.iloc[test_idx]
X_tr_sm, y_tr_sm = apply_smote(X_tr_stat, y_train)

rf_direct = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_direct.fit(X_tr_sm, y_tr_sm)

explainer_direct = shap.TreeExplainer(rf_direct)
shap_values_direct_raw = explainer_direct.shap_values(X_te_stat)

# Normalizar formato: SHAP 0.51 retorna (n_samples, n_features, n_classes)
if isinstance(shap_values_direct_raw, np.ndarray) and shap_values_direct_raw.ndim == 3:
    shap_values_direct = [shap_values_direct_raw[:, :, i] for i in range(shap_values_direct_raw.shape[2])]
elif isinstance(shap_values_direct_raw, list):
    shap_values_direct = shap_values_direct_raw
else:
    shap_values_direct = [shap_values_direct_raw]

# Beeswarm para cada classe
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for i, cls_name in enumerate(["MCAR", "MAR", "MNAR"]):
    plt.sca(axes[i])
    shap.summary_plot(shap_values_direct[i], X_te_stat, show=False, max_display=15,
                      plot_size=None)
    axes[i].set_title(f"Classe {cls_name}", fontsize=12)
plt.suptitle(f"SHAP — Direto 3-way (21 stat) — {DATA_TYPE.upper()}", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_direto_3way.png"), dpi=200, bbox_inches="tight")
plt.close()

# Mean absolute SHAP por feature (agregado)
mean_shap_direct = np.mean([np.abs(shap_values_direct[i]).mean(axis=0) for i in range(len(shap_values_direct))], axis=0)
df_shap_direct = pd.DataFrame({
    "feature": X_te_stat.columns,
    "mean_abs_shap": mean_shap_direct,
}).sort_values("mean_abs_shap", ascending=False)
df_shap_direct.to_csv(os.path.join(OUT_DIR, "shap_importance_direto.csv"), index=False)
print(f"   Top 5: {list(df_shap_direct.head(5)['feature'])}")


# ==============================================================================
# 2. SHAP: NIVEL 1 (MCAR vs NAO-MCAR) — 21 stat features
# ==============================================================================
print("\n🔬 2. SHAP — Nível 1: MCAR vs NAO-MCAR (21 stat features)...")

y_train_l1 = (y_train != 0).astype(int)
y_test_l1 = (y_test != 0).astype(int)
X_tr_l1_sm, y_tr_l1_sm = apply_smote(X_tr_stat, y_train_l1)

rf_l1 = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_l1.fit(X_tr_l1_sm, y_tr_l1_sm)

explainer_l1 = shap.TreeExplainer(rf_l1)
shap_values_l1 = normalize_shap(explainer_l1.shap_values(X_te_stat))

fig, ax = plt.subplots(figsize=(10, 8))
plt.sca(ax)
# shap_values_l1[1] = SHAP for class NAO-MCAR
shap.summary_plot(shap_values_l1[1], X_te_stat, show=False, max_display=15, plot_size=None)
ax.set_title(f"SHAP Nível 1: NAO-MCAR (21 stat) — {DATA_TYPE.upper()}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_nivel1_stat.png"), dpi=200, bbox_inches="tight")
plt.close()

mean_shap_l1 = np.abs(shap_values_l1[1]).mean(axis=0)
df_shap_l1 = pd.DataFrame({
    "feature": X_te_stat.columns,
    "mean_abs_shap_l1": mean_shap_l1,
}).sort_values("mean_abs_shap_l1", ascending=False)
df_shap_l1.to_csv(os.path.join(OUT_DIR, "shap_importance_nivel1.csv"), index=False)
print(f"   Top 5 L1: {list(df_shap_l1.head(5)['feature'])}")


# ==============================================================================
# 3. SHAP: NIVEL 2 (MAR vs MNAR) — V3 usa stat+CAAFE (25 features)
# ==============================================================================
print("\n🔬 3. SHAP — Nível 2: MAR vs MNAR (stat+CAAFE, 25 features)...")

# Filtrar NAO-MCAR no treino e teste
mask_train_nm = y_train != 0
mask_test_nm = y_test != 0

X_l2_full = X_full[FEAT_STAT_CAAFE]
X_tr_l2 = X_l2_full.iloc[train_idx][mask_train_nm.values]
X_te_l2 = X_l2_full.iloc[test_idx][mask_test_nm.values]
y_train_l2 = (y_train[mask_train_nm] == 2).astype(int)  # 0=MAR, 1=MNAR
y_test_l2 = (y_test[mask_test_nm] == 2).astype(int)

X_tr_l2_sm, y_tr_l2_sm = apply_smote(X_tr_l2, y_train_l2)

rf_l2 = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_l2.fit(X_tr_l2_sm, y_tr_l2_sm)

explainer_l2 = shap.TreeExplainer(rf_l2)
shap_values_l2 = normalize_shap(explainer_l2.shap_values(X_te_l2))

fig, ax = plt.subplots(figsize=(10, 8))
plt.sca(ax)
# shap_values_l2[1] = SHAP for class MNAR
shap.summary_plot(shap_values_l2[1], X_te_l2, show=False, max_display=15, plot_size=None)
ax.set_title(f"SHAP Nível 2: MNAR (stat+CAAFE, 25f) — {DATA_TYPE.upper()}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_nivel2_caafe.png"), dpi=200, bbox_inches="tight")
plt.close()

mean_shap_l2 = np.abs(shap_values_l2[1]).mean(axis=0)
df_shap_l2 = pd.DataFrame({
    "feature": X_te_l2.columns,
    "mean_abs_shap_l2": mean_shap_l2,
}).sort_values("mean_abs_shap_l2", ascending=False)
df_shap_l2.to_csv(os.path.join(OUT_DIR, "shap_importance_nivel2_caafe.csv"), index=False)
print(f"   Top 5 L2: {list(df_shap_l2.head(5)['feature'])}")

# CAAFE features ranking no L2
caafe_in_l2 = df_shap_l2[df_shap_l2["feature"].str.startswith("caafe_")]
print(f"   CAAFE features no L2:")
for _, row in caafe_in_l2.iterrows():
    rank = df_shap_l2.index.get_loc(row.name) + 1
    print(f"     #{rank}: {row['feature']} = {row['mean_abs_shap_l2']:.4f}")


# ==============================================================================
# 3b. SHAP NIVEL 2 com LLM (33 features) — para comparação
# ==============================================================================
print("\n🔬 3b. SHAP — Nível 2: MAR vs MNAR (ALL, 33 features)...")

X_l2_all = X_full.iloc[train_idx][mask_train_nm.values]
X_te_l2_all = X_full.iloc[test_idx][mask_test_nm.values]

X_tr_l2_all_sm, y_tr_l2_all_sm = apply_smote(X_l2_all, y_train_l2)
rf_l2_all = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_l2_all.fit(X_tr_l2_all_sm, y_tr_l2_all_sm)

explainer_l2_all = shap.TreeExplainer(rf_l2_all)
shap_values_l2_all = normalize_shap(explainer_l2_all.shap_values(X_te_l2_all))

fig, ax = plt.subplots(figsize=(10, 8))
plt.sca(ax)
shap.summary_plot(shap_values_l2_all[1], X_te_l2_all, show=False, max_display=20, plot_size=None)
ax.set_title(f"SHAP Nível 2: MNAR (ALL, 33f) — {DATA_TYPE.upper()}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_nivel2_all.png"), dpi=200, bbox_inches="tight")
plt.close()

mean_shap_l2_all = np.abs(shap_values_l2_all[1]).mean(axis=0)
df_shap_l2_all = pd.DataFrame({
    "feature": X_te_l2_all.columns,
    "mean_abs_shap_l2_all": mean_shap_l2_all,
}).sort_values("mean_abs_shap_l2_all", ascending=False)
df_shap_l2_all.to_csv(os.path.join(OUT_DIR, "shap_importance_nivel2_all.csv"), index=False)


# ==============================================================================
# 4. COMPARISON: L1 vs L2 importance (side-by-side)
# ==============================================================================
print("\n🔬 4. Comparação SHAP: L1 vs L2...")

# Merge importances (apenas features comuns = stat 21)
df_comp = df_shap_l1.merge(
    df_shap_l2[df_shap_l2["feature"].isin(STAT_COLS)][["feature", "mean_abs_shap_l2"]],
    on="feature", how="left"
).fillna(0)
df_comp["delta"] = df_comp["mean_abs_shap_l2"] - df_comp["mean_abs_shap_l1"]
df_comp = df_comp.sort_values("delta", ascending=False)
df_comp.to_csv(os.path.join(OUT_DIR, "shap_comparison_l1_vs_l2.csv"), index=False)

# Plot side-by-side bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# L1 importance
top_l1 = df_shap_l1.head(15)
axes[0].barh(range(len(top_l1)), top_l1["mean_abs_shap_l1"].values, color="#3498db")
axes[0].set_yticks(range(len(top_l1)))
axes[0].set_yticklabels(top_l1["feature"].values, fontsize=8)
axes[0].set_title("Nível 1: MCAR vs NAO-MCAR", fontsize=11)
axes[0].set_xlabel("Mean |SHAP|")
axes[0].invert_yaxis()

# L2 importance (with CAAFE)
top_l2 = df_shap_l2.head(15)
colors_l2 = ["#e74c3c" if f.startswith("caafe_") else "#2ecc71" for f in top_l2["feature"]]
axes[1].barh(range(len(top_l2)), top_l2["mean_abs_shap_l2"].values, color=colors_l2)
axes[1].set_yticks(range(len(top_l2)))
axes[1].set_yticklabels(top_l2["feature"].values, fontsize=8)
axes[1].set_title("Nível 2: MAR vs MNAR (stat+CAAFE)", fontsize=11)
axes[1].set_xlabel("Mean |SHAP|")
axes[1].invert_yaxis()

# Legend for L2
from matplotlib.patches import Patch
axes[1].legend(handles=[
    Patch(facecolor="#2ecc71", label="Stat features"),
    Patch(facecolor="#e74c3c", label="CAAFE features"),
], loc="lower right", fontsize=9)

plt.suptitle(f"SHAP Importance: L1 vs L2 — {DATA_TYPE.upper()}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_l1_vs_l2_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()


# ==============================================================================
# 5. DEPENDENCE PLOTS — top CAAFE features no L2
# ==============================================================================
print("\n🔬 5. SHAP Dependence Plots (CAAFE features no L2)...")

fig, axes = plt.subplots(1, min(4, len(CAAFE_COLS)), figsize=(5 * min(4, len(CAAFE_COLS)), 5))
if len(CAAFE_COLS) == 1:
    axes = [axes]

for i, col in enumerate(CAAFE_COLS[:4]):
    if col in X_te_l2.columns:
        col_idx = list(X_te_l2.columns).index(col)
        plt.sca(axes[i])
        shap.dependence_plot(col_idx, shap_values_l2[1], X_te_l2, show=False, ax=axes[i])
        axes[i].set_title(col, fontsize=9)

plt.suptitle(f"SHAP Dependence: CAAFE no Nível 2 — {DATA_TYPE.upper()}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_dependence_caafe.png"), dpi=200, bbox_inches="tight")
plt.close()


# ==============================================================================
# 6. ERROR ANALYSIS
# ==============================================================================
print("\n🔬 6. Error Analysis...")

# Train V3 hierárquica (L1 stat, L2 stat+CAAFE) para analisar erros
# L1
rf_l1_err = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
X_tr_l1_sm_err, y_tr_l1_sm_err = apply_smote(X_tr_stat, y_train_l1)
rf_l1_err.fit(X_tr_l1_sm_err, y_tr_l1_sm_err)
pred_l1 = rf_l1_err.predict(X_te_stat)

# L2
gb_l2_err = GradientBoostingClassifier(n_estimators=300, random_state=42)
X_tr_l2_sm_err, y_tr_l2_sm_err = apply_smote(X_tr_l2, y_train_l2)
gb_l2_err.fit(X_tr_l2_sm_err, y_tr_l2_sm_err)

# Combine
X_te_l2_for_pred = X_l2_full.iloc[test_idx]
y_pred_hier = np.zeros(len(y_test), dtype=int)
mask_nm = pred_l1 == 1
if mask_nm.any():
    pred_l2 = gb_l2_err.predict(X_te_l2_for_pred[mask_nm])
    y_pred_hier[mask_nm] = np.where(pred_l2 == 0, 1, 2)

# Error DataFrame
df_errors = pd.DataFrame({
    "idx": test_idx,
    "y_true": y_test.values,
    "y_pred": y_pred_hier,
    "correct": (y_test.values == y_pred_hier),
    "true_label": [CLASS_NAMES[v] for v in y_test.values],
    "pred_label": [CLASS_NAMES[v] for v in y_pred_hier],
})

# Add group info
if groups is not None:
    df_errors["group"] = groups.iloc[test_idx].values
    # Extract dataset name
    df_errors["dataset"] = df_errors["group"].apply(
        lambda g: re.sub(r"^(MCAR|MAR|MNAR)_", "", str(g)))

# Add file names if available
files_path = os.path.join(BASELINE_DIR, "file_names.csv")
if os.path.exists(files_path):
    file_names = pd.read_csv(files_path).squeeze("columns")
    df_errors["file"] = file_names.iloc[test_idx].values

# Error by class
print("\n   Accuracy por classe (V3 Hier+CAAFE, GBT L2):")
for cls_code, cls_name in CLASS_NAMES.items():
    mask_cls = y_test.values == cls_code
    if mask_cls.sum() > 0:
        acc_cls = (y_pred_hier[mask_cls] == cls_code).mean()
        n = mask_cls.sum()
        print(f"     {cls_name}: {acc_cls:.3f} ({int(acc_cls * n)}/{n})")

# Error by group (dataset)
if "group" in df_errors.columns:
    print("\n   Accuracy por dataset (top 5 melhores e piores):")
    group_acc = df_errors.groupby("group").agg(
        accuracy=("correct", "mean"),
        n=("correct", "count"),
        true_label=("true_label", "first"),
    ).sort_values("accuracy")

    print("   PIORES:")
    for _, row in group_acc.head(5).iterrows():
        print(f"     {row.name}: {row['accuracy']:.1%} ({row['true_label']}, n={row['n']})")
    print("   MELHORES:")
    for _, row in group_acc.tail(5).iterrows():
        print(f"     {row.name}: {row['accuracy']:.1%} ({row['true_label']}, n={row['n']})")

    group_acc.to_csv(os.path.join(OUT_DIR, "error_by_dataset.csv"))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_hier, labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(8, 6))
labels = ["MCAR", "MAR", "MNAR"]
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title(f"V3 Hier+CAAFE (GBT L2) — {DATA_TYPE.upper()}\n"
             f"Acc={accuracy_score(y_test, y_pred_hier):.1%}")
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_yticks(range(3)); ax.set_yticklabels(labels)
ax.set_xlabel("Predito"); ax.set_ylabel("Real")
plt.colorbar(im, ax=ax)
thresh = cm.max() / 2.
for i in range(3):
    for j in range(3):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_v3_hier.png"), dpi=200, bbox_inches="tight")
plt.close()

df_errors.to_csv(os.path.join(OUT_DIR, "error_analysis.csv"), index=False)


# ==============================================================================
# 7. t-SNE / UMAP
# ==============================================================================
print("\n🔬 7. t-SNE + UMAP...")

# Use stat+CAAFE features
X_vis = X_full[FEAT_STAT_CAAFE].iloc[test_idx]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vis)

# t-SNE
print("   t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_vis) - 1))
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors_class = {0: "#3498db", 1: "#2ecc71", 2: "#e74c3c"}

# By true class
ax = axes[0]
for cls_code, cls_name in CLASS_NAMES.items():
    mask = y_test.values == cls_code
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors_class[cls_code],
               label=cls_name, alpha=0.6, s=15)
ax.set_title("t-SNE — Classe Real")
ax.legend()

# By correct/incorrect
ax = axes[1]
correct = y_test.values == y_pred_hier
ax.scatter(X_tsne[correct, 0], X_tsne[correct, 1], c="#2ecc71", label="Correto", alpha=0.5, s=15)
ax.scatter(X_tsne[~correct, 0], X_tsne[~correct, 1], c="#e74c3c", label="Erro", alpha=0.7, s=20, marker="x")
ax.set_title("t-SNE — Acerto vs Erro")
ax.legend()

plt.suptitle(f"t-SNE (stat+CAAFE features) — {DATA_TYPE.upper()}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsne_features.png"), dpi=200, bbox_inches="tight")
plt.close()

# UMAP
try:
    import umap
    print("   UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for cls_code, cls_name in CLASS_NAMES.items():
        mask = y_test.values == cls_code
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=colors_class[cls_code],
                   label=cls_name, alpha=0.6, s=15)
    ax.set_title("UMAP — Classe Real")
    ax.legend()

    ax = axes[1]
    ax.scatter(X_umap[correct, 0], X_umap[correct, 1], c="#2ecc71", label="Correto", alpha=0.5, s=15)
    ax.scatter(X_umap[~correct, 0], X_umap[~correct, 1], c="#e74c3c", label="Erro", alpha=0.7, s=20, marker="x")
    ax.set_title("UMAP — Acerto vs Erro")
    ax.legend()

    plt.suptitle(f"UMAP (stat+CAAFE features) — {DATA_TYPE.upper()}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "umap_features.png"), dpi=200, bbox_inches="tight")
    plt.close()
except ImportError:
    print("   UMAP não disponível")


# ==============================================================================
# 8. SUMMARY
# ==============================================================================
summary = {
    "timestamp": datetime.now().isoformat(),
    "data_type": DATA_TYPE,
    "experiment": EXPERIMENT,
    "n_test": int(len(y_test)),
    "shap_model": "RandomForest (n=400)",
    "top5_l1": list(df_shap_l1.head(5)["feature"]),
    "top5_l2_caafe": list(df_shap_l2.head(5)["feature"]),
    "top5_l2_all": list(df_shap_l2_all.head(5)["feature"]),
    "caafe_features_ranking_l2": {
        row["feature"]: {"rank": i + 1, "mean_abs_shap": float(row["mean_abs_shap_l2"])}
        for i, (_, row) in enumerate(df_shap_l2.iterrows())
        if row["feature"].startswith("caafe_")
    },
}
with open(os.path.join(OUT_DIR, "shap_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"✅ SHAP + ERROR ANALYSIS CONCLUÍDO!")
print(f"{'='*70}")
print(f"\n💾 Salvos em: {OUT_DIR}")
for f_name in sorted(os.listdir(OUT_DIR)):
    print(f"   - {f_name}")
print(f"{'='*70}")
