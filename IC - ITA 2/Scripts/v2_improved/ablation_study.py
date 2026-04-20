"""
Estudo de Ablação — STEP 04-B.

Quantifica a contribuição marginal de cada grupo de features com 6 configurações:
  E1: Discriminativas originais (6 features)
  E2: E1 + Estatísticas invariantes + MNAR-específicas (15 features)
  E3: E2 + MechDetect (21 features) = baseline
  E4: E3 + CAAFE (25 features)
  E5: E3 + LLM v2 (29 features)
  E6: E3 + CAAFE + LLM (33 features) = todas

Testes de significância: Wilcoxon, McNemar, Bootstrap CI, Friedman + Nemenyi.

Uso:
    python ablation_study.py --data sintetico --experiment step05_pro --llm-model gemini-3.1-pro-preview
    python ablation_study.py --data real --experiment step05_pro --llm-model gemini-3.1-pro-preview
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, friedmanchisquare, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import get_output_dir, OUTPUT_BASE

warnings.filterwarnings("ignore")

# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(description="Ablation Study")
parser.add_argument("--data", choices=["sintetico", "real"], required=True)
parser.add_argument("--experiment", required=True)
parser.add_argument("--llm-model", default="gemini-3.1-pro-preview")
args = parser.parse_args()

DATA_TYPE = args.data
EXPERIMENT = args.experiment
LLM_MODEL = args.llm_model

BASELINE_DIR = get_output_dir(DATA_TYPE, "none", EXPERIMENT)
LLM_DIR = get_output_dir(DATA_TYPE, LLM_MODEL, EXPERIMENT)
OUT_DIR = os.path.join(OUTPUT_BASE, EXPERIMENT, DATA_TYPE, "ablacao")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("🔬 ESTUDO DE ABLAÇÃO — 6 CONFIGURAÇÕES (E1-E6)")
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

# ==============================================================================
# DEFINIR GRUPOS DE FEATURES
# ==============================================================================
DISC_COLS = ["auc_mask_from_Xobs", "coef_X1_abs", "log_pval_X1_mask",
             "X1_mean_diff", "X1_mannwhitney_pval", "little_proxy_score"]

STAT_INV_COLS = ["X0_missing_rate", "X0_obs_vs_full_ratio", "X0_iqr_ratio",
                 "X0_obs_skew_diff"]

MNAR_COLS = ["X0_ks_obs_vs_imputed", "X0_tail_missing_ratio", "mask_entropy",
             "X0_censoring_score", "X0_mean_shift_X1_to_X4"]

MECHDETECT_COLS = ["mechdetect_auc_complete", "mechdetect_auc_shuffled",
                   "mechdetect_auc_excluded", "mechdetect_delta_complete_shuffled",
                   "mechdetect_delta_complete_excluded", "mechdetect_mwu_pvalue"]

CAAFE_COLS = [c for c in X_full.columns if c.startswith("caafe_")]
LLM_COLS = [c for c in X_full.columns if c.startswith("llm_")]

CONFIGS = {
    "E1_disc": {
        "features": DISC_COLS,
        "desc": "Discriminativas (6f)",
        "n": len(DISC_COLS),
    },
    "E2_stat_mnar": {
        "features": DISC_COLS + STAT_INV_COLS + MNAR_COLS,
        "desc": "Disc+Stat+MNAR (15f)",
        "n": len(DISC_COLS + STAT_INV_COLS + MNAR_COLS),
    },
    "E3_baseline": {
        "features": DISC_COLS + STAT_INV_COLS + MNAR_COLS + MECHDETECT_COLS,
        "desc": "Baseline completo (21f)",
        "n": len(DISC_COLS + STAT_INV_COLS + MNAR_COLS + MECHDETECT_COLS),
    },
    "E4_caafe": {
        "features": DISC_COLS + STAT_INV_COLS + MNAR_COLS + MECHDETECT_COLS + CAAFE_COLS,
        "desc": "Baseline+CAAFE (25f)",
        "n": len(DISC_COLS + STAT_INV_COLS + MNAR_COLS + MECHDETECT_COLS + CAAFE_COLS),
    },
    "E5_llm": {
        "features": DISC_COLS + STAT_INV_COLS + MNAR_COLS + MECHDETECT_COLS + LLM_COLS,
        "desc": "Baseline+LLM (29f)",
        "n": len(DISC_COLS + STAT_INV_COLS + MNAR_COLS + MECHDETECT_COLS + LLM_COLS),
    },
    "E6_all": {
        "features": DISC_COLS + STAT_INV_COLS + MNAR_COLS + MECHDETECT_COLS + CAAFE_COLS + LLM_COLS,
        "desc": "Todas (33f)",
        "n": len(DISC_COLS + STAT_INV_COLS + MNAR_COLS + MECHDETECT_COLS + CAAFE_COLS + LLM_COLS),
    },
}

for name, cfg in CONFIGS.items():
    print(f"   {name}: {cfg['n']} features — {cfg['desc']}")

# ==============================================================================
# MODELOS
# ==============================================================================
def get_modelos(n_samples):
    if n_samples < 100:
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
            "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=0.5, random_state=42))]),
            "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1, random_state=42))]),
            "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=3))]),
            "MLP": Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42))]),
            "NaiveBayes": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        }
    return {
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=42))]),
        "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=3, random_state=42))]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "MLP": Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=42))]),
        "NaiveBayes": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
    }


def apply_smote(X_in, y_in):
    try:
        from imblearn.over_sampling import SMOTE
        min_count = pd.Series(y_in).value_counts().min() if not hasattr(y_in, "value_counts") else y_in.value_counts().min()
        if min_count >= 2:
            k = min(3, min_count - 1)
            return SMOTE(random_state=42, k_neighbors=k).fit_resample(X_in, y_in)
    except ImportError:
        pass
    return X_in, y_in


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
model_names = list(get_modelos(1000).keys())

print(f"\n📈 Split: train={len(y_train)}, test={len(y_test)}")

# ==============================================================================
# RODAR TODAS AS CONFIGURAÇÕES
# ==============================================================================
print(f"\n🏋️ Rodando 6 configs × 7 modelos...")
all_results = {}  # {config: {model: {acc, f1, y_pred, ...}}}

for cfg_name, cfg in tqdm(CONFIGS.items(), desc="Configs"):
    all_results[cfg_name] = {}
    feat_cols = cfg["features"]

    # Verificar quais colunas existem no X_full
    missing_cols = [c for c in feat_cols if c not in X_full.columns]
    if missing_cols:
        print(f"  ⚠️ {cfg_name}: colunas ausentes: {missing_cols}")
        feat_cols = [c for c in feat_cols if c in X_full.columns]

    X_sel = X_full[feat_cols]
    X_tr, X_te = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
    X_tr_sm, y_tr_sm = apply_smote(X_tr, y_train)

    for modelo_nome in model_names:
        modelo = get_modelos(len(X_tr_sm))[modelo_nome]
        modelo.fit(X_tr_sm, y_tr_sm)
        y_pred = modelo.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        all_results[cfg_name][modelo_nome] = {
            "accuracy": acc,
            "f1_macro": f1m,
            "y_pred": y_pred.copy(),
            "recall_MCAR": report.get("0", {}).get("recall", 0),
            "recall_MAR": report.get("1", {}).get("recall", 0),
            "recall_MNAR": report.get("2", {}).get("recall", 0),
        }

# ==============================================================================
# TABELA DE ABLAÇÃO
# ==============================================================================
print(f"\n{'='*70}")
print("📊 TABELA DE ABLAÇÃO")
print(f"{'='*70}")

rows = []
for cfg_name in CONFIGS:
    for m_name in model_names:
        r = all_results[cfg_name][m_name]
        rows.append({
            "config": cfg_name,
            "n_features": CONFIGS[cfg_name]["n"],
            "modelo": m_name,
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "recall_MCAR": r["recall_MCAR"],
            "recall_MAR": r["recall_MAR"],
            "recall_MNAR": r["recall_MNAR"],
        })

df_ablation = pd.DataFrame(rows)
df_ablation.to_csv(os.path.join(OUT_DIR, "ablacao_completa.csv"), index=False)

# Print pivot
pivot = df_ablation.pivot_table(index="modelo", columns="config", values="accuracy")
print(pivot.round(3).to_string())

# ==============================================================================
# BOOTSTRAP 95% CI
# ==============================================================================
print(f"\n📊 Bootstrap 95% CI...")
N_BOOTSTRAP = 1000
rng = np.random.RandomState(42)

ci_rows = []
for cfg_name in CONFIGS:
    for m_name in model_names:
        y_pred = all_results[cfg_name][m_name]["y_pred"]
        boot_accs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(y_test), len(y_test), replace=True)
            boot_accs.append(accuracy_score(y_test.values[idx], y_pred[idx]))
        ci_lo = np.percentile(boot_accs, 2.5)
        ci_hi = np.percentile(boot_accs, 97.5)
        ci_rows.append({
            "config": cfg_name,
            "modelo": m_name,
            "accuracy": all_results[cfg_name][m_name]["accuracy"],
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        })

df_ci = pd.DataFrame(ci_rows)
df_ci.to_csv(os.path.join(OUT_DIR, "bootstrap_ci.csv"), index=False)

# ==============================================================================
# TESTES DE SIGNIFICANCIA
# ==============================================================================
print(f"\n{'='*70}")
print("📊 TESTES DE SIGNIFICÂNCIA")
print(f"{'='*70}")

# Wilcoxon: para cada par consecutivo
pairs = [
    ("E1_disc", "E2_stat_mnar", "+Stat+MNAR"),
    ("E2_stat_mnar", "E3_baseline", "+MechDetect"),
    ("E3_baseline", "E4_caafe", "+CAAFE"),
    ("E3_baseline", "E5_llm", "+LLM"),
    ("E3_baseline", "E6_all", "+CAAFE+LLM"),
    ("E4_caafe", "E5_llm", "CAAFE vs LLM"),
]

sig_rows = []
for cfg_a, cfg_b, label in pairs:
    accs_a = [all_results[cfg_a][m]["accuracy"] for m in model_names]
    accs_b = [all_results[cfg_b][m]["accuracy"] for m in model_names]
    diffs = [b - a for a, b in zip(accs_a, accs_b)]

    if any(d != 0 for d in diffs):
        try:
            stat, p = wilcoxon(accs_a, accs_b)
        except Exception:
            stat, p = 0.0, 1.0
    else:
        stat, p = 0.0, 1.0

    mean_delta = np.mean(diffs)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {label:20s}: {cfg_a}→{cfg_b}  Δ_mean={mean_delta:+.4f}  W={stat:.1f}  p={p:.4f} {sig}")

    sig_rows.append({
        "config_a": cfg_a, "config_b": cfg_b, "label": label,
        "mean_delta": mean_delta, "wilcoxon_stat": stat, "wilcoxon_p": p,
        "significant_005": p < 0.05,
    })

    # McNemar for best model
    best_model = model_names[np.argmax(accs_b)]
    y_pred_a = all_results[cfg_a][best_model]["y_pred"]
    y_pred_b = all_results[cfg_b][best_model]["y_pred"]
    correct_a = (y_pred_a == y_test.values)
    correct_b = (y_pred_b == y_test.values)
    b_val = np.sum(correct_a & ~correct_b)
    c_val = np.sum(~correct_a & correct_b)
    if b_val + c_val > 0:
        mcn_stat = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
        mcn_p = 1 - chi2.cdf(mcn_stat, df=1)
    else:
        mcn_stat, mcn_p = 0.0, 1.0
    sig_rows[-1]["mcnemar_model"] = best_model
    sig_rows[-1]["mcnemar_chi2"] = mcn_stat
    sig_rows[-1]["mcnemar_p"] = mcn_p

pd.DataFrame(sig_rows).to_csv(os.path.join(OUT_DIR, "significancia.csv"), index=False)

# Friedman test
print(f"\n  Friedman test (6 configs × 7 modelos):")
config_names = list(CONFIGS.keys())
acc_matrix = np.zeros((len(model_names), len(config_names)))
for i, m in enumerate(model_names):
    for j, c in enumerate(config_names):
        acc_matrix[i, j] = all_results[c][m]["accuracy"]

try:
    friedman_stat, friedman_p = friedmanchisquare(*[acc_matrix[:, j] for j in range(len(config_names))])
    print(f"    χ²={friedman_stat:.4f}, p={friedman_p:.4f}")
    if friedman_p < 0.05:
        print(f"    ✅ Significativo — diferenças entre configs existem")
    else:
        print(f"    ❌ Não significativo")
except Exception as e:
    print(f"    Erro: {e}")
    friedman_stat, friedman_p = 0.0, 1.0

# ==============================================================================
# GRÁFICOS
# ==============================================================================

# 1. Heatmap de ablação
fig, ax = plt.subplots(figsize=(12, 6))
heat_data = np.zeros((len(config_names), len(model_names)))
for i, c in enumerate(config_names):
    for j, m in enumerate(model_names):
        heat_data[i, j] = all_results[c][m]["accuracy"]

im = ax.imshow(heat_data, cmap="RdYlGn", aspect="auto", vmin=0.2, vmax=0.9)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(config_names)))
ax.set_yticklabels([f"{c} ({CONFIGS[c]['n']}f)" for c in config_names], fontsize=8)
ax.set_title(f"Ablação de Features — {DATA_TYPE.upper()}")
plt.colorbar(im, ax=ax, label="Accuracy")
for i in range(len(config_names)):
    for j in range(len(model_names)):
        ax.text(j, i, f"{heat_data[i, j]:.1%}",
                ha="center", va="center", fontsize=7,
                color="white" if heat_data[i, j] < 0.5 else "black")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap_ablacao.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Line plot: accuracy média por config
fig, ax = plt.subplots(figsize=(10, 6))
mean_accs = [np.mean([all_results[c][m]["accuracy"] for m in model_names]) for c in config_names]
max_accs = [max(all_results[c][m]["accuracy"] for m in model_names) for c in config_names]
n_feats = [CONFIGS[c]["n"] for c in config_names]

ax.plot(n_feats, mean_accs, "o-", label="Média (7 modelos)", color="#3498db", linewidth=2, markersize=8)
ax.plot(n_feats, max_accs, "s--", label="Melhor modelo", color="#e74c3c", linewidth=2, markersize=8)
ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.3, label="Acaso")
for i, c in enumerate(config_names):
    ax.annotate(c.split("_")[0], (n_feats[i], mean_accs[i]),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
ax.set_xlabel("Número de Features")
ax.set_ylabel("Accuracy")
ax.set_title(f"Ablação: Accuracy vs N Features — {DATA_TYPE.upper()}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ablacao_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3. Bar chart with CI
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(config_names))
width = 0.1
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

for i, m in enumerate(model_names):
    accs = [all_results[c][m]["accuracy"] for c in config_names]
    ci_lo = [df_ci[(df_ci["config"] == c) & (df_ci["modelo"] == m)]["ci_lower"].values[0] for c in config_names]
    ci_hi = [df_ci[(df_ci["config"] == c) & (df_ci["modelo"] == m)]["ci_upper"].values[0] for c in config_names]
    yerr = [[a - lo for a, lo in zip(accs, ci_lo)],
            [hi - a for a, hi in zip(accs, ci_hi)]]
    ax.bar(x + i*width, accs, width, label=m, color=colors[i], yerr=yerr, capsize=2)

ax.set_xticks(x + width * len(model_names) / 2)
ax.set_xticklabels([f"{c}\n({CONFIGS[c]['n']}f)" for c in config_names], fontsize=7)
ax.set_ylabel("Accuracy")
ax.set_title(f"Ablação com 95% CI — {DATA_TYPE.upper()}")
ax.legend(fontsize=7, ncol=4)
ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ablacao_bars_ci.png"), dpi=300, bbox_inches="tight")
plt.close()

# ==============================================================================
# SUMMARY
# ==============================================================================
summary = {
    "timestamp": datetime.now().isoformat(),
    "data_type": DATA_TYPE,
    "experiment": EXPERIMENT,
    "llm_model": LLM_MODEL,
    "n_samples": int(len(y)),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "configs": {c: {"n_features": CONFIGS[c]["n"], "desc": CONFIGS[c]["desc"],
                     "mean_accuracy": float(np.mean([all_results[c][m]["accuracy"] for m in model_names])),
                     "max_accuracy": float(max(all_results[c][m]["accuracy"] for m in model_names)),
                     "best_model": model_names[np.argmax([all_results[c][m]["accuracy"] for m in model_names])],
                     } for c in config_names},
    "friedman_chi2": float(friedman_stat),
    "friedman_p": float(friedman_p),
}
with open(os.path.join(OUT_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Print final
print(f"\n{'='*70}")
print(f"✅ ABLAÇÃO CONCLUÍDA!")
print(f"{'='*70}")
print(f"\n📊 Resumo (accuracy média ± max):")
for c in config_names:
    mean_a = np.mean([all_results[c][m]["accuracy"] for m in model_names])
    max_a = max(all_results[c][m]["accuracy"] for m in model_names)
    best_m = model_names[np.argmax([all_results[c][m]["accuracy"] for m in model_names])]
    print(f"   {c:15s} ({CONFIGS[c]['n']:2d}f): mean={mean_a:.3f}  max={max_a:.3f} ({best_m})")

print(f"\n💾 Salvos em: {OUT_DIR}")
print(f"{'='*70}")
