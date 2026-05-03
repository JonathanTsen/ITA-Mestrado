"""
Classificação Hierárquica com 6 Variantes (V1-V6) — STEP 05-A.

Compara classificação direta vs hierárquica com features LLM seletivas por nível.
Carrega features baseline e LLM do mesmo experimento para montar as variantes.

Variantes:
  V1: Direto 3-way — 21 features baseline
  V2: Hierárquico puro — L1: 21 stat, L2: 21 stat
  V3: Hier + CAAFE no N2 — L1: 21 stat, L2: 25 (stat+CAAFE)
  V4: Hier + LLM no N2 — L1: 21 stat, L2: 33 (stat+CAAFE+LLM)  [CORE]
  V5: Hier + LLM em ambos — L1: 33, L2: 33  [Controle]
  V6: Direto 3-way + LLM — 33 features  [Controle]

Uso:
    python train_hierarchical_variants.py --data sintetico --experiment step05_pro --llm-model gemini-3.1-pro-preview
    python train_hierarchical_variants.py --data real --experiment step05_pro --llm-model gemini-3.1-pro-preview
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

from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import get_output_dir, get_comparison_dir, OUTPUT_BASE

warnings.filterwarnings("ignore")

# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(description="Hierárquica 6 variantes")
parser.add_argument("--data", choices=["sintetico", "real"], required=True)
parser.add_argument("--experiment", required=True)
parser.add_argument("--llm-model", default="gemini-3.1-pro-preview",
                    help="Nome do modelo LLM cujas features serão usadas")
args = parser.parse_args()

DATA_TYPE = args.data
EXPERIMENT = args.experiment
LLM_MODEL = args.llm_model

# ==============================================================================
# PATHS
# ==============================================================================
BASELINE_DIR = get_output_dir(DATA_TYPE, "none", EXPERIMENT)
LLM_DIR = get_output_dir(DATA_TYPE, LLM_MODEL, EXPERIMENT)
HIER_DIR = os.path.join(OUTPUT_BASE, EXPERIMENT, DATA_TYPE, "hierarquico_variants")
os.makedirs(HIER_DIR, exist_ok=True)

# Verifica existência dos arquivos
for label, d in [("Baseline", BASELINE_DIR), ("LLM", LLM_DIR)]:
    fpath = os.path.join(d, "X_features.csv")
    if not os.path.exists(fpath):
        print(f"❌ {label} features não encontradas: {fpath}")
        sys.exit(1)

print("=" * 70)
print("🔀 CLASSIFICAÇÃO HIERÁRQUICA — 6 VARIANTES (V1-V6)")
print("=" * 70)
print(f"📊 Dados: {DATA_TYPE}")
print(f"🤖 LLM: {LLM_MODEL}")
print(f"📂 Baseline: {BASELINE_DIR}")
print(f"📂 LLM:      {LLM_DIR}")
print(f"📂 Output:   {HIER_DIR}")
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

# Identificar colunas por tipo
STAT_COLS = [c for c in X_baseline.columns]  # 21 features
CAAFE_COLS = [c for c in X_llm_full.columns if c.startswith("caafe_")]
LLM_COLS = [c for c in X_llm_full.columns if c.startswith("llm_")]
ALL_COLS = list(X_llm_full.columns)

print(f"\n📊 Dados: {len(y)} amostras")
print(f"   Baseline: {len(STAT_COLS)} features")
print(f"   CAAFE: {len(CAAFE_COLS)} features: {CAAFE_COLS}")
print(f"   LLM: {len(LLM_COLS)} features: {LLM_COLS}")
print(f"   Total: {len(ALL_COLS)} features")
print(f"   Classes: {dict(y.value_counts().sort_index())}")

CLASS_NAMES = {0: "MCAR", 1: "MAR", 2: "MNAR"}

# Feature subsets
FEAT_STAT = STAT_COLS                          # 21
FEAT_STAT_CAAFE = STAT_COLS + CAAFE_COLS       # 25
FEAT_ALL = ALL_COLS                            # 33

# ==============================================================================
# DEFINIÇÃO DAS 6 VARIANTES
# ==============================================================================
VARIANTES = {
    "V1_direto_stat": {
        "tipo": "direto",
        "features": FEAT_STAT,
        "desc": "Direto 3-way (21 stat)",
    },
    "V2_hier_stat": {
        "tipo": "hierarquico",
        "features_l1": FEAT_STAT,
        "features_l2": FEAT_STAT,
        "desc": "Hier: L1=stat, L2=stat",
    },
    "V3_hier_caafe_n2": {
        "tipo": "hierarquico",
        "features_l1": FEAT_STAT,
        "features_l2": FEAT_STAT_CAAFE,
        "desc": "Hier: L1=stat, L2=stat+CAAFE",
    },
    "V4_hier_llm_n2": {
        "tipo": "hierarquico",
        "features_l1": FEAT_STAT,
        "features_l2": FEAT_ALL,
        "desc": "Hier: L1=stat, L2=stat+CAAFE+LLM [CORE]",
    },
    "V5_hier_llm_ambos": {
        "tipo": "hierarquico",
        "features_l1": FEAT_ALL,
        "features_l2": FEAT_ALL,
        "desc": "Hier: L1=ALL, L2=ALL [Controle]",
    },
    "V6_direto_llm": {
        "tipo": "direto",
        "features": FEAT_ALL,
        "desc": "Direto 3-way (33 ALL) [Controle]",
    },
}


# ==============================================================================
# MODELOS
# ==============================================================================
def get_modelos(n_samples: int) -> dict:
    if n_samples < 100:
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, C=0.5, random_state=42))]),
            "SVM_RBF": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1, random_state=42, probability=True))]),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=3))]),
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(32, 16),
                                      max_iter=2000, random_state=42))]),
            "NaiveBayes": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB())]),
        }
    else:
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=400, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=300, random_state=42),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=42))]),
            "SVM_RBF": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=3, random_state=42, probability=True))]),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5))]),
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                      max_iter=2000, random_state=42))]),
            "NaiveBayes": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB())]),
        }


def apply_smote(X_in, y_in):
    try:
        from imblearn.over_sampling import SMOTE
        min_count = y_in.value_counts().min() if hasattr(y_in, "value_counts") else pd.Series(y_in).value_counts().min()
        if min_count >= 2:
            k = min(3, min_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k)
            return smote.fit_resample(X_in, y_in)
    except ImportError:
        pass
    return X_in, y_in


# ==============================================================================
# SPLIT — mesmo para todas as variantes
# ==============================================================================
if groups is not None and groups.nunique() > 1:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X_baseline, y, groups))
else:
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X_baseline))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, stratify=y, random_state=42)

y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
y_test_l1 = (y_test != 0).astype(int)

print(f"\n📈 Split: train={len(y_train)}, test={len(y_test)}")
print(f"   Test dist: {dict(y_test.value_counts().sort_index())}")


# ==============================================================================
# FUNCOES DE CLASSIFICACAO
# ==============================================================================
def run_direct(X_full, y, train_idx, test_idx, feat_cols, modelo_nome):
    """Classificação direta 3-way."""
    X_sel = X_full[feat_cols]
    X_tr, X_te = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr)
    modelo = get_modelos(len(X_tr_sm))[modelo_nome]
    modelo.fit(X_tr_sm, y_tr_sm)
    y_pred = modelo.predict(X_te)
    return y_pred


def run_hierarchical(X_full, y, train_idx, test_idx, feat_l1, feat_l2, modelo_nome):
    """Classificação hierárquica: L1 (MCAR vs NAO-MCAR), L2 (MAR vs MNAR)."""
    X_l1 = X_full[feat_l1]
    X_l2 = X_full[feat_l2]

    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # Nível 1: MCAR vs NAO-MCAR
    y_tr_l1 = (y_tr != 0).astype(int)
    X_tr_l1 = X_l1.iloc[train_idx]
    X_te_l1 = X_l1.iloc[test_idx]
    X_tr_l1_sm, y_tr_l1_sm = apply_smote(X_tr_l1, y_tr_l1)
    m_l1 = get_modelos(len(X_tr_l1_sm))[modelo_nome]
    m_l1.fit(X_tr_l1_sm, y_tr_l1_sm)
    pred_l1 = m_l1.predict(X_te_l1)

    # Nível 2: MAR vs MNAR
    mask_tr_nm = y_tr != 0
    X_tr_l2 = X_l2.iloc[train_idx][mask_tr_nm.values]
    y_tr_l2 = (y_tr[mask_tr_nm] == 2).astype(int)
    X_tr_l2_sm, y_tr_l2_sm = apply_smote(X_tr_l2, y_tr_l2)

    m_l2 = get_modelos(len(X_tr_l2_sm))[modelo_nome]
    m_l2.fit(X_tr_l2_sm, y_tr_l2_sm)

    # Combinar
    X_te_l2 = X_l2.iloc[test_idx]
    y_pred = np.zeros(len(y_te), dtype=int)
    mask_notmcar = pred_l1 == 1
    y_pred[~mask_notmcar] = 0  # MCAR

    if mask_notmcar.any():
        pred_l2 = m_l2.predict(X_te_l2[mask_notmcar])
        y_pred[mask_notmcar] = np.where(pred_l2 == 0, 1, 2)  # MAR ou MNAR

    acc_l1 = accuracy_score((y_te != 0).astype(int), pred_l1)

    # Accuracy nível 2 condicionada
    mask_true_notmcar = y_te != 0
    mask_correct_notmcar = mask_notmcar & mask_true_notmcar.values
    if mask_correct_notmcar.any():
        y_te_l2_subset = y_te.values[mask_correct_notmcar]
        y_pred_l2_subset = y_pred[mask_correct_notmcar]
        acc_l2 = accuracy_score(y_te_l2_subset, y_pred_l2_subset)
    else:
        acc_l2 = float("nan")

    return y_pred, acc_l1, acc_l2


# ==============================================================================
# RODAR TODAS AS VARIANTES x TODOS OS MODELOS
# ==============================================================================
print("\n🏋️ Rodando 6 variantes × 7 modelos...")
model_names = list(get_modelos(1000).keys())
all_results = {}  # {variante: {modelo: {metrics}}}

for var_name, var_cfg in VARIANTES.items():
    print(f"\n  📐 {var_name}: {var_cfg['desc']}")
    all_results[var_name] = {}

    for modelo_nome in tqdm(model_names, desc=f"  {var_name}", leave=False):
        if var_cfg["tipo"] == "direto":
            y_pred = run_direct(X_llm_full, y, train_idx, test_idx,
                                var_cfg["features"], modelo_nome)
            acc_l1 = float("nan")
            acc_l2 = float("nan")
        else:
            y_pred, acc_l1, acc_l2 = run_hierarchical(
                X_llm_full, y, train_idx, test_idx,
                var_cfg["features_l1"], var_cfg["features_l2"], modelo_nome)

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
print("📊 TABELA COMPARATIVA — TODAS AS VARIANTES")
print(f"{'='*70}")

rows = []
for var_name in VARIANTES:
    for modelo_nome in model_names:
        r = all_results[var_name][modelo_nome]
        rows.append({
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
        })

df_all = pd.DataFrame(rows)
df_all.to_csv(os.path.join(HIER_DIR, "todas_variantes.csv"), index=False)

# Resumo por variante (média dos modelos)
df_summary = df_all.groupby("variante").agg({
    "accuracy": ["mean", "std", "max"],
    "f1_macro": ["mean", "std", "max"],
    "recall_MNAR": ["mean", "std", "max"],
    "acc_level1": "mean",
    "acc_level2": "mean",
}).round(4)
df_summary.to_csv(os.path.join(HIER_DIR, "resumo_variantes.csv"))

print("\n📊 Accuracy média por variante:")
for var_name in VARIANTES:
    subset = df_all[df_all["variante"] == var_name]
    acc_mean = subset["accuracy"].mean()
    acc_max = subset["accuracy"].max()
    mnar_mean = subset["recall_MNAR"].mean()
    best_model = subset.loc[subset["accuracy"].idxmax(), "modelo"]
    print(f"   {var_name:25s}: acc_mean={acc_mean:.3f} acc_max={acc_max:.3f} "
          f"(by {best_model}) MNAR_recall_mean={mnar_mean:.3f}")


# ==============================================================================
# COMPARACAO DETALHADA: V1 vs V2 vs V4 vs V5
# ==============================================================================
print(f"\n{'='*70}")
print("📊 COMPARAÇÃO DETALHADA POR MODELO")
print(f"{'='*70}")

comparison_rows = []
for modelo_nome in model_names:
    row = {"modelo": modelo_nome}
    for var_name in VARIANTES:
        r = all_results[var_name][modelo_nome]
        prefix = var_name.split("_")[0]  # V1, V2, etc.
        row[f"{prefix}_acc"] = r["accuracy"]
        row[f"{prefix}_f1macro"] = r["f1_macro"]
        row[f"{prefix}_mnar_recall"] = r["recall_MNAR"]
        row[f"{prefix}_mnar_f1"] = r["f1_MNAR"]
        if not np.isnan(r["acc_level1"]):
            row[f"{prefix}_acc_l1"] = r["acc_level1"]
            row[f"{prefix}_acc_l2"] = r["acc_level2"]
    comparison_rows.append(row)

    # Print
    v1 = all_results["V1_direto_stat"][modelo_nome]
    v4 = all_results["V4_hier_llm_n2"][modelo_nome]
    delta_acc = v4["accuracy"] - v1["accuracy"]
    delta_mnar = v4["recall_MNAR"] - v1["recall_MNAR"]
    sym = "✅" if delta_acc > 0 else "❌" if delta_acc < 0 else "➖"
    print(f"  {sym} {modelo_nome:20s}: V1={v1['accuracy']:.3f} V4={v4['accuracy']:.3f} "
          f"(Δ={delta_acc:+.3f}) | MNAR: V1={v1['recall_MNAR']:.3f} V4={v4['recall_MNAR']:.3f} "
          f"(Δ={delta_mnar:+.3f})")

df_comparison = pd.DataFrame(comparison_rows)
df_comparison.to_csv(os.path.join(HIER_DIR, "comparacao_por_modelo.csv"), index=False)


# ==============================================================================
# TESTES DE SIGNIFICANCIA — McNemar entre V1 e V4
# ==============================================================================
print(f"\n{'='*70}")
print("📊 TESTES DE SIGNIFICÂNCIA")
print(f"{'='*70}")

sig_rows = []
for modelo_nome in model_names:
    y_pred_v1 = all_results["V1_direto_stat"][modelo_nome]["y_pred"]
    y_pred_v4 = all_results["V4_hier_llm_n2"][modelo_nome]["y_pred"]

    # McNemar: conta discordâncias
    correct_v1 = (y_pred_v1 == y_test.values)
    correct_v4 = (y_pred_v4 == y_test.values)
    b = np.sum(correct_v1 & ~correct_v4)  # V1 certo, V4 errado
    c = np.sum(~correct_v1 & correct_v4)  # V1 errado, V4 certo

    # McNemar test (chi2 com continuity correction)
    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2
        mcnemar_p = 1 - chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_stat = 0.0
        mcnemar_p = 1.0

    sig_star = "***" if mcnemar_p < 0.001 else "**" if mcnemar_p < 0.01 else "*" if mcnemar_p < 0.05 else ""
    print(f"  {modelo_nome:20s}: V1→V4  b={b:3d}  c={c:3d}  McNemar χ²={mcnemar_stat:.2f}  p={mcnemar_p:.4f} {sig_star}")

    sig_rows.append({
        "modelo": modelo_nome,
        "v1_correct_v4_wrong": int(b),
        "v1_wrong_v4_correct": int(c),
        "mcnemar_chi2": mcnemar_stat,
        "mcnemar_p": mcnemar_p,
        "significant_005": mcnemar_p < 0.05,
    })

df_sig = pd.DataFrame(sig_rows)
df_sig.to_csv(os.path.join(HIER_DIR, "significancia_mcnemar.csv"), index=False)

# Wilcoxon: comparação pareada de accuracy entre variantes (sobre os 7 modelos)
print(f"\n  Wilcoxon signed-rank (V1 vs V4, sobre 7 modelos):")
accs_v1 = [all_results["V1_direto_stat"][m]["accuracy"] for m in model_names]
accs_v4 = [all_results["V4_hier_llm_n2"][m]["accuracy"] for m in model_names]
diffs = [a4 - a1 for a1, a4 in zip(accs_v1, accs_v4)]
if any(d != 0 for d in diffs):
    try:
        stat_w, p_w = wilcoxon(accs_v1, accs_v4)
        print(f"    statistic={stat_w:.4f}, p={p_w:.4f}")
    except Exception as e:
        print(f"    Erro: {e}")
        p_w = 1.0
else:
    print(f"    Sem diferença entre V1 e V4")
    p_w = 1.0

# Wilcoxon V2 vs V4 (hierárquica pura vs hierárquica+LLM)
accs_v2 = [all_results["V2_hier_stat"][m]["accuracy"] for m in model_names]
diffs_v2v4 = [a4 - a2 for a2, a4 in zip(accs_v2, accs_v4)]
if any(d != 0 for d in diffs_v2v4):
    try:
        stat_w2, p_w2 = wilcoxon(accs_v2, accs_v4)
        print(f"\n  Wilcoxon V2 vs V4 (efeito do LLM no N2):")
        print(f"    statistic={stat_w2:.4f}, p={p_w2:.4f}")
    except Exception as e:
        print(f"    Erro: {e}")

# Wilcoxon V4 vs V5 (LLM só N2 vs LLM em ambos)
accs_v5 = [all_results["V5_hier_llm_ambos"][m]["accuracy"] for m in model_names]
diffs_v4v5 = [a4 - a5 for a4, a5 in zip(accs_v4, accs_v5)]
if any(d != 0 for d in diffs_v4v5):
    try:
        stat_w3, p_w3 = wilcoxon(accs_v4, accs_v5)
        print(f"\n  Wilcoxon V4 vs V5 (LLM só N2 vs ambos):")
        print(f"    statistic={stat_w3:.4f}, p={p_w3:.4f}")
    except Exception as e:
        print(f"    Erro: {e}")


# ==============================================================================
# LOGO CROSS-VALIDATION (para variantes-chave: V1, V2, V4)
# ==============================================================================
if groups is not None and groups.nunique() > 2:
    print(f"\n{'='*70}")
    print("📊 LEAVE-ONE-GROUP-OUT CV (V1, V2, V4)")
    print(f"{'='*70}")

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(X_baseline, y, groups)
    print(f"   {n_folds} folds")

    cv_variantes = ["V1_direto_stat", "V2_hier_stat", "V3_hier_caafe_n2", "V4_hier_llm_n2"]
    cv_results = {v: {} for v in cv_variantes}

    for modelo_nome in tqdm(model_names, desc="LOGO CV"):
        for var_name in cv_variantes:
            var_cfg = VARIANTES[var_name]
            fold_accs = []

            for tr_i, te_i in logo.split(X_baseline, y, groups):
                y_tr, y_te = y.iloc[tr_i], y.iloc[te_i]

                if var_cfg["tipo"] == "direto":
                    feat = var_cfg["features"]
                    X_sel = X_llm_full[feat]
                    X_tr_f, X_te_f = X_sel.iloc[tr_i], X_sel.iloc[te_i]
                    X_tr_sm, y_tr_sm = apply_smote(X_tr_f, y_tr)
                    m = get_modelos(len(X_tr_sm))[modelo_nome]
                    m.fit(X_tr_sm, y_tr_sm)
                    y_pred = m.predict(X_te_f)
                else:
                    feat_l1 = var_cfg["features_l1"]
                    feat_l2 = var_cfg["features_l2"]

                    # L1
                    X_l1 = X_llm_full[feat_l1]
                    y_tr_l1 = (y_tr != 0).astype(int)
                    X_tr_l1_sm, y_tr_l1_sm = apply_smote(X_l1.iloc[tr_i], y_tr_l1)
                    m_l1 = get_modelos(len(X_tr_l1_sm))[modelo_nome]
                    m_l1.fit(X_tr_l1_sm, y_tr_l1_sm)
                    pred_l1 = m_l1.predict(X_l1.iloc[te_i])

                    # L2
                    X_l2 = X_llm_full[feat_l2]
                    mask_nm = y_tr != 0
                    X_tr_l2 = X_l2.iloc[tr_i][mask_nm.values]
                    y_tr_l2 = (y_tr[mask_nm] == 2).astype(int)

                    if len(y_tr_l2) >= 2 and y_tr_l2.nunique() >= 2:
                        X_tr_l2_sm, y_tr_l2_sm = apply_smote(X_tr_l2, y_tr_l2)
                        m_l2 = get_modelos(len(X_tr_l2_sm))[modelo_nome]
                        m_l2.fit(X_tr_l2_sm, y_tr_l2_sm)

                        y_pred = np.zeros(len(y_te), dtype=int)
                        mask_nm_pred = pred_l1 == 1
                        if mask_nm_pred.any():
                            p_l2 = m_l2.predict(X_l2.iloc[te_i][mask_nm_pred])
                            y_pred[mask_nm_pred] = np.where(p_l2 == 0, 1, 2)
                    else:
                        y_pred = np.where(pred_l1 == 0, 0, 1)

                fold_accs.append(accuracy_score(y_te, y_pred))

            cv_results[var_name][modelo_nome] = {
                "mean": np.mean(fold_accs),
                "std": np.std(fold_accs),
                "folds": fold_accs,
            }

    # Print CV results
    cv_rows = []
    for modelo_nome in model_names:
        row = {"modelo": modelo_nome}
        for var_name in cv_variantes:
            r = cv_results[var_name][modelo_nome]
            prefix = var_name.split("_")[0]
            row[f"{prefix}_cv_mean"] = r["mean"]
            row[f"{prefix}_cv_std"] = r["std"]
        cv_rows.append(row)
        v1_cv = cv_results["V1_direto_stat"][modelo_nome]["mean"]
        v4_cv = cv_results["V4_hier_llm_n2"][modelo_nome]["mean"]
        delta_cv = v4_cv - v1_cv
        sym = "✅" if delta_cv > 0 else "❌"
        print(f"  {sym} {modelo_nome:20s}: V1={v1_cv:.3f} V4={v4_cv:.3f} Δ={delta_cv:+.3f}")

    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(os.path.join(HIER_DIR, "cv_logo_variantes.csv"), index=False)


# ==============================================================================
# GRÁFICOS
# ==============================================================================

# 1. Barras: accuracy por variante (melhor modelo de cada)
fig, ax = plt.subplots(figsize=(14, 6))
var_names = list(VARIANTES.keys())
x = np.arange(len(var_names))
bar_colors = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c", "#9b59b6", "#95a5a6"]

best_accs = []
best_models = []
for v in var_names:
    best_m = max(all_results[v].items(), key=lambda x: x[1]["accuracy"])
    best_accs.append(best_m[1]["accuracy"])
    best_models.append(best_m[0])

bars = ax.bar(x, best_accs, color=bar_colors)
ax.set_ylabel("Acurácia (melhor modelo)")
ax.set_title(f"6 Variantes — {DATA_TYPE.upper()} (LLM: {LLM_MODEL})")
ax.set_xticks(x)
ax.set_xticklabels([f"{v}\n({m})" for v, m in zip(var_names, best_models)],
                    rotation=45, ha="right", fontsize=7)
ax.set_ylim([0, 1.05])
ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.3, label="Acaso (33.3%)")
for bar, acc in zip(bars, best_accs):
    ax.text(bar.get_x() + bar.get_width()/2., acc + 0.01,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HIER_DIR, "variantes_accuracy.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Grouped bars: V1 vs V2 vs V4 para todos os modelos
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
width = 0.25
x = np.arange(len(model_names))

# Accuracy
ax1 = axes[0]
for i, (var, color, label) in enumerate([
    ("V1_direto_stat", "#3498db", "V1: Direto"),
    ("V2_hier_stat", "#2ecc71", "V2: Hier puro"),
    ("V4_hier_llm_n2", "#e74c3c", "V4: Hier+LLM N2"),
]):
    accs = [all_results[var][m]["accuracy"] for m in model_names]
    bars = ax1.bar(x + i*width, accs, width, label=label, color=color)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.005,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=6, rotation=90)
ax1.set_ylabel("Acurácia")
ax1.set_title("Accuracy: V1 vs V2 vs V4")
ax1.set_xticks(x + width)
ax1.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
ax1.legend(fontsize=8)
ax1.set_ylim([0, 1.05])
ax1.axhline(y=0.333, color="gray", linestyle="--", alpha=0.3)

# Recall MNAR
ax2 = axes[1]
for i, (var, color, label) in enumerate([
    ("V1_direto_stat", "#3498db", "V1: Direto"),
    ("V2_hier_stat", "#2ecc71", "V2: Hier puro"),
    ("V4_hier_llm_n2", "#e74c3c", "V4: Hier+LLM N2"),
]):
    recalls = [all_results[var][m]["recall_MNAR"] for m in model_names]
    bars = ax2.bar(x + i*width, recalls, width, label=label, color=color)
    for bar, rec in zip(bars, recalls):
        ax2.text(bar.get_x() + bar.get_width()/2., rec + 0.005,
                f'{rec:.1%}', ha='center', va='bottom', fontsize=6, rotation=90)
ax2.set_ylabel("Recall MNAR")
ax2.set_title("Recall MNAR: V1 vs V2 vs V4")
ax2.set_xticks(x + width)
ax2.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
ax2.legend(fontsize=8)
ax2.set_ylim([0, 1.05])

plt.suptitle(f"Hierárquica: V1 vs V2 vs V4 — {DATA_TYPE.upper()}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(HIER_DIR, "v1_vs_v2_vs_v4.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3. Confusion matrix do melhor V4
best_v4_model = max(all_results["V4_hier_llm_n2"].items(),
                     key=lambda x: x[1]["accuracy"])[0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = ["MCAR", "MAR", "MNAR"]

for ax, (title, var) in zip(axes, [
    ("V1: Direto", "V1_direto_stat"),
    ("V2: Hier puro", "V2_hier_stat"),
    ("V4: Hier+LLM N2", "V4_hier_llm_n2"),
]):
    res = all_results[var][best_v4_model]
    cm = res["confusion"]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{title} — {best_v4_model}\nAcc={res['accuracy']:.1%}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    thresh = cm.max() / 2.
    for i in range(3):
        for j in range(3):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

plt.suptitle(f"Confusion Matrices — {DATA_TYPE.upper()} ({best_v4_model})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(HIER_DIR, "confusion_v1_v2_v4.png"), dpi=300, bbox_inches="tight")
plt.close()

# 4. Heatmap: Todas variantes x modelos
fig, ax = plt.subplots(figsize=(12, 6))
heat_data = np.zeros((len(var_names), len(model_names)))
for i, v in enumerate(var_names):
    for j, m in enumerate(model_names):
        heat_data[i, j] = all_results[v][m]["accuracy"]

im = ax.imshow(heat_data, cmap="RdYlGn", aspect="auto", vmin=0.2, vmax=0.9)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(var_names)))
ax.set_yticklabels(var_names, fontsize=8)
ax.set_title(f"Accuracy Heatmap — {DATA_TYPE.upper()}")
plt.colorbar(im, ax=ax, label="Accuracy")
for i in range(len(var_names)):
    for j in range(len(model_names)):
        ax.text(j, i, f"{heat_data[i, j]:.1%}",
                ha="center", va="center", fontsize=7,
                color="white" if heat_data[i, j] < 0.5 else "black")
plt.tight_layout()
plt.savefig(os.path.join(HIER_DIR, "heatmap_variantes.png"), dpi=300, bbox_inches="tight")
plt.close()


# ==============================================================================
# TRAINING SUMMARY
# ==============================================================================
summary = {
    "timestamp": datetime.now().isoformat(),
    "data_type": DATA_TYPE,
    "experiment": EXPERIMENT,
    "llm_model": LLM_MODEL,
    "n_samples": int(len(y)),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_features_baseline": len(STAT_COLS),
    "n_features_caafe": len(CAAFE_COLS),
    "n_features_llm": len(LLM_COLS),
    "n_features_total": len(ALL_COLS),
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
        "mean_accuracy": float(np.mean([all_results[var_name][m]["accuracy"]
                                         for m in model_names])),
    }

with open(os.path.join(HIER_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


# ==============================================================================
# RESUMO FINAL
# ==============================================================================
print(f"\n{'='*70}")
print(f"✅ CLASSIFICAÇÃO HIERÁRQUICA — 6 VARIANTES CONCLUÍDA!")
print(f"{'='*70}")

print(f"\n🏆 Ranking por accuracy máxima:")
ranking = sorted(
    [(v, max(all_results[v].items(), key=lambda x: x[1]["accuracy"]))
     for v in VARIANTES],
    key=lambda x: x[1][1]["accuracy"], reverse=True)

for i, (var, (modelo, metrics)) in enumerate(ranking, 1):
    print(f"   {i}. {var:25s} {metrics['accuracy']:.3f} ({modelo}) "
          f"MNAR_recall={metrics['recall_MNAR']:.3f}")

# Verifica tese: V4 > V2 > V1 e V4 > V5?
v1_best = max(r["accuracy"] for r in all_results["V1_direto_stat"].values())
v2_best = max(r["accuracy"] for r in all_results["V2_hier_stat"].values())
v4_best = max(r["accuracy"] for r in all_results["V4_hier_llm_n2"].values())
v5_best = max(r["accuracy"] for r in all_results["V5_hier_llm_ambos"].values())

print(f"\n📊 Verificação da tese:")
print(f"   V4 > V1 (hier+LLM > direto)?      {'✅ SIM' if v4_best > v1_best else '❌ NÃO'} ({v4_best:.3f} vs {v1_best:.3f})")
print(f"   V4 > V2 (LLM ajuda no N2)?         {'✅ SIM' if v4_best > v2_best else '❌ NÃO'} ({v4_best:.3f} vs {v2_best:.3f})")
print(f"   V4 > V5 (LLM só N2 > em ambos)?    {'✅ SIM' if v4_best > v5_best else '❌ NÃO'} ({v4_best:.3f} vs {v5_best:.3f})")
print(f"   V2 > V1 (hierárquica já melhora)?   {'✅ SIM' if v2_best > v1_best else '❌ NÃO'} ({v2_best:.3f} vs {v1_best:.3f})")

print(f"\n💾 Salvos em: {HIER_DIR}")
for f_name in ["todas_variantes.csv", "resumo_variantes.csv",
               "comparacao_por_modelo.csv", "significancia_mcnemar.csv",
               "variantes_accuracy.png", "v1_vs_v2_vs_v4.png",
               "confusion_v1_v2_v4.png", "heatmap_variantes.png",
               "training_summary.json"]:
    print(f"   - {f_name}")
print(f"{'='*70}")
