"""
Classificação Hierárquica: MCAR vs NAO-MCAR → MAR vs MNAR.

Nível 1: Classificador binário MCAR vs {MAR, MNAR}
Nível 2: Para amostras classificadas como NAO-MCAR, classifica MAR vs MNAR

Compara automaticamente com a classificação direta de 3 classes.

Uso:
    python train_hierarchical.py --model none --data sintetico --experiment step05
    python train_hierarchical.py --model none --data real --experiment step05
"""
import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import (
    GroupShuffleSplit, GroupKFold, LeaveOneGroupOut,
    cross_val_predict, RepeatedStratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.args import parse_common_args
from utils.paths import get_output_dir, get_comparison_dir

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURAÇÃO
# ======================================================
MODEL_NAME, DATA_TYPE, _, EXPERIMENT = parse_common_args()

OUTPUT_DIR = get_output_dir(DATA_TYPE, MODEL_NAME, EXPERIMENT)
HIER_DIR = os.path.join(get_comparison_dir(DATA_TYPE, EXPERIMENT), "hierarquico")
os.makedirs(HIER_DIR, exist_ok=True)

X_IN = os.path.join(OUTPUT_DIR, "X_features.csv")
Y_IN = os.path.join(OUTPUT_DIR, "y_labels.csv")
GROUPS_IN = os.path.join(OUTPUT_DIR, "groups.csv")

if not os.path.exists(X_IN):
    print(f"❌ Arquivo não encontrado: {X_IN}")
    print(f"   Execute primeiro: python extract_features.py --model {MODEL_NAME} --data {DATA_TYPE}")
    sys.exit(1)

ABORDAGEM = "apenas ML (baseline)" if MODEL_NAME == "none" else f"ML + LLM ({MODEL_NAME})"

print("=" * 60)
print("🔀 CLASSIFICAÇÃO HIERÁRQUICA")
print("=" * 60)
print(f"📊 Dados: {DATA_TYPE}")
print(f"🔬 Abordagem: {ABORDAGEM}")
print(f"📂 Input: {OUTPUT_DIR}")
print(f"📂 Output: {HIER_DIR}")
print("=" * 60)

# ======================================================
# CARREGA DADOS
# ======================================================
X = pd.read_csv(X_IN)
y = pd.read_csv(Y_IN).squeeze("columns")

groups = None
if os.path.exists(GROUPS_IN):
    groups = pd.read_csv(GROUPS_IN).squeeze("columns")

CLASS_NAMES = {0: "MCAR", 1: "MAR", 2: "MNAR"}
print(f"\n📊 Dados: X={X.shape}, y={y.shape}")
print(f"   Distribuição: {dict(y.value_counts().sort_index())}")

# ======================================================
# LABELS HIERÁRQUICOS
# ======================================================
# Nível 1: MCAR (0) vs NAO-MCAR (1)
y_level1 = (y != 0).astype(int)  # 0=MCAR, 1=NAO-MCAR

# Nível 2: MAR (1) vs MNAR (2) — só para amostras NAO-MCAR
mask_not_mcar = y != 0
X_level2 = X[mask_not_mcar].reset_index(drop=True)
y_level2 = y[mask_not_mcar].reset_index(drop=True)
# Remap: MAR(1)->0, MNAR(2)->1 para classificação binária
y_level2_binary = (y_level2 == 2).astype(int)  # 0=MAR, 1=MNAR
groups_level2 = groups[mask_not_mcar].reset_index(drop=True) if groups is not None else None

print(f"\n🔀 Nível 1: MCAR ({(y_level1==0).sum()}) vs NAO-MCAR ({(y_level1==1).sum()})")
print(f"🔀 Nível 2: MAR ({(y_level2_binary==0).sum()}) vs MNAR ({(y_level2_binary==1).sum()})")


# ======================================================
# MODELOS
# ======================================================
def get_modelos(n_samples: int) -> dict:
    """Modelos com hiperparâmetros adaptados."""
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


# ======================================================
# SPLIT
# ======================================================
if groups is not None and groups.nunique() > 1:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
else:
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, stratify=y, random_state=42)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
y_train_l1 = y_level1.iloc[train_idx]
y_test_l1 = y_level1.iloc[test_idx]

# Nível 2: filtra treino e teste para NAO-MCAR
mask_train_notmcar = y_train != 0
mask_test_notmcar = y_test != 0

X_train_l2 = X_train[mask_train_notmcar]
y_train_l2 = (y_train[mask_train_notmcar] == 2).astype(int)
X_test_l2 = X_test[mask_test_notmcar]
y_test_l2 = (y_test[mask_test_notmcar] == 2).astype(int)

# SMOTE para balanceamento
try:
    from imblearn.over_sampling import SMOTE
    for label, Xt, yt in [("L1", X_train, y_train_l1), ("L2", X_train_l2, y_train_l2)]:
        min_count = yt.value_counts().min()
        if min_count >= 2:
            k = min(3, min_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k)
            if label == "L1":
                X_train_l1_sm, y_train_l1_sm = smote.fit_resample(Xt, yt)
            else:
                X_train_l2_sm, y_train_l2_sm = smote.fit_resample(Xt, yt)
        else:
            if label == "L1":
                X_train_l1_sm, y_train_l1_sm = Xt, yt
            else:
                X_train_l2_sm, y_train_l2_sm = Xt, yt
except ImportError:
    X_train_l1_sm, y_train_l1_sm = X_train, y_train_l1
    X_train_l2_sm, y_train_l2_sm = X_train_l2, y_train_l2

print(f"\n📈 Split: train={len(y_train)}, test={len(y_test)}")
print(f"   L1 treino: MCAR={int((y_train_l1_sm==0).sum())}, NAO-MCAR={int((y_train_l1_sm==1).sum())}")
print(f"   L2 treino: MAR={int((y_train_l2_sm==0).sum())}, MNAR={int((y_train_l2_sm==1).sum())}")

# ======================================================
# CLASSIFICAÇÃO DIRETA (3 classes) — baseline de comparação
# ======================================================
print(f"\n🏋️ Treinando classificação DIRETA (3 classes)...")

# SMOTE para direta
try:
    from imblearn.over_sampling import SMOTE
    min_count = y_train.value_counts().min()
    if min_count >= 2:
        k = min(3, min_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train_direct_sm, y_train_direct_sm = smote.fit_resample(X_train, y_train)
    else:
        X_train_direct_sm, y_train_direct_sm = X_train, y_train
except ImportError:
    X_train_direct_sm, y_train_direct_sm = X_train, y_train

modelos_direct = get_modelos(len(X_train_direct_sm))
resultados_direct = {}
for nome, modelo in tqdm(modelos_direct.items(), desc="Direta"):
    modelo.fit(X_train_direct_sm, y_train_direct_sm)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    resultados_direct[nome] = {
        "accuracy": acc,
        "report": report,
        "confusion": cm,
        "y_pred": y_pred,
    }

# ======================================================
# CLASSIFICAÇÃO HIERÁRQUICA
# ======================================================
print(f"\n🏋️ Treinando classificação HIERÁRQUICA...")
resultados_hier = {}

for nome in tqdm(get_modelos(len(X_train_l1_sm)), desc="Hierárquica"):
    # Nível 1: MCAR vs NAO-MCAR
    modelo_l1 = get_modelos(len(X_train_l1_sm))[nome]
    modelo_l1.fit(X_train_l1_sm, y_train_l1_sm)
    pred_l1 = modelo_l1.predict(X_test)

    # Nível 2: MAR vs MNAR (apenas para preditos como NAO-MCAR)
    modelo_l2 = get_modelos(len(X_train_l2_sm))[nome]
    modelo_l2.fit(X_train_l2_sm, y_train_l2_sm)

    # Combina predições
    y_pred_hier = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        if pred_l1[i] == 0:
            y_pred_hier[i] = 0  # MCAR
        else:
            # Classifica MAR vs MNAR
            pred_l2_i = modelo_l2.predict(X_test.iloc[[i]])
            y_pred_hier[i] = 1 if pred_l2_i[0] == 0 else 2  # MAR ou MNAR

    acc = accuracy_score(y_test, y_pred_hier)
    report = classification_report(y_test, y_pred_hier, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_hier, labels=[0, 1, 2])

    # Métricas do nível 1
    acc_l1 = accuracy_score(y_test_l1, pred_l1)

    # Métricas do nível 2 (apenas amostras corretamente classificadas como NAO-MCAR)
    mask_pred_notmcar = pred_l1 == 1
    if mask_pred_notmcar.sum() > 0:
        # Aplica nível 2 a todas as preditas como NAO-MCAR
        X_test_pred_notmcar = X_test[mask_pred_notmcar]
        pred_l2_all = modelo_l2.predict(X_test_pred_notmcar)
    else:
        pred_l2_all = np.array([])

    resultados_hier[nome] = {
        "accuracy": acc,
        "report": report,
        "confusion": cm,
        "y_pred": y_pred_hier,
        "acc_level1": acc_l1,
        "n_pred_mcar": int((pred_l1 == 0).sum()),
        "n_pred_notmcar": int((pred_l1 == 1).sum()),
    }


# ======================================================
# CROSS-VALIDATION: LOGO (Leave-One-Group-Out)
# ======================================================
print(f"\n📊 Cross-Validation: LOGO + GroupKFold...")

cv_results = {"logo": {}, "groupkfold": {}}

if groups is not None and groups.nunique() > 2:
    n_groups_total = groups.nunique()

    # LOGO
    logo = LeaveOneGroupOut()
    n_logo_folds = logo.get_n_splits(X, y, groups)
    print(f"   LOGO: {n_logo_folds} folds (1 grupo por fold)")

    # GroupKFold
    n_gkf_folds = min(n_groups_total, 5)
    gkf = GroupKFold(n_splits=n_gkf_folds)
    print(f"   GroupKFold: {n_gkf_folds} folds")

    for nome in tqdm(get_modelos(len(X)), desc="CV (LOGO+GKF)"):
        # LOGO CV — classificação hierárquica
        y_pred_logo = np.full(len(y), -1)
        for train_i, test_i in logo.split(X, y, groups):
            X_tr, X_te = X.iloc[train_i], X.iloc[test_i]
            y_tr, y_te = y.iloc[train_i], y.iloc[test_i]

            # Nível 1
            y_tr_l1 = (y_tr != 0).astype(int)
            m_l1 = get_modelos(len(X_tr))[nome]
            m_l1.fit(X_tr, y_tr_l1)
            pred_l1 = m_l1.predict(X_te)

            # Nível 2
            mask_tr_nm = y_tr != 0
            X_tr_l2 = X_tr[mask_tr_nm]
            y_tr_l2 = (y_tr[mask_tr_nm] == 2).astype(int)

            if len(y_tr_l2) >= 2 and y_tr_l2.nunique() >= 2:
                m_l2 = get_modelos(len(X_tr_l2))[nome]
                m_l2.fit(X_tr_l2, y_tr_l2)

                for j, idx in enumerate(test_i):
                    if pred_l1[j] == 0:
                        y_pred_logo[idx] = 0
                    else:
                        p = m_l2.predict(X_te.iloc[[j]])
                        y_pred_logo[idx] = 1 if p[0] == 0 else 2
            else:
                for j, idx in enumerate(test_i):
                    y_pred_logo[idx] = 0 if pred_l1[j] == 0 else 1

        valid = y_pred_logo >= 0
        if valid.sum() > 0:
            acc_logo = accuracy_score(y[valid], y_pred_logo[valid])
        else:
            acc_logo = 0.0

        # GroupKFold CV — classificação hierárquica
        y_pred_gkf = np.full(len(y), -1)
        for train_i, test_i in gkf.split(X, y, groups):
            X_tr, X_te = X.iloc[train_i], X.iloc[test_i]
            y_tr = y.iloc[train_i]

            y_tr_l1 = (y_tr != 0).astype(int)
            m_l1 = get_modelos(len(X_tr))[nome]
            m_l1.fit(X_tr, y_tr_l1)
            pred_l1 = m_l1.predict(X_te)

            mask_tr_nm = y_tr != 0
            X_tr_l2 = X_tr[mask_tr_nm]
            y_tr_l2 = (y_tr[mask_tr_nm] == 2).astype(int)

            if len(y_tr_l2) >= 2 and y_tr_l2.nunique() >= 2:
                m_l2 = get_modelos(len(X_tr_l2))[nome]
                m_l2.fit(X_tr_l2, y_tr_l2)

                for j, idx in enumerate(test_i):
                    if pred_l1[j] == 0:
                        y_pred_gkf[idx] = 0
                    else:
                        p = m_l2.predict(X_te.iloc[[j]])
                        y_pred_gkf[idx] = 1 if p[0] == 0 else 2
            else:
                for j, idx in enumerate(test_i):
                    y_pred_gkf[idx] = 0 if pred_l1[j] == 0 else 1

        valid_gkf = y_pred_gkf >= 0
        if valid_gkf.sum() > 0:
            acc_gkf = accuracy_score(y[valid_gkf], y_pred_gkf[valid_gkf])
        else:
            acc_gkf = 0.0

        cv_results["logo"][nome] = acc_logo
        cv_results["groupkfold"][nome] = acc_gkf

    print(f"\n📊 CV LOGO vs GroupKFold (hierárquico):")
    for nome in cv_results["logo"]:
        logo_acc = cv_results["logo"][nome]
        gkf_acc = cv_results["groupkfold"][nome]
        delta = logo_acc - gkf_acc
        print(f"   {nome:20s}: LOGO={logo_acc:.4f} | GKF={gkf_acc:.4f} | Δ={delta:+.4f}")
else:
    print("   ⚠️ Sem grupos — LOGO não aplicável")


# ======================================================
# COMPARAÇÃO: HIERÁRQUICA vs DIRETA
# ======================================================
print(f"\n{'='*60}")
print(f"📊 COMPARAÇÃO: HIERÁRQUICA vs DIRETA")
print(f"{'='*60}")

comparison_rows = []
for nome in resultados_direct:
    d = resultados_direct[nome]
    h = resultados_hier[nome]

    delta_acc = h["accuracy"] - d["accuracy"]

    # Recall por classe
    recall_d = {CLASS_NAMES[i]: d["report"].get(str(i), {}).get("recall", 0) for i in range(3)}
    recall_h = {CLASS_NAMES[i]: h["report"].get(str(i), {}).get("recall", 0) for i in range(3)}

    row = {
        "modelo": nome,
        "acc_direta": d["accuracy"],
        "acc_hierarquica": h["accuracy"],
        "delta_acc": delta_acc,
        "recall_MCAR_direta": recall_d["MCAR"],
        "recall_MAR_direta": recall_d["MAR"],
        "recall_MNAR_direta": recall_d["MNAR"],
        "recall_MCAR_hier": recall_h["MCAR"],
        "recall_MAR_hier": recall_h["MAR"],
        "recall_MNAR_hier": recall_h["MNAR"],
        "delta_recall_MNAR": recall_h["MNAR"] - recall_d["MNAR"],
        "acc_level1": h["acc_level1"],
    }
    if nome in cv_results["logo"]:
        row["cv_logo"] = cv_results["logo"][nome]
        row["cv_groupkfold"] = cv_results["groupkfold"][nome]

    comparison_rows.append(row)

    symbol = "✅" if delta_acc > 0 else "❌" if delta_acc < 0 else "➖"
    print(f"   {symbol} {nome:20s}: Direta={d['accuracy']:.3f} | Hier={h['accuracy']:.3f} | Δ={delta_acc:+.3f}")
    print(f"      Recall MNAR: Direta={recall_d['MNAR']:.3f} → Hier={recall_h['MNAR']:.3f}")

df_comparison = pd.DataFrame(comparison_rows)
df_comparison.to_csv(os.path.join(HIER_DIR, "comparacao_hier_vs_direta.csv"), index=False)


# ======================================================
# CONFUSION MATRICES DETALHADAS
# ======================================================
conf_data = {}
for nome in resultados_direct:
    conf_data[nome] = {
        "direta": resultados_direct[nome]["confusion"].tolist(),
        "hierarquica": resultados_hier[nome]["confusion"].tolist(),
    }

with open(os.path.join(HIER_DIR, "confusion_matrices.json"), "w") as f:
    json.dump(conf_data, f, indent=2)


# ======================================================
# CV RESULTS
# ======================================================
if cv_results["logo"]:
    cv_df = pd.DataFrame({
        "modelo": list(cv_results["logo"].keys()),
        "cv_logo_acc": list(cv_results["logo"].values()),
        "cv_groupkfold_acc": list(cv_results["groupkfold"].values()),
    })
    cv_df["delta"] = cv_df["cv_logo_acc"] - cv_df["cv_groupkfold_acc"]
    cv_df.to_csv(os.path.join(HIER_DIR, "cv_logo_vs_groupkfold.csv"), index=False)


# ======================================================
# GRÁFICOS
# ======================================================
# 1. Barras: Direta vs Hierárquica por modelo
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

nomes = list(resultados_direct.keys())
x = np.arange(len(nomes))
width = 0.35

# Accuracy
ax1 = axes[0]
acc_d = [resultados_direct[n]["accuracy"] for n in nomes]
acc_h = [resultados_hier[n]["accuracy"] for n in nomes]
bars1 = ax1.bar(x - width/2, acc_d, width, label="Direta (3 classes)", color="#3498db")
bars2 = ax1.bar(x + width/2, acc_h, width, label="Hierárquica (2 níveis)", color="#e74c3c")
ax1.set_ylabel("Acurácia")
ax1.set_title("Accuracy: Direta vs Hierárquica")
ax1.set_xticks(x)
ax1.set_xticklabels(nomes, rotation=45, ha="right", fontsize=8)
ax1.legend(fontsize=8)
ax1.set_ylim([0, 1.05])
ax1.axhline(y=0.333, color="gray", linestyle="--", alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                 f'{h:.1%}', ha='center', va='bottom', fontsize=7)

# Recall MNAR
ax2 = axes[1]
recall_d = [resultados_direct[n]["report"].get("2", {}).get("recall", 0) for n in nomes]
recall_h = [resultados_hier[n]["report"].get("2", {}).get("recall", 0) for n in nomes]
bars3 = ax2.bar(x - width/2, recall_d, width, label="Direta", color="#3498db")
bars4 = ax2.bar(x + width/2, recall_h, width, label="Hierárquica", color="#e74c3c")
ax2.set_ylabel("Recall MNAR")
ax2.set_title("Recall MNAR: Direta vs Hierárquica")
ax2.set_xticks(x)
ax2.set_xticklabels(nomes, rotation=45, ha="right", fontsize=8)
ax2.legend(fontsize=8)
ax2.set_ylim([0, 1.05])
for bars in [bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                 f'{h:.1%}', ha='center', va='bottom', fontsize=7)

plt.suptitle(f"Classificação Hierárquica vs Direta — {DATA_TYPE.upper()} ({ABORDAGEM})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(HIER_DIR, "hierarquica_vs_direta.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Confusion matrices do melhor modelo (antes vs depois)
best_model = max(resultados_hier.items(), key=lambda x: x[1]["accuracy"])[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ["MCAR", "MAR", "MNAR"]

for ax, (title, res) in zip(axes, [
    ("Direta", resultados_direct[best_model]),
    ("Hierárquica", resultados_hier[best_model])
]):
    cm = res["confusion"]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{title} — {best_model}\nAcc={res['accuracy']:.1%}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

    # Valores na matriz
    thresh = cm.max() / 2.
    for i in range(3):
        for j in range(3):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

plt.suptitle(f"Confusion Matrix — {DATA_TYPE.upper()}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(HIER_DIR, "confusion_matrix_comparacao.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3. LOGO vs GroupKFold (se disponível)
if cv_results["logo"]:
    fig, ax = plt.subplots(figsize=(12, 6))
    nomes_cv = list(cv_results["logo"].keys())
    x_cv = np.arange(len(nomes_cv))

    logo_vals = [cv_results["logo"][n] for n in nomes_cv]
    gkf_vals = [cv_results["groupkfold"][n] for n in nomes_cv]

    bars1 = ax.bar(x_cv - width/2, logo_vals, width, label="LOGO", color="#2ecc71")
    bars2 = ax.bar(x_cv + width/2, gkf_vals, width, label="GroupKFold", color="#9b59b6")
    ax.set_ylabel("Acurácia")
    ax.set_title(f"LOGO vs GroupKFold (Hierárquico) — {DATA_TYPE.upper()}")
    ax.set_xticks(x_cv)
    ax.set_xticklabels(nomes_cv, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.1%}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(HIER_DIR, "logo_vs_groupkfold.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ======================================================
# TRAINING SUMMARY
# ======================================================
summary = {
    "timestamp": datetime.now().isoformat(),
    "model_name": MODEL_NAME,
    "data_type": DATA_TYPE,
    "experiment": EXPERIMENT,
    "abordagem": ABORDAGEM,
    "n_samples": int(len(X)),
    "n_features": int(X.shape[1]),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "seed": 42,
    "best_model_direct": max(resultados_direct.items(), key=lambda x: x[1]["accuracy"])[0],
    "best_acc_direct": float(max(r["accuracy"] for r in resultados_direct.values())),
    "best_model_hier": max(resultados_hier.items(), key=lambda x: x[1]["accuracy"])[0],
    "best_acc_hier": float(max(r["accuracy"] for r in resultados_hier.values())),
    "results_per_model": {},
}

for nome in resultados_direct:
    summary["results_per_model"][nome] = {
        "acc_direta": float(resultados_direct[nome]["accuracy"]),
        "acc_hierarquica": float(resultados_hier[nome]["accuracy"]),
        "recall_MNAR_direta": float(resultados_direct[nome]["report"].get("2", {}).get("recall", 0)),
        "recall_MNAR_hier": float(resultados_hier[nome]["report"].get("2", {}).get("recall", 0)),
    }

if cv_results["logo"]:
    summary["cv_logo"] = {k: float(v) for k, v in cv_results["logo"].items()}
    summary["cv_groupkfold"] = {k: float(v) for k, v in cv_results["groupkfold"].items()}

with open(os.path.join(HIER_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


# ======================================================
# RESUMO FINAL
# ======================================================
print(f"\n{'='*60}")
print(f"✅ CLASSIFICAÇÃO HIERÁRQUICA CONCLUÍDA!")
print(f"{'='*60}")

best_d = max(resultados_direct.items(), key=lambda x: x[1]["accuracy"])
best_h = max(resultados_hier.items(), key=lambda x: x[1]["accuracy"])

print(f"\n🏆 Melhor DIRETA:      {best_d[0]} ({best_d[1]['accuracy']:.4f})")
print(f"🏆 Melhor HIERÁRQUICA: {best_h[0]} ({best_h[1]['accuracy']:.4f})")

delta = best_h[1]["accuracy"] - best_d[1]["accuracy"]
if delta > 0:
    print(f"   ✅ Hierárquica melhor em {delta:+.4f}")
elif delta < 0:
    print(f"   ❌ Direta melhor em {abs(delta):.4f}")
else:
    print(f"   ➖ Empate")

# Recall MNAR
recall_mnar_d = best_d[1]["report"].get("2", {}).get("recall", 0)
recall_mnar_h = best_h[1]["report"].get("2", {}).get("recall", 0)
print(f"\n📊 Recall MNAR (melhor modelo):")
print(f"   Direta:      {recall_mnar_d:.4f}")
print(f"   Hierárquica: {recall_mnar_h:.4f}")

print(f"\n💾 Salvos em: {HIER_DIR}")
print(f"   - comparacao_hier_vs_direta.csv")
print(f"   - confusion_matrices.json")
print(f"   - hierarquica_vs_direta.png")
print(f"   - confusion_matrix_comparacao.png")
if cv_results["logo"]:
    print(f"   - cv_logo_vs_groupkfold.csv")
    print(f"   - logo_vs_groupkfold.png")
print(f"   - training_summary.json")
print(f"{'='*60}")
