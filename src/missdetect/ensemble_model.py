"""
Ensemble adaptativo: baseline + LLM como segundo estágio.

Estratégia:
1. Treina modelo baseline com features estatísticas (10 features)
2. Para amostras de teste com probabilidade máxima < THRESHOLD, usa modelo LLM (18 features)
3. Combina predições: alta confiança usa baseline, baixa confiança usa LLM

Uso:
    python ensemble_model.py --data real
    python ensemble_model.py --data sintetico
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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

CONFIDENCE_THRESHOLD = 0.6

_, DATA_TYPE, _, EXPERIMENT = parse_common_args()

# Diretórios de entrada
BASELINE_DIR = get_output_dir(DATA_TYPE, "none", EXPERIMENT)
LLM_MODEL = "gemini-3-flash-preview"
LLM_DIR = get_output_dir(DATA_TYPE, LLM_MODEL, EXPERIMENT)
ENSEMBLE_DIR = os.path.join(get_comparison_dir(DATA_TYPE, EXPERIMENT), "ensemble")
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# Verifica existência dos dados
for label, path in [("Baseline", BASELINE_DIR), ("LLM", LLM_DIR)]:
    x_path = os.path.join(path, "X_features.csv")
    if not os.path.exists(x_path):
        print(f"❌ Features {label} não encontradas: {x_path}")
        print(f"   Execute primeiro o pipeline com --data {DATA_TYPE}")
        sys.exit(1)

print("=" * 60)
print("🔀 ENSEMBLE ADAPTATIVO: BASELINE + LLM")
print("=" * 60)
print(f"📊 Dados: {DATA_TYPE}")
print(f"🎯 Threshold de confiança: {CONFIDENCE_THRESHOLD}")
print(f"📂 Output: {ENSEMBLE_DIR}")
print("=" * 60)

# ======================================================
# CARREGA DADOS
# ======================================================
X_baseline = pd.read_csv(os.path.join(BASELINE_DIR, "X_features.csv"))
y_baseline = pd.read_csv(os.path.join(BASELINE_DIR, "y_labels.csv")).squeeze("columns")

X_llm = pd.read_csv(os.path.join(LLM_DIR, "X_features.csv"))
y_llm = pd.read_csv(os.path.join(LLM_DIR, "y_labels.csv")).squeeze("columns")

assert len(y_baseline) == len(y_llm), "Baseline e LLM devem ter o mesmo número de amostras"
assert (y_baseline.values == y_llm.values).all(), "Labels devem ser idênticos"

# Carrega grupos para GroupShuffleSplit
groups_bl_path = os.path.join(BASELINE_DIR, "groups.csv")
groups_llm_path = os.path.join(LLM_DIR, "groups.csv")
groups = None
if os.path.exists(groups_bl_path):
    groups = pd.read_csv(groups_bl_path).squeeze("columns")

n_samples = len(y_baseline)
print(f"\n📊 Amostras: {n_samples}")
print(f"   Baseline features: {X_baseline.shape[1]}")
print(f"   LLM features: {X_llm.shape[1]}")
if groups is not None:
    print(f"   Grupos: {groups.nunique()}")

# ======================================================
# SPLIT (GroupShuffleSplit se disponível)
# ======================================================
if groups is not None and groups.nunique() > 1:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X_baseline, y_baseline, groups))
    X_train_bl, X_test_bl = X_baseline.iloc[train_idx], X_baseline.iloc[test_idx]
    X_train_llm, X_test_llm = X_llm.iloc[train_idx], X_llm.iloc[test_idx]
    y_train, y_test = y_baseline.iloc[train_idx], y_baseline.iloc[test_idx]
    print(f"📈 Split (GroupShuffleSplit): train={len(y_train)}, test={len(y_test)}")
    print(f"   Grupos treino: {sorted(groups.iloc[train_idx].unique())}")
    print(f"   Grupos teste:  {sorted(groups.iloc[test_idx].unique())}")
else:
    X_train_bl, X_test_bl, y_train, y_test = train_test_split(
        X_baseline, y_baseline, test_size=0.25, stratify=y_baseline, random_state=42
    )
    X_train_llm, X_test_llm, _, _ = train_test_split(
        X_llm, y_llm, test_size=0.25, stratify=y_llm, random_state=42
    )
    print(f"📈 Split: train={len(y_train)}, test={len(y_test)}")

# ======================================================
# MODELOS BASE
# ======================================================
def make_models():
    """Modelos com predict_proba para ensemble."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, random_state=42),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=3, random_state=42, probability=True))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                  max_iter=2000, random_state=42))
        ]),
        "NaiveBayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
    }

# ======================================================
# ENSEMBLE
# ======================================================
resultados = {}
relatorio_lines = []
all_decisions = []

print(f"\n🏋️ Treinando ensemble...")

for nome in make_models():
    # Treina modelo baseline
    modelo_bl = make_models()[nome]
    modelo_bl.fit(X_train_bl, y_train)

    # Treina modelo LLM com PCA para modelos sensíveis
    modelo_llm_def = make_models()[nome]
    if nome in ("SVM_RBF", "KNN", "MLP"):
        # Adiciona PCA para modelos sensíveis a dimensionalidade
        modelo_llm = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("clf", modelo_llm_def if not isinstance(modelo_llm_def, Pipeline) else modelo_llm_def.named_steps["clf"])
        ])
    else:
        modelo_llm = modelo_llm_def
    modelo_llm.fit(X_train_llm, y_train)

    # Predições do baseline com probabilidade
    proba_bl = modelo_bl.predict_proba(X_test_bl)
    pred_bl = modelo_bl.predict(X_test_bl)
    max_conf = proba_bl.max(axis=1)

    # Predições do LLM
    pred_llm = modelo_llm.predict(X_test_llm)

    # Ensemble: usa LLM apenas para amostras de baixa confiança
    pred_ensemble = pred_bl.copy()
    low_conf_mask = max_conf < CONFIDENCE_THRESHOLD
    pred_ensemble[low_conf_mask] = pred_llm[low_conf_mask]

    # Acumula decisões por amostra
    for i in range(len(y_test)):
        row = {
            "sample_idx": int(X_test_bl.index[i]),
            "y_true": int(y_test.iloc[i]),
            "modelo": nome,
            "pred_baseline": int(pred_bl[i]),
            "conf_baseline": float(max_conf[i]),
            "pred_llm": int(pred_llm[i]),
            "pred_ensemble": int(pred_ensemble[i]),
            "switched_to_llm": bool(low_conf_mask[i]),
        }
        if groups is not None:
            row["group"] = groups.iloc[X_test_bl.index[i]]
        all_decisions.append(row)

    n_low = low_conf_mask.sum()
    acc_bl = accuracy_score(y_test, pred_bl)
    acc_llm = accuracy_score(y_test, pred_llm)
    acc_ens = accuracy_score(y_test, pred_ensemble)

    resultados[nome] = {
        "acc_baseline": acc_bl,
        "acc_llm": acc_llm,
        "acc_ensemble": acc_ens,
        "n_low_conf": n_low,
        "pct_low_conf": n_low / len(y_test) * 100,
    }

    relatorio_lines.append(f"\n{'='*50}")
    relatorio_lines.append(f"=== {nome} ===")
    relatorio_lines.append(f"{'='*50}")
    relatorio_lines.append(f"Baseline: {acc_bl:.4f}")
    relatorio_lines.append(f"LLM:      {acc_llm:.4f}")
    relatorio_lines.append(f"Ensemble: {acc_ens:.4f}")
    relatorio_lines.append(f"Amostras baixa confiança: {n_low}/{len(y_test)} ({n_low/len(y_test)*100:.1f}%)")
    relatorio_lines.append(f"\nClassification Report (Ensemble):")
    relatorio_lines.append(classification_report(y_test, pred_ensemble))

    print(f"   {nome:20s}: BL={acc_bl:.3f} | LLM={acc_llm:.3f} | ENS={acc_ens:.3f} (low_conf={n_low})")

# ======================================================
# RELATÓRIO
# ======================================================
relatorio_path = os.path.join(ENSEMBLE_DIR, "relatorio_ensemble.txt")
with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write(f"RELATÓRIO ENSEMBLE ADAPTATIVO - {DATA_TYPE}\n")
    f.write(f"Threshold de confiança: {CONFIDENCE_THRESHOLD}\n")
    f.write(f"{'='*60}\n")
    f.write("\n".join(relatorio_lines))

# ======================================================
# GRÁFICO COMPARATIVO
# ======================================================
fig, ax = plt.subplots(figsize=(14, 6))
nomes = list(resultados.keys())
x = np.arange(len(nomes))
width = 0.25

bars1 = ax.bar(x - width, [r["acc_baseline"] for r in resultados.values()],
               width, label="Baseline", color="#4ecdc4")
bars2 = ax.bar(x, [r["acc_llm"] for r in resultados.values()],
               width, label="LLM", color="#ff6b6b")
bars3 = ax.bar(x + width, [r["acc_ensemble"] for r in resultados.values()],
               width, label="Ensemble", color="#45b7d1")

ax.set_ylabel("Acurácia")
ax.set_title(f"Ensemble Adaptativo - {DATA_TYPE.upper()} (threshold={CONFIDENCE_THRESHOLD})")
ax.set_xticks(x)
ax.set_xticklabels(nomes, rotation=45, ha="right")
ax.set_ylim([0, 1.05])
ax.legend()
ax.axhline(y=0.333, color="red", linestyle="--", alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f'{h:.1%}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(ENSEMBLE_DIR, "ensemble_comparacao.png"), dpi=300, bbox_inches="tight")
plt.close()

# CSV resumo
df_res = pd.DataFrame(resultados).T
df_res.to_csv(os.path.join(ENSEMBLE_DIR, "ensemble_resultados.csv"))

# CSV decisões por amostra
decisions_df = pd.DataFrame(all_decisions)
decisions_df.to_csv(os.path.join(ENSEMBLE_DIR, "ensemble_decisions.csv"), index=False)
n_switched = decisions_df["switched_to_llm"].sum()
print(f"\n   💾 ensemble_decisions.csv: {len(decisions_df)} linhas, {n_switched} switches para LLM")

# ======================================================
# RESUMO
# ======================================================
print(f"\n{'='*60}")
print(f"✅ ENSEMBLE CONCLUÍDO!")
print(f"{'='*60}")

n_better = sum(1 for r in resultados.values() if r["acc_ensemble"] > r["acc_baseline"])
n_better_llm = sum(1 for r in resultados.values() if r["acc_ensemble"] > r["acc_llm"])

print(f"\n📊 Ensemble > Baseline: {n_better}/{len(resultados)} modelos")
print(f"📊 Ensemble > LLM:      {n_better_llm}/{len(resultados)} modelos")

print(f"\n📊 RESULTADOS DETALHADOS:")
for nome, r in sorted(resultados.items(), key=lambda x: -x[1]["acc_ensemble"]):
    delta_bl = r["acc_ensemble"] - r["acc_baseline"]
    delta_llm = r["acc_ensemble"] - r["acc_llm"]
    print(f"   {nome:20s}: ENS={r['acc_ensemble']:.3f} (vs BL: {delta_bl:+.3f}, vs LLM: {delta_llm:+.3f})")

print(f"\n💾 Salvos em: {ENSEMBLE_DIR}")
print(f"{'='*60}")
