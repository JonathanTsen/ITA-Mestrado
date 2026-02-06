"""
Script de treinamento e avaliação de modelos v2.

Uso:
    python train_model.py --model <none|gemini-3-flash-preview|gpt-5.2>
    
Compara resultados com e sem features LLM.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURAÇÃO
# ======================================================
MODEL_NAME = "none"
if "--model" in sys.argv:
    idx = sys.argv.index("--model")
    if idx + 1 < len(sys.argv):
        MODEL_NAME = sys.argv[idx + 1]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_BASE = os.path.join(BASE_DIR, "Output", "v2_improved")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, MODEL_NAME)

X_IN = os.path.join(OUTPUT_DIR, "X_features.csv")
Y_IN = os.path.join(OUTPUT_DIR, "y_labels.csv")

if not os.path.exists(X_IN):
    print(f"❌ Arquivo não encontrado: {X_IN}")
    print(f"   Execute primeiro: python extract_features.py --model {MODEL_NAME}")
    sys.exit(1)

print(f"=" * 60)
print(f"🤖 TREINAMENTO DE MODELOS v2")
print(f"=" * 60)
print(f"📝 Modelo LLM: {MODEL_NAME}")
print(f"📂 Input: {OUTPUT_DIR}")
print(f"=" * 60)

# ======================================================
# CARREGA DADOS
# ======================================================
X = pd.read_csv(X_IN)
y = pd.read_csv(Y_IN).squeeze("columns")

print(f"\n📊 Dados carregados: X={X.shape}, y={y.shape}")
print(f"📊 Features: {len(X.columns)}")

# Separa features LLM das estatísticas
llm_cols = [c for c in X.columns if c.startswith("llm_")]
stat_cols = [c for c in X.columns if not c.startswith("llm_")]

print(f"   - Features estatísticas: {len(stat_cols)}")
print(f"   - Features LLM: {len(llm_cols)}")

# ======================================================
# MODELOS
# ======================================================
modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=42))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=3, random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=42))
    ]),
    "NaiveBayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
}

# ======================================================
# SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print(f"\n📈 Split: train={len(y_train)}, test={len(y_test)}")
print(f"   Distribuição train: {dict(y_train.value_counts().sort_index())}")
print(f"   Distribuição test:  {dict(y_test.value_counts().sort_index())}")

# ======================================================
# TREINAMENTO
# ======================================================
resultados = {}
relatorio_lines = []

print(f"\n🏋️ Treinando modelos...")
for nome, modelo in tqdm(modelos.items(), desc="Treinando"):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    resultados[nome] = {
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred)
    }
    
    relatorio_lines.append(f"\n{'='*50}")
    relatorio_lines.append(f"=== {nome} ===")
    relatorio_lines.append(f"{'='*50}")
    relatorio_lines.append(f"Acurácia: {acc:.4f}")
    relatorio_lines.append(classification_report(y_test, y_pred))
    relatorio_lines.append(f"Matriz de confusão:\n{confusion_matrix(y_test, y_pred)}")

# ======================================================
# ANÁLISE DE FEATURE IMPORTANCE (RandomForest)
# ======================================================
rf_model = modelos["RandomForest"]
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

relatorio_lines.append(f"\n{'='*50}")
relatorio_lines.append("=== FEATURE IMPORTANCE (RandomForest) ===")
relatorio_lines.append(f"{'='*50}")
relatorio_lines.append(feature_importance.head(20).to_string(index=False))

# Importância das features LLM vs Estatísticas
llm_importance = feature_importance[feature_importance["feature"].str.startswith("llm_")]["importance"].sum()
stat_importance = feature_importance[~feature_importance["feature"].str.startswith("llm_")]["importance"].sum()

relatorio_lines.append(f"\nImportância total LLM: {llm_importance:.4f}")
relatorio_lines.append(f"Importância total Estatísticas: {stat_importance:.4f}")
relatorio_lines.append(f"Razão LLM/Total: {llm_importance/(llm_importance+stat_importance):.4f}")

# ======================================================
# CROSS-VALIDATION
# ======================================================
relatorio_lines.append(f"\n{'='*50}")
relatorio_lines.append("=== CROSS-VALIDATION (5-fold) ===")
relatorio_lines.append(f"{'='*50}")

cv_results = {}
for nome, modelo in tqdm(modelos.items(), desc="Cross-validation"):
    scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
    cv_results[nome] = {"mean": scores.mean(), "std": scores.std()}
    relatorio_lines.append(f"{nome}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# ======================================================
# SALVA RELATÓRIO
# ======================================================
relatorio_path = os.path.join(OUTPUT_DIR, "relatorio.txt")
with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write(f"RELATÓRIO DE TREINAMENTO v2 - {MODEL_NAME}\n")
    f.write(f"{'='*60}\n")
    f.write("\n".join(relatorio_lines))

print(f"\n✅ Relatório salvo: {relatorio_path}")

# ======================================================
# GRÁFICOS
# ======================================================
# 1. Comparação de acurácias
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
ax1 = axes[0]
nomes = list(resultados.keys())
accs = [r["accuracy"] for r in resultados.values()]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(nomes)))

bars = ax1.bar(nomes, accs, color=colors, edgecolor="black")
ax1.set_ylabel("Acurácia")
ax1.set_title(f"Comparação de Modelos - {MODEL_NAME}")
ax1.set_ylim([0, 1])
ax1.tick_params(axis='x', rotation=45)
ax1.axhline(y=0.333, color='red', linestyle='--', alpha=0.5, label="Random (33.3%)")
ax1.legend()

for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

# Feature importance
ax2 = axes[1]
top_features = feature_importance.head(15)
colors2 = ['#ff6b6b' if f.startswith('llm_') else '#4ecdc4' for f in top_features["feature"]]
ax2.barh(range(len(top_features)), top_features["importance"], color=colors2)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features["feature"], fontsize=8)
ax2.set_xlabel("Importance")
ax2.set_title("Top 15 Features (RF)")
ax2.invert_yaxis()

# Legenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#ff6b6b', label='LLM'),
                   Patch(facecolor='#4ecdc4', label='Estatística')]
ax2.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "resultados.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Precisão por classe
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
classes = ["MCAR", "MAR", "MNAR"]
class_map = {"MCAR": "0", "MAR": "1", "MNAR": "2"}

for i, classe in enumerate(classes):
    ax = axes[i]
    precisoes = [r["report"][class_map[classe]]["precision"] for r in resultados.values()]
    recalls = [r["report"][class_map[classe]]["recall"] for r in resultados.values()]
    
    x = np.arange(len(nomes))
    width = 0.35
    
    ax.bar(x - width/2, precisoes, width, label="Precision", color="#3498db")
    ax.bar(x + width/2, recalls, width, label="Recall", color="#e74c3c")
    
    ax.set_ylabel("Score")
    ax.set_title(f"{classe}")
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=45, ha="right", fontsize=8)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=8)
    ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.3)

plt.suptitle(f"Precision/Recall por Classe - {MODEL_NAME}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precisao_por_classe.png"), dpi=300, bbox_inches="tight")
plt.close()

# ======================================================
# RESUMO FINAL
# ======================================================
print(f"\n{'='*60}")
print(f"✅ TREINAMENTO CONCLUÍDO!")
print(f"{'='*60}")

print(f"\n📊 RESULTADOS:")
for nome, res in sorted(resultados.items(), key=lambda x: -x[1]["accuracy"]):
    print(f"   {nome:20s}: {res['accuracy']:.4f}")

print(f"\n📊 CROSS-VALIDATION:")
for nome, res in sorted(cv_results.items(), key=lambda x: -x[1]["mean"]):
    print(f"   {nome:20s}: {res['mean']:.4f} (+/- {res['std']*2:.4f})")

best_model = max(resultados.items(), key=lambda x: x[1]["accuracy"])
print(f"\n🏆 Melhor modelo: {best_model[0]} ({best_model[1]['accuracy']:.4f})")

print(f"\n💾 Arquivos salvos em: {OUTPUT_DIR}")
print(f"   - relatorio.txt")
print(f"   - resultados.png")
print(f"   - precisao_por_classe.png")
print(f"{'='*60}")
