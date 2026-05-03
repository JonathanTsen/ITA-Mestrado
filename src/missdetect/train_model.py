"""
Script de treinamento e avaliação de modelos v2.

Uso:
    python train_model.py --model <none|gemini-3-flash-preview|gpt-5.2> [--data sintetico|real]

Exemplos:
    python train_model.py --model none                              # Baseline ML, dados sintéticos
    python train_model.py --model gemini-3-flash-preview --data real # ML + LLM, dados reais
"""

import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.args import parse_common_args
from utils.paths import get_output_dir

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURAÇÃO
# ======================================================
MODEL_NAME, DATA_TYPE, _, EXPERIMENT = parse_common_args()

OUTPUT_DIR = get_output_dir(DATA_TYPE, MODEL_NAME, EXPERIMENT)

X_IN = os.path.join(OUTPUT_DIR, "X_features.csv")
Y_IN = os.path.join(OUTPUT_DIR, "y_labels.csv")

if not os.path.exists(X_IN):
    print(f"❌ Arquivo não encontrado: {X_IN}")
    print(f"   Execute primeiro: python extract_features.py --model {MODEL_NAME} --data {DATA_TYPE}")
    sys.exit(1)

ABORDAGEM = "apenas ML (baseline)" if MODEL_NAME == "none" else f"ML + LLM ({MODEL_NAME})"

print("=" * 60)
print("🤖 TREINAMENTO DE MODELOS v2")
print("=" * 60)
print(f"📊 Dados: {DATA_TYPE}")
print(f"🔬 Abordagem: {ABORDAGEM}")
print(f"📝 Modelo LLM: {MODEL_NAME}")
print(f"📂 Input: {OUTPUT_DIR}")
print("=" * 60)

# ======================================================
# CARREGA DADOS
# ======================================================
X = pd.read_csv(X_IN)
y = pd.read_csv(Y_IN).squeeze("columns")

# Carrega grupos (dataset de origem) para GroupShuffleSplit
GROUPS_IN = os.path.join(OUTPUT_DIR, "groups.csv")
groups = None
if os.path.exists(GROUPS_IN):
    groups = pd.read_csv(GROUPS_IN).squeeze("columns")
    n_groups = groups.nunique()
    print(f"\n📊 Dados carregados: X={X.shape}, y={y.shape}, groups={n_groups}")
else:
    print(f"\n📊 Dados carregados: X={X.shape}, y={y.shape}")
print(f"📊 Features: {len(X.columns)}")

# Separa features LLM das estatísticas
llm_cols = [c for c in X.columns if c.startswith("llm_")]
stat_cols = [c for c in X.columns if not c.startswith("llm_")]

print(f"   - Features estatísticas: {len(stat_cols)}")
print(f"   - Features LLM: {len(llm_cols)}")

# ======================================================
# FEATURE SELECTION ADAPTATIVA
# ======================================================
n_samples = len(X)
max_features = max(5, n_samples // 10)

feature_selection_log = {
    "method": "SelectKBest(f_classif)",
    "n_original": int(X.shape[1]),
    "n_samples": int(n_samples),
    "max_features": int(max_features),
    "applied": False,
    "features_selected": list(X.columns),
    "features_removed": [],
    "scores": {},
    "p_values": {},
}

if X.shape[1] > max_features:
    print(f"\n🔍 Feature selection: {X.shape[1]} → {max_features} features (n={n_samples})")
    selector = SelectKBest(f_classif, k=max_features)
    X_selected = pd.DataFrame(selector.fit_transform(X, y), columns=X.columns[selector.get_support()])
    removed = set(X.columns) - set(X_selected.columns)
    print(f"   Removidas: {removed}")

    feature_selection_log["applied"] = True
    feature_selection_log["n_selected"] = int(max_features)
    feature_selection_log["features_selected"] = list(X_selected.columns)
    feature_selection_log["features_removed"] = list(removed)
    for feat, score, pval in zip(X.columns, selector.scores_, selector.pvalues_, strict=False):
        feature_selection_log["scores"][feat] = float(score)
        feature_selection_log["p_values"][feat] = float(pval)

    X = X_selected

    # Recalcula colunas LLM/stat após seleção
    llm_cols = [c for c in X.columns if c.startswith("llm_")]
    stat_cols = [c for c in X.columns if not c.startswith("llm_")]
else:
    print(f"\n🔍 Feature selection: {X.shape[1]} features mantidas (n={n_samples}, max={max_features})")


# ======================================================
# MODELOS (hiperparâmetros adaptativos ao tamanho do dataset)
# ======================================================
def get_modelos(n_samples: int, has_llm_features: bool = False) -> dict:
    """Retorna modelos com hiperparâmetros adaptados ao tamanho do dataset.

    Quando has_llm_features=True, adiciona PCA nos modelos sensíveis a
    dimensionalidade (SVM, KNN, MLP) para mitigar a maldição da dimensionalidade.
    """
    # PCA para modelos sensíveis quando LLM features estão presentes
    pca_step = ("pca", PCA(n_components=0.95, random_state=42)) if has_llm_features else None

    def _sensitive_pipeline(clf_step, use_pca=True):
        """Cria pipeline com scaler + PCA opcional + classificador."""
        steps = [("scaler", StandardScaler())]
        if use_pca and pca_step is not None:
            steps.append(pca_step)
        steps.append(("clf", clf_step))
        return Pipeline(steps)

    if n_samples < 100:
        # Dataset pequeno: modelos simples, menos overfitting
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            ),
            "LogisticRegression": _sensitive_pipeline(
                LogisticRegression(max_iter=3000, C=0.5, random_state=42), use_pca=False
            ),
            "SVM_RBF": _sensitive_pipeline(SVC(kernel="rbf", C=1, random_state=42, probability=True)),
            "KNN": _sensitive_pipeline(KNeighborsClassifier(n_neighbors=3)),
            "MLP": _sensitive_pipeline(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)),
            "NaiveBayes": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        }
    else:
        # Dataset grande: modelos originais
        return {
            "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=300, random_state=42),
            "LogisticRegression": _sensitive_pipeline(
                LogisticRegression(max_iter=3000, random_state=42), use_pca=False
            ),
            "SVM_RBF": _sensitive_pipeline(SVC(kernel="rbf", C=3, random_state=42, probability=True)),
            "KNN": _sensitive_pipeline(KNeighborsClassifier(n_neighbors=5)),
            "MLP": _sensitive_pipeline(MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=42)),
            "NaiveBayes": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        }


has_llm = len(llm_cols) > 0
modelos = get_modelos(len(X), has_llm_features=has_llm)
regime = "pequeno (n<100)" if len(X) < 100 else "grande (n>=100)"
print(f"\n⚙️ Hiperparâmetros: regime {regime}")
if has_llm:
    print("   PCA ativado para SVM/KNN/MLP (features LLM presentes)")

# ======================================================
# SPLIT (com GroupShuffleSplit se grupos disponíveis)
# ======================================================
if groups is not None and groups.nunique() > 1:
    # GroupShuffleSplit: bootstraps do mesmo dataset ficam juntos
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    print(f"\n📈 Split (GroupShuffleSplit): train={len(y_train)}, test={len(y_test)}")
    print(f"   Grupos treino: {sorted(groups_train.unique())}")
    print(f"   Grupos teste:  {sorted(groups_test.unique())}")
    overlap = set(groups_train.unique()) & set(groups_test.unique())
    if overlap:
        print(f"   ⚠️ LEAK: grupos em ambos: {overlap}")
    else:
        print("   ✅ Sem leakage: 0 grupos compartilhados")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    print(f"\n📈 Split: train={len(y_train)}, test={len(y_test)}")

print(f"   Distribuição train: {dict(pd.Series(y_train).value_counts().sort_index())}")
print(f"   Distribuição test:  {dict(pd.Series(y_test).value_counts().sort_index())}")

# SMOTE para balanceamento de classes no treino
try:
    from imblearn.over_sampling import SMOTE

    min_class_count = y_train.value_counts().min()
    if min_class_count >= 2:
        k_neighbors = min(3, min_class_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   SMOTE aplicado (k={k_neighbors}): train={len(y_train)}")
        print(f"   Distribuição pós-SMOTE: {dict(pd.Series(y_train).value_counts().sort_index())}")
    else:
        print(f"   SMOTE não aplicado: classe mínima tem apenas {min_class_count} amostra(s)")
except ImportError:
    print("   SMOTE não disponível (instale: pip install imbalanced-learn)")

# ======================================================
# TREINAMENTO
# ======================================================
resultados = {}
relatorio_lines = []
all_predictions = []
class_labels = sorted(y.unique())
class_names = {0: "MCAR", 1: "MAR", 2: "MNAR"}

print("\n🏋️ Treinando modelos...")
for nome, modelo in tqdm(modelos.items(), desc="Treinando"):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Probabilidades por classe
    if (
        hasattr(modelo, "predict_proba")
        or hasattr(modelo, "named_steps")
        and hasattr(modelo.named_steps.get("clf", modelo), "predict_proba")
    ):
        y_proba = modelo.predict_proba(X_test)
    else:
        y_proba = np.full((len(X_test), len(class_labels)), np.nan)

    # Acumula predições por amostra
    for i in range(len(X_test)):
        row = {
            "sample_idx": X_test.index[i],
            "y_true": int(y_test.iloc[i]),
            "y_pred": int(y_pred[i]),
            "modelo": nome,
            "confianca": float(np.nanmax(y_proba[i])),
        }
        for ci, cl in enumerate(class_labels):
            row[f"prob_{class_names.get(cl, cl)}"] = float(y_proba[i, ci])
        if groups is not None:
            row["group"] = groups.iloc[X_test.index[i]]
        all_predictions.append(row)

    acc = accuracy_score(y_test, y_pred)
    resultados[nome] = {
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred),
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
feature_importance = pd.DataFrame({"feature": X.columns, "importance": rf_model.feature_importances_}).sort_values(
    "importance", ascending=False
)

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
n_cv = len(X)
if groups is not None and groups.nunique() > 1:
    n_groups_total = groups.nunique()
    n_folds = min(n_groups_total, 5)
    cv_strategy = GroupKFold(n_splits=n_folds)
    cv_name = f"Group {n_folds}-Fold (por dataset de origem)"
    cv_groups = groups
elif n_cv < 50:
    cv_strategy = LeaveOneOut()
    cv_name = "Leave-One-Out"
    cv_groups = None
else:
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    cv_name = "Repeated Stratified 5-Fold (3x)"
    cv_groups = None

relatorio_lines.append(f"\n{'='*50}")
relatorio_lines.append(f"=== CROSS-VALIDATION ({cv_name}) ===")
relatorio_lines.append(f"{'='*50}")
print(f"\n📊 Estratégia CV: {cv_name} (n={n_cv})")

cv_results = {}
cv_scores_rows = []
for nome, modelo in tqdm(modelos.items(), desc="Cross-validation"):
    scores = cross_val_score(modelo, X, y, cv=cv_strategy, scoring="accuracy", groups=cv_groups)
    cv_results[nome] = {"mean": scores.mean(), "std": scores.std()}
    relatorio_lines.append(f"{nome}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    for fold_i, score in enumerate(scores):
        cv_scores_rows.append({"modelo": nome, "fold": fold_i, "score": float(score)})

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
# SALVA OUTPUTS ESTRUTURADOS
# ======================================================
print("\n💾 Salvando outputs estruturados...")

# 1. predictions.csv
predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

# 2. metrics_per_class.csv
metrics_rows = []
for nome, res in resultados.items():
    for cl_key, cl_name in class_names.items():
        cl_str = str(cl_key)
        if cl_str in res["report"]:
            metrics_rows.append(
                {
                    "modelo": nome,
                    "classe": cl_name,
                    "precision": res["report"][cl_str]["precision"],
                    "recall": res["report"][cl_str]["recall"],
                    "f1": res["report"][cl_str]["f1-score"],
                    "support": int(res["report"][cl_str]["support"]),
                }
            )
pd.DataFrame(metrics_rows).to_csv(os.path.join(OUTPUT_DIR, "metrics_per_class.csv"), index=False)

# 3. feature_importance.csv (todas as features, não só top 20)
feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

# 4. cv_scores.csv
pd.DataFrame(cv_scores_rows).to_csv(os.path.join(OUTPUT_DIR, "cv_scores.csv"), index=False)

# 5. confusion_matrices.json
conf_matrices = {}
for nome, res in resultados.items():
    conf_matrices[nome] = res["confusion"].tolist()
with open(os.path.join(OUTPUT_DIR, "confusion_matrices.json"), "w") as f:
    json.dump(conf_matrices, f, indent=2)

# 6. hyperparameters.json
hyperparams = {}
for nome, modelo in modelos.items():
    hyperparams[nome] = modelo.get_params()


# Converter tipos não serializáveis
def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if callable(obj):
        return str(obj)
    if isinstance(obj, type):
        return str(obj)
    return obj


with open(os.path.join(OUTPUT_DIR, "hyperparameters.json"), "w") as f:
    json.dump(_sanitize(hyperparams), f, indent=2, default=str)

# 7. feature_selection_log.json
with open(os.path.join(OUTPUT_DIR, "feature_selection_log.json"), "w") as f:
    json.dump(feature_selection_log, f, indent=2)

# 8. training_summary.json
training_summary = {
    "timestamp": datetime.now().isoformat(),
    "model_name": MODEL_NAME,
    "data_type": DATA_TYPE,
    "experiment": EXPERIMENT,
    "abordagem": ABORDAGEM,
    "n_samples": int(len(X)),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_features": int(X.shape[1]),
    "features": list(X.columns),
    "n_features_llm": len(llm_cols),
    "n_features_stat": len(stat_cols),
    "split_method": "GroupShuffleSplit"
    if (groups is not None and groups.nunique() > 1)
    else "StratifiedTrainTestSplit",
    "cv_method": cv_name,
    "seed": 42,
}
if groups is not None and groups.nunique() > 1:
    training_summary["groups_train"] = sorted(groups_train.unique().tolist())
    training_summary["groups_test"] = sorted(groups_test.unique().tolist())
with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
    json.dump(training_summary, f, indent=2, ensure_ascii=False)

print(f"   ✅ predictions.csv ({len(predictions_df)} linhas)")
print(f"   ✅ metrics_per_class.csv ({len(metrics_rows)} linhas)")
print(f"   ✅ feature_importance.csv ({len(feature_importance)} features)")
print(f"   ✅ cv_scores.csv ({len(cv_scores_rows)} linhas)")
print(f"   ✅ confusion_matrices.json ({len(conf_matrices)} modelos)")
print(f"   ✅ hyperparameters.json ({len(hyperparams)} modelos)")
print("   ✅ feature_selection_log.json")
print("   ✅ training_summary.json")

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
ax1.tick_params(axis="x", rotation=45)
ax1.axhline(y=0.333, color="red", linestyle="--", alpha=0.5, label="Random (33.3%)")
ax1.legend()

for bar, acc in zip(bars, accs, strict=False):
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, f"{acc:.1%}", ha="center", va="bottom", fontsize=9
    )

# Feature importance
ax2 = axes[1]
top_features = feature_importance.head(15)
colors2 = ["#ff6b6b" if f.startswith("llm_") else "#4ecdc4" for f in top_features["feature"]]
ax2.barh(range(len(top_features)), top_features["importance"], color=colors2)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features["feature"], fontsize=8)
ax2.set_xlabel("Importance")
ax2.set_title("Top 15 Features (RF)")
ax2.invert_yaxis()

# Legenda
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor="#ff6b6b", label="LLM"), Patch(facecolor="#4ecdc4", label="Estatística")]
ax2.legend(handles=legend_elements, loc="lower right")

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

    ax.bar(x - width / 2, precisoes, width, label="Precision", color="#3498db")
    ax.bar(x + width / 2, recalls, width, label="Recall", color="#e74c3c")

    ax.set_ylabel("Score")
    ax.set_title(f"{classe}")
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=45, ha="right", fontsize=8)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=8)
    ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.3)

plt.suptitle(f"Precision/Recall por Classe - {MODEL_NAME}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precisao_por_classe.png"), dpi=300, bbox_inches="tight")
plt.close()

# ======================================================
# RESUMO FINAL
# ======================================================
print(f"\n{'='*60}")
print("✅ TREINAMENTO CONCLUÍDO!")
print(f"{'='*60}")

print("\n📊 RESULTADOS:")
for nome, res in sorted(resultados.items(), key=lambda x: -x[1]["accuracy"]):
    print(f"   {nome:20s}: {res['accuracy']:.4f}")

print("\n📊 CROSS-VALIDATION:")
for nome, res in sorted(cv_results.items(), key=lambda x: -x[1]["mean"]):
    print(f"   {nome:20s}: {res['mean']:.4f} (+/- {res['std']*2:.4f})")

best_model = max(resultados.items(), key=lambda x: x[1]["accuracy"])
print(f"\n🏆 Melhor modelo: {best_model[0]} ({best_model[1]['accuracy']:.4f})")

print(f"\n💾 Arquivos salvos em: {OUTPUT_DIR}")
print("   - relatorio.txt")
print("   - resultados.png")
print("   - precisao_por_classe.png")
print("   - predictions.csv")
print("   - metrics_per_class.csv")
print("   - feature_importance.csv")
print("   - cv_scores.csv")
print("   - confusion_matrices.json")
print("   - hyperparameters.json")
print("   - feature_selection_log.json")
print("   - training_summary.json")
print(f"{'='*60}")
