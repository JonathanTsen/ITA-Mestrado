"""
Análise EXAUSTIVA de relevância de features para classificação MCAR/MAR/MNAR.

Uso:
    python analyze_feature_relevance.py --model <modelo> [--data sintetico|real]

Exemplos:
    python analyze_feature_relevance.py --model gemini-3-flash-preview
    python analyze_feature_relevance.py --model gemini-3-pro-preview --data real

Este script analisa:
1. Importância via Random Forest
2. Importância por Permutação
3. Mutual Information
4. ANOVA F-score
5. Correlação entre features
6. Análise específica de features LLM (incluindo llm_mar_evidence)
7. Recursive Feature Elimination (RFE)
8. Análise de variância/constância
9. Correlação com target
10. Análise de redundância
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import (
    mutual_info_classif, f_classif, RFE, SelectKBest, RFECV
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.args import parse_common_args
from utils.paths import get_output_dir

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAÇÃO
# ============================================================
MODEL_NAME, DATA_TYPE, _, EXPERIMENT = parse_common_args()
OUTPUT_DIR = get_output_dir(DATA_TYPE, MODEL_NAME, EXPERIMENT)

X_PATH = os.path.join(OUTPUT_DIR, "X_features.csv")
Y_PATH = os.path.join(OUTPUT_DIR, "y_labels.csv")

print("=" * 80)
print("🔍 ANÁLISE EXAUSTIVA DE RELEVÂNCIA DE FEATURES")
print("=" * 80)

# ============================================================
# CARREGA DADOS
# ============================================================
print("\n📥 Carregando dados...")
X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH)["label"]

print(f"   Shape X: {X.shape}")
print(f"   Shape y: {y.shape}")
print(f"   Classes: {dict(y.value_counts().sort_index())}")

# Identifica features LLM vs Estatísticas
llm_features = [c for c in X.columns if c.startswith("llm_")]
stat_features = [c for c in X.columns if not c.startswith("llm_")]

print(f"\n📊 Features estatísticas: {len(stat_features)}")
print(f"🤖 Features LLM: {len(llm_features)}")
print(f"   {llm_features}")

# ============================================================
# 1. ANÁLISE DE VARIÂNCIA/CONSTÂNCIA
# ============================================================
print("\n" + "=" * 80)
print("1️⃣ ANÁLISE DE VARIÂNCIA E CONSTÂNCIA")
print("=" * 80)

variance_analysis = []
for col in X.columns:
    variance_analysis.append({
        "feature": col,
        "mean": X[col].mean(),
        "std": X[col].std(),
        "min": X[col].min(),
        "max": X[col].max(),
        "nunique": X[col].nunique(),
        "pct_zeros": (X[col] == 0).mean() * 100,
        "pct_constant": (X[col] == X[col].mode().iloc[0]).mean() * 100 if len(X[col].mode()) > 0 else 0,
        "cv": X[col].std() / X[col].mean() if X[col].mean() != 0 else 0,  # Coeficiente de variação
    })

var_df = pd.DataFrame(variance_analysis)
var_df.to_csv(os.path.join(OUTPUT_DIR, "variance_analysis.csv"), index=False)

# Features com pouca variância (std muito baixo ou quase constantes)
low_var_features = var_df[
    (var_df["std"] < 0.01) | 
    (var_df["pct_constant"] > 90) |
    (var_df["nunique"] <= 3)
]["feature"].tolist()

print(f"\n⚠️ Features com BAIXA VARIÂNCIA (std < 0.01 ou >90% constante):")
for f in low_var_features:
    row = var_df[var_df["feature"] == f].iloc[0]
    print(f"   - {f}: std={row['std']:.4f}, %constante={row['pct_constant']:.1f}%, unique={row['nunique']}")

# ============================================================
# 2. RANDOM FOREST IMPORTANCE
# ============================================================
print("\n" + "=" * 80)
print("2️⃣ RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)

rf_importance = pd.DataFrame({
    "feature": X.columns,
    "rf_importance": rf.feature_importances_
}).sort_values("rf_importance", ascending=False)

print("\n🏆 Top 20 features (Random Forest):")
for i, row in rf_importance.head(20).iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']}: {row['rf_importance']:.4f}")

print("\n❌ Bottom 10 features (Random Forest):")
for i, row in rf_importance.tail(10).iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']}: {row['rf_importance']:.6f}")

# ============================================================
# 3. PERMUTATION IMPORTANCE
# ============================================================
print("\n" + "=" * 80)
print("3️⃣ PERMUTATION IMPORTANCE")
print("=" * 80)

perm_result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)

perm_importance = pd.DataFrame({
    "feature": X.columns,
    "perm_mean": perm_result.importances_mean,
    "perm_std": perm_result.importances_std
}).sort_values("perm_mean", ascending=False)

print("\n🏆 Top 20 features (Permutation):")
for i, row in perm_importance.head(20).iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']}: {row['perm_mean']:.4f} ± {row['perm_std']:.4f}")

# Features com importância NEGATIVA (pioram se removidas - podem ser ruído)
negative_perm = perm_importance[perm_importance["perm_mean"] < 0]
print(f"\n⚠️ Features com permutation importance NEGATIVO ({len(negative_perm)}):")
for i, row in negative_perm.iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']}: {row['perm_mean']:.4f}")

# ============================================================
# 4. MUTUAL INFORMATION
# ============================================================
print("\n" + "=" * 80)
print("4️⃣ MUTUAL INFORMATION")
print("=" * 80)

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({
    "feature": X.columns,
    "mi_score": mi_scores
}).sort_values("mi_score", ascending=False)

print("\n🏆 Top 20 features (Mutual Information):")
for i, row in mi_df.head(20).iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']}: {row['mi_score']:.4f}")

# Features com MI zero ou muito baixo
low_mi = mi_df[mi_df["mi_score"] < 0.01]
print(f"\n⚠️ Features com MI < 0.01 ({len(low_mi)}):")
for i, row in low_mi.iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']}: {row['mi_score']:.6f}")

# ============================================================
# 5. ANOVA F-SCORE
# ============================================================
print("\n" + "=" * 80)
print("5️⃣ ANOVA F-SCORE")
print("=" * 80)

f_scores, p_values = f_classif(X, y)
f_df = pd.DataFrame({
    "feature": X.columns,
    "f_score": f_scores,
    "p_value": p_values
}).sort_values("f_score", ascending=False)

print("\n🏆 Top 20 features (F-score):")
for i, row in f_df.head(20).iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
    print(f"   {marker} {row['feature']}: F={row['f_score']:.2f} {sig}")

# Features não significativas
non_sig = f_df[f_df["p_value"] > 0.05]
print(f"\n⚠️ Features NÃO significativas (p > 0.05) ({len(non_sig)}):")
for i, row in non_sig.iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']}: F={row['f_score']:.2f}, p={row['p_value']:.4f}")

# ============================================================
# 6. ANÁLISE ESPECÍFICA DE FEATURES LLM
# ============================================================
print("\n" + "=" * 80)
print("6️⃣ ANÁLISE ESPECÍFICA DE FEATURES LLM")
print("=" * 80)

if llm_features:
    print("\n📊 Estatísticas descritivas das features LLM:")
    llm_stats = X[llm_features].describe().T
    print(llm_stats)
else:
    print("\n⚠️ Nenhuma feature LLM presente no dataset")

if llm_features:
    # Importância relativa das LLM features
    llm_rf_imp = rf_importance[rf_importance["feature"].isin(llm_features)]
    print("\n🤖 Ranking de features LLM por RF importance:")
    for i, row in llm_rf_imp.sort_values("rf_importance", ascending=False).iterrows():
        print(f"   {row['feature']}: {row['rf_importance']:.4f}")

# Análise específica de llm_mar_evidence
print("\n" + "-" * 40)
print("🔍 ANÁLISE DETALHADA: llm_mar_evidence")
print("-" * 40)

if "llm_mar_evidence" in X.columns:
    mar_ev = X["llm_mar_evidence"]
    
    # Estatísticas por classe
    print("\n📊 Estatísticas por classe (0=MCAR, 1=MAR, 2=MNAR):")
    for cls in [0, 1, 2]:
        cls_data = mar_ev[y == cls]
        cls_name = {0: "MCAR", 1: "MAR", 2: "MNAR"}[cls]
        print(f"   {cls_name}: mean={cls_data.mean():.4f}, std={cls_data.std():.4f}, "
              f"min={cls_data.min():.4f}, max={cls_data.max():.4f}")
    
    # ANOVA teste
    groups = [mar_ev[y == cls] for cls in [0, 1, 2]]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n   ANOVA: F={f_stat:.2f}, p={p_val:.6f}")
    
    # Correlação ponto-biserial com cada classe
    print("\n   Correlação com cada classe:")
    for cls in [0, 1, 2]:
        cls_name = {0: "MCAR", 1: "MAR", 2: "MNAR"}[cls]
        binary_target = (y == cls).astype(int)
        corr, p = stats.pointbiserialr(binary_target, mar_ev)
        print(f"      vs {cls_name}: r={corr:.4f}, p={p:.6f}")
    
    # RF importance ranking
    rank = rf_importance[rf_importance["feature"] == "llm_mar_evidence"]["rf_importance"].values[0]
    total = len(rf_importance)
    position = rf_importance[rf_importance["feature"] == "llm_mar_evidence"].index[0]
    print(f"\n   RF Importance: {rank:.4f} (posição {list(rf_importance['feature']).index('llm_mar_evidence') + 1}/{total})")
    
    # Permutation importance
    perm_val = perm_importance[perm_importance["feature"] == "llm_mar_evidence"]["perm_mean"].values[0]
    print(f"   Permutation Importance: {perm_val:.4f}")
    
    # MI
    mi_val = mi_df[mi_df["feature"] == "llm_mar_evidence"]["mi_score"].values[0]
    print(f"   Mutual Information: {mi_val:.4f}")
    
    # Conclusão
    print("\n   ✅ CONCLUSÃO sobre llm_mar_evidence:")
    if rank > 0.01 and mi_val > 0.01:
        print("      RELEVANTE - mantém contribuição significativa")
    elif rank > 0.005 or mi_val > 0.005:
        print("      MARGINALMENTE RELEVANTE - pode ser mantida")
    else:
        print("      POUCO RELEVANTE - considerar remoção")
else:
    print("   ⚠️ llm_mar_evidence não encontrada no dataset")

# Salva análise de features LLM
if llm_features:
    llm_analysis_rows = []
    for feat in llm_features:
        row = {"feature": feat}
        # Média por classe
        for cls in [0, 1, 2]:
            cls_name = {0: "MCAR", 1: "MAR", 2: "MNAR"}[cls]
            row[f"mean_{cls_name}"] = float(X[feat][y == cls].mean())
        # ANOVA
        grps = [X[feat][y == cls] for cls in [0, 1, 2]]
        f_stat, p_val = stats.f_oneway(*grps)
        row["anova_F"] = float(f_stat)
        row["anova_p"] = float(p_val)
        # Correlação com target
        corr_val, corr_p = stats.spearmanr(X[feat], y)
        row["spearman_r"] = float(corr_val)
        row["spearman_p"] = float(corr_p)
        # RF importance
        rf_val = rf_importance[rf_importance["feature"] == feat]["rf_importance"].values
        row["rf_importance"] = float(rf_val[0]) if len(rf_val) > 0 else 0.0
        llm_analysis_rows.append(row)
    pd.DataFrame(llm_analysis_rows).to_csv(os.path.join(OUTPUT_DIR, "llm_feature_analysis.csv"), index=False)

# ============================================================
# 7. RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================
print("\n" + "=" * 80)
print("7️⃣ RECURSIVE FEATURE ELIMINATION (RFE)")
print("=" * 80)

rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), 
          n_features_to_select=15, step=5)
rfe.fit(X, y)

rfe_df = pd.DataFrame({
    "feature": X.columns,
    "rfe_selected": rfe.support_,
    "rfe_ranking": rfe.ranking_
}).sort_values("rfe_ranking")

print("\n🏆 Top 15 features selecionadas por RFE:")
selected = rfe_df[rfe_df["rfe_selected"]]
for i, row in selected.iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']}")

print("\n❌ Features eliminadas primeiro (menos importantes):")
eliminated_first = rfe_df[~rfe_df["rfe_selected"]].sort_values("rfe_ranking", ascending=False).head(10)
for i, row in eliminated_first.iterrows():
    marker = "🤖" if row["feature"].startswith("llm_") else "📊"
    print(f"   {marker} {row['feature']} (ranking={row['rfe_ranking']})")

# ============================================================
# 8. CORRELAÇÃO ENTRE FEATURES (REDUNDÂNCIA)
# ============================================================
print("\n" + "=" * 80)
print("8️⃣ ANÁLISE DE REDUNDÂNCIA (CORRELAÇÃO ALTA)")
print("=" * 80)

corr_matrix = X.corr()
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
high_corr_pairs = []

for i, col1 in enumerate(X.columns):
    for j, col2 in enumerate(X.columns):
        if i < j:
            corr_val = abs(corr_matrix.loc[col1, col2])
            if corr_val > 0.9:
                high_corr_pairs.append((col1, col2, corr_val))

high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\n⚠️ Pares de features altamente correlacionadas (|r| > 0.9): {len(high_corr_pairs)}")
for col1, col2, corr_val in high_corr_pairs[:15]:
    imp1 = rf_importance[rf_importance["feature"] == col1]["rf_importance"].values[0]
    imp2 = rf_importance[rf_importance["feature"] == col2]["rf_importance"].values[0]
    keep = col1 if imp1 > imp2 else col2
    remove = col2 if imp1 > imp2 else col1
    print(f"   {col1} <-> {col2}: r={corr_val:.3f}")
    print(f"      → Manter: {keep}, Remover: {remove}")

# ============================================================
# 9. IMPACTO NA PERFORMANCE (ABLATION STUDY)
# ============================================================
print("\n" + "=" * 80)
print("9️⃣ ABLATION STUDY - IMPACTO NA PERFORMANCE")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline com todas features
baseline_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X, y, cv=cv, scoring="accuracy"
)
baseline = baseline_scores.mean()
print(f"\n📊 Baseline (todas features): {baseline:.4f} ± {baseline_scores.std():.4f}")

# Sem features LLM
if llm_features:
    X_no_llm = X.drop(columns=llm_features)
    no_llm_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        X_no_llm, y, cv=cv, scoring="accuracy"
    )
    no_llm = no_llm_scores.mean()
    print(f"📊 Sem features LLM: {no_llm:.4f} ± {no_llm_scores.std():.4f} (Δ={no_llm-baseline:+.4f})")

# Apenas top 15 features (RFE)
top_features = rfe_df[rfe_df["rfe_selected"]]["feature"].tolist()
X_top = X[top_features]
top_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_top, y, cv=cv, scoring="accuracy"
)
top_acc = top_scores.mean()
print(f"📊 Apenas top 15 (RFE): {top_acc:.4f} ± {top_scores.std():.4f} (Δ={top_acc-baseline:+.4f})")

# Salva ablation results
ablation_rows = [
    {"config": "todas_features", "accuracy_mean": baseline, "accuracy_std": baseline_scores.std(),
     "n_features": X.shape[1], "delta": 0.0},
]
if llm_features:
    ablation_rows.append({
        "config": "sem_llm", "accuracy_mean": no_llm, "accuracy_std": no_llm_scores.std(),
        "n_features": X_no_llm.shape[1], "delta": no_llm - baseline,
    })
ablation_rows.append({
    "config": "top15_rfe", "accuracy_mean": top_acc, "accuracy_std": top_scores.std(),
    "n_features": len(top_features), "delta": top_acc - baseline,
})
pd.DataFrame(ablation_rows).to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)

# ============================================================
# 10. CONSOLIDAÇÃO - FEATURES PARA REMOVER
# ============================================================
print("\n" + "=" * 80)
print("🎯 CONSOLIDAÇÃO - FEATURES RECOMENDADAS PARA REMOÇÃO")
print("=" * 80)

# Critérios para remoção
features_to_remove = set()
removal_reasons = {}

def add_removal(feature, reason):
    features_to_remove.add(feature)
    if feature not in removal_reasons:
        removal_reasons[feature] = []
    removal_reasons[feature].append(reason)

# 1. Baixa variância
for f in low_var_features:
    add_removal(f, "baixa_variância")

# 2. RF importance muito baixa (< 0.005)
low_rf = rf_importance[rf_importance["rf_importance"] < 0.005]["feature"].tolist()
for f in low_rf:
    add_removal(f, "rf_importance<0.005")

# 3. Permutation importance negativa
for f in negative_perm["feature"].tolist():
    add_removal(f, "perm_importance_negativa")

# 4. MI muito baixo (< 0.01)
for f in low_mi["feature"].tolist():
    add_removal(f, "mi<0.01")

# 5. Não significativo estatisticamente
for f in non_sig["feature"].tolist():
    add_removal(f, "p>0.05")

# 6. Alta correlação (remover o menos importante)
for col1, col2, corr_val in high_corr_pairs:
    imp1 = rf_importance[rf_importance["feature"] == col1]["rf_importance"].values[0]
    imp2 = rf_importance[rf_importance["feature"] == col2]["rf_importance"].values[0]
    remove = col2 if imp1 > imp2 else col1
    add_removal(remove, f"redundante_com_{col1 if remove == col2 else col2}")

# 7. RFE eliminadas primeiro (ranking alto)
rfe_eliminated = rfe_df[rfe_df["rfe_ranking"] > len(X.columns) // 2]["feature"].tolist()
for f in rfe_eliminated:
    add_removal(f, "rfe_eliminada_cedo")

# Ordenar por número de razões
removal_scores = {f: len(reasons) for f, reasons in removal_reasons.items()}
sorted_removals = sorted(removal_scores.items(), key=lambda x: x[1], reverse=True)

print(f"\n📋 Features para remover ({len(features_to_remove)}):")
print("-" * 60)

stat_to_remove = []
llm_to_remove = []

for f, score in sorted_removals:
    reasons = removal_reasons[f]
    marker = "🤖 LLM" if f.startswith("llm_") else "📊 STAT"
    print(f"   {marker} {f}")
    print(f"      Razões ({score}): {', '.join(reasons)}")
    
    if f.startswith("llm_"):
        llm_to_remove.append(f)
    else:
        stat_to_remove.append(f)

# Features para MANTER
features_to_keep = [f for f in X.columns if f not in features_to_remove]
print(f"\n✅ Features para MANTER ({len(features_to_keep)}):")
for f in features_to_keep:
    marker = "🤖" if f.startswith("llm_") else "📊"
    imp = rf_importance[rf_importance["feature"] == f]["rf_importance"].values[0]
    print(f"   {marker} {f}: RF={imp:.4f}")

# ============================================================
# 11. SALVA RESULTADOS
# ============================================================
print("\n" + "=" * 80)
print("💾 SALVANDO RESULTADOS")
print("=" * 80)

output_file = os.path.join(OUTPUT_DIR, "feature_relevance_analysis.csv")

# Merge all analyses
final_df = rf_importance.merge(perm_importance, on="feature")
final_df = final_df.merge(mi_df, on="feature")
final_df = final_df.merge(f_df, on="feature")
final_df = final_df.merge(rfe_df, on="feature")
final_df["to_remove"] = final_df["feature"].isin(features_to_remove)
final_df["n_removal_reasons"] = final_df["feature"].apply(lambda x: removal_scores.get(x, 0))
final_df["removal_reasons"] = final_df["feature"].apply(lambda x: "|".join(removal_reasons.get(x, [])))

final_df.to_csv(output_file, index=False)
print(f"   ✅ Salvo: {output_file}")
print(f"   ✅ Salvo: variance_analysis.csv")
print(f"   ✅ Salvo: correlation_matrix.csv")
print(f"   ✅ Salvo: ablation_results.csv")
if llm_features:
    print(f"   ✅ Salvo: llm_feature_analysis.csv")

# Salva lista de remoção
removal_file = os.path.join(OUTPUT_DIR, "features_to_remove.txt")
with open(removal_file, "w") as f:
    f.write("# Features recomendadas para remoção\n")
    f.write(f"# Total: {len(features_to_remove)}\n\n")
    
    f.write("# ESTATÍSTICAS:\n")
    for feat in sorted(stat_to_remove):
        f.write(f"{feat}\n")
    
    f.write("\n# LLM:\n")
    for feat in sorted(llm_to_remove):
        f.write(f"{feat}\n")

print(f"   ✅ Salvo: {removal_file}")

# ============================================================
# RESUMO FINAL
# ============================================================
print("\n" + "=" * 80)
print("📊 RESUMO FINAL")
print("=" * 80)

print(f"""
Total de features analisadas: {len(X.columns)}
  - Estatísticas: {len(stat_features)}
  - LLM: {len(llm_features)}

Features para REMOVER: {len(features_to_remove)}
  - Estatísticas: {len(stat_to_remove)}
  - LLM: {len(llm_to_remove)}

Features para MANTER: {len(features_to_keep)}
  - Estatísticas: {len([f for f in features_to_keep if not f.startswith("llm_")])}
  - LLM: {len([f for f in features_to_keep if f.startswith("llm_")])}

Sobre llm_mar_evidence:
  - Status: {"MANTER ✅" if "llm_mar_evidence" not in features_to_remove else "REMOVER ❌"}
""")

print("=" * 80)
print("✅ ANÁLISE CONCLUÍDA!")
print("=" * 80)
