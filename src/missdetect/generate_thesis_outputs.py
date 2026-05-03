"""
Gera gráficos e tabelas para a tese do STEP05.

Consolida resultados de todos os experimentos e gera:
1. Tabela principal: comparação entre fases (v1, v2, v3, v4)
2. Heatmaps de confusion matrix (antes vs depois)
3. Barras empilhadas de feature importance por tipo
4. Box plots de CV scores
5. Scatter: accuracy sintético vs real
6. Tabela de ablação (configurações de features)
7. Testes de reprodutibilidade (3 seeds)
8. Avaliação de metas finais

Uso:
    python generate_thesis_outputs.py --experiment step05
    python generate_thesis_outputs.py --experiment step05 --data sintetico
    python generate_thesis_outputs.py --experiment step05 --data real
"""

import json
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.titlesize"] = 13
matplotlib.rcParams["axes.labelsize"] = 11

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.args import parse_common_args
from utils.paths import OUTPUT_BASE, find_result_dirs, get_comparison_dir, get_experiment_dir, get_output_dir

warnings.filterwarnings("ignore")

_, DATA_TYPE, _, EXPERIMENT = parse_common_args()

THESIS_DIR = os.path.join(get_experiment_dir(EXPERIMENT), "tese_outputs")
os.makedirs(THESIS_DIR, exist_ok=True)

print("=" * 60)
print("📝 GERAÇÃO DE OUTPUTS PARA TESE (STEP05)")
print("=" * 60)
print(f"🔬 Experimento: {EXPERIMENT}")
print(f"📂 Output: {THESIS_DIR}")
print("=" * 60)


# ======================================================
# UTILIDADES
# ======================================================
def load_experiment_data(experiment, data_type, model_name):
    """Carrega dados de um experimento."""
    out_dir = get_output_dir(data_type, model_name, experiment)
    data = {}

    for fname in [
        "training_summary.json",
        "feature_importance.csv",
        "metrics_per_class.csv",
        "cv_scores.csv",
        "confusion_matrices.json",
        "feature_selection_log.json",
    ]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            if fname.endswith(".csv"):
                data[fname.replace(".csv", "")] = pd.read_csv(fpath)
            elif fname.endswith(".json"):
                with open(fpath) as f:
                    data[fname.replace(".json", "")] = json.load(f)

    return data


def load_hier_data(experiment, data_type):
    """Carrega dados hierárquicos."""
    hier_dir = os.path.join(get_comparison_dir(data_type, experiment), "hierarquico")
    data = {}

    for fname in ["comparacao_hier_vs_direta.csv", "cv_logo_vs_groupkfold.csv", "training_summary.json"]:
        fpath = os.path.join(hier_dir, fname)
        if os.path.exists(fpath):
            if fname.endswith(".csv"):
                data[fname.replace(".csv", "")] = pd.read_csv(fpath)
            elif fname.endswith(".json"):
                with open(fpath) as f:
                    data[fname.replace(".json", "")] = json.load(f)

    return data


# ======================================================
# 1. TABELA PRINCIPAL: COMPARAÇÃO ENTRE FASES
# ======================================================
print("\n📊 1. Tabela principal de fases...")

phases = {}
# Busca resultados dos experimentos históricos
for exp_name in os.listdir(OUTPUT_BASE):
    exp_dir = os.path.join(OUTPUT_BASE, exp_name)
    if not os.path.isdir(exp_dir):
        continue

    for dt in ["sintetico", "real"]:
        baseline_dir = get_output_dir(dt, "none", exp_name)
        summary_path = os.path.join(baseline_dir, "training_summary.json")
        metrics_path = os.path.join(baseline_dir, "metrics_per_class.csv")
        cv_path = os.path.join(baseline_dir, "cv_scores.csv")

        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)

            key = f"{exp_name}/{dt}"
            entry = {
                "experiment": exp_name,
                "data_type": dt,
                "n_samples": summary.get("n_samples", 0),
                "n_features": summary.get("n_features", 0),
                "cv_method": summary.get("cv_method", ""),
                "split_method": summary.get("split_method", ""),
            }

            if os.path.exists(metrics_path):
                metrics = pd.read_csv(metrics_path)
                # Melhor accuracy (do modelo com maior accuracy)
                for modelo in metrics["modelo"].unique():
                    m_data = metrics[metrics["modelo"] == modelo]
                    acc = m_data["recall"].mean()  # Approximation
                # Busca accuracy do relatorio
                rel_path = os.path.join(baseline_dir, "relatorio.txt")
                if os.path.exists(rel_path):
                    best_acc = 0
                    with open(rel_path) as f:
                        for line in f:
                            if "Acurácia:" in line:
                                try:
                                    acc = float(line.split(":")[1].strip())
                                    best_acc = max(best_acc, acc)
                                except ValueError:
                                    pass
                    entry["best_accuracy"] = best_acc

                # Recall MNAR
                mnar_data = metrics[metrics["classe"] == "MNAR"]
                if not mnar_data.empty:
                    entry["best_recall_mnar"] = float(mnar_data["recall"].max())
                    entry["avg_recall_mnar"] = float(mnar_data["recall"].mean())

            if os.path.exists(cv_path):
                cv = pd.read_csv(cv_path)
                entry["cv_mean"] = float(cv["score"].mean())
                entry["cv_std"] = float(cv["score"].std())
                entry["cv_variance_pct"] = float(cv["score"].std() / max(cv["score"].mean(), 0.01) * 100)

            phases[key] = entry

if phases:
    df_phases = pd.DataFrame(phases.values())
    df_phases.to_csv(os.path.join(THESIS_DIR, "tabela_fases.csv"), index=False)
    print(f"   ✅ tabela_fases.csv ({len(df_phases)} entradas)")
    print(
        df_phases[["experiment", "data_type", "best_accuracy", "best_recall_mnar", "cv_variance_pct"]].to_string(
            index=False
        )
    )
else:
    print("   ⚠️ Nenhum dado encontrado")


# ======================================================
# 2. TABELA DE ABLAÇÃO (features x modelo)
# ======================================================
print("\n📊 2. Tabela de ablação...")

ablation_rows = []
for dt in ["sintetico", "real"]:
    result_dirs = find_result_dirs(dt, EXPERIMENT)
    for name, dir_path, _abordagem in result_dirs:
        rel_path = os.path.join(dir_path, "relatorio.txt")
        summary_path = os.path.join(dir_path, "training_summary.json")

        if not os.path.exists(rel_path):
            continue

        n_features = 0
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
                n_features = s.get("n_features", 0)

        with open(rel_path) as f:
            current_model = None
            for line in f:
                line = line.strip()
                if line.startswith("=== ") and line.endswith(" ==="):
                    current_model = line[4:-4]
                elif line.startswith("Acurácia:") and current_model:
                    acc = float(line.split(":")[1].strip())
                    ablation_rows.append(
                        {
                            "data_type": dt,
                            "configuracao": name,
                            "n_features": n_features,
                            "modelo_ml": current_model,
                            "accuracy": acc,
                        }
                    )

if ablation_rows:
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(os.path.join(THESIS_DIR, "tabela_ablacao.csv"), index=False)
    print(f"   ✅ tabela_ablacao.csv ({len(df_ablation)} entradas)")

    # Pivot table
    for dt in df_ablation["data_type"].unique():
        subset = df_ablation[df_ablation["data_type"] == dt]
        pivot = subset.pivot_table(index="modelo_ml", columns="configuracao", values="accuracy")
        print(f"\n   Ablação ({dt}):")
        print(pivot.to_string())
else:
    print("   ⚠️ Nenhum dado de ablação encontrado")


# ======================================================
# 3. FEATURE IMPORTANCE POR TIPO (barras empilhadas)
# ======================================================
print("\n📊 3. Feature importance por tipo...")

for dt in ["sintetico", "real"]:
    fi_path = os.path.join(get_output_dir(dt, "none", EXPERIMENT), "feature_importance.csv")
    if not os.path.exists(fi_path):
        continue

    fi = pd.read_csv(fi_path)

    # Classifica features por tipo
    def classify_feature(name):
        if name.startswith("llm_"):
            return "LLM"
        if name.startswith("emb_"):
            return "Embeddings"
        if name.startswith("caafe_"):
            return "CAAFE"
        if name.startswith("mechdetect_"):
            return "MechDetect"
        if name in ("X0_missing_rate", "X0_obs_vs_full_iqr_ratio", "X0_obs_vs_full_skew_diff", "X0_obs_count_ratio"):
            return "Estatística"
        return "Discriminativa"

    fi["tipo"] = fi["feature"].apply(classify_feature)

    # Gráfico: top 15 features coloridas por tipo
    top15 = fi.head(15)
    colors_map = {
        "Estatística": "#3498db",
        "Discriminativa": "#e74c3c",
        "MechDetect": "#2ecc71",
        "CAAFE": "#f39c12",
        "LLM": "#9b59b6",
        "Embeddings": "#1abc9c",
    }
    colors = [colors_map.get(t, "#95a5a6") for t in top15["tipo"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top15)), top15["importance"], color=colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["feature"], fontsize=9)
    ax.set_xlabel("Importância (RF)")
    ax.set_title(f"Top 15 Features — {dt.upper()}")
    ax.invert_yaxis()

    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=c, label=lbl) for lbl, c in colors_map.items() if lbl in top15["tipo"].values]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(THESIS_DIR, f"feature_importance_{dt}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Importância agregada por tipo
    tipo_agg = fi.groupby("tipo")["importance"].sum().sort_values(ascending=False)
    print(f"   Importância por tipo ({dt}):")
    for tipo, imp in tipo_agg.items():
        print(f"      {tipo}: {imp:.4f} ({imp/fi['importance'].sum()*100:.1f}%)")

    # Gráfico de pizza
    fig, ax = plt.subplots(figsize=(8, 6))
    tipo_colors = [colors_map.get(t, "#95a5a6") for t in tipo_agg.index]
    ax.pie(tipo_agg.values, labels=tipo_agg.index, autopct="%1.1f%%", colors=tipo_colors, startangle=90)
    ax.set_title(f"Importância por Tipo de Feature — {dt.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(THESIS_DIR, f"feature_importance_tipo_{dt}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✅ feature_importance_{dt}.png")
    print(f"   ✅ feature_importance_tipo_{dt}.png")


# ======================================================
# 4. BOX PLOTS DE CV SCORES
# ======================================================
print("\n📊 4. Box plots de CV scores...")

for dt in ["sintetico", "real"]:
    cv_path = os.path.join(get_output_dir(dt, "none", EXPERIMENT), "cv_scores.csv")
    if not os.path.exists(cv_path):
        continue

    cv = pd.read_csv(cv_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    modelos_cv = cv["modelo"].unique()
    data_boxes = [cv[cv["modelo"] == m]["score"].values for m in modelos_cv]

    bp = ax.boxplot(data_boxes, labels=modelos_cv, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(modelos_cv)))
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)

    ax.set_ylabel("Accuracy")
    ax.set_title(f"Distribuição CV Scores — {dt.upper()}")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(y=0.333, color="red", linestyle="--", alpha=0.3, label="Random")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(THESIS_DIR, f"cv_boxplot_{dt}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ cv_boxplot_{dt}.png")


# ======================================================
# 5. SCATTER: ACCURACY SINTÉTICO vs REAL
# ======================================================
print("\n📊 5. Scatter sintético vs real...")

acc_sint = {}
acc_real = {}

for dt, acc_dict in [("sintetico", acc_sint), ("real", acc_real)]:
    rel_path = os.path.join(get_output_dir(dt, "none", EXPERIMENT), "relatorio.txt")
    if not os.path.exists(rel_path):
        continue
    with open(rel_path) as f:
        current = None
        for line in f:
            line = line.strip()
            if line.startswith("=== ") and line.endswith(" ==="):
                current = line[4:-4]
            elif line.startswith("Acurácia:") and current:
                acc_dict[current] = float(line.split(":")[1].strip())

if acc_sint and acc_real:
    common_models = sorted(set(acc_sint.keys()) & set(acc_real.keys()))
    if common_models:
        fig, ax = plt.subplots(figsize=(8, 8))
        x_vals = [acc_sint[m] for m in common_models]
        y_vals = [acc_real[m] for m in common_models]

        ax.scatter(x_vals, y_vals, s=100, zorder=5, color="#3498db", edgecolors="black")
        for m, xv, yv in zip(common_models, x_vals, y_vals, strict=False):
            ax.annotate(m, (xv, yv), textcoords="offset points", xytext=(5, 5), fontsize=8)

        # Diagonal
        lims = [0, 1]
        ax.plot(lims, lims, "r--", alpha=0.5, label="Sintético = Real")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Accuracy (Sintético)")
        ax.set_ylabel("Accuracy (Real)")
        ax.set_title("Accuracy: Sintético vs Real")
        ax.legend()
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(os.path.join(THESIS_DIR, "scatter_sintetico_vs_real.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("   ✅ scatter_sintetico_vs_real.png")
else:
    print("   ⚠️ Dados insuficientes para scatter")


# ======================================================
# 6. HIERÁRQUICO: resumo
# ======================================================
print("\n📊 6. Resultados hierárquicos...")

for dt in ["sintetico", "real"]:
    hier_data = load_hier_data(EXPERIMENT, dt)
    if "comparacao_hier_vs_direta" in hier_data:
        df = hier_data["comparacao_hier_vs_direta"]
        print(f"\n   Hierárquico ({dt}):")
        cols = [
            "modelo",
            "acc_direta",
            "acc_hierarquica",
            "delta_acc",
            "recall_MNAR_direta",
            "recall_MNAR_hier",
            "delta_recall_MNAR",
        ]
        available_cols = [c for c in cols if c in df.columns]
        print(df[available_cols].to_string(index=False))


# ======================================================
# 7. TESTES DE REPRODUTIBILIDADE (multi-seed)
# ======================================================
print("\n📊 7. Teste de reprodutibilidade (3 seeds)...")

for dt in ["sintetico", "real"]:
    x_path = os.path.join(get_output_dir(dt, "none", EXPERIMENT), "X_features.csv")
    y_path = os.path.join(get_output_dir(dt, "none", EXPERIMENT), "y_labels.csv")
    g_path = os.path.join(get_output_dir(dt, "none", EXPERIMENT), "groups.csv")

    if not os.path.exists(x_path):
        continue

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze("columns")
    groups = pd.read_csv(g_path).squeeze("columns") if os.path.exists(g_path) else None

    seeds = [42, 123, 456]
    repro_results = []

    for seed in seeds:
        if groups is not None and groups.nunique() > 1:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            train_idx, test_idx = next(gss.split(X, y, groups))
        else:
            from sklearn.model_selection import train_test_split

            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, test_size=0.25, stratify=y, random_state=seed)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # SMOTE
        try:
            from imblearn.over_sampling import SMOTE

            min_count = y_train.value_counts().min()
            if min_count >= 2:
                k = min(3, min_count - 1)
                smote = SMOTE(random_state=seed, k_neighbors=k)
                X_train, y_train = smote.fit_resample(X_train, y_train)
        except ImportError:
            pass

        # RF e GB
        for model_name, model_cls in [
            ("RandomForest", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingClassifier(n_estimators=300, random_state=seed)),
        ]:
            model_cls.fit(X_train, y_train)
            y_pred = model_cls.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            recall_mnar = report.get("2", {}).get("recall", 0)
            repro_results.append(
                {
                    "data_type": dt,
                    "seed": seed,
                    "modelo": model_name,
                    "accuracy": acc,
                    "recall_mnar": recall_mnar,
                }
            )

    df_repro = pd.DataFrame(repro_results)
    df_repro.to_csv(os.path.join(THESIS_DIR, f"reproducibilidade_{dt}.csv"), index=False)

    # Análise
    for modelo in df_repro["modelo"].unique():
        sub = df_repro[df_repro["modelo"] == modelo]
        acc_range = sub["accuracy"].max() - sub["accuracy"].min()
        print(
            f"   {dt}/{modelo}: acc={sub['accuracy'].mean():.4f}±{sub['accuracy'].std():.4f} "
            f"(range={acc_range:.4f}, {'✅' if acc_range < 0.05 else '⚠️'} <5%)"
        )

    print(f"   ✅ reproducibilidade_{dt}.csv")


# ======================================================
# 8. AVALIAÇÃO DE METAS FINAIS
# ======================================================
print("\n📊 8. Avaliação de metas finais...")

metas = {
    "accuracy_real_70": {"meta": 0.70, "descricao": "Accuracy (melhor modelo, real) > 70%"},
    "recall_mnar_real_40": {"meta": 0.40, "descricao": "Recall MNAR (real) > 40%"},
    "cv_variancia_real_20": {"meta": 20.0, "descricao": "CV variância (real) < 20%"},
    "llm_delta_0": {"meta": 0.0, "descricao": "LLM delta vs baseline >= 0%"},
    "features_invariantes_30": {"meta": 30.0, "descricao": "Features invariantes importância > 30%"},
}

# Coleta resultados reais
real_baseline = get_output_dir("real", "none", EXPERIMENT)
real_metrics = os.path.join(real_baseline, "metrics_per_class.csv")
real_cv = os.path.join(real_baseline, "cv_scores.csv")
real_fi = os.path.join(real_baseline, "feature_importance.csv")
real_rel = os.path.join(real_baseline, "relatorio.txt")

resultados_metas = {}

# 1. Accuracy real
if os.path.exists(real_rel):
    best_acc = 0
    with open(real_rel) as f:
        for line in f:
            if "Acurácia:" in line:
                try:
                    acc = float(line.split(":")[1].strip())
                    best_acc = max(best_acc, acc)
                except ValueError:
                    pass
    resultados_metas["accuracy_real_70"] = {
        "valor": best_acc,
        "atingido": best_acc > 0.70,
    }

# 2. Recall MNAR real
if os.path.exists(real_metrics):
    metrics = pd.read_csv(real_metrics)
    mnar = metrics[metrics["classe"] == "MNAR"]
    if not mnar.empty:
        best_recall = mnar["recall"].max()
        resultados_metas["recall_mnar_real_40"] = {
            "valor": best_recall,
            "atingido": best_recall > 0.40,
        }

# 3. CV variância real
if os.path.exists(real_cv):
    cv = pd.read_csv(real_cv)
    cv_var = cv["score"].std() / max(cv["score"].mean(), 0.01) * 100
    resultados_metas["cv_variancia_real_20"] = {
        "valor": cv_var,
        "atingido": cv_var < 20.0,
    }

# 4. LLM delta
real_llm_dir = get_output_dir("real", "gemini-3-flash-preview", EXPERIMENT)
real_llm_rel = os.path.join(real_llm_dir, "relatorio.txt")
if os.path.exists(real_rel) and os.path.exists(real_llm_rel):
    best_bl = 0
    best_llm = 0
    for path, var in [(real_rel, "bl"), (real_llm_rel, "llm")]:
        best = 0
        with open(path) as f:
            for line in f:
                if "Acurácia:" in line:
                    try:
                        acc = float(line.split(":")[1].strip())
                        best = max(best, acc)
                    except ValueError:
                        pass
        if var == "bl":
            best_bl = best
        else:
            best_llm = best
    delta = best_llm - best_bl
    resultados_metas["llm_delta_0"] = {
        "valor": delta,
        "atingido": delta >= 0,
    }

# 5. Features invariantes
if os.path.exists(real_fi):
    fi = pd.read_csv(real_fi)
    invariant_features = [
        "mechdetect_auc_complete",
        "mechdetect_auc_shuffled",
        "mechdetect_auc_excluded",
        "mechdetect_delta_complete_shuffled",
        "mechdetect_delta_complete_excluded",
        "mechdetect_mwu_pvalue",
        "X0_missing_rate",
        "X0_obs_count_ratio",
        "X0_obs_vs_full_iqr_ratio",
        "X0_obs_vs_full_skew_diff",
    ]
    inv_importance = fi[fi["feature"].isin(invariant_features)]["importance"].sum()
    total_importance = fi["importance"].sum()
    pct = inv_importance / total_importance * 100 if total_importance > 0 else 0
    resultados_metas["features_invariantes_30"] = {
        "valor": pct,
        "atingido": pct > 30.0,
    }

# Exibe
print(f"\n   {'Métrica':<50} {'Meta':<15} {'Resultado':<15} {'Status'}")
print(f"   {'-'*95}")
for key, meta_info in metas.items():
    if key in resultados_metas:
        r = resultados_metas[key]
        status = "✅" if r["atingido"] else "❌"
        print(f"   {meta_info['descricao']:<50} {meta_info['meta']:<15} {r['valor']:<15.4f} {status}")
    else:
        print(f"   {meta_info['descricao']:<50} {meta_info['meta']:<15} {'N/A':<15} ⚠️")

# Salva
with open(os.path.join(THESIS_DIR, "metas_finais.json"), "w") as f:
    json.dump(
        {
            "metas": {k: {**v, "resultado": resultados_metas.get(k, {})} for k, v in metas.items()},
            "experiment": EXPERIMENT,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n   ✅ metas_finais.json")


# ======================================================
# 9. NARRATIVA CONSOLIDADA
# ======================================================
print("\n📊 9. Narrativa consolidada...")

narrative = []
narrative.append("=" * 60)
narrative.append("NARRATIVA DA TESE — RESULTADOS STEP05")
narrative.append("=" * 60)

# Coleta dados para narrativa
narrative.append("\n## 1. Dados sintéticos")
sint_rel = os.path.join(get_output_dir("sintetico", "none", EXPERIMENT), "relatorio.txt")
if os.path.exists(sint_rel):
    with open(sint_rel) as f:
        best_acc = 0
        for line in f:
            if "Acurácia:" in line:
                try:
                    acc = float(line.split(":")[1].strip())
                    best_acc = max(best_acc, acc)
                except ValueError:
                    pass
    narrative.append(f"   Melhor accuracy baseline: {best_acc:.1%}")

narrative.append("\n## 2. Dados reais")
if "accuracy_real_70" in resultados_metas:
    narrative.append(f"   Melhor accuracy baseline: {resultados_metas['accuracy_real_70']['valor']:.1%}")
if "recall_mnar_real_40" in resultados_metas:
    narrative.append(f"   Melhor recall MNAR: {resultados_metas['recall_mnar_real_40']['valor']:.1%}")
if "cv_variancia_real_20" in resultados_metas:
    narrative.append(f"   CV variância: {resultados_metas['cv_variancia_real_20']['valor']:.1f}%")

narrative.append("\n## 3. LLM contribution")
if "llm_delta_0" in resultados_metas:
    delta = resultados_metas["llm_delta_0"]["valor"]
    narrative.append(f"   Delta LLM vs baseline: {delta:+.1%}")
    if delta > 0:
        narrative.append("   → LLM contribui positivamente")
    elif delta < 0:
        narrative.append("   → LLM não melhora (possível overfitting ou ruído)")
    else:
        narrative.append("   → LLM neutro")

narrative.append("\n## 4. Classificação hierárquica")
for dt in ["sintetico", "real"]:
    hier = load_hier_data(EXPERIMENT, dt)
    if "training_summary" in hier:
        s = hier["training_summary"]
        narrative.append(
            f"   [{dt}] Direta: {s.get('best_acc_direct', 'N/A'):.4f} | "
            f"Hierárquica: {s.get('best_acc_hier', 'N/A'):.4f}"
        )

narrative.append("\n## 5. Metas atingidas")
n_atingidas = sum(1 for r in resultados_metas.values() if r.get("atingido", False))
n_total = len(metas)
narrative.append(f"   {n_atingidas}/{n_total} metas atingidas")

narrative_text = "\n".join(narrative)
with open(os.path.join(THESIS_DIR, "narrativa.txt"), "w", encoding="utf-8") as f:
    f.write(narrative_text)

print(narrative_text)

print(f"\n{'='*60}")
print("✅ OUTPUTS PARA TESE GERADOS!")
print(f"{'='*60}")
print(f"💾 Salvos em: {THESIS_DIR}")
print(f"{'='*60}")
