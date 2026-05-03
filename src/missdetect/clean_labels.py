"""
Limpeza de labels ruidosos usando Cleanlab (Confident Learning) — STEP 01.

Identifica labels potencialmente incorretos nos dados reais e gera:
1. Relatório de qualidade por dataset
2. Labels corrigidos (y_labels_clean.csv)
3. Scores de qualidade por amostra (label_quality.csv)

Uso:
    python clean_labels.py --experiment step05_pro
    python clean_labels.py --experiment step05_pro --action prune --prune-pct 20
    python clean_labels.py --experiment step05_pro --action weight
    python clean_labels.py --experiment step05_pro --action relabel
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import OUTPUT_BASE, get_output_dir

CLASS_NAMES = {0: "MCAR", 1: "MAR", 2: "MNAR"}


def get_clean_data(experiment, data_type="real", mode="weight"):
    """Retorna dados limpos para uso em train_hierarchical_v3plus.py."""
    analysis_dir = os.path.join(OUTPUT_BASE, experiment, data_type, "label_analysis")
    baseline_dir = get_output_dir(data_type, "none", experiment)

    if mode == "weight":
        X = pd.read_csv(os.path.join(baseline_dir, "X_features.csv"))
        y = pd.read_csv(os.path.join(baseline_dir, "y_labels.csv")).squeeze("columns")
        weights = pd.read_csv(os.path.join(analysis_dir, "sample_weights.csv"))["sample_weight"].values
        groups = None
        gp = os.path.join(baseline_dir, "groups.csv")
        if os.path.exists(gp):
            groups = pd.read_csv(gp).squeeze("columns")
        return X, y, weights, groups

    elif mode == "prune":
        X = pd.read_csv(os.path.join(analysis_dir, "X_features_clean.csv"))
        y = pd.read_csv(os.path.join(analysis_dir, "y_labels_clean.csv")).squeeze("columns")
        groups = None
        gp = os.path.join(analysis_dir, "groups_clean.csv")
        if os.path.exists(gp):
            groups = pd.read_csv(gp).squeeze("columns")
        return X, y, None, groups

    elif mode == "relabel":
        X = pd.read_csv(os.path.join(baseline_dir, "X_features.csv"))
        y = pd.read_csv(os.path.join(analysis_dir, "y_labels_relabeled.csv")).squeeze("columns")
        groups = None
        gp = os.path.join(baseline_dir, "groups.csv")
        if os.path.exists(gp):
            groups = pd.read_csv(gp).squeeze("columns")
        return X, y, None, groups


def main():
    parser = argparse.ArgumentParser(description="Cleanlab: limpeza de labels ruidosos")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--action", choices=["report", "prune", "weight", "relabel"], default="report")
    parser.add_argument("--prune-pct", type=float, default=15.0)
    parser.add_argument("--data", choices=["sintetico", "real"], default="real")
    args = parser.parse_args()

    EXPERIMENT = args.experiment
    DATA_TYPE = args.data
    ACTION = args.action
    PRUNE_PCT = args.prune_pct

    # Paths
    BASELINE_DIR = get_output_dir(DATA_TYPE, "none", EXPERIMENT)
    ANALYSIS_DIR = os.path.join(OUTPUT_BASE, EXPERIMENT, DATA_TYPE, "label_analysis")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("=" * 70)
    print("CLEANLAB — LIMPEZA DE LABELS RUIDOSOS")
    print("=" * 70)
    print(f"  Dados: {DATA_TYPE}  |  Acao: {ACTION}")
    print(f"  Baseline: {BASELINE_DIR}")
    print(f"  Output:   {ANALYSIS_DIR}")
    print("=" * 70)

    # Carrega dados
    X = pd.read_csv(os.path.join(BASELINE_DIR, "X_features.csv"))
    y = pd.read_csv(os.path.join(BASELINE_DIR, "y_labels.csv")).squeeze("columns")

    groups = None
    groups_path = os.path.join(BASELINE_DIR, "groups.csv")
    if os.path.exists(groups_path):
        groups = pd.read_csv(groups_path).squeeze("columns")

    print(f"\n  Dados: {len(y)} amostras, {X.shape[1]} features")
    print(f"  Classes: {dict(y.value_counts().sort_index())}")
    if groups is not None:
        print(f"  Grupos (datasets): {groups.nunique()}")

    # Probabilidades out-of-sample
    print("\n  Obtendo probabilidades out-of-sample com cross-validation...")
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=42)

    if groups is not None and groups.nunique() > 4:
        n_splits = min(groups.nunique(), 10)
        cv = GroupKFold(n_splits=n_splits)
        pred_probs = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")
        print(f"  CV: GroupKFold({n_splits})")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pred_probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
        print("  CV: StratifiedKFold(5)")

    # Cleanlab
    print("\n  Analisando qualidade dos labels com Cleanlab...")
    from cleanlab.count import compute_confident_joint
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores

    label_quality_scores = get_label_quality_scores(
        labels=y.values,
        pred_probs=pred_probs,
        method="self_confidence",
    )

    issue_indices = find_label_issues(
        labels=y.values,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        n_jobs=1,  # Evita multiprocessing no macOS
    )

    n_issues = len(issue_indices)
    pct_issues = 100 * n_issues / len(y)
    print(f"  Labels problematicos: {n_issues}/{len(y)} ({pct_issues:.1f}%)")

    # Confident Joint
    confident_joint = compute_confident_joint(labels=y.values, pred_probs=pred_probs)

    print("\n  Confident Joint (label dado x label estimado):")
    print(f"  {'':>10s} | pred MCAR | pred MAR | pred MNAR")
    for i, name in CLASS_NAMES.items():
        row = confident_joint[i]
        print(f"  {name:>10s} | {row[0]:>8d} | {row[1]:>7d} | {row[2]:>8d}")

    # Relatório por dataset
    is_issue = np.zeros(len(y), dtype=bool)
    is_issue[issue_indices] = True
    suggested_labels = pred_probs.argmax(axis=1)

    dataset_quality = []
    if groups is not None:
        print("\n  Qualidade por dataset:")
        for dataset_name in sorted(groups.unique()):
            mask = groups == dataset_name
            n_samples = mask.sum()
            avg_quality = label_quality_scores[mask].mean()
            min_quality = label_quality_scores[mask].min()
            n_issues_ds = is_issue[mask].sum()
            pct_issues_ds = 100 * n_issues_ds / n_samples

            original_label = y[mask].mode().iloc[0]
            original_name = CLASS_NAMES[original_label]
            most_predicted = pd.Series(suggested_labels[mask]).mode().iloc[0]
            predicted_name = CLASS_NAMES[most_predicted]
            agrees = original_name == predicted_name

            sym = "OK" if agrees else "!!"
            print(
                f"  [{sym}] {dataset_name:40s}: quality={avg_quality:.3f} "
                f"issues={n_issues_ds:3d}/{n_samples} ({pct_issues_ds:4.1f}%) "
                f"label={original_name:5s} -> pred={predicted_name:5s}"
            )

            dataset_quality.append(
                {
                    "dataset": dataset_name,
                    "n_samples": int(n_samples),
                    "avg_quality": float(avg_quality),
                    "min_quality": float(min_quality),
                    "n_issues": int(n_issues_ds),
                    "pct_issues": float(pct_issues_ds),
                    "original_label": int(original_label),
                    "original_name": original_name,
                    "predicted_label": int(most_predicted),
                    "predicted_name": predicted_name,
                    "label_agrees": agrees,
                }
            )

    df_quality = pd.DataFrame(dataset_quality)
    df_quality.to_csv(os.path.join(ANALYSIS_DIR, "quality_by_dataset.csv"), index=False)

    n_disagree = (~df_quality["label_agrees"]).sum() if len(df_quality) > 0 else 0
    print(f"\n  Datasets com label discordante: {n_disagree}/{len(df_quality)}")

    # Salvar scores
    df_scores = pd.DataFrame(
        {
            "index": np.arange(len(y)),
            "label_original": y.values,
            "label_name": [CLASS_NAMES[v] for v in y.values],
            "quality_score": label_quality_scores,
            "is_issue": is_issue,
            "suggested_label": suggested_labels,
            "suggested_name": [CLASS_NAMES[v] for v in suggested_labels],
            "prob_MCAR": pred_probs[:, 0],
            "prob_MAR": pred_probs[:, 1],
            "prob_MNAR": pred_probs[:, 2],
        }
    )
    if groups is not None:
        df_scores["group"] = groups.values

    df_scores.to_csv(os.path.join(ANALYSIS_DIR, "label_quality_scores.csv"), index=False)

    df_issues = df_scores.iloc[issue_indices].copy()
    df_issues["rank"] = np.arange(1, len(issue_indices) + 1)
    df_issues.to_csv(os.path.join(ANALYSIS_DIR, "label_issues_ranked.csv"), index=False)

    # Ações
    if ACTION == "prune":
        n_remove = int(len(y) * PRUNE_PCT / 100)
        n_remove = min(n_remove, len(issue_indices))
        remove_idx = issue_indices[:n_remove]
        keep_mask = np.ones(len(y), dtype=bool)
        keep_mask[remove_idx] = False

        X[keep_mask].reset_index(drop=True).to_csv(os.path.join(ANALYSIS_DIR, "X_features_clean.csv"), index=False)
        y[keep_mask].reset_index(drop=True).to_csv(os.path.join(ANALYSIS_DIR, "y_labels_clean.csv"), index=False)
        if groups is not None:
            groups[keep_mask].reset_index(drop=True).to_csv(os.path.join(ANALYSIS_DIR, "groups_clean.csv"), index=False)

        y_clean = y[keep_mask]
        print(f"\n  PRUNE: Removidas {n_remove} amostras ({PRUNE_PCT:.0f}%)")
        print(f"  Antes:  {len(y)} — {dict(y.value_counts().sort_index())}")
        print(f"  Depois: {len(y_clean)} — {dict(y_clean.value_counts().sort_index())}")

    elif ACTION == "weight":
        weights = np.clip(label_quality_scores.copy(), 0.1, 1.0)
        pd.DataFrame({"sample_weight": weights}).to_csv(os.path.join(ANALYSIS_DIR, "sample_weights.csv"), index=False)
        print(
            f"\n  WEIGHT: Pesos salvos (min={weights.min():.3f}, "
            f"mean={weights.mean():.3f}, max={weights.max():.3f})"
        )

    elif ACTION == "relabel":
        y_relabeled = y.copy()
        for idx in issue_indices:
            y_relabeled.iloc[idx] = suggested_labels[idx]
        n_changed = (y_relabeled != y).sum()
        y_relabeled.to_csv(os.path.join(ANALYSIS_DIR, "y_labels_relabeled.csv"), index=False)
        print(f"\n  RELABEL: {n_changed} labels alterados")
        print(f"  Antes:  {dict(y.value_counts().sort_index())}")
        print(f"  Depois: {dict(y_relabeled.value_counts().sort_index())}")
    else:
        print("\n  REPORT: Apenas relatorio gerado (use --action para aplicar correcoes)")

    # Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(label_quality_scores, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(
        x=np.percentile(label_quality_scores, PRUNE_PCT),
        color="red",
        linestyle="--",
        label=f"Percentil {PRUNE_PCT:.0f}%",
    )
    axes[0].set_xlabel("Label Quality Score")
    axes[0].set_ylabel("Frequencia")
    axes[0].set_title("Distribuicao de Quality Scores")
    axes[0].legend()

    for cls_id, cls_name in CLASS_NAMES.items():
        mask_cls = y == cls_id
        axes[1].hist(label_quality_scores[mask_cls], bins=30, alpha=0.6, label=cls_name)
    axes[1].set_xlabel("Label Quality Score")
    axes[1].set_ylabel("Frequencia")
    axes[1].set_title("Quality Score por Classe")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "quality_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confident_joint, cmap="YlOrRd", interpolation="nearest")
    labels_names = list(CLASS_NAMES.values())
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels_names)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels_names)
    ax.set_xlabel("Label Estimado")
    ax.set_ylabel("Label Dado")
    ax.set_title("Confident Joint")
    plt.colorbar(im, ax=ax)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(confident_joint[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if confident_joint[i, j] > confident_joint.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "confident_joint.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if groups is not None and len(df_quality) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        df_sorted = df_quality.sort_values("avg_quality")
        colors = ["#e74c3c" if not a else "#2ecc71" for a in df_sorted["label_agrees"]]
        ax.barh(range(len(df_sorted)), df_sorted["avg_quality"], color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["dataset"], fontsize=7)
        ax.set_xlabel("Average Quality Score")
        ax.set_title("Label Quality por Dataset (vermelho = label discordante)")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, "quality_by_dataset.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Summary JSON
    summary = {
        "experiment": EXPERIMENT,
        "data_type": DATA_TYPE,
        "action": ACTION,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_issues": int(n_issues),
        "pct_issues": float(pct_issues),
        "mean_quality": float(label_quality_scores.mean()),
        "median_quality": float(np.median(label_quality_scores)),
        "quality_by_class": {CLASS_NAMES[cls]: float(label_quality_scores[y == cls].mean()) for cls in CLASS_NAMES},
    }
    if groups is not None:
        summary["n_datasets_disagreeing"] = int(n_disagree)
        summary["n_datasets_total"] = int(len(df_quality))

    with open(os.path.join(ANALYSIS_DIR, "clean_labels_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("CLEANLAB CONCLUIDO!")
    print(f"{'='*70}")
    print(f"Salvos em: {ANALYSIS_DIR}")
    for f_name in sorted(os.listdir(ANALYSIS_DIR)):
        print(f"  - {f_name}")


if __name__ == "__main__":
    main()
