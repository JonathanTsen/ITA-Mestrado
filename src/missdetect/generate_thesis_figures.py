#!/usr/bin/env python3
"""
Gera todas as figuras novas para a tese (Step 3 das pendências).
Saída: ModeloTesePPGPO/figuras/
"""

import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
THESIS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "ModeloTesePPGPO"))
OUT_DIR = os.path.join(THESIS_DIR, "figuras")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output", "v2_improved")

os.makedirs(OUT_DIR, exist_ok=True)

# Consistent style
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3a — Fluxograma do Pipeline Hierárquico
# ══════════════════════════════════════════════════════════════════════════════
def fig_3a_flowchart():
    print("3a: Fluxograma do pipeline hierárquico")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    font_kw = {"ha": "center", "va": "center", "fontsize": 9, "fontweight": "bold"}

    def box(x, y, w, h, text, color="#E8F4FD", ec="#2980B9"):
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3", facecolor=color, edgecolor=ec, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, text, **font_kw)

    def arrow(x1, y1, x2, y2, text="", color="#2C3E50"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5})
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, text, fontsize=7, color="#7F8C8D", style="italic")

    # Phase labels
    ax.text(3.5, 7.6, "Phase 1: Synthetic Validation", fontsize=12, fontweight="bold", color="#2C3E50", ha="center")
    ax.text(10.5, 7.6, "Phase 2: Real-World Hierarchical", fontsize=12, fontweight="bold", color="#2C3E50", ha="center")

    # Divider
    ax.plot([7, 7], [0.3, 7.4], "--", color="#BDC3C7", lw=1.5)

    # Phase 1
    box(2, 6.8, 2.5, 0.6, "Gerador v2\n1,200 datasets", "#E8F4FD")
    box(5, 6.8, 2.2, 0.6, "Feature Extraction\n25 features", "#E8F4FD")
    box(3.5, 5.6, 3.0, 0.6, "Direct 3-way Classification\nMCAR / MAR / MNAR", "#D5F5E3")
    box(3.5, 4.3, 3.0, 0.7, "Key Finding:\nMAR separable\nMCAR ≈ MNAR", "#FDEBD0", "#E67E22")

    arrow(3.25, 6.5, 4.85, 6.95)
    arrow(5, 6.5, 3.7, 5.9)
    arrow(3.5, 5.3, 3.5, 4.65)

    # Phase 2
    box(9, 6.8, 2.2, 0.6, "23 Real Datasets\n1,132 bootstraps", "#E8F4FD")
    box(12, 6.8, 2.2, 0.6, "Feature Extraction\n25 features", "#E8F4FD")
    box(10.5, 5.5, 3.0, 0.6, "Level 1: Binary\nMCAR vs non-MCAR", "#D5F5E3", "#27AE60")
    arrow(10.15, 6.5, 10.5, 5.8)
    arrow(12, 6.5, 10.7, 5.8)

    # Routing
    box(10.5, 4.3, 2.8, 0.6, "Probabilistic Routing\nSoft 3-zone", "#F5E6FF", "#8E44AD")
    arrow(10.5, 5.2, 10.5, 4.6)

    # L2
    box(10.5, 3.1, 3.0, 0.6, "Level 2: Binary\nMAR vs MNAR", "#D5F5E3", "#27AE60")
    arrow(10.5, 4.0, 10.5, 3.4)

    # Cleanlab
    box(10.5, 1.9, 2.5, 0.6, "Cleanlab\nLabel noise weights", "#FFF3CD", "#F39C12")
    arrow(10.5, 2.8, 10.5, 2.2)

    # Final
    box(10.5, 0.7, 3.0, 0.6, "Final Prediction\n56.0% LOGO CV", "#FADBD8", "#E74C3C")
    arrow(10.5, 1.6, 10.5, 1.0)

    # Cross-phase arrow
    ax.annotate(
        "",
        xy=(8.5, 4.9),
        xytext=(5.5, 4.3),
        arrowprops={"arrowstyle": "-|>", "color": "#E67E22", "lw": 2, "linestyle": "dashed"},
    )
    ax.text(6.8, 4.9, "informs\ndesign", fontsize=7, color="#E67E22", ha="center", style="italic")

    fig.suptitle("Hierarchical Pipeline Overview", fontsize=14, fontweight="bold", y=0.98)
    save(fig, "fluxograma_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3b — Confusion Matrices (Direct RF vs Hierarchical GBT)
# ══════════════════════════════════════════════════════════════════════════════
def fig_3b_confusion():
    print("3b: Confusion matrices")
    cm_path = os.path.join(OUTPUT_DIR, "ctx_baseline", "real", "hierarquico", "confusion_matrices.json")
    with open(cm_path) as f:
        data = json.load(f)

    labels = ["MCAR", "MAR", "MNAR"]
    configs = [
        ("Direct — RandomForest", np.array(data["RandomForest"]["direta"])),
        ("Hierarchical — RandomForest", np.array(data["RandomForest"]["hierarquica"])),
        ("Direct — GradientBoosting", np.array(data["GradientBoosting"]["direta"])),
        ("Hierarchical — GradientBoosting", np.array(data["GradientBoosting"]["hierarquica"])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (title, cm) in zip(axes.flat, configs, strict=False):
        acc = np.trace(cm) / cm.sum() * 100
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_title(f"{title}\n(Acc: {acc:.1f}%)", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Confusion Matrices — Direct vs Hierarchical Classification", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "confusion_matrices_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3d — Evolução Progressiva de Accuracy
# ══════════════════════════════════════════════════════════════════════════════
def fig_3d_evolution():
    print("3d: Evolução progressiva de accuracy")
    stages = [
        ("Stage 0\nv1 direct\n(10f+8LLM)", 72.0, "synth"),
        ("Stage 1\nv2 direct\n(21f)", 76.67, "synth"),
        ("Stage 2\nv2+CAAFE\n(25f)", 77.67, "synth"),
        ("Stage 3\nv2+LLM\n(33f)", 79.33, "synth"),
        ("Stage 4\nV3 hier hard\n(LOGO)", 51.42, "real"),
        ("Stage 5\nV3+ soft3zone\n(LOGO)", 55.97, "real"),
        ("Stage 6\nV3+ CL\n(holdout)", 53.22, "holdout"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"synth": "#3498DB", "real": "#E74C3C", "holdout": "#F39C12"}
    color_labels = {"synth": "Synthetic (test set)", "real": "Real (LOGO CV)", "holdout": "Real (holdout)"}
    x = np.arange(len(stages))
    ax.bar(x, [s[1] for s in stages], color=[colors[s[2]] for s in stages], edgecolor="white", linewidth=0.5, width=0.7)

    for i, (_label, val, _) in enumerate(stages):
        ax.text(i, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 92)

    # Phase divider
    ax.axvline(3.5, color="#BDC3C7", linestyle="--", lw=1.5)
    ax.text(1.5, 88, "Phase 1: Synthetic", ha="center", fontsize=10, color="#2C3E50", fontweight="bold")
    ax.text(5, 88, "Phase 2: Real", ha="center", fontsize=10, color="#2C3E50", fontweight="bold")

    # Legend
    handles = [
        mpatches.Patch(color=c, label=lbl)
        for c, lbl in [
            (colors["synth"], color_labels["synth"]),
            (colors["real"], color_labels["real"]),
            (colors["holdout"], color_labels["holdout"]),
        ]
    ]
    ax.legend(handles=handles, loc="center left", framealpha=0.9, bbox_to_anchor=(0.0, 0.55))

    ax.set_title("Progressive Accuracy Evolution (Stage 0 → 6)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "evolucao_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3e — Label Quality Distribution
# ══════════════════════════════════════════════════════════════════════════════
def fig_3e_label_quality():
    print("3e: Label quality distribution")
    csv_path = os.path.join(OUTPUT_DIR, "step05_pro", "real", "label_analysis", "label_quality_scores.csv")
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall histogram
    ax1.hist(df["quality_score"], bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
    mean_q = df["quality_score"].mean()
    median_q = df["quality_score"].median()
    ax1.axvline(mean_q, color="#E74C3C", linestyle="--", lw=2, label=f"Mean: {mean_q:.3f}")
    ax1.axvline(median_q, color="#F39C12", linestyle="-.", lw=2, label=f"Median: {median_q:.3f}")
    ax1.axvline(0.5, color="#27AE60", linestyle=":", lw=2, label="Issue threshold: 0.5")
    ax1.set_xlabel("Label Quality Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Overall Distribution (n=1,132)")
    ax1.legend(fontsize=9)

    # Per-class
    class_colors = {"MCAR": "#E74C3C", "MAR": "#3498DB", "MNAR": "#27AE60"}
    for cls in ["MCAR", "MAR", "MNAR"]:
        subset = df[df["label_name"] == cls]["quality_score"]
        ax2.hist(
            subset,
            bins=30,
            alpha=0.5,
            color=class_colors[cls],
            label=f"{cls} (μ={subset.mean():.3f})",
            edgecolor="white",
        )
    ax2.set_xlabel("Label Quality Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution by Class")
    ax2.legend(fontsize=9)

    fig.suptitle("Cleanlab Label Quality Score Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "label_quality_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3f — Confident Joint Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig_3f_confident_joint():
    print("3f: Confident joint heatmap")
    cj = np.array(
        [
            [88, 109, 35],
            [43, 441, 66],
            [57, 215, 78],
        ]
    )
    labels = ["MCAR", "MAR", "MNAR"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cj,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1,
        linecolor="white",
    )
    ax.set_xlabel("Estimated True Label", fontsize=12)
    ax.set_ylabel("Given (Noisy) Label", fontsize=12)
    ax.set_title("Confident Joint Matrix (n=1,132)", fontsize=13, fontweight="bold")

    # Annotate key finding
    ax.text(
        1.5,
        -0.3,
        "MAR absorbs 109 MCAR + 215 MNAR samples",
        ha="center",
        fontsize=9,
        color="#E74C3C",
        style="italic",
        transform=ax.transData,
    )

    save(fig, "confident_joint_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3g — Gap Sintético vs Real por Classe
# ══════════════════════════════════════════════════════════════════════════════
def fig_3g_gap():
    print("3g: Gap sintético vs real por classe")
    classes = ["MCAR", "MAR", "MNAR"]
    synthetic = [80, 85, 65]
    real = [47, 56, 46]
    gaps = [-33, -29, -19]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, synthetic, width, label="Synthetic", color="#3498DB", edgecolor="white")
    bars2 = ax.bar(x + width / 2, real, width, label="Real (LOGO CV)", color="#E74C3C", edgecolor="white")

    # Gap annotations
    for i, gap in enumerate(gaps):
        ax.annotate(
            f"{gap}pp",
            xy=(i + width / 2, real[i]),
            xytext=(i + 0.55, real[i] + 8),
            fontsize=10,
            fontweight="bold",
            color="#E74C3C",
            arrowprops={"arrowstyle": "->", "color": "#E74C3C", "lw": 1.2},
        )

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Performance Gap: Synthetic vs Real Data by Class", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "gap_sintetico_vs_real.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3h — Cohen's d: CAAFE vs LLM Features
# ══════════════════════════════════════════════════════════════════════════════
def fig_3h_cohens_d():
    print("3h: Cohen's d CAAFE vs LLM")
    features = [
        ("tail_asymmetry", 0.84, "CAAFE"),
        ("cond_entropy_X0_mask", 0.39, "CAAFE"),
        ("kurtosis_excess", 0.29, "CAAFE"),
        ("missing_rate_by_quantile", 0.15, "CAAFE"),
        ("llm_mar_conf", 0.39, "LLM"),
        ("llm_judge_mnar_prob", 0.25, "LLM"),
        ("llm_judge_structured", 0.20, "LLM"),
        ("llm_second_order (avg)", 0.12, "LLM"),
    ]

    features.sort(key=lambda x: x[1], reverse=True)
    names = [f[0] for f in features]
    values = [f[1] for f in features]
    types = [f[2] for f in features]
    colors = ["#3498DB" if t == "CAAFE" else "#E74C3C" for t in types]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names)), values, color=colors, edgecolor="white")

    # Reference lines
    for val, label, ls in [(0.2, "Small", ":"), (0.5, "Medium", "--"), (0.8, "Large", "-.")]:
        ax.axvline(val, color="#95A5A6", linestyle=ls, lw=1, alpha=0.7)
        ax.text(val, len(names) - 0.3, label, fontsize=7, color="#95A5A6", ha="center")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Cohen's d (MAR vs MNAR discrimination)")
    ax.invert_yaxis()

    handles = [
        mpatches.Patch(color="#3498DB", label="CAAFE (pure Python)"),
        mpatches.Patch(color="#E74C3C", label="LLM (API calls)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    ax.set_title("Effect Size: CAAFE vs LLM Features", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "cohens_d_caafe_vs_llm.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3i — NaiveBayes vs XGBoost LOGO CV (all models)
# ══════════════════════════════════════════════════════════════════════════════
def fig_3i_logo_cv():
    print("3i: LOGO CV por modelo (V3+ soft3zone)")
    csv_path = os.path.join(OUTPUT_DIR, "step05_pro", "real", "hierarquico_v3plus", "cv_logo_v3plus.csv")
    df = pd.read_csv(csv_path)

    models = df["modelo"].tolist()
    means = df["V3plus_soft3zone_cv_mean"].tolist()
    stds = df["V3plus_soft3zone_cv_std"].tolist()

    # Sort by mean descending
    order = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
    models = [models[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))

    # Highlight NB and XGBoost
    colors = []
    for m in models:
        if m == "NaiveBayes":
            colors.append("#27AE60")
        elif m == "XGBoost":
            colors.append("#E74C3C")
        else:
            colors.append("#3498DB")

    ax.bar(
        x,
        [m * 100 for m in means],
        width=0.7,
        color=colors,
        edgecolor="white",
        yerr=[s * 100 for s in stds],
        capsize=3,
        error_kw={"lw": 1, "color": "#555"},
    )

    for i, (m, s) in enumerate(zip(means, stds, strict=False)):
        ax.text(i, m * 100 + s * 100 + 1.5, f"{m*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LOGO CV Accuracy (%)")
    ax.set_ylim(0, 80)

    handles = [
        mpatches.Patch(color="#27AE60", label="NaiveBayes (best)"),
        mpatches.Patch(color="#E74C3C", label="XGBoost"),
        mpatches.Patch(color="#3498DB", label="Other models"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    ax.set_title("LOGO Cross-Validation Accuracy — V3+ Soft 3-Zone Routing", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "logo_cv_models.png")


# ══════════════════════════════════════════════════════════════════════════════
# Copy existing figures
# ══════════════════════════════════════════════════════════════════════════════
def copy_existing():
    print("\nCopiando figuras existentes:")
    copies = [
        (
            os.path.join(OUTPUT_DIR, "step05_pro", "real", "shap_analysis", "shap_l1_vs_l2_comparison.png"),
            "shap_l1_vs_l2.png",
        ),
        (os.path.join(OUTPUT_DIR, "step05_pro", "real", "shap_analysis", "shap_direto_3way.png"), "shap_3way.png"),
        (
            os.path.join(OUTPUT_DIR, "step05_pro", "real", "shap_analysis", "confusion_v3_hier.png"),
            "confusion_v3plus.png",
        ),
        (
            os.path.join(OUTPUT_DIR, "step05_pro", "real", "label_analysis", "quality_distribution.png"),
            "quality_distribution_existing.png",
        ),
    ]
    for src, dst in copies:
        dst_path = os.path.join(OUT_DIR, dst)
        if os.path.exists(src):
            shutil.copy2(src, dst_path)
            print(f"  ✓ {dst} (from {os.path.basename(src)})")
        else:
            print(f"  ✗ {dst} — source not found: {src}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Output: {OUT_DIR}\n")

    fig_3a_flowchart()
    fig_3b_confusion()
    fig_3d_evolution()
    fig_3e_label_quality()
    fig_3f_confident_joint()
    fig_3g_gap()
    fig_3h_cohens_d()
    fig_3i_logo_cv()
    copy_existing()

    pngs = [f for f in os.listdir(OUT_DIR) if f.endswith(".png")]
    print(f"\nTotal: {len(pngs)} figuras em {OUT_DIR}")


if __name__ == "__main__":
    main()
