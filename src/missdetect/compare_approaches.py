"""
Teste estatístico pareado LLM-vs-ML.

Compara a abordagem `apenas_ml` (baseline) com `ml_com_llm` reportando, por
modelo e por mecanismo (MCAR / MAR / MNAR), as estatísticas que o orientador
exigiu na reunião de 07/05/2026: a média sozinha não basta — precisa de
desvio padrão e teste de significância.

Testes aplicados:
    1. Wilcoxon pareado nos 5 fold scores de cada modelo (cv_scores.csv).
    2. t-test pareado complementar (ttest_rel) nos mesmos fold scores.
    3. McNemar amostra-a-amostra por (mecanismo, modelo) usando predictions.csv,
       com inner join por (sample_idx, modelo) para garantir o pareamento.
    4. Wilcoxon pareado global agregando todos os 35 fold scores
       (7 modelos × 5 folds).
    5. Tamanho de efeito: Cohen's d_z (paramétrico) e Cliff's delta
       (não-paramétrico).

Uso:
    python -m missdetect.compare_approaches \\
        --baseline-dir results/step12_ml_only_v2b_32datasets/real/apenas_ml/baseline \\
        --llm-dir       results/step12_pro_v2b_32datasets/real/ml_com_llm/gemini-3-pro-preview \\
        --out-dir       results/comparison_v2b_pro \\
        --label         "Pro vs Baseline"
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_rel, wilcoxon

ALPHA = 0.05
MECHANISMS = ("MCAR", "MAR", "MNAR")


@dataclass
class ApproachData:
    """Predições e CV scores de uma abordagem (baseline ou LLM)."""

    label: str
    cv_scores: pd.DataFrame  # cols: modelo, fold, score
    predictions: pd.DataFrame  # cols: sample_idx, y_true, y_pred, modelo, group, ...
    models: list[str] = field(default_factory=list)


def load_approach(directory: str, label: str) -> ApproachData:
    """Carrega cv_scores.csv + predictions.csv de um diretório de resultado."""
    cv_path = os.path.join(directory, "cv_scores.csv")
    pred_path = os.path.join(directory, "predictions.csv")
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"cv_scores.csv não encontrado em {directory}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"predictions.csv não encontrado em {directory}")
    cv = pd.read_csv(cv_path)
    pred = pd.read_csv(pred_path)
    models = sorted(cv["modelo"].unique())
    return ApproachData(label=label, cv_scores=cv, predictions=pred, models=models)


def cohens_dz(diffs: np.ndarray) -> float:
    """Cohen's d_z para amostras pareadas: mean(diffs) / std(diffs).

    Retorna NaN quando o desvio é numericamente desprezível — caso contrário
    flutuações de ~1e-17 produziriam d_z astronômico em vez de "sem diferença".
    """
    if len(diffs) < 2:
        return float("nan")
    sd = float(np.std(diffs, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(diffs)) / sd


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta pareado: (#wins - #losses) / n_pairs.

    Aqui usamos a versão pareada — a e b têm mesmo tamanho e estão alinhados.
    Empates não contam como vitória nem derrota.
    """
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    diff = np.asarray(b) - np.asarray(a)
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    return (wins - losses) / len(a)


def significance_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return ""


def per_model_tests(baseline: ApproachData, llm: ApproachData) -> pd.DataFrame:
    """Wilcoxon + ttest_rel + tamanho de efeito por modelo, sobre os fold scores."""
    common_models = sorted(set(baseline.models) & set(llm.models))
    rows: list[dict] = []
    for model in common_models:
        a = baseline.cv_scores[baseline.cv_scores["modelo"] == model].sort_values("fold")["score"].to_numpy()
        b = llm.cv_scores[llm.cv_scores["modelo"] == model].sort_values("fold")["score"].to_numpy()
        if len(a) != len(b):
            raise ValueError(
                f"Número de folds diferente para {model}: baseline={len(a)} vs llm={len(b)}",
            )
        diffs = b - a
        if np.allclose(diffs, 0):
            w_stat, w_p = 0.0, 1.0
            t_stat, t_p = 0.0, 1.0
        else:
            try:
                w_stat, w_p = wilcoxon(a, b, zero_method="pratt")
            except ValueError:
                w_stat, w_p = 0.0, 1.0
            t_stat, t_p = ttest_rel(b, a)
        rows.append(
            {
                "modelo": model,
                "n_folds": len(a),
                "baseline_mean": float(np.mean(a)),
                "baseline_std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                "llm_mean": float(np.mean(b)),
                "llm_std": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
                "mean_delta": float(np.mean(diffs)),
                "wilcoxon_W": float(w_stat),
                "wilcoxon_p": float(w_p),
                "ttest_t": float(t_stat),
                "ttest_p": float(t_p),
                "cohen_dz": cohens_dz(diffs),
                "cliff_delta": cliffs_delta(a, b),
                "significant_005": bool(w_p < ALPHA),
                "marker": significance_marker(float(w_p)),
            },
        )
    return pd.DataFrame(rows)


def _mechanism_coverage(baseline: ApproachData, llm: ApproachData) -> pd.DataFrame:
    """Contagem de amostras por mecanismo em cada lado e após o inner join.

    O `sample_idx` em alguns runs é regenerado por dataset, então o inner join
    pode encontrar overlap=0 para grupos onde a indexação divergiu — isso
    precisa ser reportado para que o leitor saiba que o McNemar é parcial.
    """
    base = baseline.predictions.assign(mech=baseline.predictions["group"].str.split("_", n=1).str[0])
    extra = llm.predictions.assign(mech=llm.predictions["group"].str.split("_", n=1).str[0])
    merged = base.merge(
        extra.drop(columns=["mech"]),
        on=["sample_idx", "modelo", "group"],
        suffixes=("_base", "_llm"),
    )
    rows = []
    for mech in MECHANISMS:
        n_base = int((base["mech"] == mech).sum())
        n_llm = int((extra["mech"] == mech).sum())
        n_paired = int((merged["group"].str.startswith(mech + "_")).sum())
        rows.append(
            {
                "mecanismo": mech,
                "n_baseline": n_base,
                "n_llm": n_llm,
                "n_paired": n_paired,
                "coverage": n_paired / max(min(n_base, n_llm), 1),
            },
        )
    return pd.DataFrame(rows)


def per_mechanism_mcnemar(baseline: ApproachData, llm: ApproachData) -> pd.DataFrame:
    """McNemar pareado amostra-a-amostra por (mecanismo, modelo).

    Faz inner join em (sample_idx, modelo, group) — sample_idx sozinho não é
    único entre datasets, então o group faz parte da chave. Amostras presentes
    em apenas uma das abordagens (p.ex. LLM perdeu uma extração) são descartadas
    pelo inner join.
    """
    join_cols = ["sample_idx", "modelo", "group"]
    merged = baseline.predictions.merge(
        llm.predictions,
        on=join_cols,
        suffixes=("_base", "_llm"),
    )
    if merged.empty:
        return pd.DataFrame()

    if not (merged["y_true_base"] == merged["y_true_llm"]).all():
        mismatched = int((merged["y_true_base"] != merged["y_true_llm"]).sum())
        raise ValueError(f"y_true diverge em {mismatched} amostras após join — dados inconsistentes")

    merged["mechanism"] = merged["group"].str.split("_", n=1).str[0]
    rows: list[dict] = []
    for mech in MECHANISMS:
        for model in sorted(merged["modelo"].unique()):
            sub = merged[(merged["mechanism"] == mech) & (merged["modelo"] == model)]
            if sub.empty:
                continue
            y_true = sub["y_true_base"].to_numpy()
            correct_base = sub["y_pred_base"].to_numpy() == y_true
            correct_llm = sub["y_pred_llm"].to_numpy() == y_true
            b_only = int(np.sum(correct_base & ~correct_llm))
            c_only = int(np.sum(~correct_base & correct_llm))
            n = len(sub)
            if b_only + c_only > 0:
                chi2_stat = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
                p = float(1 - chi2.cdf(chi2_stat, df=1))
            else:
                chi2_stat, p = 0.0, 1.0
            rows.append(
                {
                    "mecanismo": mech,
                    "modelo": model,
                    "n_amostras": n,
                    "acuracia_baseline": float(correct_base.mean()),
                    "acuracia_llm": float(correct_llm.mean()),
                    "delta": float(correct_llm.mean() - correct_base.mean()),
                    "b_baseline_acerta_llm_erra": b_only,
                    "c_baseline_erra_llm_acerta": c_only,
                    "mcnemar_chi2": float(chi2_stat),
                    "mcnemar_p": p,
                    "significant_005": bool(p < ALPHA),
                    "marker": significance_marker(p),
                },
            )
    return pd.DataFrame(rows)


def overall_test(baseline: ApproachData, llm: ApproachData) -> dict:
    """Wilcoxon + ttest_rel pareados em todos os fold scores (7 modelos × 5 folds)."""
    common_models = sorted(set(baseline.models) & set(llm.models))
    a_all: list[float] = []
    b_all: list[float] = []
    for model in common_models:
        a = baseline.cv_scores[baseline.cv_scores["modelo"] == model].sort_values("fold")["score"].to_numpy()
        b = llm.cv_scores[llm.cv_scores["modelo"] == model].sort_values("fold")["score"].to_numpy()
        if len(a) != len(b):
            continue
        a_all.extend(a.tolist())
        b_all.extend(b.tolist())
    a_arr = np.asarray(a_all)
    b_arr = np.asarray(b_all)
    diffs = b_arr - a_arr
    if len(diffs) < 2 or np.allclose(diffs, 0):
        return {
            "n_pairs": len(diffs),
            "baseline_mean": float(np.mean(a_arr)) if len(a_arr) else float("nan"),
            "baseline_std": float(np.std(a_arr, ddof=1)) if len(a_arr) > 1 else 0.0,
            "llm_mean": float(np.mean(b_arr)) if len(b_arr) else float("nan"),
            "llm_std": float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 0.0,
            "mean_delta": float(np.mean(diffs)) if len(diffs) else 0.0,
            "wilcoxon_W": 0.0,
            "wilcoxon_p": 1.0,
            "ttest_t": 0.0,
            "ttest_p": 1.0,
            "cohen_dz": float("nan"),
            "cliff_delta": float("nan"),
            "significant_005": False,
        }
    try:
        w_stat, w_p = wilcoxon(a_arr, b_arr, zero_method="pratt")
    except ValueError:
        w_stat, w_p = 0.0, 1.0
    t_stat, t_p = ttest_rel(b_arr, a_arr)
    return {
        "n_pairs": len(diffs),
        "baseline_mean": float(np.mean(a_arr)),
        "baseline_std": float(np.std(a_arr, ddof=1)),
        "llm_mean": float(np.mean(b_arr)),
        "llm_std": float(np.std(b_arr, ddof=1)),
        "mean_delta": float(np.mean(diffs)),
        "wilcoxon_W": float(w_stat),
        "wilcoxon_p": float(w_p),
        "ttest_t": float(t_stat),
        "ttest_p": float(t_p),
        "cohen_dz": cohens_dz(diffs),
        "cliff_delta": cliffs_delta(a_arr, b_arr),
        "significant_005": bool(w_p < ALPHA),
    }


def plot_paired_boxplot(per_model: pd.DataFrame, out_path: str, label: str) -> None:
    """Boxplot lado a lado: baseline vs LLM por modelo."""
    if per_model.empty:
        return
    _, ax = plt.subplots(figsize=(12, 6))
    models = per_model["modelo"].tolist()
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        per_model["baseline_mean"],
        width,
        yerr=per_model["baseline_std"],
        label="Baseline (apenas ML)",
        color="#4C72B0",
        capsize=4,
    )
    bars2 = ax.bar(
        x + width / 2,
        per_model["llm_mean"],
        width,
        yerr=per_model["llm_std"],
        label="ML + LLM",
        color="#DD8452",
        capsize=4,
    )
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    for i, marker in enumerate(per_model["marker"]):
        if marker:
            top = max(per_model["baseline_mean"].iloc[i], per_model["llm_mean"].iloc[i])
            ax.text(i, top + 0.05, marker, ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Acurácia (média ± desvio padrão dos folds)")
    ax.set_title(f"Comparação pareada por modelo — {label}")
    ax.set_ylim([0, 1])
    ax.axhline(y=1 / 3, color="red", linestyle="--", alpha=0.4, label="Acaso (1/3)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_report(
    out_path: str,
    label: str,
    baseline_dir: str,
    llm_dir: str,
    per_model: pd.DataFrame,
    per_mech: pd.DataFrame,
    overall: dict,
    coverage: pd.DataFrame | None = None,
) -> None:
    """Escreve relatório em PT-BR."""
    lines = []
    lines.append("=" * 78)
    lines.append(f"COMPARAÇÃO PAREADA: {label}")
    lines.append(f"Gerado em: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Baseline (apenas ML): {baseline_dir}")
    lines.append(f"ML + LLM:             {llm_dir}")
    lines.append("")
    lines.append(f"Nível de significância: alpha = {ALPHA}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("1. POR MODELO — Wilcoxon e t-test pareados nos fold scores")
    lines.append("-" * 78)
    if per_model.empty:
        lines.append("(sem dados)")
    else:
        header = f"{'Modelo':<22}{'baseline':>14}{'llm':>14}{'Δ':>9}{'Wilcoxon p':>13}{'t p':>10}{'d_z':>8}{'sig':>5}"
        lines.append(header)
        for _, row in per_model.iterrows():
            lines.append(
                f"{row['modelo']:<22}"
                f"{row['baseline_mean']:.3f}±{row['baseline_std']:.3f}  "
                f"{row['llm_mean']:.3f}±{row['llm_std']:.3f}  "
                f"{row['mean_delta']:+.3f}  "
                f"{row['wilcoxon_p']:>11.4f}  "
                f"{row['ttest_p']:>8.4f}  "
                f"{row['cohen_dz']:>+6.2f}  "
                f"{row['marker']:>4}",
            )
    lines.append("")
    lines.append("-" * 78)
    lines.append("2. POR MECANISMO E MODELO — McNemar pareado em predições")
    lines.append("-" * 78)
    if coverage is not None and not coverage.empty:
        lines.append("Cobertura do pareamento (inner join por sample_idx + modelo + group):")
        for _, row in coverage.iterrows():
            warn = "  ⚠ overlap baixo" if row["coverage"] < 0.5 else ""
            lines.append(
                f"  {row['mecanismo']:<6} baseline={int(row['n_baseline']):>5} "
                f"llm={int(row['n_llm']):>5} pareadas={int(row['n_paired']):>5} "
                f"({100 * row['coverage']:>5.1f}%){warn}",
            )
        lines.append("")
        lines.append("Nota: sample_idx pode ser regenerado entre runs. Mecanismos com")
        lines.append("overlap baixo aparecem com poucas (ou nenhuma) linha abaixo — isso")
        lines.append("não significa que o LLM falhou, apenas que o McNemar amostra-a-amostra")
        lines.append("não pode ser computado nesses casos. O Wilcoxon nos fold scores")
        lines.append("(seção 1) é o teste principal e não sofre dessa limitação.")
        lines.append("")
    if per_mech.empty:
        lines.append("(sem dados)")
    else:
        for mech in MECHANISMS:
            sub = per_mech[per_mech["mecanismo"] == mech]
            if sub.empty:
                continue
            lines.append(f"\n  [{mech}]  n={int(sub['n_amostras'].iloc[0])} amostras")
            lines.append(
                f"    {'Modelo':<22}{'acc base':>10}{'acc llm':>10}{'Δ':>8}"
                f"{'b':>5}{'c':>5}{'χ²':>8}{'p':>10}{'sig':>5}",
            )
            for _, row in sub.iterrows():
                lines.append(
                    f"    {row['modelo']:<22}"
                    f"{row['acuracia_baseline']:>9.3f}"
                    f"{row['acuracia_llm']:>10.3f}"
                    f"{row['delta']:>+8.3f}"
                    f"{row['b_baseline_acerta_llm_erra']:>5d}"
                    f"{row['c_baseline_erra_llm_acerta']:>5d}"
                    f"{row['mcnemar_chi2']:>8.2f}"
                    f"{row['mcnemar_p']:>10.4f}"
                    f"{row['marker']:>5}",
                )
    lines.append("")
    lines.append("-" * 78)
    lines.append("3. GLOBAL — Wilcoxon pareado em todos os fold scores agregados")
    lines.append("-" * 78)
    lines.append(f"  n_pares                  : {overall['n_pairs']}")
    lines.append(f"  baseline (média ± dp)    : {overall['baseline_mean']:.4f} ± {overall['baseline_std']:.4f}")
    lines.append(f"  ml + llm (média ± dp)    : {overall['llm_mean']:.4f} ± {overall['llm_std']:.4f}")
    lines.append(f"  Δ médio (llm - baseline) : {overall['mean_delta']:+.4f}")
    lines.append(f"  Wilcoxon W               : {overall['wilcoxon_W']:.2f}")
    lines.append(f"  Wilcoxon p               : {overall['wilcoxon_p']:.6f}")
    lines.append(f"  t pareado                : {overall['ttest_t']:+.3f}")
    lines.append(f"  p (t pareado)            : {overall['ttest_p']:.6f}")
    lines.append(f"  Cohen's d_z              : {overall['cohen_dz']:+.3f}")
    lines.append(f"  Cliff's delta            : {overall['cliff_delta']:+.3f}")
    lines.append(f"  Significativo (α=0.05)?  : {'sim' if overall['significant_005'] else 'não'}")
    lines.append("")
    lines.append("Marcadores: * p<0.05  ** p<0.01  *** p<0.001")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(baseline_dir: str, llm_dir: str, out_dir: str, label: str) -> dict:
    """Executa a comparação completa e grava artefatos em out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    baseline = load_approach(baseline_dir, "baseline")
    llm = load_approach(llm_dir, "llm")

    per_model = per_model_tests(baseline, llm)
    per_mech = per_mechanism_mcnemar(baseline, llm)
    overall = overall_test(baseline, llm)
    coverage = _mechanism_coverage(baseline, llm)

    per_model.to_csv(os.path.join(out_dir, "per_model.csv"), index=False)
    per_mech.to_csv(os.path.join(out_dir, "per_mechanism.csv"), index=False)
    coverage.to_csv(os.path.join(out_dir, "mechanism_coverage.csv"), index=False)
    with open(os.path.join(out_dir, "overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    plot_paired_boxplot(per_model, os.path.join(out_dir, "comparacao_pareada.png"), label)
    write_report(
        os.path.join(out_dir, "relatorio_comparativo.txt"),
        label=label,
        baseline_dir=baseline_dir,
        llm_dir=llm_dir,
        per_model=per_model,
        per_mech=per_mech,
        overall=overall,
        coverage=coverage,
    )
    return {
        "per_model": per_model,
        "per_mechanism": per_mech,
        "overall": overall,
        "coverage": coverage,
        "out_dir": out_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teste estatístico pareado LLM-vs-ML")
    parser.add_argument("--baseline-dir", required=True, help="Diretório do baseline (apenas_ml/baseline)")
    parser.add_argument("--llm-dir", required=True, help="Diretório do LLM (ml_com_llm/<model>)")
    parser.add_argument("--out-dir", required=True, help="Diretório para salvar artefatos")
    parser.add_argument("--label", default="LLM vs Baseline", help="Rótulo da comparação para o relatório")
    return parser.parse_args()


def _main_print(result: dict, label: str) -> None:
    overall = result["overall"]
    per_model = result["per_model"]
    print("=" * 70)
    print(f"📊 COMPARAÇÃO PAREADA — {label}")
    print("=" * 70)
    print(
        f"\nGlobal: Δ={overall['mean_delta']:+.4f}  Wilcoxon p={overall['wilcoxon_p']:.4f}  d_z={overall['cohen_dz']:+.2f}"
    )
    print(f"\nPor modelo (n={len(per_model)}):")
    for _, row in per_model.iterrows():
        print(
            f"  {row['modelo']:<22} "
            f"baseline={row['baseline_mean']:.3f}±{row['baseline_std']:.3f}  "
            f"llm={row['llm_mean']:.3f}±{row['llm_std']:.3f}  "
            f"Δ={row['mean_delta']:+.3f}  p={row['wilcoxon_p']:.4f} {row['marker']}",
        )
    print(f"\n💾 Artefatos salvos em: {result['out_dir']}")


if __name__ == "__main__":
    args = parse_args()
    result = run(args.baseline_dir, args.llm_dir, args.out_dir, args.label)
    _main_print(result, args.label)
