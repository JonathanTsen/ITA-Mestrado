"""Relatório do piloto (passo 37): tempos, inelegíveis, checagens dos descritores, detectabilidade, auditoria de
atalhos, testes de L1, sd das diferenças pareadas por família e MDE.

MDE (aproximação normal para o Wilcoxon pareado, Holm com m = 4 hipóteses confirmatórias H1–H4, poder 0,8):
    MDE ≈ (z_{1−α/(2m)} + z_{0,8}) · sd_Δ / √N.
É uma aproximação: ignora a eficiência relativa do Wilcoxon (≈ 0,955, MDE ≈ 2% maior) e os quantis t. O valor para o N
do próprio piloto (8 famílias) é só ilustrativo: com N = 8, o menor p bilateral exato do Wilcoxon é 0,0078, pouco
abaixo de α/m = 0,0125 (só rejeita se as 8 diferenças tiverem o mesmo sinal), e a aproximação normal não vale.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from missdetect.rb.config import RunConfig
from missdetect.rb.models import CLASSES, macro_recall
from missdetect.rb.pipeline import feature_columns, is_oracle_panel, load_eval_frame

ALPHA = 0.05
N_HYPOTHESES = 6  # família principal do pré-registro v2 (Holm m = 6; §5.1)
POWER = 0.8
N_FAMILIES_PLANNED = (50,)  # N efetivo (adendo do freeze-v1)
SHORTCUT_LIMIT = 0.40


def mde(sd_delta: float, n_families: int, alpha: float = ALPHA, m: int = N_HYPOTHESES, power: float = POWER) -> float:
    z = stats.norm.ppf(1 - alpha / (2 * m)) + stats.norm.ppf(power)
    return float(z * sd_delta / np.sqrt(n_families))


def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def timing(run_dir: Path, frame: pd.DataFrame) -> dict:
    out: dict = {}
    inject = pd.read_parquet(run_dir / "timing_inject.parquet")
    per_family = frame.groupby("family_id").size()
    out["inject_seconds_per_instance"] = float(inject.seconds.sum() / per_family.sum())
    out["instances_per_family"] = {int(k): int(v) for k, v in per_family.items()}
    out["features_seconds_mean"] = float(frame.feature_seconds.mean())
    out["features_seconds_by_n_k"] = (
        frame.groupby(["meta_n", "meta_k"]).feature_seconds.mean().round(3).reset_index().to_dict("records")
    )
    for name in ("mdo", "mdoracle", "mddirty", "little"):
        if f"{name}_seconds" in frame:
            out[f"{name}_seconds_mean"] = float(frame[f"{name}_seconds"].mean())
            out[f"{name}_seconds_by_n"] = frame.groupby("meta_n")[f"{name}_seconds"].mean().round(3).to_dict()
    pklm_path = run_dir / "pklm.parquet"
    if pklm_path.exists():
        pklm = pd.read_parquet(pklm_path).merge(frame[["sample_id", "meta_n"]], on="sample_id")
        out["pklm_seconds_mean"] = float(pklm.seconds.mean())
        out["pklm_seconds_by_n"] = pklm.groupby("meta_n").seconds.mean().round(2).to_dict()
    per_instance = out["inject_seconds_per_instance"] + out["features_seconds_mean"]
    per_instance += sum(out.get(f"{n}_seconds_mean", 0.0) for n in ("mdo", "mdoracle", "mddirty", "little"))
    mean_per_family = float(per_family.mean())
    n_primary = float(frame[frame.generator == "symmetric"].groupby("family_id").size().mean())
    out["projection_core_hours"] = {
        str(n): {
            "inject_features_mechdetect_little": round(n * mean_per_family * per_instance / 3600, 2),
            "pklm_primary_generator": round(n * n_primary * out.get("pklm_seconds_mean", np.nan) / 3600, 2),
        }
        for n in N_FAMILIES_PLANNED
    }
    return out


def eligibility(run_dir: Path) -> dict:
    elig = pd.read_parquet(run_dir / "eligibility.parquet")
    tried = [
        {"family_id": r.family_id, **t} for r in elig.itertuples() for t in json.loads(r.tried)
    ]  # um alvo testado por linha
    tried = pd.DataFrame(tried)
    skipped = pd.read_parquet(run_dir / "skipped.parquet") if (run_dir / "skipped.parquet").exists() else None
    out = {
        "base_attempts": len(elig),
        "bases_accepted": int(elig.accepted.sum()),
        "targets_tried": len(tried),
        "targets_rejected_r2": int((~tried.r2_ok).sum()) if len(tried) else 0,
        "targets_rejected_spearman": int((~tried.spearman_ok).sum()) if len(tried) else 0,
        "targets_eligible": int(tried.eligible.sum()) if len(tried) else 0,
        "eligible_columns_by_family": elig.groupby("family_id").n_eligible_columns.max().astype(int).to_dict(),
        "levels_skipped": skipped.reason.str.extract(r"^(\w)=")[0].value_counts().to_dict()
        if skipped is not None and len(skipped)
        else {},
    }
    if len(tried):
        out["ineligible_fraction_by_family"] = (
            tried.groupby("family_id").eligible.apply(lambda s: 1 - s.mean()).round(3).to_dict()
        )
    return out


def feature_checks(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in feature_columns(frame):
        values = frame[col].astype(float)
        rho = (
            stats.spearmanr(values, frame.g0_missing_rate, nan_policy="omit").statistic
            if col != "g0_missing_rate"
            else 1.0
        )
        rows.append(
            {
                "feature": col,
                "variance": float(values.var()),
                "nan_fraction": float(values.isna().mean()),
                "abs_spearman_with_rate": float(abs(rho)),
                "flag": bool(
                    values.var() == 0 or values.isna().mean() > 0.05 or (col != "g0_missing_rate" and abs(rho) >= 0.95)
                ),
            }
        )
    return pd.DataFrame(rows)


def detectability(pred: pd.DataFrame, frame: pd.DataFrame, methods: list[str], factor: str) -> pd.DataFrame:
    """Macro-recall por nível do fator: média, sobre repetições, do macro-recall agregado no nível."""
    merged = pred[pred.method.isin(methods)].merge(frame[["sample_id", factor]], on="sample_id")
    rows = []
    for (method, level, repeat), g in merged.groupby(["method", factor, "repeat"]):
        rows.append({"method": method, factor: level, "repeat": repeat,
                     "macro_recall": macro_recall(g.y_true.to_numpy(), g.y_pred.to_numpy(), len(CLASSES))})  # fmt: skip
    table = pd.DataFrame(rows).groupby(["method", factor]).macro_recall.mean().unstack(factor)
    return table


def per_class_by_factor(pred: pd.DataFrame, frame: pd.DataFrame, method: str, factor: str) -> pd.DataFrame:
    merged = pred[pred.method == method].merge(frame[["sample_id", factor]], on="sample_id")
    merged["correct"] = merged.y_true == merged.y_pred
    return merged.groupby([factor, "y_true"]).correct.mean().unstack("y_true").rename(columns=dict(enumerate(CLASSES)))


def l1_tests(frame: pd.DataFrame, run_dir: Path, alpha: float) -> pd.DataFrame:
    """Sensibilidade, especificidade e acurácia balanceada (por família, depois média ± dp) dos testes de L1.

    Uma instância em que o teste falhou (p-valor ausente) conta como erro nas duas métricas: não detecta o não-MCAR e
    não confirma o MCAR. Little e PKLM usam o nível alpha; o teste 1 do MechDetect-O usa alpha/2, como no artigo.
    """
    data = frame[["sample_id", "family_id", "mechanism"]].copy()
    data["non_mcar"] = data.mechanism != "MCAR"
    sets = {}
    if "little_p" in frame:
        sets["little"] = data.assign(p=frame.little_p.to_numpy(), level=alpha)
    if "mdo_p_shuffled" in frame:
        sets["mechdetect_obs_test1"] = data.assign(p=frame.mdo_p_shuffled.to_numpy(), level=alpha / 2)
    pklm_path = run_dir / "pklm.parquet"
    if pklm_path.exists():
        pklm = pd.read_parquet(pklm_path)
        sets["pklm"] = data.merge(pklm[["sample_id", "pklm_p"]], on="sample_id").rename(columns={"pklm_p": "p"})
        sets["pklm"]["level"] = alpha
        # Little no mesmo subconjunto do PKLM, para a comparação pareada
        if "little" in sets:
            sub = sets["little"][sets["little"].sample_id.isin(set(pklm.sample_id))]
            sets["little_on_pklm_subset"] = sub
    rows = []
    for name, d in sets.items():
        failed = d.p.isna().to_numpy()
        reject = (d.p < d.level).to_numpy() & ~failed
        d = d.assign(hit=np.where(d.non_mcar, reject, ~reject & ~failed))
        per_family = []
        for family, g in d.groupby("family_id"):
            sens = g.loc[g.non_mcar, "hit"].mean()
            spec = g.loc[~g.non_mcar, "hit"].mean()
            per_family.append({"family_id": family, "sensitivity": sens, "specificity": spec,
                               "balanced_accuracy": (sens + spec) / 2})  # fmt: skip
        pf = pd.DataFrame(per_family)
        rows.append(
            {
                "test": name,
                "level": float(d.level.iloc[0]),
                "n_instances": len(d),
                "n_failed": int(failed.sum()),
                **{f"{m}_mean": pf[m].mean() for m in ("sensitivity", "specificity", "balanced_accuracy")},
                **{f"{m}_sd": pf[m].std(ddof=1) for m in ("sensitivity", "specificity", "balanced_accuracy")},
            }
        )
    return pd.DataFrame(rows)


def paired_deltas(per_family: pd.DataFrame, contrasts: list[tuple[str, str, str]]) -> pd.DataFrame:
    wide = per_family.pivot(index="family_id", columns="method", values="macro_recall")
    rows = []
    for label, a, b in contrasts:
        if a not in wide or b not in wide:
            continue
        delta = (wide[a] - wide[b]).dropna()
        sd = float(delta.std(ddof=1))
        rows.append(
            {
                "contrast": label, "a": a, "b": b, "n_families": len(delta),
                "mean_delta": float(delta.mean()), "sd_delta": sd,
                **{f"mde_N{n}": mde(sd, n) for n in (len(delta), *N_FAMILIES_PLANNED)},
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def run_report(run_dir: Path, config: RunConfig) -> Path:
    out_dir = run_dir / "report"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    out_dir.mkdir()
    frame_all = load_eval_frame(run_dir)
    primary_eval = run_dir / "eval-symmetric"
    pred = pd.read_parquet(primary_eval / "predictions.parquet")
    per_family = pd.read_parquet(primary_eval / "per_family.parquet")
    summary = pd.read_csv(primary_eval / "summary.csv")
    frame = frame_all[frame_all.generator == "symmetric"]
    learner = config.evaluate.learner

    report: dict = {"run": str(run_dir), "timing": timing(run_dir, frame_all), "eligibility": eligibility(run_dir)}
    checks = feature_checks(frame_all)
    checks.to_csv(out_dir / "feature_checks.csv", index=False)
    report["feature_checks_flagged"] = checks[checks.flag].feature.tolist()

    candidates = [f"direct_{learner}", *[f"hier_{learner}_{r}" for r in config.evaluate.routings]]
    for factor in ("beta", "rate", "n", "k", "direction"):
        detectability(pred, frame, [*candidates, "mechdetect_obs", "mechdetect_oracle"], factor).to_csv(
            out_dir / f"detectability_{factor}.csv"
        )
    per_class_by_factor(pred, frame, f"direct_{learner}", "beta").to_csv(
        out_dir / f"per_class_beta_direct_{learner}.csv"
    )

    nuisance = summary[summary.method == f"nuisance_{learner}"].iloc[0]
    report["shortcut_audit"] = {
        "macro_recall_mean": float(nuisance.macro_recall_mean),
        "macro_recall_sd": float(nuisance.macro_recall_sd),
        "limit": SHORTCUT_LIMIT,
        "passed": bool(nuisance.macro_recall_mean <= SHORTCUT_LIMIT),
    }
    all_eval = run_dir / "eval-all"
    if all_eval.exists():
        summary_all = pd.read_csv(all_eval / "summary.csv")
        row = summary_all[summary_all.method == f"nuisance_{learner}"].iloc[0]
        report["shortcut_audit_all_generators"] = {
            "macro_recall_mean": float(row.macro_recall_mean), "macro_recall_sd": float(row.macro_recall_sd),
            "passed": bool(row.macro_recall_mean <= SHORTCUT_LIMIT),
        }  # fmt: skip

    contrasts = [(f"H1 {c} × MechDetect-O", c, "mechdetect_obs") for c in candidates]
    contrasts += [
        (f"H2 hier_{r} × direct", f"hier_{learner}_{r}", f"direct_{learner}") for r in config.evaluate.routings
    ]
    contrasts += [(f"H3 {c} × uniform", c, "trivial_uniform") for c in candidates]
    deltas = paired_deltas(per_family, contrasts)
    deltas.to_csv(out_dir / "paired_deltas_mde.csv", index=False)
    l1 = l1_tests(frame, run_dir, ALPHA)
    l1.to_csv(out_dir / "l1_tests.csv", index=False)
    cols = ["method", "macro_recall_mean", "macro_recall_sd", "accuracy_mean", "log_loss_mean", "ece_classwise_mean"]
    cols += [f"recall_{c}_mean" for c in CLASSES]
    # Painel oráculo (vê o Y completo) em arquivo à parte: nunca é ranqueado com os métodos observed-only.
    oracle_rows = summary.method.map(is_oracle_panel)
    summary.loc[~oracle_rows, cols].to_csv(out_dir / "summary_primary.csv", index=False)
    summary.loc[oracle_rows, cols].to_csv(out_dir / "summary_oracle_panel.csv", index=False)

    beta_table = detectability(pred, frame, [f"direct_{learner}", "mechdetect_obs"], "beta")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    for method, row in beta_table.iterrows():
        labels = [str(b) for b in row.index]
        ax.plot(labels, row.to_numpy(), marker="o", label=method)
    ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=1, label="chance (1/3)")
    ax.set_xlabel("β (strength)")
    ax.set_ylabel("macro-recall (pooled over families)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "detectability_beta.png", dpi=150)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out_dir
