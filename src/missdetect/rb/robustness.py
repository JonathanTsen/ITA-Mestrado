"""Ensaios de robustez do passo 40 (exploratórios; pré-registro §6), a partir das predições gravadas (30/10/2026).

**Prior-shift** (item 5 do passo 40; ANA-080, ORA-16/22). O benchmark é balanceado; na prática, as prevalências não são.
- `expected_accuracy`: a acurácia esperada sob uma prevalência π é Σ_c π_c · recall_c. Os recalls são os do teste
  balanceado, por família (média das repetições); falha conta como erro.
- `saerens_adjust`: correção de priors de Saerens, Latinne e Decaestecker (2002) com a prevalência conhecida,
  p'(c | x) ∝ p(c | x) · π'_c / π_c, seguida da classe de maior probabilidade. Mostra quanto se recupera ajustando a
  decisão à prevalência.
- `prior_shift`: as duas decisões (`argmax` e `saerens`) em cenários nomeados e numa grade do simplex, com a média
  entre famílias e IC95% por bootstrap de famílias.
- `rare_mcar`: o L1 quando o MCAR é raro. Recall do MCAR e especificidade (não-MCAR chamado de não-MCAR) vêm do teste
  balanceado; a precisão do MCAR sob a prevalência π é π·rec / (π·rec + (1 − π)·(1 − esp)).

**Ausência também nas auxiliares** (item 6 do passo 40; ORA-17): `run_z_nan`, um ensaio limitado com NaN MCAR em Z
(ver a função).

Nenhum p-valor: tudo aqui é descritivo, e nenhuma frase de "limite" ou "teto" (passo 40, aceite).
"""

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from missdetect.rb import features as feat
from missdetect.rb.baselines.logit_simple import LOGIT_SIMPLE_FEATURES
from missdetect.rb.config import RunConfig
from missdetect.rb.evaluate import summarize_by_family
from missdetect.rb.ids import child_rng, child_seed
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.models import CLASSES, DirectClassifier
from missdetect.rb.pipeline import feature_columns, load_eval_frame
from missdetect.rb.runs import append_run_log, read_observed, write_tables_atomically
from missdetect.rb.splits import outer_splits
from missdetect.rb.stats import PRIMARY, bootstrap_hl_ci, bootstrap_mean_ci, hodges_lehmann

TRAIN_PRIOR = (1 / 3, 1 / 3, 1 / 3)
SCENARIOS: dict[str, tuple[float, float, float]] = {
    "balanceada": TRAIN_PRIOR,
    "realista (MCAR 10%, MAR 50%, MNAR 40%)": (0.10, 0.50, 0.40),
    "MCAR raro (5%)": (0.05, 0.475, 0.475),
    "MNAR raro (10%)": (0.45, 0.45, 0.10),
    "MAR dominante (60%)": (0.20, 0.60, 0.20),
}
MCAR_PREVALENCES = (0.05, 0.10, 0.20, 1 / 3)
METHODS = ("direct_hgb", "hier_hgb_fullprob", "logit_simple", "mechdetect_obs")


def _mean_ci(values: np.ndarray) -> dict:
    low, high = bootstrap_mean_ci(values)
    return {"mean": float(np.mean(values)), "ci_low": low, "ci_high": high}


def simplex_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    n = round(1 / step)
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


def _recalls(pred: pd.DataFrame) -> pd.DataFrame:
    """Recall por classe, por método, família e repetição (falha, y_pred = −1, conta como erro)."""
    hit = pred.y_pred.to_numpy() == pred.y_true.to_numpy()
    frame = pred.assign(hit=hit)
    table = frame.groupby(["method", "family_id", "repeat", "y_true"]).hit.mean().unstack("y_true")
    table.columns = [f"recall_{CLASSES[int(c)]}" for c in table.columns]
    return table.reset_index()


def expected_accuracy(pred: pd.DataFrame, prior: tuple[float, float, float]) -> pd.DataFrame:
    """Acurácia esperada sob a prevalência `prior`, por método e família (média das repetições)."""
    rec = _recalls(pred)
    weights = np.asarray(prior, dtype=float)
    rec["accuracy"] = rec[[f"recall_{c}" for c in CLASSES]].to_numpy() @ weights
    return rec.groupby(["method", "family_id"], as_index=False).accuracy.mean()


def saerens_adjust(proba: np.ndarray, train_prior: Iterable[float], new_prior: Iterable[float]) -> np.ndarray:
    ratio = np.asarray(tuple(new_prior), dtype=float) / np.asarray(tuple(train_prior), dtype=float)
    adjusted = np.asarray(proba, dtype=float) * ratio
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _saerens_pred(pred: pd.DataFrame, prior: tuple[float, float, float]) -> pd.DataFrame:
    proba = pred[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=float)
    adjusted = saerens_adjust(proba, TRAIN_PRIOR, prior).argmax(axis=1)
    return pred.assign(y_pred=np.where(pred.y_pred.to_numpy() < 0, -1, adjusted))  # falha continua falha


def prior_shift(
    pred: pd.DataFrame, methods: Iterable[str], scenarios: dict[str, tuple[float, float, float]]
) -> pd.DataFrame:  # fmt: skip
    pred = pred[pred.method.isin(tuple(methods))]
    rows = []
    for scenario, prior in scenarios.items():
        for decision, frame in (("argmax", pred), ("saerens", _saerens_pred(pred, prior))):
            acc = expected_accuracy(frame, prior)
            for method, g in acc.groupby("method"):
                rows.append({"method": method, "scenario": scenario, "prior_MCAR": prior[0], "prior_MAR": prior[1],
                             "prior_MNAR": prior[2], "decision": decision, "n_families": len(g),
                             **_mean_ci(g.accuracy.to_numpy())})  # fmt: skip
    return pd.DataFrame(rows)


def rare_mcar(pred: pd.DataFrame, methods: Iterable[str], prevalences: Iterable[float]) -> pd.DataFrame:
    """L1 com MCAR raro: recall e especificidade do teste balanceado (média por família) e a precisão sob π(MCAR)."""
    pred = pred[pred.method.isin(tuple(methods))]
    is_mcar = pred.y_true.to_numpy() == 0
    said_mcar = pred.y_pred.to_numpy() == 0
    frame = pred.assign(sens=np.where(is_mcar, said_mcar, np.nan), spec=np.where(~is_mcar, ~said_mcar, np.nan))
    per = frame.groupby(["method", "family_id", "repeat"])[["sens", "spec"]].mean().groupby(level=[0, 1]).mean()
    rows = []
    for method, g in per.groupby(level=0):
        sens, spec = float(g.sens.mean()), float(g.spec.mean())
        for pi in prevalences:
            denominator = pi * sens + (1 - pi) * (1 - spec)
            rows.append({"method": method, "mcar_prevalence": pi, "n_families": len(g), "recall_mcar": sens,
                         "specificity": spec, "precision_mcar": pi * sens / denominator if denominator else np.nan,
                         **{f"recall_mcar_{k}": v for k, v in _mean_ci(g.sens.to_numpy()).items() if k != "mean"}})  # fmt: skip
    return pd.DataFrame(rows)


def run_robustness(run_dir: Path, config: RunConfig) -> Path:
    """Etapa `robustness` (passo 40, exploratória): prior-shift e L1 com MCAR raro, a partir do `eval-symmetric`.

    Saídas em `<run>/robustness/`, gravadas juntas: `prior_shift.csv` (cenários nomeados), `prior_shift_grid.csv`
    (simplex com passo 0,1) e `rare_mcar.csv` (o marcador de conclusão)."""
    pred = pd.read_parquet(Path(run_dir) / "eval-symmetric" / "predictions.parquet")
    methods = tuple(m for m in METHODS if m in set(pred.method))
    grid = {f"grade {a:.1f}/{b:.1f}/{c:.1f}": (a, b, c) for a, b, c in simplex_grid(0.1)}
    out_dir = Path(run_dir) / "robustness"
    out_dir.mkdir(exist_ok=True)
    tables = {
        "prior_shift.csv": prior_shift(pred, methods, SCENARIOS),
        "prior_shift_grid.csv": prior_shift(pred, (PRIMARY,), grid),
        "rare_mcar.csv": rare_mcar(pred, methods, MCAR_PREVALENCES),
    }
    write_tables_atomically(out_dir, tables)
    append_run_log(Path(run_dir), config, "robustness", {"methods": list(methods), "outputs": list(tables)})
    return out_dir


# --- Item 6 do passo 40: ausência também nas auxiliares (ORA-17) ----------------------------------------------------

ZNAN_RATES = (0.05, 0.10)
ZNAN_CONTEXT_FRACTION = 0.2
ZNAN_METHODS = ("direct_hgb", "logit_simple")


def inject_z_nan(instance: ObservedInstance, rate: float, seed: int) -> ObservedInstance:
    """Apaga, MCAR, cada célula de Z com probabilidade `rate` (Y fica como está)."""
    holes = np.random.default_rng(seed).random(instance.z.shape) < rate
    return ObservedInstance(instance.sample_id, instance.y, instance.z.mask(holes))


def _znan_features(run_dir: Path, sample_id: str, context_id: str, rate: float, root_seed: int) -> dict:
    """Descritores da instância com NaN em Z; a semente depende só do contexto e da taxa (os três mecanismos de um
    contexto recebem os mesmos buracos)."""
    seed = child_seed(root_seed, "znan", context_id, rate)
    try:
        with threadpool_limits(1):
            row = {**feat.compute(inject_z_nan(read_observed(run_dir, sample_id), rate, seed)), "feature_status": "ok"}
    except Exception as exc:  # falha conta como erro, nunca some
        row = {"sample_id": sample_id, "feature_status": f"failed: {exc!r}"}
    return {**row, "znan_rate": rate}


def run_z_nan(
    run_dir: Path, config: RunConfig, n_jobs: int, rates: tuple[float, ...] = ZNAN_RATES,
    fraction: float = ZNAN_CONTEXT_FRACTION,
) -> Path:  # fmt: skip
    """Ensaio limitado com NaN MCAR em Z (passo 40, item 6; exploratório).

    Uma amostra de contextos do gerador principal (`child_rng(raiz, "znan_contexts")`) recebe 5% e 10% de NaN em Z; os
    descritores são recalculados (desvios de NaN em `features.py`). A configuração primária e a logística simples,
    treinadas com Z completo nas famílias de treino de cada fold (as mesmas partições), são aplicadas sem ajuste às
    instâncias com NaN das famílias de teste. A comparação é com as mesmas instâncias sem NaN (predições do
    `eval-symmetric`), por família: HL da mudança com IC95% por bootstrap, sem p-valor."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "robustness-znan"
    if out_dir.exists():
        return out_dir
    frame = load_eval_frame(run_dir)
    sym = frame[frame.generator == "symmetric"].reset_index(drop=True)
    contexts = np.array(sorted(sym.context_id.unique()))
    size = max(1, round(fraction * len(contexts)))
    chosen = set(child_rng(config.root_seed, "znan_contexts").choice(contexts, size=size, replace=False))
    sample = sym[sym.context_id.isin(chosen)]
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_znan_features)(run_dir, r.sample_id, r.context_id, rate, config.root_seed)
        for rate in rates for r in sample.itertuples()
    )  # fmt: skip
    znan = pd.DataFrame(rows).merge(sample[["sample_id", "family_id", "label"]], on="sample_id")
    ev = config.evaluate
    seed = child_seed(config.root_seed, "model")
    cols, simple = feature_columns(sym), list(LOGIT_SIMPLE_FEATURES)
    families = sym.family_id.reset_index(drop=True)
    preds = []
    for repeat, fold, train, test in outer_splits(families, config.root_seed, ev.n_splits, ev.n_repeats):
        test_families = set(families.iloc[test])
        target = znan[znan.family_id.isin(test_families)]
        if target.empty:
            continue
        for method in ZNAN_METHODS:
            use, learner = (cols, ev.learner) if method == PRIMARY else (simple, "logreg")
            with threadpool_limits(1):
                model = DirectClassifier(learner, {}, seed)
                model.fit(sym[use].to_numpy(float)[train], sym.label.to_numpy()[train], families.iloc[train])
                proba = model.predict_proba(target[use].to_numpy(float))
            failed = ~target.feature_status.eq("ok").to_numpy()
            proba[failed] = 1 / 3
            y_pred = np.where(failed, -1, proba.argmax(1))
            preds.append(pd.DataFrame({
                "method": [f"{method}_znan{round(100 * r)}" for r in target.znan_rate], "repeat": repeat,
                "fold": fold, "sample_id": target.sample_id.to_numpy(), "family_id": target.family_id.to_numpy(),
                "y_true": target.label.to_numpy(), "y_pred": y_pred,
                **{f"p_{c}": proba[:, k] for k, c in enumerate(CLASSES)},
            }))  # fmt: skip
    pred = pd.concat(preds, ignore_index=True)
    clean = pd.read_parquet(run_dir / "eval-symmetric" / "predictions.parquet")
    clean = clean[clean.method.isin(ZNAN_METHODS) & clean.sample_id.isin(set(sample.sample_id))]
    per_family, _ = summarize_by_family(pd.concat([pred, clean[pred.columns]], ignore_index=True))
    wide = per_family.pivot(index="family_id", columns="method", values="macro_recall")
    change = []
    for method in ZNAN_METHODS:
        for rate in rates:
            label = f"{method}_znan{round(100 * rate)}"
            pair = wide[[label, method]].dropna()
            d = (pair[label] - pair[method]).to_numpy()
            low, high = bootstrap_hl_ci(d) if len(d) > 1 else (np.nan, np.nan)
            change.append({"method": method, "znan_rate": rate, "n_families": len(d),
                           "macro_recall_clean": float(pair[method].mean()), "macro_recall_znan": float(pair[label].mean()),
                           "hodges_lehmann": hodges_lehmann(d) if len(d) else np.nan, "ci_low": low, "ci_high": high})  # fmt: skip
    for stale in run_dir.glob(".tmp-robustness-znan-*"):
        shutil.rmtree(stale)
    tmp = run_dir / f".tmp-robustness-znan-{os.getpid()}"
    tmp.mkdir()
    znan.to_parquet(tmp / "features.parquet", index=False)
    pred.to_parquet(tmp / "predictions.parquet", index=False)
    per_family.to_parquet(tmp / "per_family.parquet", index=False)
    pd.DataFrame(change).to_csv(tmp / "change.csv", index=False)
    tmp.rename(out_dir)
    append_run_log(run_dir, config, "robustness-znan", {"rates": list(rates), "fraction": fraction,
                                                         "n_contexts": len(chosen), "n_instances": len(sample),
                                                         "n_feature_failures": int((znan.feature_status != "ok").sum())})  # fmt: skip
    return out_dir
