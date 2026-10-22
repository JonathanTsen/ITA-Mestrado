"""D7: logreg × HGB conforme o tamanho do treino, no corpus sintético (CV agrupada por tabela-base).

Uso: uv run --frozen python scripts/revisao_banca/decisoes/d7_curva_aprendizado.py <diretório do corpus com features>
"""

import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from missdetect.rb.models import CLASSES, DirectClassifier, macro_recall
from missdetect.rb.splits import family_folds

FRACTIONS = (0.15, 0.4, 1.0)
LEARNERS = ("logreg", "hgb")


def load(run: str) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_parquet(f"{run}/features.parquet")
    oracle = pd.read_parquet(f"{run}/oracle.parquet")
    instances = pd.read_parquet(f"{run}/instances.parquet")
    data = features.merge(oracle[["sample_id", "mechanism", "generator"]], on="sample_id")
    data = data.merge(instances[["sample_id", "family_id"]], on="sample_id").sort_values("sample_id")
    cols = [c for c in data.columns if c[:3] in ("g0_", "g1_", "g2_", "g3_", "g4_")]
    return data.reset_index(drop=True), [c for c in cols if not c.startswith("g2_mann_whitney")]


def task(data, cols, learner, frac, rep, fold, train, test) -> dict:
    X = data[cols].to_numpy(float)
    y = data.mechanism.map({m: k for k, m in enumerate(CLASSES)}).to_numpy()
    fam = data.family_id
    rng = np.random.default_rng(1000 * rep + fold)
    families = np.unique(fam.iloc[train])
    keep = set(rng.choice(families, size=max(8, int(frac * len(families))), replace=False))
    tr = np.array([j for j in train if fam.iloc[j] in keep])
    with threadpool_limits(1):
        proba = DirectClassifier(learner, {}, 0).fit(X[tr], y[tr], fam.iloc[tr]).predict_proba(X[test])
    pred = proba.argmax(1)
    sym = (data.generator.iloc[test] == "symmetric").to_numpy()
    return {
        "learner": learner,
        "frac": frac,
        "rep": rep,
        "fold": fold,
        "n_train": len(tr),
        "macro_recall": macro_recall(y[test], pred, 3),
        "macro_recall_sym": macro_recall(y[test][sym], pred[sym], 3),
        "log_loss": float(-np.mean(np.log(np.clip(proba[np.arange(len(test)), y[test]], 1e-12, None)))),
    }


def main() -> None:
    data, cols = load(sys.argv[1])
    jobs = [
        (learner, frac, rep, fold, train, test)
        for rep in range(2)
        for fold, (train, test) in enumerate(family_folds(data.family_id, 5, seed=rep))
        for frac in FRACTIONS
        for learner in LEARNERS
    ]
    res = pd.DataFrame(Parallel(n_jobs=8)(delayed(task)(data, cols, *job) for job in jobs))
    summary = res.groupby(["frac", "learner"]).agg(
        n_train=("n_train", "mean"),
        macro_recall=("macro_recall", "mean"),
        sd=("macro_recall", "std"),
        macro_recall_sym=("macro_recall_sym", "mean"),
        log_loss=("log_loss", "mean"),
    )
    print(summary.round(3).to_string())
    wide = res.pivot_table(index=["frac", "rep", "fold"], columns="learner", values="macro_recall")
    diff = (wide.logreg - wide.hgb).groupby("frac")
    print(pd.DataFrame({"logreg_minus_hgb": diff.mean(), "logreg_wins": diff.apply(lambda v: int((v > 0).sum()))}))


if __name__ == "__main__":
    main()
