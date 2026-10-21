"""Testes do MechDetect oficial (passo 36): regra de decisão, equivalência com o código dos autores e
reprodução dos números impressos pelo notebook `accuracy_analysis.ipynb`."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from missdetect.rb.baselines.mechdetect_official import (
    decide_cube,
    load_cube,
    mechdetect_aucs,
    mechdetect_decide,
    summarize,
)

MECHDETECT_REPO = Path(os.environ.get("MECHDETECT_REPO", Path.home() / "work" / "MechDetect"))
CUBE = MECHDETECT_REPO / "src" / "auc_roc_scores" / "data" / "auc_roc_tensors_dict.npz"
OFFICIAL_TESTING = MECHDETECT_REPO / "src" / "auc_roc_scores" / "testing.py"


def _aucs(center: float, seed: int) -> np.ndarray:
    return center + np.random.default_rng(seed).normal(0.0, 0.01, size=10)


def test_decide_mcar_when_complete_equals_shuffled():
    assert mechdetect_decide(_aucs(0.5, 1), _aucs(0.5, 2), _aucs(0.5, 3)) == "MCAR"


def test_decide_mar_when_column_adds_nothing():
    assert mechdetect_decide(_aucs(0.8, 1), _aucs(0.5, 2), _aucs(0.8, 3)) == "MAR"


def test_decide_mnar_when_column_improves_its_own_mask():
    assert mechdetect_decide(_aucs(0.9, 1), _aucs(0.5, 2), _aucs(0.6, 3)) == "MNAR"


def test_decide_uses_bonferroni_half_alpha():
    # p unilateral entre alpha/2 e alpha: rejeitaria a 0,05, mas não a 0,025 (Bonferroni do notebook).
    complete = np.arange(10, 20, dtype=float)
    shuffled = np.arange(7, 17, dtype=float)
    p = stats.mannwhitneyu(complete, shuffled, alternative="greater").pvalue
    assert 0.025 <= p < 0.05
    assert mechdetect_decide(complete, shuffled, complete) == "MCAR"


@pytest.mark.skipif(not CUBE.exists(), reason="cubo oficial do MechDetect não disponível")
def test_reproduces_notebook_printed_accuracies():
    summary = summarize(decide_cube(load_cube(CUBE)))
    assert summary["n_datasets"] == 101
    assert round(summary["clean_rate_0.5"] * 100, 2) == 89.04  # "Avg. accuracy on with 0.5 error rate: 89.04%"
    assert round(summary["dirty_all_rates"], 4) == 0.5438  # "Accuracy on dirty data: 0.5438"


@pytest.mark.skipif(not OFFICIAL_TESTING.exists(), reason="repositório oficial do MechDetect não disponível")
def test_aucs_match_official_train_and_evaluate_models():
    spec = importlib.util.spec_from_file_location("mechdetect_testing_oficial", OFFICIAL_TESTING)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    rng = np.random.default_rng(7)
    n = 400
    data = pd.DataFrame({"y": rng.normal(size=n), "z1": rng.normal(size=n), "z2": rng.normal(size=n)})
    data["y"] += 0.8 * data["z1"]
    mask = pd.Series((data["y"] > data["y"].quantile(0.6)).astype(int), name="y")

    ours = mechdetect_aucs(data, mask, "y", seed=1234)
    theirs = official.train_and_evaluate_models(
        data=data.copy(),
        target=mask,
        column="y",
        model=official.HistGradientBoostingClassifier(max_iter=200),
        seed=1234,
        cv_folds=10,
    )
    np.testing.assert_allclose(ours, theirs)
    assert ours.shape == (4, 10)
    assert ours[0].mean() > ours[1].mean()  # a máscara depende de y: complete > shuffled
