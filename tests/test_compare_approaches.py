"""Tests for compare_approaches.py — paired statistical tests LLM vs ML."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from missdetect.compare_approaches import (
    cliffs_delta,
    cohens_dz,
    load_approach,
    overall_test,
    per_mechanism_mcnemar,
    per_model_tests,
    run,
)


def _make_cv_scores(model_scores: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for model, scores in model_scores.items():
        for fold, score in enumerate(scores):
            rows.append({"modelo": model, "fold": fold, "score": score})
    return pd.DataFrame(rows)


def _make_predictions(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _write_pair(tmp_path, baseline_dir_name, llm_dir_name, baseline_data, llm_data):
    base_dir = tmp_path / baseline_dir_name
    llm_dir = tmp_path / llm_dir_name
    base_dir.mkdir()
    llm_dir.mkdir()
    baseline_data["cv_scores"].to_csv(base_dir / "cv_scores.csv", index=False)
    baseline_data["predictions"].to_csv(base_dir / "predictions.csv", index=False)
    llm_data["cv_scores"].to_csv(llm_dir / "cv_scores.csv", index=False)
    llm_data["predictions"].to_csv(llm_dir / "predictions.csv", index=False)
    return str(base_dir), str(llm_dir)


def test_cohens_dz_simple():
    diffs = np.array([0.1, 0.05, 0.08, 0.12, 0.07])
    dz = cohens_dz(diffs)
    expected = float(np.mean(diffs) / np.std(diffs, ddof=1))
    assert dz == pytest.approx(expected, rel=1e-6)


def test_cohens_dz_zero_std_returns_nan():
    diffs = np.array([0.1, 0.1, 0.1])
    assert np.isnan(cohens_dz(diffs))


def test_cliffs_delta_all_wins():
    a = np.array([0.5, 0.5, 0.5, 0.5])
    b = np.array([0.6, 0.6, 0.6, 0.6])
    assert cliffs_delta(a, b) == pytest.approx(1.0)


def test_cliffs_delta_all_losses():
    a = np.array([0.6, 0.6, 0.6, 0.6])
    b = np.array([0.5, 0.5, 0.5, 0.5])
    assert cliffs_delta(a, b) == pytest.approx(-1.0)


def test_cliffs_delta_ties_zero():
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([0.5, 0.5, 0.5])
    assert cliffs_delta(a, b) == pytest.approx(0.0)


def test_per_model_tests_detects_uniform_improvement():
    """When LLM beats baseline on every fold, Wilcoxon p must be marginal and Cliff's delta = 1."""
    base = _make_cv_scores(
        {
            "RandomForest": [0.50, 0.51, 0.49, 0.52, 0.50],
            "MLP": [0.45, 0.46, 0.44, 0.45, 0.46],
        },
    )
    # LLM consistentemente acima, mas com variação real nos deltas
    llm = _make_cv_scores(
        {
            "RandomForest": [0.62, 0.59, 0.60, 0.63, 0.55],
            "MLP": [0.58, 0.54, 0.52, 0.56, 0.55],
        },
    )
    from missdetect.compare_approaches import ApproachData

    a = ApproachData(label="b", cv_scores=base, predictions=pd.DataFrame(), models=["MLP", "RandomForest"])
    b = ApproachData(label="l", cv_scores=llm, predictions=pd.DataFrame(), models=["MLP", "RandomForest"])
    df = per_model_tests(a, b)
    assert len(df) == 2
    for _, row in df.iterrows():
        assert row["mean_delta"] > 0
        assert row["wilcoxon_p"] < 0.10  # 5 folds, sinal uniforme, fica perto de p ≈ 0.0625
        assert row["cohen_dz"] > 0
        assert row["cliff_delta"] == pytest.approx(1.0)


def test_per_model_tests_no_difference_marks_not_significant():
    base = _make_cv_scores({"RandomForest": [0.5, 0.6, 0.55, 0.52, 0.58]})
    llm = _make_cv_scores({"RandomForest": [0.5, 0.6, 0.55, 0.52, 0.58]})
    from missdetect.compare_approaches import ApproachData

    a = ApproachData(label="b", cv_scores=base, predictions=pd.DataFrame(), models=["RandomForest"])
    b = ApproachData(label="l", cv_scores=llm, predictions=pd.DataFrame(), models=["RandomForest"])
    df = per_model_tests(a, b)
    assert df["wilcoxon_p"].iloc[0] == 1.0
    assert df["mean_delta"].iloc[0] == 0.0
    assert df["significant_005"].iloc[0] is False or not bool(df["significant_005"].iloc[0])


def test_per_mechanism_mcnemar_aggregates_by_mechanism_prefix():
    """Group prefix MCAR/MAR/MNAR controls the mechanism bucket."""
    base_pred = _make_predictions(
        [{"sample_idx": i, "y_true": 0, "y_pred": 0, "modelo": "RF", "group": "MCAR_d1"} for i in range(10)]
        + [{"sample_idx": 100 + i, "y_true": 1, "y_pred": 0, "modelo": "RF", "group": "MAR_d2"} for i in range(10)],
    )
    llm_pred = _make_predictions(
        [{"sample_idx": i, "y_true": 0, "y_pred": 0, "modelo": "RF", "group": "MCAR_d1"} for i in range(10)]
        + [{"sample_idx": 100 + i, "y_true": 1, "y_pred": 1, "modelo": "RF", "group": "MAR_d2"} for i in range(10)],
    )
    from missdetect.compare_approaches import ApproachData

    a = ApproachData(label="b", cv_scores=pd.DataFrame(), predictions=base_pred, models=["RF"])
    b = ApproachData(label="l", cv_scores=pd.DataFrame(), predictions=llm_pred, models=["RF"])
    df = per_mechanism_mcnemar(a, b)
    mcar = df[df["mecanismo"] == "MCAR"].iloc[0]
    mar = df[df["mecanismo"] == "MAR"].iloc[0]
    assert mcar["acuracia_baseline"] == pytest.approx(1.0)
    assert mcar["acuracia_llm"] == pytest.approx(1.0)
    assert mcar["mcnemar_chi2"] == 0.0
    assert mar["acuracia_baseline"] == pytest.approx(0.0)
    assert mar["acuracia_llm"] == pytest.approx(1.0)
    assert mar["mcnemar_chi2"] > 0
    assert mar["c_baseline_erra_llm_acerta"] == 10


def test_per_mechanism_mcnemar_inner_join_drops_unaligned_samples():
    """Sample present only in baseline must be ignored (not crash)."""
    base_pred = _make_predictions(
        [
            {"sample_idx": 1, "y_true": 0, "y_pred": 0, "modelo": "RF", "group": "MCAR_d1"},
            {"sample_idx": 2, "y_true": 0, "y_pred": 1, "modelo": "RF", "group": "MCAR_d1"},
        ],
    )
    llm_pred = _make_predictions(
        [{"sample_idx": 1, "y_true": 0, "y_pred": 0, "modelo": "RF", "group": "MCAR_d1"}],
    )
    from missdetect.compare_approaches import ApproachData

    a = ApproachData(label="b", cv_scores=pd.DataFrame(), predictions=base_pred, models=["RF"])
    b = ApproachData(label="l", cv_scores=pd.DataFrame(), predictions=llm_pred, models=["RF"])
    df = per_mechanism_mcnemar(a, b)
    assert len(df) == 1
    assert int(df["n_amostras"].iloc[0]) == 1


def test_per_mechanism_mcnemar_raises_on_y_true_mismatch():
    """Inconsistent y_true across approaches indicates corrupt data — must raise."""
    base_pred = _make_predictions(
        [{"sample_idx": 1, "y_true": 0, "y_pred": 0, "modelo": "RF", "group": "MCAR_d1"}],
    )
    llm_pred = _make_predictions(
        [{"sample_idx": 1, "y_true": 1, "y_pred": 1, "modelo": "RF", "group": "MCAR_d1"}],
    )
    from missdetect.compare_approaches import ApproachData

    a = ApproachData(label="b", cv_scores=pd.DataFrame(), predictions=base_pred, models=["RF"])
    b = ApproachData(label="l", cv_scores=pd.DataFrame(), predictions=llm_pred, models=["RF"])
    with pytest.raises(ValueError, match="y_true diverge"):
        per_mechanism_mcnemar(a, b)


def test_overall_test_aggregates_all_folds():
    base = _make_cv_scores(
        {
            "RF": [0.5, 0.5, 0.5, 0.5, 0.5],
            "MLP": [0.4, 0.4, 0.4, 0.4, 0.4],
        },
    )
    llm = _make_cv_scores(
        {
            "RF": [0.6, 0.6, 0.6, 0.6, 0.6],
            "MLP": [0.5, 0.5, 0.5, 0.5, 0.5],
        },
    )
    from missdetect.compare_approaches import ApproachData

    a = ApproachData(label="b", cv_scores=base, predictions=pd.DataFrame(), models=["MLP", "RF"])
    b = ApproachData(label="l", cv_scores=llm, predictions=pd.DataFrame(), models=["MLP", "RF"])
    out = overall_test(a, b)
    assert out["n_pairs"] == 10
    assert out["mean_delta"] == pytest.approx(0.10)
    assert out["wilcoxon_p"] < 0.01  # 10 pares com sinal uniforme bate < 0.01
    assert out["significant_005"] is True


def test_run_writes_all_artifacts(tmp_path):
    base_cv = _make_cv_scores(
        {"RF": [0.5, 0.51, 0.49, 0.52, 0.50], "MLP": [0.45, 0.46, 0.44, 0.45, 0.46]},
    )
    llm_cv = _make_cv_scores(
        {"RF": [0.60, 0.61, 0.59, 0.62, 0.60], "MLP": [0.55, 0.56, 0.54, 0.55, 0.56]},
    )
    base_pred = _make_predictions(
        [
            {"sample_idx": i, "y_true": 0, "y_pred": 0, "modelo": m, "group": "MCAR_d1"}
            for i in range(5)
            for m in ["RF", "MLP"]
        ],
    )
    llm_pred = _make_predictions(
        [
            {"sample_idx": i, "y_true": 0, "y_pred": 0, "modelo": m, "group": "MCAR_d1"}
            for i in range(5)
            for m in ["RF", "MLP"]
        ],
    )
    base_dir, llm_dir = _write_pair(
        tmp_path,
        "base",
        "llm",
        {"cv_scores": base_cv, "predictions": base_pred},
        {"cv_scores": llm_cv, "predictions": llm_pred},
    )
    out_dir = str(tmp_path / "out")
    result = run(base_dir, llm_dir, out_dir, label="Test")
    assert os.path.exists(os.path.join(out_dir, "per_model.csv"))
    assert os.path.exists(os.path.join(out_dir, "per_mechanism.csv"))
    assert os.path.exists(os.path.join(out_dir, "overall.json"))
    assert os.path.exists(os.path.join(out_dir, "relatorio_comparativo.txt"))
    assert os.path.exists(os.path.join(out_dir, "comparacao_pareada.png"))
    with open(os.path.join(out_dir, "overall.json"), encoding="utf-8") as f:
        overall = json.load(f)
    assert overall["mean_delta"] == pytest.approx(0.10, abs=1e-9)
    assert "per_model" in result and len(result["per_model"]) == 2


def test_load_approach_missing_file_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="cv_scores"):
        load_approach(str(d), "x")
