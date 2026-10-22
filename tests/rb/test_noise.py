"""Testes do passo 44 (RQ3): ruído de rótulo controlado só no treino, flags do Confident Learning e mitigações."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.evaluate import check_pairing, run_method
from missdetect.rb.models import DirectClassifier
from missdetect.rb.noise import (
    MITIGATIONS,
    TEST_RATE,
    cleanlab_scores,
    corrupt_labels,
    degradation_curves,
    fit_mitigated,
    flag_quality,
    flag_summary,
    noise_settings,
    overlap_flags,
    per_family_noise,
    rq3_tests,
    run_setting,
)

MCAR, MAR, MNAR = 0, 1, 2


def _trios(n_families: int = 4, contexts_per_family: int = 20):
    """Rótulos em trios (um contexto = as três instâncias MCAR, MAR e MNAR da mesma tabela, alvo e auxiliares)."""
    contexts, families, y = [], [], []
    for f in range(n_families):
        for c in range(contexts_per_family):
            for label in (MCAR, MAR, MNAR):
                contexts.append(f"f{f}c{c}")
                families.append(f)
                y.append(label)
    return np.array(y), pd.Series(contexts), pd.Series(families)


def test_rate_zero_keeps_every_label():
    y, contexts, families = _trios()
    for kind in ("symmetric", "mar_mnar"):
        noisy = corrupt_labels(y, contexts, families, 0.0, kind, np.random.default_rng(0))
        assert np.array_equal(noisy, y)


def test_input_labels_are_not_modified():
    y, contexts, families = _trios()
    before = y.copy()
    corrupt_labels(y, contexts, families, 0.4, "symmetric", np.random.default_rng(0))
    assert np.array_equal(y, before)


@pytest.mark.parametrize("rate", [0.1, 0.2, 0.4])
def test_symmetric_corrupts_whole_contexts_at_the_exact_rate_per_family(rate):
    y, contexts, families = _trios(contexts_per_family=20)
    noisy = corrupt_labels(y, contexts, families, rate, "symmetric", np.random.default_rng(1))
    changed = pd.Series(noisy != y).groupby(contexts).agg(["all", "any"])
    assert (changed["all"] == changed["any"]).all()  # as três instâncias de um contexto trocam juntas
    per_family = changed["all"].groupby(changed.index.str.split("c").str[0]).sum()
    assert (per_family == round(rate * 20)).all()


def test_symmetric_moves_to_each_other_class_with_equal_probability():
    y, contexts, families = _trios(n_families=10, contexts_per_family=100)
    noisy = corrupt_labels(y, contexts, families, 0.4, "symmetric", np.random.default_rng(2))
    hit = noisy != y
    shift = (noisy[hit] - y[hit]) % 3
    assert set(np.unique(shift)) == {1, 2}
    assert abs(np.mean(shift == 1) - 0.5) < 0.05


def test_mar_mnar_swaps_only_mar_and_mnar():
    y, contexts, families = _trios()
    noisy = corrupt_labels(y, contexts, families, 0.4, "mar_mnar", np.random.default_rng(3))
    assert np.array_equal(noisy[y == MCAR], y[y == MCAR])
    hit = noisy != y
    assert hit.any()
    assert set(zip(y[hit], noisy[hit], strict=True)) == {(MAR, MNAR), (MNAR, MAR)}
    swapped = pd.Series(hit).groupby(contexts).sum()
    assert set(swapped.unique()) <= {0, 2}  # o MAR e o MNAR do contexto trocam juntos


def test_corruption_is_reproducible_and_depends_on_the_seed():
    y, contexts, families = _trios()
    a = corrupt_labels(y, contexts, families, 0.2, "symmetric", np.random.default_rng(5))
    b = corrupt_labels(y, contexts, families, 0.2, "symmetric", np.random.default_rng(5))
    c = corrupt_labels(y, contexts, families, 0.2, "symmetric", np.random.default_rng(6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_unknown_kind_is_an_error():
    y, contexts, families = _trios()
    with pytest.raises(ValueError):
        corrupt_labels(y, contexts, families, 0.2, "other", np.random.default_rng(0))


def test_flag_quality_precision_recall_and_na_without_positives():
    families = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    corrupted = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    flagged = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    out = flag_quality(flagged, corrupted, families).set_index("family_id")
    assert out.loc[0, "precision"] == pytest.approx(0.5)
    assert out.loc[0, "recall"] == pytest.approx(0.5)
    assert out.loc[0, "flagged_fraction"] == pytest.approx(0.5)
    assert np.isnan(out.loc[1, "precision"])  # nada sinalizado
    assert np.isnan(out.loc[1, "recall"])  # nada corrompido


def _blobs(n_families: int = 8, per_family: int = 60, seed: int = 0, shift: float = 1.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(3, size=n_families * per_family)
    X = rng.normal(size=(len(y), 3)) + np.eye(3)[y] * shift
    families = pd.Series(np.repeat(np.arange(n_families), per_family))
    return X, y, families


def test_unit_weights_reproduce_the_unweighted_classifier():
    X, y, families = _blobs()
    a = DirectClassifier("hgb", {}, 0).fit(X, y, families).predict_proba(X[:50])
    b = DirectClassifier("hgb", {}, 0).fit(X, y, families, sample_weight=np.ones(len(y))).predict_proba(X[:50])
    np.testing.assert_allclose(a, b)


def test_weights_shift_the_classifier_towards_heavier_labels():
    X, y, families = _blobs()
    X_test = np.random.default_rng(9).normal(size=(300, 3))  # pontos ambíguos, longe dos centros
    base = DirectClassifier("hgb", {}, 0).fit(X, y, families).predict_proba(X_test)[:, MCAR].mean()
    weights = np.where(y == MCAR, 10.0, 1.0)
    heavy = DirectClassifier("hgb", {}, 0).fit(X, y, families, sample_weight=weights).predict_proba(X_test)[:, MCAR]
    assert heavy.mean() > base + 0.05


def _setting(rate: float, kind: str = "symmetric", seed: int = 0):
    X, y, families = _blobs(n_families=8, per_family=60, seed=seed)
    contexts = pd.Series([f"c{i // 3}" for i in range(len(y))])
    noisy = corrupt_labels(y, contexts, families, rate, kind, np.random.default_rng(seed))
    ids = pd.Series([f"s{i}" for i in range(len(y))])
    pred, flags = run_setting(
        X, y, noisy, families, ids, root_seed=3, learner="hgb", model_seed=0, n_splits=4, n_repeats=1
    )
    return X, y, noisy, families, ids, pred, flags


def test_cleanlab_flags_corrupted_labels_better_than_chance():
    X, y, families = _blobs(n_families=8, per_family=80, shift=3.0)  # classes bem separadas
    contexts = pd.Series([f"c{i // 3}" for i in range(len(y))])
    noisy = corrupt_labels(y, contexts, families, 0.2, "symmetric", np.random.default_rng(0))
    flagged, quality = cleanlab_scores(X, noisy, families, "hgb", 0)
    corrupted = noisy != y
    assert flagged.any()
    assert corrupted[flagged].mean() > 2 * corrupted.mean()
    assert quality[corrupted].mean() < quality[~corrupted].mean()


def test_test_labels_are_always_the_true_ones():
    _, y, noisy, _, ids, pred, _ = _setting(0.4)
    assert (noisy != y).any()
    truth = dict(zip(ids, y, strict=True))
    assert (pred.y_true == pred.sample_id.map(truth)).all()


def test_mitigations_are_paired_and_cover_every_instance_once_per_repeat():
    _, y, _, _, _, pred, _ = _setting(0.2)
    assert set(pred.mitigation) == set(MITIGATIONS)
    assert check_pairing(pred.assign(method=pred.mitigation))
    assert pred.groupby(["mitigation", "repeat"]).size().eq(len(y)).all()


def test_flags_cover_only_training_rows():
    _, _, _, _, _, pred, flags = _setting(0.2)
    for (repeat, fold), g in flags.groupby(["repeat", "fold"]):
        test_ids = set(pred[(pred.repeat == repeat) & (pred.fold == fold)].sample_id)
        assert not set(g.sample_id) & test_ids
    assert {"corrupted", "flagged", "quality"} <= set(flags.columns)


def test_clean_labels_without_mitigation_reproduce_the_primary_evaluation():
    X, y, _, families, ids, pred, _ = _setting(0.0)
    ref, _ = run_method("direct_hgb", lambda: DirectClassifier("hgb", {}, 0), X, y, families, ids, 3, 4, 1)
    mine = pred[pred.mitigation == "none"].sort_values(["repeat", "fold", "sample_id"]).reset_index(drop=True)
    ref = ref.sort_values(["repeat", "fold", "sample_id"]).reset_index(drop=True)
    np.testing.assert_allclose(mine[["p_MCAR", "p_MAR", "p_MNAR"]], ref[["p_MCAR", "p_MAR", "p_MNAR"]])


def test_prune_that_removes_a_whole_class_is_an_error():
    X, y, families = _blobs()
    flagged = y == MNAR
    with pytest.raises(ValueError, match="classe"):
        fit_mitigated("prune", X, y, families, "hgb", 0, flagged, np.ones(len(y)))


def test_without_corruption_flags_measure_class_overlap():
    """ORA-14: com rótulos corretos, a fração sinalizada cresce com a sobreposição entre as classes."""
    fractions = {}
    for shift in (1.5, 3.0):
        X, y, families = _blobs(n_families=8, per_family=80, shift=shift)
        flagged, _ = cleanlab_scores(X, y, families, "hgb", 0)
        fractions[shift] = flagged.mean()
    assert fractions[1.5] > 3 * fractions[3.0]


def test_settings_have_one_clean_reference_and_three_seeds_per_noisy_cell():
    settings = noise_settings()
    assert settings[0] == ("clean", 0.0, 0)
    noisy = settings[1:]
    assert len(noisy) == 2 * 3 * 3  # 2 tipos × 3 taxas > 0 × 3 sementes
    assert {r for _, r, _ in noisy} == {0.1, 0.2, 0.4}
    assert TEST_RATE == 0.2


def _noise_pred(effect: dict, n_families: int = 12, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Predições sintéticas: acerto por família ~ base da família + efeito da (tipo, taxa, mitigação)."""
    rng = np.random.default_rng(0)
    rows = []
    settings = [("clean", 0.0, 0)] + [
        (k, r, s) for k in ("symmetric", "mar_mnar") for r in (0.1, 0.2, 0.4) for s in seeds
    ]
    for kind, rate, seed in settings:
        for mitigation in MITIGATIONS:
            for f in range(n_families):
                p_hit = np.clip(0.7 + 0.02 * f - rate * 0.5 + effect.get((kind, mitigation), 0.0), 0, 1)
                for i in range(60):
                    y = i % 3
                    hit = rng.random() < p_hit
                    pred = y if hit else (y + 1) % 3
                    rows.append({"noise_kind": kind, "noise_rate": rate, "noise_seed": seed, "mitigation": mitigation,
                                 "repeat": 0, "fold": 0, "sample_id": f"f{f}s{i}", "family_id": f, "y_true": y,
                                 "y_pred": pred, "p_MCAR": float(pred == 0), "p_MAR": float(pred == 1),
                                 "p_MNAR": float(pred == 2), "status": "ok"})  # fmt: skip
    return pd.DataFrame(rows)


def test_per_family_averages_seeds_and_repeats():
    pred = _noise_pred({})
    per_family = per_family_noise(pred)
    cells = per_family.groupby(["noise_kind", "noise_rate", "mitigation"]).family_id.nunique()
    assert (cells == 12).all()
    assert per_family.groupby(["noise_kind", "noise_rate", "mitigation", "family_id"]).size().eq(1).all()


def test_rq3_tests_are_the_four_preregistered_contrasts_with_holm():
    pred = _noise_pred({("symmetric", "weights"): 0.15})
    tests = rq3_tests(per_family_noise(pred)).set_index("hypothesis")
    assert list(tests.index) == ["RQ3_weights_symmetric", "RQ3_prune_symmetric", "RQ3_weights_mar_mnar",
                                 "RQ3_prune_mar_mnar"]  # fmt: skip
    assert (tests.p_holm >= tests.p_value - 1e-12).all()
    assert tests.p_holm.min() == pytest.approx(min(4 * tests.p_value.min(), 1.0))
    assert tests.loc["RQ3_weights_symmetric", "decision"] == "superior"


def test_degradation_curves_start_from_the_shared_clean_point():
    curves = degradation_curves(per_family_noise(_noise_pred({})))
    for kind in ("symmetric", "mar_mnar"):
        c = curves[curves.noise_kind == kind]
        assert sorted(c.noise_rate.unique()) == [0.0, 0.1, 0.2, 0.4]
        assert {"macro_recall_mean", "ci_low", "ci_high", "n_families"} <= set(c.columns)
    clean = curves[curves.noise_rate == 0.0].set_index(["noise_kind", "mitigation"]).macro_recall_mean
    assert clean.loc["symmetric"].equals(clean.loc["mar_mnar"])


def test_flag_summary_reports_precision_recall_per_cell():
    flags = pd.DataFrame({
        "noise_kind": "symmetric", "noise_rate": 0.2, "noise_seed": 0, "repeat": 0, "fold": 0,
        "sample_id": [f"s{i}" for i in range(8)], "family_id": [0, 0, 0, 0, 1, 1, 1, 1],
        "corrupted": [1, 1, 0, 0, 1, 0, 0, 0], "flagged": [1, 0, 1, 0, 1, 0, 0, 0], "quality": 0.5,
    })  # fmt: skip
    flags[["corrupted", "flagged"]] = flags[["corrupted", "flagged"]].astype(bool)
    summary = flag_summary(flags).iloc[0]
    assert summary.precision_mean == pytest.approx((0.5 + 1.0) / 2)
    assert summary.recall_mean == pytest.approx((0.5 + 1.0) / 2)


def test_overlap_flags_by_beta_and_r2_use_only_the_clean_setting():
    flags = pd.DataFrame({
        "noise_kind": ["clean"] * 6 + ["symmetric"] * 2, "noise_rate": [0.0] * 6 + [0.2] * 2, "noise_seed": 0,
        "repeat": 0, "fold": 0, "sample_id": ["a", "b", "c", "d", "e", "f", "a", "b"],
        "family_id": [0, 0, 0, 1, 1, 1, 0, 0], "corrupted": False,
        "flagged": [1, 1, 0, 0, 0, 0, 1, 1], "quality": 0.5,
    })  # fmt: skip
    flags["flagged"] = flags.flagged.astype(bool)
    info = pd.DataFrame({"sample_id": list("abcdef"), "beta": [1.0, 1.0, 8.0, 1.0, 8.0, 8.0],
                         "r2_cv": [0.1, 0.1, 0.9, 0.1, 0.9, 0.9]})  # fmt: skip
    by_beta, by_r2 = overlap_flags(flags, info)
    beta = by_beta.set_index("beta").flagged_fraction_mean
    assert beta.loc[1.0] == pytest.approx((1.0 + 0.0) / 2)  # família 0: a, b sinalizadas; família 1: d não
    assert beta.loc[8.0] == pytest.approx(0.0)
    assert set(by_r2.r2_bin) == {"<0,2", "≥0,5"}
