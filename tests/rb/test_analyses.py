"""Testes das análises secundárias e exploratórias e do painel oráculo (pré-registro, itens 6–7)."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.analyses import (
    bootstrap_mean_ci,
    boundary_flags,
    descriptive_classifiers,
    l1_per_family,
    l2_evaluation,
    per_mechanism,
)
from missdetect.rb.baselines import mechdetect_obs, oracle_descriptors
from missdetect.rb.inject import ObservedInstance, symmetric_mask


def _instance(mechanism: str, n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    y = 0.6 * z[:, 0] + rng.normal(scale=0.8, size=n)
    v = {"MCAR": np.zeros(n), "MAR": z[:, 0], "MNAR": y}[mechanism]
    mask = symmetric_mask(v, 0.3, 0.0 if mechanism == "MCAR" else np.inf, "upper", rng)
    y_obs = y.copy()
    y_obs[mask] = np.nan
    return ObservedInstance(f"s{seed}", y_obs, pd.DataFrame(z, columns=["a", "b", "c"])), y


def test_l1_per_family_counts_failures_as_errors():
    d = pd.DataFrame(
        {
            "method": "t",
            "repeat": 0,
            "sample_id": list("abcd"),
            "family_id": 1,
            "non_mcar": [False, False, True, True],
            "reject": [False, False, True, False],
            "failed": [False, True, False, True],
        }
    )
    row = l1_per_family(d).iloc[0]
    assert row.specificity == pytest.approx(0.5)  # a falha sob MCAR não é "não rejeitou"
    assert row.sensitivity == pytest.approx(0.5)
    assert row.balanced_accuracy == pytest.approx(0.5)


def test_l2_evaluation_recovers_p2_from_fullprob():
    p1, p2 = 0.2, 0.9  # P(MCAR) e P(MAR | não-MCAR)
    rows = []
    for y_true in (1, 2):
        rows.append({"method": "hier_hgb_fullprob", "repeat": 0, "family_id": 1, "y_true": y_true, "status": "ok",
                     "p_MCAR": p1, "p_MAR": (1 - p1) * p2, "p_MNAR": (1 - p1) * (1 - p2), "y_pred": 1})  # fmt: skip
    res = l2_evaluation(pd.DataFrame(rows)).iloc[0]
    assert res.l2_diagnostic_macro_recall == pytest.approx(0.5)  # prevê MAR nas duas: recall MAR 1, MNAR 0
    assert res.non_mcar_routed_fraction == pytest.approx(1.0)


def test_descriptive_classifiers_ranks():
    rng = np.random.default_rng(0)
    rows = []
    for f in range(20):
        for k, m in enumerate(("direct_hgb", "direct_logreg", "direct_rf", "direct_knn")):
            rows.append({"method": m, "family_id": f, "macro_recall": 0.5 + 0.05 * (3 - k) + rng.normal(0, 0.01)})
    res = descriptive_classifiers(pd.DataFrame(rows))
    assert res["friedman_p"] < 1e-6
    assert min(res["mean_ranks"], key=res["mean_ranks"].get) == "direct_hgb"


def test_bootstrap_mean_ci_contains_mean():
    values = np.random.default_rng(1).normal(0.6, 0.05, 50)
    low, high = bootstrap_mean_ci(values)
    assert low < values.mean() < high


def test_boundary_flags(tmp_path):
    (tmp_path / "oracle_complete").mkdir()
    pd.DataFrame({"y_complete": [0.0] * 30 + list(np.arange(1, 71.0))}).to_parquet(
        tmp_path / "oracle_complete" / "c1.parquet"
    )
    pd.DataFrame({"y_complete": np.arange(100.0)}).to_parquet(tmp_path / "oracle_complete" / "c2.parquet")
    flags = boundary_flags(tmp_path).set_index("context_id")
    assert bool(flags.loc["c1", "flagged"]) and not bool(flags.loc["c2", "flagged"])


def test_per_mechanism_has_three_classes_with_holm():
    rows = []
    rng = np.random.default_rng(2)
    for f in range(30):
        for m, shift in (("a", 0.1), ("b", 0.0)):
            rows.append({"method": m, "family_id": f, **{f"recall_{c}": 0.5 + shift + rng.normal(0, 0.02)
                                                         for c in ("MCAR", "MAR", "MNAR")}})  # fmt: skip
    res = per_mechanism(pd.DataFrame(rows), "X", "a", "b")
    assert len(res) == 3 and (res.p_holm >= res.p_value).all()


def test_dirty_regime_is_degenerate_under_mcar():
    instance, _ = _instance("MCAR", seed=3)
    assert mechdetect_obs.dirty(instance)["mddirty_decision"] != "MCAR"


def test_oracle_descriptors_see_the_mechanism():
    mnar, y_mnar = _instance("MNAR", seed=4)
    mcar, y_mcar = _instance("MCAR", seed=5)
    hi = oracle_descriptors.compute(mnar, y_mnar)
    lo = oracle_descriptors.compute(mcar, y_mcar)
    assert hi["orc_auc_mask_from_y"] > 0.9 and abs(lo["orc_auc_mask_from_y"] - 0.5) < 0.1
    assert hi["orc_auc_yz_minus_z"] > 0.05
    with pytest.raises(ValueError):
        oracle_descriptors.compute(mnar, y_mnar + 1.0)


def test_l2_evaluation_counts_failures_as_errors():
    base = {"method": "hier_hgb_fullprob", "repeat": 0, "family_id": 1, "y_pred": 1,
            "p_MCAR": 0.2, "p_MAR": 0.72, "p_MNAR": 0.08}  # fmt: skip
    rows = [{**base, "y_true": 1, "status": "ok"}, {**base, "y_true": 1, "status": "failed: x"}]
    res = l2_evaluation(pd.DataFrame(rows)).iloc[0]
    assert res.l2_diagnostic_macro_recall == pytest.approx(0.5)  # 1 acerto em 2 MAR; a falha é erro
    assert res.non_mcar_routed_fraction == pytest.approx(0.5)  # a falha não é roteada
