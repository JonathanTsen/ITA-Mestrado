"""Testes da análise confirmatória (passo 46) com diferenças conhecidas."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.stats import (
    DELTA,
    bayes_signed_rank,
    confirmatory,
    decide,
    decide_equivalence,
    hodges_lehmann,
    holm,
    level_estimate,
    llm_hypothesis,
    rank_biserial,
    wilcoxon_p,
    wilcoxon_tost,
)


def test_hodges_lehmann_is_median_of_walsh_averages():
    assert hodges_lehmann(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)
    assert hodges_lehmann(np.array([0.0, 0.0, 10.0])) == pytest.approx(2.5)  # Walsh: 0, 0, 0, 5, 5, 10


def test_rank_biserial_extremes_and_zeros():
    assert rank_biserial(np.array([0.1, 0.2, 0.3])) == pytest.approx(1.0)
    assert rank_biserial(np.array([-0.1, -0.2])) == pytest.approx(-1.0)
    assert rank_biserial(np.array([0.0, 0.0])) == 0.0
    assert rank_biserial(np.array([0.1, -0.1])) == pytest.approx(0.0)


def test_holm_step_down():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm({"x": 0.9, "y": 0.8}) == pytest.approx({"y": 1.0, "x": 1.0})


def test_wilcoxon_detects_shift_and_handles_zeros():
    rng = np.random.default_rng(0)
    assert wilcoxon_p(rng.normal(0.05, 0.02, 40)) < 1e-6
    assert wilcoxon_p(np.zeros(10)) == 1.0
    assert wilcoxon_p(rng.normal(0, 0.02, 40)) > 0.01


def test_bayes_sign_convention_right_means_method_better():
    base = np.linspace(0.4, 0.6, 30)
    left, rope, right = bayes_signed_rank(base + 0.06, base)
    assert right > 0.95 and left < 0.05
    left, rope, right = bayes_signed_rank(base - 0.06, base)
    assert left > 0.95
    rng = np.random.default_rng(1)
    left, rope, right = bayes_signed_rank(base + rng.normal(0, 0.002, 30), base)
    assert rope > 0.95


def test_decide_table():
    assert decide(0.01, 0.05) == "superior"
    assert decide(0.01, -0.05) == "inferior"
    assert decide(0.01, 0.005) == "superior (efeito menor que δ)"
    assert decide(0.30, 0.01) == "evidência insuficiente"


def test_decide_equivalence():
    assert decide_equivalence(0.01) == "equivalentes dentro de ±δ"
    assert decide_equivalence(0.20) == "equivalência não demonstrada"


def test_tost_detects_equivalence_only_inside_the_margin():
    rng = np.random.default_rng(0)
    assert wilcoxon_tost(rng.normal(0.0, 0.01, 50), DELTA) < 0.001
    assert wilcoxon_tost(rng.normal(0.05, 0.01, 50), DELTA) > 0.5
    assert wilcoxon_tost(rng.normal(0.0, 0.08, 50), DELTA) > 0.05  # dispersão grande: não dá para afirmar


def _per_family(effects: dict[str, float], n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.4, 0.7, n)
    rows = []
    for method, shift in effects.items():
        scores = base + shift + rng.normal(0, 0.01, n) if method != "trivial_uniform" else np.full(n, 1 / 3)
        rows += [{"method": method, "family_id": f, "macro_recall": s} for f, s in enumerate(scores)]
    return pd.DataFrame(rows)


def _inputs(n: int = 40):
    main = _per_family({"direct_hgb": 0.0, "mechdetect_obs": -0.10, "hier_hgb_fullprob": 0.0, "logit_simple": -0.03}, n)
    cross = _per_family({"cross_direct_hgb": 0.0, "cross_logit_simple": -0.04, "mechdetect_obs": -0.08}, n, seed=5)
    rng = np.random.default_rng(7)
    base = rng.normal(0.70, 0.05, n)
    auc = pd.concat([
        pd.DataFrame({"family_id": range(n), "method": "direct_hgb", "auc": base + 0.05}),
        pd.DataFrame({"family_id": range(n), "method": "logit_simple", "auc": base + rng.normal(0, 0.01, n)}),
    ])  # fmt: skip
    return main, cross, auc


def test_confirmatory_family_has_six_tests_with_holm():
    res = confirmatory(*_inputs()).set_index("hypothesis")
    assert list(res.index) == ["H1", "H2", "H2e", "H3", "H5", "H6"]
    assert res.p_holm.min() == pytest.approx(min(6 * res.p_value.min(), 1.0))
    assert res.loc["H1", "decision"] == "superior" and res.loc["H1", "hodges_lehmann"] == pytest.approx(0.10, abs=0.01)
    assert res.loc["H1", "ci_low"] < 0.10 < res.loc["H1", "ci_high"]
    assert res.loc["H3", "decision"] == "superior" and res.loc["H3", "comparator"] == "logit_simple"
    assert res.loc["H5", "method"] == "cross_direct_hgb" and res.loc["H5", "comparator"] == "cross_logit_simple"
    assert res.loc["H5", "decision"] == "superior"
    assert res.loc["H6", "method"] == "direct_hgb" and res.loc["H6", "comparator"] == "logit_simple"
    assert res.loc["H6", "decision"] == "superior" and res.loc["H6", "hodges_lehmann"] == pytest.approx(0.05, abs=0.01)
    assert res.loc["H2", "decision"] == "evidência insuficiente"
    assert res.loc["H2e", "test"] == "tost" and res.loc["H2e", "decision"] == "equivalentes dentro de ±δ"


def test_llm_hypothesis_is_its_own_family():
    llm = _per_family({"direct_hgb": 0.0, "llm_a1": -0.08}, n=30, seed=3)
    res = llm_hypothesis(llm).set_index("hypothesis")
    assert list(res.index) == ["H4"]
    assert res.loc["H4", "n_families"] == 30
    assert res.loc["H4", "p_holm"] == pytest.approx(res.loc["H4", "p_value"])  # m = 1
    assert res.loc["H4", "decision"] == "inferior"  # d = llm_a1 − primária < 0: o LLM é inferior


def test_level_estimate_has_a_bootstrap_interval():
    main, _, _ = _inputs()
    level = level_estimate(main, "direct_hgb")
    truth = main[main.method == "direct_hgb"].macro_recall.mean()
    assert level["mean"] == pytest.approx(truth)
    assert level["ci_low"] < truth < level["ci_high"] and level["n_families"] == 40


def test_level_estimate_without_the_method_is_empty_not_an_error():
    main, _, _ = _inputs()
    level = level_estimate(main, "ausente")
    assert level["n_families"] == 0 and np.isnan(level["mean"]) and np.isnan(level["ci_low"])


# Pré-registro, seção 5: uma hipótese que não puder rodar entra no Holm com p = 1, e m continua 6 (emenda C1, 29/10).
def test_missing_method_enters_holm_with_p_one_and_a_reason():
    main, cross, auc = _inputs()
    res = confirmatory(main[main.method != "mechdetect_obs"], cross, auc).set_index("hypothesis")
    assert list(res.index) == ["H1", "H2", "H2e", "H3", "H5", "H6"]
    h1 = res.loc["H1"]
    assert h1.p_value == 1.0 and h1.p_holm == 1.0 and h1.n_families == 0
    assert h1.decision.startswith("não rodou") and "mechdetect_obs" in h1.decision
    others = res.drop(index="H1")
    assert others.p_holm.min() == pytest.approx(min(6 * others.p_value.min(), 1.0))  # m continua 6
    assert res.loc["H3", "decision"] == "superior"


def test_missing_table_enters_holm_with_p_one():
    main, _, auc = _inputs()
    res = confirmatory(main, None, auc).set_index("hypothesis")
    assert len(res) == 6 and res.loc["H5", "p_value"] == 1.0
    assert res.loc["H5", "decision"].startswith("não rodou")


def test_too_few_paired_families_is_unavailable():
    main, cross, auc = _inputs()
    lonely = cross[(cross.method != "cross_logit_simple") | (cross.family_id == 0)]  # só 1 família pareada
    res = confirmatory(main, lonely, auc).set_index("hypothesis")
    assert res.loc["H5", "decision"].startswith("não rodou") and res.loc["H5", "p_value"] == 1.0


def test_llm_hypothesis_without_the_llm_method_does_not_crash():
    res = llm_hypothesis(_per_family({"direct_hgb": 0.0}, n=30)).set_index("hypothesis")
    assert list(res.index) == ["H4"] and res.loc["H4", "p_value"] == 1.0
    assert res.loc["H4", "decision"].startswith("não rodou")
