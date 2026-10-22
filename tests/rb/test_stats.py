"""Testes da análise confirmatória (passo 46) com diferenças conhecidas."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.stats import (
    bayes_signed_rank,
    confirmatory,
    decide,
    hodges_lehmann,
    holm,
    rank_biserial,
    wilcoxon_p,
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
    assert decide(0.01, 0.05, 0.0) == "superior"
    assert decide(0.01, -0.05, 0.0) == "inferior"
    assert decide(0.30, 0.001, 0.97) == "praticamente equivalente"
    assert decide(0.30, 0.01, 0.50) == "evidência insuficiente"
    assert decide(0.01, 0.005, 0.97) == "superior, dentro da ROPE"


def _per_family(effects: dict[str, float], n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.4, 0.7, n)
    rows = []
    for method, shift in effects.items():
        scores = base + shift + rng.normal(0, 0.01, n) if method != "trivial_uniform" else np.full(n, 1 / 3)
        rows += [{"method": method, "family_id": f, "macro_recall": s} for f, s in enumerate(scores)]
    return pd.DataFrame(rows)


def test_confirmatory_known_effects_and_missing_h4():
    pf = _per_family({"direct_hgb": 0.0, "mechdetect_obs": -0.10, "hier_hgb_fullprob": 0.0, "trivial_uniform": 0.0})
    res = confirmatory(pf).set_index("hypothesis")
    assert set(res.index) == {"H1", "H2", "H3"}
    assert res.loc["H1", "decision"] == "superior" and res.loc["H1", "hodges_lehmann"] == pytest.approx(0.10, abs=0.01)
    assert res.loc["H1", "ci_low"] < 0.10 < res.loc["H1", "ci_high"]
    assert res.loc["H3", "decision"].startswith("superior")
    assert res.loc["H2", "decision"] in {"praticamente equivalente", "evidência insuficiente"}
    # H4 ausente entra no Holm com p = 1 (m = 4): o menor p é multiplicado por 4
    assert res.p_holm.min() == pytest.approx(min(4 * res.p_value.min(), 1.0))


def test_confirmatory_with_llm_panel():
    main = _per_family({"direct_hgb": 0.0, "mechdetect_obs": -0.10, "hier_hgb_fullprob": 0.0, "trivial_uniform": 0.0})
    llm = _per_family({"direct_hgb": 0.0, "llm_a1": -0.08}, n=30, seed=3)
    res = confirmatory(main, llm).set_index("hypothesis")
    assert res.loc["H4", "n_families"] == 30
    assert res.loc["H4", "decision"] == "inferior"  # d = llm_a1 − primária < 0: o LLM é inferior
