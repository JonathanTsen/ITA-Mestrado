"""Testes do passo 32: elegibilidade, bases/contextos aninhados, geradores e separação observado × oráculo."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.inject import (
    MECHANISMS,
    ObservedInstance,
    build_instance,
    column_is_eligible,
    context_rows,
    make_context,
    mdatagen_mask,
    pair_eligibility,
    sample_base,
    symmetric_mask,
    tab_err_mask,
)

ROOT = 123


def _table(n: int = 2500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 9))
    y = 0.6 * z[:, 0] + 0.3 * z[:, 1] + rng.normal(scale=0.8, size=n)
    frame = pd.DataFrame(z, columns=[f"z{j}" for j in range(9)])
    frame["y"] = y
    return frame.round(4)


def _base(table: pd.DataFrame, rule: str = "maxcorr"):
    base, report = sample_base(table, list(table.columns), 1, 99, 0, ROOT, driver_rule=rule, n_max=2000)
    assert base is not None, report
    return base


def test_symmetric_mask_exact_count_and_extremes():
    rng = np.random.default_rng(0)
    v = rng.normal(size=1000)
    for rate in (0.1, 0.3, 0.5):
        assert symmetric_mask(v, rate, 2.0, "upper", rng).sum() == round(rate * 1000)
    upper = symmetric_mask(v, 0.2, np.inf, "upper", rng)
    lower = symmetric_mask(v, 0.2, np.inf, "lower", rng)
    both = symmetric_mask(v, 0.2, np.inf, "both", rng)
    assert v[upper].min() >= np.quantile(v, 0.8) - 1e-9
    assert v[lower].max() <= np.quantile(v, 0.2) + 1e-9
    assert np.median(np.abs(v[both] - np.median(v))) > np.median(np.abs(v - np.median(v)))


def test_symmetric_strength_is_monotone_and_beta_zero_is_uniform():
    v = np.random.default_rng(1).normal(size=2000)
    means = []
    for beta in (0.0, 1.0, 4.0, np.inf):
        draws = [v[symmetric_mask(v, 0.3, beta, "upper", np.random.default_rng(s))].mean() for s in range(30)]
        means.append(np.mean(draws))
    assert abs(means[0]) < 0.05  # beta = 0: seleção não depende de v
    assert means[0] < means[1] < means[2] < means[3]


def test_eligibility_rejects_deterministic_target_and_tied_columns():
    table = _table()
    table["f"] = 2.0 * table["z0"] + 1.0  # Y ≈ função de Z
    flags = pair_eligibility(table.iloc[:1000], "f", [f"z{j}" for j in range(9)], seed=0)
    assert not flags["eligible"] and not flags["spearman_ok"]
    assert not column_is_eligible(pd.Series(np.repeat([0.0, 1.0, 2.0], 400)))
    assert column_is_eligible(table["z1"])


def test_three_mechanisms_share_context_and_only_target_is_masked():
    table = _table()
    base = _base(table)
    context = make_context(base, 1000, 4, 0.3, 2.0, "upper", ROOT)
    sub = context_rows(table, base, context)
    instances = {m: build_instance(sub, context, "symmetric", m) for m in MECHANISMS}
    z_frames = [obs.z for obs, _, _ in instances.values()]
    assert all(z.equals(z_frames[0]) for z in z_frames)
    assert list(z_frames[0].columns) == list(context.auxiliaries) and context.target not in z_frames[0].columns
    assert not z_frames[0].isna().any().any()
    for obs, oracle, mask in instances.values():
        assert np.isnan(obs.y).sum() == mask.sum() == round(0.3 * 1000)
        assert oracle.realized_rate == pytest.approx(0.3)
    masks = [mask for _, _, mask in instances.values()]
    assert not np.array_equal(masks[1], masks[2])  # MAR (driver) e MNAR (alvo) diferem


def test_nested_k_and_n_share_rows_and_mask():
    table = _table()
    base = _base(table, rule="random")
    assert base.auxiliary_order[0] == base.driver
    contexts = {k: make_context(base, 1000, k, 0.2, 1.0, "lower", ROOT) for k in (2, 4, 8)}
    built = {k: build_instance(context_rows(table, base, c), c, "symmetric", "MNAR") for k, c in contexts.items()}
    assert np.array_equal(built[2][2], built[8][2])  # mesma máscara para todos os k
    assert list(built[4][0].z.columns)[:2] == list(built[2][0].z.columns)  # Z_2 ⊂ Z_4
    small = make_context(base, 500, 4, 0.2, 1.0, "lower", ROOT)
    rows_small = context_rows(table, base, small)
    rows_big = context_rows(table, base, contexts[4])
    assert set(map(tuple, rows_small.to_numpy())) <= set(map(tuple, rows_big.to_numpy()))


def test_observed_instance_carries_no_oracle_fields():
    fields = {f.name for f in dataclasses.fields(ObservedInstance)}
    assert fields == {"sample_id", "y", "z"}


@pytest.mark.parametrize(
    ("mechanism", "variant"),
    [("MCAR", "random"), ("MAR", "lowest"), ("MAR", "rank"), ("MAR", "median"), ("MNAR", "run_0"), ("MNAR", "run_1")],
)
def test_mdatagen_wrapper_is_seeded_and_touches_only_target(mechanism, variant):
    sub = _table(1000).iloc[:1000]
    first, rate = mdatagen_mask(sub, "y", "z0", mechanism, variant, 0.3, seed=7)
    second, _ = mdatagen_mask(sub, "y", "z0", mechanism, variant, 0.3, seed=7)
    assert np.array_equal(first, second)
    assert rate == pytest.approx(0.3)


def test_mdatagen_run_is_truncation_on_target():
    sub = _table(1000).iloc[:1000]
    mask, _ = mdatagen_mask(sub, "y", "z0", "MNAR", "run_0", 0.2, seed=1)
    assert sub["y"][mask].max() <= sub["y"][~mask].min()  # threshold 0 = menores valores de Y


def test_mdatagen_mbov_realized_rate_falls_short():
    """Documenta o defeito do MBOV_randomness (np.delete não atribuído): taxa realizada abaixo da pedida."""
    sub = _table(2000).iloc[:2000]
    deficits = [0.3 - mdatagen_mask(sub, "y", "z0", "MNAR", "mbov_0.5", 0.3, seed=s)[1] for s in range(5)]
    assert np.mean(deficits) > 0.01


def test_tab_err_wrapper_is_seeded_and_exact():
    sub = _table(1000).iloc[:1000]
    for mechanism in MECHANISMS:
        first, rate = tab_err_mask(sub, "y", "z0", mechanism, 0.3, seed=5)
        second, _ = tab_err_mask(sub, "y", "z0", mechanism, 0.3, seed=5)
        assert np.array_equal(first, second) and rate == pytest.approx(0.3)
