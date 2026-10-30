"""Testes do painel de LLM (pré-registro §4, H4): 30 famílias, 6 contextos por família, repetição e ordem."""

import numpy as np
import pandas as pd

from missdetect.rb.llm.panel import (
    family_order,
    panel_contexts,
    panel_families,
    panel_instances,
    repeat_subsample,
)


def _design():
    contexts, bases, instances, oracle = [], [], [], []
    for family in (1, 2):
        n_max, ks = (768, [2]) if family == 1 else (5000, [2, 4, 7])
        for rate in (0.1, 0.2, 0.3, 0.4, 0.5):
            base = f"b{family}-{rate}"
            bases.append({"base_id": base, "family_id": family, "n_max": n_max})
            for n in sorted({min(500, n_max), min(1000, n_max), min(2000, n_max)}):
                for k in ks:
                    ctx = f"{base}-{n}-{k}"
                    contexts.append({"context_id": ctx, "base_id": base, "family_id": family, "n": n, "k": k,
                                     "rate": rate})  # fmt: skip
                    for generator in ("symmetric", "mdatagen"):
                        for mech in ("MCAR", "MAR", "MNAR"):
                            sid = f"{ctx}-{generator}-{mech}"
                            instances.append({"sample_id": sid, "context_id": ctx, "family_id": family})
                            oracle.append({"sample_id": sid, "mechanism": mech, "generator": generator})
    return pd.DataFrame(contexts), pd.DataFrame(bases), pd.DataFrame(instances), pd.DataFrame(oracle)


def test_panel_families_sample_30_with_the_registered_seed():
    effective = list(range(100, 150))
    chosen = panel_families(effective, 20261101)
    assert len(chosen) == 30 and chosen == sorted(chosen) and set(chosen) <= set(effective)
    assert chosen == panel_families(list(reversed(effective)), 20261101)
    assert panel_families([5, 3, 4], 20261101) == [3, 4, 5]


def test_panel_contexts_rates_k_and_n_rule():
    contexts, bases, _, _ = _design()
    chosen = panel_contexts(contexts, bases, [1, 2])
    big = chosen[chosen.family_id == 2]
    assert len(big) == 6 and set(big.n) == {1000} and set(big.k) == {2, 7}
    assert sorted(big.rate.unique()) == [0.1, 0.3, 0.5]
    small = chosen[chosen.family_id == 1]
    assert len(small) == 3 and set(small.n) == {768} and set(small.k) == {2}  # um só k disponível


def test_panel_instances_symmetric_trios_only():
    contexts, bases, instances, oracle = _design()
    panel = panel_instances(instances, oracle, panel_contexts(contexts, bases, [1, 2]))
    assert len(panel) == (6 + 3) * 3
    assert set(panel.generator) == {"symmetric"}
    assert (panel.groupby("context_id").mechanism.nunique() == 3).all()


def test_repeat_subsample_is_ten_percent_and_reproducible():
    ids = pd.Series([f"s{i:03d}" for i in range(540)])
    chosen = repeat_subsample(ids, 20261101)
    assert len(chosen) == 54 and len(set(chosen)) == 54 and set(chosen) <= set(ids)
    assert chosen == repeat_subsample(ids.sample(frac=1, random_state=0), 20261101)


def test_family_order_is_a_seeded_permutation():
    families = [3, 1, 2, 9]
    order = family_order(families, 20261101)
    assert sorted(order) == [1, 2, 3, 9] and order == family_order([9, 2, 1, 3], 20261101)
    assert isinstance(order[0], int) and not isinstance(order[0], np.integer)
