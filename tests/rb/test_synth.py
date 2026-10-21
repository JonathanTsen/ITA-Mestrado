"""Testes do passo 33: corpus sintético (desenho balanceado, dependência controlada, elegibilidade, reprodutível)."""

import numpy as np
import pandas as pd
from scipy import stats

from missdetect.rb.config import parse_config
from missdetect.rb.inject import MECHANISMS
from missdetect.rb.synth import SYNTH_FAMILY_OFFSET, design, generate_base, sample_table

CFG = parse_config({"name": "s", "root_seed": 7, "synth": {"n_bases": 60}}).synth


def test_design_is_balanced_and_reproducible():
    items = design(CFG, 7)
    assert items == design(CFG, 7)
    assert len(items) == 60
    for attr, levels in (("n", CFG.n_grid), ("k", CFG.k_grid), ("rate", CFG.rate_grid), ("direction", CFG.directions)):
        counts = pd.Series([getattr(i, attr) for i in items]).value_counts()
        assert set(counts.index) == set(levels)
        assert counts.max() - counts.min() <= 1
    gens = pd.Series([i.generator for i in items]).value_counts()
    assert gens["symmetric"] == 40 and gens["mdatagen"] == 10 and gens["tab_err"] == 10


def test_latent_dependence_controls_r2():
    rng = np.random.default_rng(0)
    cfg = parse_config({"name": "s", "root_seed": 1, "synth": {"marginals": ["normal"]}}).synth
    r2_latent, r2_lin = [], []
    for _ in range(20):
        table, meta = sample_table(3000, 4, cfg, rng)
        z = table.drop(columns="t").to_numpy()
        z = np.column_stack([np.ones(len(z)), z])
        coef, *_ = np.linalg.lstsq(z, table.t.to_numpy(), rcond=None)
        resid = table.t.to_numpy() - z @ coef
        r2_lin.append(1 - resid.var() / table.t.var(ddof=0))
        r2_latent.append(meta["latent_r2"])
    assert stats.spearmanr(r2_latent, r2_lin).statistic > 0.95
    assert max(r2_latent) <= 0.9


def test_marginals_have_requested_shapes():
    rng = np.random.default_rng(1)
    cfg = parse_config({"name": "s", "root_seed": 1, "synth": {"marginals": ["discrete"]}}).synth
    table, _ = sample_table(2000, 2, cfg, rng)
    for col in table:
        assert table[col].nunique() >= 20
        assert np.allclose(table[col], np.round(table[col]))


def test_generate_base_three_mechanisms_same_context(tmp_path):
    item = design(CFG, 7)[0]
    out = generate_base(tmp_path, item, CFG, 7)
    assert len(out.contexts) == 1 and len(out.instances) == 3
    assert sorted(o["mechanism"] for o in out.oracle) == sorted(MECHANISMS)
    assert {r["family_id"] for r in out.instances} == {SYNTH_FAMILY_OFFSET + item.base_idx}
    assert {r["context_id"] for r in out.instances} == {out.contexts[0]["context_id"]}
    accepted = [b for b in out.bases if b["accepted"]]
    assert len(accepted) == 1 and accepted[0]["eligible"]
    rates = {round(r["realized_rate"], 3) for r in out.instances}
    assert len(rates) == 1
    again = generate_base(tmp_path / "again", item, CFG, 7)
    assert [o["mask_sha256"] for o in again.oracle] == [o["mask_sha256"] for o in out.oracle]
