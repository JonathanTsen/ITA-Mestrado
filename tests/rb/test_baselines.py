"""Testes do passo 39: Little, MechDetect-O, painel oráculo e PKLM."""

import shutil

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from missdetect.rb.baselines import little, mechdetect_obs
from missdetect.rb.baselines.pklm_r import run_pklm
from missdetect.rb.inject import ObservedInstance, symmetric_mask
from missdetect.rb.runs import write_observed


def _instance(mechanism: str, n: int = 1000, seed: int = 0, rate: float = 0.3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    y = 0.7 * z[:, 0] + rng.normal(scale=0.7, size=n)
    v = {"MCAR": np.zeros(n), "MAR": z[:, 0], "MNAR": y}[mechanism]
    beta = 0.0 if mechanism == "MCAR" else np.inf
    mask = symmetric_mask(v, rate, beta, "upper", rng)
    y_obs = y.copy()
    y_obs[mask] = np.nan
    return ObservedInstance(f"s{seed}", y_obs, pd.DataFrame(z, columns=["a", "b", "c"])), y


def test_little_mle_matches_complete_data_estimates():
    instance, y = _instance("MCAR", n=5000)
    mu, sigma = little.little_mle(instance.y, instance.z.to_numpy())
    full = np.column_stack([y, instance.z.to_numpy()])
    np.testing.assert_allclose(mu, full.mean(axis=0), atol=0.05)
    np.testing.assert_allclose(sigma, np.cov(full, rowvar=False), atol=0.06)


def test_little_size_and_power():
    p_mcar = [little.run(_instance("MCAR", seed=s)[0])["little_p"] for s in range(60)]
    assert 0.0 <= np.mean(np.asarray(p_mcar) < 0.05) <= 0.15
    assert stats.kstest(p_mcar, "uniform").pvalue > 0.01
    p_mar = [little.run(_instance("MAR", seed=s)[0])["little_p"] for s in range(10)]
    assert np.mean(np.asarray(p_mar) < 0.05) == 1.0
    assert little.run(_instance("MCAR")[0])["little_df"] == 3


def test_mechdetect_obs_decisions():
    decisions = {m: mechdetect_obs.observed_only(_instance(m, seed=1)[0])["mdo_decision"] for m in ("MCAR", "MAR")}
    assert decisions == {"MCAR": "MCAR", "MAR": "MAR"}
    mnar = mechdetect_obs.observed_only(_instance("MNAR", seed=1)[0])
    assert mnar["mdo_decision"] != "MCAR"


def test_mechdetect_oracle_uses_complete_y_and_checks_it():
    instance, y = _instance("MNAR", seed=2)
    assert mechdetect_obs.oracle_clean(instance, y)["mdoracle_decision"] == "MNAR"
    with pytest.raises(ValueError):
        mechdetect_obs.oracle_clean(instance, y + 1.0)


def test_decision_proba():
    proba = mechdetect_obs.decision_proba(pd.Series(["MAR", "MCAR", "MNAR"]))
    np.testing.assert_array_equal(proba.argmax(axis=1), [1, 0, 2])
    with pytest.raises(ValueError):
        mechdetect_obs.decision_proba(pd.Series(["X"]))


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R não instalado")
def test_pklm_via_rscript(tmp_path):
    ids = []
    for mechanism, seed in (("MCAR", 4), ("MAR", 5)):
        instance, _ = _instance(mechanism, n=300, seed=seed)
        write_observed(tmp_path, instance)
        ids.append(instance.sample_id)
    out = run_pklm(tmp_path, ids, num_proj=10, nrep=20, seed=1, n_jobs=2)
    assert list(out.sample_id) == ids
    assert (out.status == "ok").all()
    assert out.pklm_p.between(0, 1).all()
    assert out.set_index("sample_id").loc["s5", "pklm_p"] < 0.05
