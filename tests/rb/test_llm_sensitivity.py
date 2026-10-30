"""Testes da sensibilidade de 200 linhas da H4: mesmas linhas que o LLM vê, mesmas partições, saídas por família."""

import json

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.config import parse_config
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.llm import sensitivity
from missdetect.rb.llm.render import llm_rows
from missdetect.rb.runs import read_observed, write_observed
from missdetect.rb.splits import outer_splits

NAMES = ["g0_missing_rate", "g1_mean", "g2_corr"]


@pytest.fixture
def run(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    instances, oracle = [], []
    for family in range(1, 7):
        for rate in (0.1, 0.3, 0.5):
            ctx = f"c{family}r{rate}"
            rng = np.random.default_rng(family * 10 + int(rate * 10))
            z = pd.DataFrame({"drv": rng.normal(size=1000), "aux": rng.normal(size=1000)})
            y_full = z.drv.to_numpy() + rng.normal(size=1000)
            for index, (mech, score) in enumerate([("MCAR", rng.random(1000)), ("MAR", z.drv.to_numpy()),
                                                   ("MNAR", y_full)]):  # fmt: skip
                y = y_full.copy()
                y[np.argsort(score)[: int(rate * 1000)]] = np.nan
                sid = f"{ctx}x{index}"
                rel = write_observed(run_dir, ObservedInstance(sid, y, z))
                instances.append({"sample_id": sid, "context_id": ctx, "family_id": family, "observed_path": rel,
                                  "realized_rate": rate})  # fmt: skip
                oracle.append({"sample_id": sid, "mechanism": mech, "generator": "symmetric"})
    pd.DataFrame(instances).to_parquet(run_dir / "instances.parquet", index=False)
    pd.DataFrame(oracle).to_parquet(run_dir / "oracle.parquet", index=False)
    (run_dir / "feature_schema.json").write_text(
        json.dumps({"version": "34b", "features": [{"name": n} for n in NAMES]})
    )
    lengths: list[int] = []

    def fake_compute(instance):
        lengths.append(len(instance.y))
        mask = np.isnan(instance.y)
        return {"sample_id": instance.sample_id, "g0_missing_rate": mask.mean(), "g1_mean": np.nanmean(instance.y),
                "g2_corr": abs(np.corrcoef(mask, instance.z.iloc[:, 0])[0, 1])}  # fmt: skip

    monkeypatch.setattr(sensitivity.feat, "compute", fake_compute)
    config = parse_config({"name": "s", "root_seed": 9, "output_root": str(tmp_path),
                           "evaluate": {"n_splits": 3, "n_repeats": 2}})  # fmt: skip
    return run_dir, config, lengths


def test_subsample_uses_the_rows_the_llm_sees(run):
    run_dir, _, _ = run
    instance = read_observed(run_dir, "c1r0.1x0")
    sub = sensitivity.subsample(instance, seed=5)
    rows = llm_rows(1000, 5)
    np.testing.assert_array_equal(sub.z.drv.to_numpy(), instance.z.drv.to_numpy()[rows])
    assert len(sub.y) == 200


def test_sensitivity_retrains_on_200_rows_in_the_same_partitions(run, tmp_path):
    run_dir, config, lengths = run
    panel = [f"c{f}r{r}x{i}" for f in (1, 2, 3) for r in (0.1, 0.5) for i in range(3)]
    llm = pd.DataFrame({"method": "llm_a1", "repeat": 0, "sample_id": panel,
                        "family_id": [int(s[1]) for s in panel], "y_true": [int(s[-1]) for s in panel],
                        "y_pred": 0, "p_MCAR": 0.8, "p_MAR": 0.1, "p_MNAR": 0.1})  # fmt: skip
    llm_path = tmp_path / "llm.parquet"
    llm.to_parquet(llm_path, index=False)
    out = sensitivity.run_llm_sensitivity(run_dir, config, n_jobs=1, panel_ids=panel, llm_predictions=llm_path)
    assert set(lengths) == {200}
    pred = pd.read_parquet(out / "predictions.parquet")
    assert set(pred.method) == {"direct_hgb_200"} and len(pred) == 54 * 2
    families = pd.read_parquet(run_dir / "instances.parquet").sort_values("sample_id").family_id.reset_index(drop=True)
    expected = {}
    for repeat, fold, _, test in outer_splits(families, config.root_seed, 3, 2):
        for family in families.iloc[test].unique():
            expected[(repeat, int(family))] = fold
    got = pred.groupby(["repeat", "family_id"]).fold.first().to_dict()
    assert got == expected
    per_family = pd.read_parquet(out / "per_family.parquet")
    assert set(per_family.family_id) == {1, 2, 3}
    comparison = pd.read_csv(out / "h4_sensitivity.csv")
    assert comparison.hypothesis.tolist() == ["H4-200"] and comparison.n_families.iloc[0] == 3
    assert (run_dir / "run.llm-sensitivity.json").exists()
    assert sensitivity.run_llm_sensitivity(run_dir, config, n_jobs=1, panel_ids=panel) == out  # já feito
