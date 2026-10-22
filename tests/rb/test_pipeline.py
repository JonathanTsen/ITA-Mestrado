"""Teste ponta a ponta (passos 30, 32, 34, 35, 39): inject → features → baselines → evaluate num run pequeno.

Verifica a separação física observado × oráculo, o checkpoint por sample_id, o pareamento 1:1 entre todos os métodos
(inclusive os de regra) e que nada é sobrescrito.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from missdetect.rb import pipeline
from missdetect.rb.bench import generate_family, load_sentinel_exclusions
from missdetect.rb.config import parse_config
from missdetect.rb.evaluate import check_pairing

N_FAMILIES = 8


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("rb")
    csv_dir = root / "csv"
    csv_dir.mkdir()
    rows = []
    for f in range(N_FAMILIES):
        rng = np.random.default_rng(f)
        z = rng.normal(size=(600, 4))
        y = 0.6 * z[:, 0] + rng.normal(scale=0.8, size=600)
        frame = pd.DataFrame(np.column_stack([y, z]).round(4), columns=["a", "b", "c", "d", "e"])
        frame["target"] = rng.integers(2, size=600)
        frame.to_csv(csv_dir / f"{100 + f}.csv")
        rows.append({"openml_id": 100 + f, "family_id": 100 + f, "representative": True, "role": "dev",
                     "numeric_columns": "a;b;c;d;e"})  # fmt: skip
    sources = root / "sources.csv"
    pd.DataFrame(rows).to_csv(sources, index=False)
    config = parse_config(
        {
            "name": "e2e",
            "root_seed": 9,
            "output_root": str(root / "results"),
            "sources": {"table": str(sources), "csv_dir": str(csv_dir), "roles": ["dev"]},
            "inject": {
                "n_grid": [500],
                "k_grid": [2],
                "rate_grid": [0.3],
                "beta_grid": [float("inf")],
                "secondary": [{"generator": "tab_err", "n": 500}],
            },
            "evaluate": {"n_splits": 2, "n_repeats": 1, "descriptive_learners": []},
            "baselines": {"mechdetect_cv_folds": 3},
        }
    )
    run_dir = pipeline.run_inject(config, n_jobs=2)
    pipeline.run_features(run_dir, config, n_jobs=2)
    pipeline.run_baselines(run_dir, config, n_jobs=2, skip_pklm=True)
    pipeline.run_evaluate(run_dir, config, n_jobs=2)
    return run_dir, config


def test_generation_layout(run):
    run_dir, _ = run
    instances = pd.read_parquet(run_dir / "instances.parquet")
    oracle = pd.read_parquet(run_dir / "oracle.parquet")
    assert len(instances) == N_FAMILIES * 2 * 3  # 1 contexto × (simétrico + tab_err) × 3 mecanismos
    assert not {"mechanism", "generator", "variant"} & set(instances.columns)
    assert set(oracle.sample_id) == set(instances.sample_id)
    for path in instances.observed_path:
        assert not any(label in path.upper() for label in ("MCAR", "MAR", "MNAR"))
        assert list(pd.read_parquet(run_dir / path).columns[:1]) == ["y"]
    assert (run_dir / "run.json").exists()


def test_complete_y_agrees_with_observed(run):
    run_dir, _ = run
    instances = pd.read_parquet(run_dir / "instances.parquet")
    for row in instances.itertuples():
        y_obs = pd.read_parquet(run_dir / row.observed_path).y.to_numpy()
        y_complete = pd.read_parquet(run_dir / "oracle_complete" / f"{row.context_id}.parquet").y_complete.to_numpy()
        observed = ~np.isnan(y_obs)
        np.testing.assert_allclose(y_obs[observed], y_complete[observed])


def test_features_and_baselines_cover_all_instances(run):
    run_dir, _ = run
    ids = set(pd.read_parquet(run_dir / "instances.parquet").sample_id)
    features = pd.read_parquet(run_dir / "features.parquet")
    baselines = pd.read_parquet(run_dir / "baselines.parquet")
    assert set(features.sample_id) == ids and (features.feature_status == "ok").all()
    assert set(baselines.sample_id) == ids
    assert (baselines[["mdo_status", "mdoracle_status", "little_status"]] == "ok").all().all()


def test_evaluate_pairing_and_methods(run):
    run_dir, _ = run
    pred = pd.read_parquet(run_dir / "eval-all" / "predictions.parquet")
    assert check_pairing(pred)
    for method in ("direct_hgb", "hier_hgb_fullprob", "trivial_uniform", "logit_simple", "nuisance_hgb",
                   "g4only_hgb", "mechdetect_obs", "mechdetect_oracle"):  # fmt: skip
        assert method in set(pred.method)
    assert pred.groupby("method").size().nunique() == 1
    assert (run_dir / "eval-all" / "summary.csv").exists()


def test_nothing_is_overwritten(run):
    run_dir, config = run
    with pytest.raises(FileExistsError):
        pipeline.run_features(run_dir, config, n_jobs=1)
    with pytest.raises(FileExistsError):
        pipeline.run_evaluate(run_dir, config, n_jobs=1)
    with pytest.raises(FileExistsError):
        pipeline.run_inject(config, n_jobs=1)


def test_features_resume_from_checkpoint(run):
    run_dir, config = run
    full = pd.read_parquet(run_dir / "features.parquet")
    resumed = run_dir.parent / "resumed"
    resumed.mkdir()
    for name in ("instances.parquet",):
        (resumed / name).write_bytes((run_dir / name).read_bytes())
    (resumed / "observed").symlink_to(run_dir / "observed")
    parts = resumed / "features_parts"
    parts.mkdir()
    full.iloc[:10].to_parquet(parts / "part-00000.parquet", index=False)
    pipeline.run_features(resumed, config, n_jobs=2)
    again = pd.read_parquet(resumed / "features.parquet").set_index("sample_id").sort_index()
    cols = [c for c in again.columns if c.startswith("g")]
    pd.testing.assert_frame_equal(again[cols], full.set_index("sample_id").sort_index()[cols])


def test_sentinel_columns_are_excluded_from_the_table(run, tmp_path):
    _, config = run
    sources = pd.read_csv(config.sources.table)
    source = sources.iloc[0]
    out = generate_family(tmp_path, Path(config.sources.csv_dir), source, config.inject, 9, frozenset({"a", "b"}))
    assert out.contexts
    for context in out.contexts:
        assert context["target"] not in {"a", "b"}
        assert not {"a", "b"} & set(context["auxiliaries"].split(";"))
    assert {r["reason"] for r in out.skipped} >= {"coluna excluída (sentinela): a", "coluna excluída (sentinela): b"}


def test_versioned_sentinel_list_is_read():
    exclusions = load_sentinel_exclusions(
        str(Path(__file__).resolve().parents[2] / "data_rb" / "sentinel_exclusions.csv")
    )
    assert exclusions == {37: {"plas", "pres", "skin", "insu", "mass"}}


def test_family_status_is_recorded(run):
    run_dir, _ = run
    status = pd.read_parquet(run_dir / "family_status.parquet")
    assert len(status) == N_FAMILIES and (status.n_bases > 0).all()
    assert status.n_instances.sum() == len(pd.read_parquet(run_dir / "instances.parquet"))
