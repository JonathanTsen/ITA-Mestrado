"""Teste ponta a ponta (passos 30, 32, 34, 35, 39): inject → features → baselines → evaluate num run pequeno.

Verifica a separação física observado × oráculo, o checkpoint por sample_id, o pareamento 1:1 entre todos os métodos
(inclusive os de regra) e que nada é sobrescrito.
"""

import json
import shutil
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
            "evaluate": {"n_splits": 2, "n_repeats": 1, "descriptive_learners": ["logreg"]},
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


@pytest.fixture(scope="module")
def extended(run, tmp_path_factory):
    """Comandos do pré-registro (itens 6–7) encadeados sobre o run pequeno, mais um mini-corpus sintético."""
    run_dir, config = run
    pipeline.run_oracle_features(run_dir, config, n_jobs=2)
    pipeline.run_evaluate(run_dir, config, n_jobs=2, generators=("symmetric",))
    pipeline.run_ablation(run_dir, config, n_jobs=2)
    pipeline.run_logo(run_dir, config, n_jobs=2)
    pipeline.run_cross_generator(run_dir, config, n_jobs=2)
    pipeline.run_secondary(run_dir, config)
    pipeline.run_stats(run_dir, config)
    synth_config = parse_config(
        {
            "name": "mini-synth",
            "root_seed": 4,
            "output_root": str(tmp_path_factory.mktemp("synth")),
            "synth": {"n_bases": 12, "n_grid": [500], "k_grid": [2, 4]},
        }
    )
    synth_dir = pipeline.run_synth(synth_config, n_jobs=2)
    pipeline.run_features(synth_dir, synth_config, n_jobs=2)
    transfer_dir = pipeline.run_transfer(synth_dir, run_dir, config)
    return run_dir, transfer_dir


def test_oracle_panel_is_separate_and_paired(extended):
    run_dir, _ = extended
    oracle = pd.read_parquet(run_dir / "oracle_features.parquet")
    assert (oracle.oracle_status == "ok").all()
    pred = pd.read_parquet(run_dir / "eval-symmetric" / "predictions.parquet")
    assert {"direct_hgb_oracle_desc", "mechdetect_oracle", "mechdetect_dirty"} <= set(pred.method)
    assert check_pairing(pred)


def test_ablation_has_four_groups_and_holm(extended):
    run_dir, _ = extended
    tests = pd.read_csv(run_dir / "eval-ablation" / "ablation_tests.csv")
    assert list(tests.hypothesis) == [f"RQ2_G{k}" for k in range(1, 5)]
    assert (tests.p_holm >= tests.p_value - 1e-12).all()


def test_logo_and_transfer_mark_feature_failures(extended):
    run_dir, transfer_dir = extended
    for path in (run_dir / "eval-logo" / "predictions.parquet", transfer_dir / "predictions.parquet"):
        pred = pd.read_parquet(path)
        failed = pred[pred.status != "ok"]
        assert (failed.y_pred == -1).all()


def test_logo_tests_on_held_out_generator(extended):
    run_dir, _ = extended
    pred = pd.read_parquet(run_dir / "eval-logo" / "predictions.parquet")
    oracle = pd.read_parquet(run_dir / "oracle.parquet")[["sample_id", "generator"]]
    merged = pred.merge(oracle, on="sample_id")
    for method, g in merged.groupby("method"):
        assert set(g.generator) == {method.split("_", 1)[1]}


def test_secondary_and_stats_outputs(extended):
    run_dir, _ = extended
    l1 = pd.read_csv(run_dir / "secondary" / "l1_tests.csv")
    assert set(l1.hypothesis) == {"L1_vs_Little", "L1_vs_MechDetectO_teste1"}  # sem PKLM neste run
    assert l1.p_holm.min() == pytest.approx(min(3 * l1.p_value.min(), 1.0))  # Holm com m = 3 fixo (PKLM com p = 1)
    sens = pd.read_csv(run_dir / "secondary" / "boundary_sensitivity.csv").iloc[0]
    assert sens.n_families_common <= sens.n_families_total
    assert (run_dir / "secondary" / "l2_evaluation.csv").exists()
    assert (run_dir / "secondary" / "boundary_sensitivity.csv").exists()
    stats_out = pd.read_csv(run_dir / "stats_confirmatory.csv")
    assert list(stats_out.hypothesis) == ["H1", "H2", "H2e", "H3", "H5", "H6"]
    assert stats_out.p_holm.min() == pytest.approx(min(6 * stats_out.p_value.min(), 1.0))  # Holm m = 6
    level = pd.read_csv(run_dir / "stats_level.csv")
    assert {"direct_hgb", "cross_direct_hgb"} <= set(level.method)
    oracle = pd.read_csv(run_dir / "secondary" / "oracle_panel_tests.csv")
    assert list(oracle.hypothesis) == ["ORA_symmetric"]
    assert oracle.p_holm.iloc[0] == pytest.approx(
        min(3 * oracle.p_value.iloc[0], 1.0)
    )  # m = 3 (geradores sem eval: p = 1)
    assert list(pd.read_csv(run_dir / "secondary" / "g4_tests.csv").hypothesis) == ["G4"]
    assert list(pd.read_csv(run_dir / "secondary" / "external_mdo_tests.csv").hypothesis) == ["EXT_mdo"]
    h6 = stats_out.set_index("hypothesis").loc["H6"]
    assert (h6.method, h6.comparator) == ("direct_hgb", "logit_simple")
    decomposition = pd.read_csv(run_dir / "stats_h3_decomposition.csv")
    assert {"learner", "descriptors"} <= set(decomposition.component)


def test_stats_leaves_nothing_behind_when_it_fails(run, extended, tmp_path, monkeypatch):
    """C1 (29/10): se o `stats` falha, nenhuma saída fica pela metade, e a re-execução funciona depois."""
    run_dir, config = run
    copy = tmp_path / "copy"
    copy.mkdir()
    for sub in ("eval-symmetric", "eval-cross"):
        shutil.copytree(run_dir / sub, copy / sub)

    def boom(per_family):
        raise RuntimeError("falha no meio")

    monkeypatch.setattr(pipeline, "_h3_decomposition", boom)
    with pytest.raises(RuntimeError):
        pipeline.run_stats(copy, config)
    assert not list(copy.glob("stats_*")) and not list(copy.glob(".stats_*"))
    monkeypatch.undo()
    pipeline.run_stats(copy, config)
    assert (copy / "stats_confirmatory.csv").exists() and (copy / "stats_level.csv").exists()


def test_stats_llm_is_a_separate_step(run, extended, tmp_path):
    """C1 (29/10): a H4 roda depois da produção, sem tocar a saída confirmatória principal."""
    run_dir, config = run
    before = (run_dir / "stats_confirmatory.csv").read_bytes()
    rng = np.random.default_rng(0)
    rows = [{"method": m, "family_id": f, "macro_recall": rng.uniform(0.3, 0.7)}
            for m in ("llm_a1", "direct_hgb") for f in range(8)]  # fmt: skip
    path = tmp_path / "llm_per_family.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    out = pipeline.run_stats_llm(run_dir, config, path)
    res = pd.read_csv(out)
    assert list(res.hypothesis) == ["H4"] and res.n_families.iloc[0] == 8
    assert (run_dir / "stats_confirmatory.csv").read_bytes() == before
    assert json.loads((run_dir / "run.stats-llm.json").read_text())["llm_per_family_sha256"]
    with pytest.raises(FileExistsError):
        pipeline.run_stats_llm(run_dir, config, path)


def test_secondary_extra_writes_the_amendment_analyses(run, extended, tmp_path):
    """C4 (emendas de 29/10): P6, P7, P8, P11, P12, P13 e P15 a partir das avaliações gravadas, de uma vez."""
    run_dir, config = run
    nominal = tmp_path / "nominal.csv"
    pd.DataFrame({"family_id": [100], "target": ["*"], "reason": ["teste"]}).to_csv(nominal, index=False)
    out = pipeline.run_secondary_extra(run_dir, config, nominal_list=nominal)
    names = {p.name for p in out.iterdir()}
    for name in ("p6_recall_by_regime.csv", "p7_trio_concordance_summary.csv", "p8_generator_effect.csv",
                 "p11_by_repeat.csv", "p12_tests.csv", "p13_l1_auc_summary.csv", "p15_nominal_sensitivity.csv"):  # fmt: skip
        assert name in names, name
    assert not [n for n in names if n.startswith(".")]  # sem temporários
    regime = pd.read_csv(out / "p6_recall_by_regime.csv")
    assert {"inf", "inf/lower", "inf/upper", "inf/both"} & set(regime.regime)
    assert len(pd.read_csv(out / "p12_tests.csv")) == 2
    p15 = pd.read_csv(out / "p15_nominal_sensitivity.csv")
    assert list(p15.hypothesis) == ["H1", "H2", "H2e", "H3", "H5", "H6"] and p15.n_contexts_excluded.iloc[0] > 0
    with pytest.raises(FileExistsError):
        pipeline.run_secondary_extra(run_dir, config, nominal_list=nominal)


def test_ampute_is_a_test_only_external_generator(run, extended):
    """P24 (29/10): instâncias do ampute em <run>/ampute/ (sem tocar o benchmark), com descritores, e a avaliação dos
    modelos treinados só no simétrico, nas mesmas partições, com a secundária m = 1 e a mudança simétrico → ampute."""
    run_dir, config = run
    oracle_before = (run_dir / "oracle.parquet").read_bytes()
    amp_dir = pipeline.run_inject_ampute(run_dir, config, n_jobs=2)
    assert (run_dir / "oracle.parquet").read_bytes() == oracle_before
    amp_oracle = pd.read_parquet(amp_dir / "oracle.parquet")
    oracle = pd.read_parquet(run_dir / "oracle.parquet")
    external = set(oracle.loc[oracle.generator != "symmetric", "context_id"])
    assert set(amp_oracle.context_id) == external and (amp_oracle.generator == "ampute").all()
    assert len(amp_oracle) == 3 * len(external) and not amp_oracle.deterministic.any()
    features = pd.read_parquet(amp_dir / "features.parquet")
    assert set(features.sample_id) == set(amp_oracle.sample_id) and (features.feature_status == "ok").all()
    out = pipeline.run_cross_ampute(run_dir, config, n_jobs=2)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert set(pred.method) == {"cross_direct_hgb", "cross_logit_simple"} and check_pairing(pred)
    tests = pd.read_csv(out / "ampute_tests.csv")
    assert list(tests.hypothesis) == ["P24_ampute"]
    change = pd.read_csv(out / "ampute_change.csv")
    assert change.quantity.iloc[0] == "change_ampute_minus_symmetric" and change.n_families.iloc[0] > 0
    rates = pd.read_csv(out / "ampute_realized_rates.csv")
    assert set(rates.mechanism) == {"MCAR", "MAR", "MNAR"}
    with pytest.raises(FileExistsError):
        pipeline.run_cross_ampute(run_dir, config, n_jobs=1)


def test_transfer_predicts_every_real_instance(extended):
    run_dir, transfer_dir = extended
    pred = pd.read_parquet(transfer_dir / "predictions.parquet")
    n_real = len(pd.read_parquet(run_dir / "instances.parquet"))
    assert pred.groupby("method").size().eq(n_real).all()
    assert "transfer_all" in set(pred.method)
    matrix = pd.read_csv(transfer_dir / "transfer_matrix.csv")
    assert {"all", "symmetric"} <= set(matrix.train)


@pytest.fixture(scope="module")
def noisy(run, extended):
    """RQ3 (passo 44) sobre o run pequeno (depois do `extended`, que grava eval-symmetric), uma semente por célula."""
    run_dir, config = run
    return run_dir, pipeline.run_noise(run_dir, config, n_jobs=2, n_seeds=1), config


def test_noise_outputs_and_label_immutability(noisy):
    run_dir, out_dir, _ = noisy
    for name in ("predictions.parquet", "flags.parquet", "per_family.parquet", "rq3_tests.csv",
                 "degradation_curves.csv", "flag_summary.csv", "overlap_by_beta.csv", "overlap_by_r2.csv",
                 "settings.csv", "degradation.png"):  # fmt: skip
        assert (out_dir / name).exists(), name
    log = json.loads((run_dir / "run.noise.json").read_text())
    assert log["labels_sha256_before"] == log["labels_sha256_after"]
    assert log["oracle_sha256_before"] == log["oracle_sha256_after"]
    pred = pd.read_parquet(out_dir / "predictions.parquet")
    oracle = pd.read_parquet(run_dir / "oracle.parquet")[["sample_id", "mechanism"]]
    truth = oracle.set_index("sample_id").mechanism.map({"MCAR": 0, "MAR": 1, "MNAR": 2})
    assert (pred.y_true.to_numpy() == pred.sample_id.map(truth).to_numpy()).all()
    tests = pd.read_csv(out_dir / "rq3_tests.csv")
    assert len(tests) == 4


def test_clean_noise_reference_equals_the_primary_evaluation(noisy):
    run_dir, out_dir, _ = noisy
    pred = pd.read_parquet(out_dir / "predictions.parquet")
    clean = pred[(pred.noise_kind == "clean") & (pred.mitigation == "none")]
    primary = pd.read_parquet(run_dir / "eval-symmetric" / "predictions.parquet")
    primary = primary[primary.method == "direct_hgb"]
    cols = ["repeat", "fold", "sample_id"]
    merged = clean.merge(primary, on=cols, suffixes=("_noise", "_primary"))
    assert len(merged) == len(primary) == len(clean)
    np.testing.assert_allclose(merged.p_MCAR_noise, merged.p_MCAR_primary)


def test_noise_is_not_overwritten(noisy):
    run_dir, _, config = noisy
    with pytest.raises(FileExistsError):
        pipeline.run_noise(run_dir, config, n_jobs=1, n_seeds=1)


@pytest.fixture(scope="module")
def final_model(run):
    """Passo 45: modelo final congelado (aqui, treinado no run pequeno)."""
    run_dir, config = run
    return run_dir, pipeline.run_fit_final(run_dir, config), config


def test_final_model_is_frozen_with_its_hash(final_model):
    run_dir, model_dir, config = final_model
    meta = json.loads((model_dir / "model.json").read_text())
    assert meta["model_sha256"] == pipeline.sha256_path(model_dir / "model.joblib")
    assert meta["generator"] == "symmetric"
    instances = pd.read_parquet(run_dir / "oracle.parquet")
    assert meta["n_instances"] == int((instances.generator == "symmetric").sum())
    model, loaded = pipeline.load_final(model_dir)
    assert loaded["feature_columns"] == meta["feature_columns"]
    assert model.predict_proba(np.zeros((1, len(meta["feature_columns"])))).shape == (1, 3)
    with pytest.raises(FileExistsError):
        pipeline.run_fit_final(run_dir, config)


def test_tampered_final_model_is_refused(final_model, tmp_path):
    _, model_dir, _ = final_model
    copy = tmp_path / "final_model"
    copy.mkdir()
    (copy / "model.json").write_text((model_dir / "model.json").read_text())
    (copy / "model.joblib").write_bytes((model_dir / "model.joblib").read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        pipeline.load_final(copy)


def test_case_study_reports_probabilities_and_never_accuracy(final_model, tmp_path):
    _, model_dir, _ = final_model
    rng = np.random.default_rng(0)
    tables = tmp_path / "tables"
    tables.mkdir()
    z = rng.normal(size=(400, 3))
    y = z[:, 0] + rng.normal(size=400)
    y[z[:, 1] > 0.8] = np.nan
    pd.DataFrame({"a": z[:, 0], "b": z[:, 1], "c": z[:, 2], "y": y}).to_csv(tables / "toy.csv", index=False)
    base = {"source": "toy", "target": "y", "missing_rule": "natural", "z_include": "", "z_exclude": "",
            "hypothesis": "MAR", "evidence": "construção", "citation": "-", "exclusion_reason": ""}  # fmt: skip
    registry = tmp_path / "registry.csv"
    pd.DataFrame([
        base | {"case_id": "toy_y", "include": "True"},
        base | {"case_id": "toy_out", "include": "False", "exclusion_reason": "duplicata"},
        base | {"case_id": "toy_bad", "target": "nope", "include": "True"},
    ]).to_csv(registry, index=False)  # fmt: skip
    config = parse_config({"name": "case-study", "root_seed": 1, "output_root": str(tmp_path / "results")})
    out = pipeline.run_case_study(model_dir, registry, tables, config, n_jobs=1)
    cases = pd.read_csv(out / "case_predictions.csv").set_index("case_id")
    assert cases.loc["toy_y", "status"] == "ok"
    assert cases.loc[["toy_y"], ["p_MCAR", "p_MAR", "p_MNAR"]].sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert cases.loc["toy_bad", "status"].startswith("failed")
    assert not any(w in c.lower() for c in cases.columns for w in ("accuracy", "f1", "agree", "concord", "correct"))
    excluded = pd.read_csv(out / "excluded.csv")
    assert list(excluded.case_id) == ["toy_out"]
    log = json.loads((out / "run.case-study.json").read_text())
    assert log["model_sha256"] == json.loads((model_dir / "model.json").read_text())["model_sha256"]


def test_final_model_columns_and_schema_are_protected(final_model, tmp_path, monkeypatch):
    _, model_dir, _ = final_model
    copy = tmp_path / "final_model"
    copy.mkdir()
    (copy / "model.joblib").write_bytes((model_dir / "model.joblib").read_bytes())
    meta = json.loads((model_dir / "model.json").read_text())
    meta["feature_columns"] = list(reversed(meta["feature_columns"]))
    (copy / "model.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="colunas"):
        pipeline.load_final(copy)
    monkeypatch.setattr(pipeline.feat, "FEATURE_SCHEMA_VERSION", "outra")
    with pytest.raises(ValueError, match="schema"):
        pipeline.load_final(model_dir)


def test_case_study_checks_tables_against_the_manifest(final_model, tmp_path):
    _, model_dir, _ = final_model
    rng = np.random.default_rng(1)
    tables = tmp_path / "tables"
    tables.mkdir()
    z = rng.normal(size=(300, 2))
    y = z[:, 0] + rng.normal(size=300)
    y[z[:, 1] > 0.7] = np.nan
    pd.DataFrame({"a": z[:, 0], "b": z[:, 1], "y": y}).to_csv(tables / "toy.csv", index=False)
    pd.DataFrame({"a": z[:, 0], "b": z[:, 1], "y": y}).to_csv(tables / "other.csv", index=False)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"source": ["toy", "other"],
                  "table_sha256": [pipeline.sha256_path(tables / "toy.csv"), "0" * 64]}).to_csv(manifest, index=False)  # fmt: skip
    base = {"target": "y", "missing_rule": "natural", "z_include": "", "z_exclude": "", "hypothesis": "MAR",
            "evidence": "-", "citation": "-", "include": "True", "exclusion_reason": ""}  # fmt: skip
    registry = tmp_path / "registry.csv"
    pd.DataFrame([base | {"case_id": "ok", "source": "toy"}, base | {"case_id": "bad", "source": "other"}]).to_csv(
        registry, index=False
    )
    config = parse_config({"name": "case-study", "root_seed": 2, "output_root": str(tmp_path / "results")})
    out = pipeline.run_case_study(model_dir, registry, tables, config, n_jobs=1, manifest=manifest)
    cases = pd.read_csv(out / "case_predictions.csv").set_index("case_id")
    assert cases.loc["ok", "status"] == "ok"
    assert cases.loc["ok", "table_sha256"] == pipeline.sha256_path(tables / "toy.csv")
    assert "sha256" in cases.loc["bad", "status"]
    assert "z_determines_mask" in cases.columns


def test_noise_leaves_no_partial_output_when_an_analysis_fails(run, extended, monkeypatch, tmp_path):
    run_dir, config = run
    copy = tmp_path / run_dir.name
    shutil.copytree(run_dir, copy, ignore=shutil.ignore_patterns("noise", "run.noise.json"))
    monkeypatch.setattr(pipeline.noise, "rq3_tests", lambda per_family: (_ for _ in ()).throw(RuntimeError("falha")))
    with pytest.raises(RuntimeError, match="falha"):
        pipeline.run_noise(copy, config, n_jobs=1, n_seeds=1)
    assert not (copy / "noise").exists()


def test_cross_generator_uses_primary_partitions_and_tests_only_external_generators(extended):
    run_dir, _ = extended
    pred = pd.read_parquet(run_dir / "eval-cross" / "predictions.parquet")
    assert set(pred.method) == {"cross_direct_hgb", "cross_logit_simple", "ingen_direct_hgb", "mechdetect_obs"}
    assert check_pairing(pred)
    generator = pd.read_parquet(run_dir / "oracle.parquet").set_index("sample_id").generator
    assert (pred.sample_id.map(generator) != "symmetric").all()
    primary = pd.read_parquet(run_dir / "eval-symmetric" / "predictions.parquet")
    primary_folds = primary.groupby(["repeat", "fold"]).family_id.agg(set)
    for (repeat, fold), fams in pred.groupby(["repeat", "fold"]).family_id.agg(set).items():
        assert fams <= primary_folds[(repeat, fold)]  # mesmas famílias de teste da avaliação primária
    summary = pd.read_csv(run_dir / "eval-cross" / "cross_summary.csv")
    assert "loss_ingen_minus_cross" in set(summary.quantity)
