"""Etapas de um run: injeção, corpus sintético, descritores, baselines e avaliação (passos 30, 32–35, 37 e 39).

Cada etapa lê e grava no diretório do run (ver `runs.py`), nunca sobrescreve e registra `run.<etapa>.json`.
As etapas que veem as instâncias (descritores, MechDetect-O, Little, PKLM) leem só `observed/`; apenas a avaliação e
o painel oráculo leem `oracle.parquet` / `oracle_complete/`.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from missdetect.rb import features as feat
from missdetect.rb.baselines import little, mechdetect_obs
from missdetect.rb.baselines.logit_simple import LOGIT_SIMPLE_FEATURES
from missdetect.rb.baselines.pklm_r import run_pklm
from missdetect.rb.bench import generate_family, load_sentinel_exclusions
from missdetect.rb.config import RunConfig
from missdetect.rb.evaluate import (
    N_CLASSES,
    Passthrough,
    check_pairing,
    run_method,
    summarize_by_family,
    trivial_factory,
)
from missdetect.rb.ids import child_rng, child_seed
from missdetect.rb.models import CLASSES, DirectClassifier, HierarchicalClassifier
from missdetect.rb.runs import (
    append_run_log,
    new_run_dir,
    read_complete,
    read_observed,
    sha256_path,
    write_json,
    write_parquet,
)
from missdetect.rb.synth import design as synth_design
from missdetect.rb.synth import generate_base

FEATURE_CHUNK = 100
NUISANCE = ("g0_missing_rate", "meta_n", "meta_k", "meta_n_unique_yobs")


# ---------------------------------------------------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------------------------------------------------
def _family_worker(run_dir: Path, csv_dir: Path, source: pd.Series, config: RunConfig, excluded: frozenset[str]):
    with threadpool_limits(1):
        return generate_family(run_dir, csv_dir, source, config.inject, config.root_seed, excluded)


def run_inject(config: RunConfig, n_jobs: int) -> Path:
    sources = pd.read_csv(config.sources.table)
    chosen = sources[sources.representative & sources.role.isin(config.sources.roles)].sort_values("family_id")
    if chosen.empty:
        raise ValueError("nenhuma família com os papéis pedidos")
    exclusions = load_sentinel_exclusions(config.sources.sentinel_exclusions)
    run_dir = new_run_dir(config)
    append_run_log(
        run_dir,
        config,
        "inject",
        {
            "families": chosen.family_id.astype(int).tolist(),
            "sentinel_exclusions": {str(k): sorted(v) for k, v in exclusions.items()},
        },
    )
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_family_worker)(
            run_dir, Path(config.sources.csv_dir), row, config, frozenset(exclusions.get(int(row.openml_id), set()))
        )
        for _, row in chosen.iterrows()
    )
    _write_generation(run_dir, outputs, extra_tables=("eligibility", "skipped"))
    write_parquet(
        pd.DataFrame([{"family_id": o.family_id, "seconds": o.seconds} for o in outputs]),
        run_dir / "timing_inject.parquet",
    )
    return run_dir


def _write_generation(run_dir: Path, outputs: list, extra_tables: tuple[str, ...] = ()) -> None:
    for name in ("bases", "contexts", "instances", "oracle", *extra_tables):
        rows = [r for o in outputs for r in getattr(o, name)]
        frame = pd.DataFrame(rows)
        if name == "instances" and frame.sample_id.duplicated().any():
            raise RuntimeError("sample_id duplicado")
        write_parquet(frame, run_dir / f"{name}.parquet")


def _synth_worker(run_dir: Path, items: list, config: RunConfig):
    with threadpool_limits(1):
        return [generate_base(run_dir, item, config.synth, config.root_seed) for item in items]


def run_synth(config: RunConfig, n_jobs: int) -> Path:
    run_dir = new_run_dir(config)
    append_run_log(run_dir, config, "synth")
    items = synth_design(config.synth, config.root_seed)
    chunks = [items[i : i + 25] for i in range(0, len(items), 25)]
    nested = Parallel(n_jobs=n_jobs)(delayed(_synth_worker)(run_dir, chunk, config) for chunk in chunks)
    outputs = [o for group in nested for o in group]
    _write_generation(run_dir, outputs)
    instances = pd.read_parquet(run_dir / "instances.parquet")
    oracle = pd.read_parquet(run_dir / "oracle.parquet")
    manifest = instances.merge(oracle[["sample_id", "mask_sha256"]], on="sample_id")
    manifest["observed_sha256"] = [sha256_path(run_dir / p) for p in manifest.observed_path]
    write_parquet(manifest, run_dir / "synth_manifest.parquet")
    return run_dir


# ---------------------------------------------------------------------------------------------------------------------
# Descritores
# ---------------------------------------------------------------------------------------------------------------------
def _feature_row(run_dir: Path, sample_id: str) -> dict:
    t0 = time.perf_counter()
    instance = read_observed(run_dir, sample_id)
    y_obs = instance.y[~np.isnan(instance.y)]
    meta = {"meta_n": len(instance.y), "meta_k": instance.z.shape[1], "meta_n_unique_yobs": len(np.unique(y_obs))}
    try:
        with threadpool_limits(1):
            row = {**feat.compute(instance), "feature_status": "ok"}
    except Exception as exc:  # falha vira status, nunca é descartada (teste 7 do passo 30)
        row = {"sample_id": sample_id, "schema": feat.FEATURE_SCHEMA_VERSION, "feature_status": f"failed: {exc!r}"}
    return {**row, **meta, "feature_seconds": time.perf_counter() - t0}


def run_features(run_dir: Path, config: RunConfig, n_jobs: int) -> Path:
    """Descritores de todas as instâncias, com checkpoint por `sample_id` em `features_parts/`."""
    out = run_dir / "features.parquet"
    if out.exists():
        raise FileExistsError(f"{out} já existe; não sobrescrevo.")
    parts = run_dir / "features_parts"
    parts.mkdir(exist_ok=True)
    done = set()
    for part in sorted(parts.glob("part-*.parquet")):
        done |= set(pd.read_parquet(part, columns=["sample_id"]).sample_id)
    todo = [s for s in pd.read_parquet(run_dir / "instances.parquet").sample_id if s not in done]
    start = len(list(parts.glob("part-*.parquet")))
    for j, i in enumerate(range(0, len(todo), FEATURE_CHUNK * n_jobs)):
        chunk = todo[i : i + FEATURE_CHUNK * n_jobs]
        rows = Parallel(n_jobs=n_jobs)(delayed(_feature_row)(run_dir, s) for s in chunk)
        write_parquet(pd.DataFrame(rows), parts / f"part-{start + j:05d}.parquet")
    frame = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part-*.parquet"))], ignore_index=True)
    if frame.sample_id.duplicated().any():
        raise RuntimeError("sample_id duplicado nos checkpoints de descritores")
    write_parquet(frame, out)
    write_json(run_dir / "feature_schema.json", {"version": feat.FEATURE_SCHEMA_VERSION, "features": feat.schema()})
    append_run_log(run_dir, config, "features", {"n_instances": len(frame), "n_jobs": n_jobs})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------------------------------
def _baseline_row(run_dir: Path, sample_id: str, context_id: str, config: RunConfig) -> dict:
    cfg = config.baselines
    instance = read_observed(run_dir, sample_id)
    row: dict = {"sample_id": sample_id}
    with threadpool_limits(1):
        for name, fn in (
            ("mdo", lambda: mechdetect_obs.observed_only(instance, cfg.alpha, cfg.mechdetect_cv_folds)),
            ("mdoracle", lambda: mechdetect_obs.oracle_clean(
                instance, read_complete(run_dir, context_id), cfg.alpha, cfg.mechdetect_cv_folds)),
            ("little", lambda: little.run(instance)),
        ):  # fmt: skip
            t0 = time.perf_counter()
            try:
                row.update(fn())
                row[f"{name}_status"] = "ok"
            except Exception as exc:
                row[f"{name}_status"] = f"failed: {exc!r}"
            row[f"{name}_seconds"] = time.perf_counter() - t0
    return row


def pklm_subset(instances: pd.DataFrame, oracle: pd.DataFrame, max_instances: int | None, seed: int) -> list[str]:
    """Contextos inteiros do gerador primário, sorteados igualmente por família (todas as instâncias se None)."""
    frame = instances.merge(oracle[["sample_id", "generator"]], on="sample_id")
    frame = frame[frame.generator == "symmetric"]
    if max_instances is None:
        return frame.sample_id.tolist()
    rng = child_rng(seed, "pklm_subset")
    families = sorted(frame.family_id.unique())
    per_family = max(1, max_instances // (3 * len(families)))
    chosen = []
    for family in families:
        contexts = np.sort(frame[frame.family_id == family].context_id.unique())
        pick = set(rng.choice(contexts, size=min(per_family, len(contexts)), replace=False))
        chosen += frame[frame.context_id.isin(pick)].sample_id.tolist()
    return chosen


def run_baselines(
    run_dir: Path, config: RunConfig, n_jobs: int, skip_pklm: bool = False, only_pklm: bool = False
) -> Path:
    """MechDetect-O, painel oráculo e Little em `baselines.parquet`; PKLM em `pklm.parquet` (etapas separáveis)."""
    instances = pd.read_parquet(run_dir / "instances.parquet")
    extra: dict = {"n_jobs": n_jobs}
    if not only_pklm:
        out = run_dir / "baselines.parquet"
        if out.exists():
            raise FileExistsError(f"{out} já existe; não sobrescrevo.")
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_baseline_row)(run_dir, s, c, config)
            for s, c in zip(instances.sample_id, instances.context_id, strict=True)
        )
        write_parquet(pd.DataFrame(rows), out)
        extra["baselines_instances"] = len(rows)
    if not skip_pklm:
        out = run_dir / "pklm.parquet"
        if out.exists():
            raise FileExistsError(f"{out} já existe; não sobrescrevo.")
        oracle = pd.read_parquet(run_dir / "oracle.parquet")
        subset = pklm_subset(instances, oracle, config.baselines.pklm_max_instances, config.root_seed)
        cfg = config.baselines
        write_parquet(
            run_pklm(run_dir, subset, cfg.pklm_num_proj, cfg.pklm_nrep, config.root_seed % 2**31, n_jobs), out
        )
        extra["pklm_instances"] = len(subset)
    label = "baselines" if not only_pklm else "baselines-pklm"
    append_run_log(run_dir, config, label, extra)
    return run_dir


# ---------------------------------------------------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------------------------------------------------
def load_eval_frame(run_dir: Path, generators: tuple[str, ...] = ()) -> pd.DataFrame:
    """Descritores + rótulo + fatores do contexto (a avaliação é a única etapa que lê o oráculo)."""
    features = pd.read_parquet(run_dir / "features.parquet")
    instances = pd.read_parquet(run_dir / "instances.parquet")
    oracle = pd.read_parquet(run_dir / "oracle.parquet")
    contexts = pd.read_parquet(run_dir / "contexts.parquet")
    frame = (
        instances.merge(features, on="sample_id", how="left")
        .merge(oracle[["sample_id", "mechanism", "generator", "variant", "deterministic"]], on="sample_id")
        .merge(contexts[["context_id", "n", "k", "rate", "beta", "direction", "base_id"]], on="context_id")
    )
    baselines = run_dir / "baselines.parquet"
    if baselines.exists():
        frame = frame.merge(pd.read_parquet(baselines), on="sample_id", how="left")
    if generators:
        frame = frame[frame.generator.isin(generators)]
    frame["label"] = frame.mechanism.map({m: i for i, m in enumerate(CLASSES)}).astype(int)
    return frame.sort_values("sample_id").reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c[:3] in ("g0_", "g1_", "g2_", "g3_", "g4_")]


def method_specs(frame: pd.DataFrame, config: RunConfig) -> dict:
    """Métodos avaliados: nome → (fábrica, matriz X). Todos passam pelo mesmo `run_method`."""
    ev = config.evaluate
    seed = child_seed(config.root_seed, "model")
    X_all = frame[feature_columns(frame)].to_numpy(float)
    specs: dict = {}
    for kind in ("uniform", "majority", "prior"):
        specs[f"trivial_{kind}"] = (trivial_factory(kind, seed), X_all)
    specs[f"direct_{ev.learner}"] = (lambda: DirectClassifier(ev.learner, {}, seed), X_all)
    for routing in ev.routings:
        specs[f"hier_{ev.learner}_{routing}"] = (
            lambda routing=routing: HierarchicalClassifier(ev.learner, {}, routing, seed),
            X_all,
        )
    for learner in ev.descriptive_learners:
        specs[f"direct_{learner}"] = (lambda learner=learner: DirectClassifier(learner, {}, seed), X_all)
    specs["logit_simple"] = (
        lambda: DirectClassifier("logreg", {}, seed),
        frame[list(LOGIT_SIMPLE_FEATURES)].to_numpy(float),
    )
    g4 = [c for c in frame.columns if c.startswith("g4_")]
    specs[f"g4only_{ev.learner}"] = (lambda: DirectClassifier(ev.learner, {}, seed), frame[g4].to_numpy(float))
    nuisance = frame[list(NUISANCE)].copy()
    for generator in sorted(frame.generator.unique()):
        nuisance[f"gen_{generator}"] = (frame.generator == generator).astype(float)
    specs[f"nuisance_{ev.learner}"] = (lambda: DirectClassifier(ev.learner, {}, seed), nuisance.to_numpy(float))
    if "mdo_decision" in frame:
        for name, prefix in RULE_METHODS.items():
            ok = frame[f"{prefix}_status"].eq("ok")
            proba = np.full((len(frame), N_CLASSES), np.nan)
            proba[ok.to_numpy()] = mechdetect_obs.decision_proba(frame.loc[ok, f"{prefix}_decision"])
            specs[name] = (Passthrough, proba)
    return specs


def _method_worker(name: str, factory, X, frame: pd.DataFrame, config: RunConfig):
    ev = config.evaluate
    with threadpool_limits(1):
        t0 = time.perf_counter()
        pred, choices = run_method(
            name, factory, X, frame.label.to_numpy(), frame.family_id, frame.sample_id, config.root_seed,
            ev.n_splits, ev.n_repeats,
        )  # fmt: skip
        pred["seconds"] = time.perf_counter() - t0
    return pred, choices


RULE_METHODS = {"mechdetect_obs": "mdo", "mechdetect_oracle": "mdoracle"}


def mark_failures(pred: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Instâncias em que a entrada do método falhou contam como erro (teste 7 do passo 30).

    Métodos com descritores: falha na extração. Métodos de regra: falha no baseline. Triviais: nunca falham."""
    feature_failed = set(frame.loc[~frame.feature_status.eq("ok"), "sample_id"])
    rows = pred.sample_id.isin(feature_failed) & ~pred.method.isin(list(RULE_METHODS))
    rows &= ~pred.method.str.startswith("trivial_")
    reason = pd.Series("failed: features", index=pred.index)
    for method, prefix in RULE_METHODS.items():
        if f"{prefix}_status" in frame:
            failed = set(frame.loc[~frame[f"{prefix}_status"].eq("ok"), "sample_id"])
            hit = pred.method.eq(method) & pred.sample_id.isin(failed)
            rows |= hit
            reason[hit] = f"failed: {prefix}"
    pred.loc[rows, [f"p_{c}" for c in CLASSES]] = 1 / N_CLASSES
    pred.loc[rows, "y_pred"] = -1
    pred.loc[rows, "status"] = reason[rows]
    return pred


def run_evaluate(run_dir: Path, config: RunConfig, n_jobs: int, generators: tuple[str, ...] = ()) -> Path:
    label = "eval-" + ("-".join(generators) if generators else "all")
    out_dir = run_dir / label
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    frame = load_eval_frame(run_dir, generators)
    specs = method_specs(frame, config)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_method_worker)(name, factory, X, frame, config) for name, (factory, X) in specs.items()
    )
    pred = pd.concat([p for p, _ in results], ignore_index=True)
    pred = mark_failures(pred, frame)
    if not check_pairing(pred):
        raise RuntimeError("pareamento 1:1 violado")
    out_dir.mkdir()
    write_parquet(pred, out_dir / "predictions.parquet")
    per_family, summary = summarize_by_family(pred)
    write_parquet(per_family, out_dir / "per_family.parquet")
    summary.reset_index().to_csv(out_dir / "summary.csv", index=False)
    write_json(out_dir / "choices.json", {"choices": [c for _, cs in results for c in cs]})
    append_run_log(run_dir, config, label, {"n_instances": len(frame), "methods": list(specs), "n_jobs": n_jobs})
    return out_dir
