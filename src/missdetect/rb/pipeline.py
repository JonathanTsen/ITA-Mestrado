"""Etapas de um run: injeção, corpus sintético, descritores, baselines e avaliação (passos 30, 32–35, 37 e 39).

Cada etapa lê e grava no diretório do run (ver `runs.py`), nunca sobrescreve e registra `run.<etapa>.json`.
As etapas que veem as instâncias (descritores, MechDetect-O, Little, PKLM) leem só `observed/`; apenas a avaliação e
o painel oráculo leem `oracle.parquet` / `oracle_complete/`.
"""

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

from missdetect.rb import case_study, noise
from missdetect.rb import features as feat
from missdetect.rb.analyses import (
    bootstrap_mean_ci,
    boundary_sensitivity,
    descriptive_classifiers,
    l1_secondary,
    l2_evaluation,
    per_mechanism,
)
from missdetect.rb.baselines import little, mechdetect_obs
from missdetect.rb.baselines.logit_simple import LOGIT_SIMPLE_FEATURES
from missdetect.rb.baselines.oracle_descriptors import ORACLE_COLUMNS
from missdetect.rb.baselines.oracle_descriptors import compute as oracle_compute
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
from missdetect.rb.splits import outer_splits
from missdetect.rb.stats import compare_family, confirmatory
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
    # Famílias sem nenhuma base elegível saem do run por regra: ficam registradas aqui, nunca somem em silêncio.
    status = pd.DataFrame(
        [{"family_id": o.family_id, "n_bases": len(o.bases), "n_contexts": len(o.contexts), "n_instances": len(o.instances)}
         for o in outputs]
    )  # fmt: skip
    write_parquet(status, run_dir / "family_status.parquet")
    without = status[status.n_bases == 0].family_id.tolist()
    if without:
        print(f"famílias sem base elegível (excluídas por regra): {without}")
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
            ("mddirty", lambda: mechdetect_obs.dirty(instance, cfg.alpha, cfg.mechdetect_cv_folds)),
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
    oracle_features = run_dir / "oracle_features.parquet"
    if oracle_features.exists():
        frame = frame.merge(pd.read_parquet(oracle_features), on="sample_id", how="left")
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
    for name, prefix in RULE_METHODS.items():
        if f"{prefix}_decision" in frame:
            ok = frame[f"{prefix}_status"].eq("ok")
            proba = np.full((len(frame), N_CLASSES), np.nan)
            proba[ok.to_numpy()] = mechdetect_obs.decision_proba(frame.loc[ok, f"{prefix}_decision"])
            specs[name] = (Passthrough, proba)
    if all(c in frame for c in ORACLE_COLUMNS):
        X_oracle = frame[[*feature_columns(frame), *ORACLE_COLUMNS]].to_numpy(float)
        specs[f"direct_{ev.learner}_oracle_desc"] = (lambda: DirectClassifier(ev.learner, {}, seed), X_oracle)
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


RULE_METHODS = {"mechdetect_obs": "mdo", "mechdetect_oracle": "mdoracle", "mechdetect_dirty": "mddirty"}
# Painel de informação privilegiada ou degenerada: reportado à parte, nunca ranqueado com os métodos observed-only.
ORACLE_PANEL = ("mechdetect_oracle", "mechdetect_dirty")
ORACLE_SUFFIX = "_oracle_desc"


def is_oracle_panel(method: str) -> bool:
    return method in ORACLE_PANEL or method.endswith(ORACLE_SUFFIX)


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
    if "oracle_status" in frame:
        failed = set(frame.loc[~frame.oracle_status.eq("ok"), "sample_id"])
        hit = pred.method.str.endswith("_oracle_desc") & pred.sample_id.isin(failed)
        rows |= hit
        reason[hit] = "failed: oracle_descriptors"
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
    if not (run_dir / "oracle_features.parquet").exists():
        print("aviso: sem oracle_features.parquet; o painel oráculo não terá o classificador com descritores-oráculo")
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


def run_stats(run_dir: Path, config: RunConfig, llm_per_family: Path | None = None) -> Path:
    """H1–H4 com Holm (m = 4) a partir das métricas por família da avaliação no gerador principal."""
    out = run_dir / "stats_confirmatory.csv"
    if out.exists():
        raise FileExistsError(f"{out} já existe; não sobrescrevo.")
    main = pd.read_parquet(run_dir / "eval-symmetric" / "per_family.parquet")
    llm = pd.read_parquet(llm_per_family) if llm_per_family else None
    result = confirmatory(main, llm)
    result.to_csv(out, index=False)
    append_run_log(
        run_dir, config, "stats", {"hypotheses": result.hypothesis.tolist(), "llm_panel": bool(llm_per_family)}
    )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Painel oráculo, ablação, leave-one-generator-out, transferência e análises secundárias (pré-registro, itens 6–7)
# ---------------------------------------------------------------------------------------------------------------------
def _oracle_row(run_dir: Path, sample_id: str, context_id: str) -> dict:
    try:
        with threadpool_limits(1):
            row = oracle_compute(read_observed(run_dir, sample_id), read_complete(run_dir, context_id))
        row["oracle_status"] = "ok"
    except Exception as exc:
        row = {"sample_id": sample_id, "oracle_status": f"failed: {exc!r}"}
    return row


def run_oracle_features(run_dir: Path, config: RunConfig, n_jobs: int) -> Path:
    """Descritores-oráculo (com o Y completo) para o painel de informação privilegiada."""
    out = run_dir / "oracle_features.parquet"
    if out.exists():
        raise FileExistsError(f"{out} já existe; não sobrescrevo.")
    instances = pd.read_parquet(run_dir / "instances.parquet")
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_oracle_row)(run_dir, s, c) for s, c in zip(instances.sample_id, instances.context_id, strict=True)
    )
    write_parquet(pd.DataFrame(rows), out)
    append_run_log(run_dir, config, "oracle-features", {"n_instances": len(rows)})
    return out


def _write_eval(out_dir: Path, pred: pd.DataFrame, extra: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    out_dir.mkdir()
    write_parquet(pred, out_dir / "predictions.parquet")
    per_family, summary = summarize_by_family(pred)
    write_parquet(per_family, out_dir / "per_family.parquet")
    summary.reset_index().to_csv(out_dir / "summary.csv", index=False)
    for name, table in (extra or {}).items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    return per_family


GROUPS = ("g1_", "g2_", "g3_", "g4_")


def run_ablation(run_dir: Path, config: RunConfig, n_jobs: int) -> Path:
    """RQ2 (secundária, Holm m = 4): configuração primária com todos os descritores × sem o grupo G_k, no gerador
    principal, com as mesmas partições da avaliação primária."""
    out_dir = run_dir / "eval-ablation"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    frame = load_eval_frame(run_dir, ("symmetric",))
    learner = config.evaluate.learner
    seed = child_seed(config.root_seed, "model")
    cols = feature_columns(frame)
    specs = {f"direct_{learner}": (lambda: DirectClassifier(learner, {}, seed), frame[cols].to_numpy(float))}
    for k, prefix in enumerate(GROUPS, start=1):
        keep = [c for c in cols if not c.startswith(prefix)]
        specs[f"direct_{learner}_minus_G{k}"] = (
            lambda: DirectClassifier(learner, {}, seed),
            frame[keep].to_numpy(float),
        )
    results = Parallel(n_jobs=n_jobs)(
        delayed(_method_worker)(name, factory, X, frame, config) for name, (factory, X) in specs.items()
    )
    pred = mark_failures(pd.concat([p for p, _ in results], ignore_index=True), frame)
    if not check_pairing(pred):
        raise RuntimeError("pareamento 1:1 violado")
    per_family = summarize_by_family(pred)[0]
    hypotheses = {f"RQ2_G{k}": (f"direct_{learner}", f"direct_{learner}_minus_G{k}") for k in range(1, 5)}
    tests = compare_family(per_family, hypotheses)
    _write_eval(out_dir, pred, {"ablation_tests": tests})
    append_run_log(run_dir, config, "ablation", {"methods": list(specs), "n_instances": len(frame)})
    return out_dir


def run_logo(run_dir: Path, config: RunConfig, n_jobs: int) -> Path:
    """Leave-one-generator-out (exploratória): treino nos outros geradores das famílias de treino, teste no gerador
    deixado de fora das famílias de teste; mais a referência "mesmo gerador" (treino e teste no gerador g)."""
    out_dir = run_dir / "eval-logo"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    frame = load_eval_frame(run_dir)
    learner = config.evaluate.learner
    seed = child_seed(config.root_seed, "model")
    X = frame[feature_columns(frame)].to_numpy(float)
    y = frame.label.to_numpy()
    families = frame.family_id.reset_index(drop=True)
    generators = sorted(frame.generator.unique())
    tasks = [
        (g, mode, repeat, fold, train, test)
        for g in generators
        for mode in ("logo", "ingen")
        for repeat, fold, train, test in outer_splits(families, config.root_seed, config.evaluate.n_splits, config.evaluate.n_repeats)
    ]  # fmt: skip

    def task(g, mode, repeat, fold, train, test):
        gen = frame.generator.to_numpy()
        tr = train[gen[train] != g] if mode == "logo" else train[gen[train] == g]
        te = test[gen[test] == g]
        if len(te) == 0:
            return pd.DataFrame()
        status = "ok"
        try:
            if len(np.unique(families.iloc[tr])) < 4:
                raise ValueError("menos de 4 famílias de treino")
            with threadpool_limits(1):
                model = DirectClassifier(learner, {}, seed).fit(X[tr], y[tr], families.iloc[tr])
                proba = model.predict_proba(X[te])
        except Exception as exc:  # falha conta como erro, nunca some
            status = f"failed: {type(exc).__name__}: {exc}"
            proba = np.full((len(te), N_CLASSES), 1 / N_CLASSES)
        pred_class = proba.argmax(1) if status == "ok" else np.full(len(te), -1)
        return pd.DataFrame({
            "method": f"{mode}_{g}", "repeat": repeat, "fold": fold, "sample_id": frame.sample_id.iloc[te].to_numpy(),
            "family_id": families.iloc[te].to_numpy(), "y_true": y[te], "y_pred": pred_class,
            **{f"p_{c}": proba[:, k] for k, c in enumerate(CLASSES)}, "status": status,
        })  # fmt: skip

    pred = pd.concat(Parallel(n_jobs=n_jobs)(delayed(task)(*t) for t in tasks), ignore_index=True)
    pred = mark_failures(pred, frame)
    per_family = summarize_by_family(pred)[0]
    summary = []
    for method, g in per_family.groupby("method"):
        low, high = bootstrap_mean_ci(g.macro_recall.to_numpy())
        summary.append({"method": method, "n_families": len(g), "macro_recall_mean": g.macro_recall.mean(),
                        "ci_low": low, "ci_high": high})  # fmt: skip
    _write_eval(out_dir, pred, {"logo_summary": pd.DataFrame(summary)})
    append_run_log(run_dir, config, "logo", {"generators": generators})
    return out_dir


def run_transfer(synth_run: Path, real_run: Path, config: RunConfig) -> Path:
    """Transferência sintético → real (exploratória): a configuração primária é treinada só no corpus sintético (escolhas
    internas por CV agrupada por tabela-base), congelada e aplicada sem ajuste a todas as instâncias reais. Um modelo
    por conjunto de geradores de treino (todos e cada um), para separar mudança de domínio de mudança de gerador."""
    out_dir = real_run / f"transfer-{synth_run.name}"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    synth = load_eval_frame(synth_run)
    real = load_eval_frame(real_run)
    cols = feature_columns(real)
    if cols != feature_columns(synth):
        raise ValueError("descritores diferentes entre o corpus sintético e o run real")
    learner = config.evaluate.learner
    seed = child_seed(config.root_seed, "model")
    parts, skipped = [], []
    for train_set in ("all", *sorted(synth.generator.unique())):
        tr = synth if train_set == "all" else synth[synth.generator == train_set]
        if tr.family_id.nunique() < 4:  # a calibração interna exige 4 folds por família
            skipped.append(train_set)
            continue
        with threadpool_limits(1):
            model = DirectClassifier(learner, {}, seed).fit(
                tr[cols].to_numpy(float), tr.label.to_numpy(), tr.family_id.reset_index(drop=True)
            )
            proba = model.predict_proba(real[cols].to_numpy(float))
        parts.append(pd.DataFrame({
            "method": f"transfer_{train_set}", "repeat": 0, "fold": 0, "sample_id": real.sample_id.to_numpy(),
            "family_id": real.family_id.to_numpy(), "test_generator": real.generator.to_numpy(),
            "y_true": real.label.to_numpy(), "y_pred": proba.argmax(1),
            **{f"p_{c}": proba[:, k] for k, c in enumerate(CLASSES)}, "status": "ok",
        }))  # fmt: skip
    pred = mark_failures(pd.concat(parts, ignore_index=True), real)
    matrix = []
    for (method, test_gen), g in pred.groupby(["method", "test_generator"]):
        per_family = summarize_by_family(g.drop(columns="test_generator"))[0]
        low, high = bootstrap_mean_ci(per_family.macro_recall.to_numpy())
        matrix.append({"train": method.removeprefix("transfer_"), "test_generator": test_gen,
                       "n_families": len(per_family), "macro_recall_mean": per_family.macro_recall.mean(),
                       "ci_low": low, "ci_high": high})  # fmt: skip
    _write_eval(out_dir, pred.drop(columns="test_generator"), {"transfer_matrix": pd.DataFrame(matrix)})
    append_run_log(real_run, config, f"transfer-{synth_run.name}", {"synth_run": str(synth_run), "skipped": skipped})
    return out_dir


def run_secondary(run_dir: Path, config: RunConfig) -> Path:
    """L1 secundária, recall por mecanismo (H1, H2), avaliação do L2, classificadores descritivos (CD) e sensibilidade
    da massa na fronteira, a partir da avaliação primária já gravada."""
    out_dir = run_dir / "secondary"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    out_dir.mkdir()
    learner = config.evaluate.learner
    pred = pd.read_parquet(run_dir / "eval-symmetric" / "predictions.parquet")
    per_family = pd.read_parquet(run_dir / "eval-symmetric" / "per_family.parquet")
    l1_family, l1_tests = l1_secondary(run_dir, l1_method=f"hier_{learner}_fullprob")
    l1_family.to_csv(out_dir / "l1_per_family.csv", index=False)
    l1_tests.to_csv(out_dir / "l1_tests.csv", index=False)
    mech = []
    for name, (m, c) in {
        "H1": (f"direct_{learner}", "mechdetect_obs"),
        "H2": (f"hier_{learner}_fullprob", f"direct_{learner}"),
    }.items():
        if {m, c} <= set(per_family.method):
            mech.append(per_mechanism(per_family, name, m, c))
    if mech:
        pd.concat(mech, ignore_index=True).to_csv(out_dir / "per_mechanism.csv", index=False)
    l2_evaluation(pred, method=f"hier_{learner}_fullprob").to_csv(out_dir / "l2_evaluation.csv", index=False)
    descriptive = tuple(f"direct_{m}" for m in (learner, *config.evaluate.descriptive_learners))
    if len(set(descriptive) & set(per_family.method)) >= 3:  # Friedman exige pelo menos 3 classificadores
        cd = descriptive_classifiers(per_family, out_dir / "cd_diagram.png", methods=descriptive)
        cd.pop("nemenyi_p").to_csv(out_dir / "nemenyi_p.csv")
        write_json(out_dir / "descriptive_classifiers.json", cd)
    boundary_sensitivity(run_dir, pred, f"direct_{learner}").to_csv(out_dir / "boundary_sensitivity.csv", index=False)
    append_run_log(run_dir, config, "secondary", {"outputs": sorted(p.name for p in out_dir.iterdir())})
    return out_dir


# ---------------------------------------------------------------------------------------------------------------------
# RQ3: ruído de rótulo controlado (passo 44; pré-registro, item 6)
# ---------------------------------------------------------------------------------------------------------------------
def _noise_worker(setting, X, y, frame: pd.DataFrame, config: RunConfig):
    kind, rate, seed = setting
    noisy = y.copy()
    if kind != "clean":
        rng = child_rng(config.root_seed, "noise", kind, rate, seed)
        noisy = noise.corrupt_labels(y, frame.context_id, frame.family_id, rate, kind, rng)
    ev = config.evaluate
    pred, flags = noise.run_setting(
        X, y, noisy, frame.family_id, frame.sample_id, config.root_seed, ev.learner,
        child_seed(config.root_seed, "model"), ev.n_splits, ev.n_repeats,
    )  # fmt: skip
    cell = {"noise_kind": kind, "noise_rate": rate, "noise_seed": seed}
    realized = {
        **cell,
        "n": len(y),
        "n_corrupted": int((noisy != y).sum()),
        "realized_rate": float(np.mean(noisy != y)),
    }
    return pred.assign(**cell), flags.assign(**cell), realized


def _labels_sha(y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(y, dtype=np.int64).tobytes()).hexdigest()


def _degradation_figure(curves: pd.DataFrame, path: Path) -> None:
    fig = Figure(figsize=(9, 3.6))
    for i, kind in enumerate(noise.KINDS):
        ax = fig.add_subplot(1, 2, i + 1)
        for mitigation in noise.MITIGATIONS:
            c = curves[(curves.noise_kind == kind) & (curves.mitigation == mitigation)].sort_values("noise_rate")
            ax.plot(c.noise_rate, c.macro_recall_mean, marker="o", label=mitigation)
            ax.fill_between(c.noise_rate, c.ci_low, c.ci_high, alpha=0.15)
        ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=1)
        ax.set_title("symmetric" if kind == "symmetric" else "MAR ↔ MNAR")
        ax.set_xlabel("training label noise rate")
        ax.set_ylabel("macro-recall (clean test)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def run_noise(run_dir: Path, config: RunConfig, n_jobs: int, n_seeds: int = noise.N_SEEDS) -> Path:
    """RQ3 (secundária, Holm m = 4): ruído de rótulo só no treino da configuração primária, no gerador principal, com
    as mesmas partições da avaliação primária; teste sempre com os rótulos verdadeiros (hash conferido antes e depois).
    """
    out_dir = run_dir / "noise"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    oracle_sha_before = sha256_path(run_dir / "oracle.parquet")
    frame = load_eval_frame(run_dir, ("symmetric",))
    frame = frame.merge(pd.read_parquet(run_dir / "bases.parquet")[["base_id", "r2_cv"]], on="base_id", how="left")
    X = frame[feature_columns(frame)].to_numpy(float)
    y = frame.label.to_numpy()
    labels_sha_before = _labels_sha(y)
    settings = noise.noise_settings(n_seeds)
    results = Parallel(n_jobs=n_jobs)(delayed(_noise_worker)(s, X, y, frame, config) for s in settings)
    labels_sha_after = _labels_sha(frame.label.to_numpy())
    oracle_sha_after = sha256_path(run_dir / "oracle.parquet")
    if (labels_sha_before, oracle_sha_before) != (labels_sha_after, oracle_sha_after):
        raise RuntimeError("os rótulos verdadeiros mudaram durante a RQ3")
    pred = pd.concat([p for p, _, _ in results], ignore_index=True)
    pred = mark_failures(pred.assign(method="noise_" + pred.mitigation), frame).drop(columns="method")
    flags = pd.concat([f for _, f, _ in results], ignore_index=True)
    per_family = noise.per_family_noise(pred)
    curves = noise.degradation_curves(per_family)
    by_beta, by_r2 = noise.overlap_flags(flags, frame[["sample_id", "beta", "r2_cv"]])
    out_dir.mkdir()
    write_parquet(pred, out_dir / "predictions.parquet")
    write_parquet(flags, out_dir / "flags.parquet")
    write_parquet(per_family, out_dir / "per_family.parquet")
    noise.rq3_tests(per_family).to_csv(out_dir / "rq3_tests.csv", index=False)
    curves.to_csv(out_dir / "degradation_curves.csv", index=False)
    noise.flag_summary(flags).to_csv(out_dir / "flag_summary.csv", index=False)
    by_beta.to_csv(out_dir / "overlap_by_beta.csv", index=False)
    by_r2.to_csv(out_dir / "overlap_by_r2.csv", index=False)
    pd.DataFrame([r for _, _, r in results]).to_csv(out_dir / "settings.csv", index=False)
    _degradation_figure(curves, out_dir / "degradation.png")
    append_run_log(
        run_dir, config, "noise",
        {"settings": [list(s) for s in settings], "n_instances": len(frame),
         "labels_sha256_before": labels_sha_before, "labels_sha256_after": labels_sha_after,
         "oracle_sha256_before": oracle_sha_before, "oracle_sha256_after": oracle_sha_after},
    )  # fmt: skip
    return out_dir


# ---------------------------------------------------------------------------------------------------------------------
# Modelo final congelado e estudo de caso com ausência natural (passo 45; decisão D3)
# ---------------------------------------------------------------------------------------------------------------------
def run_fit_final(run_dir: Path, config: RunConfig) -> Path:
    """Configuração primária treinada em todas as instâncias do gerador principal do run, congelada com SHA-256."""
    out_dir = run_dir / "final_model"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    frame = load_eval_frame(run_dir, ("symmetric",))
    frame = frame[frame.feature_status.eq("ok")].reset_index(drop=True)
    cols = feature_columns(frame)
    learner = config.evaluate.learner
    with threadpool_limits(1):
        model = DirectClassifier(learner, {}, child_seed(config.root_seed, "model")).fit(
            frame[cols].to_numpy(float), frame.label.to_numpy(), frame.family_id
        )
    out_dir.mkdir()
    joblib.dump(model, out_dir / "model.joblib")
    meta = {
        "model_sha256": sha256_path(out_dir / "model.joblib"), "source_run": run_dir.name, "learner": learner,
        "feature_schema": feat.FEATURE_SCHEMA_VERSION, "feature_columns": cols, "generator": "symmetric",
        "n_instances": len(frame), "families": sorted(int(f) for f in frame.family_id.unique()),
        "k_range": [int(frame.k.min()), int(frame.k.max())], "choices": model.choices_,
    }  # fmt: skip
    write_json(out_dir / "model.json", meta)
    append_run_log(run_dir, config, "fit-final", {k: meta[k] for k in ("model_sha256", "n_instances", "families")})
    return out_dir


def load_final(model_dir: Path) -> tuple[DirectClassifier, dict]:
    """Carrega o modelo congelado só se o SHA-256 do arquivo conferir com o registrado."""
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if sha256_path(model_dir / "model.joblib") != meta["model_sha256"]:
        raise ValueError(f"o hash do modelo em {model_dir} não confere com model.json")
    return joblib.load(model_dir / "model.joblib"), meta


def _case_row(spec: dict, tables_dir: Path, model, meta: dict) -> dict:
    row = {k: spec[k] for k in ("case_id", "source", "target", "missing_rule", "hypothesis", "evidence", "citation")}
    try:
        table = pd.read_csv(tables_dir / f"{spec['source']}.csv")
        instance, info = case_study.build_case(table, spec)
        with threadpool_limits(1):
            descriptors = feat.compute(instance)
            proba = model.predict_proba(np.array([[descriptors[c] for c in meta["feature_columns"]]], dtype=float))[0]
        try:
            little_p = little.run(instance)["little_p"]
        except ValueError:
            little_p = np.nan
        low, high = meta["k_range"]
        row |= info | {f"p_{c}": float(proba[i]) for i, c in enumerate(CLASSES)}
        row |= {"most_probable": CLASSES[int(np.argmax(proba))], "k_in_training_range": low <= info["k"] <= high,
                "little_p": little_p, "status": "ok"}  # fmt: skip
    except Exception as exc:  # o caso fica registrado com o motivo da falha, nunca some
        row |= {"status": f"failed: {type(exc).__name__}: {exc}"}
    return row


def run_case_study(model_dir: Path, registry: Path, tables_dir: Path, config: RunConfig, n_jobs: int) -> Path:
    """Aplica o modelo congelado, sem ajuste, aos casos do registro: P(MCAR), P(MAR) e P(MNAR) por coluna, ao lado da
    hipótese documentada na literatura. Nenhuma métrica de acerto é calculada (VIT-07)."""
    model, meta = load_final(model_dir)
    specs = case_study.parse_registry(registry)
    run_dir = new_run_dir(config)
    included = [s for s in specs if s["include"]]
    rows = Parallel(n_jobs=n_jobs)(delayed(_case_row)(s, tables_dir, model, meta) for s in included)
    cases = pd.DataFrame(rows).assign(model_sha256=meta["model_sha256"])
    cases.to_csv(run_dir / "case_predictions.csv", index=False)
    excluded = pd.DataFrame(
        [{"case_id": s["case_id"], "exclusion_reason": s["exclusion_reason"]} for s in specs if not s["include"]],
        columns=["case_id", "exclusion_reason"],
    )
    excluded.to_csv(run_dir / "excluded.csv", index=False)
    append_run_log(
        run_dir, config, "case-study",
        {"model_dir": str(model_dir), "model_sha256": meta["model_sha256"], "model_source_run": meta["source_run"],
         "registry": str(registry), "registry_sha256": sha256_path(registry), "tables_dir": str(tables_dir),
         "n_cases": len(included), "n_excluded": len(excluded)},
    )  # fmt: skip
    return run_dir
