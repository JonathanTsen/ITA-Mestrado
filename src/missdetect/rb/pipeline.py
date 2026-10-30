"""Etapas de um run: injeção, corpus sintético, descritores, baselines e avaliação (passos 30, 32–35, 37 e 39).

Cada etapa lê e grava no diretório do run (ver `runs.py`), nunca sobrescreve e registra `run.<etapa>.json`.
As etapas que veem as instâncias (descritores, MechDetect-O, Little, PKLM) leem só `observed/`; apenas a avaliação e
o painel oráculo leem `oracle.parquet` / `oracle_complete/`.
"""

import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

from missdetect.rb import ampute as amp
from missdetect.rb import case_study, noise
from missdetect.rb import extra_analyses as extra
from missdetect.rb import features as feat
from missdetect.rb.analyses import (
    bootstrap_mean_ci,
    boundary_sensitivity,
    descriptive_classifiers,
    generator_family_tests,
    l1_decisions,
    l1_per_family,
    l1_secondary,
    l2_evaluation,
    mar_mnar_auc,
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
    instance_row,
    new_run_dir,
    observed_path,
    oracle_row,
    read_complete,
    read_observed,
    repo_root,
    sha256_path,
    write_json,
    write_observed,
    write_parquet,
    write_tables_atomically,
)
from missdetect.rb.splits import outer_splits
from missdetect.rb.stats import (
    H4,
    PRIMARY,
    bootstrap_hl_ci,
    compare_family,
    confirmatory,
    hodges_lehmann,
    level_estimate,
    llm_hypothesis,
    llm_sensitivity_row,
    paired_differences,
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


def _h3_decomposition(per_family: pd.DataFrame) -> pd.DataFrame:
    """Descritivo: separa a diferença da H3 em aprendiz (HGB × logística, 28 descritores) e descritores (28 × 4, com
    logística), por Hodges–Lehmann com IC por bootstrap. Não é teste."""
    rows = []
    parts = {"learner": (PRIMARY, "direct_logreg"), "descriptors": ("direct_logreg", "logit_simple")}
    for component, (a, b) in parts.items():
        if {a, b} <= set(per_family.method):
            d = paired_differences(per_family, a, b).to_numpy()
            low, high = bootstrap_hl_ci(d)
            rows.append({"component": component, "method": a, "comparator": b, "n_families": len(d),
                         "hodges_lehmann": hodges_lehmann(d), "ci_low": low, "ci_high": high})  # fmt: skip
    columns = ["component", "method", "comparator", "n_families", "hodges_lehmann", "ci_low", "ci_high"]
    return pd.DataFrame(rows, columns=columns)


def run_stats(run_dir: Path, config: RunConfig) -> Path:
    """Família confirmatória principal (H1, H2, H2e, H3, H5, H6; Holm m = 6). A H4 fica no `stats-llm`.

    Lê `eval-symmetric/` (H1–H3 e a AUC MAR × MNAR da H6) e `eval-cross/` (H5). Sem o `eval-cross`, a etapa para:
    é um passo esquecido (rode `cross-generator` antes), não uma hipótese que não pode rodar. Uma hipótese sem dados
    entra no Holm com p = 1, com o motivo (pré-registro, seção 5).

    As saídas são gravadas juntas (C1): `stats_confirmatory.csv`, a última, marca a conclusão. Se a etapa for
    interrompida, nada fica pela metade, e ela pode ser refeita."""
    out = run_dir / "stats_confirmatory.csv"
    if out.exists():
        raise FileExistsError(f"{out} já existe; não sobrescrevo.")
    cross_path = run_dir / "eval-cross" / "per_family.parquet"
    if not cross_path.exists():
        raise FileNotFoundError(f"{cross_path} não existe: rode `cross-generator` antes do `stats` (H5)")
    main = pd.read_parquet(run_dir / "eval-symmetric" / "per_family.parquet")
    cross = pd.read_parquet(cross_path)
    pred = pd.read_parquet(run_dir / "eval-symmetric" / "predictions.parquet")
    auc = pd.concat([mar_mnar_auc(pred, m).assign(method=m) for m in (PRIMARY, "logit_simple")], ignore_index=True)
    result = confirmatory(main, cross, auc)
    level = [
        level_estimate(main, PRIMARY),
        level_estimate(cross, "cross_direct_hgb"),
        level_estimate(auc, PRIMARY, "auc"),
    ]
    write_tables_atomically(
        run_dir,
        {
            "stats_level.csv": pd.DataFrame(level),
            "stats_h6_auc_by_family.csv": auc,
            "stats_h3_decomposition.csv": _h3_decomposition(main),
            "stats_confirmatory.csv": result,  # por último: marcador de conclusão
        },
    )
    append_run_log(run_dir, config, "stats", {"hypotheses": result.hypothesis.tolist()})
    return out


def run_stats_llm(run_dir: Path, config: RunConfig, llm_per_family: Path, probes: Path | None = None) -> Path:
    """H4 (LLM-A1 × configuração primária, no painel de LLM; família própria, m = 1), numa etapa separada (C1).

    Roda depois do painel de LLM sem tocar a saída confirmatória principal (`stats_confirmatory.csv`).
    Memorização (P21), com o `results.csv` das sondas: as famílias reconhecidas na renderização cega saem da H4 (sem
    reposição); o reconhecimento da tabela crua é covariável, e a H4 restrita às famílias não reconhecidas na crua é
    reportada à parte, descritiva (`stats_llm_covariate.csv`). Num run confirmatório, as sondas são obrigatórias."""
    if probes is None and "confirm" in config.sources.roles:
        raise ValueError("stats-llm confirmatório precisa das sondas de memorização (--probes; pré-registro §4, P21)")
    per_family = pd.read_parquet(llm_per_family)
    excluded: list[int] = []
    raw_recognized: list[int] = []
    tables: dict[str, pd.DataFrame] = {}
    if probes is not None:
        results = pd.read_csv(probes)
        excluded = sorted(results.loc[(results.table == "blind") & results.recognized, "family_id"].astype(int))
        raw_recognized = sorted(results.loc[(results.table == "raw") & results.recognized, "family_id"].astype(int))
        per_family = per_family[~per_family.family_id.isin(excluded)]
        unrecognized = per_family[~per_family.family_id.isin(raw_recognized)]
        tables["stats_llm_covariate.csv"] = pd.DataFrame(
            [llm_sensitivity_row(unrecognized, H4[0], H4[1]) | {"hypothesis": "H4-não-reconhecidas-na-crua"}]
        )
    tables["stats_llm.csv"] = llm_hypothesis(per_family)
    write_tables_atomically(run_dir, tables)
    extra = {
        "llm_per_family": str(llm_per_family), "llm_per_family_sha256": sha256_path(llm_per_family),
        "probes": str(probes) if probes else None, "probes_sha256": sha256_path(probes) if probes else None,
        "excluded_families_blind": excluded, "recognized_families_raw": raw_recognized,
    }  # fmt: skip
    append_run_log(run_dir, config, "stats-llm", extra)
    return run_dir / "stats_llm.csv"


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
    # Painel oráculo pareado por gerador (Holm m = 3) e as estatísticas do MechDetect como descritores (m = 1)
    oracle = generator_family_tests(run_dir, f"direct_{learner}_oracle_desc", "mechdetect_oracle", "ORA")
    if oracle is not None:
        oracle.to_csv(out_dir / "oracle_panel_tests.csv", index=False)
    if {f"direct_{learner}", f"g4only_{learner}"} <= set(per_family.method):
        compare_family(per_family, {"G4": (f"direct_{learner}", f"g4only_{learner}")}).to_csv(
            out_dir / "g4_tests.csv", index=False
        )
    # Geradores externos contra o MechDetect-O, nas instâncias da H5 (m = 1)
    cross = run_dir / "eval-cross" / "per_family.parquet"
    if cross.exists():
        compare_family(pd.read_parquet(cross), {"EXT_mdo": ("cross_direct_hgb", "mechdetect_obs")}).to_csv(
            out_dir / "external_mdo_tests.csv", index=False
        )
    # Gerador da mdatagen com a sua própria CV (Holm m = 2)
    mdatagen = run_dir / "eval-mdatagen" / "per_family.parquet"
    if mdatagen.exists():
        hypotheses = {
            "MDG_simples": (f"direct_{learner}", "logit_simple"),
            "MDG_mdo": (f"direct_{learner}", "mechdetect_obs"),
        }
        compare_family(pd.read_parquet(mdatagen), hypotheses).to_csv(out_dir / "mdatagen_tests.csv", index=False)
    append_run_log(run_dir, config, "secondary", {"outputs": sorted(p.name for p in out_dir.iterdir())})
    return out_dir


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def run_secondary_extra(run_dir: Path, config: RunConfig, nominal_list: Path | None = None) -> Path:
    """Análises das emendas de 29/10 ao pré-registro (§0.4: P6, P7, P8, P11, P12, P13, P15), a partir das avaliações
    gravadas. Rode depois do `evaluate --generators symmetric`, do `cross-generator` e do `secondary`.

    As saídas vão juntas para `secondary-extra/` (C1): o último arquivo marca a conclusão, e nada é sobrescrito."""
    out_dir = run_dir / "secondary-extra"
    cross_dir = run_dir / "eval-cross"
    if not (cross_dir / "predictions.parquet").exists():
        raise FileNotFoundError(f"{cross_dir} não existe: rode `cross-generator` antes do `secondary-extra`")
    out_dir.mkdir(exist_ok=True)
    simple = "logit_simple"
    l1_method = f"hier_{config.evaluate.learner}_fullprob"
    sym = pd.read_parquet(run_dir / "eval-symmetric" / "predictions.parquet")
    sym_family = pd.read_parquet(run_dir / "eval-symmetric" / "per_family.parquet")
    cross = pd.read_parquet(cross_dir / "predictions.parquet")
    cross_family = pd.read_parquet(cross_dir / "per_family.parquet")
    instances = pd.read_parquet(run_dir / "instances.parquet")[["sample_id", "context_id"]]
    contexts = pd.read_parquet(run_dir / "contexts.parquet")
    regimes = instances.merge(contexts[["context_id", "beta", "direction"]], on="context_id")
    generators = pd.read_parquet(run_dir / "oracle.parquet")[["sample_id", "generator"]]
    nominal_path = nominal_list or repo_root() / "data_rb" / "nominal_targets.csv"
    excluded = extra.nominal_contexts(contexts, pd.read_csv(nominal_path))
    concordance = pd.concat([extra.trio_concordance(sym, instances, m) for m in (PRIMARY, simple)], ignore_index=True)
    errors = extra.decision_errors(sym)
    l1 = extra.l1_auc(
        sym, _read_optional(run_dir / "baselines.parquet"), _read_optional(run_dir / "pklm.parquet"), l1_method
    )
    tables = {
        "p6_recall_by_regime.csv": pd.concat([extra.recall_by_regime(sym, regimes, m) for m in (PRIMARY, simple)]),
        "p6_h6_auc_by_regime.csv": pd.concat([extra.h6_auc_by_regime(sym, regimes, m) for m in (PRIMARY, simple)]),
        "p7_h6_outside_generator.csv": extra.h6_outside_generator(cross),
        "p7_trio_concordance.csv": concordance,
        "p7_trio_concordance_summary.csv": extra.trio_concordance_summary(concordance),
        "p8_class_decomposition.csv": pd.concat(
            [
                extra.class_decomposition(cross_family, extra.CROSS_PRIMARY, extra.CROSS_SIMPLE, "H5"),
                extra.class_decomposition(sym_family, PRIMARY, simple, "gerador_principal"),
            ]
        ),
        "p8_generator_effect.csv": extra.generator_effect(cross, sym, regimes, generators),
        "p8_specialization.csv": extra.specialization(cross, generators),
        "p11_by_repeat.csv": extra.by_repeat(sym, cross),
        "p12_decision_errors.csv": errors,
        "p12_l1_false_mcar.csv": extra.l1_false_mcar(l1_per_family(l1_decisions(run_dir, l1_method=l1_method))),
        "p12_tests.csv": extra.decision_error_tests(errors),
        "p13_l1_auc.csv": l1,
        "p13_l1_auc_summary.csv": extra.l1_auc_summary(l1),
        "p15_nominal_sensitivity.csv": extra.without_contexts(sym, cross, instances, excluded),  # último: marcador
    }
    write_tables_atomically(out_dir, tables)
    append_run_log(
        run_dir, config, "secondary-extra",
        {"outputs": list(tables), "nominal_list": str(nominal_path), "nominal_list_sha256": sha256_path(nominal_path),
         "nominal_contexts_excluded": sorted(excluded)},
    )  # fmt: skip
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
    noise.check_test_labels(pred, frame.set_index("sample_id").label)  # os workers recebem cópias: confere aqui
    pred = mark_failures(pred.assign(method="noise_" + pred.mitigation), frame).drop(columns="method")
    flags = pd.concat([f for _, f, _ in results], ignore_index=True)
    per_family = noise.per_family_noise(pred)
    curves = noise.degradation_curves(per_family)
    tests = noise.rq3_tests(per_family)
    summary = noise.flag_summary(flags)
    by_beta, by_r2 = noise.overlap_flags(flags, frame[["sample_id", "beta", "r2_cv"]])
    out_dir.mkdir()  # só depois de todas as análises: uma falha acima não deixa `noise/` pela metade
    try:
        write_parquet(pred, out_dir / "predictions.parquet")
        write_parquet(flags, out_dir / "flags.parquet")
        write_parquet(per_family, out_dir / "per_family.parquet")
        tests.to_csv(out_dir / "rq3_tests.csv", index=False)
        curves.to_csv(out_dir / "degradation_curves.csv", index=False)
        summary.to_csv(out_dir / "flag_summary.csv", index=False)
        by_beta.to_csv(out_dir / "overlap_by_beta.csv", index=False)
        by_r2.to_csv(out_dir / "overlap_by_r2.csv", index=False)
        pd.DataFrame([r for _, _, r in results]).to_csv(out_dir / "settings.csv", index=False)
        _degradation_figure(curves, out_dir / "degradation.png")
    except Exception:
        shutil.rmtree(out_dir)
        raise
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
    """Configuração primária treinada nas instâncias do gerador principal do run com descritores válidos, congelada
    com SHA-256. O arquivo guarda o modelo junto com a ordem das colunas e a versão do schema, cobertos pelo hash."""
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
    joblib.dump(
        {"model": model, "feature_columns": cols, "feature_schema": feat.FEATURE_SCHEMA_VERSION},
        out_dir / "model.joblib",
    )
    meta = {
        "model_sha256": sha256_path(out_dir / "model.joblib"), "source_run": run_dir.name, "learner": learner,
        "feature_schema": feat.FEATURE_SCHEMA_VERSION, "feature_columns": cols, "generator": "symmetric",
        "n_instances": len(frame), "families": sorted(int(f) for f in frame.family_id.unique()),
        "k_range": [int(frame.k.min()), int(frame.k.max())], "n_range": [int(frame.n.min()), int(frame.n.max())],
        "rate_range": [float(frame.rate.min()), float(frame.rate.max())], "choices": model.choices_,
    }  # fmt: skip
    write_json(out_dir / "model.json", meta)
    append_run_log(run_dir, config, "fit-final", {k: meta[k] for k in ("model_sha256", "n_instances", "families")})
    return out_dir


def load_final(model_dir: Path) -> tuple[DirectClassifier, dict]:
    """Carrega o modelo congelado só se o SHA-256 do arquivo conferir com o registrado, se as colunas do `model.json`
    forem as guardadas no arquivo e se o schema dos descritores for o atual."""
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if sha256_path(model_dir / "model.joblib") != meta["model_sha256"]:
        raise ValueError(f"o hash do modelo em {model_dir} não confere com model.json")
    payload = joblib.load(model_dir / "model.joblib")
    if payload["feature_columns"] != meta["feature_columns"]:
        raise ValueError(f"as colunas de {model_dir / 'model.json'} não são as guardadas no modelo")
    if payload["feature_schema"] != feat.FEATURE_SCHEMA_VERSION:
        raise ValueError(f"schema do modelo {payload['feature_schema']} ≠ schema atual {feat.FEATURE_SCHEMA_VERSION}")
    return payload["model"], meta


def _in_range(value: float, bounds: list | None) -> bool | None:
    """Dentro da faixa do treino? None quando o modelo é anterior ao registro dessa faixa (C5, 29/10)."""
    return None if bounds is None else bool(bounds[0] <= value <= bounds[1])


def _case_prediction(instance, model, meta: dict) -> dict:
    with threadpool_limits(1):
        descriptors = feat.compute(instance)
        proba = model.predict_proba(np.array([[descriptors[c] for c in meta["feature_columns"]]], dtype=float))[0]
    try:
        little_p = little.run(instance)["little_p"]
    except ValueError:
        little_p = np.nan
    n, k, rate = len(instance.y), instance.z.shape[1], float(np.isnan(instance.y).mean())
    return {
        "n": n, "k": k, "missing_rate": rate, **{f"p_{c}": float(proba[i]) for i, c in enumerate(CLASSES)},
        "most_probable": CLASSES[int(np.argmax(proba))], "little_p": little_p,
        "n_in_training_range": _in_range(n, meta.get("n_range")), "k_in_training_range": _in_range(k, meta["k_range"]),
        "rate_in_training_range": _in_range(rate, meta.get("rate_range")),
    }  # fmt: skip


def _case_rows(spec: dict, tables_dir: Path, model, meta: dict, expected_sha: dict[str, str], seed: int) -> list[dict]:
    """Uma linha por caso, ou uma por subamostra se o caso tem mais de `SUBSAMPLE_N` linhas (D3, 29/10)."""
    row = {k: spec[k] for k in ("case_id", "source", "target", "missing_rule", "hypothesis", "evidence", "citation")}
    try:
        path = tables_dir / f"{spec['source']}.csv"
        row["table_sha256"] = sha256_path(path)
        if expected_sha and row["table_sha256"] != expected_sha.get(spec["source"]):
            raise ValueError("o sha256 da tabela não confere com o manifesto")
        instance, info = case_study.build_case(pd.read_csv(path), spec)
        row |= info | {"n_case": info["n"], "missing_rate_case": info["missing_rate"]}
        row["z_determines_mask"] = (
            max(info["mask_auc_from_z"], info["mask_auc_from_z_hgb"]) >= case_study.MASK_DETERMINED_AUC
        )
        if len(instance.y) <= case_study.SUBSAMPLE_N:
            return [row | _case_prediction(instance, model, meta) | {"subsample": "full", "status": "ok"}]
        rows = []
        for r in range(case_study.N_SUBSAMPLES):
            sub = case_study.subsample_instance(instance, case_study.SUBSAMPLE_N, child_seed(seed, spec["case_id"], r))
            rows.append(row | _case_prediction(sub, model, meta) | {"subsample": str(r), "status": "ok"})
        return rows
    except Exception as exc:  # o caso fica registrado com o motivo da falha, nunca some
        return [row | {"subsample": "full", "status": f"failed: {type(exc).__name__}: {exc}"}]


def _case_summary(cases: pd.DataFrame) -> pd.DataFrame:
    """Por caso: média, mínimo e máximo das probabilidades entre as subamostras (sem nenhuma métrica de acerto)."""
    ok = cases[cases.status == "ok"]
    rows = []
    for case_id, g in ok.groupby("case_id", sort=False):
        row = {"case_id": case_id, "n_subsamples": len(g), "n_case": g.n_case.iloc[0]}
        for c in CLASSES:
            row |= {f"p_{c}_mean": g[f"p_{c}"].mean(), f"p_{c}_min": g[f"p_{c}"].min(), f"p_{c}_max": g[f"p_{c}"].max()}
        row["most_probable_counts"] = g.most_probable.value_counts().to_dict()
        rows.append(row)
    return pd.DataFrame(rows)


def run_case_study(
    model_dir: Path, registry: Path, tables_dir: Path, config: RunConfig, n_jobs: int, manifest: Path | None = None
) -> Path:
    """Aplica o modelo congelado, sem ajuste, aos casos do registro: P(MCAR), P(MAR) e P(MNAR) por coluna, ao lado da
    hipótese documentada na fonte. Nenhuma métrica de acerto é calculada (VIT-07). Com `manifest`, cada tabela só é
    usada se o seu SHA-256 conferir com o registrado na obtenção. Casos com mais de 2.000 linhas entram em 10
    subamostras de 2.000 (D3, 29/10), e as faixas de n, k e taxa do treino são sinalizadas (C5)."""
    model, meta = load_final(model_dir)
    specs = case_study.parse_registry(registry)
    expected = {} if manifest is None else dict(pd.read_csv(manifest)[["source", "table_sha256"]].to_numpy())
    run_dir = new_run_dir(config)
    included = [s for s in specs if s["include"]]
    seed = child_seed(config.root_seed, "case_subsample")
    results = Parallel(n_jobs=n_jobs)(delayed(_case_rows)(s, tables_dir, model, meta, expected, seed) for s in included)
    cases = pd.DataFrame([r for rows in results for r in rows]).assign(model_sha256=meta["model_sha256"])
    cases.to_csv(run_dir / "case_predictions.csv", index=False)
    _case_summary(cases).to_csv(run_dir / "case_summary.csv", index=False)
    excluded = pd.DataFrame(
        [{"case_id": s["case_id"], "exclusion_reason": s["exclusion_reason"]} for s in specs if not s["include"]],
        columns=["case_id", "exclusion_reason"],
    )
    excluded.to_csv(run_dir / "excluded.csv", index=False)
    append_run_log(
        run_dir, config, "case-study",
        {"model_dir": str(model_dir), "model_sha256": meta["model_sha256"], "model_source_run": meta["source_run"],
         "registry": str(registry), "registry_sha256": sha256_path(registry), "tables_dir": str(tables_dir),
         "manifest": str(manifest), "manifest_sha256": sha256_path(manifest) if manifest else None,
         "n_cases": len(included), "n_excluded": len(excluded)},
    )  # fmt: skip
    return run_dir


# ---------------------------------------------------------------------------------------------------------------------
# Geradores externos (H5, confirmatória; pré-registro v2)
# ---------------------------------------------------------------------------------------------------------------------
EXTERNAL_METHODS = ("cross_direct_hgb", "cross_logit_simple", "ingen_direct_hgb", "mechdetect_obs")


def run_cross_generator(run_dir: Path, config: RunConfig, n_jobs: int) -> Path:
    """H5: a configuração primária treinada só no gerador principal, nas mesmas partições da avaliação primária, aplicada
    sem ajuste às instâncias dos geradores externos (mdatagen e tab_err) das famílias de teste.

    Métodos, todos nas mesmas instâncias: `cross_direct_hgb` (a H5), `mechdetect_obs` (a secundária EXT_mdo),
    `cross_logit_simple` (a logística simples, também treinada no principal) e `ingen_direct_hgb` (a configuração
    primária treinada nos próprios geradores externos das famílias de treino, referência para a perda fora do gerador,
    descritiva)."""
    out_dir = run_dir / "eval-cross"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    if not (run_dir / "baselines.parquet").exists():  # C5 (b-L1): sem ele, o MechDetect-O viraria "sempre MCAR"
        raise FileNotFoundError(
            f"{run_dir / 'baselines.parquet'} não existe: rode `baselines` antes do `cross-generator`"
        )
    frame = load_eval_frame(run_dir)
    sym = frame[frame.generator == "symmetric"].reset_index(drop=True)
    ext = frame[frame.generator != "symmetric"].reset_index(drop=True)
    if ext.empty:
        raise ValueError("o run não tem instâncias de geradores externos")
    ev = config.evaluate
    seed = child_seed(config.root_seed, "model")
    cols, simple = feature_columns(frame), list(LOGIT_SIMPLE_FEATURES)
    mdo_ok = ext.mdo_status.eq("ok").to_numpy() if "mdo_status" in ext else np.zeros(len(ext), bool)
    mdo_proba = np.full((len(ext), N_CLASSES), np.nan)
    if mdo_ok.any():
        mdo_proba[mdo_ok] = mechdetect_obs.decision_proba(ext.loc[mdo_ok, "mdo_decision"])
    families = sym.family_id.reset_index(drop=True)

    def task(method, repeat, fold, train, test):
        train_fams = set(families.iloc[train])
        test_fams = set(families.iloc[test])
        te = np.flatnonzero(ext.family_id.isin(test_fams).to_numpy())
        if len(te) == 0:
            return pd.DataFrame()
        status = "ok"
        try:
            with threadpool_limits(1):
                if method == "mechdetect_obs":
                    proba = Passthrough().predict_proba(mdo_proba[te])
                elif method == "ingen_direct_hgb":
                    tr = np.flatnonzero(ext.family_id.isin(train_fams).to_numpy())
                    model = DirectClassifier(ev.learner, {}, seed)
                    model.fit(ext[cols].to_numpy(float)[tr], ext.label.to_numpy()[tr], ext.family_id.iloc[tr])
                    proba = model.predict_proba(ext[cols].to_numpy(float)[te])
                else:
                    use = cols if method == "cross_direct_hgb" else simple
                    learner = ev.learner if method == "cross_direct_hgb" else "logreg"
                    model = DirectClassifier(learner, {}, seed)
                    model.fit(sym[use].to_numpy(float)[train], sym.label.to_numpy()[train], families.iloc[train])
                    proba = model.predict_proba(ext[use].to_numpy(float)[te])
        except Exception as exc:  # falha conta como erro, nunca some
            status = f"failed: {type(exc).__name__}: {exc}"
            proba = np.full((len(te), N_CLASSES), 1 / N_CLASSES)
        y_pred = proba.argmax(1) if status == "ok" else np.full(len(te), -1)
        return pd.DataFrame({
            "method": method, "repeat": repeat, "fold": fold, "sample_id": ext.sample_id.iloc[te].to_numpy(),
            "family_id": ext.family_id.iloc[te].to_numpy(), "y_true": ext.label.to_numpy()[te], "y_pred": y_pred,
            **{f"p_{c}": proba[:, k] for k, c in enumerate(CLASSES)}, "status": status,
        })  # fmt: skip

    tasks = [
        (method, repeat, fold, train, test)
        for repeat, fold, train, test in outer_splits(families, config.root_seed, ev.n_splits, ev.n_repeats)
        for method in EXTERNAL_METHODS
    ]
    pred = pd.concat(Parallel(n_jobs=n_jobs)(delayed(task)(*t) for t in tasks), ignore_index=True)
    pred = mark_failures(pred, ext)
    if not check_pairing(pred):
        raise RuntimeError("pareamento 1:1 violado")
    per_family = summarize_by_family(pred)[0]
    _write_eval(out_dir, pred, {"cross_summary": _cross_summary(pred, per_family, ext)})
    append_run_log(
        run_dir, config, "cross-generator",
        {"methods": list(EXTERNAL_METHODS), "n_instances": len(ext), "generators": sorted(ext.generator.unique())},
    )  # fmt: skip
    return out_dir


def _cross_summary(pred: pd.DataFrame, per_family: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Descritivo: macro-recall por método (todos os geradores externos e cada um) e a perda fora do gerador
    (ingen − cross, Hodges–Lehmann das diferenças por família com IC por bootstrap)."""
    rows = []
    for method, g in per_family.groupby("method"):
        low, high = bootstrap_mean_ci(g.macro_recall.to_numpy())
        rows.append({"quantity": "macro_recall", "method": method, "generator": "external", "n_families": len(g),
                     "value": g.macro_recall.mean(), "ci_low": low, "ci_high": high})  # fmt: skip
    by_gen = pred.merge(ext[["sample_id", "generator"]], on="sample_id")
    for generator, part in by_gen.groupby("generator"):
        pf = summarize_by_family(part.drop(columns="generator"))[0]
        for method, g in pf.groupby("method"):
            low, high = bootstrap_mean_ci(g.macro_recall.to_numpy())
            rows.append({"quantity": "macro_recall", "method": method, "generator": generator, "n_families": len(g),
                         "value": g.macro_recall.mean(), "ci_low": low, "ci_high": high})  # fmt: skip
    wide = per_family.pivot(index="family_id", columns="method", values="macro_recall")
    d = (wide["ingen_direct_hgb"] - wide["cross_direct_hgb"]).dropna().to_numpy()
    if len(d):
        low, high = bootstrap_hl_ci(d)
        rows.append({"quantity": "loss_ingen_minus_cross", "method": "cross_direct_hgb", "generator": "external",
                     "n_families": len(d), "value": hodges_lehmann(d), "ci_low": low, "ci_high": high})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Gerador externo estocástico da literatura, só para teste (ampute; pré-registro §6, emenda P24)
# ---------------------------------------------------------------------------------------------------------------------
AMPUTE_DIR = "ampute"
AMPUTE_METHODS = ("cross_direct_hgb", "cross_logit_simple")


def _ampute_context(run_dir: Path, out_dir: Path, context: pd.Series, anchor_id: str) -> list[tuple[dict, dict]]:
    """As três instâncias do ampute num contexto. Z e Y completo vêm do contexto já gerado (na ordem observada). Uma
    instância já gravada por uma execução interrompida é reaproveitada (a máscara é determinística pela semente)."""
    z = read_observed(run_dir, anchor_id).z
    y_complete = read_complete(run_dir, str(context.context_id))
    rows = []
    for mechanism in amp.MECHANISMS:
        instance, oracle, _ = amp.build_ampute_instance(context, z, y_complete, mechanism)
        rel = observed_path(instance.sample_id)
        if not (out_dir / rel).exists():
            write_observed(out_dir, instance)
        rows.append((instance_row(instance, oracle, rel), oracle_row(oracle)))
    return rows


def run_inject_ampute(run_dir: Path, config: RunConfig, n_jobs: int) -> Path:
    """P24: instâncias do ampute (Schouten, Lugtig e Vink, 2018) nos contextos dos geradores externos, só para teste,
    em `<run>/ampute/`, com os descritores. Não calcula predição: pode rodar antes do adendo LLM (P21).

    Retoma de onde parou: instâncias já gravadas são reaproveitadas, e os descritores usam o checkpoint do `features`."""
    out_dir = run_dir / AMPUTE_DIR
    if (out_dir / "features.parquet").exists():
        raise FileExistsError(f"{out_dir / 'features.parquet'} já existe; não sobrescrevo.")
    oracle = pd.read_parquet(run_dir / "oracle.parquet")
    contexts = pd.read_parquet(run_dir / "contexts.parquet")
    external = set(oracle.loc[oracle.generator != "symmetric", "context_id"])
    if not external:
        raise ValueError("o run não tem contextos de geradores externos")
    anchors = oracle[
        (oracle.generator == "symmetric") & (oracle.mechanism == "MCAR") & oracle.context_id.isin(external)
    ]
    anchor_of = dict(zip(anchors.context_id, anchors.sample_id, strict=True))
    chosen = contexts[contexts.context_id.isin(external)].sort_values("context_id")
    if not (out_dir / "instances.parquet").exists():
        out_dir.mkdir(exist_ok=True)
        for name in ("oracle.parquet", "contexts.parquet"):  # sobras de uma interrupção antes do marcador
            (out_dir / name).unlink(missing_ok=True)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_ampute_context)(run_dir, out_dir, row, anchor_of[row.context_id]) for _, row in chosen.iterrows()
        )
        pairs = [pair for rows in results for pair in rows]
        write_parquet(chosen, out_dir / "contexts.parquet")
        write_parquet(pd.DataFrame([o for _, o in pairs]), out_dir / "oracle.parquet")
        write_parquet(pd.DataFrame([i for i, _ in pairs]), out_dir / "instances.parquet")  # último: marcador
    run_features(out_dir, config, n_jobs)
    amp_oracle = pd.read_parquet(out_dir / "oracle.parquet")
    append_run_log(
        run_dir, config, "inject-ampute",
        {"n_contexts": len(chosen), "n_instances": len(amp_oracle), "pyampute": "0.0.3",
         "realized_rate_by_mechanism": amp_oracle.groupby("mechanism").realized_rate.mean().round(4).to_dict()},
    )  # fmt: skip
    return out_dir


def run_cross_ampute(run_dir: Path, config: RunConfig, n_jobs: int) -> Path:
    """P24: a configuração primária e a logística simples, treinadas só no gerador principal das famílias de treino de
    cada fold (as mesmas partições da avaliação primária), aplicadas sem ajuste às instâncias do ampute das famílias de
    teste. Grava a secundária (m = 1), a mudança simétrico → ampute e as taxas realizadas, em `eval-ampute/`."""
    out_dir = run_dir / "eval-ampute"
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} já existe; não sobrescrevo.")
    frame = load_eval_frame(run_dir)
    sym = frame[frame.generator == "symmetric"].reset_index(drop=True)  # mesma ordem e partições do cross-generator
    ext = load_eval_frame(run_dir / AMPUTE_DIR)
    ev = config.evaluate
    seed = child_seed(config.root_seed, "model")
    cols, simple = feature_columns(sym), list(LOGIT_SIMPLE_FEATURES)
    families = sym.family_id.reset_index(drop=True)

    def task(method, repeat, fold, train, test):
        te = np.flatnonzero(ext.family_id.isin(set(families.iloc[test])).to_numpy())
        if len(te) == 0:
            return pd.DataFrame()
        use, learner = (cols, ev.learner) if method == "cross_direct_hgb" else (simple, "logreg")
        status = "ok"
        try:
            with threadpool_limits(1):
                model = DirectClassifier(learner, {}, seed)
                model.fit(sym[use].to_numpy(float)[train], sym.label.to_numpy()[train], families.iloc[train])
                proba = model.predict_proba(ext[use].to_numpy(float)[te])
        except Exception as exc:  # falha conta como erro, nunca some
            status = f"failed: {type(exc).__name__}: {exc}"
            proba = np.full((len(te), N_CLASSES), 1 / N_CLASSES)
        y_pred = proba.argmax(1) if status == "ok" else np.full(len(te), -1)
        return pd.DataFrame({
            "method": method, "repeat": repeat, "fold": fold, "sample_id": ext.sample_id.iloc[te].to_numpy(),
            "family_id": ext.family_id.iloc[te].to_numpy(), "y_true": ext.label.to_numpy()[te], "y_pred": y_pred,
            **{f"p_{c}": proba[:, k] for k, c in enumerate(CLASSES)}, "status": status,
        })  # fmt: skip

    tasks = [
        (method, repeat, fold, train, test)
        for repeat, fold, train, test in outer_splits(families, config.root_seed, ev.n_splits, ev.n_repeats)
        for method in AMPUTE_METHODS
    ]
    pred = pd.concat(Parallel(n_jobs=n_jobs)(delayed(task)(*t) for t in tasks), ignore_index=True)
    pred = mark_failures(pred, ext)
    if not check_pairing(pred):
        raise RuntimeError("pareamento 1:1 violado")
    per_family = summarize_by_family(pred)[0]
    tests = compare_family(per_family, {"P24_ampute": ("cross_direct_hgb", "cross_logit_simple")})
    sym_family = pd.read_parquet(run_dir / "eval-symmetric" / "per_family.parquet")
    amp_score = per_family[per_family.method == "cross_direct_hgb"].set_index("family_id").macro_recall
    sym_score = sym_family[sym_family.method == PRIMARY].set_index("family_id").macro_recall
    d = (amp_score - sym_score).dropna().to_numpy()
    low, high = bootstrap_hl_ci(d) if len(d) else (np.nan, np.nan)
    change = pd.DataFrame([{"quantity": "change_ampute_minus_symmetric", "method": PRIMARY, "n_families": len(d),
                            "hodges_lehmann": hodges_lehmann(d) if len(d) else np.nan, "ci_low": low, "ci_high": high,
                            "mean_ampute": amp_score.mean(), "mean_symmetric": sym_score.mean()}])  # fmt: skip
    amp_oracle = pd.read_parquet(run_dir / AMPUTE_DIR / "oracle.parquet")
    rates = amp_oracle.groupby("mechanism").realized_rate.agg(["mean", "min", "max", "count"]).reset_index()
    _write_eval(out_dir, pred, {"ampute_tests": tests, "ampute_change": change, "ampute_realized_rates": rates})
    append_run_log(run_dir, config, "cross-ampute", {"methods": list(AMPUTE_METHODS), "n_instances": len(ext)})
    return out_dir
