"""Análises do LLM que retreinam a configuração primária (30/10/2026): a sensibilidade de 200 linhas da H4 e a
secundária RQ4-A2.

**Sensibilidade de 200 linhas** (pré-registro §4, H4: "informação desigual, declarada").

O LLM vê no máximo 200 linhas sorteadas; a configuração primária vê a instância inteira. Na sensibilidade pré-registrada,
a configuração primária é **retreinada** com subamostras de ≤ 200 linhas das instâncias de treino, nas mesmas partições,
e aplicada à mesma subamostra que o LLM recebe (aplicar o modelo treinado com n ≥ 500 a 200 linhas seria extrapolação
contra a primária).

- **Subamostra:** em toda instância do gerador principal, as linhas de `render.llm_rows` com a semente do contexto
  (`child_seed(raiz, "llm_render", contexto)`): nas instâncias do painel, são exatamente as linhas que o A1 mostra.
- **Descritores:** os 28 descritores 34b recalculados nas subamostras, com checkpoint por parte.
- **Modelo:** a configuração primária (`DirectClassifier` com o aprendiz e a semente do `evaluate`), na CV externa com
  as mesmas sementes, logo nas mesmas partições por família.
- **Saídas** em `<run>/llm-sensitivity/`: as predições, a tabela por família restrita ao painel (`direct_hgb_200`) e, se
  as predições do LLM forem dadas, a comparação descritiva LLM-A1 × primária com 200 linhas (HL, IC e p sem Holm).
"""

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from missdetect.rb import features as feat
from missdetect.rb.config import RunConfig
from missdetect.rb.evaluate import run_method, summarize_by_family
from missdetect.rb.ids import child_seed
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.llm.render import llm_rows
from missdetect.rb.models import CLASSES, DirectClassifier
from missdetect.rb.runs import append_run_log, read_observed, write_parquet
from missdetect.rb.stats import llm_sensitivity_row

METHOD_SUFFIX = "_200"
CHUNK = 64


def subsample(instance: ObservedInstance, seed: int) -> ObservedInstance:
    rows = llm_rows(len(instance.y), seed)
    return ObservedInstance(instance.sample_id, instance.y[rows], instance.z.iloc[rows].reset_index(drop=True))


def _feature_row(run_dir: Path, sample_id: str, seed: int) -> dict:
    t0 = time.perf_counter()
    try:
        with threadpool_limits(1):
            row = {**feat.compute(subsample(read_observed(run_dir, sample_id), seed)), "feature_status": "ok"}
    except Exception as exc:  # falha vira status e conta como erro, nunca é descartada
        row = {"sample_id": sample_id, "feature_status": f"failed: {exc!r}"}
    return {**row, "feature_seconds": time.perf_counter() - t0}


def features_200(run_dir: Path, instances: pd.DataFrame, root_seed: int, work: Path, n_jobs: int) -> pd.DataFrame:
    parts = work / "features_parts"
    parts.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    for part in sorted(parts.glob("part-*.parquet")):
        done |= set(pd.read_parquet(part, columns=["sample_id"]).sample_id)
    todo = instances[~instances.sample_id.isin(done)]
    start = len(list(parts.glob("part-*.parquet")))
    for j, i in enumerate(range(0, len(todo), CHUNK * n_jobs)):
        chunk = todo.iloc[i : i + CHUNK * n_jobs]
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_feature_row)(run_dir, r.sample_id, child_seed(root_seed, "llm_render", r.context_id))
            for r in chunk.itertuples()
        )
        write_parquet(pd.DataFrame(rows), parts / f"part-{start + j:05d}.parquet")
    frame = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part-*.parquet"))], ignore_index=True)
    if frame.sample_id.duplicated().any():
        raise RuntimeError("sample_id duplicado nos checkpoints da sensibilidade")
    return frame


def run_llm_sensitivity(
    run_dir: Path, config: RunConfig, n_jobs: int, panel_ids: list[str], llm_predictions: Path | None = None
) -> Path:  # fmt: skip
    run_dir = Path(run_dir)
    out_dir = run_dir / "llm-sensitivity"
    if out_dir.exists():
        return out_dir
    work = run_dir / "llm-sensitivity-work"
    instances = pd.read_parquet(run_dir / "instances.parquet").merge(
        pd.read_parquet(run_dir / "oracle.parquet")[["sample_id", "mechanism", "generator"]], on="sample_id"
    )
    instances = instances[instances.generator == "symmetric"].sort_values("sample_id").reset_index(drop=True)
    descriptors = features_200(run_dir, instances, config.root_seed, work, n_jobs)
    names = [f["name"] for f in json.loads((run_dir / "feature_schema.json").read_text())["features"]]
    frame = instances.merge(descriptors, on="sample_id", how="left").sort_values("sample_id").reset_index(drop=True)
    ev = config.evaluate
    method = f"direct_{ev.learner}{METHOD_SUFFIX}"
    seed = child_seed(config.root_seed, "model")
    y = frame.mechanism.map({m: i for i, m in enumerate(CLASSES)}).to_numpy()
    with threadpool_limits(1):
        pred, _ = run_method(
            method, lambda: DirectClassifier(ev.learner, {}, seed), frame[names].to_numpy(float), y, frame.family_id,
            frame.sample_id, config.root_seed, ev.n_splits, ev.n_repeats,
        )  # fmt: skip
    failed = set(frame.loc[~frame.feature_status.eq("ok"), "sample_id"])
    rows = pred.sample_id.isin(failed)
    pred.loc[rows, [f"p_{c}" for c in CLASSES]] = 1 / 3
    pred.loc[rows, ["y_pred", "status"]] = [-1, "failed: features"]
    panel_pred = pred[pred.sample_id.isin(panel_ids)]
    per_family, _ = summarize_by_family(panel_pred)
    comparison = None
    if llm_predictions is not None:
        llm = pd.read_parquet(llm_predictions)
        llm = llm[llm.sample_id.isin(panel_ids)]
        columns = ["method", "repeat", "family_id", "sample_id", "y_true", "y_pred", *[f"p_{c}" for c in CLASSES]]
        both, _ = summarize_by_family(pd.concat([llm[columns], panel_pred[columns]], ignore_index=True))
        comparison = pd.DataFrame([llm_sensitivity_row(both, str(llm.method.iloc[0]), method)])
    for stale in run_dir.glob(".tmp-llm-sensitivity-*"):
        shutil.rmtree(stale)
    tmp = run_dir / f".tmp-llm-sensitivity-{os.getpid()}"
    tmp.mkdir()
    descriptors.to_parquet(tmp / "features_200.parquet", index=False)
    pred.to_parquet(tmp / "predictions.parquet", index=False)
    per_family.to_parquet(tmp / "per_family.parquet", index=False)
    if comparison is not None:
        comparison.to_csv(tmp / "h4_sensitivity.csv", index=False)
    tmp.rename(out_dir)
    extra = {"method": method, "n_instances": len(frame), "n_panel": len(panel_ids), "n_feature_failures": len(failed),
             "llm_predictions": str(llm_predictions) if llm_predictions else None}  # fmt: skip
    append_run_log(run_dir, config, "llm-sensitivity", extra)
    return out_dir


def run_a2_ablation(run_dir: Path, config: RunConfig, a2_predictions: Path) -> Path:
    """Secundária RQ4-A2 (m = 1; pré-registro §6, emenda P23): pipeline + descritores do LLM × pipeline, no painel.

    Os dois braços são avaliados por CV de famílias dentro do painel (mesmas partições, mesma semente): a configuração
    primária com os 28 descritores × a mesma configuração com os 28 descritores mais as 3 probabilidades do LLM-A2
    (uma falha do A2 entra como 1/3, 1/3, 1/3). Teste: Wilcoxon-Pratt do macro-recall por família, sem Holm (m = 1)."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "llm-a2-ablation"
    if out_dir.exists():
        return out_dir
    a2 = pd.read_parquet(a2_predictions)
    if "replicate" in a2:
        a2 = a2[a2.replicate == 0]
    names = [f["name"] for f in json.loads((run_dir / "feature_schema.json").read_text())["features"]]
    oracle = pd.read_parquet(run_dir / "oracle.parquet")[["sample_id", "mechanism"]]
    frame = (
        a2[["sample_id", "family_id", "p_MCAR", "p_MAR", "p_MNAR"]]
        .rename(columns={f"p_{c}": f"llm_a2_{c}" for c in CLASSES})
        .merge(
            pd.read_parquet(run_dir / "features.parquet", columns=["sample_id", "feature_status", *names]),
            on="sample_id",
        )
        .merge(oracle, on="sample_id")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    ev = config.evaluate
    seed = child_seed(config.root_seed, "model")
    y = frame.mechanism.map({m: i for i, m in enumerate(CLASSES)}).to_numpy()
    arms = {
        f"direct_{ev.learner}_panel": names,
        f"direct_{ev.learner}_panel_plus_llm_a2": [*names, *(f"llm_a2_{c}" for c in CLASSES)],
    }
    preds = []
    with threadpool_limits(1):
        for method, columns in arms.items():
            pred, _ = run_method(
                method, lambda: DirectClassifier(ev.learner, {}, seed), frame[columns].to_numpy(float), y,
                frame.family_id, frame.sample_id, config.root_seed, ev.n_splits, ev.n_repeats,
            )  # fmt: skip
            preds.append(pred)
    pred = pd.concat(preds, ignore_index=True)
    failed = set(frame.loc[~frame.feature_status.eq("ok"), "sample_id"])
    rows = pred.sample_id.isin(failed)
    pred.loc[rows, [f"p_{c}" for c in CLASSES]] = 1 / 3
    pred.loc[rows, ["y_pred", "status"]] = [-1, "failed: features"]
    per_family, _ = summarize_by_family(pred)
    plus, base = arms
    comparison = pd.DataFrame([llm_sensitivity_row(per_family, plus, base) | {"hypothesis": "RQ4-A2"}])
    for stale in run_dir.glob(".tmp-llm-a2-ablation-*"):
        shutil.rmtree(stale)
    tmp = run_dir / f".tmp-llm-a2-ablation-{os.getpid()}"
    tmp.mkdir()
    pred.to_parquet(tmp / "predictions.parquet", index=False)
    per_family.to_parquet(tmp / "per_family.parquet", index=False)
    comparison.to_csv(tmp / "rq4_a2.csv", index=False)
    tmp.rename(out_dir)
    append_run_log(run_dir, config, "llm-a2-ablation", {"a2_predictions": str(a2_predictions), "n_instances": len(frame),
                                                         "n_families": int(frame.family_id.nunique())})  # fmt: skip
    return out_dir
