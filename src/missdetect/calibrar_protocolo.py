"""calibrar_protocolo.py — Camada E: calibração de thresholds e scores Bayesianos.

Roda as Camadas A-C de validar_rotulos_v2.py em uma amostra balanceada dos
1.200 sintéticos (com ground truth) e produz dois artefatos:

  1. data/calibration.json — thresholds via Youden's J por score, mais
     métricas de validação do protocolo nos sintéticos.

  2. data/calibration_scores.npz — vetores de 10 scores por mecanismo, em
     formato numpy nativo (seguro). validar_rotulos_v2.py refita um KDE
     Gaussiano on-the-fly a partir destes pontos quando rodado com
     --bayes-scores.

Uso:
    uv run python -m missdetect.calibrar_protocolo \\
        --output-dir data --n-per-class 80 --n-permutations 50

    # Com mais permutações (mais lento, mais preciso):
    uv run python -m missdetect.calibrar_protocolo \\
        --output-dir data --n-per-class 100 --n-permutations 200
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validar_rotulos_v2 import (
    DATA_PATHS,
    DEFAULT_THRESHOLDS,
    VEC_KEYS,
    diagnose_bayes,
    diagnose_rules,
    fit_kde_from_scores,
    layer_a_mcar,
    layer_b_mar,
    layer_c_mnar,
    scores_to_vec,
)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _youden_threshold(y_true: np.ndarray, score: np.ndarray, higher_means_positive: bool = True) -> dict:
    """Threshold que maximiza Youden's J = TPR - FPR sobre uma ROC.

    higher_means_positive=False inverte (útil para p-values: p baixo → positivo).
    """
    valid = ~np.isnan(score)
    if valid.sum() < 10 or len(np.unique(y_true[valid])) < 2:
        return {"threshold": float("nan"), "youden_j": 0.0, "auc": 0.5}
    s = score[valid]
    y = y_true[valid]
    if not higher_means_positive:
        s = -s
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    idx = int(np.argmax(j))
    raw_threshold = float(thr[idx])
    threshold = -raw_threshold if not higher_means_positive else raw_threshold
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 vs <2
    auc = float(trapz(tpr, fpr))
    return {"threshold": threshold, "youden_j": float(j[idx]), "auc": auc}


def _process_one_file(task: tuple) -> dict | None:
    """Processa um único arquivo: Camadas A-C + serialização para row-dict.

    Função top-level para uso com ``ProcessPoolExecutor.map``. Camadas A e B
    rodam em threads paralelas (Nível 2). Permutações dentro de cada camada
    paralelas se ``n_workers_perm > 1`` (Nível 3).

    Retorna ``None`` em caso de falha de leitura do arquivo (loop continua).
    """
    fpath, mech, n_permutations, n_workers_perm = task
    try:
        df = pd.read_csv(fpath, sep="\t")
    except Exception:
        return None

    with ThreadPoolExecutor(max_workers=2) as tpool:
        fut_a = tpool.submit(layer_a_mcar, df, n_permutations, n_workers_perm)
        fut_b = tpool.submit(layer_b_mar, df, n_permutations, n_workers_perm)
        c = layer_c_mnar(df)
        a = fut_a.result()
        b = fut_b.result()

    scores = {"layer_a": a, "layer_b": b, "layer_c": c}
    vec = scores_to_vec(scores)
    row = {"file": fpath.name, "true_label": mech, **a, **b, **c}
    row.update(dict(zip(VEC_KEYS, vec, strict=False)))
    return {"row": row, "mech": mech, "vec": vec.tolist()}


def _append_to_checkpoint(row: dict, checkpoint_path: Path) -> None:
    """Append-por-linha ao CSV de checkpoint (chamado apenas no processo principal)."""
    write_header = not checkpoint_path.exists()
    pd.DataFrame([row]).to_csv(checkpoint_path, mode="a", header=write_header, index=False)


def _collect_scores(
    n_per_class: int,
    n_permutations: int,
    seed: int = 42,
    checkpoint_path: Path | None = None,
    n_workers: int = 1,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Roda Camadas A-C em n_per_class arquivos por mecanismo dos sintéticos.

    Se ``checkpoint_path`` for fornecido, cada dataset processado é appendado
    a um CSV nesse caminho. Em uma re-execução com o mesmo path, datasets já
    no CSV são pulados (retomada após crash/sleep).

    Args:
        n_workers: número de processos para paralelizar datasets (Nível 1).
            Default 1 = sequencial. Quando >1, cada worker recebe ``cpu//n_workers``
            cores para paralelizar permutações (Nível 3) e usa 2 threads para
            executar Camadas A e B em paralelo (Nível 2).

    Atenção: mudar ``n_per_class``, ``n_permutations`` ou ``seed`` invalida
    um checkpoint existente — apague o CSV antes de relançar.
    """
    rng = random.Random(seed)
    paths = DATA_PATHS["sintetico"]

    cores = max(1, os.cpu_count() or 1)
    # P0 fix: sempre paralelliza permutações, independente de n_workers
    n_workers_perm = max(1, cores // max(1, n_workers))

    rows: list[dict] = []
    vecs_by_class: dict[str, list[np.ndarray]] = {"MCAR": [], "MAR": [], "MNAR": []}

    processed: set[tuple[str, str]] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        rows = existing.to_dict(orient="records")
        for r in rows:
            vec = np.array([r[k] for k in VEC_KEYS], dtype=float)
            vecs_by_class[r["true_label"]].append(vec)
            processed.add((str(r["true_label"]), str(r["file"])))
        print(f"[checkpoint] retomando: {len(processed)} datasets já em {checkpoint_path}")

    all_tasks: list[tuple] = []
    for mech, p in paths.items():
        if not p.exists():
            print(f"[!] {mech} dir não existe: {p}")
            continue
        files = sorted([f for f in os.listdir(p) if f.endswith(".txt")])
        rng.shuffle(files)
        files = files[:n_per_class]
        remaining = [f for f in files if (mech, f) not in processed]
        skipped = len(files) - len(remaining)
        if skipped:
            print(f"[{mech}] {skipped} já no checkpoint; processando {len(remaining)}")
        else:
            print(f"[{mech}] amostrando {len(remaining)} arquivos")
        for fname in remaining:
            all_tasks.append((p / fname, mech, n_permutations, n_workers_perm))

    if not all_tasks:
        df_scores = pd.DataFrame(rows)
        arrays = {cls: np.array(vs) for cls, vs in vecs_by_class.items() if vs}
        return df_scores, arrays

    desc = "datasets [seq]" if n_workers == 1 else f"datasets [{n_workers}p×{n_workers_perm}c]"

    if n_workers == 1:
        results_iter = (_process_one_file(t) for t in all_tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
        results_iter = pool.map(_process_one_file, all_tasks, chunksize=1)

    try:
        for result in tqdm(results_iter, total=len(all_tasks), desc=desc):
            if result is None:
                continue
            row = result["row"]
            mech = result["mech"]
            vec = np.array(result["vec"], dtype=float)
            vecs_by_class[mech].append(vec)
            rows.append(row)
            if checkpoint_path is not None:
                _append_to_checkpoint(row, checkpoint_path)
    finally:
        if n_workers > 1:
            pool.shutdown(wait=True)

    df_scores = pd.DataFrame(rows)
    arrays = {cls: np.array(vs) for cls, vs in vecs_by_class.items() if vs}
    return df_scores, arrays


def _calibrate_thresholds(df: pd.DataFrame) -> dict:
    """Calibra thresholds via Youden's J sobre os scores sintéticos."""
    out: dict = {}

    is_non_mcar = (df["true_label"] != "MCAR").astype(int).values
    is_mnar = (df["true_label"] == "MNAR").astype(int).values

    out["little_p"] = _youden_threshold(is_non_mcar, df["little_p"].values, higher_means_positive=False)
    out["pklm_p"] = _youden_threshold(is_non_mcar, df["pklm_p"].values, higher_means_positive=False)
    out["levene_p"] = _youden_threshold(is_non_mcar, df["levene_p"].values, higher_means_positive=False)

    is_mar = (df["true_label"] == "MAR").astype(int).values
    out["auc_obs"] = _youden_threshold(is_mar, df["auc_obs"].values, higher_means_positive=True)
    out["mi_max"] = _youden_threshold(is_mar, df["mi_max"].values, higher_means_positive=True)

    out["caafe_auc_self_delta"] = _youden_threshold(
        is_mnar, df["caafe_auc_self_delta"].values, higher_means_positive=True
    )
    out["caafe_kl_density"] = _youden_threshold(is_mnar, df["caafe_kl_density"].values, higher_means_positive=True)
    out["caafe_kurt_excess_abs"] = _youden_threshold(
        is_mnar, np.abs(df["caafe_kurt_excess"].values), higher_means_positive=True
    )
    out["caafe_cond_entropy"] = _youden_threshold(is_mnar, df["caafe_cond_entropy"].values, higher_means_positive=True)

    return out


def _build_threshold_dict(calibration: dict) -> dict:
    """Mapeia métricas calibradas para os nomes esperados em diagnose_rules."""
    th = dict(DEFAULT_THRESHOLDS)
    if not np.isnan(calibration["auc_obs"]["threshold"]):
        th["auc_mar"] = calibration["auc_obs"]["threshold"]
    if not np.isnan(calibration["mi_max"]["threshold"]):
        th["mi_max_mar"] = calibration["mi_max"]["threshold"]
    if not np.isnan(calibration["caafe_auc_self_delta"]["threshold"]):
        th["auc_self_delta"] = calibration["caafe_auc_self_delta"]["threshold"]
    if not np.isnan(calibration["caafe_kl_density"]["threshold"]):
        th["kl_density"] = calibration["caafe_kl_density"]["threshold"]
    if not np.isnan(calibration["caafe_kurt_excess_abs"]["threshold"]):
        th["kurt_abs"] = calibration["caafe_kurt_excess_abs"]["threshold"]
    if not np.isnan(calibration["caafe_cond_entropy"]["threshold"]):
        th["cond_entropy"] = calibration["caafe_cond_entropy"]["threshold"]
    return th


def _scores_from_row(row: pd.Series) -> dict:
    """Reconstrói o dict de scores usado por diagnose_rules/diagnose_bayes."""
    return {
        "layer_a": {
            "little_p": row["little_p"],
            "pklm_p": row["pklm_p"],
            "pklm_stat": row["pklm_stat"],
            "levene_p": row["levene_p"],
            "n_tests_reject": row["n_tests_reject"],
            "n_tests_valid": row["n_tests_valid"],
            "rejects_mcar": row["rejects_mcar"],
        },
        "layer_b": {
            "auc_obs": row["auc_obs"],
            "auc_p": row["auc_p"],
            "auc_z": row["auc_z"],
            "mi_max": row["mi_max"],
            "mi_mean": row["mi_mean"],
        },
        "layer_c": {
            "caafe_auc_self_delta": row["caafe_auc_self_delta"],
            "caafe_kl_density": row["caafe_kl_density"],
            "caafe_kurt_excess": row["caafe_kurt_excess"],
            "caafe_cond_entropy": row["caafe_cond_entropy"],
        },
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas comuns para predições MCAR/MAR/MNAR."""
    accuracy = float(np.mean(y_pred == y_true))
    cm = confusion_matrix(y_true, y_pred, labels=["MCAR", "MAR", "MNAR"])
    per_class = {}
    for i, cls in enumerate(["MCAR", "MAR", "MNAR"]):
        n = int((y_true == cls).sum())
        tp = int(cm[i, i])
        per_class[cls] = {
            "n": n,
            "recall": float(tp / n) if n else 0.0,
            "predictions": int((y_pred == cls).sum()),
        }
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def _eval_protocol(df_scores: pd.DataFrame, arrays: dict, thresholds: dict, bandwidth: float = 0.5) -> dict:
    """Mede accuracy do protocolo nos sintéticos sob 3 modos: default/calibrated/bayes."""
    kde = fit_kde_from_scores(arrays, bandwidth=bandwidth)
    results = {"default": [], "calibrated": [], "bayes": []}

    for _, row in df_scores.iterrows():
        scores = _scores_from_row(row)
        results["default"].append(diagnose_rules(scores)["prediction"])
        results["calibrated"].append(diagnose_rules(scores, thresholds=thresholds)["prediction"])
        results["bayes"].append(diagnose_bayes(scores, kde)["prediction"])

    metrics = {}
    y_true = df_scores["true_label"].values
    for mode, preds in results.items():
        metrics[mode] = _classification_metrics(y_true, np.array(preds))
    return metrics


def _eval_bayes_cv(
    df_scores: pd.DataFrame,
    bandwidth: float = 0.5,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Avalia Bayes/KDE com Stratified K-Fold para evitar treino=teste.

    Reusa os vetores 10D já coletados em ``df_scores``: em cada fold o KDE é
    fitado apenas nos folds de treino e prediz o fold deixado fora.
    """
    y_true = df_scores["true_label"].astype(str).values
    X_vecs = df_scores[list(VEC_KEYS)].astype(float).values

    min_class = min(int(np.sum(y_true == cls)) for cls in ("MCAR", "MAR", "MNAR"))
    if min_class < 2:
        return {
            "accuracy": float("nan"),
            "accuracy_std": float("nan"),
            "fold_accuracies": [],
            "confusion_matrix": [],
            "per_class": {},
            "n_splits": 0,
            "error": "classes insuficientes para cross-validation estratificada",
        }

    n_splits = min(n_splits, min_class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_preds = np.empty(len(y_true), dtype=object)
    fold_accuracies: list[float] = []

    for train_idx, test_idx in skf.split(X_vecs, y_true):
        arrays_fold = {
            cls: X_vecs[train_idx][y_true[train_idx] == cls]
            for cls in ("MCAR", "MAR", "MNAR")
        }
        kde_fold = fit_kde_from_scores(arrays_fold, bandwidth=bandwidth)

        fold_preds = []
        for i in test_idx:
            pred = diagnose_bayes(_scores_from_row(df_scores.iloc[i]), kde_fold)["prediction"]
            cv_preds[i] = pred
            fold_preds.append(pred)
        fold_accuracies.append(float(np.mean(np.array(fold_preds) == y_true[test_idx])))

    metrics = _classification_metrics(y_true, cv_preds)
    metrics["accuracy_std"] = float(np.std(fold_accuracies))
    metrics["fold_accuracies"] = fold_accuracies
    metrics["n_splits"] = int(n_splits)
    return metrics


def _select_bandwidth(arrays: dict[str, np.ndarray]) -> float:
    """Seleciona bandwidth ótimo para KDE via GridSearchCV (5-fold) nos scores sintéticos."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    X_all = np.vstack(list(arrays.values()))
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=5, n_jobs=-1)
    grid.fit(X_all)
    bw = float(grid.best_params_["bandwidth"])
    print(f"[P7] bandwidth ótimo: {bw:.4f} (score={grid.best_score_:.4f})")
    return bw


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data"),
        help="Diretório onde salvar calibration.json e calibration_scores.npz",
    )
    parser.add_argument("--n-per-class", type=int, default=80)
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--auto-bandwidth", action="store_true", help="Seleciona bandwidth ótimo via GridSearchCV.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CSV onde appendar cada dataset processado; permite retomar após crash/sleep.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help=(
            "Número de processos para paralelizar datasets (Nível 1). "
            "Default 1 = sequencial. Sugerido para Apple Silicon 12c: 4."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CALIBRAÇÃO DO PROTOCOLO V2 (Camada E)")
    print("=" * 70)
    print(f"Sintéticos: {args.n_per_class} por classe (×3 = {args.n_per_class * 3})")
    print(f"Permutações por dataset: {args.n_permutations}")
    print(f"Workers (Nível 1 - datasets): {args.n_workers}")
    if args.n_workers > 1:
        cores = max(1, os.cpu_count() or 1)
        print(f"  → {cores // args.n_workers} cores/worker p/ permutações (Nível 3)")
    print(f"Output: {out_dir}")
    print("=" * 70)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[checkpoint] CSV: {checkpoint_path}")

    df_scores, arrays = _collect_scores(
        n_per_class=args.n_per_class,
        n_permutations=args.n_permutations,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        n_workers=args.n_workers,
    )

    if df_scores.empty:
        print("[erro] nenhum score coletado; abortando")
        return

    print(f"\n[ok] {len(df_scores)} datasets sintéticos processados")

    bandwidth = args.bandwidth
    if args.auto_bandwidth:
        print("\n[P7] selecionando bandwidth ótimo via GridSearchCV...")
        bandwidth = _select_bandwidth(arrays)
    else:
        print(f"\n[bandwidth] usando valor fixo: {bandwidth}")

    print("\n[calibração] computando thresholds via Youden's J...")
    calibration = _calibrate_thresholds(df_scores)
    thresholds = _build_threshold_dict(calibration)

    print("\n[avaliação] medindo accuracy do protocolo nos sintéticos (3 modos)...")
    metrics = _eval_protocol(df_scores, arrays, thresholds, bandwidth=bandwidth)
    print("[avaliação] medindo Bayes com 5-fold cross-validation...")
    metrics["bayes_cv"] = _eval_bayes_cv(df_scores, bandwidth=bandwidth, n_splits=5, seed=args.seed)

    print("\n" + "=" * 70)
    print("RESULTADOS DA CALIBRAÇÃO")
    print("=" * 70)
    for mode, m in metrics.items():
        suffix = ""
        if mode == "bayes_cv":
            suffix = f" ± {m['accuracy_std']:.1%} ({m['n_splits']}-fold)"
        print(f"\n[{mode}] accuracy = {m['accuracy']:.1%}{suffix}")
        for cls, d in m["per_class"].items():
            print(f"  {cls}: recall={d['recall']:.1%} (n={d['n']}, pred={d['predictions']})")

    cal_path = out_dir / "calibration.json"
    npz_path = out_dir / "calibration_scores.npz"

    output_json = {
        "n_per_class": args.n_per_class,
        "n_permutations": args.n_permutations,
        "bandwidth": bandwidth,
        "raw_calibration": calibration,
        "thresholds": thresholds,
        "validation_metrics": metrics,
    }
    with open(cal_path, "w") as f:
        json.dump(output_json, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)

    np.savez(npz_path, **arrays)

    print(f"\n[ok] calibration salvo em {cal_path}")
    print(f"[ok] scores Bayesianos salvos em {npz_path}")
    print("\nPara usar:")
    print(f"  uv run python -m missdetect.validar_rotulos_v2 --data real --calibration {cal_path}")
    print(f"  uv run python -m missdetect.validar_rotulos_v2 --data real --bayes-scores {npz_path}")


if __name__ == "__main__":
    _main()
