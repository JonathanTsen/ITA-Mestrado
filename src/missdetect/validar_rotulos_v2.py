"""validar_rotulos_v2.py — Protocolo v2 de validação de rótulos em camadas.

Camada A (MCAR): voto majoritário entre Little, PKLM (Spohn 2024),
  e Levene-stratified. Pelo menos 2 de 3 testes rejeitam → rejeita MCAR.

Camada B (MAR): AUC de RandomForest prevendo mask a partir de X1..X4
  com permutation p-value (200 permutações), além de mutual information.
  Captura linearidade + interações + não-linearidade.

Camada C (MNAR): 4 scores CAAFE-MNAR (tail asymmetry, kurtosis excess,
  conditional entropy, missing rate por quartil) thresholdados.

Reconciliação: regras simples ou likelihoods Bayesianas via KDE
  ajustado em sintéticos (Camada D, ver calibrar_protocolo.py).

Uso:
    uv run python -m missdetect.validar_rotulos_v2 --data real --experiment v2
    uv run python -m missdetect.validar_rotulos_v2 --data sintetico \\
        --calibration data/calibration.json --max-files-per-class 50
    uv run python -m missdetect.validar_rotulos_v2 --data real \\
        --bayes-scores data/calibration_scores.npz
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines.pklm import pklm_test
from features.caafe_mnar import extract_caafe_mnar_features

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

DATA_PATHS = {
    "sintetico": {m: DATA_DIR / "synthetic" / m for m in ("MCAR", "MAR", "MNAR")},
    "real": {m: DATA_DIR / "real" / "processed" / m for m in ("MCAR", "MAR", "MNAR")},
}

DEFAULT_THRESHOLDS = {
    "auc_mar": 0.65,
    "auc_p": 0.05,
    "mi_max_mar": 0.05,
    "auc_self_delta": 0.05,
    "kl_density": 0.10,
    "kurt_abs": 0.50,
    "cond_entropy": 0.05,
}

VEC_KEYS = (
    "log10_little_p",
    "log10_pklm_p",
    "log10_levene_p",
    "auc_obs",
    "auc_z",
    "mi_max",
    "caafe_auc_self_delta",
    "caafe_kl_density",
    "kurt_excess",
    "cond_entropy",
)


# ============================================================
# CAMADA A — DETECÇÃO DE MCAR (3 testes, voto majoritário)
# ============================================================
def little_mcar_test(df: pd.DataFrame, missing_col: str = "X0") -> float:
    """Little's MCAR via missmecha; fallback Fisher-combined t-tests."""
    try:
        from missmecha import MCARTest

        return float(MCARTest.little_mcar_test(df))
    except Exception:
        return _little_proxy(df, missing_col)


def _little_proxy(df: pd.DataFrame, missing_col: str) -> float:
    mask = df[missing_col].isna()
    if mask.sum() == 0 or (~mask).sum() == 0:
        return float("nan")
    p_vals = []
    for col in [c for c in df.columns if c != missing_col]:
        obs = df.loc[~mask, col].dropna()
        mis = df.loc[mask, col].dropna()
        if len(obs) >= 3 and len(mis) >= 3:
            _, p = stats.ttest_ind(obs, mis, equal_var=False)
            p_vals.append(p)
    if not p_vals:
        return float("nan")
    chi2 = -2 * sum(np.log(max(p, 1e-300)) for p in p_vals)
    return float(1 - stats.chi2.cdf(chi2, df=2 * len(p_vals)))


def levene_stratified(df: pd.DataFrame, missing_col: str = "X0") -> float:
    """Levene's test: variância de Xi difere entre grupos observed/missing.

    Complementa Little (que vê médias) detectando heterocedasticidade
    associada à máscara. Combina Xi via Bonferroni (min(p)*k).
    """
    mask = df[missing_col].isna()
    if mask.sum() < 5 or (~mask).sum() < 5:
        return float("nan")
    p_vals = []
    for col in [c for c in df.columns if c != missing_col]:
        a = df.loc[~mask, col].dropna()
        b = df.loc[mask, col].dropna()
        if len(a) >= 5 and len(b) >= 5:
            _, p = stats.levene(a, b, center="median")
            p_vals.append(p)
    if not p_vals:
        return float("nan")
    return float(min(min(p_vals) * len(p_vals), 1.0))


def layer_a_mcar(df: pd.DataFrame, n_permutations: int = 200, n_workers: int = 1) -> dict:
    p_little = little_mcar_test(df)
    pklm_res = pklm_test(df, missing_col="X0", n_permutations=n_permutations, n_workers=n_workers)
    p_pklm = pklm_res["pklm_pvalue"]
    p_levene = levene_stratified(df)

    rejects = [p < 0.05 for p in (p_little, p_pklm, p_levene) if not np.isnan(p)]
    n_valid = len(rejects)
    n_rej = sum(rejects)
    rejects_mcar = (n_rej >= 2) if n_valid >= 2 else (n_rej >= 1 if n_valid == 1 else False)

    return {
        "little_p": p_little,
        "pklm_p": p_pklm,
        "pklm_stat": pklm_res["pklm_statistic"],
        "levene_p": p_levene,
        "n_tests_reject": n_rej,
        "n_tests_valid": n_valid,
        "rejects_mcar": rejects_mcar,
    }


# ============================================================
# CAMADA B — EVIDÊNCIA DE MAR (AUC + permutation + MI)
# ============================================================
def _cv_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_splits = min(5, max(2, min(n_pos, n_neg)))
    if n_splits < 2 or n_pos == 0 or n_neg == 0:
        return 0.5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=seed, n_jobs=1)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        scores.append(roc_auc_score(y[te], proba))
    return float(np.mean(scores)) if scores else 0.5


def _single_perm_auc(seed: int, X: np.ndarray, mask_template: np.ndarray) -> float:
    """Computa AUC de RF para uma permutação. Usa `seed` para shuffle e RF.

    Função top-level para suportar joblib em qualquer backend.
    """
    rng_local = np.random.RandomState(seed)
    m = mask_template.copy()
    rng_local.shuffle(m)
    return _cv_auc(X, m, seed)


def auc_mask_from_xobs(
    df: pd.DataFrame,
    missing_col: str = "X0",
    n_permutations: int = 200,
    random_state: int = 42,
    n_workers: int = 1,
) -> dict:
    """AUC de RF prevendo mask a partir de X1..X4 + permutation p-value.

    Args:
        n_workers: paralelismo do loop de permutações (default 1 = sequencial).
    """
    mask = df[missing_col].isna().astype(int).values
    if mask.sum() < 5 or (1 - mask).sum() < 5:
        return {"auc_obs": 0.5, "auc_p": 1.0, "auc_z": 0.0}

    other_cols = [c for c in df.columns if c != missing_col]
    X = df[other_cols].values
    X = SimpleImputer(strategy="median").fit_transform(X)

    rng = np.random.RandomState(random_state)
    auc_obs = _cv_auc(X, mask, rng.randint(2**31))

    # Pré-gerar seeds (reprodutibilidade independente da ordem de execução)
    perm_seeds = rng.randint(0, 2**31, size=n_permutations)

    aucs_perm = np.array(
        Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(_single_perm_auc)(int(seed), X, mask) for seed in perm_seeds
        )
    )

    p_value = float(np.mean(aucs_perm >= auc_obs))
    mu, sd = float(aucs_perm.mean()), float(max(aucs_perm.std(), 1e-9))
    z = (auc_obs - mu) / sd

    return {"auc_obs": float(auc_obs), "auc_p": p_value, "auc_z": float(z)}


def mutual_info_mask_xobs(df: pd.DataFrame, missing_col: str = "X0", random_state: int = 42) -> dict:
    mask = df[missing_col].isna().astype(int).values
    if mask.sum() < 5 or (1 - mask).sum() < 5:
        return {"mi_max": 0.0, "mi_mean": 0.0}
    other_cols = [c for c in df.columns if c != missing_col]
    X = df[other_cols].values
    X = SimpleImputer(strategy="median").fit_transform(X)
    mi = mutual_info_classif(X, mask, random_state=random_state)
    return {"mi_max": float(np.max(mi)), "mi_mean": float(np.mean(mi))}


def layer_b_mar(df: pd.DataFrame, n_permutations: int = 200, n_workers: int = 1) -> dict:
    return {
        **auc_mask_from_xobs(df, n_permutations=n_permutations, n_workers=n_workers),
        **mutual_info_mask_xobs(df),
    }


# ============================================================
# CAMADA C — EVIDÊNCIA DE MNAR (CAAFE thresholdadas)
# ============================================================
def layer_c_mnar(df: pd.DataFrame) -> dict:
    feats = extract_caafe_mnar_features(df)
    return {
        "caafe_auc_self_delta": feats["caafe_auc_self_delta"],
        "caafe_kl_density": feats["caafe_kl_density"],
        "caafe_kurt_excess": feats["caafe_kurtosis_excess"],
        "caafe_cond_entropy": feats["caafe_cond_entropy_X0_mask"],
    }


# ============================================================
# RECONCILIAÇÃO — regra simples ou Bayesiana (Camada D)
# ============================================================
def scores_to_vec(scores: dict) -> np.ndarray:
    """Mapeia scores das 3 camadas para um vetor de 10 dimensões.

    Ordem das dimensões definida em VEC_KEYS.
    """
    a, b, c = scores["layer_a"], scores["layer_b"], scores["layer_c"]
    return np.array(
        [
            np.log10(max(a["little_p"], 1e-10)) if not np.isnan(a["little_p"]) else 0.0,
            np.log10(max(a["pklm_p"], 1e-10)) if not np.isnan(a["pklm_p"]) else 0.0,
            np.log10(max(a["levene_p"], 1e-10)) if not np.isnan(a["levene_p"]) else 0.0,
            b["auc_obs"],
            b["auc_z"],
            b["mi_max"],
            c["caafe_auc_self_delta"],
            c["caafe_kl_density"],
            c["caafe_kurt_excess"],
            c["caafe_cond_entropy"],
        ]
    )


def diagnose_rules(scores: dict, thresholds: dict | None = None) -> dict:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    a, b, c = scores["layer_a"], scores["layer_b"], scores["layer_c"]

    if not a["rejects_mcar"]:
        return {
            "prediction": "MCAR",
            "confidence": 0.7,
            "rationale": f"Camada A: {a['n_tests_reject']}/{a['n_tests_valid']} testes rejeitam (insuficiente)",
        }

    mar_evidence = (b["auc_obs"] > th["auc_mar"] and b["auc_p"] < th["auc_p"]) or (b["mi_max"] > th["mi_max_mar"])
    mnar_count = sum(
        [
            c["caafe_auc_self_delta"] > th["auc_self_delta"],
            c["caafe_kl_density"] > th["kl_density"],
            abs(c["caafe_kurt_excess"]) > th["kurt_abs"],
            c["caafe_cond_entropy"] > th["cond_entropy"],
        ]
    )

    if mar_evidence and mnar_count <= 1:
        return {
            "prediction": "MAR",
            "confidence": 0.6,
            "rationale": f"AUC mask~Xobs={b['auc_obs']:.2f} (p={b['auc_p']:.3f}); CAAFE={mnar_count}/4",
        }
    if mnar_count >= 3 and not mar_evidence:
        return {
            "prediction": "MNAR",
            "confidence": 0.6,
            "rationale": f"CAAFE-MNAR {mnar_count}/4 acima do limiar; sem AUC alto",
        }
    if mar_evidence and mnar_count >= 2:
        return {
            "prediction": "MAR",
            "confidence": 0.4,
            "rationale": f"Evidências mistas: AUC={b['auc_obs']:.2f}, CAAFE={mnar_count}/4 (MAR predomina)",
        }
    if mnar_count >= 2:
        return {
            "prediction": "MNAR",
            "confidence": 0.4,
            "rationale": f"CAAFE-MNAR {mnar_count}/4 (incerto, sem AUC alto)",
        }
    return {
        "prediction": "MAR",
        "confidence": 0.3,
        "rationale": f"Rejeita MCAR mas sem evidência clara MNAR; AUC={b['auc_obs']:.2f}, CAAFE={mnar_count}/4",
    }


def fit_kde_from_scores(scores_by_class: dict[str, np.ndarray], bandwidth: float = 0.5) -> dict:
    """Ajusta um KernelDensity gaussiano por mecanismo a partir de matrizes (n, d=10)."""
    return {cls: KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(arr) for cls, arr in scores_by_class.items()}


def diagnose_bayes(scores: dict, kde_by_class: dict, prior: dict | None = None) -> dict:
    """Camada D: posterior via likelihood × prior (KDE multivariado por mecanismo).

    Args:
        prior: {"MCAR": p1, "MAR": p2, "MNAR": p3} com p1+p2+p3=1.
               Default None → uniforme {1/3, 1/3, 1/3}.
    """
    _prior = prior if prior is not None else {"MCAR": 1 / 3, "MAR": 1 / 3, "MNAR": 1 / 3}
    vec = scores_to_vec(scores).reshape(1, -1)
    log_lik = {m: float(kde_by_class[m].score_samples(vec)[0]) for m in ("MCAR", "MAR", "MNAR")}
    log_max = max(log_lik.values())
    p = {m: np.exp(log_lik[m] - log_max) * _prior[m] for m in log_lik}
    s = sum(p.values())
    p = {m: float(v / s) for m, v in p.items()}
    pred = max(p, key=p.get)
    sorted_p = sorted(p.values(), reverse=True)
    confidence = float(sorted_p[0] - sorted_p[1])
    rationale = f"Bayes: P(MCAR)={p['MCAR']:.2f}, P(MAR)={p['MAR']:.2f}, P(MNAR)={p['MNAR']:.2f}"
    if confidence < 0.4:
        rationale = "AMBIGUO " + rationale
    return {
        "prediction": pred,
        "confidence": confidence,
        "p_mcar": p["MCAR"],
        "p_mar": p["MAR"],
        "p_mnar": p["MNAR"],
        "rationale": rationale,
    }


# ============================================================
# Função pública: validar 1 dataset
# ============================================================
def validate_one(
    df: pd.DataFrame,
    n_permutations: int = 200,
    thresholds: dict | None = None,
    bayes_kde: dict | None = None,
    prior: dict | None = None,
    n_workers: int = 1,
    parallel_layers: bool = False,
) -> dict:
    """Valida um dataset rodando as 3 camadas e diagnóstico final.

    Args:
        n_workers: paralelismo do loop de permutações dentro das camadas A/B
            (default 1 = sequencial).
        parallel_layers: se True, executa as camadas A e B em duas threads
            paralelas (Nível 2). Útil quando chamado dentro de um worker
            de processo do Nível 1.
        prior: prior Bayesiano {"MCAR": p1, "MAR": p2, "MNAR": p3}.
               None → uniforme. Ignorado se bayes_kde é None.
    """
    if parallel_layers:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(layer_a_mcar, df, n_permutations, n_workers)
            fut_b = pool.submit(layer_b_mar, df, n_permutations, n_workers)
            c = layer_c_mnar(df)
            a = fut_a.result()
            b = fut_b.result()
    else:
        a = layer_a_mcar(df, n_permutations=n_permutations, n_workers=n_workers)
        b = layer_b_mar(df, n_permutations=n_permutations, n_workers=n_workers)
        c = layer_c_mnar(df)
    scores = {"layer_a": a, "layer_b": b, "layer_c": c}
    if bayes_kde is not None:
        diag = diagnose_bayes(scores, bayes_kde, prior=prior)
    else:
        diag = diagnose_rules(scores, thresholds=thresholds)
    return {**scores, "diagnosis": diag}


# ============================================================
# CLI
# ============================================================
def _flatten_row(file_name: str, true_label: str, missing_rate: float, res: dict) -> dict:
    a, b, c = res["layer_a"], res["layer_b"], res["layer_c"]
    d = res["diagnosis"]
    return {
        "file": file_name,
        "true_label": true_label,
        "missing_rate": round(missing_rate, 4),
        "little_p": a["little_p"],
        "pklm_p": a["pklm_p"],
        "pklm_stat": a["pklm_stat"],
        "levene_p": a["levene_p"],
        "rejects_mcar": a["rejects_mcar"],
        "auc_obs": b["auc_obs"],
        "auc_p": b["auc_p"],
        "auc_z": b["auc_z"],
        "mi_max": b["mi_max"],
        "mi_mean": b["mi_mean"],
        "caafe_auc_self_delta": c["caafe_auc_self_delta"],
        "caafe_kl_density": c["caafe_kl_density"],
        "caafe_kurt_excess": c["caafe_kurt_excess"],
        "caafe_cond_entropy": c["caafe_cond_entropy"],
        "prediction": d["prediction"],
        "confidence": d["confidence"],
        "p_mcar": d.get("p_mcar"),
        "p_mar": d.get("p_mar"),
        "p_mnar": d.get("p_mnar"),
        "rationale": d["rationale"],
    }


def _load_calibration_thresholds(path: str) -> tuple[dict | None, float | None]:
    """Retorna (thresholds, bandwidth) do calibration.json. bandwidth=None se não salvo."""
    with open(path) as f:
        data = json.load(f)
    return data.get("thresholds"), data.get("bandwidth")


def _load_bayes_scores(path: str, bandwidth: float = 0.5) -> dict:
    """Carrega scores sintéticos (.npz) e refita KDE por mecanismo.

    Espera arrays 'MCAR', 'MAR', 'MNAR' de shape (n, 10).
    """
    with np.load(path) as data:
        arrays = {cls: data[cls] for cls in ("MCAR", "MAR", "MNAR") if cls in data.files}
    if set(arrays.keys()) != {"MCAR", "MAR", "MNAR"}:
        raise ValueError(f"Arrays MCAR/MAR/MNAR esperados em {path}; encontrei {list(arrays.keys())}")
    return fit_kde_from_scores(arrays, bandwidth=bandwidth)


def _process_one_validation_file(task: tuple) -> tuple[dict | None, str, str | None]:
    """Worker top-level para validação paralela de datasets reais/sintéticos."""
    fpath, mech, n_permutations, thresholds, bayes_kde, prior, n_workers_perm, parallel_layers = task
    try:
        df = pd.read_csv(fpath, sep="\t")
    except Exception as e:
        return None, fpath.name, str(e)
    res = validate_one(
        df,
        n_permutations=n_permutations,
        thresholds=thresholds,
        bayes_kde=bayes_kde,
        prior=prior,
        n_workers=n_workers_perm,
        parallel_layers=parallel_layers,
    )
    missing_rate = float(df["X0"].isna().mean())
    flat = _flatten_row(fpath.name, mech, missing_rate, res)
    return flat, fpath.name, None


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["sintetico", "real"], required=True)
    parser.add_argument("--experiment", default="v2_protocol")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--calibration", default=None, help="Path para data/calibration.json")
    parser.add_argument(
        "--bayes-scores",
        default=None,
        help="Path para data/calibration_scores.npz (refita KDE em runtime)",
    )
    parser.add_argument("--bandwidth", type=float, default=0.5, help="Largura de banda do KDE Bayesiano")
    parser.add_argument("--max-files-per-class", type=int, default=None, help="Limita arquivos por classe (debug)")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help=(
            "Número de processos para paralelizar datasets (Nível 1). "
            "Default 1 = sequencial. Sugerido para Apple Silicon 12c: 4."
        ),
    )
    parser.add_argument(
        "--prior-mnar",
        type=float,
        default=None,
        help=(
            "Prior P(MNAR) para diagnóstico Bayesiano (0-1). "
            "P(MCAR)=P(MAR)=(1-prior_mnar)/2. Default None → uniforme."
        ),
    )
    args = parser.parse_args()

    paths = DATA_PATHS[args.data]
    out_dir = RESULTS_DIR / args.experiment / args.data / "validacao_rotulos_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = None
    bayes_kde = None
    prior: dict | None = None
    bandwidth = args.bandwidth
    if args.calibration:
        thresholds, bw_from_json = _load_calibration_thresholds(args.calibration)
        if bw_from_json is not None:
            bandwidth = bw_from_json
            print(f"[ok] bandwidth lido do JSON: {bandwidth:.4f}")
        print(f"[ok] thresholds calibrados de {args.calibration}")
    if args.bayes_scores:
        bayes_kde = _load_bayes_scores(args.bayes_scores, bandwidth=bandwidth)
        print(f"[ok] KDE Bayesiano refittado de {args.bayes_scores} (bw={bandwidth})")
    if args.prior_mnar is not None:
        p_mnar = float(args.prior_mnar)
        p_rest = (1.0 - p_mnar) / 2.0
        prior = {"MCAR": p_rest, "MAR": p_rest, "MNAR": p_mnar}
        print(f"[ok] prior informativo: MCAR={p_rest:.2f}, MAR={p_rest:.2f}, MNAR={p_mnar:.2f}")

    mode = "Bayesiano" if bayes_kde else ("Regras (calibradas)" if thresholds else "Regras (default)")
    print("=" * 70)
    print("PROTOCOLO V2 DE VALIDAÇÃO DE RÓTULOS")
    print("=" * 70)
    print(f"Dados: {args.data}")
    print(f"Permutações: {args.n_permutations}")
    print(f"Modo: {mode}")
    print(f"Output: {out_dir}")
    print("=" * 70)

    cores = max(1, os.cpu_count() or 1)
    # P0 fix: sempre paralelliza permutações, independente de n_workers
    n_workers_perm = max(1, cores // max(1, args.n_workers))
    # ThreadPoolExecutor é seguro com threads, ativar sempre
    parallel_layers = True

    all_tasks: list[tuple] = []
    for mech, p in paths.items():
        if not p.exists():
            print(f"[!] {mech} dir não existe: {p}")
            continue
        files = sorted([f for f in os.listdir(p) if f.endswith(".txt")])
        if args.max_files_per_class:
            files = files[: args.max_files_per_class]
        print(f"\n[{mech}] {len(files)} arquivos em {p}")
        for fname in files:
            all_tasks.append(
                (
                    p / fname,
                    mech,
                    args.n_permutations,
                    thresholds,
                    bayes_kde,
                    prior,
                    n_workers_perm,
                    parallel_layers,
                )
            )

    desc = "files [seq]" if args.n_workers == 1 else f"files [{args.n_workers}p×{n_workers_perm}c]"
    rows: list[dict] = []
    if args.n_workers == 1 or not all_tasks:
        results_iter = (_process_one_validation_file(t) for t in all_tasks)
        pool = None
    else:
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx)
        results_iter = pool.map(_process_one_validation_file, all_tasks, chunksize=1)

    try:
        for flat, fname, err in tqdm(results_iter, total=len(all_tasks), desc=desc):
            if err is not None:
                print(f"  [erro] {fname}: {err}")
                continue
            if flat is not None:
                rows.append(flat)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "validacao_v2.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nResultados em {csv_path}")

    if df_out.empty:
        return

    df_out["consistent"] = df_out["true_label"] == df_out["prediction"]
    n_consistent = int(df_out["consistent"].sum())
    n_total = len(df_out)
    print(f"\nAccuracy (predição vs rótulo literário): {n_consistent / n_total:.1%} ({n_consistent}/{n_total})")
    print("\nConfusion matrix (linha=verdadeiro, coluna=predito):")
    cm = pd.crosstab(df_out["true_label"], df_out["prediction"], margins=True)
    print(cm.to_string())

    summary = {
        "mode": mode,
        "n_total": int(n_total),
        "n_consistent": int(n_consistent),
        "accuracy_vs_literature": float(n_consistent / n_total),
        "n_permutations": args.n_permutations,
        "per_class": {
            cls: {
                "n": int((df_out["true_label"] == cls).sum()),
                "consistent": int(((df_out["true_label"] == cls) & df_out["consistent"]).sum()),
            }
            for cls in ("MCAR", "MAR", "MNAR")
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Resumo em {out_dir / 'summary.json'}")


if __name__ == "__main__":
    _main()
