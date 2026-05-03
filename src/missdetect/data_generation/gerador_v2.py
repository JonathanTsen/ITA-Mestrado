"""
gerador_v2.py — Geração de dados sintéticos com múltiplas variantes por mecanismo.

Variantes implementadas:
  MCAR (3):  uniform masking, fixed selection, block-based
  MAR  (5):  logistic(X1), threshold(X1), rank(X1), quantile-group(X1), multi-predictor(X1+X2)
  MNAR (4):  self-censoring logistic, quantile threshold, tail censoring, self-masking+noise

Para cada variante: 100 datasets com distribuições base variadas.
Total: (3+5+4) variantes × 100 datasets = 1200 datasets.

Formato: X0 (com missing), X1-X4 (completas), 1000 linhas, tab-separated.

Uso:
    cd "IC - ITA 2/Scripts"
    uv run python gerador_v2.py [--n-per-variant 100]
"""

import os
import shutil
import argparse
import numpy as np
import pandas as pd

# ======================================================
# CONFIG
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BASE_DIR), "Dataset", "synthetic_data")

N_ROWS = 1000
N_COLS = 5
COLNAMES = [f"X{i}" for i in range(N_COLS)]

# Distribuições base para diversidade
DISTRIBUTIONS = {
    "uniform": lambda rng, n: rng.uniform(0, 1, n),
    "normal": lambda rng, n: np.clip(rng.normal(0.5, 0.15, n), 0, 1),
    "exponential": lambda rng, n: np.clip(rng.exponential(0.3, n), 0, 1),
    "beta": lambda rng, n: rng.beta(2, 5, n),
}
DIST_NAMES = list(DISTRIBUTIONS.keys())


# ======================================================
# VARIANTES DE MECANISMO
# ======================================================

# --- MCAR ---

def mcar_uniform(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 1: cada célula de X0 tem prob independente de ser missing."""
    out = X.copy()
    mask = rng.random(len(out)) < rate
    if mask.sum() == 0:
        mask[rng.integers(0, len(out))] = True
    out.loc[mask, "X0"] = np.nan
    return out


def mcar_fixed(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 2: exatamente N posições aleatórias fixas."""
    out = X.copy()
    n_miss = max(1, int(rate * len(out)))
    idx = rng.choice(len(out), size=n_miss, replace=False)
    out.iloc[idx, 0] = np.nan
    return out


def mcar_block(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 3: blocos contíguos de missing (simula falha de sensor)."""
    out = X.copy()
    n = len(out)
    n_miss = max(1, int(rate * n))
    n_blocks = rng.integers(1, max(2, n_miss // 5) + 1)
    block_size = max(1, n_miss // n_blocks)
    indices = set()
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, n - block_size))
        for j in range(block_size):
            if start + j < n:
                indices.add(start + j)
    indices = list(indices)[:n_miss]
    if not indices:
        indices = [rng.integers(0, n)]
    out.iloc[indices, 0] = np.nan
    return out


MCAR_VARIANTS = {
    "uniform": mcar_uniform,
    "fixed": mcar_fixed,
    "block": mcar_block,
}


# --- MAR ---

def mar_logistic(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 1: P(X0 missing) = sigmoid(β * standardize(X1))."""
    out = X.copy()
    x1 = out["X1"].to_numpy()
    z = (x1 - x1.mean()) / (x1.std() + 1e-9)
    prob = 1.0 / (1.0 + np.exp(-z))
    prob = prob * (rate / (prob.mean() + 1e-12))
    prob = np.clip(prob, 0.0, 1.0)
    mask = rng.random(len(out)) < prob
    if mask.sum() == 0:
        mask[np.argmax(prob)] = True
    out.loc[mask, "X0"] = np.nan
    return out


def mar_threshold(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 2: X0 missing quando X1 > percentil (1-rate)."""
    out = X.copy()
    threshold = np.quantile(out["X1"], 1 - rate)
    mask = out["X1"] >= threshold
    if mask.sum() == 0:
        mask.iloc[out["X1"].argmax()] = True
    out.loc[mask, "X0"] = np.nan
    return out


def mar_rank(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 3: X0 missing nos ranks mais altos de X1."""
    out = X.copy()
    n_miss = max(1, int(rate * len(out)))
    ranks = out["X1"].rank(method="first", ascending=False)
    mask = ranks <= n_miss
    out.loc[mask, "X0"] = np.nan
    return out


def mar_quantile_group(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 4: missing distribuído por quartis de X1 com probabilidades diferentes."""
    out = X.copy()
    x1 = out["X1"]
    q25, q50, q75 = x1.quantile([0.25, 0.5, 0.75])
    # Prob de missing aumenta com X1
    probs = np.where(x1 <= q25, rate * 0.2,
            np.where(x1 <= q50, rate * 0.6,
            np.where(x1 <= q75, rate * 1.4,
                     rate * 2.0)))
    probs = np.clip(probs, 0, 1)
    mask = rng.random(len(out)) < probs
    if mask.sum() == 0:
        mask.iloc[x1.argmax()] = True
    out.loc[mask, "X0"] = np.nan
    return out


def mar_multi_predictor(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 5: P(X0 missing) depende de X1 e X2 conjuntamente."""
    out = X.copy()
    x1 = (out["X1"].to_numpy() - out["X1"].mean()) / (out["X1"].std() + 1e-9)
    x2 = (out["X2"].to_numpy() - out["X2"].mean()) / (out["X2"].std() + 1e-9)
    z = 0.7 * x1 + 0.3 * x2
    prob = 1.0 / (1.0 + np.exp(-z))
    prob = prob * (rate / (prob.mean() + 1e-12))
    prob = np.clip(prob, 0.0, 1.0)
    mask = rng.random(len(out)) < prob
    if mask.sum() == 0:
        mask[np.argmax(prob)] = True
    out.loc[mask, "X0"] = np.nan
    return out


MAR_VARIANTS = {
    "logistic": mar_logistic,
    "threshold": mar_threshold,
    "rank": mar_rank,
    "quantile_group": mar_quantile_group,
    "multi_predictor": mar_multi_predictor,
}


# --- MNAR ---

def mnar_self_logistic(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 1: P(X0 missing) = sigmoid(β * standardize(X0))."""
    out = X.copy()
    x0 = out["X0"].to_numpy()
    z = (x0 - x0.mean()) / (x0.std() + 1e-9)
    prob = 1.0 / (1.0 + np.exp(-z))
    prob = prob * (rate / (prob.mean() + 1e-12))
    prob = np.clip(prob, 0.0, 1.0)
    mask = rng.random(len(out)) < prob
    if mask.sum() == 0:
        mask[np.argmax(prob)] = True
    out.loc[mask, "X0"] = np.nan
    return out


def mnar_quantile_threshold(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 2: X0 missing quando X0 > quantil alto."""
    out = X.copy()
    threshold = np.quantile(out["X0"], 1 - rate)
    mask = out["X0"] >= threshold
    if mask.sum() == 0:
        mask.iloc[out["X0"].argmax()] = True
    out.loc[mask, "X0"] = np.nan
    return out


def mnar_tail_censoring(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 3: X0 missing nos extremos (caudas inferior e superior)."""
    out = X.copy()
    n_miss = max(1, int(rate * len(out)))
    n_low = n_miss // 2
    n_high = n_miss - n_low
    x0 = out["X0"]
    low_idx = x0.nsmallest(n_low).index
    high_idx = x0.nlargest(n_high).index
    out.loc[low_idx.union(high_idx), "X0"] = np.nan
    return out


def mnar_self_noisy(X: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Tipo 4: self-masking com ruído — P(missing) depende de X0 + noise."""
    out = X.copy()
    x0 = out["X0"].to_numpy()
    noise = rng.normal(0, 0.1, len(out))
    z = (x0 + noise - (x0.mean())) / (x0.std() + 1e-9)
    prob = 1.0 / (1.0 + np.exp(-z))
    prob = prob * (rate / (prob.mean() + 1e-12))
    prob = np.clip(prob, 0.0, 1.0)
    mask = rng.random(len(out)) < prob
    if mask.sum() == 0:
        mask[np.argmax(prob)] = True
    out.loc[mask, "X0"] = np.nan
    return out


MNAR_VARIANTS = {
    "self_logistic": mnar_self_logistic,
    "quantile_threshold": mnar_quantile_threshold,
    "tail_censoring": mnar_tail_censoring,
    "self_noisy": mnar_self_noisy,
}


ALL_VARIANTS = {
    "MCAR": MCAR_VARIANTS,
    "MAR": MAR_VARIANTS,
    "MNAR": MNAR_VARIANTS,
}


# ======================================================
# GERAÇÃO
# ======================================================

def generate_base_data(rng: np.random.Generator, dist_name: str) -> pd.DataFrame:
    """Gera matriz base com distribuição especificada."""
    dist_fn = DISTRIBUTIONS[dist_name]
    data = np.column_stack([dist_fn(rng, N_ROWS) for _ in range(N_COLS)])
    return pd.DataFrame(data, columns=COLNAMES)


def main():
    parser = argparse.ArgumentParser(description="Gerador v2 com múltiplas variantes MissMecha")
    parser.add_argument("--n-per-variant", type=int, default=100,
                        help="Datasets por variante (default: 100)")
    parser.add_argument("--clean", action="store_true", default=True,
                        help="Limpar pastas de saída antes de gerar")
    args = parser.parse_args()

    N_PER_VARIANT = args.n_per_variant

    paths = {
        "MCAR": os.path.join(OUT, "MCAR"),
        "MAR": os.path.join(OUT, "MAR"),
        "MNAR": os.path.join(OUT, "MNAR"),
    }

    if args.clean:
        for p in paths.values():
            if os.path.isdir(p):
                shutil.rmtree(p)

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    total = 0
    for mech, variants in ALL_VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"  {mech} — {len(variants)} variantes × {N_PER_VARIANT} datasets")
        print(f"{'='*50}")

        for var_name, var_fn in variants.items():
            count = 0
            for k in range(N_PER_VARIANT):
                seed = hash((mech, var_name, k)) % (2**31)
                rng = np.random.default_rng(seed)

                dist_name = DIST_NAMES[k % len(DIST_NAMES)]
                X = generate_base_data(rng, dist_name)

                missing_rate = int(rng.integers(1, 11))
                rate = missing_rate / 100.0

                try:
                    X_miss = var_fn(X, rate, rng)
                except Exception as e:
                    print(f"    ERRO {mech}/{var_name}/k={k}: {e}")
                    continue

                # Sanity check: missing só em X0
                for c in COLNAMES[1:]:
                    if X_miss[c].isna().any():
                        X_miss[c] = X_miss[c].fillna(X_miss[c].mean())

                if X_miss["X0"].isna().sum() == 0:
                    continue

                fname = f"{mech}_{var_name}_seed{seed}_mr{missing_rate}.txt"
                X_miss.to_csv(os.path.join(paths[mech], fname), sep="\t", index=False)
                count += 1
                total += 1

            print(f"  {var_name}: {count} datasets gerados")

    print(f"\n{'='*60}")
    print(f"Total: {total} datasets sintéticos gerados")
    for mech in ["MCAR", "MAR", "MNAR"]:
        n = len([f for f in os.listdir(paths[mech]) if f.endswith(".txt")])
        print(f"  {mech}: {n} arquivos")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
