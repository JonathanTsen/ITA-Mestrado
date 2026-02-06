import os
import shutil
import numpy as np
import pandas as pd

from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.univariate.uMAR import uMAR
from mdatagen.univariate.uMNAR import uMNAR


def _get_generated_dataset(gen):
    """
    Na sua versão do mdatagen, o dataset pode vir pronto em 'dataset' ou 'X'.
    """
    if hasattr(gen, "dataset") and gen.dataset is not None:
        return gen.dataset.copy()
    if hasattr(gen, "X") and gen.X is not None:
        return gen.X.copy()
    raise AttributeError(
        f"Não encontrei dataset gerado em {type(gen).__name__}. "
        f"Atributos disponíveis: {[m for m in dir(gen) if not m.startswith('_')]}"
    )


def _apply_fallback_manual(X: pd.DataFrame, mech: str, missing_rate_pct: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Garante missing APENAS em X0 usando um gerador manual simples e controlado.
    missing_rate_pct: 1..10 (percentual)
    """
    X_out = X.copy()
    n = len(X_out)
    p = missing_rate_pct / 100.0

    if mech == "MCAR":
        # missing totalmente aleatório
        u = rng.random(n)
        idx = np.where(u < p)[0]
        if len(idx) == 0:
            idx = np.array([rng.integers(0, n)])
        X_out.loc[idx, "X0"] = np.nan

    elif mech == "MAR":
        # missing depende de X1 (observável)
        x1 = X_out["X1"].to_numpy()
        z = (x1 - x1.mean()) / (x1.std() + 1e-9)
        prob = 1.0 / (1.0 + np.exp(-z))  # sigmoid(z) em [0,1]
        # escala para a taxa desejada
        prob = prob * (p / (prob.mean() + 1e-12))
        prob = np.clip(prob, 0.0, 1.0)

        u = rng.random(n)
        idx = np.where(u < prob)[0]
        if len(idx) == 0:
            idx = np.array([np.argmax(prob)])
        X_out.loc[idx, "X0"] = np.nan

    elif mech == "MNAR":
        # missing depende do próprio X0 (não observável)
        x0 = X_out["X0"].to_numpy()
        z = (x0 - x0.mean()) / (x0.std() + 1e-9)
        prob = 1.0 / (1.0 + np.exp(-z))
        prob = prob * (p / (prob.mean() + 1e-12))
        prob = np.clip(prob, 0.0, 1.0)

        u = rng.random(n)
        idx = np.where(u < prob)[0]
        if len(idx) == 0:
            idx = np.array([np.argmax(prob)])
        X_out.loc[idx, "X0"] = np.nan

    else:
        raise ValueError("Mecanismo inválido. Use MCAR, MAR, MNAR.")

    return X_out


# ======================================================
# CONFIG
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BASE_DIR), "Dataset")

paths = {
    "MCAR": os.path.join(OUT, "MCAR"),
    "MAR": os.path.join(OUT, "MAR"),
    "MNAR": os.path.join(OUT, "MNAR"),
}

# ✅ RECOMENDADO: limpar pastas para não acumular (isso resolve os 92 arquivos)
CLEAN_OUTPUT = True
if CLEAN_OUTPUT:
    for p in paths.values():
        if os.path.isdir(p):
            shutil.rmtree(p)

for p in paths.values():
    os.makedirs(p, exist_ok=True)

N = 1000
P = 5
N_DATASETS = 1000
colnames = [f"X{i}" for i in range(P)]

# ======================================================
# GERAÇÃO
# ======================================================
for mech in ["MCAR", "MAR", "MNAR"]:
    for k in range(N_DATASETS):
        seed = 10_000 + k
        rng = np.random.default_rng(seed)

        # matriz completa uniform[0,1]
        X = pd.DataFrame(rng.uniform(0, 1, size=(N, P)), columns=colnames)

        # ✅ taxa 1% a 10% INCLUSIVO (antes você estava gerando só 1..9)
        missing_rate = int(rng.integers(1, 11))  # 1..10

        y_dummy = np.zeros(N)

        # tenta via mdatagen
        try:
            if mech == "MCAR":
                gen = uMCAR(X=X, y=y_dummy, missing_rate=missing_rate, x_miss="X0")
            elif mech == "MAR":
                gen = uMAR(X=X, y=y_dummy, missing_rate=missing_rate, x_miss="X0", x_obs="X1")
            else:
                gen = uMNAR(X=X, y=y_dummy, missing_rate=missing_rate, x_miss="X0")

            X_miss = _get_generated_dataset(gen)

        except Exception:
            # se der qualquer erro, cai pro manual
            X_miss = X.copy()

        # ✅ Sanity check: garante missing em X0
        rate_x0 = X_miss["X0"].isna().mean()
        if rate_x0 == 0.0:
            # fallback manual controlado
            X_miss = _apply_fallback_manual(X, mech, missing_rate, rng)

        # ✅ Garante que só X0 tem missing
        for c in colnames[1:]:
            if X_miss[c].isna().any():
                X_miss[c] = X_miss[c].fillna(X_miss[c].mean())

        # salva
        fname = f"{mech}_seed{seed}_mr{missing_rate}.txt"
        X_miss.to_csv(os.path.join(paths[mech], fname), sep="\t", index=False)

print(f"✅ Banco sintético gerado ({N_DATASETS} por classe) e X0 garantidamente com missing.")