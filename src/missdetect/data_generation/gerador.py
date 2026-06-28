"""
gerador.py — Geração de dados sintéticos (MCAR/MAR/MNAR) usando a biblioteca mdatagen.

Correção (jun/2026): a versão anterior instanciava uMCAR/uMAR/uMNAR mas NUNCA
chamava o método gerador (.random()/.rank()/.run()); lia ``gen.dataset`` (uma
cópia do DataFrame SEM NaN). O sanity-check ``if rate_x0 == 0.0`` disparava
sempre e o resultado vinha do fallback manual, de forma silenciosa — ver
``docs/archive/00_auditoria_inicial/01_gerador.md`` (BUG-G1/G2). Agora o método
correto é chamado, a coluna ``target`` que a mdatagen anexa é removida, e o uso
de fallback é contabilizado e logado (resolve BUG-G1/G2). Execução protegida por
``if __name__ == "__main__"`` (BUG-G3) e limpeza de pastas só com ``--clean``
(BUG-G4).

Observação importante: o benchmark sintético principal da dissertação (12
variantes × 100) é gerado por ``gerador_v2.py`` (NumPy/pandas). Este script
gera a variante de 3 mecanismos via mdatagen e serve como prova de conceito /
validação cruzada do gerador — NÃO substitui ``gerador_v2.py``.
"""

import argparse
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd

try:
    from mdatagen.univariate.uMAR import uMAR
    from mdatagen.univariate.uMCAR import uMCAR
    from mdatagen.univariate.uMNAR import uMNAR

    MDATAGEN_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depende do ambiente
    MDATAGEN_AVAILABLE = False
    _IMPORT_ERROR = exc

logger = logging.getLogger("gerador")

N_ROWS = 1000
N_COLS = 5
COLNAMES = [f"X{i}" for i in range(N_COLS)]


def _generate_with_mdatagen(X: pd.DataFrame, y: np.ndarray, mech: str, missing_rate: int) -> pd.DataFrame:
    """
    Gera o DataFrame com NaN usando a mdatagen, chamando o método gerador correto.

    A mdatagen exige que, após instanciar a classe, se CHAME um método que insere
    os NaN e RETORNA o DataFrame. Os atributos pós-construção (``gen.dataset``) são
    apenas cópias sem missing. Cada método retorna o DataFrame com uma coluna extra
    ``target`` (= y), que é removida aqui.

    MCAR -> uMCAR(...).random()         (faltas completamente aleatórias)
    MAR  -> uMAR(..., x_obs="X1").rank()  (faltas dependem de X1 observada)
    MNAR -> uMNAR(..., threshold=0).run() (faltas dependem do próprio X0)
    """
    if mech == "MCAR":
        gen = uMCAR(X=X, y=y, missing_rate=missing_rate, x_miss="X0")
        out = gen.random()
    elif mech == "MAR":
        gen = uMAR(X=X, y=y, missing_rate=missing_rate, x_miss="X0", x_obs="X1")
        out = gen.rank()
    elif mech == "MNAR":
        gen = uMNAR(X=X, y=y, missing_rate=missing_rate, x_miss="X0", threshold=0)
        out = gen.run()
    else:
        raise ValueError("Mecanismo inválido. Use MCAR, MAR, MNAR.")

    return out.drop(columns=["target"], errors="ignore")


def _apply_fallback_manual(
    X: pd.DataFrame, mech: str, missing_rate_pct: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Fallback manual (NumPy) usado apenas se a mdatagen não estiver disponível ou
    falhar. Garante missing APENAS em X0. missing_rate_pct: 1..10 (percentual).
    """
    X_out = X.copy()
    n = len(X_out)
    p = missing_rate_pct / 100.0

    if mech == "MCAR":
        u = rng.random(n)
        idx = np.where(u < p)[0]
        if len(idx) == 0:
            idx = np.array([rng.integers(0, n)])
        X_out.loc[idx, "X0"] = np.nan

    elif mech == "MAR":
        x1 = X_out["X1"].to_numpy()
        z = (x1 - x1.mean()) / (x1.std() + 1e-9)
        prob = 1.0 / (1.0 + np.exp(-z))
        prob = prob * (p / (prob.mean() + 1e-12))
        prob = np.clip(prob, 0.0, 1.0)
        u = rng.random(n)
        idx = np.where(u < prob)[0]
        if len(idx) == 0:
            idx = np.array([np.argmax(prob)])
        X_out.loc[idx, "X0"] = np.nan

    elif mech == "MNAR":
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


def generate(n_datasets: int, out_dir: str, clean: bool = False) -> dict:
    """
    Gera ``n_datasets`` por mecanismo (MCAR/MAR/MNAR) preferindo a mdatagen, com
    fallback manual contabilizado. Salva um manifesto JSON indicando, por dataset,
    qual fonte (mdatagen/fallback) gerou os NaN. Retorna os contadores.
    """
    paths = {m: os.path.join(out_dir, m) for m in ("MCAR", "MAR", "MNAR")}
    if clean:
        for p in paths.values():
            if os.path.isdir(p):
                shutil.rmtree(p)
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    if not MDATAGEN_AVAILABLE:
        logger.warning("mdatagen indisponível (%s); usando apenas o fallback manual.", _IMPORT_ERROR)

    counters = {"mdatagen": 0, "fallback": 0}
    manifest = []

    for mech in ("MCAR", "MAR", "MNAR"):
        for k in range(n_datasets):
            seed = 10_000 + k
            rng = np.random.default_rng(seed)

            X = pd.DataFrame(rng.uniform(0, 1, size=(N_ROWS, N_COLS)), columns=COLNAMES)
            missing_rate = int(rng.integers(1, 11))  # 1..10 inclusive
            y = np.zeros(N_ROWS)

            X_miss = None
            source = None
            if MDATAGEN_AVAILABLE:
                try:
                    X_miss = _generate_with_mdatagen(X, y, mech, missing_rate)
                    if X_miss["X0"].isna().mean() == 0.0:
                        raise RuntimeError("mdatagen retornou X0 sem nenhum missing")
                    source = "mdatagen"
                except Exception as exc:
                    logger.warning(
                        "mdatagen falhou (%s, seed=%d, mr=%d): %s -> fallback manual",
                        mech, seed, missing_rate, exc,
                    )
                    X_miss = None

            if X_miss is None:
                X_miss = _apply_fallback_manual(X, mech, missing_rate, rng)
                source = "fallback"

            counters[source] += 1

            # garante que só X0 tem missing
            for c in COLNAMES[1:]:
                if X_miss[c].isna().any():
                    X_miss[c] = X_miss[c].fillna(X_miss[c].mean())

            fname = f"{mech}_seed{seed}_mr{missing_rate}.txt"
            X_miss.to_csv(os.path.join(paths[mech], fname), sep="\t", index=False)
            manifest.append(
                {"file": fname, "mechanism": mech, "seed": seed,
                 "missing_rate": missing_rate, "source": source}
            )

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"counters": counters, "datasets": manifest}, fh, indent=2)

    logger.info("Geração concluída: mdatagen=%d, fallback=%d (manifest em %s/manifest.json)",
                counters["mdatagen"], counters["fallback"], out_dir)
    return counters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera datasets sintéticos (MCAR/MAR/MNAR) via mdatagen, com fallback manual contabilizado."
    )
    parser.add_argument("--n-datasets", type=int, default=1000,
                        help="datasets por mecanismo (default: 1000)")
    parser.add_argument("--sample", type=int, default=None,
                        help="atalho de prova de conceito: gera N datasets por mecanismo")
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_mdatagen")
    parser.add_argument("--out", default=default_out, help="diretório de saída")
    parser.add_argument("--clean", action="store_true",
                        help="limpa as pastas de saída antes de gerar (DESTRUTIVO)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n = args.sample if args.sample is not None else args.n_datasets
    counters = generate(n, args.out, clean=args.clean)
    print(f"mdatagen usado: {counters['mdatagen']}, fallback: {counters['fallback']}")
