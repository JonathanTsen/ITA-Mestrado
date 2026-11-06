"""Gerador externo estocástico da literatura, só para teste (pré-registro §1 e §6, emenda P24 de 29/10/2026).

É o procedimento de amputação de Schouten, Lugtig e Vink (2018), *Generating missing values for simulation purposes:
a multivariate amputation procedure* (J. Stat. Comput. Simul., DOI 10.1080/00949655.2018.1491577), na implementação
`pyampute` 0.0.3 (`mice::ampute` em R), aplicado só à coluna-alvo Y de um contexto já gerado:
- MCAR: mecanismo `MCAR`;
- MAR: peso só no driver, uma auxiliar observada;
- MNAR: peso só em Y;
- direção lower → `sigmoid-left`, upper → `sigmoid-right`, both → `sigmoid-tail`; `prop` = taxa do contexto.

O ampute sorteia linha a linha, com a probabilidade dada pelo sigmoide do escore padronizado, então a contagem de
ausências varia entre os mecanismos de um contexto. A regra dos trios com a mesma contagem não se aplica a este
gerador, que só entra no teste; as taxas realizadas são reportadas. As instâncias ficam em `<run>/ampute/`,
separadas do benchmark congelado (`inject.py` não muda).

Efeitos colaterais do `pyampute` 0.0.3, contidos aqui: ele usa o RNG global do NumPy (`np.random.seed`) e o logger
raiz (configura `logging.basicConfig` e emite avisos de tabela de busca a cada chamada). O RNG global não é usado pelo
resto do pacote (que usa `np.random.Generator`), e os avisos são silenciados só durante a chamada.
"""

import logging

import numpy as np
import pandas as pd
from pyampute.ampute import MultivariateAmputation

from missdetect.rb.ids import child_seed, stable_id
from missdetect.rb.inject import ObservedInstance, OracleRecord

GENERATOR = "ampute"
MECHANISMS = ("MCAR", "MAR", "MNAR")
AMPUTE_FUNC = {"lower": "sigmoid-left", "upper": "sigmoid-right", "both": "sigmoid-tail"}


def ampute_mask(
    y_complete: np.ndarray, z: pd.DataFrame, driver: str, mechanism: str, direction: str, rate: float, seed: int
) -> np.ndarray:
    """Máscara de ausência de Y (True = ausente) pelo ampute, com a semente dada (reprodutível)."""
    X = np.column_stack([np.asarray(y_complete, dtype=float), z.to_numpy(dtype=float)])
    pattern: dict = {
        "incomplete_vars": [0],
        "mechanism": mechanism,
        "score_to_probability_func": AMPUTE_FUNC[direction],
    }
    if mechanism != "MCAR":
        weights = np.zeros(X.shape[1])
        weights[0 if mechanism == "MNAR" else 1 + list(z.columns).index(driver)] = 1.0
        pattern["weights"] = weights
    logging.disable(logging.WARNING)
    try:
        amputed = MultivariateAmputation(prop=rate, patterns=[pattern], seed=int(seed) % 2**32).fit_transform(X)
    finally:
        logging.disable(logging.NOTSET)
    return np.isnan(amputed[:, 0])


def build_ampute_instance(
    context: pd.Series, z: pd.DataFrame, y_complete: np.ndarray, mechanism: str
) -> tuple[ObservedInstance, OracleRecord, np.ndarray]:
    """Uma instância do ampute no contexto, com Z e Y completo já na ordem observada do contexto."""
    seed = child_seed(int(context.mask_seed), GENERATOR, mechanism)
    mask = ampute_mask(y_complete, z, str(context.driver), mechanism, str(context.direction), float(context.rate), seed)
    y = np.asarray(y_complete, dtype=float).copy()
    y[mask] = np.nan
    spec = {"context_id": str(context.context_id), "generator": GENERATOR, "mechanism": mechanism, "variant": "sigmoid"}
    sample_id = stable_id(spec)
    oracle = OracleRecord(
        sample_id=sample_id,
        context_id=str(context.context_id),
        family_id=int(context.family_id),
        mechanism=mechanism,
        generator=GENERATOR,
        variant=AMPUTE_FUNC[str(context.direction)],
        deterministic=False,
        realized_rate=float(mask.mean()),
        mask_sha256=stable_id({"mask": np.flatnonzero(mask).tolist()}, length=64),
    )
    return ObservedInstance(sample_id=sample_id, y=y, z=z.reset_index(drop=True)), oracle, mask
