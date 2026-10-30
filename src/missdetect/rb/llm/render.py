"""Renderização cega das instâncias para o LLM (passo 42 do plano; pré-registro §4, emenda P20; 30/10/2026).

Braços:
- **A1, classificador direto cego:** a instância vira uma tabela CSV com colunas `c0…ck`, em inteiros de 0 a 999
  (transformação afim positiva por coluna sobre os valores observados das linhas mostradas, seguida de arredondamento),
  com no máximo 200 linhas sorteadas e linhas e colunas permutadas. A ausência é o token `NA`.
- **A2, extrator de descritores:** só os 28 descritores 34b, em JSON numérico, com os nomes pelo que medem.

O renderizador recebe só o que o método vê (Y com NaN e Z) e uma semente derivada do contexto, que é a mesma para os três
mecanismos. Nunca recebe o rótulo, o `sample_id`, o oráculo, caminhos nem nomes de colunas. O teste byte a byte
(`tests/rb/test_llm_render.py`) confere que mudar o rótulo, os nomes, os valores ausentes de Y ou aplicar uma
transformação afim positiva aos dados não muda o prompt. As instruções e o esquema são congelados por hash no adendo
`prereg-v1-llm`; sem exemplos, sem prior de classes e sem limiares (passo 42, §2).
"""

import json

import numpy as np
import pandas as pd

from missdetect.rb.inject import ObservedInstance

RENDERER_VERSION = "1"
MAX_ROWS = 200
SCALE_MAX = 999
NA_TOKEN = "NA"
CLASSES = ("MCAR", "MAR", "MNAR")

_DEFINITIONS = """\
- MCAR: whether a value is missing does not depend on any values in the data.
- MAR: whether a value is missing depends only on the values of the other, complete columns.
- MNAR: whether a value is missing depends on the missing value itself, even after accounting for the other columns.

Give the probability of each mechanism. The three probabilities must sum to 1."""

A1_SYSTEM = f"""\
You will receive a table in CSV format. It is a random sample of at most {MAX_ROWS} rows from a larger dataset. The \
column names (c0, c1, ...) are anonymous, and the rows and columns are in random order. Each column was rescaled by an \
increasing linear transformation and rounded to integers from 0 to {SCALE_MAX}. Exactly one column has missing values, \
written as {NA_TOKEN}; the other columns are complete.

Classify the missingness mechanism of the column with missing values:
{_DEFINITIONS}"""

A2_SYSTEM = f"""\
You will receive summary statistics, in JSON, computed from a table in which exactly one numeric column (Y) has missing \
values and the other columns are complete. Each statistic is named after what it measures, with a prefix for its group:
- g0: the missing rate of Y;
- g1: the shape of the distribution of the observed values of Y;
- g2: how well the missingness indicator of Y is predicted by the complete columns (AUC, Kolmogorov-Smirnov distance \
and mutual information, summarized over the columns by the maximum, the mean and the second largest value);
- g3: how Y relates to the complete columns: the fit of a prediction of Y from them on the rows where Y is observed, \
how those predictions differ between rows with Y observed and missing, and the shape of the residuals;
- g4: whether the missingness indicator is better predicted from the complete columns and the predictions of Y than a \
randomly permuted indicator.
A statistic is null when it is undefined.

Classify the missingness mechanism of Y:
{_DEFINITIONS}"""

A1_SCHEMA = {
    "type": "object",
    "properties": {c: {"type": "number", "minimum": 0, "maximum": 1} for c in CLASSES},
    "required": list(CLASSES),
    "additionalProperties": False,
}
A2_SCHEMA = A1_SCHEMA


def scale_to_int(values: np.ndarray) -> np.ndarray:
    """Transformação afim positiva que leva os valores observados a inteiros de 0 a 999 (NaN continua NaN)."""
    v = np.asarray(values, dtype=float)
    observed = v[~np.isnan(v)]
    if observed.size == 0:
        return v.copy()
    low, high = observed.min(), observed.max()
    if high == low:
        return np.where(np.isnan(v), np.nan, 0.0)
    return np.rint((v - low) / (high - low) * SCALE_MAX)


def blind_table(instance: ObservedInstance, seed: int) -> pd.DataFrame:
    """Tabela cega do A1: ≤ 200 linhas sorteadas, colunas permutadas e renomeadas, inteiros de 0 a 999 por coluna."""
    rng = np.random.default_rng(seed)
    n = len(instance.y)
    rows = rng.choice(n, size=min(MAX_ROWS, n), replace=False)
    columns = [np.asarray(instance.y, dtype=float)] + [instance.z[c].to_numpy(dtype=float) for c in instance.z.columns]
    order = rng.permutation(len(columns))
    return pd.DataFrame({f"c{j}": scale_to_int(columns[i][rows]) for j, i in enumerate(order)})


def render_a1(instance: ObservedInstance, seed: int) -> str:
    """Mensagem do usuário no A1: a tabela cega em CSV, sem espaços, com `NA` nas ausências."""
    table = blind_table(instance, seed)
    lines = [",".join(table.columns)]
    for row in table.to_numpy(dtype=float):
        lines.append(",".join(NA_TOKEN if np.isnan(v) else str(int(v)) for v in row))
    return "\n".join(lines) + "\n"


def render_a2(features: dict[str, float]) -> str:
    """Mensagem do usuário no A2: os descritores em JSON com chaves ordenadas, 4 algarismos significativos, null = NaN."""
    clean = {
        name: (None if value is None or not np.isfinite(value) else float(f"{float(value):.4g}"))
        for name, value in features.items()
    }
    return json.dumps(dict(sorted(clean.items())), separators=(",", ":"))
