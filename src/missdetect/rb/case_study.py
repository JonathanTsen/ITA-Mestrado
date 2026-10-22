"""Estudo de caso com ausência natural (passo 45; decisão D3): sem acurácia, F1 nem concordância como métrica de acerto.

O pré-processamento é refeito para **não fabricar mecanismo** (IND-22): nada de `cap_missing_rate`, jitter, colunas de
ruído ou imputação. Uma ausência só é criada por uma regra documentada no registro (`data_rb/case_study_sources.csv`):
- `natural`: só os valores ausentes como a fonte os distribui;
- `zero_impossible`: zero fisicamente impossível pela definição da variável vira ausente (mesmo critério dos sentinelas
  da D4; ex.: insulina e prega cutânea da Pima);
- `below_lod:<coluna>`: a fonte marca na coluna de comentário (= 1) os resultados abaixo do limite de detecção, que ela
  preenche com um valor de substituição; esses viram ausentes, e as linhas sem resultado laboratorial (código ausente)
  ficam fora da população do caso (ex.: NHANES).

Auxiliares Z: as colunas listadas em `z_include`, ou, sem lista, **todas** as colunas completas que sejam numéricas ou
binárias (binárias viram 0/1 pela ordem das categorias), menos o alvo, os identificadores, as constantes e `z_exclude`.
Colunas que codificam a máscara ou o próprio Y (indicadores de medição, Y em outra unidade, outro analito censurado no
mesmo limite de detecção, colunas sentinela da D4, somas que contêm Y) vão para `z_exclude` com o motivo na evidência.
Como diagnóstico, `mask_auc_from_z` mede o quanto Z prevê a máscara (AUC de CV de uma logística); acima de 0,99, Z
praticamente determina a máscara, e o caso é marcado.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from missdetect.rb.inject import ObservedInstance
from missdetect.rb.sources import id_like

REGISTRY_COLUMNS = (
    "case_id", "source", "target", "missing_rule", "z_include", "z_exclude", "hypothesis", "evidence", "citation",
    "include", "exclusion_reason",
)  # fmt: skip
MIN_OBSERVED = 10
MASK_DETERMINED_AUC = 0.99


def _split(value: str) -> list[str]:
    return [v for v in str(value).split(";") if v]


def parse_registry(path: Path) -> list[dict]:
    """Casos do estudo; todo caso fora (`include = False`) precisa de motivo registrado."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REGISTRY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"registro sem as colunas {sorted(missing)}")
    specs = []
    for row in frame.to_dict("records"):
        row["include"] = row["include"].strip().lower() == "true"
        if not row["include"] and not row["exclusion_reason"].strip():
            raise ValueError(f"caso {row['case_id']} excluído sem motivo registrado")
        specs.append(row)
    return specs


def auxiliary_columns(table: pd.DataFrame, target: str, exclude: list[str]) -> list[str]:
    """Colunas completas, numéricas ou binárias, sem o alvo, identificadores, constantes e exclusões."""
    skip = {target, *exclude, *id_like(table)}
    cols = []
    for col in table.columns:
        values = table[col]
        if col in skip or values.isna().any() or values.nunique() < 2:
            continue
        numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        if numeric or values.nunique() == 2:
            cols.append(str(col))
    return cols


def _encode(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = {}
    for col in cols:
        values = table[col]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            out[col] = values.to_numpy(float)
        else:
            levels = sorted(values.astype(str).unique())
            out[col] = (values.astype(str) == levels[1]).to_numpy(float)
    return pd.DataFrame(out)


def mask_predictability(z: pd.DataFrame, m: np.ndarray) -> float:
    """AUC de CV (logística padronizada, até 5 folds estratificados) da máscara prevista a partir de Z."""
    m = np.asarray(m, dtype=int)
    folds = int(min(5, m.sum(), (1 - m).sum()))
    if folds < 2:
        return float("nan")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    proba = cross_val_predict(model, z.to_numpy(float), m, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(m, proba))


def build_case(table: pd.DataFrame, spec: dict) -> tuple[ObservedInstance, dict]:
    """Instância observada de um caso: Y com as ausências da fonte (e só as da regra documentada) e as auxiliares."""
    rule, target = spec["missing_rule"], spec["target"]
    exclude = _split(spec["z_exclude"])
    frame, n_dropped, code = table, 0, None
    if rule.startswith("below_lod:"):
        code = rule.split(":", 1)[1]
        keep = table[code].notna()
        n_dropped = int((~keep).sum())
        frame = table[keep].reset_index(drop=True)
        exclude.append(code)
    elif rule not in ("natural", "zero_impossible"):
        raise ValueError(f"regra de ausência desconhecida: {rule}")
    if not pd.api.types.is_numeric_dtype(frame[target]):
        raise ValueError(f"o alvo {target} não é numérico")
    y = frame[target].to_numpy(float).copy()
    natural = np.isnan(y)
    hit = np.zeros(len(y), bool)
    if rule == "zero_impossible":
        hit = y == 0
    elif code is not None:
        hit = frame[code].to_numpy(float) == 1
    y[hit] = np.nan
    if spec["z_include"]:
        cols = _split(spec["z_include"])
        incomplete = [c for c in cols if frame[c].isna().any()]
        if incomplete:
            raise ValueError(f"auxiliares com ausências: {incomplete}")
    else:
        cols = auxiliary_columns(frame, target, exclude)
    m = np.isnan(y)
    if not m.any():
        raise ValueError(f"caso {spec['case_id']} sem ausências no alvo")
    if (~m).sum() < MIN_OBSERVED:
        raise ValueError(f"caso {spec['case_id']} com menos de {MIN_OBSERVED} valores observados")
    if not cols:
        raise ValueError(f"caso {spec['case_id']}: nenhuma auxiliar completa (numérica ou binária) pela regra")
    z = _encode(frame, cols)
    meta = {
        "case_id": spec["case_id"], "n": len(y), "k": len(cols), "z_columns": ";".join(cols),
        "n_missing_natural": int(natural.sum()), "n_missing_rule": int((hit & ~natural).sum()),
        "n_rows_dropped": n_dropped, "missing_rate": float(m.mean()), "mask_auc_from_z": mask_predictability(z, m),
    }  # fmt: skip
    return ObservedInstance(spec["case_id"], y, z), meta
