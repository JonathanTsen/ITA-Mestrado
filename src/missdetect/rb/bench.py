"""Desenho e geração do benchmark de dados reais com ausência injetada (passos 32 e 37 do plano).

Desenho por família (plano 32, seção 5):
- uma base por taxa (alvo, driver e ordem das auxiliares sorteados; regra do driver balanceada entre as bases);
- em cada base, os níveis de n (subamostras aninhadas) e, em cada n, os níveis de k (auxiliares aninhadas, mesma
  máscara entre os k);
- β e direção atribuídos às células (base, n) em listas balanceadas e embaralhadas por família;
- gerador principal (simétrico) em todos os contextos; secundários (mdatagen, pares determinísticos; tab_err) nos
  contextos com n igual ao n do secundário;
- níveis de n ou k que colidem por falta de linhas ou de colunas são descartados e contados.

Cada instância grava só o observado em `observed/`; o oráculo vai para tabelas separadas (ver `runs.py`).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from missdetect.rb.config import InjectConfig
from missdetect.rb.ids import child_rng
from missdetect.rb.inject import (
    MECHANISMS,
    Base,
    Context,
    UnequalMissingCount,
    build_triple,
    context_rows,
    make_context,
    record,
    sample_base,
)
from missdetect.rb.runs import instance_row, oracle_row, write_complete, write_observed
from missdetect.rb.sources import id_like, read_local_csv

# Pares determinísticos da mdatagen por direção (MAR sobre o driver ↔ MNAR sobre o próprio Y; MCAR uniforme).
MDATAGEN_BY_DIRECTION = {
    "lower": {"MCAR": "random", "MAR": "lowest", "MNAR": "run_0"},
    "upper": {"MCAR": "random", "MAR": "highest", "MNAR": "run_1"},
    "both": {"MCAR": "random", "MAR": "mix", "MNAR": "run_0.5"},
}


@dataclass
class FamilyOutput:
    family_id: int
    eligibility: list[dict] = field(default_factory=list)
    bases: list[dict] = field(default_factory=list)
    contexts: list[dict] = field(default_factory=list)
    instances: list[dict] = field(default_factory=list)
    oracle: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def _json_scalar(value):
    return value.item() if hasattr(value, "item") else str(value)


def load_table(csv_dir: Path, openml_id: int, numeric_columns: list[str]) -> pd.DataFrame:
    """Tabela completa sem identificadores, só com as colunas numéricas não constantes do perfil (passo 31)."""
    frame = read_local_csv(Path(csv_dir) / f"{openml_id}.csv")
    frame = frame.drop(columns=id_like(frame))
    return frame[numeric_columns].astype(float)


def _balanced(values: tuple, size: int, rng: np.random.Generator) -> list:
    reps = -(-size // len(values))
    return list(rng.permutation(np.array(list(values) * reps, dtype=object))[:size])


def design_bases(
    table: pd.DataFrame, numeric: list[str], family_id: int, openml_id: int, cfg: InjectConfig, root_seed: int
) -> tuple[list[tuple[float, Base]], list[dict]]:
    """Uma base por taxa. Devolve [(taxa, base)] e os relatórios de elegibilidade de todas as tentativas."""
    rng = child_rng(root_seed, "design", family_id, "bases")
    rules = _balanced(cfg.driver_rules, len(cfg.rate_grid), rng)
    bases, reports = [], []
    for rate_idx, rate in enumerate(cfg.rate_grid):
        for attempt in range(cfg.max_base_attempts):
            base_idx = rate_idx * 1000 + attempt
            base, report = sample_base(
                table, numeric, family_id, openml_id, base_idx, root_seed,
                driver_rule=str(rules[rate_idx]), n_max=max(cfg.n_grid),
            )  # fmt: skip
            reports.append({"family_id": family_id, "openml_id": openml_id, "base_idx": base_idx, "rate": rate,
                            "driver_rule": str(rules[rate_idx]), "accepted": base is not None,
                            "n_eligible_columns": report["n_eligible_columns"], "tried": report["tried"]})  # fmt: skip
            if base is not None:
                bases.append((rate, base))
                break
    return bases, reports


def design_contexts(
    bases: list[tuple[float, Base]], cfg: InjectConfig, root_seed: int, family_id: int
) -> tuple[list[Context], list[dict]]:
    """Contextos (base × n × k) com β e direção balanceados por célula (base, n)."""
    cells, skipped = [], []
    for rate, base in bases:
        seen_n = set()
        for n in cfg.n_grid:
            n_eff = min(n, base.n_max)
            if n_eff in seen_n:
                skipped.append({"family_id": family_id, "base_id": base.base_id, "reason": f"n={n} colide ({n_eff})"})
                continue
            seen_n.add(n_eff)
            cells.append((rate, base, n))
    rng = child_rng(root_seed, "design", family_id, "cells")
    betas = _balanced(cfg.beta_grid, len(cells), rng)
    directions = _balanced(cfg.directions, len(cells), rng)
    contexts = []
    for (rate, base, n), beta, direction in zip(cells, betas, directions, strict=True):
        seen_k = set()
        for k in cfg.k_grid:
            k_eff = min(k, len(base.auxiliary_order))
            if k_eff in seen_k:
                skipped.append({"family_id": family_id, "base_id": base.base_id, "reason": f"k={k} colide ({k_eff})"})
                continue
            seen_k.add(k_eff)
            contexts.append(make_context(base, n, k, rate, float(beta), str(direction), root_seed))
    return contexts, skipped


def _generators_for(context: Context, base: Base, cfg: InjectConfig) -> list[tuple[str, dict[str, str]]]:
    out = [("symmetric", dict.fromkeys(MECHANISMS, ""))]
    for sec in cfg.secondary:
        if context.n != min(sec.n, base.n_max):
            continue
        if sec.generator == "mdatagen":
            out.append(("mdatagen", MDATAGEN_BY_DIRECTION[context.direction]))
        else:
            out.append((sec.generator, dict.fromkeys(MECHANISMS, "")))
    return out


def generate_family(run_dir: Path, csv_dir: Path, source: pd.Series, cfg: InjectConfig, root_seed: int) -> FamilyOutput:
    """Gera e grava todas as instâncias de uma família (representante do passo 31)."""
    t0 = pd.Timestamp.now()
    family_id, openml_id = int(source.family_id), int(source.openml_id)
    numeric = str(source.numeric_columns).split(";")
    table = load_table(csv_dir, openml_id, numeric)
    out = FamilyOutput(family_id=family_id)
    bases, reports = design_bases(table, numeric, family_id, openml_id, cfg, root_seed)
    out.eligibility = [{**r, "tried": json.dumps(r["tried"], default=_json_scalar)} for r in reports]
    out.bases = [{**record(base), "rate": rate} for rate, base in bases]
    base_by_id = {b.base_id: b for _, b in bases}
    contexts, out.skipped = design_contexts(bases, cfg, root_seed, family_id)
    for context in contexts:
        base = base_by_id[context.base_id]
        sub = context_rows(table, base, context)
        out.contexts.append(record(context))
        order = np.random.default_rng(context.shuffle_seed).permutation(context.n)
        write_complete(run_dir, context.context_id, sub[context.target].to_numpy(float)[order])
        for generator, variants in _generators_for(context, base, cfg):
            try:
                triple = build_triple(sub, context, generator, variants)
            except UnequalMissingCount as exc:
                out.skipped.append({"family_id": family_id, "base_id": base.base_id, "reason": str(exc)})
                continue
            for observed, oracle, _ in triple:
                rel = write_observed(run_dir, observed)
                out.instances.append(instance_row(observed, oracle, rel))
                out.oracle.append(oracle_row(oracle))
    out.seconds = (pd.Timestamp.now() - t0).total_seconds()
    return out
