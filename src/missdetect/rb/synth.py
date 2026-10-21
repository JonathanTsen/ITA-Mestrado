"""Corpus sintético: tabelas com dependência realista entre colunas e as mesmas máscaras do benchmark real (passo 33).

Tabela-base: cópula gaussiana com p = k + 1 colunas.
- Z latente ~ N(0, C), com C sorteada por `scipy.stats.random_correlation` (autovalores de uma Dirichlet);
- Y latente = sqrt(R²)·s + sqrt(1 − R²)·ε, com s = Z·w padronizado; R² latente ~ U(0, `latent_r2_max`), o que
  controla a força da dependência entre Y e Z;
- cada coluna recebe uma marginal sorteada (normal, log-normal, exponencial, beta, uniforme, mistura de duas normais,
  discreta com >= 20 níveis) aplicada a Φ(latente).

Máscaras: os mesmos geradores do passo 32 (`inject.build_instance`), com os três mecanismos sobre a mesma base; cada
tabela-base é um grupo (`family_id`) nas divisões de CV. As mesmas regras de elegibilidade do passo 32 valem; uma base
inelegível é sorteada de novo (tentativas registradas). Não há MCAR "em bloco": a ordem das linhas é arbitrária.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import random_correlation, spearmanr

from missdetect.rb.bench import MDATAGEN_BY_DIRECTION
from missdetect.rb.config import SynthConfig
from missdetect.rb.ids import child_rng, child_seed, stable_id
from missdetect.rb.inject import (
    Base,
    UnequalMissingCount,
    build_triple,
    column_is_eligible,
    make_context,
    pair_eligibility,
    record,
)
from missdetect.rb.runs import instance_row, oracle_row, write_complete, write_observed

SYNTH_FAMILY_OFFSET = 1_000_000  # family_id sintético = offset + índice da base (não colide com IDs OpenML)
TARGET = "t"


@dataclass(frozen=True)
class SynthDesign:
    base_idx: int
    n: int
    k: int
    rate: float
    beta: float
    direction: str
    generator: str
    driver_rule: str


@dataclass
class SynthOutput:
    bases: list[dict] = field(default_factory=list)
    contexts: list[dict] = field(default_factory=list)
    instances: list[dict] = field(default_factory=list)
    oracle: list[dict] = field(default_factory=list)


def design(cfg: SynthConfig, root_seed: int) -> list[SynthDesign]:
    """Fatores de cada base em listas balanceadas e embaralhadas (n, k, taxa, β, direção, gerador, regra do driver)."""
    rng = child_rng(root_seed, "synth", "design")
    size = cfg.n_bases

    def balanced(values) -> list:
        reps = -(-size // len(values))
        return list(rng.permutation(np.array(list(values) * reps, dtype=object))[:size])

    names = [g for g, _ in cfg.generator_shares]
    weights = np.array([w for _, w in cfg.generator_shares], dtype=float)
    counts = np.floor(weights / weights.sum() * size).astype(int)
    counts[0] += size - counts.sum()
    generators = list(rng.permutation(np.repeat(np.array(names, dtype=object), counts)))
    columns = [balanced(v) for v in (cfg.n_grid, cfg.k_grid, cfg.rate_grid, cfg.beta_grid, cfg.directions)]
    rules = balanced(("maxcorr", "random"))
    return [
        SynthDesign(i, int(n), int(k), float(r), float(b), str(d), str(g), str(rule))
        for i, (n, k, r, b, d, g, rule) in enumerate(zip(*columns, generators, rules, strict=True))
    ]


def _marginal(name: str, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = np.clip(u, 1e-9, 1 - 1e-9)
    if name == "normal":
        return stats.norm.ppf(u, loc=rng.normal(0, 5), scale=rng.uniform(0.5, 5))
    if name == "lognormal":
        return stats.lognorm.ppf(u, s=rng.uniform(0.3, 1.2), scale=np.exp(rng.normal(0, 1)))
    if name == "exponential":
        return stats.expon.ppf(u, scale=rng.uniform(0.5, 10))
    if name == "beta":
        return stats.beta.ppf(u, a=rng.uniform(0.5, 5), b=rng.uniform(0.5, 5))
    if name == "uniform":
        low = rng.normal(0, 10)
        return low + u * rng.uniform(1, 20)
    if name == "mixture":
        # Mistura de duas normais: quantis por interpolação da CDF numa grade fina.
        w, gap = rng.uniform(0.2, 0.8), rng.uniform(2, 6)
        grid = np.linspace(-6, gap + 6, 4001)
        cdf = w * stats.norm.cdf(grid) + (1 - w) * stats.norm.cdf(grid, loc=gap)
        return np.interp(u, cdf, grid)
    if name == "discrete":
        trials = int(rng.integers(40, 121))
        return stats.binom.ppf(u, trials, rng.uniform(0.3, 0.7))
    raise ValueError(name)


def sample_table(n: int, k: int, cfg: SynthConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    eig = rng.dirichlet(np.ones(k)) * k
    corr = random_correlation.rvs(eig, random_state=rng) if k > 1 else np.ones((1, 1))
    z_lat = rng.multivariate_normal(np.zeros(k), corr, size=n, method="cholesky")
    w = rng.normal(size=k)
    s = z_lat @ w / np.sqrt(w @ corr @ w)
    r2 = float(rng.uniform(0, cfg.latent_r2_max))
    y_lat = np.sqrt(r2) * s + np.sqrt(1 - r2) * rng.normal(size=n)
    latent = np.column_stack([y_lat, z_lat])
    marginals = [str(m) for m in rng.choice(cfg.marginals, size=k + 1)]
    columns = {}
    for j, name in enumerate(marginals):
        col = TARGET if j == 0 else f"c{j}"
        columns[col] = _marginal(name, stats.norm.cdf(latent[:, j]), rng)
    meta = {"latent_r2": r2, "marginals": ";".join(marginals), "max_abs_latent_corr_z": float(np.abs(
        corr[np.triu_indices(k, 1)]).max()) if k > 1 else 0.0}  # fmt: skip
    return pd.DataFrame(columns), meta


def _driver(table: pd.DataFrame, auxiliaries: list[str], rule: str, rng: np.random.Generator) -> str:
    if rule == "maxcorr":
        corr = {c: abs(spearmanr(table[TARGET], table[c]).statistic) for c in auxiliaries}
        return max(sorted(corr), key=corr.get)
    return str(rng.choice(auxiliaries))


def generate_base(run_dir: Path, item: SynthDesign, cfg: SynthConfig, root_seed: int) -> SynthOutput:
    """Sorteia uma tabela elegível para o desenho `item` e grava as três instâncias (mesmo contexto)."""
    out = SynthOutput()
    family_id = SYNTH_FAMILY_OFFSET + item.base_idx
    for attempt in range(cfg.max_attempts_per_base):
        rng = child_rng(root_seed, "synth", "table", item.base_idx, attempt)
        table, meta = sample_table(item.n, item.k, cfg, rng)
        auxiliaries = [c for c in table.columns if c != TARGET]
        ok_columns = column_is_eligible(table[TARGET]) and any(column_is_eligible(table[c]) for c in auxiliaries)
        flags = {"r2_cv": np.nan, "max_abs_spearman": np.nan, "eligible": False}
        if ok_columns:
            flags = pair_eligibility(table, TARGET, auxiliaries, child_seed(root_seed, "synth", "elig", item.base_idx))
        row = {"base_idx": item.base_idx, "attempt": attempt, "columns_ok": ok_columns, **meta, **flags}
        if not (ok_columns and flags["eligible"]):
            out.bases.append({**row, "accepted": False})
            continue
        drivers = [c for c in auxiliaries if column_is_eligible(table[c])]
        driver = _driver(table, drivers, item.driver_rule, rng)
        rest = [str(c) for c in rng.permutation([c for c in auxiliaries if c != driver])]
        spec = {"synth_base": item.base_idx, "attempt": attempt, "driver": driver, "rule": item.driver_rule}
        base = Base(
            base_id=stable_id(spec), family_id=family_id, openml_id=-1, base_idx=item.base_idx, target=TARGET,
            driver=driver, driver_rule=item.driver_rule, auxiliary_order=(driver, *rest),
            rows_seed=int(child_seed(root_seed, "synth", "rows", item.base_idx)), n_max=item.n,
            r2_cv=flags["r2_cv"], max_abs_spearman=flags["max_abs_spearman"],
        )  # fmt: skip
        context = make_context(base, item.n, item.k, item.rate, item.beta, item.direction, root_seed)
        variants = MDATAGEN_BY_DIRECTION[item.direction] if item.generator == "mdatagen" else {}
        try:
            triple = build_triple(table, context, item.generator, variants)
        except UnequalMissingCount:
            out.bases.append({**row, "accepted": False, "unequal_count": True})
            continue
        out.bases.append({**row, "accepted": True, **record(base), "generator": item.generator})
        out.contexts.append(record(context))
        order = np.random.default_rng(context.shuffle_seed).permutation(context.n)
        write_complete(run_dir, context.context_id, table[TARGET].to_numpy(float)[order])
        for observed, oracle, _ in triple:
            rel = write_observed(run_dir, observed)
            out.instances.append(instance_row(observed, oracle, rel))
            out.oracle.append(oracle_row(oracle))
        return out
    return out
