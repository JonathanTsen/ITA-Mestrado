"""Injeção controlada de ausências na coluna-alvo (passo 32 do plano).

Definições (Cap2, `sec:mechanism_definitions`): MCAR, M ⫫ (Y, Z); MAR, M depende só do driver Z_d ∈ Z;
MNAR, M depende do próprio Y.

Componentes:
- elegibilidade do par (Y, Z), calculada na tabela completa antes de qualquer máscara;
- contextos pareados: tabela, subamostra, alvo, driver, auxiliares aninhadas, taxa, força e direção, com os três
  mecanismos aplicados ao mesmo contexto;
- geradores: simétrico (próprio), mdatagen 0.2.0 e tab_err, sempre restritos à coluna-alvo e semeados.

Fatos verificados no código das bibliotecas (registrados em `execucao/32-injecao.md` do plano):
- mdatagen 0.2.0: `uMAR` e `uMNAR` não aceitam `seed` e usam `np.random` e `random` globais; ambos acrescentam uma
  coluna `target` à saída; `uMAR.rank` indexa por rótulo (exige índice 0..n-1); `uMNAR.run(deterministic=False)`
  também ordena pela própria coluna (o aleatório só desempata); `mMNAR.MBOV_randomness` não atribui o
  `np.delete(...)`, então as posições aleatórias podem coincidir com as determinísticas e a taxa realizada fica
  abaixo da pedida.
- tab_err 0.2.1: mecanismos com `seed` são reproduzíveis; EAR/ENAR marcam uma janela contígua (início sorteado) dos
  valores ordenados do driver ou do alvo.
"""

import random
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from mdatagen.multivariate.mMNAR import mMNAR
from mdatagen.univariate.uMAR import uMAR
from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.univariate.uMNAR import uMNAR
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from tab_err import error_mechanism

from missdetect.rb.ids import child_rng, child_seed, stable_id

Mechanism = Literal["MCAR", "MAR", "MNAR"]
MECHANISMS: tuple[Mechanism, ...] = ("MCAR", "MAR", "MNAR")
DIRECTIONS = ("lower", "upper", "both")

MIN_DISTINCT = 20
MAX_MODE_FREQUENCY = 0.5
MAX_R2_Y_GIVEN_Z = 0.95
MAX_ABS_SPEARMAN = 0.95
MAX_RATE_DEVIATION = 0.01  # desvio máximo da taxa realizada (mdatagen) antes de substituir a variante


# ---------------------------------------------------------------------------------------------------------------------
# Elegibilidade
# ---------------------------------------------------------------------------------------------------------------------
def column_is_eligible(values: pd.Series) -> bool:
    """>= 20 valores distintos e valor mais frequente em no máximo metade das linhas."""
    counts = values.value_counts(dropna=True)
    return len(counts) >= MIN_DISTINCT and counts.iloc[0] / len(values) <= MAX_MODE_FREQUENCY


def pair_eligibility(table: pd.DataFrame, target: str, auxiliaries: list[str], seed: int) -> dict:
    """Critérios do par (Y, Z) na tabela completa: R² de Y dado Z por validação cruzada e |Spearman| máximo."""
    y = table[target].to_numpy(dtype=float)
    z = table[auxiliaries].to_numpy(dtype=float)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2 = float(cross_val_score(HistGradientBoostingRegressor(random_state=seed), z, y, cv=cv, scoring="r2").mean())
    rho = max(abs(spearmanr(y, table[c].to_numpy(dtype=float)).statistic) for c in auxiliaries)
    flags = {
        "r2_cv": r2,
        "max_abs_spearman": float(rho),
        "r2_ok": r2 <= MAX_R2_Y_GIVEN_Z,
        "spearman_ok": rho <= MAX_ABS_SPEARMAN,
    }
    flags["eligible"] = flags["r2_ok"] and flags["spearman_ok"]
    return flags


# ---------------------------------------------------------------------------------------------------------------------
# Bases e contextos
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Base:
    """Escolhas que não dependem dos fatores do ensaio: permutação das linhas, alvo, driver e ordem das auxiliares.

    Os contextos de uma base usam as primeiras n linhas da permutação (subamostras aninhadas) e as primeiras k
    auxiliares da ordem (Z_2 ⊂ Z_4 ⊂ Z_8, com o driver sempre em primeiro), o que permite estimar os efeitos de n e
    de k pareados dentro da mesma base.
    """

    base_id: str
    family_id: int
    openml_id: int
    base_idx: int
    target: str
    driver: str
    driver_rule: str
    auxiliary_order: tuple[str, ...]
    rows_seed: int
    n_max: int
    r2_cv: float
    max_abs_spearman: float


@dataclass(frozen=True)
class Context:
    context_id: str
    base_id: str
    family_id: int
    openml_id: int
    target: str
    driver: str
    auxiliaries: tuple[str, ...]
    n: int
    k: int
    rate: float
    beta: float
    direction: str
    mask_seed: int  # não depende de k: os níveis de k de uma mesma base compartilham a máscara
    shuffle_seed: int


def _base_permutation(n_rows: int, rows_seed: int) -> np.ndarray:
    return np.random.default_rng(rows_seed).permutation(n_rows)


def sample_base(
    table: pd.DataFrame,
    numeric_columns: list[str],
    family_id: int,
    openml_id: int,
    base_idx: int,
    root_seed: int,
    *,
    driver_rule: str,
    n_max: int = 2000,
) -> tuple[Base | None, dict]:
    """Sorteia alvo, driver e ordem das auxiliares; devolve (None, relatório) se nenhum alvo for elegível.

    A elegibilidade é calculada na maior subamostra (n_max linhas), com todas as auxiliares candidatas: um par
    elegível com todas as auxiliares também o é com qualquer subconjunto delas.
    """
    rng = child_rng(root_seed, "base", family_id, openml_id, base_idx)
    rows_seed = int(child_seed(root_seed, "rows", family_id, openml_id, base_idx))
    n_eff = min(n_max, len(table))
    sub = table.iloc[np.sort(_base_permutation(len(table), rows_seed)[:n_eff])].reset_index(drop=True)
    eligible = [c for c in numeric_columns if column_is_eligible(sub[c])]
    report: dict = {"n_eligible_columns": len(eligible), "tried": []}
    for target in rng.permutation(eligible):
        target = str(target)
        drivers = [c for c in eligible if c != target]
        others = [c for c in numeric_columns if c != target]
        if not drivers:
            continue
        flags = pair_eligibility(sub, target, others, child_seed(root_seed, "elig", openml_id, base_idx, target))
        report["tried"].append({"target": target, **flags})
        if not flags["eligible"]:
            continue
        if driver_rule == "maxcorr":
            corr = {c: abs(spearmanr(sub[target], sub[c]).statistic) for c in drivers}
            driver = max(sorted(corr), key=corr.get)
        else:
            driver = str(rng.choice(drivers))
        rest = [str(c) for c in rng.permutation([c for c in others if c != driver])]
        spec = {
            "family_id": family_id,
            "openml_id": openml_id,
            "base_idx": base_idx,
            "target": target,
            "driver": driver,
            "driver_rule": driver_rule,
        }
        base = Base(
            base_id=stable_id(spec),
            family_id=family_id,
            openml_id=openml_id,
            base_idx=base_idx,
            target=target,
            driver=driver,
            driver_rule=driver_rule,
            auxiliary_order=(driver, *rest),
            rows_seed=rows_seed,
            n_max=n_eff,
            r2_cv=flags["r2_cv"],
            max_abs_spearman=flags["max_abs_spearman"],
        )
        return base, report
    return None, report


def make_context(base: Base, n: int, k: int, rate: float, beta: float, direction: str, root_seed: int) -> Context:
    n_eff = min(n, base.n_max)
    auxiliaries = base.auxiliary_order[: min(k, len(base.auxiliary_order))]
    mask_spec = {"base_id": base.base_id, "n": n_eff, "rate": rate, "beta": beta, "direction": direction}
    spec = {**mask_spec, "k": len(auxiliaries)}
    return Context(
        context_id=stable_id(spec),
        base_id=base.base_id,
        family_id=base.family_id,
        openml_id=base.openml_id,
        target=base.target,
        driver=base.driver,
        auxiliaries=tuple(auxiliaries),
        n=n_eff,
        k=len(auxiliaries),
        rate=rate,
        beta=beta,
        direction=direction,
        mask_seed=int(child_seed(root_seed, "mask", stable_id(mask_spec))),
        shuffle_seed=int(child_seed(root_seed, "shuffle", stable_id(mask_spec))),
    )


def context_rows(table: pd.DataFrame, base: Base, context: Context) -> pd.DataFrame:
    """Subamostra do contexto: as primeiras n linhas da permutação da base, na ordem original da tabela."""
    rows = np.sort(_base_permutation(len(table), base.rows_seed)[: context.n])
    return table.iloc[rows].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
# Geradores
# ---------------------------------------------------------------------------------------------------------------------
def rank_scores(v: np.ndarray, direction: str) -> np.ndarray:
    s = (rankdata(v) - 0.5) / len(v) - 0.5
    if direction == "lower":
        return -s
    if direction == "both":
        return np.abs(s)
    return s


def symmetric_mask(v: np.ndarray, rate: float, beta: float, direction: str, rng: np.random.Generator) -> np.ndarray:
    """Exatamente round(rate·n) linhas, sorteadas sem reposição com pesos exp(beta·s) (Gumbel-top-k).

    MCAR: beta = 0. MAR: v = driver. MNAR: v = alvo completo. beta = inf: as maiores pontuações (desempate pela ordem
    das linhas, que é aleatória porque a subamostra foi sorteada).
    """
    n = len(v)
    m = round(rate * n)
    s = rank_scores(v, direction)
    if np.isinf(beta):
        idx = np.argsort(-s, kind="stable")[:m]
    else:
        keys = beta * s + rng.gumbel(size=n)
        idx = np.argsort(-keys, kind="stable")[:m]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


MDATAGEN_VARIANTS: dict[Mechanism, tuple[str, ...]] = {
    "MCAR": ("random", "binomial"),
    "MAR": ("lowest", "highest", "mix", "rank", "median"),
    "MNAR": ("run_0", "run_1", "run_0.5", "mbov_0.25", "mbov_0.5"),
}
MDATAGEN_DETERMINISTIC = {"lowest", "highest", "mix", "run_0", "run_1", "run_0.5"}


def mdatagen_mask(
    sub: pd.DataFrame, target: str, driver: str, mechanism: Mechanism, variant: str, rate: float, seed: int
):
    """Máscara de `target` gerada pela mdatagen 0.2.0, com todas as fontes de aleatoriedade semeadas.

    Devolve (máscara booleana, taxa realizada). Erro se a biblioteca alterar outra coluna.
    """
    X = sub.reset_index(drop=True)
    random.seed(seed)
    np.random.seed(seed)
    y_dummy = np.zeros(len(X), dtype=int)
    pct = round(rate * 100)
    if mechanism == "MCAR":
        gen = uMCAR(X=X, y=y_dummy, missing_rate=pct, x_miss=target, seed=seed)
        out = gen.random() if variant == "random" else gen.binomial()
    elif mechanism == "MAR":
        gen = uMAR(X=X, y=y_dummy, missing_rate=pct, x_miss=target, x_obs=driver)
        out = getattr(gen, variant)()
    elif variant.startswith("run_"):
        gen = uMNAR(X=X, y=y_dummy, missing_rate=pct, x_miss=target, threshold=float(variant.split("_")[1]))
        out = gen.run(deterministic=True)
    else:
        gen = mMNAR(X=X, y=y_dummy, n_Threads=1)
        out = gen.MBOV_randomness(missing_rate=pct, randomness=float(variant.split("_")[1]), columns=[target])
    out = out.drop(columns=["target"], errors="ignore").reset_index(drop=True)
    others = [c for c in X.columns if c != target]
    if out[others].isna().any().any() or not np.allclose(out[others].to_numpy(float), X[others].to_numpy(float)):
        raise RuntimeError("mdatagen alterou colunas auxiliares")
    mask = out[target].isna().to_numpy()
    return mask, float(mask.mean())


def tab_err_mask(sub: pd.DataFrame, target: str, driver: str, mechanism: Mechanism, rate: float, seed: int):
    """Máscara de `target` gerada pelo tab_err (API de baixo nível, só a coluna-alvo, mecanismo semeado)."""
    if mechanism == "MCAR":
        mech = error_mechanism.ECAR(seed=seed)
    elif mechanism == "MAR":
        mech = error_mechanism.EAR(condition_to_column=driver, seed=seed)
    else:
        mech = error_mechanism.ENAR(seed=seed)
    error_mask = mech.sample(sub.reset_index(drop=True), target, rate)
    if error_mask.drop(columns=[target]).to_numpy().any():
        raise RuntimeError("tab_err marcou colunas auxiliares")
    mask = error_mask[target].to_numpy(dtype=bool)
    return mask, float(mask.mean())


# ---------------------------------------------------------------------------------------------------------------------
# Instâncias
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ObservedInstance:
    """O que o método vê: Y com NaN e as auxiliares. Não contém rótulo, gerador nem parâmetros."""

    sample_id: str
    y: np.ndarray
    z: pd.DataFrame


@dataclass(frozen=True)
class OracleRecord:
    """O que só o gerador e o avaliador leem."""

    sample_id: str
    context_id: str
    family_id: int
    mechanism: str
    generator: str
    variant: str
    deterministic: bool
    realized_rate: float
    mask_sha256: str


def build_instance(
    sub: pd.DataFrame, context: Context, generator: str, mechanism: Mechanism, variant: str = ""
) -> tuple[ObservedInstance, OracleRecord, np.ndarray]:
    """Gera uma instância do contexto. A permutação final das linhas é a mesma para os três mecanismos."""
    seed = context.mask_seed
    if generator == "symmetric":
        rng = child_rng(seed, "symmetric", mechanism)
        if mechanism == "MCAR":
            mask = symmetric_mask(np.zeros(context.n), context.rate, 0.0, context.direction, rng)
        else:
            v = sub[context.driver if mechanism == "MAR" else context.target].to_numpy(dtype=float)
            mask = symmetric_mask(v, context.rate, context.beta, context.direction, rng)
        deterministic = mechanism != "MCAR" and np.isinf(context.beta)
        realized = float(mask.mean())
    elif generator == "mdatagen":
        mask, realized = mdatagen_mask(sub, context.target, context.driver, mechanism, variant, context.rate, seed)
        deterministic = variant in MDATAGEN_DETERMINISTIC
    elif generator == "tab_err":
        mask, realized = tab_err_mask(sub, context.target, context.driver, mechanism, context.rate, seed)
        deterministic = mechanism != "MCAR"
    else:
        raise ValueError(generator)

    order = np.random.default_rng(context.shuffle_seed).permutation(context.n)
    y = sub[context.target].to_numpy(dtype=float).copy()
    y[mask] = np.nan
    z = sub[list(context.auxiliaries)].iloc[order].reset_index(drop=True)
    spec = {"context_id": context.context_id, "generator": generator, "mechanism": mechanism, "variant": variant}
    sample_id = stable_id(spec)
    mask_sha = stable_id({"mask": np.flatnonzero(mask[order]).tolist()}, length=64)
    observed = ObservedInstance(sample_id=sample_id, y=y[order], z=z)
    oracle = OracleRecord(
        sample_id=sample_id,
        context_id=context.context_id,
        family_id=context.family_id,
        mechanism=mechanism,
        generator=generator,
        variant=variant,
        deterministic=bool(deterministic),
        realized_rate=realized,
        mask_sha256=mask_sha,
    )
    return observed, oracle, mask[order]


def record(item: Base | Context) -> dict:
    """Linha para `contexts.parquet` (tuplas viram texto separado por ";")."""
    out = asdict(item)
    for key, value in out.items():
        if isinstance(value, tuple):
            out[key] = ";".join(value)
    return out
