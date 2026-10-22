"""Análise estatística confirmatória (passo 46), fixada junto com o pré-registro (passo 38).

Para cada hipótese "método × comparador", sobre os escores por família (macro-recall médio das 3 repetições):
- diferenças d_f = método − comparador, por família (só as famílias presentes nos dois);
- Wilcoxon de postos sinalizados bilateral, `zero_method="pratt"`, aproximação normal (`method="asymptotic"`), sem
  correção de continuidade; p ajustado por Holm dentro da família de hipóteses;
- estimando comparativo: Hodges–Lehmann (mediana das médias de Walsh d_i + d_j / 2, i ≤ j), com IC95% percentil por
  bootstrap de famílias (10.000 réplicas, semente fixa; IC não ajustado por multiplicidade);
- correlação rank-biserial pareada (fórmula de Kerby): (soma dos postos de |d| positivos − soma dos negativos) / soma
  total, sem as diferenças nulas;
- diferença média ± dp, só descritiva;
- teste bayesiano de postos sinalizados (Benavoli et al.) com ROPE ±0,02: `baycomp.SignedRankTest.probs(comparador,
  método)`, de modo que "direita" = método melhor; `prior=0.5`, 50.000 amostras, semente fixa. As probabilidades são
  as frações da posterior em que cada região (esquerda, ROPE, direita) é a mais provável.

Tabela de decisão (uma única regra, pré-registrada):
- "superior": p ajustado < 0,05 e HL > 0;  "inferior": p ajustado < 0,05 e HL < 0;
- "praticamente equivalente": P(ROPE) ≥ 0,95 (independentemente do p);
- "evidência insuficiente": nos demais casos.
Se as duas primeiras e a terceira valerem ao mesmo tempo (diferença significativa mas menor que a ROPE), o rótulo é
"superior/inferior, dentro da ROPE".
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from baycomp import SignedRankTest
from scipy import stats

ALPHA = 0.05
ROPE = 0.02
N_BOOT = 10_000
SEED = 20261101
BAYES_SAMPLES = 50_000
BAYES_PRIOR = 0.5
BAYES_THRESHOLD = 0.95


@dataclass(frozen=True)
class Comparison:
    hypothesis: str
    method: str
    comparator: str
    n_families: int
    mean_diff: float
    sd_diff: float
    hodges_lehmann: float
    ci_low: float
    ci_high: float
    rank_biserial: float
    p_value: float
    p_holm: float
    p_left: float
    p_rope: float
    p_right: float
    decision: str


def paired_differences(
    per_family: pd.DataFrame, method: str, comparator: str, metric: str = "macro_recall"
) -> pd.Series:
    """d_f = método − comparador, por família, a partir da tabela longa (method, family_id, métrica)."""
    wide = per_family.pivot(index="family_id", columns="method", values=metric)
    return (wide[method] - wide[comparator]).dropna()


def hodges_lehmann(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def bootstrap_hl_ci(d: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """IC95% percentil do Hodges–Lehmann, reamostrando famílias com reposição."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([hodges_lehmann(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def rank_biserial(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def wilcoxon_p(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return 1.0
    res = stats.wilcoxon(d, zero_method="pratt", correction=False, alternative="two-sided", method="asymptotic")
    return float(res.pvalue)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """p ajustados por Holm (step-down, monótonos, limitados a 1)."""
    names = sorted(p_values, key=p_values.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def bayes_signed_rank(
    method_scores: np.ndarray, comparator_scores: np.ndarray, rope: float = ROPE
) -> tuple[float, ...]:
    """(P esquerda = comparador melhor, P ROPE, P direita = método melhor)."""
    p_left, p_rope, p_right = SignedRankTest.probs(
        np.asarray(comparator_scores, dtype=float),
        np.asarray(method_scores, dtype=float),
        rope,
        nsamples=BAYES_SAMPLES,
        prior=BAYES_PRIOR,
        random_state=SEED,
    )
    return float(p_left), float(p_rope), float(p_right)


def decide(p_holm: float, hl: float, p_rope: float, alpha: float = ALPHA) -> str:
    significant = p_holm < alpha and hl != 0
    equivalent = p_rope >= BAYES_THRESHOLD
    if significant:
        label = "superior" if hl > 0 else "inferior"
        return f"{label}, dentro da ROPE" if equivalent else label
    if equivalent:
        return "praticamente equivalente"
    return "evidência insuficiente"


PRIMARY = "direct_hgb"
CONFIRMATORY = {
    "H1": (PRIMARY, "mechdetect_obs"),
    "H2": ("hier_hgb_fullprob", "direct_hgb"),
    "H3": (PRIMARY, "trivial_uniform"),
}
H4 = ("llm_a1", PRIMARY)


def _raw_comparison(per_family: pd.DataFrame, name: str, method: str, comparator: str, metric: str) -> dict:
    wide = per_family.pivot(index="family_id", columns="method", values=metric)
    pair = wide[[method, comparator]].dropna()
    d = (pair[method] - pair[comparator]).to_numpy()
    low, high = bootstrap_hl_ci(d)
    left, rope, right = bayes_signed_rank(pair[method].to_numpy(), pair[comparator].to_numpy())
    return {
        "hypothesis": name, "method": method, "comparator": comparator, "n_families": len(d),
        "mean_diff": float(np.mean(d)), "sd_diff": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
        "hodges_lehmann": hodges_lehmann(d), "ci_low": low, "ci_high": high, "rank_biserial": rank_biserial(d),
        "p_value": wilcoxon_p(d), "p_left": left, "p_rope": rope, "p_right": right,
    }  # fmt: skip


def _adjust(raw: list[dict], extra_p: dict[str, float] | None = None) -> pd.DataFrame:
    """Holm sobre todas as hipóteses da família (inclusive as que não rodaram, com p = 1) e rótulo de decisão."""
    p_values = {r["hypothesis"]: r["p_value"] for r in raw} | (extra_p or {})
    adjusted = holm(p_values)
    rows = []
    for r in raw:
        p_holm = adjusted[r["hypothesis"]]
        rows.append(asdict(Comparison(**r, p_holm=p_holm, decision=decide(p_holm, r["hodges_lehmann"], r["p_rope"]))))
    return pd.DataFrame(rows)


def compare_family(
    per_family: pd.DataFrame, hypotheses: dict[str, tuple[str, str]], metric: str = "macro_recall"
) -> pd.DataFrame:
    """Testa uma família de hipóteses {nome: (método, comparador)} da mesma tabela, com Holm dentro dela."""
    raw = [_raw_comparison(per_family, name, m, c, metric) for name, (m, c) in hypotheses.items()]
    return _adjust(raw)


def confirmatory(per_family_main: pd.DataFrame, per_family_llm: pd.DataFrame | None = None) -> pd.DataFrame:
    """H1–H3 nas famílias confirmatórias e H4 no painel de LLM, com Holm sobre as quatro (m = 4).

    Se a H4 não rodar (sem chaves ou corte de orçamento, decisão D5), ela entra no Holm com p = 1, e m continua 4.
    """
    raw = [_raw_comparison(per_family_main, name, m, c, "macro_recall") for name, (m, c) in CONFIRMATORY.items()]
    extra = {}
    if per_family_llm is not None:
        raw.append(_raw_comparison(per_family_llm, "H4", H4[0], H4[1], "macro_recall"))
    else:
        extra["H4"] = 1.0
    return _adjust(raw, extra)
