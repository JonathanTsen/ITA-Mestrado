"""Análise estatística confirmatória (passo 46), fixada junto com o pré-registro (passo 38, versão 2 de 27/10/2026).

Para cada hipótese "método × comparador", sobre os escores por família (macro-recall médio das 3 repetições):
- diferenças d_f = método − comparador, por família (só as famílias presentes nos dois);
- Wilcoxon de postos sinalizados bilateral, `zero_method="pratt"`, aproximação normal (`method="asymptotic"`), sem
  correção de continuidade; p ajustado por Holm dentro da família de hipóteses;
- equivalência (só onde pré-registrada, H2e): TOST de Wilcoxon com margem δ = 0,02, isto é, dois testes unilaterais
  sobre d + δ (H0: mediana ≤ −δ) e d − δ (H0: mediana ≥ δ); o p é o maior dos dois e entra no mesmo Holm;
- teste de uma amostra (H6): AUC MAR × MNAR por família contra 0,5, com o mesmo Wilcoxon sobre AUC − 0,5;
- estimando comparativo: Hodges–Lehmann (mediana das médias de Walsh d_i + d_j / 2, i ≤ j), com IC95% percentil por
  bootstrap de famílias (10.000 réplicas, semente fixa). O IC não é ajustado por multiplicidade e é só descritivo: a
  decisão vem do p ajustado;
- correlação rank-biserial pareada (fórmula de Kerby) e diferença média ± dp, só descritivas;
- teste bayesiano de postos sinalizados (Benavoli et al.) com ROPE ±δ: `baycomp.SignedRankTest.probs(comparador,
  método)`, de modo que "direita" = método melhor; `prior=0.5`, 50.000 amostras, semente fixa. **Só descritivo** na
  versão 2 do pré-registro: a equivalência é decidida pelo TOST.

Família confirmatória principal (Holm m = 6): H1, H2 (diferença), H2e (equivalência), H3, H5 e H6. A H4 (LLM) é uma
família própria (m = 1). Uma hipótese que não puder rodar entra no Holm da sua família com p = 1.

Tabela de decisão (uma única regra, pré-registrada):
- teste de diferença: "superior" ou "inferior" se o p ajustado < 0,05, pelo sinal do Hodges–Lehmann. Se |HL| < δ, o
  rótulo ganha "(efeito menor que δ)": a diferença é significativa, mas fica abaixo da relevância prática. Nos demais
  casos, "evidência insuficiente";
- teste de equivalência: "equivalentes dentro de ±δ" se o p ajustado < 0,05; senão, "equivalência não demonstrada".
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from baycomp import SignedRankTest
from scipy import stats

ALPHA = 0.05
DELTA = 0.02  # margem de equivalência e de relevância prática, em macro-recall
ROPE = DELTA
N_BOOT = 10_000
SEED = 20261101
BAYES_SAMPLES = 50_000
BAYES_PRIOR = 0.5


@dataclass(frozen=True)
class Comparison:
    hypothesis: str
    method: str
    comparator: str
    test: str
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


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """IC95% percentil da média entre famílias (bootstrap de famílias)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
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


def wilcoxon_tost(d: np.ndarray, delta: float = DELTA) -> float:
    """p do TOST de Wilcoxon: equivalência dentro de ±delta quando as duas nulas unilaterais são rejeitadas."""
    d = np.asarray(d, dtype=float)
    options = {"zero_method": "pratt", "correction": False, "method": "asymptotic"}
    lower = stats.wilcoxon(d + delta, alternative="greater", **options).pvalue  # H0: mediana ≤ −δ
    upper = stats.wilcoxon(d - delta, alternative="less", **options).pvalue  # H0: mediana ≥ δ
    return float(max(lower, upper))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """p ajustados por Holm (step-down, monótonos, limitados a 1)."""
    names = sorted(p_values, key=lambda name: p_values[name])
    m = len(names)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def bayes_signed_rank(
    method_scores: np.ndarray, comparator_scores: np.ndarray, rope: float = ROPE
) -> tuple[float, ...]:
    """(P esquerda = comparador melhor, P ROPE, P direita = método melhor). Só descritivo."""
    p_left, p_rope, p_right = SignedRankTest.probs(
        np.asarray(comparator_scores, dtype=float),
        np.asarray(method_scores, dtype=float),
        rope,
        nsamples=BAYES_SAMPLES,
        prior=BAYES_PRIOR,
        random_state=SEED,
    )
    return float(p_left), float(p_rope), float(p_right)


def decide(p_holm: float, hl: float, delta: float = DELTA, alpha: float = ALPHA) -> str:
    """Rótulo de um teste de diferença."""
    if p_holm < alpha and hl != 0:
        label = "superior" if hl > 0 else "inferior"
        return f"{label} (efeito menor que δ)" if abs(hl) < delta else label
    return "evidência insuficiente"


def decide_equivalence(p_holm: float, alpha: float = ALPHA) -> str:
    """Rótulo de um teste de equivalência (TOST)."""
    return "equivalentes dentro de ±δ" if p_holm < alpha else "equivalência não demonstrada"


PRIMARY = "direct_hgb"
DIFFERENCE = {  # hipótese: (método, comparador, tabela de escores)
    "H1": (PRIMARY, "mechdetect_obs", "main"),
    "H2": ("hier_hgb_fullprob", PRIMARY, "main"),
    "H3": (PRIMARY, "logit_simple", "main"),
    "H5": ("cross_direct_hgb", "mechdetect_obs", "cross"),
}
ORDER = ("H1", "H2", "H2e", "H3", "H5", "H6")
H4 = ("llm_a1", PRIMARY)


def _pair(per_family: pd.DataFrame, method: str, comparator: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    wide = per_family.pivot(index="family_id", columns="method", values=metric)
    pair = wide[[method, comparator]].dropna()
    return pair[method].to_numpy(float), pair[comparator].to_numpy(float)


def _describe(name: str, method: str, comparator: str, test: str, a: np.ndarray, b: np.ndarray, p: float) -> dict:
    d = a - b
    low, high = bootstrap_hl_ci(d)
    left, rope, right = bayes_signed_rank(a, b)
    return {
        "hypothesis": name, "method": method, "comparator": comparator, "test": test, "n_families": len(d),
        "mean_diff": float(np.mean(d)), "sd_diff": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
        "hodges_lehmann": hodges_lehmann(d), "ci_low": low, "ci_high": high, "rank_biserial": rank_biserial(d),
        "p_value": p, "p_left": left, "p_rope": rope, "p_right": right,
    }  # fmt: skip


def _raw_comparison(per_family: pd.DataFrame, name: str, method: str, comparator: str, metric: str) -> dict:
    a, b = _pair(per_family, method, comparator, metric)
    return _describe(name, method, comparator, "wilcoxon", a, b, wilcoxon_p(a - b))


def _raw_equivalence(per_family: pd.DataFrame, name: str, method: str, comparator: str, metric: str) -> dict:
    a, b = _pair(per_family, method, comparator, metric)
    return _describe(name, method, comparator, "tost", a, b, wilcoxon_tost(a - b))


def _raw_one_sample(name: str, method: str, values: np.ndarray, reference: float, label: str) -> dict:
    values = np.asarray(values, dtype=float)
    ref = np.full(len(values), reference)
    return _describe(name, method, label, "wilcoxon_1amostra", values, ref, wilcoxon_p(values - reference))


def _adjust(raw: list[dict], extra_p: dict[str, float] | None = None) -> pd.DataFrame:
    """Holm sobre todas as hipóteses da família (inclusive as que não rodaram, com p = 1) e rótulo de decisão."""
    p_values = {r["hypothesis"]: r["p_value"] for r in raw} | (extra_p or {})
    adjusted = holm(p_values)
    rows = []
    for r in raw:
        p_holm = adjusted[r["hypothesis"]]
        label = decide_equivalence(p_holm) if r["test"] == "tost" else decide(p_holm, r["hodges_lehmann"])
        rows.append(asdict(Comparison(**r, p_holm=p_holm, decision=label)))
    return pd.DataFrame(rows)


def compare_family(
    per_family: pd.DataFrame, hypotheses: dict[str, tuple[str, str]], metric: str = "macro_recall"
) -> pd.DataFrame:
    """Testa uma família de hipóteses {nome: (método, comparador)} da mesma tabela, com Holm dentro dela."""
    raw = [_raw_comparison(per_family, name, m, c, metric) for name, (m, c) in hypotheses.items()]
    return _adjust(raw)


def confirmatory(
    per_family_main: pd.DataFrame,
    per_family_cross: pd.DataFrame | None = None,
    auc_mar_mnar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Família confirmatória principal (Holm m = 6).

    - `per_family_main`: escores por família no gerador principal (H1, H2, H2e, H3);
    - `per_family_cross`: escores por família nos geradores externos, com o modelo treinado só no principal (H5);
    - `auc_mar_mnar`: AUC MAR × MNAR por família da configuração primária, colunas `family_id` e `auc` (H6).

    Uma tabela ausente faz a hipótese correspondente entrar no Holm com p = 1."""
    tables = {"main": per_family_main, "cross": per_family_cross}
    raw, missing = [], {}
    for name in ORDER:
        if name == "H2e":
            method, comparator, _ = DIFFERENCE["H2"]
            raw.append(_raw_equivalence(per_family_main, name, method, comparator, "macro_recall"))
        elif name == "H6":
            if auc_mar_mnar is None:
                missing[name] = 1.0
            else:
                raw.append(_raw_one_sample(name, PRIMARY, auc_mar_mnar.auc.to_numpy(), 0.5, "0,5"))
        else:
            method, comparator, table = DIFFERENCE[name]
            if tables[table] is None:
                missing[name] = 1.0
            else:
                raw.append(_raw_comparison(tables[table], name, method, comparator, "macro_recall"))
    return _adjust(raw, missing)


def llm_hypothesis(per_family_llm: pd.DataFrame) -> pd.DataFrame:
    """H4 (LLM-A1 × configuração primária, no painel de LLM): família própria, m = 1."""
    return _adjust([_raw_comparison(per_family_llm, "H4", H4[0], H4[1], "macro_recall")])


def level_estimate(per_family: pd.DataFrame, method: str = PRIMARY, metric: str = "macro_recall") -> dict:
    """Estimando de nível: média entre famílias do escore por família, com IC95% por bootstrap de famílias."""
    values = per_family.loc[per_family.method == method, metric].dropna().to_numpy(float)
    low, high = bootstrap_mean_ci(values)
    return {"method": method, "n_families": len(values), "mean": float(values.mean()), "ci_low": low, "ci_high": high}
