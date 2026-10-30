"""Painel de LLM (pré-registro §4, H4; passo 43; 30/10/2026).

- **Famílias:** 30 das famílias efetivas, sorteadas com `child_rng(raiz, "llm_panel")` (todas, se houver ≤ 30, como nas 8
  de desenvolvimento do piloto).
- **Contextos:** em cada família, os do gerador principal com r ∈ {0,1; 0,3; 0,5} × k ∈ {2, maior k disponível na base}
  e n = min(1000, nº de linhas da base): 6 contextos e 18 instâncias por família (menos, se a base só tem k = 2).
- **Repetição (P20):** 10% das instâncias do painel, sorteadas com `child_rng(raiz, "llm_repeat")`, são perguntadas
  de novo; a concordância é descritiva.
- **Ordem (P22):** as famílias rodam numa ordem sorteada com `child_rng(raiz, "llm_order")`; se o orçamento acabar, a H4
  usa as famílias completas.
"""

import numpy as np
import pandas as pd

from missdetect.rb.ids import child_rng

N_PANEL_FAMILIES = 30
PANEL_RATES = (0.1, 0.3, 0.5)
PANEL_N = 1000
REPEAT_FRACTION = 0.10
GENERATOR = "symmetric"


def panel_families(effective: list[int], root_seed: int, n_families: int = N_PANEL_FAMILIES) -> list[int]:
    families = sorted(int(f) for f in effective)
    if len(families) <= n_families:
        return families
    rng = child_rng(root_seed, "llm_panel")
    return sorted(int(f) for f in rng.choice(families, size=n_families, replace=False))


def family_order(families: list[int], root_seed: int) -> list[int]:
    ordered = sorted(int(f) for f in families)
    return [ordered[i] for i in child_rng(root_seed, "llm_order").permutation(len(ordered))]


def panel_contexts(contexts: pd.DataFrame, bases: pd.DataFrame, families: list[int]) -> pd.DataFrame:
    ctx = contexts[contexts.family_id.isin(families)].copy()
    ctx = ctx[np.isclose(ctx.rate.to_numpy(float)[:, None], PANEL_RATES).any(axis=1)]
    n_max = ctx.base_id.map(bases.set_index("base_id").n_max)
    ctx = ctx[ctx.n == np.minimum(PANEL_N, n_max)]
    k_max = ctx.groupby("base_id").k.transform("max")
    ctx = ctx[(ctx.k == 2) | (ctx.k == k_max)]
    return ctx.sort_values(["family_id", "rate", "k"]).reset_index(drop=True)


def panel_instances(instances: pd.DataFrame, oracle: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    """As instâncias do gerador principal nos contextos do painel, com o rótulo (só para a avaliação)."""
    merged = instances[instances.context_id.isin(contexts.context_id)].merge(
        oracle[["sample_id", "mechanism", "generator"]], on="sample_id"
    )
    merged = merged[merged.generator == GENERATOR]
    return merged.sort_values(["family_id", "context_id", "mechanism"]).reset_index(drop=True)


def repeat_subsample(sample_ids: pd.Series, root_seed: int, fraction: float = REPEAT_FRACTION) -> list[str]:
    ids = sorted(str(s) for s in sample_ids)
    size = round(fraction * len(ids))
    chosen = child_rng(root_seed, "llm_repeat").choice(len(ids), size=size, replace=False)
    return sorted(ids[i] for i in chosen)
