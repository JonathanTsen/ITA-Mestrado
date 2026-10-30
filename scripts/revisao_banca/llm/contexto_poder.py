"""Poder de uma hipótese confirmatória sobre o contexto (30/10/2026; exploratório, nas 16 bases inventadas de
desenvolvimento). Diferença por base do macro-recall entre dois braços (por exemplo S7 − S0); dp entre bases; poder do
Wilcoxon-Pratt bilateral (α = 0,05) por simulação normal, para N bases confirmatórias.
Uso: uv run --frozen python scripts/revisao_banca/llm/contexto_poder.py <saída do conjunto 1> <saída do conjunto 2>
     <braço A> <braço B>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, wilcoxon

from missdetect.rb.evaluate import summarize_by_family

out1, out2, arm_a, arm_b = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
diffs = []
for out in (out1, out2):
    scores = {}
    for arm in (arm_a, arm_b):
        p = pd.read_parquet(out / f"predicoes_{arm}.parquet")
        scores[arm] = summarize_by_family(p)[0].set_index("family_id").macro_recall
    diffs.append(scores[arm_b] - scores[arm_a])
d = pd.concat(diffs).to_numpy()
mean, sd, n0 = float(d.mean()), float(d.std(ddof=1)), len(d)
sd_high = float(np.sqrt((n0 - 1) * sd**2 / chi2.ppf(0.025, n0 - 1)))
print(f"{arm_b} − {arm_a} por base: média {mean:+.3f}, dp {sd:.3f} (n = {n0} bases); dp pessimista {sd_high:.3f}")
rng = np.random.default_rng(20261101)


def power(delta: float, sd_: float, n: int, sims: int = 4000) -> float:
    hits = sum(
        wilcoxon(rng.normal(delta, sd_, n), zero_method="pratt", method="asymptotic", correction=False).pvalue < 0.05
        for _ in range(sims)
    )
    return hits / sims


for label, sd_ in (("dp observado", sd), ("dp pessimista", sd_high)):
    for delta in (0.05, 0.075, 0.10):
        row = []
        for n in (12, 16, 20, 24, 30):
            row.append(f"N={n}: {power(delta, sd_, n):.2f}")
        print(f"{label}, efeito {100 * delta:.1f} p.p. → " + "; ".join(row))
