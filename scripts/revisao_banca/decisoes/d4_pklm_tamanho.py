"""D4: tamanho do PKLM sob MCAR puro (dados gaussianos, uma coluna com ausência), na configuração do piloto.

Uso: uv run --frozen python scripts/revisao_banca/decisoes/d4_pklm_tamanho.py <dir vazio> <réplicas> <num.proj> <nrep>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from missdetect.rb.baselines.pklm_r import run_pklm
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.runs import write_observed


def main() -> None:
    out, n_rep, num_proj, nrep = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    ids = []
    for s in range(n_rep):
        rng = np.random.default_rng(1000 + s)
        n, k = 1000, 4
        z = rng.multivariate_normal(np.zeros(k), 0.5 * np.eye(k) + 0.5, size=n)
        y = z @ rng.normal(size=k) * 0.5 + rng.normal(size=n)
        y[rng.permutation(n)[:300]] = np.nan
        instance = ObservedInstance(f"inst{s:03d}", y, pd.DataFrame(z, columns=[f"z{j}" for j in range(k)]))
        write_observed(out, instance)
        ids.append(instance.sample_id)
    res = run_pklm(out, ids, num_proj, nrep, seed=7, n_jobs=8)
    rejected = int((res.pklm_p < 0.05).sum())
    print(f"num.proj={num_proj} nrep={nrep}: rejeição a 5% = {rejected / len(res):.3f} ({rejected}/{len(res)})")


if __name__ == "__main__":
    main()
