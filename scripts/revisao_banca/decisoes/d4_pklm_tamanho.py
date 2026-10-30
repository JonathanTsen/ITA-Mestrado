"""D4: tamanho do PKLM sob MCAR puro (dados gaussianos, uma coluna com ausência), na configuração do piloto.

Uso: uv run --frozen python scripts/revisao_banca/decisoes/d4_pklm_tamanho.py <dir vazio> <réplicas> <num.proj> <nrep>
     [<saída.csv>] [<semente do PKLM>]

Com <saída.csv>, grava o p-valor de cada tabela (a execução de 21/10 só imprimiu o total e não pôde ser conferida).
As tabelas usam as sementes 1000 + s, então as 48 primeiras de qualquer execução são as mesmas. A semente do PKLM é 7
(E7, E19); o R4 do plano 38c usa a da produção (20261101, a `root_seed` passada pelo pipeline).
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
    csv_out = Path(sys.argv[5]) if len(sys.argv) > 5 else None
    pklm_seed = int(sys.argv[6]) if len(sys.argv) > 6 else 7
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
    res = run_pklm(out, ids, num_proj, nrep, seed=pklm_seed, n_jobs=8)
    if csv_out is not None:
        res.assign(num_proj=num_proj, nrep=nrep).to_csv(csv_out, index=False)
    rejected = int((res.pklm_p < 0.05).sum())
    print(f"num.proj={num_proj} nrep={nrep}: rejeição a 5% = {rejected / len(res):.3f} ({rejected}/{len(res)})")


if __name__ == "__main__":
    main()
