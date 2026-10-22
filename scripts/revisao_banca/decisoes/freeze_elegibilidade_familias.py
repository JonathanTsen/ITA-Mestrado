"""Quantas famílias aceitam bases com as regras congeladas (freeze-v1), só com a elegibilidade nas tabelas completas.

Não gera máscara nem instância: roda `design_bases`, como a injeção faz no início, e grava, por família, as bases aceitas
e o motivo das rejeições.

Uso: uv run --frozen python scripts/revisao_banca/decisoes/freeze_elegibilidade_familias.py <saída.csv>
"""

import json
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from missdetect.rb.bench import design_bases, load_sentinel_exclusions, load_table
from missdetect.rb.config import load_config
from missdetect.rb.inject import column_is_eligible


def family_report(row: pd.Series, config) -> dict:
    excluded = load_sentinel_exclusions(config.sources.sentinel_exclusions).get(int(row.openml_id), set())
    numeric = [c for c in str(row.numeric_columns).split(";") if c not in excluded]
    table = load_table(Path(config.sources.csv_dir), int(row.openml_id), numeric)
    with threadpool_limits(1):
        bases, reports = design_bases(table, numeric, int(row.family_id), int(row.openml_id), config.inject, config.root_seed)
    tried = [t for r in reports for t in r["tried"]]
    return {
        "family_id": int(row.family_id),
        "openml_id": int(row.openml_id),
        "role": row.role,
        "n_rows": int(row.n_rows),
        "n_numeric": len(numeric),
        "n_eligible_columns": int(sum(column_is_eligible(table[c]) for c in numeric)),
        "bases_accepted": len(bases),
        "targets_tried": len(tried),
        "rejected_r2": int(sum(not t["r2_ok"] for t in tried)),
        "rejected_spearman": int(sum(not t["spearman_ok"] for t in tried)),
        "tried_detail": json.dumps(tried, default=float)[:2000],
    }


def main() -> None:
    config = load_config(Path("configs/rb/inject-v1.yaml"))
    sources = pd.read_csv(config.sources.table)
    chosen = sources[sources.representative].sort_values("family_id")
    rows = Parallel(n_jobs=8)(delayed(family_report)(r, config) for _, r in chosen.iterrows())
    out = pd.DataFrame(rows)
    out.to_csv(sys.argv[1], index=False)
    for role in ("confirm", "dev"):
        sub = out[out.role == role]
        print(f"{role}: {len(sub)} famílias; com bases aceitas: {(sub.bases_accepted > 0).sum()}")
    print(out[out.bases_accepted == 0][["family_id", "openml_id", "role", "n_numeric", "n_eligible_columns", "targets_tried", "rejected_r2", "rejected_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
