"""D11: varre as colunas numéricas dos representantes (passo 31) atrás de massa num valor de fronteira (zero ou mínimo).

Uso: uv run --frozen python scripts/revisao_banca/decisoes/d11_zeros_sentinela.py (precisa dos CSV do MechDetect em
/tmp/rb_mechdetect_run/mech-detect-datasets).
"""

from pathlib import Path

import pandas as pd

from missdetect.rb.inject import column_is_eligible
from missdetect.rb.sources import id_like, read_local_csv

src = pd.read_csv("data_rb/sources_local.csv")
src = src[src.representative]
csv_dir = Path("/tmp/rb_mechdetect_run/mech-detect-datasets")
rows = []
for r in src.itertuples():
    f = read_local_csv(csv_dir / f"{r.openml_id}.csv")
    f = f.drop(columns=id_like(f))
    for col in str(r.numeric_columns).split(";"):
        v = f[col].astype(float)
        vc = v.value_counts(normalize=True)
        mode_val, mode_frac = vc.index[0], vc.iloc[0]
        rows.append(
            {
                "family": r.family_id,
                "role": r.role,
                "openml": r.openml_id,
                "col": col,
                "eligible": column_is_eligible(v),
                "mode_frac": mode_frac,
                "mode_is_zero": mode_val == 0,
                "mode_is_min": mode_val == v.min(),
                "zero_frac": float((v == 0).mean()),
            }
        )
d = pd.DataFrame(rows)
e = d[d.eligible]
print("colunas numéricas:", len(d), "| elegíveis (≥20 distintos, moda ≤ 0,5):", len(e))
for thr in (0.05, 0.10, 0.20, 0.30):
    flag = e[(e.mode_frac >= thr) & (e.mode_is_zero | e.mode_is_min)]
    print(f"elegíveis com massa na fronteira ≥ {thr:.0%}: {len(flag)} colunas em {flag.family.nunique()} famílias")
flag = e[(e.mode_frac >= 0.10) & (e.mode_is_zero | e.mode_is_min)]
print(
    flag.sort_values("mode_frac", ascending=False)
    .head(25)[["family", "role", "openml", "col", "mode_frac", "mode_is_zero"]]
    .to_string(index=False)
)
