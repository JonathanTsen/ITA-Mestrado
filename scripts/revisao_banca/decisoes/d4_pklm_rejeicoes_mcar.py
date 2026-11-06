"""D4: onde o PKLM rejeitou instâncias MCAR do piloto e se isso se associa a empates, n ou k.

Uso: uv run --frozen python -I scripts/revisao_banca/decisoes/d4_pklm_rejeicoes_mcar.py <run do piloto>
"""

import sys

import pandas as pd

R = sys.argv[1]
p = pd.read_parquet(f"{R}/pklm.parquet")
o = pd.read_parquet(f"{R}/oracle.parquet")
i = pd.read_parquet(f"{R}/instances.parquet")
c = pd.read_parquet(f"{R}/contexts.parquet")
b = pd.read_parquet(f"{R}/baselines.parquet")
m = (
    p.merge(o[["sample_id", "mechanism"]], on="sample_id")
    .merge(i, on="sample_id")
    .merge(c[["context_id", "n", "k", "rate"]], on="context_id")
    .merge(b[["sample_id", "little_p"]], on="sample_id")
)
mc = m[m.mechanism == "MCAR"].copy()
rows = []
for r in mc.itertuples():
    obs = pd.read_parquet(f"{R}/{r.observed_path}")
    z = obs.drop(columns="y")
    uniq = z.nunique() / len(z)
    rows.append(
        {
            "sample_id": r.sample_id,
            "min_unique_frac_z": float(uniq.min()),
            "max_mode_frac_z": float(max(z[c].value_counts(normalize=True).iloc[0] for c in z)),
        }
    )
mc = mc.merge(pd.DataFrame(rows), on="sample_id")
mc["rej_pklm"] = mc.pklm_p < 0.05
mc["rej_little"] = mc.little_p < 0.05
print("MCAR:", len(mc), "PKLM rejeita:", int(mc.rej_pklm.sum()), "Little rejeita:", int(mc.rej_little.sum()))
print(mc.groupby("family_id")[["rej_pklm", "rej_little"]].agg(["sum", "size"]))
print(mc.groupby("rej_pklm")[["n", "k", "rate", "min_unique_frac_z", "max_mode_frac_z"]].mean().round(3))
print("p-valores PKLM (MCAR) quantis:", mc.pklm_p.quantile([0.05, 0.1, 0.25, 0.5]).round(3).to_dict())
print("valores distintos de p:", mc.pklm_p.nunique(), "mínimo p:", mc.pklm_p.min())
