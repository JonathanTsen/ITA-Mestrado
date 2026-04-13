"""
expandir_dados_reais.py — Busca, baixa e processa datasets reais para classificação
de mecanismos de missing data.

Fontes: OpenML (via sklearn), URLs diretas.
Resultado: datasets processados em Dataset/real_data/processado/{MCAR,MAR,MNAR}/

Uso:
    cd "IC - ITA 2/Scripts"
    uv run python expandir_dados_reais.py
"""

import os
import ssl
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Contorna SSL em macOS
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "Dataset", "real_data")
OUTPUT_DIR = os.path.join(DATASET_DIR, "processado")

RNG = np.random.default_rng(42)
TARGET_MISSING_RATE = 0.10

for mech in ["MCAR", "MAR", "MNAR"]:
    os.makedirs(os.path.join(OUTPUT_DIR, mech), exist_ok=True)


# ======================================================
# UTILIDADES
# ======================================================

def impute_with_sample(series: pd.Series) -> pd.Series:
    observed = series.dropna().values
    mask = series.isna()
    if mask.sum() == 0 or len(observed) == 0:
        return series
    series = series.copy()
    series[mask] = RNG.choice(observed, size=mask.sum())
    return series


def cap_missing_rate(df: pd.DataFrame, target: float = TARGET_MISSING_RATE) -> pd.DataFrame:
    current = df["X0"].isna().mean()
    if current <= target:
        return df
    df = df.copy()
    nan_idx = df.index[df["X0"].isna()].tolist()
    n_keep = max(1, int(target * len(df)))
    n_impute = len(nan_idx) - n_keep
    if n_impute <= 0:
        return df
    to_impute = RNG.choice(nan_idx, size=n_impute, replace=False)
    observed = df["X0"].dropna().values
    df.loc[to_impute, "X0"] = RNG.choice(observed, size=n_impute)
    print(f"    Cap: {current*100:.1f}% -> {df['X0'].isna().mean()*100:.1f}%")
    return df


def select_columns(df_raw: pd.DataFrame, x0_col: str,
                   aux_cols: list[str] | None = None) -> pd.DataFrame:
    out = pd.DataFrame()
    out["X0"] = pd.to_numeric(df_raw[x0_col], errors="coerce").values

    if aux_cols:
        for i, col in enumerate(aux_cols[:4], 1):
            out[f"X{i}"] = pd.to_numeric(df_raw[col], errors="coerce").values
    else:
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != x0_col]
        num_cols.sort(key=lambda c: df_raw[c].isna().mean())
        for i, col in enumerate(num_cols[:4], 1):
            out[f"X{i}"] = df_raw[col].values

    for i in range(len(out.columns), 5):
        out[f"X{i}"] = RNG.uniform(0, 1, len(out))

    return out


def normalize_and_save(df: pd.DataFrame, mechanism: str, name: str) -> str | None:
    if df["X0"].isna().sum() == 0:
        print(f"    SKIP: sem missing em X0")
        return None

    df = cap_missing_rate(df)

    for col in df.columns:
        obs = df[col].dropna()
        if len(obs) == 0:
            continue
        cmin, cmax = obs.min(), obs.max()
        if cmax - cmin > 1e-12:
            df[col] = (df[col] - cmin) / (cmax - cmin)
        else:
            df.loc[df[col].notna(), col] = 0.5

    for c in ["X1", "X2", "X3", "X4"]:
        if c in df.columns and df[c].isna().any():
            df[c] = impute_with_sample(df[c])

    fname = f"{mechanism}_{name}.txt"
    path = os.path.join(OUTPUT_DIR, mechanism, fname)
    df.to_csv(path, sep="\t", index=False)
    miss = df["X0"].isna().mean() * 100
    print(f"  -> {fname}: {len(df)} rows, {miss:.1f}% missing")
    return path


def process_openml(openml_name: str, version: int, x0_col: str,
                   mechanism: str, dataset_name: str,
                   aux_cols: list[str] | None = None) -> bool:
    try:
        print(f"\n  Baixando {openml_name} v{version} (OpenML)...")
        data = fetch_openml(name=openml_name, version=version, as_frame=True, parser="auto")
        df_raw = data.frame
        print(f"    Shape: {df_raw.shape}")

        df = select_columns(df_raw, x0_col, aux_cols)
        result = normalize_and_save(df, mechanism, dataset_name)
        return result is not None
    except Exception as e:
        print(f"    ERRO: {e}")
        return False


def process_url(url: str, x0_col: str, mechanism: str, dataset_name: str,
                aux_cols: list[str] | None = None, **kwargs) -> bool:
    try:
        print(f"\n  Baixando {dataset_name} (URL)...")
        df_raw = pd.read_csv(url, **kwargs)
        print(f"    Shape: {df_raw.shape}")
        df = select_columns(df_raw, x0_col, aux_cols)
        result = normalize_and_save(df, mechanism, dataset_name)
        return result is not None
    except Exception as e:
        print(f"    ERRO: {e}")
        return False


# ======================================================
# DATASETS
# ======================================================

print("=" * 60)
print("EXPANDINDO DATASETS REAIS")
print("=" * 60)

success_count = 0

# ==============================
# MCAR — Missing administrativo/aleatório
# ==============================
# Já existem: breastcancer_barenuclei, oceanbuoys_humidity, oceanbuoys_airtemp

print(f"\n{'='*50}")
print("MCAR — Missing por falha administrativa/logística")
print(f"{'='*50}")

# cylinder-bands: dados de manufatura, missing por falha de sensor
if process_openml("cylinder-bands", 2, "blade_pressure", "MCAR", "cylinderbands_bladepressure",
                  aux_cols=["press_speed", "ink_temperature", "viscosity", "roughness"]):
    success_count += 1

# cylinder-bands: ESA_Voltage
if process_openml("cylinder-bands", 2, "ESA_Voltage", "MCAR", "cylinderbands_esavoltage",
                  aux_cols=["press_speed", "ink_temperature", "viscosity", "roughness"]):
    success_count += 1

# hypothyroid: T4U (exame tireoidiano não solicitado - MCAR plausível para exames de rotina)
if process_openml("hypothyroid", 1, "T4U", "MCAR", "hypothyroid_t4u",
                  aux_cols=["age", "TSH", "TT4", "FTI"]):
    success_count += 1

# autoMpg: horsepower (dados não registrados aleatoriamente)
if process_openml("autoMpg", 1, "horsepower", "MCAR", "autompg_horsepower",
                  aux_cols=["displacement", "weight", "acceleration", "cylinders"]):
    success_count += 1

# ==============================
# MAR — Missing depende de variáveis observadas
# ==============================
# Já existem: airquality_ozone, mammographic_density, titanic_age

print(f"\n{'='*50}")
print("MAR — Missing depende de variáveis observáveis")
print(f"{'='*50}")

# sick (thyroid): T3 — teste solicitado com base em sintomas (MAR)
if process_openml("sick", 1, "T3", "MAR", "sick_t3",
                  aux_cols=["age", "TSH", "TT4", "FTI"]):
    success_count += 1

# sick: TSH — idem
if process_openml("sick", 1, "TSH", "MAR", "sick_tsh",
                  aux_cols=["age", "T3", "TT4", "FTI"]):
    success_count += 1

# chronic-kidney-disease: hemo — exame depende da severidade do caso
if process_openml("chronic-kidney-disease", 1, "hemo", "MAR", "kidney_hemo",
                  aux_cols=["bp", "age", "bgr", "bu"]):
    success_count += 1

# heart-h: chol — colesterol não medido depende de fatores clínicos
if process_openml("heart-h", 1, "chol", "MAR", "hearth_chol",
                  aux_cols=["age", "trestbps", "thalach", "oldpeak"]):
    success_count += 1

# Titanic (URL direta, versão completa)
if process_url(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    x0_col="Age", mechanism="MAR", dataset_name="titanic_age_v2",
    aux_cols=["Pclass", "SibSp", "Parch", "Fare"],
):
    success_count += 1

# colic: respiratory_rate — exame depende da severidade
if process_openml("colic", 1, "respiratory_rate", "MAR", "colic_resprate",
                  aux_cols=["pulse", "rectal_temperature", "packed_cell_volume", "total_protein"]):
    success_count += 1

# ==============================
# MNAR — Missing depende do próprio valor faltante
# ==============================
# Já existem: pima_insulin, mroz_wages, adult_capitalgain

print(f"\n{'='*50}")
print("MNAR — Missing depende do valor faltante")
print(f"{'='*50}")

# chronic-kidney-disease: pot (potássio) — valores extremos não reportados
if process_openml("chronic-kidney-disease", 1, "pot", "MNAR", "kidney_pot",
                  aux_cols=["bp", "age", "bgr", "bu"]):
    success_count += 1

# chronic-kidney-disease: sod (sódio) — idem
if process_openml("chronic-kidney-disease", 1, "sod", "MNAR", "kidney_sod",
                  aux_cols=["bp", "age", "bgr", "bu"]):
    success_count += 1

# colic: nasogastric_reflux_PH — valor difícil de medir em extremos
if process_openml("colic", 1, "nasogastric_reflux_PH", "MNAR", "colic_refluxph",
                  aux_cols=["pulse", "rectal_temperature", "packed_cell_volume", "total_protein"]):
    success_count += 1

# cylinder-bands: varnish_pct — qualidade-dependente
if process_openml("cylinder-bands", 2, "varnish_pct", "MNAR", "cylinderbands_varnishpct",
                  aux_cols=["press_speed", "ink_temperature", "viscosity", "roughness"]):
    success_count += 1

# hypothyroid: TBG — exame solicitado apenas quando resultado esperado é anormal (MNAR)
# Skippar: 100% missing, não útil

# sick: TBG — idem
# Skippar: 100% missing

# ======================================================
# RESUMO
# ======================================================
print(f"\n{'='*60}")
print(f"EXPANSAO CONCLUIDA: {success_count} novos datasets adicionados")
print(f"{'='*60}")

for mech in ["MCAR", "MAR", "MNAR"]:
    mech_dir = os.path.join(OUTPUT_DIR, mech)
    files = sorted([f for f in os.listdir(mech_dir) if f.endswith(".txt")])
    print(f"\n{mech}/ ({len(files)} arquivos)")
    for f in files:
        df = pd.read_csv(os.path.join(mech_dir, f), sep="\t")
        miss = df["X0"].isna().mean() * 100
        print(f"  {f}: {len(df)} rows, {miss:.1f}% missing")
