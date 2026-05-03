"""
preparar_dados_reais.py
=======================
Converte os datasets reais baixados para o formato esperado pelo pipeline
de classificação de mecanismos de missing data.

Formato de saída (mesmo do gerador.py):
  - Colunas: X0, X1, X2, X3, X4 (tab-separated)
  - X0 contém os valores missing
  - X1-X4 são completas (sem missing)
  - 1 arquivo .txt por dataset

Melhorias v2:
  - Imputação de preditoras por amostragem da distribuição observada
    (preserva variância, ao contrário de fillna(mean))
  - Cap de taxa de missing para o range do treino sintético (≤10%)
  - Jitter gaussiano em variáveis ordinais (Mammographic) para
    simular continuidade compatível com os dados sintéticos

Uso:
  cd "IC - ITA 2/Scripts"
  python preparar_dados_reais.py
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "Dataset", "real_data")
OUTPUT_DIR = os.path.join(DATASET_DIR, "processado")

TARGET_MISSING_RATE = 0.10
JITTER_SCALE = 0.02

RNG = np.random.default_rng(42)

for mech in ["MCAR", "MAR", "MNAR"]:
    os.makedirs(os.path.join(OUTPUT_DIR, mech), exist_ok=True)


# ======================================================================
# Funções auxiliares
# ======================================================================


def impute_with_sample(series: pd.Series) -> pd.Series:
    """Replace NaN with random samples from observed values (preserves variance)."""
    observed = series.dropna().values
    mask = series.isna()
    n_missing = mask.sum()
    if n_missing == 0 or len(observed) == 0:
        return series
    series = series.copy()
    series[mask] = RNG.choice(observed, size=n_missing)
    return series


def cap_missing_rate(df: pd.DataFrame, target: float = TARGET_MISSING_RATE) -> pd.DataFrame:
    """Reduce X0 missing rate to target by imputing excess NaN with observed samples.

    A random subset of NaN positions is kept, preserving the original mechanism
    pattern (MCAR/MAR/MNAR) in the remaining missing values.
    """
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

    new_rate = df["X0"].isna().mean()
    print(f"    Cap missing: {current*100:.1f}% → {new_rate*100:.1f}%")
    return df


def add_jitter(df: pd.DataFrame, columns: list[str], scale: float = JITTER_SCALE) -> pd.DataFrame:
    """Add small Gaussian noise to ordinal columns to simulate continuity."""
    df = df.copy()
    for col in columns:
        mask = df[col].notna()
        n = mask.sum()
        noise = RNG.normal(0, scale, size=n)
        df.loc[mask, col] = np.clip(df.loc[mask, col].values + noise, 0.0, 1.0)
    return df


def normalize_col(df: pd.DataFrame, col: str, observed_only: bool = False):
    """Min-max normalize a single column in-place."""
    vals = df[col].dropna() if observed_only else df[col]
    cmin, cmax = vals.min(), vals.max()
    if cmax - cmin > 1e-12:
        df[col] = (df[col] - cmin) / (cmax - cmin)
    else:
        df.loc[df[col].notna(), col] = 0.5


def select_and_rename(df: pd.DataFrame, x0_col: str, other_cols: list[str], pad_to: int = 5) -> pd.DataFrame:
    """Select columns, rename to X0..X4, impute predictor NaN with samples."""
    out = pd.DataFrame()
    out["X0"] = df[x0_col].values

    for i, col in enumerate(other_cols, start=1):
        out[f"X{i}"] = df[col].values

    n = len(out)
    current = len(out.columns)
    for i in range(current, pad_to):
        out[f"X{i}"] = RNG.uniform(0, 1, n)

    for col in [f"X{i}" for i in range(1, pad_to)]:
        if col in out.columns and out[col].isna().any():
            n_missing = out[col].isna().sum()
            print(f"    Imputing {col}: {n_missing} NaN → sample from observed")
            out[col] = impute_with_sample(out[col])

    return out


def process_and_save(df: pd.DataFrame, mechanism: str, name: str, jitter_cols: list[str] | None = None):
    """Cap missing rate → normalize → jitter (optional) → save."""
    df = cap_missing_rate(df)

    normalize_col(df, "X0", observed_only=True)
    for col in ["X1", "X2", "X3", "X4"]:
        normalize_col(df, col)

    if jitter_cols:
        df = add_jitter(df, jitter_cols)

    fname = f"{mechanism}_{name}.txt"
    path = os.path.join(OUTPUT_DIR, mechanism, fname)
    df.to_csv(path, sep="\t", index=False)
    miss_rate = df["X0"].isna().mean() * 100
    print(f"  ✓ {fname}: {len(df)} rows, {miss_rate:.1f}% missing em X0")


# ======================================================================
# 1. MCAR — Oceanbuoys / TAO
# ======================================================================
print("\n=== MCAR: Oceanbuoys (TAO) ===")
tao_path = os.path.join(DATASET_DIR, "MCAR", "oceanbuoys_tao.csv")
tao = pd.read_csv(tao_path)

# Variante 1: humidity como X0 (93 NaN, 12.6%)
print("  Variante: humidity")
df_mcar = select_and_rename(
    tao,
    x0_col="humidity",
    other_cols=["sea.surface.temp", "air.temp", "uwind", "vwind"],
)
process_and_save(df_mcar, "MCAR", "oceanbuoys_humidity")

# Variante 2: air.temp como X0 (81 NaN, 11.0%)
print("  Variante: air.temp")
df_mcar2 = select_and_rename(
    tao,
    x0_col="air.temp",
    other_cols=["sea.surface.temp", "humidity", "uwind", "vwind"],
)
process_and_save(df_mcar2, "MCAR", "oceanbuoys_airtemp")


# ======================================================================
# 2. MAR — Airquality
# ======================================================================
print("\n=== MAR: Airquality ===")
aq_path = os.path.join(DATASET_DIR, "MAR", "airquality.csv")
aq = pd.read_csv(aq_path)

# Ozone: 37 NaN (24.2%), missingness correlaciona com Wind e Temp (MAR)
df_mar = select_and_rename(
    aq,
    x0_col="Ozone",
    other_cols=["Wind", "Temp", "Solar.R", "Month"],
)
process_and_save(df_mar, "MAR", "airquality_ozone")


# ======================================================================
# MAR — Mammographic Mass
# ======================================================================
print("\n=== MAR: Mammographic Mass ===")
mammo_path = os.path.join(DATASET_DIR, "MAR", "mammographic_mass_raw.csv")
mammo = pd.read_csv(
    mammo_path,
    header=None,
    names=["BIRADS", "Age", "Shape", "Margin", "Density", "Severity"],
    na_values="?",
)
for col in ["BIRADS", "Age", "Shape", "Margin", "Density"]:
    mammo[col] = pd.to_numeric(mammo[col], errors="coerce")
mammo_clean = mammo.dropna(subset=["BIRADS", "Age", "Shape", "Margin"]).copy()

# Density: 56 NaN (6.3%), missingness depende de BIRADS e Age (MAR)
df_mar2 = select_and_rename(
    mammo_clean,
    x0_col="Density",
    other_cols=["BIRADS", "Age", "Shape", "Margin"],
)
# Jitter em variáveis ordinais (X0=Density, X1=BIRADS, X3=Shape, X4=Margin)
# X2=Age é contínua, não precisa de jitter
process_and_save(df_mar2, "MAR", "mammographic_density", jitter_cols=["X0", "X1", "X3", "X4"])


# ======================================================================
# 3. MNAR — Pima Indians Diabetes (Insulin)
# ======================================================================
print("\n=== MNAR: Pima Diabetes (Insulin) ===")
pima_path = os.path.join(DATASET_DIR, "MNAR", "pima_diabetes_raw.csv")
pima = pd.read_csv(
    pima_path,
    header=None,
    names=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigree",
        "Age",
        "Outcome",
    ],
)
# Zeros biologicamente impossíveis → NaN
pima_proc = pima.copy()
pima_proc["Insulin"] = pima_proc["Insulin"].replace(0, np.nan)
pima_proc["Glucose"] = pima_proc["Glucose"].replace(0, np.nan)
pima_proc["BloodPressure"] = pima_proc["BloodPressure"].replace(0, np.nan)
pima_proc["BMI"] = pima_proc["BMI"].replace(0, np.nan)

# Insulin: 374 NaN (48.7%) → será capped para ~10%
df_mnar = select_and_rename(
    pima_proc,
    x0_col="Insulin",
    other_cols=["Glucose", "BloodPressure", "BMI", "Age"],
)
process_and_save(df_mnar, "MNAR", "pima_insulin")


# ======================================================================
# MNAR — Mroz Wages
# ======================================================================
print("\n=== MNAR: Mroz Wages ===")
mroz_path = os.path.join(DATASET_DIR, "MNAR", "mroz_wages.csv")
mroz = pd.read_csv(mroz_path)

# lwg = NaN para mulheres fora da força de trabalho (lfp="no")
# Valores no CSV para lfp="no" são salários estimados (Heckman), não observados
mroz_proc = mroz.copy()
mroz_proc.loc[mroz_proc["lfp"] == "no", "lwg"] = np.nan
mroz_proc["wc_num"] = (mroz_proc["wc"] == "yes").astype(float)

# lwg: 325 NaN (43.2%) → será capped para ~10%
df_mnar2 = select_and_rename(
    mroz_proc,
    x0_col="lwg",
    other_cols=["age", "inc", "k5", "wc_num"],
)
process_and_save(df_mnar2, "MNAR", "mroz_wages")


# ======================================================================
# Resumo
# ======================================================================
print("\n" + "=" * 60)
print("RESUMO DOS DATASETS PROCESSADOS")
print("=" * 60)

for mech in ["MCAR", "MAR", "MNAR"]:
    mech_dir = os.path.join(OUTPUT_DIR, mech)
    files = sorted(os.listdir(mech_dir))
    print(f"\n{mech}/")
    for f in files:
        df = pd.read_csv(os.path.join(mech_dir, f), sep="\t")
        miss = df["X0"].isna().mean() * 100
        x0_unique = df["X0"].dropna().nunique()
        complete = sum(1 for c in df.columns if c != "X0" and df[c].isna().sum() == 0)
        print(
            f"  {f}: {df.shape[0]} rows, X0 missing={miss:.1f}%, "
            f"X0 unique={x0_unique}, {complete} preditoras completas"
        )
