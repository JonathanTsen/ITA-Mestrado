"""
preparar_dados_reais.py
=======================
Converte os datasets reais baixados para o formato esperado pelo pipeline
de classificação de mecanismos de missing data.

Formato de saída (mesmo do gerador.py):
  - Colunas: X0, X1, X2, X3, X4 (tab-separated)
  - X0 contém os valores missing
  - X1-X4 são completas (sem missing)
  - 1 arquivo .txt por dataset, salvo em Dataset/real/{MECANISMO}/

Uso:
  cd "IC - ITA 2/Scripts"
  python preparar_dados_reais.py
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "Dataset", "real")
OUTPUT_DIR = os.path.join(DATASET_DIR, "processado")

os.makedirs(os.path.join(OUTPUT_DIR, "MCAR"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "MAR"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "MNAR"), exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza cada coluna para [0, 1] (min-max scaling)."""
    for col in df.columns:
        cmin, cmax = df[col].min(), df[col].max()
        if cmax - cmin > 1e-12:
            df[col] = (df[col] - cmin) / (cmax - cmin)
        else:
            df[col] = 0.5
    return df


def select_and_rename(df: pd.DataFrame, x0_col: str, other_cols: list[str],
                      pad_to: int = 5) -> pd.DataFrame:
    """
    Seleciona colunas, renomeia para X0..XN, e preenche com colunas
    aleatórias se necessário para chegar a pad_to colunas.
    X0 = coluna com missing. X1..X4 = completas.
    """
    out = pd.DataFrame()
    out["X0"] = df[x0_col].values

    for i, col in enumerate(other_cols, start=1):
        out[f"X{i}"] = df[col].values

    # Preenche colunas restantes com ruído uniforme (se tiver menos de 5)
    rng = np.random.default_rng(42)
    n = len(out)
    current = len(out.columns)
    for i in range(current, pad_to):
        out[f"X{i}"] = rng.uniform(0, 1, n)

    # Garante que X1..X4 não tenham missing (preenche com média)
    for col in [f"X{i}" for i in range(1, pad_to)]:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(out[col].mean())

    return out


def save_dataset(df: pd.DataFrame, mechanism: str, name: str):
    """Salva no formato tab-separated com o nome padronizado."""
    fname = f"{mechanism}_{name}.txt"
    path = os.path.join(OUTPUT_DIR, mechanism, fname)
    df.to_csv(path, sep="\t", index=False)
    miss_rate = df["X0"].isna().mean() * 100
    print(f"  {fname}: {len(df)} rows, {miss_rate:.1f}% missing em X0")


# ======================================================================
# 1. MCAR — Oceanbuoys / TAO
# ======================================================================
print("\n=== MCAR: Oceanbuoys (TAO) ===")
tao_path = os.path.join(DATASET_DIR, "MCAR", "oceanbuoys_tao.csv")
tao = pd.read_csv(tao_path)

# humidity tem 93 missing (MCAR por falha de equipamento)
# Usamos humidity como X0, e sea.surface.temp, air.temp, uwind, vwind como X1-X4
# Primeiro: preencher air.temp missing (81 NaN) para que sirva como X1 completa
tao_clean = tao.copy()
tao_clean["air.temp"] = tao_clean["air.temp"].fillna(tao_clean["air.temp"].mean())
tao_clean["sea.surface.temp"] = tao_clean["sea.surface.temp"].fillna(
    tao_clean["sea.surface.temp"].mean()
)

df_mcar = select_and_rename(
    tao_clean,
    x0_col="humidity",
    other_cols=["sea.surface.temp", "air.temp", "uwind", "vwind"],
)
# Normaliza colunas completas (X1-X4) para [0,1]. X0 normaliza só os observados.
for col in ["X1", "X2", "X3", "X4"]:
    cmin, cmax = df_mcar[col].min(), df_mcar[col].max()
    if cmax - cmin > 1e-12:
        df_mcar[col] = (df_mcar[col] - cmin) / (cmax - cmin)

# Normaliza X0 mantendo NaN
x0_obs = df_mcar["X0"].dropna()
x0_min, x0_max = x0_obs.min(), x0_obs.max()
if x0_max - x0_min > 1e-12:
    df_mcar["X0"] = (df_mcar["X0"] - x0_min) / (x0_max - x0_min)

save_dataset(df_mcar, "MCAR", "oceanbuoys_humidity")

# Variante: air.temp como X0 (81 missing)
tao_clean2 = tao.copy()
tao_clean2["humidity"] = tao_clean2["humidity"].fillna(tao_clean2["humidity"].mean())
tao_clean2["sea.surface.temp"] = tao_clean2["sea.surface.temp"].fillna(
    tao_clean2["sea.surface.temp"].mean()
)

df_mcar2 = select_and_rename(
    tao_clean2,
    x0_col="air.temp",
    other_cols=["sea.surface.temp", "humidity", "uwind", "vwind"],
)
for col in ["X1", "X2", "X3", "X4"]:
    cmin, cmax = df_mcar2[col].min(), df_mcar2[col].max()
    if cmax - cmin > 1e-12:
        df_mcar2[col] = (df_mcar2[col] - cmin) / (cmax - cmin)
x0_obs2 = df_mcar2["X0"].dropna()
x0_min2, x0_max2 = x0_obs2.min(), x0_obs2.max()
if x0_max2 - x0_min2 > 1e-12:
    df_mcar2["X0"] = (df_mcar2["X0"] - x0_min2) / (x0_max2 - x0_min2)

save_dataset(df_mcar2, "MCAR", "oceanbuoys_airtemp")


# ======================================================================
# 2. MAR — Airquality
# ======================================================================
print("\n=== MAR: Airquality ===")
aq_path = os.path.join(DATASET_DIR, "MAR", "airquality.csv")
aq = pd.read_csv(aq_path)

# Ozone tem 37 missing (~24%). Missingness correlaciona com Temp e Wind (MAR)
# Solar.R tem 7 missing - preencher para usar como preditora
aq_clean = aq.copy()
aq_clean["Solar.R"] = aq_clean["Solar.R"].fillna(aq_clean["Solar.R"].mean())

df_mar = select_and_rename(
    aq_clean,
    x0_col="Ozone",
    other_cols=["Wind", "Temp", "Solar.R", "Month"],
)
for col in ["X1", "X2", "X3", "X4"]:
    cmin, cmax = df_mar[col].min(), df_mar[col].max()
    if cmax - cmin > 1e-12:
        df_mar[col] = (df_mar[col] - cmin) / (cmax - cmin)
x0_obs_mar = df_mar["X0"].dropna()
x0_min_m, x0_max_m = x0_obs_mar.min(), x0_obs_mar.max()
if x0_max_m - x0_min_m > 1e-12:
    df_mar["X0"] = (df_mar["X0"] - x0_min_m) / (x0_max_m - x0_min_m)

save_dataset(df_mar, "MAR", "airquality_ozone")


# MAR — Mammographic Mass
print("\n=== MAR: Mammographic Mass ===")
mammo_path = os.path.join(DATASET_DIR, "MAR", "mammographic_mass_raw.csv")
mammo = pd.read_csv(
    mammo_path,
    header=None,
    names=["BIRADS", "Age", "Shape", "Margin", "Density", "Severity"],
    na_values="?",
)
# Density tem 76 missing. Missingness depende de BIRADS e Age (MAR)
mammo_clean = mammo.dropna(subset=["BIRADS", "Age", "Shape", "Margin"]).copy()
mammo_clean["BIRADS"] = pd.to_numeric(mammo_clean["BIRADS"], errors="coerce")
mammo_clean["Age"] = pd.to_numeric(mammo_clean["Age"], errors="coerce")
mammo_clean["Shape"] = pd.to_numeric(mammo_clean["Shape"], errors="coerce")
mammo_clean["Margin"] = pd.to_numeric(mammo_clean["Margin"], errors="coerce")
mammo_clean["Density"] = pd.to_numeric(mammo_clean["Density"], errors="coerce")
mammo_clean = mammo_clean.dropna(subset=["BIRADS", "Age", "Shape", "Margin"])

df_mar2 = select_and_rename(
    mammo_clean,
    x0_col="Density",
    other_cols=["BIRADS", "Age", "Shape", "Margin"],
)
for col in ["X1", "X2", "X3", "X4"]:
    cmin, cmax = df_mar2[col].min(), df_mar2[col].max()
    if cmax - cmin > 1e-12:
        df_mar2[col] = (df_mar2[col] - cmin) / (cmax - cmin)
x0_obs_m2 = df_mar2["X0"].dropna()
x0_min_m2, x0_max_m2 = x0_obs_m2.min(), x0_obs_m2.max()
if x0_max_m2 - x0_min_m2 > 1e-12:
    df_mar2["X0"] = (df_mar2["X0"] - x0_min_m2) / (x0_max_m2 - x0_min_m2)

save_dataset(df_mar2, "MAR", "mammographic_density")


# ======================================================================
# 3. MNAR — Pima Indians Diabetes (Insulin)
# ======================================================================
print("\n=== MNAR: Pima Diabetes (Insulin) ===")
pima_path = os.path.join(DATASET_DIR, "MNAR", "pima_diabetes_raw.csv")
pima = pd.read_csv(
    pima_path,
    header=None,
    names=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome",
    ],
)
# Insulin: 374 zeros (48.7%) = missing (MNAR: não medido quando médico
# não suspeita diabetes, que depende do próprio nível de insulina)
pima_proc = pima.copy()
pima_proc["Insulin"] = pima_proc["Insulin"].replace(0, np.nan)
# Glucose zeros também são missing biológico - preencher
pima_proc["Glucose"] = pima_proc["Glucose"].replace(0, np.nan)
pima_proc["Glucose"] = pima_proc["Glucose"].fillna(pima_proc["Glucose"].mean())
pima_proc["BloodPressure"] = pima_proc["BloodPressure"].replace(0, np.nan)
pima_proc["BloodPressure"] = pima_proc["BloodPressure"].fillna(
    pima_proc["BloodPressure"].mean()
)
pima_proc["BMI"] = pima_proc["BMI"].replace(0, np.nan)
pima_proc["BMI"] = pima_proc["BMI"].fillna(pima_proc["BMI"].mean())

df_mnar = select_and_rename(
    pima_proc,
    x0_col="Insulin",
    other_cols=["Glucose", "BloodPressure", "BMI", "Age"],
)
for col in ["X1", "X2", "X3", "X4"]:
    cmin, cmax = df_mnar[col].min(), df_mnar[col].max()
    if cmax - cmin > 1e-12:
        df_mnar[col] = (df_mnar[col] - cmin) / (cmax - cmin)
x0_obs_mn = df_mnar["X0"].dropna()
x0_min_mn, x0_max_mn = x0_obs_mn.min(), x0_obs_mn.max()
if x0_max_mn - x0_min_mn > 1e-12:
    df_mnar["X0"] = (df_mnar["X0"] - x0_min_mn) / (x0_max_mn - x0_min_mn)

save_dataset(df_mnar, "MNAR", "pima_insulin")


# MNAR — Mroz Wages
print("\n=== MNAR: Mroz Wages ===")
mroz_path = os.path.join(DATASET_DIR, "MNAR", "mroz_wages.csv")
mroz = pd.read_csv(mroz_path)

# lwg (log wage) = NaN quando lfp="no" (mulher não trabalha)
# A decisão de não trabalhar depende do próprio salário potencial → MNAR
mroz_proc = mroz.copy()
mroz_proc.loc[mroz_proc["lfp"] == "no", "lwg"] = np.nan

# Converter variáveis categóricas
mroz_proc["wc_num"] = (mroz_proc["wc"] == "yes").astype(float)
mroz_proc["hc_num"] = (mroz_proc["hc"] == "yes").astype(float)

df_mnar2 = select_and_rename(
    mroz_proc,
    x0_col="lwg",
    other_cols=["age", "inc", "k5", "wc_num"],
)
for col in ["X1", "X2", "X3", "X4"]:
    cmin, cmax = df_mnar2[col].min(), df_mnar2[col].max()
    if cmax - cmin > 1e-12:
        df_mnar2[col] = (df_mnar2[col] - cmin) / (cmax - cmin)
x0_obs_mn2 = df_mnar2["X0"].dropna()
x0_min_mn2, x0_max_mn2 = x0_obs_mn2.min(), x0_obs_mn2.max()
if x0_max_mn2 - x0_min_mn2 > 1e-12:
    df_mnar2["X0"] = (df_mnar2["X0"] - x0_min_mn2) / (x0_max_mn2 - x0_min_mn2)

save_dataset(df_mnar2, "MNAR", "mroz_wages")


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
        complete_cols = sum(1 for c in df.columns if c != "X0" and df[c].isna().sum() == 0)
        print(f"  {f}: {df.shape[0]} rows x {df.shape[1]} cols, "
              f"X0 missing={miss:.1f}%, {complete_cols} colunas completas")
