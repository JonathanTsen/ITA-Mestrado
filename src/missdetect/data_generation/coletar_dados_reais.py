"""
Coleta datasets reais adicionais com mecanismos de missing data conhecidos.

Baixa datasets do UCI ML Repository e OpenML, processa para o formato
padrão (X0-X4, tab-separated) e salva em Dataset/real_data/processado/.

Datasets adicionados:
  MCAR:
    - Wisconsin Breast Cancer: Bare Nuclei (16 missing, erros de registro)
    - Heart Disease Cleveland: ca (4 missing, equipamento)
  MAR:
    - Titanic: Age depende de Pclass (MAR clássico)
    - Heart Disease Cleveland: thal depende de outros atributos
  MNAR:
    - Chronic Kidney Disease: hemoglobin depende da gravidade
    - Adult Income: capital-gain missing correlaciona com o próprio valor

Uso:
    cd "IC - ITA 2/Scripts"
    python coletar_dados_reais.py
"""
import os
import io
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Fix SSL para macOS (certificados não instalados)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "Dataset", "real_data")
OUTPUT_DIR = os.path.join(DATASET_DIR, "processado")
TARGET_MISSING_RATE = 0.10
JITTER_SCALE = 0.02

RNG = np.random.default_rng(42)

for mech in ["MCAR", "MAR", "MNAR"]:
    os.makedirs(os.path.join(OUTPUT_DIR, mech), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, mech), exist_ok=True)


# ======================================================================
# Funções auxiliares (mesmas do preparar_dados_reais.py)
# ======================================================================
def impute_with_sample(series: pd.Series) -> pd.Series:
    observed = series.dropna().values
    mask = series.isna()
    n_missing = mask.sum()
    if n_missing == 0 or len(observed) == 0:
        return series
    series = series.copy()
    series[mask] = RNG.choice(observed, size=n_missing)
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
    new_rate = df["X0"].isna().mean()
    print(f"    Cap missing: {current*100:.1f}% -> {new_rate*100:.1f}%")
    return df


def normalize_col(df: pd.DataFrame, col: str, observed_only: bool = False):
    vals = df[col].dropna() if observed_only else df[col]
    cmin, cmax = vals.min(), vals.max()
    if cmax - cmin > 1e-12:
        df[col] = (df[col] - cmin) / (cmax - cmin)
    else:
        df.loc[df[col].notna(), col] = 0.5


def select_and_rename(df: pd.DataFrame, x0_col: str, other_cols: list[str],
                      pad_to: int = 5) -> pd.DataFrame:
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
            out[col] = impute_with_sample(out[col])
    return out


def process_and_save(df: pd.DataFrame, mechanism: str, name: str,
                     jitter_cols: list[str] | None = None):
    df = cap_missing_rate(df)
    normalize_col(df, "X0", observed_only=True)
    for col in ["X1", "X2", "X3", "X4"]:
        normalize_col(df, col)
    if jitter_cols:
        from preparar_dados_reais import add_jitter
        df = add_jitter(df, jitter_cols)
    fname = f"{mechanism}_{name}.txt"
    path = os.path.join(OUTPUT_DIR, mechanism, fname)
    df.to_csv(path, sep="\t", index=False)
    miss_rate = df["X0"].isna().mean() * 100
    print(f"  -> {fname}: {len(df)} rows, {miss_rate:.1f}% missing em X0")


def _ssl_context():
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_csv(url: str) -> pd.DataFrame:
    """Baixa CSV de uma URL."""
    import urllib.request
    print(f"    Baixando: {url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(data))


def download_text(url: str) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ======================================================================
# 1. MCAR: Wisconsin Breast Cancer (via raw GitHub mirror)
# ======================================================================
print("\n=== MCAR: Wisconsin Breast Cancer ===")
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.data"
    text = download_text(url)
    names = ["id", "clump_thickness", "cell_size", "cell_shape", "marginal_adhesion",
             "epithelial_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli",
             "mitoses", "class"]
    wbc = pd.read_csv(io.StringIO(text), header=None, names=names, na_values="?")

    df_mcar1 = select_and_rename(
        wbc, x0_col="bare_nuclei",
        other_cols=["clump_thickness", "cell_size", "cell_shape", "marginal_adhesion"],
    )
    process_and_save(df_mcar1, "MCAR", "breastcancer_barenuclei")
    print("  OK!")
except Exception as e:
    print(f"  ERRO: {e}")

# ======================================================================
# 2. MCAR: Heart Disease Cleveland (via raw GitHub mirror)
# ======================================================================
print("\n=== MCAR: Heart Disease Cleveland (ca) ===")
heart = None
try:
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv"
    text = download_text(url)
    heart = pd.read_csv(io.StringIO(text))
    # Se não tiver 'ca', tenta formato UCI clássico
    if "ca" not in heart.columns:
        url2 = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/heart.csv"
        text2 = download_text(url2)
        heart = pd.read_csv(io.StringIO(text2))

    df_mcar2 = select_and_rename(
        heart, x0_col="ca",
        other_cols=["age", "trestbps", "chol", "thalach"],
    )
    process_and_save(df_mcar2, "MCAR", "heartdisease_ca",
                     jitter_cols=["X0"])
    print("  OK!")
except Exception as e:
    print(f"  ERRO: {e}")

# ======================================================================
# 3. MAR: Titanic (Age depende de Pclass)
# ======================================================================
print("\n=== MAR: Titanic (Age) ===")
try:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    titanic = download_csv(url)

    # Age: 177 NaN (19.9%) - MAR: 3rd class passageiros tem mais missing
    df_mar1 = select_and_rename(
        titanic, x0_col="Age",
        other_cols=["Pclass", "SibSp", "Parch", "Fare"],
    )
    process_and_save(df_mar1, "MAR", "titanic_age")
    print("  OK!")
except Exception as e:
    print(f"  ERRO: {e}")

# ======================================================================
# 4. MAR: Heart Disease Cleveland (thal depende de restecg e outros)
# ======================================================================
print("\n=== MAR: Heart Disease Cleveland (thal) ===")
try:
    if heart is not None and "thal" in heart.columns:
        df_mar2 = select_and_rename(
            heart, x0_col="thal",
            other_cols=["age", "cp", "thalach", "oldpeak"],
        )
        process_and_save(df_mar2, "MAR", "heartdisease_thal",
                         jitter_cols=["X0"])
        print("  OK!")
    else:
        print("  SKIP: heart dataset não carregado ou sem coluna 'thal'")
except Exception as e:
    print(f"  ERRO: {e}")

# ======================================================================
# 5. MNAR: Chronic Kidney Disease (hemoglobin) via GitHub mirror
# ======================================================================
print("\n=== MNAR: Chronic Kidney Disease (hemoglobin) ===")
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/chronic_kidney_disease.csv"
    text = download_text(url)
    ckd = pd.read_csv(io.StringIO(text), na_values="?")
    for col in ckd.columns:
        ckd[col] = pd.to_numeric(ckd[col], errors="coerce")

    # Tenta encontrar hemoglobin
    hemo_col = None
    for c in ckd.columns:
        if "hemo" in c.lower() or c == "hemo":
            hemo_col = c
            break

    if hemo_col is None:
        # Tenta pela posição (coluna 10 no dataset original)
        cols = list(ckd.columns)
        print(f"  Colunas encontradas: {cols[:15]}...")
        # Procura coluna com NaN que pode ser hemoglobin
        for c in cols:
            if ckd[c].isna().sum() > 10 and ckd[c].dropna().mean() > 5:
                hemo_col = c
                print(f"  Usando coluna '{c}' como hemoglobin proxy")
                break

    if hemo_col:
        # Preditores: primeiras colunas numéricas sem muitos NaN
        pred_candidates = [c for c in ckd.columns
                          if c != hemo_col and ckd[c].isna().mean() < 0.1
                          and ckd[c].nunique() > 3]
        pred_cols = pred_candidates[:4]
        if len(pred_cols) >= 2:
            df_mnar1 = select_and_rename(ckd, x0_col=hemo_col, other_cols=pred_cols)
            process_and_save(df_mnar1, "MNAR", "ckd_hemoglobin")
            print("  OK!")
        else:
            print(f"  Insuficientes preditores: {pred_cols}")
    else:
        print("  Coluna hemoglobin não encontrada")
except Exception as e:
    print(f"  ERRO: {e}")

# ======================================================================
# 6. MNAR: Adult Income (capital-gain) via GitHub mirror
# ======================================================================
print("\n=== MNAR: Adult Income (capital-gain) ===")
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
    text = download_text(url)
    names = ["age", "workclass", "fnlwgt", "education", "education_num",
             "marital_status", "occupation", "relationship", "race", "sex",
             "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    adult = pd.read_csv(io.StringIO(text), header=None, names=names,
                        na_values=" ?", skipinitialspace=True)

    adult_proc = adult.copy()
    adult_proc.loc[adult_proc["capital_gain"] == 0, "capital_gain"] = np.nan

    adult_sample = adult_proc.sample(n=1000, random_state=42)

    df_mnar2 = select_and_rename(
        adult_sample, x0_col="capital_gain",
        other_cols=["age", "education_num", "hours_per_week", "capital_loss"],
    )
    process_and_save(df_mnar2, "MNAR", "adult_capitalgain")
    print("  OK!")
except Exception as e:
    print(f"  ERRO: {e}")

# ======================================================================
# Resumo
# ======================================================================
print("\n" + "=" * 60)
print("DATASETS ADICIONAIS COLETADOS")
print("=" * 60)

for mech in ["MCAR", "MAR", "MNAR"]:
    mech_dir = os.path.join(OUTPUT_DIR, mech)
    files = sorted(os.listdir(mech_dir))
    print(f"\n{mech}/ ({len(files)} arquivos)")
    for f in files:
        df = pd.read_csv(os.path.join(mech_dir, f), sep="\t")
        miss = df["X0"].isna().mean() * 100
        print(f"  {f}: {len(df)} rows, {miss:.1f}% missing")
