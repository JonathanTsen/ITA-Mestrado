"""
validar_rotulos.py — Validação estatística de rótulos de mecanismo de missing data.

Implementa 3 testes conforme STEP03:
  1. Little's MCAR test (via MissMecha): p > 0.05 → consistente com MCAR
  2. Correlação ponto-biserial mask-Xi: |corr| > 0.1 com p < 0.05 → evidência MAR
  3. KS test observados vs imputados: p < 0.05 → evidência MNAR

Gera relatório com diagnóstico e rótulo recomendado.

Uso:
    cd "IC - ITA 2/Scripts/v2_improved"
    uv run python validar_rotulos.py --data real [--experiment <name>]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.args import parse_common_args
from utils.paths import get_comparison_dir

warnings.filterwarnings("ignore")

_, DATA_TYPE, _, EXPERIMENT = parse_common_args()

VALIDATION_DIR = os.path.join(get_comparison_dir(DATA_TYPE, EXPERIMENT), "validacao_rotulos")
os.makedirs(VALIDATION_DIR, exist_ok=True)

PROCESSADO_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "Dataset", "real_data", "processado")

print("=" * 60)
print("VALIDACAO DE ROTULOS — 3 TESTES ESTATISTICOS")
print("=" * 60)
print(f"Dados: {DATA_TYPE}")
print(f"Diretorio: {PROCESSADO_DIR}")
print("=" * 60)


# ======================================================
# TESTE 1: Little's MCAR test (MissMecha)
# ======================================================
def test_little_mcar(df: pd.DataFrame) -> float:
    """Aplica Little's MCAR test. Retorna p-valor."""
    try:
        from missmecha import MCARTest
        p_value = MCARTest.little_mcar_test(df)
        return float(p_value)
    except Exception as e:
        print(f"    [Little] Erro: {e}")
        # Fallback: proxy baseado em t-tests
        return _little_proxy(df)


def _little_proxy(df: pd.DataFrame) -> float:
    """Proxy do Little's test via t-tests pareados."""
    mask = df["X0"].isna()
    if mask.sum() == 0 or (~mask).sum() == 0:
        return np.nan

    p_vals = []
    for col in [c for c in df.columns if c != "X0"]:
        obs = df.loc[~mask, col].dropna()
        mis = df.loc[mask, col].dropna()
        if len(obs) >= 3 and len(mis) >= 3:
            _, p = stats.ttest_ind(obs, mis, equal_var=False)
            p_vals.append(p)

    if not p_vals:
        return np.nan
    # Combina p-valores via Fisher's method
    chi2 = -2 * sum(np.log(max(p, 1e-300)) for p in p_vals)
    combined_p = 1 - stats.chi2.cdf(chi2, df=2 * len(p_vals))
    return combined_p


# ======================================================
# TESTE 2: Correlação mask-Xi (evidência MAR)
# ======================================================
def test_mar_correlation(df: pd.DataFrame) -> dict:
    """Correlação ponto-biserial entre máscara de X0 e cada Xi."""
    mask = df["X0"].isna().astype(int)
    results = {}
    for col in [c for c in df.columns if c != "X0"]:
        vals = df[col].dropna()
        mask_aligned = mask[vals.index]
        if len(vals) < 10 or mask_aligned.nunique() < 2:
            results[col] = {"corr": np.nan, "p": np.nan}
            continue
        corr, p = stats.pointbiserialr(mask_aligned, vals)
        results[col] = {"corr": corr, "p": p}
    return results


# ======================================================
# TESTE 3: KS test observados vs imputados (evidência MNAR)
# ======================================================
def test_mnar_ks(df: pd.DataFrame) -> dict:
    """Compara distribuição de X0 observado vs X0 imputado com mediana."""
    x0 = df["X0"].copy()
    mask = x0.isna()
    x0_obs = x0.dropna().values

    if len(x0_obs) < 10 or mask.sum() < 3:
        return {"ks_stat": np.nan, "ks_p": np.nan}

    # Imputar missing com mediana
    median_val = np.median(x0_obs)
    x0_imputed = x0.fillna(median_val).values

    # KS test entre observados e imputados completos
    ks_stat, ks_p = stats.ks_2samp(x0_obs, x0_imputed)
    return {"ks_stat": ks_stat, "ks_p": ks_p}


# ======================================================
# DIAGNÓSTICO
# ======================================================
def diagnose(label: str, little_p: float, max_corr: float, mar_sig: bool,
             ks_p: float) -> tuple[str, str]:
    """Retorna (diagnóstico, rótulo_recomendado)."""
    mcar_consistent = little_p > 0.05 if not np.isnan(little_p) else None
    ks_significant = ks_p < 0.05 if not np.isnan(ks_p) else False

    # Regras do STEP03
    if mcar_consistent and not mar_sig and not ks_significant:
        diag = "MCAR confirmado"
        recommended = "MCAR"
    elif not mcar_consistent and mar_sig:
        diag = "Evidencia MAR (rejeita MCAR + correlacao mask-Xi)"
        recommended = "MAR"
    elif not mcar_consistent and ks_significant and not mar_sig:
        diag = "Possivelmente MNAR (rejeita MCAR + KS significativo)"
        recommended = "MNAR"
    elif mcar_consistent is None:
        diag = "Inconclusivo (Little test falhou)"
        recommended = label
    else:
        # Testes conflitantes
        diag = "Ambiguo (testes conflitantes)"
        recommended = label

    consistency = "CONSISTENTE" if recommended == label else "INCONSISTENTE"
    return f"{consistency}: {diag}", recommended


# ======================================================
# PROCESSA CADA ARQUIVO
# ======================================================
all_results = []

for mechanism in ["MCAR", "MAR", "MNAR"]:
    mech_dir = os.path.join(PROCESSADO_DIR, mechanism)
    if not os.path.isdir(mech_dir):
        print(f"\nDiretorio nao encontrado: {mech_dir}")
        continue

    files = sorted([f for f in os.listdir(mech_dir) if f.endswith(".txt")])

    print(f"\n{'='*50}")
    print(f"{mechanism} ({len(files)} arquivos)")
    print(f"{'='*50}")

    for fname in files:
        filepath = os.path.join(mech_dir, fname)
        df = pd.read_csv(filepath, sep="\t")
        missing_rate = df["X0"].isna().mean()

        print(f"\n  {fname} ({len(df)} rows, {missing_rate*100:.1f}% missing)")

        # Teste 1: Little's MCAR
        little_p = test_little_mcar(df)
        mcar_str = f"p={little_p:.4f}" if not np.isnan(little_p) else "FALHOU"
        reject_str = "rejeita MCAR" if (not np.isnan(little_p) and little_p < 0.05) else "nao rejeita"
        print(f"    [Little] {mcar_str} -> {reject_str}")

        # Teste 2: Correlação MAR
        mar_results = test_mar_correlation(df)
        max_corr = 0.0
        mar_p_min = 1.0
        best_col = ""
        for col, vals in mar_results.items():
            if not np.isnan(vals["corr"]) and abs(vals["corr"]) > abs(max_corr):
                max_corr = vals["corr"]
                mar_p_min = vals["p"]
                best_col = col

        mar_sig = mar_p_min < 0.05 and abs(max_corr) > 0.1
        print(f"    [MAR corr] max|corr|={abs(max_corr):.4f} ({best_col}), p={mar_p_min:.4f} -> {'evidencia MAR' if mar_sig else 'sem evidencia'}")

        # Teste 3: KS test MNAR
        ks_results = test_mnar_ks(df)
        ks_stat = ks_results["ks_stat"]
        ks_p = ks_results["ks_p"]
        ks_str = f"stat={ks_stat:.4f}, p={ks_p:.4f}" if not np.isnan(ks_stat) else "FALHOU"
        ks_sig = ks_p < 0.05 if not np.isnan(ks_p) else False
        print(f"    [KS MNAR] {ks_str} -> {'evidencia MNAR' if ks_sig else 'sem evidencia'}")

        # Diagnóstico
        diagnosis, recommended = diagnose(mechanism, little_p, max_corr, mar_sig, ks_p)
        print(f"    DIAGNOSTICO: {diagnosis}")
        if recommended != mechanism:
            print(f"    *** ROTULO RECOMENDADO: {recommended} (atual: {mechanism}) ***")

        all_results.append({
            "arquivo": fname,
            "rotulo_atual": mechanism,
            "n_rows": len(df),
            "missing_rate": round(missing_rate, 4),
            "little_p": round(little_p, 6) if not np.isnan(little_p) else np.nan,
            "max_corr_Xi": round(max_corr, 4),
            "corr_col": best_col,
            "mar_p_min": round(mar_p_min, 6),
            "ks_stat": round(ks_stat, 4) if not np.isnan(ks_stat) else np.nan,
            "ks_p": round(ks_p, 6) if not np.isnan(ks_p) else np.nan,
            "diagnostico": diagnosis,
            "rotulo_recomendado": recommended,
        })

# ======================================================
# SALVA RESULTADOS
# ======================================================
df_val = pd.DataFrame(all_results)
csv_path = os.path.join(VALIDATION_DIR, "validacao_rotulos.csv")
df_val.to_csv(csv_path, index=False)

report_path = os.path.join(VALIDATION_DIR, "validacao_rotulos.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("VALIDACAO DE ROTULOS — 3 TESTES ESTATISTICOS\n")
    f.write("=" * 60 + "\n")
    f.write("Testes: Little's MCAR | Correlacao mask-Xi | KS obs vs imputado\n\n")

    for _, row in df_val.iterrows():
        f.write(f"{row['arquivo']}:\n")
        f.write(f"  Rotulo atual: {row['rotulo_atual']}\n")
        f.write(f"  Missing rate: {row['missing_rate']*100:.1f}%\n")
        f.write(f"  Little p-value: {row['little_p']}\n")
        f.write(f"  Max |corr| mask-Xi: {row['max_corr_Xi']} ({row['corr_col']})\n")
        f.write(f"  KS stat: {row['ks_stat']}, p={row['ks_p']}\n")
        f.write(f"  Diagnostico: {row['diagnostico']}\n")
        f.write(f"  Rotulo recomendado: {row['rotulo_recomendado']}\n\n")

print(f"\n{'='*60}")
print(f"VALIDACAO CONCLUIDA!")
print(f"{'='*60}")
print(f"Resultados: {csv_path}")
print(f"Relatorio: {report_path}")

# Resumo
n_consistent = sum(1 for r in all_results if "CONSISTENTE" in r["diagnostico"] and "INCONSISTENTE" not in r["diagnostico"])
n_inconsistent = sum(1 for r in all_results if "INCONSISTENTE" in r["diagnostico"])
n_total = len(all_results)
print(f"\nResumo: {n_consistent}/{n_total} consistentes, {n_inconsistent}/{n_total} inconsistentes")

if n_inconsistent > 0:
    print("\nDatasets inconsistentes:")
    for r in all_results:
        if "INCONSISTENTE" in r["diagnostico"]:
            print(f"  {r['arquivo']}: {r['rotulo_atual']} -> {r['rotulo_recomendado']}")
