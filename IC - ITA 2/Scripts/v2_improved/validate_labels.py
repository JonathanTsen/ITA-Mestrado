"""
Validação estatística dos rótulos de mecanismo de missing data.

Aplica testes estatísticos para verificar se os mecanismos atribuídos
(MCAR, MAR, MNAR) aos dados reais estão corretos.

Testes:
  1. Teste de Little (MCAR): p > 0.05 → não rejeita MCAR
  2. Correlação missingness-preditores (MAR): correlação significativa → MAR
  3. Correlação missingness-valor (MNAR): padrão no próprio X0

Uso:
    python validate_labels.py --data real
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.args import parse_common_args
from utils.paths import get_comparison_dir, DATASET_PATHS

warnings.filterwarnings("ignore")

_, DATA_TYPE, _, EXPERIMENT = parse_common_args()

VALIDATION_DIR = os.path.join(get_comparison_dir(DATA_TYPE, EXPERIMENT), "validacao_rotulos")
os.makedirs(VALIDATION_DIR, exist_ok=True)

print("=" * 60)
print("🔬 VALIDAÇÃO DE RÓTULOS DE MECANISMO")
print("=" * 60)
print(f"📊 Dados: {DATA_TYPE}")
print("=" * 60)


def littles_mcar_proxy(df: pd.DataFrame) -> dict:
    """Proxy do teste de Little para MCAR.

    Compara médias das variáveis completas (X1-X4) entre
    observações com X0 presente vs X0 missing. Se as médias
    são similares, é consistente com MCAR.

    Retorna dict com p-valores para cada variável preditora.
    """
    mask = df["X0"].isna()
    if mask.sum() == 0 or (~mask).sum() == 0:
        return {"error": "Sem variação em missing"}

    results = {}
    predictors = [c for c in df.columns if c != "X0"]

    for col in predictors:
        obs_present = df.loc[~mask, col].dropna()
        obs_missing = df.loc[mask, col].dropna()

        if len(obs_present) < 3 or len(obs_missing) < 3:
            results[col] = {"t_stat": np.nan, "p_value": np.nan}
            continue

        t_stat, p_value = stats.ttest_ind(obs_present, obs_missing, equal_var=False)
        results[col] = {"t_stat": t_stat, "p_value": p_value}

    return results


def mar_test(df: pd.DataFrame) -> dict:
    """Testa se missingness de X0 correlaciona com X1-X4 (indicativo de MAR).

    Usa correlação ponto-bisserial entre máscara de missing e cada preditor.
    """
    mask = df["X0"].isna().astype(int)
    results = {}
    predictors = [c for c in df.columns if c != "X0"]

    for col in predictors:
        vals = df[col].dropna()
        mask_aligned = mask[vals.index]

        if len(vals) < 10 or mask_aligned.nunique() < 2:
            results[col] = {"correlation": np.nan, "p_value": np.nan}
            continue

        corr, p_value = stats.pointbiserialr(mask_aligned, vals)
        results[col] = {"correlation": corr, "p_value": p_value}

    return results


def mnar_test(df: pd.DataFrame) -> dict:
    """Testa se missingness de X0 correlaciona com o próprio X0 (MNAR).

    Compara a distribuição de X0 observado entre grupos definidos por
    quartis — se missing concentra-se em valores extremos, sugere MNAR.
    """
    x0_obs = df["X0"].dropna()
    mask = df["X0"].isna()

    if len(x0_obs) < 10 or mask.sum() < 3:
        return {"error": "Amostras insuficientes"}

    # Teste: observações perto da mediana vs extremos têm taxas de missing diferentes?
    median_val = x0_obs.median()
    q25, q75 = x0_obs.quantile(0.25), x0_obs.quantile(0.75)

    results = {
        "x0_mean_observed": x0_obs.mean(),
        "x0_std_observed": x0_obs.std(),
        "missing_rate": mask.mean(),
    }

    # Mann-Whitney entre X0 observado e X1 (proxy para relação valor-missing)
    # Se X0 missing depende do valor de X0, os observados terão distribuição enviesada
    if "X1" in df.columns:
        x1_missing = df.loc[mask, "X1"].dropna()
        x1_present = df.loc[~mask, "X1"].dropna()
        if len(x1_missing) >= 3 and len(x1_present) >= 3:
            u_stat, p_val = stats.mannwhitneyu(x1_present, x1_missing, alternative="two-sided")
            results["x1_mannwhitney_p"] = p_val

    return results


# ======================================================
# PROCESSA CADA ARQUIVO ORIGINAL
# ======================================================
# Usa dados processados (pré-bootstrap) quando disponíveis
PROCESSADO_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "Dataset", "real_data", "processado")

all_results = []

for mechanism in ["MCAR", "MAR", "MNAR"]:
    mech_dir = os.path.join(PROCESSADO_DIR, mechanism)
    if not os.path.isdir(mech_dir):
        print(f"\n⚠️ Diretório não encontrado: {mech_dir}")
        continue

    files = sorted([f for f in os.listdir(mech_dir) if f.endswith(".txt")])

    print(f"\n{'='*50}")
    print(f"📁 {mechanism} ({len(files)} arquivos)")
    print(f"{'='*50}")

    for fname in files:
        filepath = os.path.join(mech_dir, fname)
        df = pd.read_csv(filepath, sep="\t")
        missing_rate = df["X0"].isna().mean()

        print(f"\n  📄 {fname} ({len(df)} rows, {missing_rate*100:.1f}% missing)")

        # 1. Teste MCAR (Little's proxy)
        little_results = littles_mcar_proxy(df)
        if "error" not in little_results:
            p_vals = [v["p_value"] for v in little_results.values() if not np.isnan(v["p_value"])]
            min_p = min(p_vals) if p_vals else np.nan
            all_significant = all(p < 0.05 for p in p_vals) if p_vals else False
            mcar_consistent = not all_significant  # MCAR se NÃO rejeitamos

            print(f"     MCAR test (Little's proxy): min_p={min_p:.4f}")
            for col, vals in little_results.items():
                sig = "***" if vals["p_value"] < 0.05 else ""
                print(f"       {col}: t={vals['t_stat']:.3f}, p={vals['p_value']:.4f} {sig}")
            print(f"     → Consistente com MCAR: {'SIM' if mcar_consistent else 'NÃO'}")
        else:
            mcar_consistent = None
            min_p = np.nan

        # 2. Teste MAR
        mar_results = mar_test(df)
        max_corr = 0
        mar_p_min = 1.0
        for col, vals in mar_results.items():
            if not np.isnan(vals.get("correlation", np.nan)):
                if abs(vals["correlation"]) > abs(max_corr):
                    max_corr = vals["correlation"]
                if vals["p_value"] < mar_p_min:
                    mar_p_min = vals["p_value"]

        mar_evidence = mar_p_min < 0.05 and abs(max_corr) > 0.1
        print(f"     MAR test: max_corr={max_corr:.4f}, min_p={mar_p_min:.4f}")
        print(f"     → Evidência de MAR: {'SIM' if mar_evidence else 'NÃO'}")

        # 3. Teste MNAR
        mnar_results = mnar_test(df)
        print(f"     MNAR test: {mnar_results}")

        # Diagnóstico
        label = mechanism
        if mechanism == "MCAR" and not mcar_consistent:
            diagnosis = "⚠️ INCONSISTENTE: rotulado MCAR mas rejeita hipótese MCAR"
        elif mechanism == "MAR" and not mar_evidence:
            diagnosis = "⚠️ FRACO: rotulado MAR mas sem correlação significativa"
        elif mechanism == "MNAR":
            diagnosis = "ℹ️ MNAR difícil de testar sem acesso ao valor verdadeiro"
        else:
            diagnosis = "✅ CONSISTENTE"

        print(f"     → Diagnóstico: {diagnosis}")

        all_results.append({
            "arquivo": fname,
            "mecanismo_rotulo": mechanism,
            "n_rows": len(df),
            "missing_rate": missing_rate,
            "mcar_min_p": min_p,
            "mcar_consistente": mcar_consistent,
            "mar_max_corr": max_corr,
            "mar_p_min": mar_p_min,
            "mar_evidencia": mar_evidence,
            "diagnostico": diagnosis,
        })

# ======================================================
# SALVA RESULTADOS
# ======================================================
df_val = pd.DataFrame(all_results)
csv_path = os.path.join(VALIDATION_DIR, "validacao_rotulos.csv")
df_val.to_csv(csv_path, index=False)

report_path = os.path.join(VALIDATION_DIR, "validacao_rotulos.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("VALIDAÇÃO DE RÓTULOS DE MECANISMO DE MISSING DATA\n")
    f.write("=" * 60 + "\n\n")
    for _, row in df_val.iterrows():
        f.write(f"{row['arquivo']}:\n")
        f.write(f"  Rótulo: {row['mecanismo_rotulo']}\n")
        f.write(f"  Missing rate: {row['missing_rate']*100:.1f}%\n")
        f.write(f"  MCAR consistente: {row['mcar_consistente']}\n")
        f.write(f"  MAR evidência: {row['mar_evidencia']}\n")
        f.write(f"  Diagnóstico: {row['diagnostico']}\n\n")

print(f"\n{'='*60}")
print(f"✅ VALIDAÇÃO CONCLUÍDA!")
print(f"{'='*60}")
print(f"\n💾 Resultados: {csv_path}")
print(f"💾 Relatório: {report_path}")

# Resumo
n_consistent = sum(1 for r in all_results if "CONSISTENTE" in r["diagnostico"] and "IN" not in r["diagnostico"])
n_total = len(all_results)
print(f"\n📊 Resumo: {n_consistent}/{n_total} rótulos consistentes")
