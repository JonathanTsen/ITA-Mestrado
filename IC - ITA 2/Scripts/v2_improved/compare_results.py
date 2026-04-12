"""
Script para comparar resultados entre configurações.

Uso:
    python compare_results.py --data sintetico          # Compara dentro de dados sintéticos
    python compare_results.py --data real                # Compara dentro de dados reais
    python compare_results.py --compare-all              # Comparação cruzada sintético vs real
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.args import parse_common_args
from utils.paths import get_comparison_dir, find_result_dirs, OUTPUT_BASE, get_experiment_dir


def parse_relatorio(filepath):
    """Extrai acurácias do relatório."""
    results = {}
    current_model = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== ") and line.endswith(" ==="):
                current_model = line[4:-4]
            elif line.startswith("Acurácia:") and current_model:
                acc = float(line.split(":")[1].strip())
                results[current_model] = acc

    return results


def compare_data_type(data_type, experiment="default"):
    """Compara resultados dentro de um tipo de dado."""
    result_dirs = find_result_dirs(data_type, experiment)

    if not result_dirs:
        print(f"❌ Nenhum resultado encontrado para dados '{data_type}'!")
        print(f"   Execute primeiro: python extract_features.py --model <modelo> --data {data_type}")
        print(f"   Depois: python train_model.py --model <modelo> --data {data_type}")
        return None

    print(f"\n📂 Resultados encontrados ({data_type}):")
    for name, _, abordagem in result_dirs:
        print(f"   - {name} [{abordagem}]")

    all_results = {}
    for name, dir_path, _ in result_dirs:
        rel_path = os.path.join(dir_path, "relatorio.txt")
        all_results[name] = parse_relatorio(rel_path)

    df_compare = pd.DataFrame(all_results)
    print(f"\n📊 COMPARAÇÃO DE RESULTADOS ({data_type.upper()}):")
    print(df_compare.to_string())

    # Calcula diferenças vs baseline
    baseline_col = "baseline (apenas ML)"
    if baseline_col in df_compare.columns and len(df_compare.columns) > 1:
        print(f"\n📈 DIFERENÇA vs BASELINE (apenas ML):")
        for col in df_compare.columns:
            if col != baseline_col:
                diff = df_compare[col] - df_compare[baseline_col]
                print(f"\n   {col}:")
                for model, d in diff.items():
                    symbol = "✅" if d > 0 else "❌" if d < 0 else "➖"
                    print(f"     {symbol} {model}: {d:+.4f}")

    # Gráfico comparativo
    comparison_base = get_comparison_dir(data_type, experiment)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_compare.index))
    width = 0.8 / len(df_compare.columns)
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_compare.columns)))

    for i, col in enumerate(df_compare.columns):
        offset = (i - len(df_compare.columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_compare[col], width, label=col, color=colors[i])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

    data_label = "Dados Sintéticos" if data_type == "sintetico" else "Dados Reais"
    ax.set_ylabel('Acurácia')
    ax.set_title(f'Comparação de Abordagens - {data_label}')
    ax.set_xticks(x)
    ax.set_xticklabels(df_compare.index, rotation=45, ha='right')
    ax.legend(title="Configuração")
    ax.set_ylim([0, 1])
    ax.axhline(y=0.333, color='red', linestyle='--', alpha=0.5, label="Random")

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_base, "comparacao.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n💾 Gráfico salvo: {os.path.join(comparison_base, 'comparacao.png')}")

    csv_path = os.path.join(comparison_base, "comparacao.csv")
    df_compare.to_csv(csv_path)
    print(f"💾 CSV salvo: {csv_path}")

    return df_compare


def compare_all(experiment="default"):
    """Comparação cruzada entre dados sintéticos e reais."""
    print(f"\n{'=' * 60}")
    print(f"🔄 COMPARAÇÃO CRUZADA: SINTÉTICO vs REAL")
    print(f"{'=' * 60}")

    dfs = {}
    for data_type in ("sintetico", "real"):
        csv_path = os.path.join(get_comparison_dir(data_type, experiment), "comparacao.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            dfs[data_type] = df
        else:
            print(f"⚠️ Comparação não encontrada para '{data_type}'. Execute primeiro com --data {data_type}")

    if len(dfs) < 2:
        print("❌ Necessário ter resultados de ambos os tipos para comparação cruzada.")
        return

    # Combina em um único DataFrame
    rows = []
    for data_type, df in dfs.items():
        for config_col in df.columns:
            for ml_model, acc in df[config_col].items():
                rows.append({
                    "tipo_dado": data_type,
                    "configuracao": config_col,
                    "modelo_ml": ml_model,
                    "acuracia": acc,
                })

    df_cross = pd.DataFrame(rows)
    cross_path = os.path.join(get_experiment_dir(experiment), "comparacao_sintetico_vs_real.csv")
    df_cross.to_csv(cross_path, index=False)
    print(f"\n💾 Comparação cruzada salva: {cross_path}")
    print(df_cross.to_string(index=False))


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    _, DATA_TYPE, _, EXPERIMENT = parse_common_args()
    do_compare_all = "--compare-all" in sys.argv

    if do_compare_all:
        # Roda comparação para cada tipo e depois cruzada
        for dt in ("sintetico", "real"):
            compare_data_type(dt, EXPERIMENT)
        compare_all(EXPERIMENT)
    else:
        compare_data_type(DATA_TYPE, EXPERIMENT)
