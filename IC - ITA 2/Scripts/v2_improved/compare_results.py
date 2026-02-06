"""
Script para comparar resultados com e sem LLM.

Uso:
    python compare_results.py
    
Compara os resultados de diferentes configurações.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_BASE = os.path.join(BASE_DIR, "Output", "v2_improved")

# Encontra todas as pastas de resultados
result_dirs = []
for d in os.listdir(OUTPUT_BASE):
    dir_path = os.path.join(OUTPUT_BASE, d)
    if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "relatorio.txt")):
        result_dirs.append(d)

if not result_dirs:
    print("❌ Nenhum resultado encontrado!")
    print(f"   Execute primeiro: python extract_features.py --model <modelo>")
    print(f"   Depois: python train_model.py --model <modelo>")
    exit(1)

print(f"📂 Resultados encontrados: {result_dirs}")

# Parse resultados
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

all_results = {}
for d in result_dirs:
    rel_path = os.path.join(OUTPUT_BASE, d, "relatorio.txt")
    all_results[d] = parse_relatorio(rel_path)

# Cria DataFrame comparativo
df_compare = pd.DataFrame(all_results)
print(f"\n📊 COMPARAÇÃO DE RESULTADOS:")
print(df_compare.to_string())

# Calcula diferenças
if "none" in df_compare.columns and len(df_compare.columns) > 1:
    print(f"\n📈 DIFERENÇA vs BASELINE (none):")
    for col in df_compare.columns:
        if col != "none":
            diff = df_compare[col] - df_compare["none"]
            print(f"\n   {col}:")
            for model, d in diff.items():
                symbol = "✅" if d > 0 else "❌" if d < 0 else "➖"
                print(f"     {symbol} {model}: {d:+.4f}")

# Gráfico comparativo
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(df_compare.index))
width = 0.8 / len(df_compare.columns)

colors = plt.cm.Set2(np.linspace(0, 1, len(df_compare.columns)))

for i, col in enumerate(df_compare.columns):
    offset = (i - len(df_compare.columns)/2 + 0.5) * width
    bars = ax.bar(x + offset, df_compare[col], width, label=col, color=colors[i])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_ylabel('Acurácia')
ax.set_title('Comparação: Features Estatísticas vs Features + LLM')
ax.set_xticks(x)
ax.set_xticklabels(df_compare.index, rotation=45, ha='right')
ax.legend(title="Configuração")
ax.set_ylim([0, 1])
ax.axhline(y=0.333, color='red', linestyle='--', alpha=0.5, label="Random")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "comparacao_geral.png"), dpi=300, bbox_inches='tight')
print(f"\n💾 Gráfico salvo: {os.path.join(OUTPUT_BASE, 'comparacao_geral.png')}")

# Salva CSV
df_compare.to_csv(os.path.join(OUTPUT_BASE, "comparacao_resultados.csv"))
print(f"💾 CSV salvo: {os.path.join(OUTPUT_BASE, 'comparacao_resultados.csv')}")
