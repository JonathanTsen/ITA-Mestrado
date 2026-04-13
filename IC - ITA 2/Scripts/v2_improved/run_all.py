"""
Script para executar o pipeline completo de extração e treinamento.

Uso:
    python run_all.py [--data sintetico|real|all] [--test]

Exemplos:
    python run_all.py                       # Dados sintéticos (default)
    python run_all.py --data real            # Dados reais
    python run_all.py --data all             # Ambos + comparação cruzada
    python run_all.py --data all --test      # Modo teste

Executa por tipo de dado:
1. Extração de features sem LLM (baseline)
2. Extração de features com LLM
3. Treinamento para cada configuração
4. Comparação de resultados
5. (se --data all) Comparação cruzada sintético vs real
"""
import os
import sys
import subprocess

MODELS = ["none", "gemini-3-flash-preview"]
TEST_MODE = "--test" in sys.argv

# Parse --data (aceita sintetico, real, all)
DATA_TYPES = ["sintetico"]
if "--data" in sys.argv:
    idx = sys.argv.index("--data")
    if idx + 1 < len(sys.argv):
        val = sys.argv[idx + 1]
        if val == "all":
            DATA_TYPES = ["sintetico", "real"]
        elif val in ("sintetico", "real"):
            DATA_TYPES = [val]
        else:
            print(f"❌ --data deve ser 'sintetico', 'real' ou 'all', recebido: '{val}'")
            sys.exit(1)

# Parse --experiment
EXPERIMENT = "default"
if "--experiment" in sys.argv:
    idx = sys.argv.index("--experiment")
    if idx + 1 < len(sys.argv):
        EXPERIMENT = sys.argv[idx + 1]

# Parse --llm-approach (STEP04: v2, judge, embeddings, caafe)
LLM_APPROACH = "judge"  # Default: novo approach do STEP04
if "--llm-approach" in sys.argv:
    idx = sys.argv.index("--llm-approach")
    if idx + 1 < len(sys.argv):
        LLM_APPROACH = sys.argv[idx + 1]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Salvar metadata do experimento
sys.path.insert(0, SCRIPT_DIR)
from utils.paths import save_experiment_config
save_experiment_config(EXPERIMENT)

print("=" * 60)
print("🚀 PIPELINE COMPLETO v2 IMPROVED")
print("=" * 60)

if TEST_MODE:
    print("🧪 MODO TESTE ATIVADO")

print(f"🔬 Experimento: {EXPERIMENT}")
print(f"📊 Tipos de dados: {DATA_TYPES}")
print(f"📋 Modelos a processar: {MODELS}")
print(f"🧠 Abordagem LLM: {LLM_APPROACH}")
print("=" * 60)

for data_type in DATA_TYPES:
    data_label = "DADOS SINTÉTICOS" if data_type == "sintetico" else "DADOS REAIS"

    print(f"\n{'#' * 60}")
    print(f"# {data_label}")
    print(f"{'#' * 60}")

    # Se dados reais, regenerar amostras bootstrap
    if data_type == "real":
        print(f"\n🔄 Gerando amostras bootstrap dos dados reais...")
        bootstrap_script = os.path.join(SCRIPT_DIR, "..", "subdividir_dados_reais.py")
        result = subprocess.run([sys.executable, bootstrap_script], cwd=SCRIPT_DIR)
        if result.returncode != 0:
            print(f"❌ Erro ao gerar bootstraps")
            continue

    for model in MODELS:
        abordagem = "apenas ML (baseline)" if model == "none" else f"ML + LLM ({model})"

        print(f"\n{'=' * 60}")
        print(f"📦 {data_label} | {abordagem}")
        print(f"{'=' * 60}")

        # 1. Extração de features
        print(f"\n🔧 Extraindo features...")
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "extract_features.py"),
               "--model", model, "--data", data_type, "--experiment", EXPERIMENT,
               "--llm-approach", LLM_APPROACH]
        if TEST_MODE:
            cmd.append("--test")

        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        if result.returncode != 0:
            print(f"❌ Erro na extração para {model} ({data_type})")
            continue

        # 2. Treinamento
        print(f"\n🤖 Treinando modelos...")
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_model.py"),
               "--model", model, "--data", data_type, "--experiment", EXPERIMENT]

        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        if result.returncode != 0:
            print(f"❌ Erro no treinamento para {model} ({data_type})")
            continue

        print(f"\n✅ {abordagem} ({data_type}) processado com sucesso!")

    # 3. Comparação por tipo de dado
    print(f"\n{'=' * 60}")
    print(f"📊 COMPARANDO RESULTADOS ({data_type.upper()})")
    print(f"{'=' * 60}")

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "compare_results.py"),
           "--data", data_type, "--experiment", EXPERIMENT]
    subprocess.run(cmd, cwd=SCRIPT_DIR)

    # 4. Ensemble adaptativo (requer baseline + LLM)
    if len(MODELS) > 1:
        print(f"\n{'=' * 60}")
        print(f"🔀 ENSEMBLE ADAPTATIVO ({data_type.upper()})")
        print(f"{'=' * 60}")

        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "ensemble_model.py"),
               "--data", data_type, "--experiment", EXPERIMENT]
        subprocess.run(cmd, cwd=SCRIPT_DIR)

    # 5. Validação de rótulos (apenas dados reais)
    if data_type == "real":
        print(f"\n{'=' * 60}")
        print(f"🔬 VALIDAÇÃO DE RÓTULOS ({data_type.upper()})")
        print(f"{'=' * 60}")

        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "validar_rotulos.py"),
               "--data", data_type, "--experiment", EXPERIMENT]
        subprocess.run(cmd, cwd=SCRIPT_DIR)

    # 6. Classificação MNAR Focused vs Diffuse
    print(f"\n{'=' * 60}")
    print(f"🔍 CLASSIFICAÇÃO MNAR FOCUSED vs DIFFUSE ({data_type.upper()})")
    print(f"{'=' * 60}")

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "classificar_mnar.py"),
           "--data", data_type, "--experiment", EXPERIMENT]
    subprocess.run(cmd, cwd=SCRIPT_DIR)

    # 7. Classificação Hierárquica + LOGO CV (STEP05)
    print(f"\n{'=' * 60}")
    print(f"🔀 CLASSIFICAÇÃO HIERÁRQUICA + LOGO CV ({data_type.upper()})")
    print(f"{'=' * 60}")

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_hierarchical.py"),
           "--model", "none", "--data", data_type, "--experiment", EXPERIMENT]
    subprocess.run(cmd, cwd=SCRIPT_DIR)

# 8. Comparação cruzada (se rodou ambos)
if len(DATA_TYPES) > 1:
    print(f"\n{'=' * 60}")
    print(f"🔄 COMPARAÇÃO CRUZADA: SINTÉTICO vs REAL")
    print(f"{'=' * 60}")

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "compare_results.py"),
           "--compare-all", "--experiment", EXPERIMENT]
    subprocess.run(cmd, cwd=SCRIPT_DIR)

# 9. Geração de outputs para tese (STEP05)
print(f"\n{'=' * 60}")
print(f"📝 GERAÇÃO DE OUTPUTS PARA TESE")
print(f"{'=' * 60}")

cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_thesis_outputs.py"),
       "--experiment", EXPERIMENT]
subprocess.run(cmd, cwd=SCRIPT_DIR)

print(f"\n{'=' * 60}")
print(f"✅ PIPELINE COMPLETO FINALIZADO!")
print(f"{'=' * 60}")
