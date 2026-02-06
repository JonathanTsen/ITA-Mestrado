"""
Script para executar o pipeline completo de extração e treinamento.

Uso:
    python run_all.py [--test]
    
Executa:
1. Extração de features sem LLM (baseline)
2. Extração de features com LLM
3. Treinamento para cada configuração
4. Comparação de resultados
"""
import os
import sys
import subprocess

# Configurações
MODELS = ["none", "gemini-3-flash-preview"]  # Adicione outros modelos conforme necessário
TEST_MODE = "--test" in sys.argv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("🚀 PIPELINE COMPLETO v2 IMPROVED")
print("=" * 60)

if TEST_MODE:
    print("🧪 MODO TESTE ATIVADO")
    
print(f"📋 Modelos a processar: {MODELS}")
print("=" * 60)

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"📦 PROCESSANDO: {model}")
    print(f"{'='*60}")
    
    # 1. Extração de features
    print(f"\n🔧 Extraindo features...")
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "extract_features.py"), "--model", model]
    if TEST_MODE:
        cmd.append("--test")
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"❌ Erro na extração para {model}")
        continue
    
    # 2. Treinamento
    print(f"\n🤖 Treinando modelos...")
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_model.py"), "--model", model]
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"❌ Erro no treinamento para {model}")
        continue
    
    print(f"\n✅ {model} processado com sucesso!")

# 3. Comparação final
print(f"\n{'='*60}")
print(f"📊 COMPARANDO RESULTADOS")
print(f"{'='*60}")

cmd = [sys.executable, os.path.join(SCRIPT_DIR, "compare_results.py")]
subprocess.run(cmd, cwd=SCRIPT_DIR)

print(f"\n{'='*60}")
print(f"✅ PIPELINE COMPLETO FINALIZADO!")
print(f"{'='*60}")
