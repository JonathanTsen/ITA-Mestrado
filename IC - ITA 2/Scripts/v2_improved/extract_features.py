"""
Script principal para extração de features v2 melhorada.

Uso:
    python extract_features.py --model <none|gemini-3-flash-preview|gpt-5.2> [--test]
    
Exemplos:
    python extract_features.py --model none              # Apenas features estatísticas
    python extract_features.py --model gemini-3-flash-preview  # Com LLM
    python extract_features.py --model none --test       # Modo teste (50 arquivos)
"""
import os
import sys
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Adiciona diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.statistical import extract_statistical_features
from features.discriminative import extract_discriminative_features
from llm.extractor_v2 import LLMFeatureExtractorV2, get_llm_fallback_features_v2

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURAÇÃO
# ======================================================
MODEL_TO_PROVIDER = {
    "gpt-5.2": "openai",
    "gpt-5-mini": "openai", 
    "gemini-3-pro-preview": "gemini",
    "gemini-3-flash-preview": "gemini",
}

# Parse argumentos
MODEL_NAME = "none"
TEST_MODE = False
MAX_WORKERS = 100

if "--model" in sys.argv:
    idx = sys.argv.index("--model")
    if idx + 1 < len(sys.argv):
        MODEL_NAME = sys.argv[idx + 1]

if "--test" in sys.argv:
    TEST_MODE = True

USE_LLM = MODEL_NAME != "none" and MODEL_NAME in MODEL_TO_PROVIDER

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

DATASET_PATHS = {
    "MCAR": os.path.join(BASE_DIR, "Dataset", "MCAR"),
    "MAR": os.path.join(BASE_DIR, "Dataset", "MAR"),
    "MNAR": os.path.join(BASE_DIR, "Dataset", "MNAR"),
}

# Output organizado por versão e modelo
OUTPUT_BASE = os.path.join(BASE_DIR, "Output", "v2_improved")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP = {"MCAR": 0, "MAR": 1, "MNAR": 2}

X_OUT = os.path.join(OUTPUT_DIR, "X_features.csv")
Y_OUT = os.path.join(OUTPUT_DIR, "y_labels.csv")

print(f"=" * 60)
print(f"🚀 EXTRAÇÃO DE FEATURES v2 MELHORADA")
print(f"=" * 60)
print(f"📝 Modelo: {MODEL_NAME}")
print(f"🤖 Usar LLM: {USE_LLM}")
print(f"📂 Output: {OUTPUT_DIR}")
if TEST_MODE:
    print(f"🧪 MODO TESTE: apenas 50 arquivos")
print(f"=" * 60)

# ======================================================
# INICIALIZA LLM (se necessário)
# ======================================================
llm_extractor = None
if USE_LLM:
    provider = MODEL_TO_PROVIDER[MODEL_NAME]
    print(f"🔧 Inicializando LLM v2: {MODEL_NAME} ({provider})")
    llm_extractor = LLMFeatureExtractorV2(MODEL_NAME, provider)


# ======================================================
# FUNÇÃO DE EXTRAÇÃO COMPLETA
# ======================================================
def extract_all_features(df: pd.DataFrame) -> dict:
    """Extrai todas as features de um DataFrame."""
    feats = {}
    
    # 1. Features estatísticas básicas
    feats.update(extract_statistical_features(df))
    
    # 2. Features discriminativas (MCAR/MAR/MNAR)
    feats.update(extract_discriminative_features(df))
    
    # 3. Features LLM (apenas se habilitado)
    if USE_LLM and llm_extractor is not None:
        llm_feats = llm_extractor.extract_features(df)
        feats.update(llm_feats)
    
    return feats


def process_file(args):
    """Processa um arquivo e retorna (features, label, index)."""
    filepath, classe, idx = args
    try:
        df = pd.read_csv(filepath, sep="\t")
        feats = extract_all_features(df)
        return (feats, LABEL_MAP[classe], idx, None)
    except Exception as e:
        return (None, LABEL_MAP[classe], idx, str(e))


# ======================================================
# COLETA ARQUIVOS
# ======================================================
tasks = []
for classe, pasta in DATASET_PATHS.items():
    if not os.path.exists(pasta):
        print(f"⚠️ Pasta não encontrada: {pasta}")
        continue
    arquivos = sorted([a for a in os.listdir(pasta) if a.endswith(".txt")])
    for arq in arquivos:
        filepath = os.path.join(pasta, arq)
        tasks.append((filepath, classe, len(tasks)))

if TEST_MODE:
    tasks = tasks[:50]

print(f"📁 Total de arquivos: {len(tasks)}")

# ======================================================
# CHECKPOINT SYSTEM
# ======================================================
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, ".checkpoint.json")
processed_files = set()
results = [None] * len(tasks)

if os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            processed_files = set(checkpoint.get("processed", []))
        print(f"📂 Checkpoint: {len(processed_files)} já processados")
    except Exception:
        pass

# Carrega resultados parciais
if os.path.exists(X_OUT) and os.path.exists(Y_OUT):
    try:
        X_partial = pd.read_csv(X_OUT)
        y_partial = pd.read_csv(Y_OUT)
        for i in range(min(len(X_partial), len(y_partial))):
            if i < len(results):
                results[i] = (X_partial.iloc[i].to_dict(), y_partial.iloc[i]['label'])
        print(f"📥 Resultados parciais: {len(X_partial)} linhas")
    except Exception:
        pass

# Filtra tarefas
tasks_to_process = [(fp, cl, idx) for fp, cl, idx in tasks if fp not in processed_files]
print(f"⏭️ Arquivos restantes: {len(tasks_to_process)}")

# ======================================================
# PROCESSAMENTO
# ======================================================
def save_checkpoint():
    """Salva checkpoint e CSVs parciais."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"processed": list(processed_files)}, f)
    
    X_partial = [r[0] for r in results if r is not None]
    y_partial = [r[1] for r in results if r is not None]
    
    if X_partial:
        pd.DataFrame(X_partial).replace([np.inf, -np.inf], 0).fillna(0).to_csv(X_OUT, index=False)
        pd.Series(y_partial, name="label").to_csv(Y_OUT, index=False)


if len(tasks_to_process) > 0:
    errors = []
    
    if USE_LLM and MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_file, t): t for t in tasks_to_process}
            
            for future in tqdm(as_completed(futures), total=len(tasks_to_process), 
                              desc="🤖 Extraindo features"):
                feats, label, idx, error = future.result()
                
                if error:
                    errors.append((tasks[idx][0], error))
                    continue
                
                results[idx] = (feats, label)
                processed_files.add(tasks[idx][0])
                
                if len(processed_files) % 20 == 0:
                    save_checkpoint()
    else:
        for t in tqdm(tasks_to_process, desc="📂 Extraindo features"):
            feats, label, idx, error = process_file(t)
            
            if error:
                errors.append((t[0], error))
                continue
            
            results[idx] = (feats, label)
            processed_files.add(t[0])
            
            if len(processed_files) % 20 == 0:
                save_checkpoint()
    
    if errors:
        print(f"\n⚠️ {len(errors)} erros durante processamento")
        for fp, err in errors[:5]:
            print(f"   - {os.path.basename(fp)}: {err}")

# ======================================================
# SALVA RESULTADOS FINAIS
# ======================================================
X_all = [r[0] for r in results if r is not None]
y_all = [r[1] for r in results if r is not None]

if len(X_all) == 0:
    print("❌ Nenhum resultado válido!")
    sys.exit(1)

X = pd.DataFrame(X_all).replace([np.inf, -np.inf], 0).fillna(0)
y = pd.Series(y_all, name="label")

X.to_csv(X_OUT, index=False)
y.to_csv(Y_OUT, index=False)

# Remove checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"\n{'=' * 60}")
print(f"✅ EXTRAÇÃO CONCLUÍDA!")
print(f"{'=' * 60}")
print(f"📊 Shape X: {X.shape}")
print(f"📊 Shape y: {y.shape}")
print(f"📊 Features: {list(X.columns)}")
print(f"💾 Salvo em: {OUTPUT_DIR}")
print(f"{'=' * 60}")

# Mostra distribuição de classes
print(f"\n📈 Distribuição de classes:")
for label, count in y.value_counts().sort_index().items():
    label_name = {0: "MCAR", 1: "MAR", 2: "MNAR"}[label]
    print(f"   {label_name}: {count} ({count/len(y)*100:.1f}%)")
