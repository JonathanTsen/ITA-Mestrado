"""
Script principal para extração de features v2 melhorada.

Uso:
    python extract_features.py --model <none|gemini-3-flash-preview|gpt-5.2> [--data sintetico|real] [--test]

Exemplos:
    python extract_features.py --model none                              # Baseline ML, dados sintéticos
    python extract_features.py --model gemini-3-flash-preview            # ML + LLM, dados sintéticos
    python extract_features.py --model none --data real                  # Baseline ML, dados reais
    python extract_features.py --model gemini-3-flash-preview --data real  # ML + LLM, dados reais
    python extract_features.py --model none --test                       # Modo teste (50 arquivos)
"""

import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Adiciona diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.advanced_l2 import extract_advanced_l2_features
from features.caafe_mnar import extract_caafe_mnar_features
from features.discriminative import extract_discriminative_features
from features.mechdetect import extract_mechdetect_features
from features.statistical import extract_statistical_features
from llm.context_aware import LLMContextAwareExtractor
from llm.embeddings import EmbeddingFeatureExtractor
from llm.extractor_v2 import LLMFeatureExtractorV2
from llm.judge_mnar import LLMJudgeMNAR
from llm.self_consistency import SelfConsistencyExtractor
from utils.args import parse_common_args, parse_llm_approach
from utils.paths import (
    MODEL_TO_PROVIDER,
    V2_DIR,
    get_dataset_paths,
    get_output_dir,
)

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURAÇÃO
# ======================================================
MODEL_NAME, DATA_TYPE, TEST_MODE, EXPERIMENT = parse_common_args()
LLM_APPROACH = parse_llm_approach()
_default_workers = 10 if LLM_APPROACH in ("context", "self_consistency") else 100
if "--workers" in sys.argv:
    idx = sys.argv.index("--workers")
    _default_workers = int(sys.argv[idx + 1])
MAX_WORKERS = _default_workers

# --metadata-variant: 'default' (original) ou 'neutral' (sem revelar mecanismo)
METADATA_VARIANT = "default"
if "--metadata-variant" in sys.argv:
    idx = sys.argv.index("--metadata-variant")
    if idx + 1 < len(sys.argv):
        METADATA_VARIANT = sys.argv[idx + 1]
    if METADATA_VARIANT not in ("default", "neutral"):
        print(f"❌ --metadata-variant deve ser 'default' ou 'neutral', recebido: '{METADATA_VARIANT}'")
        sys.exit(1)

# --datasets-include: filtra por parent-dataset (1 prefixo por linha; '#' = comentário)
DATASETS_INCLUDE: set[str] | None = None
if "--datasets-include" in sys.argv:
    idx = sys.argv.index("--datasets-include")
    if idx + 1 < len(sys.argv):
        _di_path = sys.argv[idx + 1]
        if not os.path.exists(_di_path):
            print(f"❌ --datasets-include: arquivo não encontrado: {_di_path}")
            sys.exit(1)
        with open(_di_path) as _f:
            DATASETS_INCLUDE = {line.strip() for line in _f if line.strip() and not line.startswith("#")}
        print(f"📋 Filtro --datasets-include: {len(DATASETS_INCLUDE)} parent-datasets de {_di_path}")

USE_LLM_API = MODEL_NAME != "none" and MODEL_NAME in MODEL_TO_PROVIDER
# CAAFE features são puras Python — habilitadas quando approach é "caafe" OU junto com LLM
USE_CAAFE = LLM_APPROACH in ("caafe", "context", "self_consistency") or USE_LLM_API
# Advanced L2 features (STEP 03) — habilitadas via --advanced-l2
USE_ADVANCED_L2 = "--advanced-l2" in sys.argv
# Embeddings usam modelo local (sentence-transformers) — não precisa de API
USE_EMBEDDINGS = LLM_APPROACH == "embeddings"
# Judge e v2 precisam de API
USE_LLM = USE_LLM_API and LLM_APPROACH in ("v2", "judge")
# Context-aware precisa de API
USE_CONTEXT = USE_LLM_API and LLM_APPROACH == "context"
# Self-consistency precisa de API (5 perspectivas paralelas)
USE_SC = USE_LLM_API and LLM_APPROACH == "self_consistency"

load_dotenv(os.path.join(V2_DIR, ".env"))

DATASET_PATHS = get_dataset_paths(DATA_TYPE)

OUTPUT_DIR = get_output_dir(DATA_TYPE, MODEL_NAME, EXPERIMENT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP = {"MCAR": 0, "MAR": 1, "MNAR": 2}

X_OUT = os.path.join(OUTPUT_DIR, "X_features.csv")
Y_OUT = os.path.join(OUTPUT_DIR, "y_labels.csv")

# Dados reais têm poucos arquivos, --test não faz sentido
if TEST_MODE and DATA_TYPE == "real":
    print("ℹ️ --test ignorado para dados reais (poucos arquivos)")
    TEST_MODE = False

ABORDAGEM = "apenas ML (baseline)" if MODEL_NAME == "none" else f"ML + LLM ({MODEL_NAME})"

print("=" * 60)
print("🚀 EXTRAÇÃO DE FEATURES v2 MELHORADA")
print("=" * 60)
print(f"📊 Dados: {DATA_TYPE}")
print(f"🔬 Abordagem: {ABORDAGEM}")
print(f"📝 Modelo LLM: {MODEL_NAME}")
print(f"🤖 Usar LLM API: {USE_LLM}")
print(f"🧠 Abordagem LLM: {LLM_APPROACH}")
print(f"📐 CAAFE features: {USE_CAAFE}")
print(f"📐 Advanced L2: {USE_ADVANCED_L2}")
print(f"🔤 Embeddings: {USE_EMBEDDINGS}")
print(f"🌐 Context-aware: {USE_CONTEXT}")
print(f"🔄 Self-consistency: {USE_SC}")
print(f"📖 Metadata variant: {METADATA_VARIANT}")
print(f"📂 Output: {OUTPUT_DIR}")
if TEST_MODE:
    print("🧪 MODO TESTE: apenas 50 arquivos")
print("=" * 60)

# ======================================================
# INICIALIZA LLM / EMBEDDINGS (conforme abordagem)
# ======================================================
llm_extractor = None
llm_judge = None
embedding_extractor = None
context_extractor = None
sc_extractor = None

if USE_SC:
    provider = MODEL_TO_PROVIDER[MODEL_NAME]
    print(f"🔧 Inicializando Self-Consistency: {MODEL_NAME} ({provider}) " f"[metadata_variant={METADATA_VARIANT}]")
    sc_extractor = SelfConsistencyExtractor(
        MODEL_NAME,
        provider,
        metadata_variant=METADATA_VARIANT,
    )

if USE_CONTEXT:
    provider = MODEL_TO_PROVIDER[MODEL_NAME]
    print(f"🔧 Inicializando LLM Context-Aware: {MODEL_NAME} ({provider}) " f"[metadata_variant={METADATA_VARIANT}]")
    context_extractor = LLMContextAwareExtractor(
        MODEL_NAME,
        provider,
        metadata_variant=METADATA_VARIANT,
    )

if USE_LLM:
    provider = MODEL_TO_PROVIDER[MODEL_NAME]

    if LLM_APPROACH == "v2":
        print(f"🔧 Inicializando LLM v2: {MODEL_NAME} ({provider})")
        llm_extractor = LLMFeatureExtractorV2(MODEL_NAME, provider)
    elif LLM_APPROACH == "judge":
        print(f"🔧 Inicializando LLM Judge MNAR: {MODEL_NAME} ({provider})")
        llm_judge = LLMJudgeMNAR(MODEL_NAME, provider)

if USE_EMBEDDINGS:
    print("🔧 Inicializando Embedding Extractor (sentence-transformers local)")
    embedding_extractor = EmbeddingFeatureExtractor(n_components=10, cache_dir=OUTPUT_DIR)

if USE_CAAFE and not USE_LLM and not USE_EMBEDDINGS:
    print("🔧 Usando features CAAFE-MNAR (sem chamada LLM)")


# ======================================================
# FUNÇÃO DE EXTRAÇÃO COMPLETA
# ======================================================
def extract_all_features(df: pd.DataFrame, filename: str = "") -> dict:
    """Extrai todas as features de um DataFrame."""
    feats = {}

    # 1. Features estatísticas básicas (4 features)
    feats.update(extract_statistical_features(df))

    # 2. Features discriminativas (11 features)
    feats.update(extract_discriminative_features(df))

    # 3. Features MechDetect (6 features)
    feats.update(extract_mechdetect_features(df))

    # 4. Features CAAFE-MNAR (4 features, puras Python)
    if USE_CAAFE:
        feats.update(extract_caafe_mnar_features(df))

    # 4b. Features avançadas L2 (7 features, STEP 03)
    if USE_ADVANCED_L2:
        feats.update(extract_advanced_l2_features(df))

    # 5. Features LLM API (v2 ou judge)
    if USE_LLM:
        if LLM_APPROACH == "v2" and llm_extractor is not None:
            feats.update(llm_extractor.extract_features(df))
        elif LLM_APPROACH == "judge" and llm_judge is not None:
            feats.update(llm_judge.judge(df))

    # 6. Features embedding (sentence-transformers local, sem API)
    if USE_EMBEDDINGS and embedding_extractor is not None:
        feats.update(embedding_extractor.extract_features(df))

    # 7. Features LLM context-aware (9 features, STEP context)
    if USE_CONTEXT and context_extractor is not None:
        feats.update(
            context_extractor.extract_features(
                df,
                filename=filename,
                data_type=DATA_TYPE,
            )
        )

    # 8. Features self-consistency (8 features, 5 perspectivas + CISC)
    if USE_SC and sc_extractor is not None:
        feats.update(
            sc_extractor.extract_features(
                df,
                filename=filename,
                data_type=DATA_TYPE,
            )
        )

    return feats


def process_file(args):
    """Processa um arquivo e retorna (features, label, index)."""
    filepath, classe, idx = args
    try:
        df = pd.read_csv(filepath, sep="\t")
        filename = os.path.basename(filepath)
        feats = extract_all_features(df, filename=filename)
        return (feats, LABEL_MAP[classe], idx, None)
    except Exception as e:
        return (None, LABEL_MAP[classe], idx, str(e))


# ======================================================
# COLETA ARQUIVOS
# ======================================================
import re as _re_filter

_BOOT_RE = _re_filter.compile(r"_boot\d+\.txt$")

tasks = []
_skipped_by_filter = 0
for classe, pasta in DATASET_PATHS.items():
    if not os.path.exists(pasta):
        print(f"⚠️ Pasta não encontrada: {pasta}")
        continue
    arquivos = sorted([a for a in os.listdir(pasta) if a.endswith(".txt")])
    for arq in arquivos:
        if DATASETS_INCLUDE is not None:
            parent = _BOOT_RE.sub("", arq)
            if parent not in DATASETS_INCLUDE:
                _skipped_by_filter += 1
                continue
        filepath = os.path.join(pasta, arq)
        tasks.append((filepath, classe, len(tasks)))

if DATASETS_INCLUDE is not None:
    print(f"📋 Filtro aplicado: {len(tasks)} arquivos incluídos, {_skipped_by_filter} excluídos")

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
        with open(CHECKPOINT_FILE) as f:
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
                results[i] = (X_partial.iloc[i].to_dict(), y_partial.iloc[i]["label"])
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
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed": list(processed_files), "total": len(tasks)}, f)

    X_partial = [r[0] for r in results if r is not None]
    y_partial = [r[1] for r in results if r is not None]

    if X_partial:
        X_ckpt = pd.DataFrame(X_partial).replace([np.inf, -np.inf], np.nan)
        # Checkpoint: fillna(0) temporário (imputação final ocorre ao salvar resultados)
        X_ckpt.fillna(0).to_csv(X_OUT, index=False)
        pd.Series(y_partial, name="label").to_csv(Y_OUT, index=False)


if len(tasks_to_process) > 0:
    errors = []

    if (USE_LLM or USE_CONTEXT or USE_SC) and MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_file, t): t for t in tasks_to_process}

            for future in tqdm(as_completed(futures), total=len(tasks_to_process), desc="🤖 Extraindo features"):
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
valid_indices = [i for i, r in enumerate(results) if r is not None]
X_all = [results[i][0] for i in valid_indices]
y_all = [results[i][1] for i in valid_indices]

if len(X_all) == 0:
    print("❌ Nenhum resultado válido!")
    sys.exit(1)

X = pd.DataFrame(X_all).replace([np.inf, -np.inf], np.nan)

# Imputação diferenciada por tipo de feature
llm_cols = [c for c in X.columns if c.startswith("llm_")]
emb_cols = [c for c in X.columns if c.startswith("emb_")]
model_cols = llm_cols + emb_cols  # Features que vêm de modelos (LLM/embeddings)
stat_cols = [c for c in X.columns if c not in model_cols]

# Features estatísticas + CAAFE: NaN -> 0 (ausência de sinal)
X[stat_cols] = X[stat_cols].fillna(0)

# Features LLM/embeddings: NaN -> mediana das amostras com resposta válida
if model_cols:
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    X[model_cols] = pd.DataFrame(imputer.fit_transform(X[model_cols]), columns=model_cols, index=X.index)

# Flush embedding cache se usado
if embedding_extractor is not None:
    embedding_extractor.flush_cache()

y = pd.Series(y_all, name="label")

# Extrai grupo (dataset de origem) de cada arquivo para GroupShuffleSplit
# Ex: "MCAR_breastcancer_barenuclei_boot001.txt" -> "MCAR_breastcancer_barenuclei"
import re

groups = []
for i in valid_indices:
    filepath = tasks[i][0]
    fname = os.path.basename(filepath)
    # Remove _bootNNN.txt para obter o grupo (dataset de origem)
    group = re.sub(r"_boot\d+\.txt$", "", fname)
    if group == fname.replace(".txt", ""):
        # Dados sintéticos ou sem bootstrap: cada arquivo é seu próprio grupo
        group = fname.replace(".txt", "")
    groups.append(group)

groups_series = pd.Series(groups, name="group")

X.to_csv(X_OUT, index=False)
y.to_csv(Y_OUT, index=False)
groups_series.to_csv(os.path.join(OUTPUT_DIR, "groups.csv"), index=False)

# Remove checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"\n{'=' * 60}")
print("✅ EXTRAÇÃO CONCLUÍDA!")
print(f"{'=' * 60}")
print(f"📊 Shape X: {X.shape}")
print(f"📊 Shape y: {y.shape}")
print(f"📊 Features: {list(X.columns)}")
print(f"💾 Salvo em: {OUTPUT_DIR}")
print(f"{'=' * 60}")

# Mostra distribuição de classes
print("\n📈 Distribuição de classes:")
for label, count in y.value_counts().sort_index().items():
    label_name = {0: "MCAR", 1: "MAR", 2: "MNAR"}[label]
    print(f"   {label_name}: {count} ({count/len(y)*100:.1f}%)")
