"""
Parsing centralizado de argumentos CLI do pipeline v2.
"""
import sys

# Abordagens LLM disponíveis (STEP04)
LLM_APPROACHES = ("v2", "judge", "embeddings", "caafe", "context", "self_consistency")


def parse_common_args() -> tuple[str, str, bool, str]:
    """Parse --model, --data, --test e --experiment de sys.argv.

    Retorna (model_name, data_type, test_mode, experiment).
    """
    model_name = "none"
    data_type = "sintetico"
    test_mode = False
    experiment = "default"

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_name = sys.argv[idx + 1]

    if "--data" in sys.argv:
        idx = sys.argv.index("--data")
        if idx + 1 < len(sys.argv):
            data_type = sys.argv[idx + 1]
        if data_type not in ("sintetico", "real"):
            print(f"❌ --data deve ser 'sintetico' ou 'real', recebido: '{data_type}'")
            sys.exit(1)

    if "--test" in sys.argv:
        test_mode = True

    if "--experiment" in sys.argv:
        idx = sys.argv.index("--experiment")
        if idx + 1 < len(sys.argv):
            experiment = sys.argv[idx + 1]

    return model_name, data_type, test_mode, experiment


def parse_llm_approach() -> str:
    """Parse --llm-approach de sys.argv.

    Retorna a abordagem LLM escolhida:
    - 'v2': prompt original de 8 features (default, backward-compatible)
    - 'judge': desambiguação binária MCAR vs MNAR (4 features)
    - 'embeddings': sentence-transformers embedding (10 features)
    - 'caafe': features CAAFE-MNAR puras Python (4 features, sem LLM)
    - 'context': features context-aware com domínio + contra-argumentação (9 features)
    - 'self_consistency': 5 perspectivas com votação CISC (8 features)
    """
    approach = "v2"

    if "--llm-approach" in sys.argv:
        idx = sys.argv.index("--llm-approach")
        if idx + 1 < len(sys.argv):
            approach = sys.argv[idx + 1]
        if approach not in LLM_APPROACHES:
            print(f"❌ --llm-approach deve ser {LLM_APPROACHES}, recebido: '{approach}'")
            sys.exit(1)

    return approach
