"""
Centraliza paths do projeto para todos os scripts do pipeline v2.
"""
import os

# utils/ está em Scripts/v2_improved/utils/
# _THIS_DIR = .../Scripts/v2_improved/utils/
# BASE_DIR = IC - ITA 2/ (3 níveis acima de _THIS_DIR)
# SCRIPT_DIR = Scripts/ (2 níveis acima de _THIS_DIR)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(_THIS_DIR)                        # Scripts/v2_improved/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))  # IC - ITA 2/
SCRIPT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))   # Scripts/

OUTPUT_BASE = os.path.join(BASE_DIR, "Output", "v2_improved")

DATASET_PATHS = {
    "sintetico": {
        "MCAR": os.path.join(BASE_DIR, "Dataset", "synthetic_data", "MCAR"),
        "MAR": os.path.join(BASE_DIR, "Dataset", "synthetic_data", "MAR"),
        "MNAR": os.path.join(BASE_DIR, "Dataset", "synthetic_data", "MNAR"),
    },
    "real": {
        "MCAR": os.path.join(BASE_DIR, "Dataset", "real_data", "processado_chunks", "MCAR"),
        "MAR": os.path.join(BASE_DIR, "Dataset", "real_data", "processado_chunks", "MAR"),
        "MNAR": os.path.join(BASE_DIR, "Dataset", "real_data", "processado_chunks", "MNAR"),
    },
}

MODEL_TO_PROVIDER = {
    "gpt-5.2": "openai",
    "gpt-5-mini": "openai",
    "gemini-3-pro-preview": "gemini",
    "gemini-3-flash-preview": "gemini",
    "gemini-3.1-pro-preview": "gemini",
}


def get_experiment_dir(experiment: str = "default") -> str:
    """Retorna o diretório raiz de um experimento."""
    return os.path.join(OUTPUT_BASE, experiment)


def get_output_dir(data_type: str, model_name: str, experiment: str = "default") -> str:
    """Retorna o diretório de output para um experimento.

    Regra:
      - model_name == "none" → {experiment}/{data_type}/apenas_ml/baseline/
      - qualquer outro       → {experiment}/{data_type}/ml_com_llm/{model_name}/
    """
    exp_base = get_experiment_dir(experiment)
    if model_name == "none":
        return os.path.join(exp_base, data_type, "apenas_ml", "baseline")
    return os.path.join(exp_base, data_type, "ml_com_llm", model_name)


def get_comparison_dir(data_type: str, experiment: str = "default") -> str:
    """Retorna o diretório raiz de um tipo de dado (para comparações)."""
    return os.path.join(get_experiment_dir(experiment), data_type)


def get_dataset_paths(data_type: str) -> dict:
    """Retorna {"MCAR": ..., "MAR": ..., "MNAR": ...} para o tipo de dado."""
    if data_type not in DATASET_PATHS:
        raise ValueError(f"Tipo de dado inválido: '{data_type}'. Use 'sintetico' ou 'real'.")
    return DATASET_PATHS[data_type]


def find_result_dirs(data_type: str, experiment: str = "default") -> list[tuple[str, str, str]]:
    """Encontra todos os diretórios com relatorio.txt dentro de um data type.

    Retorna lista de (display_name, dir_path, abordagem) onde abordagem é
    'apenas_ml' ou 'ml_com_llm'.
    """
    base = get_comparison_dir(data_type, experiment)
    results = []

    # Procura em apenas_ml/baseline/
    apenas_ml_dir = os.path.join(base, "apenas_ml", "baseline")
    if os.path.isdir(apenas_ml_dir) and os.path.exists(os.path.join(apenas_ml_dir, "relatorio.txt")):
        results.append(("baseline (apenas ML)", apenas_ml_dir, "apenas_ml"))

    # Procura em ml_com_llm/*/
    ml_llm_base = os.path.join(base, "ml_com_llm")
    if os.path.isdir(ml_llm_base):
        for d in sorted(os.listdir(ml_llm_base)):
            dir_path = os.path.join(ml_llm_base, d)
            if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "relatorio.txt")):
                results.append((f"{d} (ML + LLM)", dir_path, "ml_com_llm"))

    return results


def save_experiment_config(experiment: str, description: str = "") -> None:
    """Salva experiment_config.json no diretório do experimento (se não existir)."""
    import json
    from datetime import datetime

    exp_dir = get_experiment_dir(experiment)
    config_path = os.path.join(exp_dir, "experiment_config.json")

    if os.path.exists(config_path):
        return

    os.makedirs(exp_dir, exist_ok=True)
    config = {
        "experiment": experiment,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": description,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
