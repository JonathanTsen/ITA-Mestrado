"""
Parsing centralizado de argumentos CLI do pipeline v2.
"""
import sys


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
