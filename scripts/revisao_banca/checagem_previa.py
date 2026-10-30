"""Checagem prévia antes de retomar o trabalho ou de rodar uma etapa longa (plano 38c, "Como retomar"; 29/10/2026).

Uso (no ITA-Mestrado):
    uv run --frozen python -I scripts/revisao_banca/checagem_previa.py [--config configs/rb/inject-v1.yaml]

- CSV-fonte do MechDetect: o `/tmp` é apagado quando o Mac reinicia. Se faltar algum, reextrai de
  `~/work/MechDetect/src/auc_roc_scores/datasets.tar.gz` e confere o SHA-256 de cada arquivo contra o registro das fontes.
- Estado do código: HEAD, árvore limpa ou suja e as tags do pré-registro (`prereg-v1`, `prereg-v1-llm`).
- Processos do pipeline em andamento, que disputam a CPU com uma etapa nova.

Sai com código 1 se os CSV não puderem ser garantidos.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from missdetect.rb.config import load_config
from missdetect.rb.runs import LLM_ADDENDUM_TAG, PREREG_TAG, git_state, repo_root
from missdetect.rb.sources import ensure_source_csvs

TARBALL = Path.home() / "work" / "MechDetect" / "src" / "auc_roc_scores" / "datasets.tar.gz"
PROCESSES = "missdetect.rb.cli|d4_pklm_tamanho|d7_curva"


def _run(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/rb/inject-v1.yaml"))
    args = parser.parse_args()
    root = repo_root()
    config = load_config(args.config)
    sources = pd.read_csv(root / config.sources.table)
    try:
        extracted = ensure_source_csvs(Path(config.sources.csv_dir), TARBALL, sources)
        print(f"CSV-fonte: ok ({len(sources)} conferidos pelo SHA-256; {extracted} reextraídos)")
    except (ValueError, FileNotFoundError) as exc:
        print(f"CSV-fonte: PROBLEMA: {exc}")
        return 1
    state = git_state(root)
    print(f"código: HEAD {state['commit'][:7]} ({state['branch']}), árvore {'SUJA' if state['dirty'] else 'limpa'}")
    for tag in (PREREG_TAG, LLM_ADDENDUM_TAG):
        commit = _run("git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
        print(f"tag {tag}: {commit[:7] if commit else 'ainda não existe'}")
    running = _run("pgrep", "-fl", PROCESSES)
    print("processos do pipeline em andamento:", running if running else "nenhum")
    return 0


if __name__ == "__main__":
    sys.exit(main())
