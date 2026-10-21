"""PKLM (teste MCAR não paramétrico; Spohn et al., 2023) pelo pacote R `PKLMtest`, chamado via `Rscript` (passo 39).

As instâncias observadas são exportadas como CSV (coluna `y` com NA + auxiliares) e processadas em lotes por
processos R paralelos (`pklm.R`). `num.proj` e `nrep` vêm da configuração e ficam no `run.json`. Uma instância que
falha no R volta com `status = "failed: ..."` e p-valor NaN (contada como erro na avaliação, nunca descartada).

Uso só na tarefa L1 (rejeita MCAR ou não).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from missdetect.rb.runs import read_observed

R_SCRIPT = Path(__file__).with_name("pklm.R")


def rscript() -> str:
    path = shutil.which("Rscript")
    if path is None:
        raise RuntimeError("Rscript não encontrado no PATH")
    return path


def run_pklm(run_dir: Path, sample_ids: list[str], num_proj: int, nrep: int, seed: int, n_jobs: int) -> pd.DataFrame:
    """p-valores do PKLM para `sample_ids`, em `n_jobs` processos R."""
    if not sample_ids:
        return pd.DataFrame(columns=["sample_id", "pklm_p", "seconds", "status"])
    with tempfile.TemporaryDirectory(prefix="rb_pklm_") as tmp:
        tmp_dir = Path(tmp)
        paths = []
        for sample_id in sample_ids:
            instance = read_observed(run_dir, sample_id)
            frame = instance.z.copy()
            frame.insert(0, "y", instance.y)
            path = tmp_dir / f"{sample_id}.csv"
            frame.to_csv(path, index=False, na_rep="NA")
            paths.append(str(path))
        items = pd.DataFrame({"sample_id": sample_ids, "path": paths})
        procs = []
        for j, chunk in enumerate(np.array_split(np.arange(len(items)), max(1, min(n_jobs, len(items))))):
            listing, output = tmp_dir / f"list_{j}.csv", tmp_dir / f"out_{j}.csv"
            items.iloc[chunk].to_csv(listing, index=False)
            cmd = [rscript(), str(R_SCRIPT), str(listing), str(output), str(num_proj), str(nrep), str(seed)]
            procs.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE), output))
        frames = []
        for proc, output in procs:
            _, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Rscript falhou ({proc.returncode}): {err.decode()[-2000:]}")
            frames.append(pd.read_csv(output))
    return pd.concat(frames, ignore_index=True)
