"""Diretórios de run, `run.json` e gravação separada de observado × oráculo (passo 30 do plano).

Layout de um run (`results_rb/<data>-<hash da config>/`, nunca sobrescrito):

    run.json                      commits, sha256 do uv.lock, config, sementes, versões, máquina, data
    contexts.parquet              uma linha por contexto (e bases.parquet, com os relatórios de elegibilidade)
    instances.parquet             sample_id, context_id, family_id, observed_path, realized_rate — sem mecanismo
    observed/<sample_id>.parquet  coluna `y` (NaN nas linhas ausentes) + auxiliares; o caminho não contém a classe
    oracle.parquet                mecanismo, gerador, variante, determinismo, taxa realizada, hash da máscara
    oracle_complete/<context_id>.parquet   Y completo do contexto (só o painel oráculo e os testes leem)

As funções de descritores e de baselines observed-only recebem apenas `ObservedInstance`, lida de `observed/`.

Modelo de ameaça dos identificadores: `sample_id` e `context_id` são SHA-256 de especificações conhecidas (inclusive o
mecanismo), então quem tiver o código e a config consegue recuperar o rótulo a partir deles, e o `context_id` revela os
trios irmãos. Eles servem só à contabilidade (checkpoint, pareamento, junção com o oráculo) e nunca podem entrar num
descritor, num prompt de LLM ou em qualquer entrada de método. Os métodos usam só os valores de `y` e `z`.
"""

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from missdetect.rb.config import RunConfig
from missdetect.rb.inject import ObservedInstance, OracleRecord

LABEL_PATTERN = ("MCAR", "MAR", "MNAR")
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "mdatagen", "tab-err", "pyarrow", "joblib")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return out.stdout.strip()


def git_state(repo: Path) -> dict:
    return {
        "path": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
    }


# Trava das etapas confirmatórias (pré-registro §9, emenda P3; ordem do LLM, P21). O pipeline principal roda no commit
# da tag do pré-registro; as etapas do LLM, no commit da tag do adendo. Antes do adendo, só rodam as etapas que não
# calculam predições. Na produção, o pipeline principal roda numa worktree fixada em `prereg-v1`.
PREREG_TAG = "prereg-v1"
LLM_ADDENDUM_TAG = "prereg-v1-llm"
NO_PREDICTION_COMMANDS = frozenset({"inject", "features", "oracle-features"})
LLM_COMMANDS = frozenset({"llm", "stats-llm"})


def confirmatory_guard(repo: Path, command: str) -> None:
    """Recusa uma etapa com papel `confirm` fora do commit da tag certa, com a árvore suja ou antes do adendo LLM."""
    tag = LLM_ADDENDUM_TAG if command in LLM_COMMANDS else PREREG_TAG
    tagged = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    if not tagged:
        raise RuntimeError(f"{command}: a tag {tag} não existe; as etapas confirmatórias só rodam depois dela")
    head = _git(repo, "rev-parse", "HEAD")
    if head != tagged:
        raise RuntimeError(f"{command}: HEAD ({head[:7]}) não é o commit da tag {tag} ({tagged[:7]})")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(
            f"{command}: árvore suja; faça commit ou descarte as mudanças antes de uma etapa confirmatória"
        )
    addendum = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{LLM_ADDENDUM_TAG}^{{commit}}")
    if command not in NO_PREDICTION_COMMANDS | LLM_COMMANDS and not addendum:
        raise RuntimeError(
            f"{command}: calcula predições, então só roda depois da tag {LLM_ADDENDUM_TAG} (pré-registro §4, P21)"
        )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def r_versions() -> dict:
    """Versões do R e do PKLMtest (o PKLM roda via Rscript); vazio se o R não estiver instalado."""
    script = (
        'cat(R.version.string, "|", tryCatch(as.character(packageVersion("PKLMtest")), error=function(e) "ausente"))'
    )
    try:
        out = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, check=True, timeout=60)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    r_version, _, pklm = out.stdout.partition("|")
    return {"R": r_version.strip(), "PKLMtest": pklm.strip()}


def run_metadata(config: RunConfig, command: str) -> dict:
    root = repo_root()
    lock = root / "uv.lock"
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "command": command,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_name": config.name,
        "config_hash": config.config_hash(),
        "config": config.to_dict(),
        "root_seed": config.root_seed,
        "code": git_state(root),
        "plan": git_state(Path(config.plan_repo).expanduser()) if config.plan_repo else None,
        "uv_lock_sha256": sha256_path(lock) if lock.exists() else None,
        "python": platform.python_version(),
        "versions": versions,
        "machine": {"platform": platform.platform(), "processor": platform.processor(), "node": platform.node()},
        "r": r_versions(),
    }


def new_run_dir(config: RunConfig, when: datetime | None = None) -> Path:
    """Cria `<output_root>/<AAAAMMDD>-<nome>-<hash>`; erro se já existir (nunca sobrescreve)."""
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%d")
    run_dir = Path(config.output_root) / f"{stamp}-{config.name}-{config.config_hash()}"
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} já existe; não sobrescrevo resultados.")
    run_dir.mkdir(parents=True)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"{path} já existe; não sobrescrevo.")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_run_log(run_dir: Path, config: RunConfig, command: str, extra: dict | None = None) -> None:
    """`run.json` na criação do run; cada comando posterior acrescenta `run.<comando>.json`."""
    payload = {**run_metadata(config, command), **(extra or {})}
    name = "run.json" if command in ("inject", "synth") else f"run.{command}.json"
    write_json(run_dir / name, payload)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} já existe; não sobrescrevo.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def write_tables_atomically(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Grava todas as tabelas (CSV) ou nenhuma, para que uma etapa interrompida possa ser refeita (C1, 29/10/2026).

    Cada tabela vai primeiro para um arquivo temporário; só quando todas foram gravadas os temporários são renomeados,
    na ordem dada. A última tabela é o **marcador de conclusão**: se ela existe, a etapa terminou e nada é sobrescrito.
    Se não existe, as outras saídas que estiverem no diretório são sobras de uma interrupção entre as renomeações e são
    refeitas."""
    names = list(tables)
    marker = out_dir / names[-1]
    if marker.exists():
        raise FileExistsError(f"{marker} já existe; não sobrescrevo.")
    tmp = {name: out_dir / f".{name}.tmp" for name in names}
    try:
        for name in names:
            tables[name].to_csv(tmp[name], index=False)
        for name in names:
            os.replace(tmp[name], out_dir / name)
    finally:
        for path in tmp.values():
            path.unlink(missing_ok=True)


def observed_path(sample_id: str) -> str:
    path = f"observed/{sample_id}.parquet"
    if any(label in path.upper() for label in LABEL_PATTERN):
        raise ValueError(f"caminho observado contém um rótulo: {path}")
    return path


def write_observed(run_dir: Path, instance: ObservedInstance) -> str:
    rel = observed_path(instance.sample_id)
    frame = instance.z.copy()
    frame.columns = [str(c) for c in frame.columns]
    if "y" in frame.columns:
        raise ValueError("auxiliar chamada 'y' colide com a coluna observada")
    frame.insert(0, "y", instance.y)
    write_parquet(frame, run_dir / rel)
    return rel


def read_observed(run_dir: Path, sample_id: str) -> ObservedInstance:
    frame = pd.read_parquet(run_dir / observed_path(sample_id))
    return ObservedInstance(sample_id=sample_id, y=frame["y"].to_numpy(dtype=float), z=frame.drop(columns="y"))


def write_complete(run_dir: Path, context_id: str, y_complete: np.ndarray) -> None:
    path = run_dir / "oracle_complete" / f"{context_id}.parquet"
    if not path.exists():  # os geradores de um mesmo contexto compartilham Y completo e ordem das linhas
        write_parquet(pd.DataFrame({"y_complete": y_complete}), path)


def read_complete(run_dir: Path, context_id: str) -> np.ndarray:
    return pd.read_parquet(run_dir / "oracle_complete" / f"{context_id}.parquet")["y_complete"].to_numpy(float)


def oracle_row(oracle: OracleRecord) -> dict:
    return asdict(oracle)


def instance_row(instance: ObservedInstance, oracle: OracleRecord, rel_path: str) -> dict:
    """Linha de `instances.parquet`: metadados de controle, sem mecanismo nem gerador (os IDs não entram nos métodos)."""
    return {
        "sample_id": instance.sample_id,
        "context_id": oracle.context_id,
        "family_id": oracle.family_id,
        "observed_path": rel_path,
        "realized_rate": oracle.realized_rate,
    }
