"""Configuração declarativa dos runs (passo 30 do plano).

Um arquivo YAML por run em `configs/rb/`. A leitura valida as chaves (chave desconhecida → erro), converte para
dataclasses imutáveis e calcula o hash da configuração canônica, que entra no nome do diretório do run
(`results_rb/<data>-<hash>/`) e no `run.json`.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from missdetect.rb.ids import stable_id


@dataclass(frozen=True)
class SourcesConfig:
    table: str = "data_rb/sources_local.csv"
    csv_dir: str = ""
    roles: tuple[str, ...] = ("dev",)
    sentinel_exclusions: str = ""  # CSV openml_id,column,...: colunas que codificam "não medido" (excluídas da tabela)


@dataclass(frozen=True)
class SecondaryGenerator:
    generator: str
    n: int = 1000


@dataclass(frozen=True)
class InjectConfig:
    n_grid: tuple[int, ...] = (500, 1000, 2000)
    k_grid: tuple[int, ...] = (2, 4, 8)
    rate_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    beta_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, float("inf"))
    directions: tuple[str, ...] = ("lower", "upper", "both")
    driver_rules: tuple[str, ...] = ("maxcorr", "random")
    max_base_attempts: int = 20
    secondary: tuple[SecondaryGenerator, ...] = ()


@dataclass(frozen=True)
class SynthConfig:
    n_bases: int = 1800
    n_grid: tuple[int, ...] = (500, 1000, 2000)
    k_grid: tuple[int, ...] = (2, 4, 8)
    rate_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    beta_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, float("inf"))
    directions: tuple[str, ...] = ("lower", "upper", "both")
    latent_r2_max: float = 0.9
    marginals: tuple[str, ...] = ("normal", "lognormal", "exponential", "beta", "uniform", "mixture", "discrete")
    generator_shares: tuple[tuple[str, float], ...] = (("symmetric", 4.0), ("mdatagen", 1.0), ("tab_err", 1.0))
    max_attempts_per_base: int = 20


@dataclass(frozen=True)
class FeaturesConfig:
    n_jobs: int = 8


@dataclass(frozen=True)
class EvaluateConfig:
    learner: str = "hgb"
    descriptive_learners: tuple[str, ...] = ("logreg",)
    routings: tuple[str, ...] = ("hard", "threshold", "soft", "fullprob")
    n_splits: int = 5
    n_repeats: int = 3


@dataclass(frozen=True)
class BaselinesConfig:
    alpha: float = 0.05
    mechdetect_cv_folds: int = 10
    pklm_num_proj: int = 100
    pklm_nrep: int = 200
    pklm_max_instances: int | None = None
    n_jobs: int = 8


@dataclass(frozen=True)
class RunConfig:
    name: str
    root_seed: int
    output_root: str = "results_rb"
    plan_repo: str = ""
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)

    def to_dict(self) -> dict:
        return _plain(dataclasses.asdict(self))

    def config_hash(self) -> str:
        return stable_id(self.to_dict(), length=12)


VALID_GENERATORS = {"symmetric", "mdatagen", "tab_err"}
VALID_DIRECTIONS = {"lower", "upper", "both"}
VALID_DRIVER_RULES = {"maxcorr", "random"}
VALID_ROUTINGS = {"hard", "threshold", "soft", "fullprob"}
VALID_ROLES = {"dev", "confirm"}


def _plain(value: Any) -> Any:
    """Tuplas → listas e inf → "inf", para JSON e hash estáveis."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    return value


def _tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple(v) for v in value)
    return value


def _build(cls: type, data: dict | None, where: str) -> Any:
    data = data or {}
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{where}: chaves desconhecidas {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        if key in ("secondary",):
            kwargs[key] = tuple(_build(SecondaryGenerator, item, f"{where}.{key}") for item in value)
        elif key == "generator_shares":
            kwargs[key] = tuple((str(k), float(v)) for k, v in value.items())
        elif isinstance(value, list):
            kwargs[key] = tuple(float(v) if key.endswith(("rate_grid", "beta_grid")) else v for v in value)
        else:
            kwargs[key] = _tuple(value)
    return cls(**kwargs)


def _validate(config: RunConfig) -> None:
    inj, syn, ev = config.inject, config.synth, config.evaluate
    for name, grid in (("inject.rate_grid", inj.rate_grid), ("synth.rate_grid", syn.rate_grid)):
        if not all(0 < r < 1 for r in grid):
            raise ValueError(f"{name}: taxas devem estar em (0, 1)")
    for name, grid in (("inject.beta_grid", inj.beta_grid), ("synth.beta_grid", syn.beta_grid)):
        if not all(b > 0 for b in grid):
            raise ValueError(f"{name}: beta deve ser > 0 (beta = 0 é o MCAR)")
    for name, dirs in (("inject.directions", inj.directions), ("synth.directions", syn.directions)):
        if not set(dirs) <= VALID_DIRECTIONS:
            raise ValueError(f"{name}: direções válidas são {sorted(VALID_DIRECTIONS)}")
    if not set(inj.driver_rules) <= VALID_DRIVER_RULES:
        raise ValueError(f"inject.driver_rules: regras válidas são {sorted(VALID_DRIVER_RULES)}")
    gens = {s.generator for s in inj.secondary} | {g for g, _ in syn.generator_shares}
    if not gens <= VALID_GENERATORS:
        raise ValueError(f"geradores válidos são {sorted(VALID_GENERATORS)}")
    if not set(ev.routings) <= VALID_ROUTINGS:
        raise ValueError(f"evaluate.routings: válidos são {sorted(VALID_ROUTINGS)}")
    if not set(config.sources.roles) <= VALID_ROLES:
        raise ValueError(f"sources.roles: válidos são {sorted(VALID_ROLES)}")


def parse_config(data: dict) -> RunConfig:
    data = dict(data)
    sections = {
        "sources": SourcesConfig,
        "inject": InjectConfig,
        "synth": SynthConfig,
        "features": FeaturesConfig,
        "evaluate": EvaluateConfig,
        "baselines": BaselinesConfig,
    }
    top = {f.name for f in dataclasses.fields(RunConfig)}
    unknown = set(data) - top
    if unknown:
        raise ValueError(f"chaves desconhecidas no topo: {sorted(unknown)}")
    for key in ("name", "root_seed"):
        if key not in data:
            raise ValueError(f"chave obrigatória ausente: {key}")
    kwargs = {k: v for k, v in data.items() if k not in sections}
    for key, cls in sections.items():
        if key in data:
            kwargs[key] = _build(cls, data[key], key)
    config = RunConfig(**kwargs)
    _validate(config)
    return config


def load_config(path: Path) -> RunConfig:
    with Path(path).open(encoding="utf-8") as fh:
        return parse_config(yaml.safe_load(fh))
