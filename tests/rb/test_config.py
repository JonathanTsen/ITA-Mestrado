"""Testes do passo 30: configuração declarativa (validação e hash estável)."""

from pathlib import Path

import pytest

from missdetect.rb.config import load_config, parse_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "rb"


def test_versioned_configs_load():
    for path in sorted(CONFIGS.glob("*.yaml")):
        config = load_config(path)
        assert config.name and len(config.config_hash()) == 12


def test_hash_is_stable_and_sensitive():
    a = parse_config({"name": "x", "root_seed": 1, "inject": {"beta_grid": [1.0, float("inf")]}})
    b = parse_config({"root_seed": 1, "name": "x", "inject": {"beta_grid": [1.0, float("inf")]}})
    assert a.config_hash() == b.config_hash()
    c = parse_config({"name": "x", "root_seed": 2, "inject": {"beta_grid": [1.0, float("inf")]}})
    assert a.config_hash() != c.config_hash()
    assert a.to_dict()["inject"]["beta_grid"] == [1.0, "inf"]


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "x"},
        {"name": "x", "root_seed": 1, "extra": 1},
        {"name": "x", "root_seed": 1, "inject": {"rate_grid": [0.0, 0.2]}},
        {"name": "x", "root_seed": 1, "inject": {"beta_grid": [0.0]}},
        {"name": "x", "root_seed": 1, "inject": {"unknown_key": 1}},
        {"name": "x", "root_seed": 1, "inject": {"secondary": [{"generator": "mbov"}]}},
        {"name": "x", "root_seed": 1, "evaluate": {"routings": ["soft3zone"]}},
        {"name": "x", "root_seed": 1, "sources": {"roles": ["test"]}},
    ],
)
def test_invalid_configs_raise(bad):
    with pytest.raises(ValueError):
        parse_config(bad)


def test_secondary_and_shares_are_parsed():
    config = parse_config(
        {
            "name": "x",
            "root_seed": 1,
            "inject": {"secondary": [{"generator": "mdatagen", "n": 500}]},
            "synth": {"generator_shares": {"symmetric": 2, "tab_err": 1}},
        }
    )
    assert config.inject.secondary[0].generator == "mdatagen"
    assert config.inject.secondary[0].n == 500
    assert config.synth.generator_shares == (("symmetric", 2.0), ("tab_err", 1.0))
