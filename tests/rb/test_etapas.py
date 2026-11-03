"""Execução em duas etapas (emenda P25, decisão do autor de 03/11/2026): o sorteio congelado e as configurações."""

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

from missdetect.rb.config import load_config

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "revisao_banca"))  # o script do sorteio fica fora do pacote
etapas = importlib.import_module("etapas")


def test_the_frozen_split_is_the_seeded_draw_and_covers_everything_once(monkeypatch):
    monkeypatch.chdir(REPO)
    recorded = json.loads((REPO / "data_rb/etapas/etapas.json").read_text())
    split = etapas.sortear()
    assert split == {k: recorded[k] for k in ("benchmark", "h7")}
    bench, h7 = split["benchmark"], split["h7"]
    effective = json.loads((REPO / "data_rb/freeze_v1_addendum.json").read_text())["effective_families"]
    assert sorted(bench["etapa1"] + bench["etapa2"]) == sorted(effective) and len(bench["etapa1"]) == 25
    assert len(set(bench["etapa1"]) & set(bench["painel_completo"])) == 15
    assert sorted(h7["etapa1"] + h7["etapa2"]) == list(range(990201, 990231)) and len(h7["etapa1"]) == 15


def test_stage_configs_differ_from_the_frozen_ones_only_in_the_source_table(monkeypatch):
    monkeypatch.chdir(REPO)
    recorded = json.loads((REPO / "data_rb/etapas/etapas.json").read_text())
    for name, key in (("inject-v1", "benchmark"), ("contexto-h7", "h7")):
        frozen = load_config(REPO / f"configs/rb/{name}.yaml").to_dict()
        for stage in (1, 2):
            config = load_config(REPO / f"configs/rb/{name}-etapa{stage}.yaml")
            data = config.to_dict()
            assert data["sources"].pop("table") != frozen["sources"]["table"]
            assert {**data, "sources": {**data["sources"]}} == {
                **frozen,
                "sources": {k: v for k, v in frozen["sources"].items() if k != "table"},
            }
            table = pd.read_csv(config.sources.table)
            assert sorted(table.family_id.unique()) == recorded[key][f"etapa{stage}"]
    assert load_config(REPO / "configs/rb/inject-v1.yaml").config_hash() == "52ed6f93a2a0"  # o freeze não muda
