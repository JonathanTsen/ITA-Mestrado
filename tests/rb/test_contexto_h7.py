"""H7 confirmatória (adendo LLM, §10; decisão do autor de 03/11/2026): bases, travas e análise, sem chamar a API."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from missdetect.rb import cli
from missdetect.rb.config import load_config
from missdetect.rb.llm.cache import CLASSES
from missdetect.rb.llm.client import AuthorizationRequired

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "revisao_banca" / "llm"
sys.path.insert(0, str(SCRIPTS))  # os scripts do experimento importam uns aos outros pelo diretório
h7 = importlib.import_module("contexto_h7")
dominios = importlib.import_module("contexto_h7_dominios")
sintetico = importlib.import_module("contexto_sintetico")
REPO = Path(__file__).resolve().parents[2]


def test_the_30_domains_are_new_and_well_formed():
    dev = [*sintetico.DOMAINS, *sintetico.DOMAINS_2]
    ids = [d.source_id for d in dominios.DOMAINS_H7]
    assert ids == list(range(990201, 990231))
    assert not set(ids) & {d.source_id for d in dev}
    assert not {d.name for d in dominios.DOMAINS_H7} & {d.name for d in dev}
    for domain in dominios.DOMAINS_H7:
        table = domain.generate(np.random.default_rng([sintetico.GENERATOR_SEED, domain.source_id]), sintetico.N_ROWS)
        assert list(table.columns) == list(domain.columns) and len(table) == 1000, domain.name
        assert np.isfinite(table.to_numpy(float)).all(), domain.name
        assert table.nunique().min() >= 30, domain.name  # contínuas, como os 16 de desenvolvimento
        sintetico.describe(domain, table)  # falha se o contexto falar de dados faltantes


def test_gerar_is_deterministic_and_marks_the_bases_confirmatory(tmp_path):
    h7.gerar(tmp_path / "a")
    h7.gerar(tmp_path / "b")
    a, b = (json.loads((tmp_path / x / "manifest.json").read_text()) for x in ("a", "b"))
    assert a["files"] == b["files"] and len(a["files"]) == 32 and a["role"] == "confirm"
    sources = pd.read_csv(tmp_path / "a" / "sources.csv")
    assert set(sources.role) == {"confirm"} and len(sources) == 30


def test_the_committed_config_is_confirmatory_and_points_to_the_frozen_bases():
    config = load_config(REPO / h7.CONFIG)
    assert tuple(config.sources.roles) == ("confirm",) and config.name == "contexto-h7"
    assert Path(config.sources.csv_dir) == Path("data_rb/contexto_h7/csv")


def test_panel_has_18_instances_per_base_and_both_arms_see_the_same_rows(tmp_path, monkeypatch):
    """Duas bases, papel `dev` e outra semente: confere o desenho sem gerar as instâncias confirmatórias."""
    data = tmp_path / "data"
    sintetico.gerar_bases(data, dominios.DOMAINS_H7[:2], "teste", "dev", h7.DOMAINS_FILE)
    raw = yaml.safe_load((REPO / h7.CONFIG).read_text())
    raw |= {"name": "h7-teste", "root_seed": 7, "output_root": str(tmp_path / "results")}
    raw["sources"] |= {"table": str(data / "sources.csv"), "csv_dir": str(data / "csv"), "roles": ["dev"]}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(REPO)
    assert cli.main(["inject", "--config", str(config_path)]) == 0
    run_dir = next((tmp_path / "results").iterdir())
    panels = sintetico._panel_with_keys(run_dir, load_config(config_path), h7.ARMS, "gemini-3.8-flash")
    s0, s7 = panels["S0"], panels["S7"]
    assert len(s0) == len(s7) == 36 and list(s0.sample_id) == list(s7.sample_id)
    assert set(s0.groupby("family_id").size()) == {18}
    assert set(s0.groupby(["family_id", "mechanism"]).size()) == {6}
    assert not set(s0.key) & set(s7.key)


def test_perguntar_goes_through_the_guard_and_the_authorization_before_any_call(monkeypatch):
    calls = []

    def guard(repo, command):
        calls.append(command)
        raise RuntimeError("trava")

    monkeypatch.setattr(h7, "confirmatory_guard", guard)
    monkeypatch.chdir(REPO)
    with pytest.raises(RuntimeError, match="trava"):
        h7.perguntar(Path("x"), REPO / h7.CONFIG, "gemini-3.8-flash", None, budget_only=True)
    assert calls == ["llm"]
    monkeypatch.setattr(h7, "confirmatory_guard", lambda repo, command: None)
    with pytest.raises(AuthorizationRequired):
        h7.perguntar(Path("x"), REPO / h7.CONFIG, "gemini-4-argon", None, budget_only=True)


def _predictions(rng: np.random.Generator, accuracy_by_family: dict[int, float], arm: str) -> pd.DataFrame:
    rows = []
    for family, accuracy in accuracy_by_family.items():
        for i in range(18):
            y = i % 3
            hit = rng.random() < accuracy
            pred = y if hit else (y + 1 + rng.integers(0, 2)) % 3
            p = np.full(3, 0.1)
            p[pred] = 0.8
            rows.append({"method": f"llm_{arm.lower()}", "repeat": 0, "sample_id": f"{family}-{i}", "family_id": family,
                         "y_true": y, "y_pred": pred, "status": "ok",
                         **dict(zip([f"p_{c}" for c in CLASSES], p, strict=True))})  # fmt: skip
    return pd.DataFrame(rows)


def test_analysis_is_the_per_base_wilcoxon_pratt_with_hl_and_reads_null_results_honestly():
    rng = np.random.default_rng(0)
    families = range(990201, 990231)
    s0 = _predictions(rng, dict.fromkeys(families, 0.45), "S0")
    s7 = _predictions(rng, dict.fromkeys(families, 0.75), "S7")
    summary, by_family = h7.analisar_h7(s0, s7)
    assert summary["n_bases"] == 30 and summary["n_instancias"] == 540 and len(by_family) == 30
    assert summary["p_wilcoxon_pratt"] < 0.05 and summary["hodges_lehmann"] > 0.15
    assert summary["ic95_baixo"] < summary["hodges_lehmann"] < summary["ic95_alto"]
    assert summary["leitura"] == "S7 acima de S0"
    assert by_family.d.to_numpy() == pytest.approx(by_family.macro_recall_S7 - by_family.macro_recall_S0)
    same = h7.analisar_h7(s0, s0.assign(method="llm_s7"))[0]
    assert same["p_wilcoxon_pratt"] == 1.0 and same["leitura"].startswith("evidência insuficiente")
    with pytest.raises(ValueError, match="mesmas instâncias"):
        h7.analisar_h7(s0, s7.iloc[1:])


def test_h7_counts_floating_point_residue_as_a_tie():
    """Desvio D4: base com o mesmo acerto nos dois braços é empate, mesmo com resíduo de ponto flutuante."""
    rng = np.random.default_rng(1)
    s0 = _predictions(rng, dict.fromkeys(range(990201, 990211), 0.5), "S0")
    s7 = s0.assign(method="llm_s7")
    summary, by_family = h7.analisar_h7(s0, s7)
    assert (by_family.d == 0).all() and summary["bases_empate"] == 10
    assert summary["bases_S7_melhor"] == summary["bases_S7_pior"] == 0
