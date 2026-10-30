"""Testes do passo 30: diretório de run sem sobrescrita, run.json e separação física observado × oráculo."""

import json
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.config import parse_config
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.runs import (
    LLM_ADDENDUM_TAG,
    PREREG_TAG,
    append_run_log,
    confirmatory_guard,
    new_run_dir,
    observed_path,
    read_observed,
    write_observed,
    write_parquet,
    write_tables_atomically,
)


def _config(tmp_path):
    return parse_config({"name": "t", "root_seed": 3, "output_root": str(tmp_path / "results")})


def test_run_dir_is_never_overwritten(tmp_path):
    config = _config(tmp_path)
    when = datetime(2026, 10, 22, tzinfo=UTC)
    run_dir = new_run_dir(config, when)
    assert run_dir.name == f"20261022-t-{config.config_hash()}"
    with pytest.raises(FileExistsError):
        new_run_dir(config, when)
    write_parquet(pd.DataFrame({"a": [1]}), run_dir / "x.parquet")
    with pytest.raises(FileExistsError):
        write_parquet(pd.DataFrame({"a": [2]}), run_dir / "x.parquet")


def test_run_json_records_provenance(tmp_path):
    config = _config(tmp_path)
    run_dir = new_run_dir(config)
    append_run_log(run_dir, config, "inject", {"families": [1]})
    payload = json.loads((run_dir / "run.json").read_text())
    for key in ("code", "uv_lock_sha256", "config", "config_hash", "root_seed", "versions", "machine", "created_utc"):
        assert key in payload
    assert payload["code"]["commit"]
    assert payload["families"] == [1]
    with pytest.raises(FileExistsError):
        append_run_log(run_dir, config, "inject")


def test_no_label_in_observed(tmp_path):
    """Teste 6 do passo 30: o arquivo observado só tem y (com NaN) e as auxiliares; o caminho não tem a classe."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    y[:10] = np.nan
    instance = ObservedInstance("abc123", y, pd.DataFrame({"z1": rng.normal(size=50), "z2": rng.normal(size=50)}))
    rel = write_observed(tmp_path, instance)
    assert not any(label in rel.upper() for label in ("MCAR", "MAR", "MNAR"))
    frame = pd.read_parquet(tmp_path / rel)
    assert list(frame.columns) == ["y", "z1", "z2"]
    back = read_observed(tmp_path, "abc123")
    np.testing.assert_array_equal(np.isnan(back.y), np.isnan(y))
    assert set(vars(back)) == {"sample_id", "y", "z"}
    with pytest.raises(ValueError):
        observed_path("MNAR-1")


class _Boom:
    def to_csv(self, *args, **kwargs):
        raise RuntimeError("falha no meio da gravação")


def test_tables_are_written_all_or_nothing(tmp_path):
    """C1 (29/10): uma etapa interrompida não deixa saída pela metade; o último arquivo marca a conclusão."""
    good = pd.DataFrame({"a": [1]})
    with pytest.raises(RuntimeError):
        write_tables_atomically(tmp_path, {"x.csv": good, "y.csv": _Boom(), "marker.csv": good})
    assert not list(tmp_path.iterdir())  # nem arquivo final nem temporário
    write_tables_atomically(tmp_path, {"x.csv": good, "marker.csv": good})
    assert {p.name for p in tmp_path.iterdir()} == {"x.csv", "marker.csv"}
    with pytest.raises(FileExistsError):
        write_tables_atomically(tmp_path, {"x.csv": good, "marker.csv": good})


def test_leftovers_of_an_interrupted_rename_are_redone(tmp_path):
    """Sem o marcador (último arquivo), saídas que já existem são sobras de uma interrupção e são refeitas."""
    (tmp_path / "x.csv").write_text("parcial", encoding="utf-8")
    write_tables_atomically(tmp_path, {"x.csv": pd.DataFrame({"a": [1]}), "marker.csv": pd.DataFrame({"a": [2]})})
    assert pd.read_csv(tmp_path / "x.csv").a.tolist() == [1]
    assert pd.read_csv(tmp_path / "marker.csv").a.tolist() == [2]


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)  # fmt: skip


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("1", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "c1")
    return repo


def test_confirmatory_steps_need_the_prereg_tag_on_a_clean_head(tmp_path):
    """C3 (pré-registro §9, P3): as etapas `confirm` só rodam no commit da tag, com a árvore limpa."""
    repo = _repo(tmp_path)
    with pytest.raises(RuntimeError, match="não existe"):
        confirmatory_guard(repo, "inject")
    _git(repo, "tag", PREREG_TAG)
    confirmatory_guard(repo, "inject")
    (repo / "a.txt").write_text("2", encoding="utf-8")
    with pytest.raises(RuntimeError, match="árvore suja"):
        confirmatory_guard(repo, "features")
    _git(repo, "commit", "-q", "-am", "c2")
    with pytest.raises(RuntimeError, match="não é o commit da tag"):
        confirmatory_guard(repo, "inject")


def test_prediction_steps_wait_for_the_llm_addendum(tmp_path):
    """P21: antes da tag do adendo LLM, só rodam as etapas sem predição (inject, features, oracle-features)."""
    repo = _repo(tmp_path)
    _git(repo, "tag", PREREG_TAG)
    for command in ("inject", "features", "oracle-features"):
        confirmatory_guard(repo, command)
    for command in ("baselines", "evaluate", "cross-generator", "stats", "secondary"):
        with pytest.raises(RuntimeError, match=LLM_ADDENDUM_TAG):
            confirmatory_guard(repo, command)
    _git(repo, "tag", LLM_ADDENDUM_TAG)
    confirmatory_guard(repo, "evaluate")


def test_llm_steps_run_on_the_addendum_commit(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, "tag", PREREG_TAG)
    (repo / "b.txt").write_text("llm", encoding="utf-8")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "adendo")
    _git(repo, "tag", LLM_ADDENDUM_TAG)
    confirmatory_guard(repo, "stats-llm")
    with pytest.raises(RuntimeError, match="não é o commit da tag"):
        confirmatory_guard(repo, "evaluate")  # o pipeline principal roda no commit do prereg-v1
