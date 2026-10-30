"""Testes do passo 30: diretório de run sem sobrescrita, run.json e separação física observado × oráculo."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.config import parse_config
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.runs import (
    append_run_log,
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
