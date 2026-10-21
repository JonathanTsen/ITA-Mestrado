"""Testes do passo 31: filtros, famílias (automáticas e manuais), representantes e papéis."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.sources import (
    MECHDETECT_IDS,
    assign_families,
    build_from_local,
    candidate_pairs,
    id_like,
    profile_table,
)

CSV_DIR = Path(os.environ.get("MECHDETECT_CSV_DIR", "/tmp/rb_mechdetect_run/mech-detect-datasets"))
OVERRIDES = Path(__file__).resolve().parents[2] / "data_rb" / "family_overrides.csv"


def _frame(n: int, seed: int, columns: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(size=n).round(3) for c in columns})


def test_filters_and_id_columns():
    frame = _frame(600, 0, ["a", "b", "c"])
    frame["row_id"] = np.arange(600)
    profile = profile_table(1, frame, "test", "x")
    assert profile.included and profile.id_like_columns == "row_id" and profile.n_numeric_nonconstant == 3
    small = profile_table(2, _frame(100, 0, ["a", "b", "c"]), "test", "x")
    assert not small.included and "n_rows<500" in small.exclusion_reason
    with_nan = _frame(600, 0, ["a", "b", "c"])
    with_nan.iloc[0, 0] = np.nan
    assert "has_missing" in profile_table(3, with_nan, "test", "x").exclusion_reason
    assert id_like(pd.DataFrame({"user_id": [1, 2], "valid": [1, 1]})) == ["user_id"]


def test_subset_version_is_merged_and_distinct_tables_are_not():
    base = _frame(800, 1, ["x1", "x2", "x3", "x4"])
    subset = base.iloc[:500, :3].copy()  # versão derivada: menos linhas e colunas
    other = _frame(800, 2, ["x1", "x2", "x3", "x4"])  # mesmos nomes genéricos, outros valores
    pairs = candidate_pairs({10: base, 11: subset, 12: other})
    merged = {(r.id_a, r.id_b) for r in pairs.itertuples() if r.auto_merge}
    assert (10, 11) in merged and (10, 12) not in merged and (11, 12) not in merged


def test_manual_merge_and_split():
    pairs = pd.DataFrame([{"id_a": 1, "id_b": 2, "auto_merge": True}, {"id_a": 3, "id_b": 4, "auto_merge": False}])
    overrides = pd.DataFrame(
        [
            {"id_a": 3, "id_b": 4, "action": "merge", "reason": "r"},
            {"id_a": 1, "id_b": 2, "action": "split", "reason": "r"},
        ]
    )
    families = assign_families([1, 2, 3, 4, 5], pairs, overrides)
    assert families[1] != families[2] and families[3] == families[4] == 3 and families[5] == 5


@pytest.mark.skipif(not CSV_DIR.exists(), reason="CSV do MechDetect não disponíveis")
def test_local_benchmark_counts_and_roles():
    table, _ = build_from_local(CSV_DIR, pd.read_csv(OVERRIDES, comment="#"))
    assert len(table) == len(MECHDETECT_IDS) == 101
    assert table[table.included].n_missing.sum() == 0
    used = table[table.representative]
    assert used.family_id.is_unique and (used.role == "dev").sum() == 8
    assert set(table.loc[~table.representative, "role"]) == {"not_used"}
