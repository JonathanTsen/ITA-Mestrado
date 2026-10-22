"""Testes do passo 45: estudo de caso com ausência natural, sem fabricar mecanismo (IND-22) e sem métrica de acerto."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.case_study import auxiliary_columns, build_case, parse_registry


def _table(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "patient_id": np.arange(n),  # identificador
            "age": rng.normal(50, 10, n).round(1),
            "sex": rng.choice(["F", "M"], n),  # binária
            "stage": rng.choice(["I", "II", "III"], n),  # categórica com mais de 2 níveis
            "const": 1.0,
            "lab": rng.normal(size=n),
            "partial": rng.normal(size=n),
            "y": rng.normal(10, 2, n),
        }
    )
    frame.loc[rng.choice(n, 30, replace=False), "y"] = np.nan
    frame.loc[rng.choice(n, 10, replace=False), "partial"] = np.nan
    return frame


def _spec(**kw) -> dict:
    base = {"case_id": "t", "source": "t", "target": "y", "missing_rule": "natural", "z_include": "",
            "z_exclude": "", "hypothesis": "MAR", "evidence": "x", "citation": "x", "include": True,
            "exclusion_reason": ""}  # fmt: skip
    return base | kw


def test_auxiliaries_are_complete_numeric_or_binary_columns():
    cols = auxiliary_columns(_table(), "y", exclude=[])
    assert cols == ["age", "sex", "lab"]  # sem id, constante, categórica de 3 níveis, coluna com NaN e o alvo


def test_auxiliaries_respect_exclusions():
    assert auxiliary_columns(_table(), "y", exclude=["lab"]) == ["age", "sex"]


def test_natural_case_keeps_every_observed_value_and_every_missing_one():
    table = _table()
    instance, meta = build_case(table, _spec())
    source = table["y"].to_numpy()
    assert np.array_equal(np.isnan(instance.y), np.isnan(source))
    assert np.array_equal(instance.y[~np.isnan(source)], source[~np.isnan(source)])
    assert list(instance.z.columns) == ["age", "sex", "lab"]
    assert not instance.z.isna().any().any()
    np.testing.assert_array_equal(instance.z["age"].to_numpy(), table["age"].to_numpy())
    assert set(instance.z["sex"].unique()) == {0.0, 1.0}
    assert meta["n_missing_natural"] == 30 and meta["n_missing_rule"] == 0 and meta["k"] == 3


def test_zero_rule_only_turns_zeros_into_missing():
    table = _table()
    table.loc[:4, "y"] = 0.0
    instance, meta = build_case(table, _spec(missing_rule="zero_impossible"))
    source = table["y"].to_numpy()
    zero = source == 0
    assert np.isnan(instance.y[zero]).all()
    keep = ~zero & ~np.isnan(source)
    assert np.array_equal(instance.y[keep], source[keep])
    assert meta["n_missing_rule"] == int(zero.sum())


def test_below_lod_rule_uses_the_source_comment_code():
    table = _table()
    table["y"] = table["y"].fillna(1.0)
    table["y_code"] = 0.0
    table.loc[:9, "y_code"] = 1.0  # abaixo do limite de detecção (valor de preenchimento na fonte)
    table.loc[10:14, "y_code"] = np.nan  # sem resultado laboratorial: fora da população do estudo
    instance, meta = build_case(table, _spec(missing_rule="below_lod:y_code", z_exclude="y_code"))
    assert len(instance.y) == len(table) - 5
    assert np.isnan(instance.y).sum() == 10
    assert meta["n_rows_dropped"] == 5 and meta["n_missing_rule"] == 10


def test_explicit_auxiliaries_must_be_complete():
    table = _table()
    instance, _ = build_case(table, _spec(z_include="age;lab"))
    assert list(instance.z.columns) == ["age", "lab"]
    with pytest.raises(ValueError, match="partial"):
        build_case(table, _spec(z_include="age;partial"))


def test_non_numeric_target_is_an_error():
    table = _table()
    with pytest.raises(ValueError, match="numérico"):
        build_case(table, _spec(target="stage"))


def test_registry_requires_a_reason_for_every_exclusion(tmp_path):
    path = tmp_path / "registry.csv"
    pd.DataFrame([_spec(case_id="a"), _spec(case_id="b", include=False, exclusion_reason="zero é valor real")]).to_csv(
        path, index=False
    )
    specs = parse_registry(path)
    assert [s["case_id"] for s in specs] == ["a", "b"]
    assert specs[1]["include"] is False
    pd.DataFrame([_spec(case_id="c", include=False)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="motivo"):
        parse_registry(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.assign(y=t["y"].fillna(1.0)), "sem ausências"),
        (lambda t: t.assign(y=np.where(np.arange(len(t)) < 5, 1.0, np.nan)), "observados"),
        (lambda t: t[["patient_id", "stage", "y"]], "nenhuma auxiliar completa"),
    ],
)
def test_unusable_cases_fail_with_the_specific_reason(change, message):
    with pytest.raises(ValueError, match=message):
        build_case(change(_table()), _spec())


def test_versioned_registry_is_valid_and_matches_the_manifest():
    root = Path(__file__).resolve().parents[2] / "data_rb"
    specs = parse_registry(root / "case_study_sources.csv")
    ids = [s["case_id"] for s in specs]
    assert len(ids) == len(set(ids))
    sources = set(pd.read_csv(root / "case_study_manifest.csv").source)
    for spec in specs:
        if spec["include"]:
            assert spec["source"] in sources, spec["case_id"]
            assert spec["missing_rule"] in ("natural", "zero_impossible") or spec["missing_rule"].startswith(
                "below_lod:"
            )
            assert spec["hypothesis"] and spec["evidence"], spec["case_id"]
