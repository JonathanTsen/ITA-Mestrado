"""Testes da renderização cega do LLM (passo 42; pré-registro §4, emenda P20): o prompt não pode carregar o rótulo."""

import json

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.inject import ObservedInstance
from missdetect.rb.llm.render import (
    A1_SCHEMA,
    A1_SYSTEM,
    DESCRIPTION_MAX_CHARS,
    MAX_ROWS,
    NA_TOKEN,
    blind_table,
    llm_rows,
    neutral_description,
    render_a1,
    render_a2,
    render_a3,
    scale_to_int,
)


def _instance(sample_id: str = "s0", n: int = 500, seed: int = 0, names=("preg", "pedi")) -> ObservedInstance:
    rng = np.random.default_rng(seed)
    z = pd.DataFrame({name: rng.normal(size=n) * (j + 1) + 10 * j for j, name in enumerate(names)})
    y = 3.0 * z.iloc[:, 0].to_numpy() + rng.normal(size=n) + 50
    y[rng.random(n) < 0.3] = np.nan
    return ObservedInstance(sample_id=sample_id, y=y, z=z)


def test_scale_to_int_is_increasing_affine_over_observed_values():
    v = np.array([2.0, np.nan, 5.0, 3.5, 8.0])
    out = scale_to_int(v)
    assert np.isnan(out[1])
    np.testing.assert_array_equal(out[[0, 2, 3, 4]], [0, 500, 250, 999])
    assert np.isnan(scale_to_int(np.array([np.nan, np.nan]))).all()
    np.testing.assert_array_equal(scale_to_int(np.array([4.0, 4.0, np.nan]))[:2], [0, 0])


def test_blind_table_shape_names_and_values():
    inst = _instance()
    table = blind_table(inst, seed=7)
    assert len(table) == MAX_ROWS and list(table.columns) == ["c0", "c1", "c2"]
    with_na = [c for c in table.columns if table[c].isna().any()]
    assert len(with_na) == 1
    values = table.to_numpy(dtype=float)
    observed = values[~np.isnan(values)]
    assert observed.min() >= 0 and observed.max() <= 999 and np.all(observed == np.round(observed))
    small = ObservedInstance("s", inst.y[:150], inst.z.iloc[:150])
    assert len(blind_table(small, seed=7)) == 150


def test_prompt_does_not_depend_on_label_ids_names_or_hidden_values():
    """Byte a byte: mudar o rótulo (que muda o sample_id), os nomes das colunas, os valores ausentes de Y ou uma
    transformação afim positiva dos dados não muda o prompt do A1 nem o do A2."""
    base = _instance(sample_id="id-do-mcar")
    reference = render_a1(base, seed=11)
    renamed = ObservedInstance("id-do-mnar", base.y.copy(), base.z.set_axis(["glucose", "insulin"], axis=1))
    assert render_a1(renamed, seed=11) == reference
    affine = ObservedInstance("x", 2.5 * base.y - 40, base.z * 3 + 7)
    assert render_a1(affine, seed=11) == reference
    features = {"g0_missing_rate": 0.3, "g2_cv_auc_boosting": 0.61}
    assert render_a2(features) == render_a2(dict(reversed(features.items())))


def test_prompt_changes_with_the_seed_and_only_with_observed_data():
    inst = _instance()
    assert render_a1(inst, seed=1) != render_a1(inst, seed=2)
    assert render_a1(inst, seed=1) == render_a1(inst, seed=1)
    other_mask = inst.y.copy()
    other_mask[np.flatnonzero(~np.isnan(other_mask))[:20]] = np.nan
    assert render_a1(ObservedInstance("s0", other_mask, inst.z), seed=1) != render_a1(inst, seed=1)


def test_table_text_has_only_generic_tokens():
    inst = _instance(names=("preg", "pedi"))
    text = render_a1(inst, seed=3)
    header, *rows = text.strip().split("\n")
    assert header == "c0,c1,c2"
    allowed = set("0123456789,") | set(NA_TOKEN)
    assert all(set(row) <= allowed for row in rows)
    assert "preg" not in text and "pedi" not in text and " " not in text


@pytest.mark.parametrize("word", ["prior", "example", "%"])
def test_instructions_have_no_priors_or_examples(word):
    assert word not in A1_SYSTEM.lower()


def test_schema_is_only_the_three_probabilities():
    assert A1_SCHEMA["required"] == ["MCAR", "MAR", "MNAR"]
    assert set(A1_SCHEMA["properties"]) == {"MCAR", "MAR", "MNAR"}
    assert A1_SCHEMA["additionalProperties"] is False
    json.dumps(A1_SCHEMA)


def test_llm_rows_are_the_rows_the_blind_table_shows():
    n = 700
    inst = ObservedInstance("s", np.arange(n, dtype=float), pd.DataFrame({"a": np.arange(n) * 2.0}))
    table = blind_table(inst, seed=21)
    rows = llm_rows(n, seed=21)
    assert len(rows) == MAX_ROWS and not table.isna().any().any()
    np.testing.assert_array_equal(np.argsort(table.c0.to_numpy()), np.argsort(rows))


PIMA = """**Author**: [Vincent Sigillito](vgs@aplcen.apl.jhu.edu)

**Source**: [Obtained from UCI](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)

**Please cite**: [UCI citation policy](https://archive.ics.uci.edu/ml/citation_policy.html)

1. Title: Pima Indians Diabetes Database

4. Relevant Information: Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. The diagnostic, binary-valued variable investigated is whether the patient shows signs of diabetes according to World Health Organization criteria.

8. Missing Attribute Values: None. Values were imputed by the mean.
9. Class Distribution: (class value 1 is interpreted as "tested positive for diabetes")"""


def test_neutral_description_drops_identity_and_missingness_words():
    text = neutral_description(PIMA, "diabetes")
    low = text.lower()
    for word in (
        "missing",
        "imput",
        "sigillito",
        "uci",
        "http",
        "@",
        "title",
        "cite",
        "pima indians diabetes database",
    ):
        assert word not in low, word
    assert "females at least 21 years old" in text and len(text) <= DESCRIPTION_MAX_CHARS
    assert (
        neutral_description("Built by the kdd_ipums_la_97-small team. Income data.", "kdd_ipums_la_97-small")
        == "Income data."
    )
    assert neutral_description("", "x") == ""
    for term in ("Censored values.", "Unit nonresponse here.", "Data are MCAR.", "Non-response was low."):
        assert neutral_description(term + " Ok.", "x") == "Ok."


def test_a3_is_the_a1_table_with_real_names_and_a_label_free_description():
    base = _instance(sample_id="id-do-mcar", names=("preg", "pedi"))
    a1 = render_a1(base, seed=11)
    a3 = render_a3(base, seed=11, target_name="age", description="Patients aged 21 or more.")
    table = a3.split("Table:\n", 1)[1]
    assert table.splitlines()[1:] == a1.splitlines()[1:]  # mesmas linhas e valores
    assert sorted(table.splitlines()[0].split(",")) == ["age", "pedi", "preg"]
    renamed_label = ObservedInstance("id-do-mnar", base.y.copy(), base.z.copy())
    assert render_a3(renamed_label, seed=11, target_name="age", description="Patients aged 21 or more.") == a3
    assert "Patients aged 21 or more." in a3 and "NA" in table
