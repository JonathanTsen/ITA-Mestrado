"""Testes do passo 34 (e testes 4 e 5 do passo 30): oráculo lacrado, invariância a linhas/colunas, k e indefinidos."""

import time

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.features import compute, schema
from missdetect.rb.inject import ObservedInstance, build_instance, context_rows, make_context, sample_base

ROOT = 7


def _table(n: int = 2200, k: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    y = 0.7 * z[:, 0] + 0.4 * z[:, 1] + rng.normal(scale=0.7, size=n)
    frame = pd.DataFrame(z, columns=[f"z{j}" for j in range(k)])
    frame["y"] = y
    return frame.round(5)


def _instances(table: pd.DataFrame, n: int = 600, k: int = 4, beta: float = np.inf):
    base, report = sample_base(table, list(table.columns), 1, 1, 0, ROOT, driver_rule="maxcorr", n_max=n)
    assert base is not None, report
    context = make_context(base, n, k, 0.3, beta, "upper", ROOT)
    sub = context_rows(table, base, context)
    return sub, context, {m: build_instance(sub, context, "symmetric", m) for m in ("MCAR", "MAR", "MNAR")}


def _assert_same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        if key == "sample_id":
            continue
        va, vb = a[key], b[key]
        if isinstance(va, float) and np.isnan(va):
            assert isinstance(vb, float) and np.isnan(vb), key
        else:
            assert va == pytest.approx(vb, abs=1e-9), key


def test_invariant_to_row_and_column_permutation_and_names():
    _, _, built = _instances(_table())
    obs = built["MAR"][0]
    rng = np.random.default_rng(3)
    rows = rng.permutation(len(obs.y))
    cols = list(rng.permutation(obs.z.columns))
    shuffled = ObservedInstance(
        sample_id=obs.sample_id,
        y=obs.y[rows],
        z=obs.z.iloc[rows][cols].set_axis([f"c{j}" for j in range(len(cols))], axis=1).reset_index(drop=True),
    )
    _assert_same(compute(obs), compute(shuffled))


def test_oracle_sealed_values_at_missing_rows_do_not_matter():
    """Teste 4 do passo 30: trocar Y nas linhas ausentes da tabela completa não muda nenhum descritor."""
    sub, context, built = _instances(_table())
    _, _, mask = built["MNAR"]
    altered = sub.copy()
    order = np.random.default_rng(context.shuffle_seed).permutation(context.n)
    original_rows = order[mask]  # linhas (na ordem da subamostra) que ficaram ausentes
    altered.loc[original_rows, context.target] = 1e6
    again = build_instance(altered, context, "symmetric", "MNAR")
    assert np.array_equal(np.isnan(again[0].y), mask)
    _assert_same(compute(built["MNAR"][0]), compute(again[0]))


def test_works_for_any_number_of_auxiliaries():
    table = _table()
    keys = {k: set(compute(_instances(table, k=k)[2]["MAR"][0])) for k in (1, 2, 8)}
    assert keys[1] == keys[2] == keys[8]


def test_undefined_values_are_nan_and_counted():
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    y[:3] = np.nan  # só 3 ausentes: AUCs com CV ficam indefinidos
    out = compute(ObservedInstance("u", y, pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])))
    assert np.isnan(out["g2_cv_auc_boosting"]) and out["status_undefined"] > 0


def test_descriptors_carry_the_intended_signal():
    """Sanidade (não é critério de aceite): com beta = inf, G2 separa MAR de MCAR e G1 desloca a forma sob MNAR."""
    table = _table(seed=1)
    out = {m: compute(inst[0]) for m, inst in _instances(table, n=1000)[2].items()}
    assert out["MAR"]["g2_cv_auc_boosting"] > out["MCAR"]["g2_cv_auc_boosting"] + 0.2
    assert out["MNAR"]["g1_skewness"] < out["MCAR"]["g1_skewness"]  # cauda superior removida


def test_schema_and_cost():
    names = [s["name"] for s in schema()]
    assert len(names) == len(set(names)) and all(n.startswith(("g0", "g1", "g2", "g3", "g4")) for n in names)
    _, _, built = _instances(_table(n=2200, k=8), n=2000, k=8)
    start = time.perf_counter()
    compute(built["MNAR"][0])
    assert time.perf_counter() - start < 20  # meta do plano: < 2 s; medido e registrado em execucao/34-features.md
