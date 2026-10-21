"""Testes do passo 35 (e testes 2, 3, 7 e 8 do passo 30): partições, pareamento, seleção cega, falhas, métricas."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.evaluate import (
    check_pairing,
    classwise_ece,
    metrics,
    run_method,
    summarize_by_family,
    trivial_factory,
)
from missdetect.rb.models import DirectClassifier, HierarchicalClassifier
from missdetect.rb.splits import inner_splits, outer_splits

ROOT = 11


def _data(n_families: int = 12, per_family: int = 30, seed: int = 0):
    """Descritores sintéticos com sinal: MCAR ~ ruído; MAR desloca x0; MNAR desloca x0 e x1."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_families):
        for _ in range(per_family):
            c = int(rng.integers(3))
            x = rng.normal(size=4)
            x[0] += 1.5 * (c > 0)
            x[1] += 1.5 * (c == 2)
            rows.append((f, c, *x))
    frame = pd.DataFrame(rows, columns=["family", "y", "x0", "x1", "x2", "x3"])
    X = frame[["x0", "x1", "x2", "x3"]].to_numpy()
    sample_ids = pd.Series([f"s{i}" for i in range(len(frame))])
    return X, frame.y.to_numpy(), frame.family, sample_ids


def test_outer_and_inner_splits_are_disjoint_by_family():
    _, _, families, _ = _data()
    seen = {}
    for repeat, _, train, test in outer_splits(families, ROOT):
        assert not set(families.iloc[train]) & set(families.iloc[test])
        seen.setdefault(repeat, []).extend(families.iloc[test].unique())
        tf = families.iloc[train].reset_index(drop=True)
        for itrain, itest in inner_splits(tf, seed=1):
            assert not set(tf.iloc[itrain]) & set(tf.iloc[itest])
    for families_tested in seen.values():  # cada família é testada exatamente uma vez por repetição
        assert sorted(families_tested) == sorted(families.unique())


def test_methods_are_paired_one_to_one():
    X, y, families, ids = _data()
    a, _ = run_method("uniform", trivial_factory("uniform", 0), X, y, families, ids, ROOT)
    b, _ = run_method("majority", trivial_factory("majority", 0), X, y, families, ids, ROOT)
    assert check_pairing(pd.concat([a, b]))


def test_failures_are_counted_as_errors():
    X, y, families, ids = _data()

    class Broken:
        def fit(self, *args):
            raise RuntimeError("falha simulada")

    pred, _ = run_method("broken", Broken, X, y, families, ids, ROOT, n_repeats=1)
    assert len(pred) == len(y) and (pred.y_pred == -1).all() and pred.status.str.startswith("failed").all()
    assert (
        metrics(pred.y_true.to_numpy(), pred[["p_MCAR", "p_MAR", "p_MNAR"]].to_numpy(), pred.y_pred.to_numpy())[
            "accuracy"
        ]
        == 0.0
    )


def test_selection_is_blind_to_test_labels():
    """Permutar os rótulos de uma família não muda as escolhas internas dos folds em que ela está no teste."""
    X, y, families, ids = _data()
    factory = lambda: HierarchicalClassifier("logreg", {}, "threshold", seed=0)  # noqa: E731
    _, base = run_method("h", factory, X, y, families, ids, ROOT, n_repeats=1)
    y2 = y.copy()
    target_family = 3
    idx = np.flatnonzero(families.to_numpy() == target_family)
    y2[idx] = np.random.default_rng(0).permutation(y2[idx])
    _, changed = run_method("h", factory, X, y2, families, ids, ROOT, n_repeats=1)
    for (_, _, _, test), c0, c1 in zip(outer_splits(families, ROOT, n_repeats=1), base, changed, strict=True):
        if target_family in set(families.iloc[test]):
            assert c0 == c1


def test_imputation_is_fit_on_training_data_only():
    """Teste 8 do passo 30: a mediana usada para imputar descritores vem só do treino."""
    X, y, families, _ = _data()
    X = X.copy()
    X[::7, 0] = np.nan
    train = np.arange(200)
    model = DirectClassifier("logreg", {}, seed=0).fit(X[train], y[train], families.iloc[train])
    imputer = model.model_.learner.named_steps["simpleimputer"]
    assert imputer.statistics_[0] == pytest.approx(np.nanmedian(X[train, 0]))
    assert imputer.statistics_[0] != pytest.approx(np.nanmedian(X[:, 0]))


def test_metric_definitions():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y]
    m = metrics(y, proba, y)
    assert m["accuracy"] == m["macro_recall"] == 1.0 and m["brier"] == pytest.approx(0.0)
    uniform = np.full((6, 3), 1 / 3)
    assert metrics(y, uniform, np.zeros(6, dtype=int))["macro_recall"] == pytest.approx(1 / 3)
    assert metrics(y, uniform, np.zeros(6, dtype=int))["brier"] == pytest.approx(2 / 3)
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=20000)
    y_cal = np.array([rng.choice(3, p=row) for row in p])
    assert classwise_ece(p, y_cal) < 0.02  # probabilidades calibradas por construção


def test_end_to_end_direct_and_hierarchical():
    X, y, families, ids = _data(seed=1)
    preds, choices = [], []
    for name, factory in [
        ("direct", lambda: DirectClassifier("logreg", {}, seed=0)),
        ("hier_soft", lambda: HierarchicalClassifier("logreg", {}, "soft", seed=0)),
        ("hier_full", lambda: HierarchicalClassifier("logreg", {}, "fullprob", seed=0)),
    ]:
        p, c = run_method(name, factory, X, y, families, ids, ROOT, n_repeats=1)
        preds.append(p)
        choices.extend(c)
    pred = pd.concat(preds)
    assert check_pairing(pred) and (pred.status == "ok").all()
    _, summary = summarize_by_family(pred)
    assert (summary["macro_recall_mean"] > 0.5).all()
    assert all("zones" in c for c in choices if c["method"] == "hier_soft")
