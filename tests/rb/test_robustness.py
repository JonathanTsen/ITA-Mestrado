"""Testes dos ensaios de robustez do passo 40 (exploratórios): prior-shift e L1 com MCAR raro."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.config import parse_config
from missdetect.rb.robustness import (
    SCENARIOS,
    expected_accuracy,
    prior_shift,
    rare_mcar,
    run_robustness,
    saerens_adjust,
    simplex_grid,
)


def _pred(method: str = "m", recalls=(0.9, 0.6, 0.3), families=(1, 2, 3, 4), n_per_class: int = 10) -> pd.DataFrame:
    """Predições com recall exato por classe; o erro vai sempre para a classe seguinte."""
    rows = []
    for family in families:
        for c, r in enumerate(recalls):
            hits = round(r * n_per_class)
            for i in range(n_per_class):
                y_pred = c if i < hits else (c + 1) % 3
                proba = np.full(3, 0.1)
                proba[y_pred] = 0.8
                rows.append({"method": method, "repeat": 0, "family_id": family, "sample_id": f"{family}-{c}-{i}",
                             "y_true": c, "y_pred": y_pred, "p_MCAR": proba[0], "p_MAR": proba[1],
                             "p_MNAR": proba[2]})  # fmt: skip
    return pd.DataFrame(rows)


def test_expected_accuracy_is_the_prior_weighted_recall():
    acc = expected_accuracy(_pred(), (0.1, 0.5, 0.4))
    assert acc.accuracy.tolist() == pytest.approx([0.1 * 0.9 + 0.5 * 0.6 + 0.4 * 0.3] * 4)
    assert expected_accuracy(_pred(), (1 / 3, 1 / 3, 1 / 3)).accuracy.iloc[0] == pytest.approx(0.6)


def test_failures_count_as_errors():
    pred = _pred(recalls=(1.0, 1.0, 1.0))
    pred.loc[pred.y_true == 0, "y_pred"] = -1
    assert expected_accuracy(pred, (1.0, 0.0, 0.0)).accuracy.iloc[0] == 0.0


def test_saerens_adjustment_moves_the_decision_with_the_prior():
    proba = np.array([[0.4, 0.35, 0.25]])
    same = saerens_adjust(proba, (1 / 3,) * 3, (1 / 3,) * 3)
    np.testing.assert_allclose(same, proba)
    rare_mcar_prior = saerens_adjust(proba, (1 / 3,) * 3, (0.05, 0.5, 0.45))
    assert rare_mcar_prior.argmax() == 1 and rare_mcar_prior.sum() == pytest.approx(1.0)


def test_prior_shift_table_has_every_scenario_with_both_decisions():
    table = prior_shift(_pred(), ("m",), SCENARIOS)
    assert set(table.scenario) == set(SCENARIOS) and set(table.decision) == {"argmax", "saerens"}
    balanced = table[(table.scenario == "balanceada") & (table.decision == "argmax")]
    assert balanced["mean"].iloc[0] == pytest.approx(0.6) and balanced.n_families.iloc[0] == 4
    grid = simplex_grid(0.1)
    assert len(grid) == 66 and all(sum(p) == pytest.approx(1) for p in grid)


def test_rare_mcar_precision_falls_with_prevalence():
    table = rare_mcar(_pred(recalls=(0.9, 0.8, 0.8)), ("m",), (0.05, 1 / 3))
    rare, balanced = table.sort_values("mcar_prevalence").itertuples()
    assert rare.recall_mcar == pytest.approx(0.9) and balanced.recall_mcar == pytest.approx(0.9)
    assert rare.precision_mcar < balanced.precision_mcar
    # erros MAR → MNAR e MNAR → MCAR: 2 de 20 não-MCAR viram MCAR (especificidade 0,9)
    assert balanced.specificity == pytest.approx(0.9)
    assert balanced.precision_mcar == pytest.approx((1 / 3 * 0.9) / (1 / 3 * 0.9 + 2 / 3 * 0.1))


def test_run_robustness_writes_the_three_tables(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "eval-symmetric").mkdir(parents=True)
    pred = pd.concat([_pred("direct_hgb"), _pred("logit_simple", recalls=(0.7, 0.7, 0.7))], ignore_index=True)
    pred.to_parquet(run_dir / "eval-symmetric" / "predictions.parquet", index=False)
    out = run_robustness(run_dir, parse_config({"name": "r", "root_seed": 1, "output_root": str(tmp_path)}))
    shift = pd.read_csv(out / "prior_shift.csv")
    assert set(shift.method) == {"direct_hgb", "logit_simple"} and len(pd.read_csv(out / "prior_shift_grid.csv")) == 132
    assert (out / "rare_mcar.csv").exists() and (run_dir / "run.robustness.json").exists()
