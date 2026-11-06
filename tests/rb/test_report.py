"""Testes do relatório do piloto (passo 37): testes de L1 contam falhas como erro; MDE."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.report import N_FAMILIES_PLANNED, N_HYPOTHESES, l1_tests, mde


def test_l1_failure_counts_as_error(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "family_id": [1, 1, 1, 1],
            "mechanism": ["MCAR", "MCAR", "MAR", "MNAR"],
            "little_p": [0.5, np.nan, 0.001, np.nan],
        }
    )
    row = l1_tests(frame, tmp_path, 0.05).set_index("test").loc["little"]
    assert row.n_failed == 2
    assert row.specificity_mean == pytest.approx(0.5)  # MCAR com falha não conta como "não rejeitou"
    assert row.sensitivity_mean == pytest.approx(0.5)  # MNAR com falha não conta como rejeição


def test_mde_matches_formula():
    assert mde(0.06, 42, m=4) == pytest.approx((2.4977 + 0.8416) * 0.06 / np.sqrt(42), rel=1e-3)


def test_mde_uses_the_preregistered_family():
    """C2 (29/10): Holm m = 6 e N = 50, como no §5.1 do pré-registro (fator 3,48)."""
    assert N_HYPOTHESES == 6 and N_FAMILIES_PLANNED == (50,)
    assert mde(0.058, 50) * np.sqrt(50) / 0.058 == pytest.approx(3.48, abs=0.01)
