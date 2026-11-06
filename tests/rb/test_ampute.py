"""Testes do gerador externo estocástico da literatura (ampute de Schouten et al. 2018; emenda P24 de 29/10/2026)."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.ampute import AMPUTE_FUNC, ampute_mask


def _data(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    driver = rng.normal(size=n)
    y = 0.3 * driver + rng.normal(size=n)
    z = pd.DataFrame({"w": rng.normal(size=n), "drv": driver, "v": rng.normal(size=n)})
    return y, z


def _corr(mask: np.ndarray, x: np.ndarray) -> float:
    return float(np.corrcoef(mask.astype(float), x)[0, 1])


def test_direction_maps_to_the_sigmoid_of_the_paper():
    assert AMPUTE_FUNC == {"lower": "sigmoid-left", "upper": "sigmoid-right", "both": "sigmoid-tail"}


@pytest.mark.parametrize("direction", ["lower", "upper", "both"])
def test_mechanisms_depend_on_what_they_should(direction):
    y, z = _data()
    score = {"lower": lambda x: -x, "upper": lambda x: x, "both": np.abs}[direction]
    mcar = ampute_mask(y, z, "drv", "MCAR", direction, 0.3, seed=1)
    mar = ampute_mask(y, z, "drv", "MAR", direction, 0.3, seed=1)
    mnar = ampute_mask(y, z, "drv", "MNAR", direction, 0.3, seed=1)
    assert abs(_corr(mcar, score(z.drv.to_numpy()))) < 0.1 and abs(_corr(mcar, score(y))) < 0.1
    assert _corr(mar, score(z.drv.to_numpy())) > 0.2
    assert _corr(mnar, score(y)) > 0.2
    for mask in (mcar, mar, mnar):
        assert abs(mask.mean() - 0.3) < 0.05  # sorteio linha a linha: a contagem varia em torno da taxa


def test_mar_depends_only_on_the_driver():
    """Mudar Y ou outra auxiliar não muda a máscara MAR (peso só no driver)."""
    y, z = _data()
    base = ampute_mask(y, z, "drv", "MAR", "upper", 0.3, seed=5)
    z2 = z.assign(w=np.random.default_rng(9).normal(size=len(z)))
    y2 = np.random.default_rng(8).normal(size=len(y))
    np.testing.assert_array_equal(base, ampute_mask(y2, z2, "drv", "MAR", "upper", 0.3, seed=5))


def test_mnar_depends_only_on_y():
    y, z = _data()
    base = ampute_mask(y, z, "drv", "MNAR", "lower", 0.3, seed=5)
    z2 = z.assign(drv=np.random.default_rng(9).normal(size=len(z)))
    np.testing.assert_array_equal(base, ampute_mask(y, z2, "drv", "MNAR", "lower", 0.3, seed=5))


def test_same_seed_same_mask_and_input_untouched():
    y, z = _data()
    y0, z0 = y.copy(), z.copy()
    a = ampute_mask(y, z, "drv", "MNAR", "both", 0.2, seed=3)
    b = ampute_mask(y, z, "drv", "MNAR", "both", 0.2, seed=3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(y, y0)
    pd.testing.assert_frame_equal(z, z0)
