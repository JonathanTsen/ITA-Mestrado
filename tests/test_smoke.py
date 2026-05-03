"""Smoke tests: package importable, headline features compute on a fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_package_importable():
    """The package itself imports and exposes a version string."""
    import missdetect

    assert isinstance(missdetect.__version__, str)
    assert missdetect.__version__.count(".") == 2  # semver-shaped


def test_statistical_features_shape():
    """extract_statistical_features returns the documented 4-key dict."""
    from missdetect.features.statistical import extract_statistical_features

    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame(
        {
            "X0": rng.normal(size=200),
            "X1": rng.normal(size=200),
            "X2": rng.normal(size=200),
            "X3": rng.normal(size=200),
            "X4": rng.normal(size=200),
        }
    )
    # Inject ~20% MCAR missingness in X0
    df.loc[rng.choice(200, size=40, replace=False), "X0"] = np.nan

    feats = extract_statistical_features(df)

    expected_keys = {
        "X0_missing_rate",
        "X0_obs_vs_full_ratio",
        "X0_iqr_ratio",
        "X0_obs_skew_diff",
    }
    assert set(feats.keys()) >= expected_keys, (
        f"missing keys: {expected_keys - set(feats.keys())}"
    )
    # Missing rate is the empirical fraction (~0.20)
    assert 0.15 < feats["X0_missing_rate"] < 0.25


def test_statistical_features_no_missing_is_safe():
    """Edge case: no missing values yields well-defined defaults."""
    from missdetect.features.statistical import extract_statistical_features

    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "X0": rng.normal(size=50),
            "X1": rng.normal(size=50),
            "X2": rng.normal(size=50),
            "X3": rng.normal(size=50),
            "X4": rng.normal(size=50),
        }
    )

    feats = extract_statistical_features(df)
    assert feats["X0_missing_rate"] == 0.0


def test_statistical_features_all_missing_is_safe():
    """Edge case: all values missing should not crash."""
    from missdetect.features.statistical import extract_statistical_features

    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "X0": [np.nan] * 50,
            "X1": rng.normal(size=50),
            "X2": rng.normal(size=50),
            "X3": rng.normal(size=50),
            "X4": rng.normal(size=50),
        }
    )

    feats = extract_statistical_features(df)
    assert feats["X0_missing_rate"] == 1.0


@pytest.mark.parametrize("submodule", ["features", "llm", "baselines", "utils"])
def test_subpackages_importable(submodule):
    """All advertised subpackages are present and importable."""
    import importlib

    mod = importlib.import_module(f"missdetect.{submodule}")
    assert mod is not None
