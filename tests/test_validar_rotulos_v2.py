"""Testes do protocolo v2 de validação de rótulos.

Cobre Camadas A (MCAR), B (MAR), C (MNAR), reconciliação por regras e
agregação Bayesiana via KDE. Usa datasets sintéticos pequenos gerados em
runtime para velocidade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_mcar_df(n: int = 300, missing_rate: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """X0..X4 i.i.d. normais; mask de X0 uniformemente aleatória (MCAR puro)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {f"X{i}": rng.normal(size=n) for i in range(5)},
    )
    miss_idx = rng.choice(n, size=int(n * missing_rate), replace=False)
    df.loc[miss_idx, "X0"] = np.nan
    return df


def _make_mar_df(n: int = 300, missing_rate: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Mask de X0 depende fortemente de X1 (MAR forte)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"X{i}": rng.normal(size=n) for i in range(5)})
    p_miss = 1.0 / (1.0 + np.exp(-3.0 * df["X1"].values))
    p_miss = p_miss / p_miss.mean() * missing_rate
    p_miss = np.clip(p_miss, 0.0, 0.95)
    miss_mask = rng.random(n) < p_miss
    df.loc[miss_mask, "X0"] = np.nan
    return df


def _make_mnar_df(n: int = 300, missing_rate: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Mask de X0 depende do próprio X0 (truncamento extremo, MNAR)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"X{i}": rng.normal(size=n) for i in range(5)})
    threshold = np.quantile(df["X0"].values, 1 - missing_rate)
    df.loc[df["X0"] > threshold, "X0"] = np.nan
    return df


# ============================================================
# Camada A
# ============================================================
def test_layer_a_returns_expected_keys():
    from missdetect.validar_rotulos_v2 import layer_a_mcar

    df = _make_mcar_df(n=200, missing_rate=0.15)
    res = layer_a_mcar(df, n_permutations=10)
    assert {"little_p", "pklm_p", "levene_p", "rejects_mcar", "n_tests_reject", "n_tests_valid"} <= set(res)


@pytest.mark.slow
def test_layer_a_does_not_reject_mcar_strongly():
    """Em MCAR puro, ao menos um dos testes não rejeita (Tipo I não absurdo)."""
    from missdetect.validar_rotulos_v2 import layer_a_mcar

    df = _make_mcar_df(n=400, missing_rate=0.2, seed=7)
    res = layer_a_mcar(df, n_permutations=30)
    assert res["n_tests_reject"] <= 2, f"Esperava ≤2 rejeições em MCAR; obteve {res}"


@pytest.mark.slow
def test_layer_a_rejects_mar_strongly():
    """MAR forte (prob de mask depende de X1) deve rejeitar MCAR."""
    from missdetect.validar_rotulos_v2 import layer_a_mcar

    df = _make_mar_df(n=400, missing_rate=0.25, seed=11)
    res = layer_a_mcar(df, n_permutations=30)
    assert res["rejects_mcar"], f"MAR forte deveria rejeitar MCAR; obteve {res}"


# ============================================================
# Camada B
# ============================================================
def test_layer_b_returns_expected_keys():
    from missdetect.validar_rotulos_v2 import layer_b_mar

    df = _make_mar_df(n=200, missing_rate=0.2)
    res = layer_b_mar(df, n_permutations=20)
    assert {"auc_obs", "auc_p", "auc_z", "mi_max", "mi_mean"} <= set(res)
    assert 0.0 <= res["auc_obs"] <= 1.0


@pytest.mark.slow
def test_layer_b_auc_high_for_mar():
    """AUC mask~Xobs deve ser claramente acima de 0.5 em MAR forte."""
    from missdetect.validar_rotulos_v2 import layer_b_mar

    df = _make_mar_df(n=400, missing_rate=0.25, seed=13)
    res = layer_b_mar(df, n_permutations=30)
    assert res["auc_obs"] > 0.6, f"AUC deveria ser > 0.6 em MAR forte; obteve {res['auc_obs']:.3f}"


@pytest.mark.slow
def test_layer_b_auc_near_chance_for_mcar():
    """AUC mask~Xobs deve ficar perto de 0.5 em MCAR puro."""
    from missdetect.validar_rotulos_v2 import layer_b_mar

    df = _make_mcar_df(n=400, missing_rate=0.2, seed=17)
    res = layer_b_mar(df, n_permutations=30)
    assert abs(res["auc_obs"] - 0.5) < 0.15, f"AUC deveria ser ~0.5 em MCAR; obteve {res['auc_obs']:.3f}"


# ============================================================
# Camada C
# ============================================================
def test_layer_c_returns_four_features():
    from missdetect.validar_rotulos_v2 import layer_c_mnar

    df = _make_mnar_df(n=200, missing_rate=0.2)
    res = layer_c_mnar(df)
    assert set(res) == {
        "caafe_auc_self_delta",
        "caafe_kl_density",
        "caafe_kurt_excess",
        "caafe_cond_entropy",
    }


def test_layer_c_features_finite():
    from missdetect.validar_rotulos_v2 import layer_c_mnar

    df = _make_mnar_df(n=200, missing_rate=0.25)
    res = layer_c_mnar(df)
    for k, v in res.items():
        assert np.isfinite(v), f"{k} = {v} não é finito"


# ============================================================
# Reconciliação (regras e Bayes)
# ============================================================
def test_scores_to_vec_returns_10_dims():
    from missdetect.validar_rotulos_v2 import VEC_KEYS, layer_a_mcar, layer_b_mar, layer_c_mnar, scores_to_vec

    df = _make_mcar_df(n=150, missing_rate=0.2)
    a = layer_a_mcar(df, n_permutations=10)
    b = layer_b_mar(df, n_permutations=10)
    c = layer_c_mnar(df)
    vec = scores_to_vec({"layer_a": a, "layer_b": b, "layer_c": c})
    assert vec.shape == (10,)
    assert len(VEC_KEYS) == 10


def test_diagnose_rules_returns_valid_mechanism():
    from missdetect.validar_rotulos_v2 import diagnose_rules, layer_a_mcar, layer_b_mar, layer_c_mnar

    df = _make_mar_df(n=200, missing_rate=0.2)
    a = layer_a_mcar(df, n_permutations=10)
    b = layer_b_mar(df, n_permutations=10)
    c = layer_c_mnar(df)
    diag = diagnose_rules({"layer_a": a, "layer_b": b, "layer_c": c})
    assert diag["prediction"] in {"MCAR", "MAR", "MNAR"}
    assert 0.0 <= diag["confidence"] <= 1.0


def test_diagnose_bayes_probabilities_sum_to_one():
    from missdetect.validar_rotulos_v2 import (
        diagnose_bayes,
        fit_kde_from_scores,
        layer_a_mcar,
        layer_b_mar,
        layer_c_mnar,
    )

    rng = np.random.default_rng(0)
    arrays = {
        cls: rng.normal(size=(20, 10)) + offset
        for cls, offset in zip(("MCAR", "MAR", "MNAR"), (-1.0, 0.0, 1.0), strict=True)
    }
    kde = fit_kde_from_scores(arrays, bandwidth=0.5)

    df = _make_mcar_df(n=150, missing_rate=0.2)
    a = layer_a_mcar(df, n_permutations=10)
    b = layer_b_mar(df, n_permutations=10)
    c = layer_c_mnar(df)
    diag = diagnose_bayes({"layer_a": a, "layer_b": b, "layer_c": c}, kde)
    s = diag["p_mcar"] + diag["p_mar"] + diag["p_mnar"]
    assert abs(s - 1.0) < 1e-6, f"Probabilidades não somam 1: {s}"
    assert diag["prediction"] in {"MCAR", "MAR", "MNAR"}


def test_fit_kde_with_minimal_data():
    from missdetect.validar_rotulos_v2 import fit_kde_from_scores

    rng = np.random.default_rng(0)
    arrays = {cls: rng.normal(size=(5, 10)) for cls in ("MCAR", "MAR", "MNAR")}
    kde = fit_kde_from_scores(arrays, bandwidth=0.5)
    assert set(kde.keys()) == {"MCAR", "MAR", "MNAR"}


# ============================================================
# Função de alto nível
# ============================================================
def test_validate_one_runs_end_to_end():
    from missdetect.validar_rotulos_v2 import validate_one

    df = _make_mcar_df(n=200, missing_rate=0.2)
    res = validate_one(df, n_permutations=10)
    assert "diagnosis" in res
    assert res["diagnosis"]["prediction"] in {"MCAR", "MAR", "MNAR"}
    assert "layer_a" in res and "layer_b" in res and "layer_c" in res


# ============================================================
# Reprodutibilidade do paralelismo (mesma seed → mesmos resultados)
# ============================================================
def test_pklm_reproducible_seq_vs_par():
    """pklm_test com n_workers=1 e n_workers=2 deve produzir resultados idênticos."""
    from missdetect.baselines.pklm import pklm_test

    df = _make_mar_df(n=300, missing_rate=0.25, seed=21)
    res_seq = pklm_test(df, n_permutations=20, random_state=42, n_workers=1)
    res_par = pklm_test(df, n_permutations=20, random_state=42, n_workers=2)
    assert res_seq["pklm_statistic"] == pytest.approx(res_par["pklm_statistic"])
    assert res_seq["pklm_pvalue"] == pytest.approx(res_par["pklm_pvalue"])


def test_auc_reproducible_seq_vs_par():
    """auc_mask_from_xobs com n_workers=1 e n_workers=2 deve produzir resultados idênticos."""
    from missdetect.validar_rotulos_v2 import auc_mask_from_xobs

    df = _make_mar_df(n=300, missing_rate=0.25, seed=23)
    res_seq = auc_mask_from_xobs(df, n_permutations=20, random_state=42, n_workers=1)
    res_par = auc_mask_from_xobs(df, n_permutations=20, random_state=42, n_workers=2)
    assert res_seq["auc_obs"] == pytest.approx(res_par["auc_obs"])
    assert res_seq["auc_p"] == pytest.approx(res_par["auc_p"])
    assert res_seq["auc_z"] == pytest.approx(res_par["auc_z"])


def test_validate_one_parallel_layers_matches_sequential():
    """validate_one com parallel_layers=True produz resultado idêntico ao sequencial."""
    from missdetect.validar_rotulos_v2 import validate_one

    df = _make_mar_df(n=250, missing_rate=0.2, seed=29)
    seq = validate_one(df, n_permutations=15, parallel_layers=False)
    par = validate_one(df, n_permutations=15, parallel_layers=True)
    assert seq["layer_a"]["pklm_p"] == pytest.approx(par["layer_a"]["pklm_p"])
    assert seq["layer_b"]["auc_obs"] == pytest.approx(par["layer_b"]["auc_obs"])
    assert seq["diagnosis"]["prediction"] == par["diagnosis"]["prediction"]


def test_bayes_cv_uses_left_out_folds():
    """CV do Bayes deve estimar KDE sem avaliar no mesmo fold de treino."""
    from missdetect.calibrar_protocolo import _eval_bayes_cv
    from missdetect.validar_rotulos_v2 import VEC_KEYS, scores_to_vec

    rows = []
    templates = {
        "MCAR": {
            "little_p": 0.8,
            "pklm_p": 0.8,
            "levene_p": 0.8,
            "n_tests_reject": 0,
            "rejects_mcar": False,
            "auc_obs": 0.5,
            "auc_z": 0.0,
            "mi_max": 0.005,
            "caafe_kurt_excess": 0.0,
            "caafe_cond_entropy": 0.15,
        },
        "MAR": {
            "little_p": 0.01,
            "pklm_p": 0.02,
            "levene_p": 0.01,
            "n_tests_reject": 3,
            "rejects_mcar": True,
            "auc_obs": 0.82,
            "auc_z": 3.0,
            "mi_max": 0.12,
            "caafe_kurt_excess": 0.0,
            "caafe_cond_entropy": 0.15,
        },
        "MNAR": {
            "little_p": 0.8,
            "pklm_p": 0.8,
            "levene_p": 0.8,
            "n_tests_reject": 0,
            "rejects_mcar": False,
            "auc_obs": 0.5,
            "auc_z": 0.0,
            "mi_max": 0.005,
            "caafe_kurt_excess": 1.2,
            "caafe_cond_entropy": 0.55,
        },
    }

    for cls, base in templates.items():
        for i in range(8):
            row = {
                "file": f"{cls}_{i}.txt",
                "true_label": cls,
                **base,
                "pklm_stat": 0.0,
                "n_tests_valid": 3,
                "auc_p": 0.01,
                "mi_mean": base["mi_max"] / 2,
                "caafe_auc_self_delta": 0.0,
                "caafe_kl_density": 0.0,
            }
            row["auc_obs"] += i * 0.001
            scores = {
                "layer_a": {
                    "little_p": row["little_p"],
                    "pklm_p": row["pklm_p"],
                    "levene_p": row["levene_p"],
                },
                "layer_b": {
                    "auc_obs": row["auc_obs"],
                    "auc_z": row["auc_z"],
                    "mi_max": row["mi_max"],
                },
                "layer_c": {
                    "caafe_auc_self_delta": row["caafe_auc_self_delta"],
                    "caafe_kl_density": row["caafe_kl_density"],
                    "caafe_kurt_excess": row["caafe_kurt_excess"],
                    "caafe_cond_entropy": row["caafe_cond_entropy"],
                },
            }
            row.update(dict(zip(VEC_KEYS, scores_to_vec(scores), strict=False)))
            rows.append(row)

    metrics = _eval_bayes_cv(pd.DataFrame(rows), bandwidth=0.5, n_splits=4, seed=42)
    assert metrics["n_splits"] == 4
    assert len(metrics["fold_accuracies"]) == 4
    assert metrics["accuracy"] > 0.9
    assert set(metrics["per_class"]) == {"MCAR", "MAR", "MNAR"}
