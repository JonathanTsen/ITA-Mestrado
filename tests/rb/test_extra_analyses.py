"""Testes das análises das emendas de 29/10 (P6, P7, P8, P11, P12, P13, P15) com resultados conhecidos."""

import numpy as np
import pandas as pd
import pytest

from missdetect.rb.extra_analyses import (
    by_repeat,
    class_decomposition,
    decision_error_tests,
    decision_errors,
    generator_effect,
    l1_auc,
    l1_false_mcar,
    nominal_contexts,
    recall_by_regime,
    recalls_by_family,
    specialization,
    trio_concordance,
    without_contexts,
)

CLASS = {"MCAR": 0, "MAR": 1, "MNAR": 2}


def _pred(rows: list[tuple]) -> pd.DataFrame:
    """rows: (method, family_id, repeat, sample_id, y_true, y_pred, p_MNAR_share)."""
    out = []
    for method, family, repeat, sample, y_true, y_pred, share in rows:
        p_mcar = 0.2
        out.append({"method": method, "family_id": family, "repeat": repeat, "sample_id": sample, "y_true": y_true,
                    "y_pred": y_pred, "p_MCAR": p_mcar, "p_MAR": (1 - p_mcar) * (1 - share),
                    "p_MNAR": (1 - p_mcar) * share, "status": "ok"})  # fmt: skip
    return pd.DataFrame(out)


def _trio_pred(method: str, families: int = 4, repeats: int = 1, mnar_as: int = 2, share_mnar: float = 0.8):
    """Um contexto por família, com as três instâncias; o MNAR é previsto como `mnar_as`."""
    rows = []
    for f in range(families):
        for r in range(repeats):
            rows += [
                (method, f, r, f"{f}-mcar", 0, 0, 0.5),
                (method, f, r, f"{f}-mar", 1, 1, 1 - share_mnar),
                (method, f, r, f"{f}-mnar", 2, mnar_as, share_mnar),
            ]
    return _pred(rows)


def test_recalls_by_family_counts_failures_as_errors():
    pred = _trio_pred("m", families=1)
    pred.loc[pred.sample_id == "0-mar", "y_pred"] = -1  # falha
    rec = recalls_by_family(pred).iloc[0]
    assert rec.recall_MCAR == 1.0 and rec.recall_MAR == 0.0 and rec.recall_MNAR == 1.0
    assert rec.macro_recall == pytest.approx(2 / 3)


def test_recall_by_regime_shows_censoring_as_mar():
    """P6: com β = ∞ (direção lower), o MNAR previsto como MAR dá recall de MNAR 0 nesse regime."""
    inf = _trio_pred("direct_hgb", families=4, mnar_as=1)
    two = _trio_pred("direct_hgb", families=4, mnar_as=2)
    two["sample_id"] = "b2-" + two.sample_id
    pred = pd.concat([inf, two], ignore_index=True)
    regimes = pd.DataFrame({"sample_id": pred.sample_id.unique()})
    regimes["beta"] = np.where(regimes.sample_id.str.startswith("b2-"), 2.0, np.inf)
    regimes["direction"] = "lower"
    table = recall_by_regime(pred, regimes, "direct_hgb").set_index(["regime", "metric"])
    assert table.loc[("inf", "recall_MNAR"), "mean"] == 0.0
    assert table.loc[("inf/lower", "recall_MNAR"), "mean"] == 0.0
    assert table.loc[("2.0", "recall_MNAR"), "mean"] == 1.0
    assert table.loc[("inf", "recall_MNAR"), "n_families"] == 4


def test_trio_concordance_compares_mnar_with_its_own_mar():
    """P7: 1 quando a instância MNAR recebe escore maior que a MAR do mesmo contexto; empate vale 1/2."""
    pred = _trio_pred("direct_hgb", families=2, share_mnar=0.8)
    pred.loc[pred.sample_id == "1-mnar", "p_MNAR"] = pred.loc[pred.sample_id == "1-mar", "p_MNAR"].to_numpy()
    pred.loc[pred.sample_id == "1-mnar", "p_MAR"] = pred.loc[pred.sample_id == "1-mar", "p_MAR"].to_numpy()
    contexts = pd.DataFrame({"sample_id": pred.sample_id.unique()})
    contexts["context_id"] = contexts.sample_id.str.split("-").str[0]
    conc = trio_concordance(pred, contexts, "direct_hgb").set_index("family_id").concordance
    assert conc.loc[0] == 1.0 and conc.loc[1] == 0.5


def test_class_decomposition_finds_where_the_difference_is():
    """P8: a diferença entre dois métodos é separada por classe."""
    rng = np.random.default_rng(0)
    rows = []
    for f in range(12):
        base = rng.uniform(0.5, 0.7)
        rows.append(
            {"method": "a", "family_id": f, "recall_MCAR": base - 0.10, "recall_MAR": base, "recall_MNAR": base}
        )
        rows.append({"method": "b", "family_id": f, "recall_MCAR": base, "recall_MAR": base, "recall_MNAR": base})
    per_family = pd.DataFrame(rows)
    per_family["macro_recall"] = per_family[["recall_MCAR", "recall_MAR", "recall_MNAR"]].mean(axis=1)
    table = class_decomposition(per_family, "a", "b", "H5").set_index("metric")
    assert table.loc["recall_MCAR", "hodges_lehmann"] == pytest.approx(-0.10)
    assert table.loc["recall_MAR", "hodges_lehmann"] == pytest.approx(0.0)


def test_generator_effect_and_specialization():
    """P8: efeito do gerador = externo − recorte β = ∞ do simétrico; especialização = treinado no externo − no simétrico."""
    sym = _trio_pred("direct_hgb", families=3)  # acerta tudo no simétrico com β = ∞
    regimes = pd.DataFrame({"sample_id": sym.sample_id.unique(), "beta": np.inf, "direction": "lower"})
    ext = _trio_pred("cross_direct_hgb", families=3, mnar_as=1)  # erra o MNAR no externo
    ext["sample_id"] = "e-" + ext.sample_id
    ingen = ext.assign(method="ingen_direct_hgb", y_pred=ext.y_true)
    cross = pd.concat([ext, ingen], ignore_index=True)
    generators = pd.DataFrame({"sample_id": ext.sample_id.unique(), "generator": "mdatagen"})
    effect = generator_effect(cross, sym, regimes, generators).set_index("generator")
    assert effect.loc["mdatagen", "hodges_lehmann"] == pytest.approx(-1 / 3)
    spec = specialization(cross, generators).set_index("generator")
    assert spec.loc["mdatagen", "hodges_lehmann"] == pytest.approx(1 / 3)


def test_decision_errors_and_their_tests():
    """P12: falso MCAR e MNAR → MAR por família; o teste usa 1 − taxa (maior é melhor), com Holm m = 2."""
    rows = []
    for f in range(10):
        rows += [("direct_hgb", f, 0, f"{f}a", 1, 1, 0.2), ("direct_hgb", f, 0, f"{f}b", 2, 2, 0.8)]
        rows += [("logit_simple", f, 0, f"{f}a", 1, 0, 0.2), ("logit_simple", f, 0, f"{f}b", 2, 1, 0.2)]
    per_family = decision_errors(_pred(rows)).set_index(["method", "family_id"])
    assert per_family.loc[("logit_simple", 0), "false_mcar"] == 0.5
    assert per_family.loc[("logit_simple", 0), "mnar_as_mar"] == 1.0
    assert per_family.loc[("direct_hgb", 0), "avoid_false_mcar"] == 1.0
    tests = decision_error_tests(per_family.reset_index())
    assert len(tests) == 2 and (tests.decision == "superior").all()
    assert tests.p_holm.min() == pytest.approx(min(2 * tests.p_value.min(), 1.0))


def test_l1_auc_is_one_for_perfect_separation():
    """P13: AUC sem limiar; falha recebe o pior escore."""
    rows = []
    for f in range(3):
        rows += [("hier", f, 0, f"{f}x", 0, 0, 0.5), ("hier", f, 0, f"{f}y", 1, 1, 0.5)]
    pred = _pred(rows)
    pred.loc[pred.y_true == 0, "p_MCAR"] = 0.9
    pred.loc[pred.y_true == 1, "p_MCAR"] = 0.1
    baselines = pd.DataFrame({"sample_id": pred.sample_id.unique()})
    baselines["little_p"] = np.where(baselines.sample_id.str.endswith("x"), 0.8, 0.01)
    baselines["mdo_p_shuffled"] = np.nan  # falhou em tudo
    table = l1_auc(pred, baselines, None, "hier")
    by = table.groupby("method").auc.mean()
    assert by["l1_hier"] == 1.0 and by["little"] == 1.0
    assert by["mdo_test1"] == 0.0


def test_nominal_contexts_closed_list():
    """P15: a família inteira com "*" ou só as bases cujo Y é o código listado."""
    contexts = pd.DataFrame({
        "context_id": ["a", "b", "c", "d"],
        "family_id": [4135, 42493, 42493, 1],
        "target": ["ROLE_CODE", "Flight", "Time", "x"],
    })  # fmt: skip
    nominal = pd.DataFrame({"family_id": [4135, 42493], "target": ["*", "Flight"]})
    assert nominal_contexts(contexts, nominal) == {"a", "b"}


def test_by_repeat_runs_the_confirmatory_family_once_per_repeat():
    """P11: a família principal é refeita em cada repetição (descritivo)."""
    methods = ("direct_hgb", "hier_hgb_fullprob", "logit_simple", "mechdetect_obs")
    sym = pd.concat([_trio_pred(m, families=6, repeats=2) for m in methods], ignore_index=True)
    cross = pd.concat([_trio_pred(m, families=6, repeats=2) for m in ("cross_direct_hgb", "cross_logit_simple")])
    table = by_repeat(sym, cross)
    assert sorted(table.repeat.unique()) == [0, 1]
    for _, g in table.groupby("repeat"):
        assert list(g.hypothesis) == ["H1", "H2", "H2e", "H3", "H5", "H6"]


def test_l1_false_mcar_is_one_minus_sensitivity():
    """P12: para os testes binários do L1, o falso MCAR é 1 − sensibilidade (não há MNAR → MAR)."""
    table = pd.DataFrame({"method": ["little", "little"], "family_id": [0, 1], "sensitivity": [0.75, 0.25]})
    row = l1_false_mcar(table).set_index("method").loc["little"]
    assert row["mean"] == pytest.approx(0.5) and row["n_families"] == 2


def test_without_contexts_matches_the_full_family_when_nothing_is_excluded():
    """P15: sem nenhuma base excluída, a sensibilidade repete a análise principal."""
    methods = ("direct_hgb", "hier_hgb_fullprob", "logit_simple", "mechdetect_obs")
    rng = np.random.default_rng(3)
    sym = pd.concat([_trio_pred(m, families=6, repeats=2) for m in methods], ignore_index=True)
    sym["y_pred"] = np.where(rng.uniform(size=len(sym)) < 0.3, rng.integers(0, 3, len(sym)), sym.y_pred)
    instances = pd.DataFrame({"sample_id": sym.sample_id.unique()})
    instances["context_id"] = instances.sample_id.str.split("-").str[0]
    full = without_contexts(sym, None, instances, set())
    some = without_contexts(sym, None, instances, {"0"})
    assert full.n_contexts_excluded.iloc[0] == 0 and some.n_contexts_excluded.iloc[0] == 1
    assert (full.n_families >= some.n_families).all()
