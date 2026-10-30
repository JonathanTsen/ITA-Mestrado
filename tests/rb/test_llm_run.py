"""Teste ponta a ponta da etapa `llm` num run mínimo, com backend falso (nenhuma chamada à API)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from google.genai import types

from missdetect.rb.config import parse_config
from missdetect.rb.ids import child_seed
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.llm.cache import ResponseCache
from missdetect.rb.llm.client import PAID, AuthorizationRequired
from missdetect.rb.llm.render import render_a1
from missdetect.rb.llm.run import build_panel, cache_root, run_llm, run_probes
from missdetect.rb.runs import read_observed, write_observed
from missdetect.rb.stats import PRIMARY

FLASH = "gemini-3.8-flash"
DESCRIPTORS = ["g0_missing_rate", "g2_cv_auc_boosting"]


class Fake:
    def __init__(self, answer):
        self.answer = answer
        self.generated: list[dict] = []
        self.counted: list[str] = []

    def tiers(self):
        return [PAID]

    def generate(self, tier, request):
        self.generated.append(request)
        text = self.answer(request)
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))],
            model_version="gemini-3.8-flash-001", response_id=f"r{len(self.generated)}",
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1000, candidates_token_count=20, thoughts_token_count=500
            ),
        )  # fmt: skip

    def count_tokens(self, tier, model, text):
        self.counted.append(model)
        return 1000

    def batch_create(self, *args):
        raise AssertionError("o teste é síncrono")

    def batch_get(self, *args):
        raise AssertionError("o teste é síncrono")


@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contexts, bases, instances, oracle, features, preds = [], [], [], [], [], []
    for family in (1, 2):
        for rate in (0.1, 0.2, 0.3, 0.5):
            base = f"b{family}{rate}"
            bases.append({"base_id": base, "family_id": family, "n_max": 3000})
            for n in (500, 1000):
                ctx = f"c{family}{rate}{n}"
                contexts.append({"context_id": ctx, "base_id": base, "family_id": family, "n": n, "k": 2,
                                 "rate": rate, "openml_id": 100 + family, "target": "outcome"})  # fmt: skip
                rng = np.random.default_rng(len(contexts))
                z = pd.DataFrame({"drv": rng.normal(size=n), "aux": rng.normal(size=n)})
                for index, mech in enumerate(("MCAR", "MAR", "MNAR")):
                    sid = f"{ctx}x{index}"
                    y = z.drv.to_numpy() + rng.normal(size=n)
                    y[rng.random(n) < rate] = np.nan
                    rel = write_observed(run_dir, ObservedInstance(sid, y, z))
                    instances.append({"sample_id": sid, "context_id": ctx, "family_id": family, "observed_path": rel,
                                      "realized_rate": rate})  # fmt: skip
                    oracle.append({"sample_id": sid, "context_id": ctx, "family_id": family, "mechanism": mech,
                                   "generator": "symmetric"})  # fmt: skip
                    features.append({"sample_id": sid, "g0_missing_rate": rate, "g2_cv_auc_boosting": rng.random()})
                    for repeat in range(3):
                        label = ("MCAR", "MAR", "MNAR").index(mech)
                        preds.append({"method": PRIMARY, "repeat": repeat, "fold": 0, "sample_id": sid,
                                      "family_id": family, "y_true": label, "y_pred": label, "p_MCAR": 0.2,
                                      "p_MAR": 0.3, "p_MNAR": 0.5, "status": "ok", "seconds": 0.0})  # fmt: skip
    for name, rows in [("contexts", contexts), ("bases", bases), ("instances", instances), ("oracle", oracle),
                       ("features", features)]:  # fmt: skip
        pd.DataFrame(rows).to_parquet(run_dir / f"{name}.parquet", index=False)
    (run_dir / "eval-symmetric").mkdir()
    pd.DataFrame(preds).to_parquet(run_dir / "eval-symmetric" / "predictions.parquet", index=False)
    schema = {"version": "34b", "features": [{"name": n} for n in DESCRIPTORS]}
    (run_dir / "feature_schema.json").write_text(json.dumps(schema))
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    rng = np.random.default_rng(0)
    for family in (1, 2):
        table = pd.DataFrame({"preg": rng.integers(0, 10, 300).astype(float), "mass": rng.normal(30, 5, 300).round(1)})
        table["target"] = rng.choice(["yes", "no"], 300)
        table.to_csv(csv_dir / f"{100 + family}.csv")
    metadata = tmp_path / "openml_metadata.csv"
    pd.DataFrame({"openml_id": [101, 102], "name": ["diabetes", "electricity"], "default_target": ["class", "class"]}
                 ).to_csv(metadata, index=False)  # fmt: skip
    config = parse_config(
        {"name": "llm-test", "root_seed": 9, "output_root": str(tmp_path / "results"),
         "sources": {"csv_dir": str(csv_dir), "roles": ["dev"]}}
    )  # fmt: skip
    run_dir.joinpath("_metadata").write_text(str(metadata))
    descriptions = tmp_path / "openml_descriptions.csv"
    pd.DataFrame({"openml_id": [101, 102], "status": "ok", "name": ["diabetes", "kdd_x_97"],
                  "description": ["**Author**: Someone\n\nAdult patients. Missing values: none.",
                                  "The kdd_x_97 extract. Household income records."]}).to_csv(descriptions, index=False)  # fmt: skip
    run_dir.joinpath("_descriptions").write_text(str(descriptions))
    return run_dir, config


def _mcar_always(request):
    return '{"MCAR": 0.8, "MAR": 0.1, "MNAR": 0.1}'


def test_panel_is_the_registered_selection_plus_ten_percent(run):
    run_dir, config = run
    panel = build_panel(run_dir, config)
    main = panel[panel.replicate == 0].merge(pd.read_parquet(run_dir / "contexts.parquet"), on="context_id")
    assert len(main) == 2 * 3 * 3 and set(main.rate) == {0.1, 0.3, 0.5} and set(main.n) == {1000}
    assert (panel.replicate == 1).sum() == 2  # round(0,1 × 18)
    assert panel.replicate.is_monotonic_increasing  # a repetição vem depois do painel inteiro


def test_llm_step_writes_predictions_h4_table_and_never_pays_twice(run):
    run_dir, config = run
    fake = Fake(_mcar_always)
    out = run_llm(run_dir, config, model=FLASH, arm="A1", mode="sync", backend=fake, sleep=lambda s: None)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert len(pred) == 18 and set(pred.method) == {"llm_a1"} and (pred.y_pred == 0).all()
    assert set(pred.model_version) == {"gemini-3.8-flash-001"} and set(pred.tier) == {PAID}
    per_family = pd.read_parquet(out / "per_family.parquet")
    assert set(per_family.method) == {"llm_a1", PRIMARY} and len(per_family) == 4
    llm = per_family[per_family.method == "llm_a1"]
    assert np.allclose(llm.macro_recall, 1 / 3) and np.allclose(
        per_family[per_family.method == PRIMARY].macro_recall, 1
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_valid"] == 18 and summary["n_calls"] == 20 and summary["cost_usd"] > 0
    assert (run_dir / f"run.llm-{FLASH}-A1.json").exists()
    assert set(fake.counted) == {FLASH}
    agreement = pd.read_csv(out / "repeat_agreement.csv")
    assert agreement.n.iloc[0] == 2 and agreement.agreement.iloc[0] == 1.0
    calls = len(fake.generated)
    assert calls == 20
    assert all("drv" not in r["user"] and "aux" not in r["user"] for r in fake.generated)
    shutil.rmtree(out)
    (run_dir / f"run.llm-{FLASH}-A1.json").unlink()
    run_llm(run_dir, config, model=FLASH, arm="A1", mode="sync", backend=fake, sleep=lambda s: None)
    assert len(fake.generated) == calls  # tudo veio do caderno


def test_invalid_answers_become_failures_counted_as_errors(run):
    run_dir, config = run
    fake = Fake(lambda request: "não sei")
    out = run_llm(run_dir, config, model=FLASH, arm="A1", mode="sync", backend=fake, sleep=lambda s: None)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert (pred.status == "failed").all() and (pred.y_pred == -1).all()
    assert np.allclose(pred[["p_MCAR", "p_MAR", "p_MNAR"]].to_numpy(), 1 / 3)
    assert len(fake.generated) == 20 * 3  # a primeira resposta mais 2 retentativas por pergunta


def test_ties_are_broken_among_the_tied_classes(run):
    run_dir, config = run
    fake = Fake(lambda request: '{"MCAR": 0.0, "MAR": 0.5, "MNAR": 0.5}')
    out = run_llm(run_dir, config, model=FLASH, arm="A1", mode="sync", backend=fake, sleep=lambda s: None)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert set(pred.y_pred) == {1, 2}


def test_budget_only_makes_no_generation_call_and_argon_needs_authorization(run):
    run_dir, config = run
    fake = Fake(_mcar_always)
    path = run_llm(run_dir, config, model=FLASH, arm="A2", mode="batch", backend=fake, budget_only=True)
    check = json.loads(path.read_text().splitlines()[-1])
    assert fake.generated == [] and check["n_pending"] == 20 and check["projected_usd"] > 0
    with pytest.raises(AuthorizationRequired):
        run_llm(run_dir, config, model="gemini-4-argon", arm="A1", mode="batch", backend=fake, budget_only=True)
    assert fake.generated == [] and set(fake.counted) == {FLASH}
    assert not any(cache_root(run_dir, config).rglob("attempt-*.json"))


def test_a2_prompt_is_only_the_descriptors(run):
    run_dir, config = run
    fake = Fake(_mcar_always)
    run_llm(run_dir, config, model=FLASH, arm="A2", mode="sync", backend=fake, sleep=lambda s: None)
    users = {r["user"] for r in fake.generated}
    assert all(set(json.loads(u)) == set(DESCRIPTORS) for u in users)
    cache = ResponseCache(cache_root(run_dir, config))
    assert cache.digest()


def test_without_the_primary_predictions_the_calls_are_made_and_the_outputs_wait(run):
    run_dir, config = run
    saved = run_dir / "eval-symmetric-saved"
    (run_dir / "eval-symmetric").rename(saved)
    fake = Fake(_mcar_always)
    note = run_llm(run_dir, config, model=FLASH, arm="A1", mode="sync", backend=fake, sleep=lambda s: None)
    assert note.name == "aguardando-eval-symmetric.txt" and not (run_dir / "llm" / FLASH / "A1").exists()
    calls = len(fake.generated)
    saved.rename(run_dir / "eval-symmetric")
    out = run_llm(run_dir, config, model=FLASH, arm="A1", mode="sync", backend=fake, sleep=lambda s: None)
    assert (out / "per_family.parquet").exists() and len(fake.generated) == calls == 20


def _probe_answer(blind_named: bool):
    def answer(request):
        user = request["user"] if isinstance(request["user"], str) else request["user"][-1]["content"]
        if "What is the name of this dataset?" in user:
            if "c0," in user and not blind_named:
                return "I do not know"
            return "This is the Pima diabetes dataset" if "preg" in user or blind_named else "unknown"
        return "x"

    return answer


def test_probes_record_then_read_and_exclude_only_blind_recognition(run):
    run_dir, config = run
    metadata = Path((run_dir / "_metadata").read_text())
    fake = Fake(_probe_answer(blind_named=False))
    out = run_probes(run_dir, config, model=FLASH, mode="sync", backend=fake, sleep=lambda s: None, metadata=metadata)
    results = pd.read_csv(out / "results.csv")
    assert len(results) == 4 and set(results.table) == {"raw", "blind"}
    raw = results[results.table == "raw"].set_index("family_id")
    assert bool(raw.loc[1, "name_match"]) and bool(raw.loc[1, "recognized"])  # "diabetes" está na resposta
    assert not bool(raw.loc[2, "name_match"])
    assert not results[results.table == "blind"].recognized.any()
    assert pd.read_csv(out / "excluded_families.csv").empty
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_failed_questions"] == 0 and summary["n_questions"] == len(
        {json.dumps(r, sort_keys=True) for r in fake.generated}
    )
    raw_csv = (run_dir / "llm" / FLASH / "_work" / "probes" / "tables" / "1-raw.csv").read_text().splitlines()
    assert raw_csv[0] == "preg,mass,class" and raw_csv[1].split(",")[0].isdigit()  # sem índice, inteiros sem ".0"
    blind = (run_dir / "llm" / FLASH / "_work" / "probes" / "tables" / "1-blind.csv").read_text().splitlines()
    assert blind[0] == "c0,c1,c2"


def test_probes_exclude_a_family_recognized_in_the_blind_rendering(run):
    run_dir, config = run
    metadata = Path((run_dir / "_metadata").read_text())
    fake = Fake(_probe_answer(blind_named=True))
    out = run_probes(run_dir, config, model=FLASH, mode="sync", backend=fake, sleep=lambda s: None, metadata=metadata)
    assert pd.read_csv(out / "excluded_families.csv").family_id.tolist() == [1]
    with pytest.raises(AuthorizationRequired):
        run_probes(run_dir, config, model="gemini-4-argon", mode="batch", backend=fake, metadata=metadata)


def test_limit_asks_only_the_first_questions_and_writes_only_a_usage_summary(run):
    """Teste do Argon (≈ 30 chamadas): --limit pergunta só as N primeiras, na ordem de execução, sem saídas finais."""
    run_dir, config = run
    fake = Fake(_mcar_always)
    path = run_llm(run_dir, config, model=FLASH, arm="A1", mode="sync", backend=fake, sleep=lambda s: None, limit=5)
    assert len(fake.generated) == 5 and not (run_dir / "llm" / FLASH / "A1").exists()
    summary = json.loads(path.read_text())
    assert summary["n_questions"] == 5 and summary["n_valid"] == 5 and summary["cost_usd"] > 0
    assert summary["mean_thoughts_tokens"] == 500 and summary["limit"] == 5
    order = build_panel(run_dir, config)
    first = order[order.replicate == 0].sample_id.iloc[0]
    assert render_a1_user_of(run_dir, config, first) in {r["user"] for r in fake.generated}


def render_a1_user_of(run_dir, config, sample_id):
    contexts = pd.read_parquet(run_dir / "instances.parquet").set_index("sample_id").context_id
    seed = child_seed(config.root_seed, "llm_render", contexts.loc[sample_id])
    return render_a1(read_observed(run_dir, sample_id), seed)


def test_a3_uses_real_names_and_the_neutral_description(run):
    run_dir, config = run
    descriptions = Path((run_dir / "_descriptions").read_text())
    fake = Fake(_mcar_always)
    run_llm(run_dir, config, model=FLASH, arm="A3", mode="sync", backend=fake, sleep=lambda s: None, limit=4,
            descriptions=descriptions)  # fmt: skip
    users = [r["user"] for r in fake.generated]
    assert all(u.startswith("Description: ") and "Table:\n" in u for u in users)
    header = users[0].split("Table:\n", 1)[1].splitlines()[0]
    assert sorted(header.split(",")) == ["aux", "drv", "outcome"]
    joined = " ".join(users)
    assert "Missing" not in joined and "Someone" not in joined and "kdd_x_97" not in joined
    bad = descriptions.with_name("sem.csv")
    pd.read_csv(descriptions).iloc[:1].to_csv(bad, index=False)
    with pytest.raises(ValueError, match="sem descrição"):
        run_llm(run_dir, config, model=FLASH, arm="A3", mode="sync", backend=fake, descriptions=bad, budget_only=True)


def test_name_matching_uses_the_alias_list_and_feature_test_has_a_chance_baseline(tmp_path):
    from missdetect.rb.llm.probes import feature_chance, name_matches

    assert name_matches("This is Breast Cancer Wisconsin (Original)", ["BNG(breast-w)", "breast cancer wisconsin"])
    assert not name_matches("Statlog (Shuttle)", ["BNG(breast-w)", "breast cancer wisconsin"])
    assert not name_matches("pc4", ["pc3"])
    csv = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7], "b": range(10)}).to_csv(csv, index=False)
    share, p_value = feature_chance(csv, "a", matches=4, n_queries=10)
    assert share == pytest.approx(0.4) and p_value > 0.05  # 4 de 10 é o que o chute da moda daria
    share, p_value = feature_chance(csv, "b", matches=4, n_queries=10)
    assert share == pytest.approx(0.1) and p_value < 0.05
