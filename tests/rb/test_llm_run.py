"""Teste ponta a ponta da etapa `llm` num run mínimo, com backend falso (nenhuma chamada à API)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from google.genai import types

from missdetect.rb.config import parse_config
from missdetect.rb.inject import ObservedInstance
from missdetect.rb.llm.cache import ResponseCache
from missdetect.rb.llm.client import PAID, AuthorizationRequired
from missdetect.rb.llm.run import build_panel, cache_root, run_llm
from missdetect.rb.runs import write_observed
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
                                 "rate": rate})  # fmt: skip
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
    config = parse_config({"name": "llm-test", "root_seed": 9, "output_root": str(tmp_path / "results")})
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
    assert not any((cache_root(config)).rglob("attempt-*.json"))


def test_a2_prompt_is_only_the_descriptors(run):
    run_dir, config = run
    fake = Fake(_mcar_always)
    run_llm(run_dir, config, model=FLASH, arm="A2", mode="sync", backend=fake, sleep=lambda s: None)
    users = {r["user"] for r in fake.generated}
    assert all(set(json.loads(u)) == set(DESCRIPTORS) for u in users)
    cache = ResponseCache(cache_root(config))
    assert cache.digest()
