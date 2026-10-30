"""Testes do cliente do Gemini com um backend falso (nenhuma chamada à API; passo 42 e emendas P20–P22)."""

import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from missdetect.rb.llm.cache import ResponseCache
from missdetect.rb.llm.client import (
    FREE,
    PAID,
    AuthorizationRequired,
    Runner,
    build_request,
    contents_of,
    generation_config,
    is_daily_quota,
    load_api_keys,
    require_authorization,
    retry_delay,
)

FLASH, ARGON = "gemini-3.8-flash", "gemini-4-argon"
VALID = '{"MCAR": 0.2, "MAR": 0.3, "MNAR": 0.5}'


def _response(text: str = VALID, rid: str = "r1") -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        model_version="gemini-3.8-flash-001",
        response_id=rid,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20, thoughts_token_count=300, total_token_count=420
        ),
    )


def _quota_error(daily: bool) -> errors.ClientError:
    quota = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if daily else "GenerateRequestsPerMinute-FreeTier"
    details = [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"},
    ]
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}})


class FakeBackend:
    def __init__(self, tiers=(PAID,), script=None):
        self._tiers = sorted(tiers)
        self.script = list(script or [])
        self.calls: list[tuple[str, str]] = []
        self.jobs: dict[str, SimpleNamespace] = {}

    def tiers(self):
        return self._tiers

    def generate(self, tier, request):
        self.calls.append((tier, request["model"]))
        item = self.script.pop(0) if self.script else VALID
        if isinstance(item, Exception):
            raise item
        return _response(item, rid=f"r{len(self.calls)}")

    def count_tokens(self, tier, model, text):
        self.calls.append((tier, "count:" + model))
        return len(text) // 4

    def batch_create(self, model, inlined, display_name):
        name = f"batches/job{len(self.jobs)}"
        responses = [
            types.InlinedResponse(response=_response(rid=f"b{i}"), metadata=r.metadata) for i, r in enumerate(inlined)
        ]
        self.jobs[name] = SimpleNamespace(
            name=name, state=types.JobState.JOB_STATE_RUNNING, update_time=None,
            dest=types.BatchJobDestination(inlined_responses=responses),
        )  # fmt: skip
        self.calls.append((PAID, "batch:" + model))
        return self.jobs[name]

    def batch_get(self, name):
        return self.jobs[name]


def _request(user="tabela", model=FLASH, replicate=0):
    return build_request(
        arm="A1", model=model, system="instruções", user=user, schema={"type": "object"}, thinking_level="medium",
        max_output_tokens=8192, renderer_version="1", replicate=replicate,
    )  # fmt: skip


def _runner(tmp_path, backend, model=FLASH, **kwargs):
    return Runner(backend, ResponseCache(tmp_path / "cache"), model, sleep=lambda s: None, max_workers=2, **kwargs)


def test_argon_needs_explicit_authorization_and_never_calls_without_it(tmp_path):
    require_authorization(FLASH, None)
    with pytest.raises(AuthorizationRequired):
        require_authorization(ARGON, None)
    with pytest.raises(AuthorizationRequired):
        require_authorization(ARGON, "   ")
    backend = FakeBackend(tiers=(FREE, PAID))
    with pytest.raises(AuthorizationRequired):
        _runner(tmp_path, backend, model=ARGON)
    assert backend.calls == []
    runner = _runner(tmp_path, backend, model=ARGON, authorization="autor, 30/10 10:00")
    assert runner.tier == PAID  # o Argon não tem plano gratuito


def test_load_api_keys_reads_only_the_known_variables(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=pago\nGEMINI_API_KEY_FREE=gratis\nOUTRA=x\n")
    assert load_api_keys(env) == {PAID: "pago", FREE: "gratis"}
    assert load_api_keys(tmp_path / "nao-existe") == {}


def test_generation_config_sends_no_sampling_parameters_nor_tools():
    config = generation_config(_request())
    assert config.temperature is None and config.top_p is None and config.top_k is None and config.seed is None
    assert config.tools is None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert config.response_json_schema == {"type": "object"} and config.response_mime_type == "application/json"
    assert generation_config({**_request(), "schema": None}).response_mime_type is None


def test_sync_stores_metadata_and_never_pays_twice(tmp_path):
    backend = FakeBackend()
    runner = _runner(tmp_path, backend)
    key = runner.cache.register(_request())
    runner.run_sync([key])
    record = runner.cache.first_valid(key)
    assert record["parsed"] == pytest.approx([0.2, 0.3, 0.5])
    assert record["model_version"] == "gemini-3.8-flash-001" and record["response_id"] == "r1"
    assert record["usage"]["thoughts_token_count"] == 300 and record["tier"] == PAID and record["mode"] == "sync"
    runner.run_sync([key])
    assert len(backend.calls) == 1


def test_invalid_output_is_retried_up_to_three_attempts(tmp_path):
    backend = FakeBackend(script=["lixo", VALID])
    runner = _runner(tmp_path, backend)
    key = runner.cache.register(_request())
    runner.run_sync([key])
    attempts = runner.cache.attempts(key)
    assert [a["valid"] for a in attempts] == [False, True] and attempts[0]["invalid_reason"] == "não é json"
    backend = FakeBackend(script=["a", "b", "c", VALID])
    runner = _runner(tmp_path / "x", backend)
    key = runner.cache.register(_request())
    runner.run_sync([key])
    assert len(backend.calls) == 3 and runner.cache.first_valid(key) is None


def test_free_tier_first_then_paid_when_the_daily_quota_ends(tmp_path):
    events = tmp_path / "events.jsonl"
    backend = FakeBackend(tiers=(FREE, PAID), script=[VALID, _quota_error(daily=True), VALID])
    runner = _runner(tmp_path, backend, events=events)
    runner.max_workers = 1
    keys = [runner.cache.register(_request(user=f"t{i}")) for i in range(2)]
    runner.run_sync(keys)
    tiers = sorted(runner.cache.first_valid(k)["tier"] for k in keys)
    assert tiers == [FREE, PAID]
    assert [c[0] for c in backend.calls] == [FREE, FREE, PAID]
    log = [json.loads(line) for line in events.read_text().splitlines()]
    assert log[0]["event"] == "tier_switch" and log[0]["to"] == PAID
    assert "gratis" not in events.read_text() and "pago" not in events.read_text()


def test_free_tier_per_minute_limit_waits_instead_of_switching(tmp_path):
    waits: list[float] = []
    backend = FakeBackend(tiers=(FREE, PAID), script=[_quota_error(daily=False), VALID])
    runner = Runner(backend, ResponseCache(tmp_path), FLASH, sleep=waits.append)
    key = runner.cache.register(_request())
    runner.run_sync([key])
    assert waits == [8.0] and runner.tier == FREE and runner.cache.first_valid(key)["tier"] == FREE


def test_transient_errors_do_not_consume_attempts_and_permanent_errors_stop(tmp_path):
    backend = FakeBackend(script=[errors.ServerError(503, {"error": {"code": 503}}), VALID])
    runner = _runner(tmp_path, backend)
    key = runner.cache.register(_request())
    runner.run_sync([key])
    assert len(runner.cache.attempts(key)) == 1 and runner.cache.first_valid(key) is not None
    backend = FakeBackend(script=[errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})])
    runner = _runner(tmp_path / "y", backend)
    key = runner.cache.register(_request())
    with pytest.raises(errors.ClientError):
        runner.run_sync([key])
    assert runner.cache.attempts(key) == []


def test_error_helpers():
    assert is_daily_quota(_quota_error(daily=True)) and not is_daily_quota(_quota_error(daily=False))
    assert retry_delay(_quota_error(daily=False), 30.0) == 8.0
    assert retry_delay(errors.ServerError(503, {"error": {"code": 503}}), 30.0) == 30.0


def test_batch_records_the_job_before_waiting_and_collects_once(tmp_path):
    backend = FakeBackend(tiers=(FREE, PAID))
    runner = _runner(tmp_path, backend)
    jobs = tmp_path / "jobs" / "jobs.jsonl"
    keys = [runner.cache.register(_request(user=f"t{i}")) for i in range(3)]
    names = runner.submit_batch(keys, jobs, "piloto")
    recorded = [json.loads(line) for line in jobs.read_text().splitlines()]
    assert [r["name"] for r in recorded] == names and len(recorded[0]["requests"]) == 3
    assert runner.collect_batches(jobs) == {names[0]: "JOB_STATE_RUNNING"}
    assert runner.submit_batch(keys, jobs, "piloto") == []  # nada a submeter: tudo está num job pendente
    backend.jobs[names[0]].state = types.JobState.JOB_STATE_SUCCEEDED
    assert runner.collect_batches(jobs) == {}
    for key in keys:
        record = runner.cache.first_valid(key)
        assert record["mode"] == "batch" and record["tier"] == PAID and record["batch_job"] == names[0]
    assert runner.collect_batches(jobs) == {}  # recolhido uma vez só
    assert [c for c in backend.calls if c[1].startswith("batch:")] == [(PAID, "batch:" + FLASH)]


def test_failed_batch_is_resubmitted_by_run_batch(tmp_path):
    backend = FakeBackend()
    runner = _runner(tmp_path, backend)
    jobs = tmp_path / "jobs.jsonl"
    keys = [runner.cache.register(_request(user="t"))]
    first = runner.submit_batch(keys, jobs, "piloto")[0]
    backend.jobs[first].state = types.JobState.JOB_STATE_FAILED
    backend.jobs[first].dest = None
    original_create = backend.batch_create

    def create_done(model, inlined, display_name):
        job = original_create(model, inlined, display_name)
        job.state = types.JobState.JOB_STATE_SUCCEEDED
        return job

    backend.batch_create = create_done
    runner.run_batch(keys, jobs, "piloto", poll_seconds=0)
    assert runner.cache.first_valid(keys[0])["batch_job"] != first


def test_conversations_become_gemini_turns():
    request = {**_request(), "user": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                      {"role": "user", "content": "c"}]}  # fmt: skip
    turns = contents_of(request)
    assert [t.role for t in turns] == ["user", "model", "user"] and turns[1].parts[0].text == "b"
    assert contents_of(_request()) == "tabela"
