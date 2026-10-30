"""Testes do caderno de respostas do LLM (passo 42, §3; emenda P20): cada pergunta é paga uma vez e nada é apagado."""

import json

import pytest

from missdetect.rb.llm.cache import MAX_ATTEMPTS, ResponseCache, parse_probabilities, request_key

REQUEST = {"model": "gemini-3.8-flash", "system": "s", "user": "u", "schema": {"type": "object"}, "thinking": "medium"}


def test_key_is_canonical_and_covers_every_field():
    assert request_key(REQUEST) == request_key(dict(reversed(REQUEST.items())))
    for field, value in [("user", "u2"), ("model", "gemini-4-argon"), ("thinking", "low"), ("schema", {})]:
        assert request_key({**REQUEST, field: value}) != request_key(REQUEST)
    assert request_key({**REQUEST, "replicate": 1}) != request_key(REQUEST)


def test_store_is_write_once_and_keeps_the_request(tmp_path):
    cache = ResponseCache(tmp_path)
    key = cache.register(REQUEST)
    assert json.loads((tmp_path / key[:2] / key / "request.json").read_text()) == REQUEST
    cache.store(key, 0, {"text": "{}", "valid": False, "invalid_reason": "x"})
    with pytest.raises(FileExistsError):
        cache.store(key, 0, {"text": "outra", "valid": True})
    assert cache.register(REQUEST) == key  # registrar de novo não sobrescreve


def test_first_valid_attempt_counts_and_attempts_are_limited(tmp_path):
    cache = ResponseCache(tmp_path)
    key = cache.register(REQUEST)
    assert cache.next_attempt(key) == 0 and cache.first_valid(key) is None
    cache.store(key, 0, {"valid": False, "invalid_reason": "json"})
    assert cache.next_attempt(key) == 1
    cache.store(key, 1, {"valid": True, "probabilities": [0.2, 0.3, 0.5]})
    cache.store(key, 2, {"valid": True, "probabilities": [0.9, 0.05, 0.05]})
    assert cache.first_valid(key)["probabilities"] == [0.2, 0.3, 0.5]
    assert cache.next_attempt(key) is None


def test_attempts_exhausted_after_max_invalid(tmp_path):
    cache = ResponseCache(tmp_path)
    key = cache.register(REQUEST)
    for attempt in range(MAX_ATTEMPTS):
        cache.store(key, attempt, {"valid": False, "invalid_reason": "json"})
    assert cache.next_attempt(key) is None and cache.first_valid(key) is None
    assert MAX_ATTEMPTS == 3  # a primeira resposta mais até 2 retentativas por saída inválida


def test_digest_changes_when_anything_is_added(tmp_path):
    cache = ResponseCache(tmp_path)
    empty = cache.digest()
    key = cache.register(REQUEST)
    after_request = cache.digest()
    cache.store(key, 0, {"valid": True})
    assert len({empty, after_request, cache.digest()}) == 3


@pytest.mark.parametrize(
    ("text", "expected", "reason"),
    [
        ('{"MCAR": 0.2, "MAR": 0.3, "MNAR": 0.5}', (0.2, 0.3, 0.5), None),
        ('{"MCAR": 0.33, "MAR": 0.33, "MNAR": 0.33}', (1 / 3, 1 / 3, 1 / 3), None),
        ('{"MAR": 0.3, "MNAR": 0.5, "MCAR": 0.2}', (0.2, 0.3, 0.5), None),
        ("não é json", None, "json"),
        ('{"MCAR": 0.2, "MAR": 0.3}', None, "chaves"),
        ('{"MCAR": 0.2, "MAR": 0.3, "MNAR": 0.5, "x": 1}', None, "chaves"),
        ('{"MCAR": -0.1, "MAR": 0.6, "MNAR": 0.5}', None, "faixa"),
        ('{"MCAR": 0.5, "MAR": 0.5, "MNAR": 0.5}', None, "soma"),
        ('{"MCAR": "0.2", "MAR": 0.3, "MNAR": 0.5}', None, "número"),
        ("", None, "vazia"),
    ],
)
def test_parse_probabilities(text, expected, reason):
    probabilities, why = parse_probabilities(text)
    if expected is None:
        assert probabilities is None and why is not None and reason in why
    else:
        assert why is None and probabilities == pytest.approx(expected)
