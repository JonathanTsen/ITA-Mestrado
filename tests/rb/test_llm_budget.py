"""Testes do orçamento do LLM (emenda P22): preços datados, gasto pelo caderno e trava do teto."""

from datetime import date

import pytest

from missdetect.rb.llm.budget import (
    BudgetExceeded,
    TokenCounter,
    call_cost,
    check_budget,
    observed_output_tokens,
    price,
    projection_usd,
    spent_usd,
)
from missdetect.rb.llm.cache import ResponseCache


def test_prices_change_on_the_announced_dates():
    assert price("gemini-4-argon", date(2026, 11, 30)) == (2.0, 10.0)
    assert price("gemini-4-argon", date(2026, 12, 1)) == (4.0, 20.0)
    assert price("gemini-3.8-flash", date(2026, 12, 31)) == (0.75, 3.75)
    assert price("gemini-3.8-flash", date(2027, 1, 1)) == (1.50, 7.50)


def test_call_cost_batch_and_free_tier():
    full = call_cost("gemini-4-argon", date(2026, 11, 1), 1e6, 1e6, batch=False)
    assert full == pytest.approx(12.0)
    assert call_cost("gemini-4-argon", date(2026, 11, 1), 1e6, 1e6, batch=True) == pytest.approx(6.0)
    assert call_cost("gemini-3.8-flash", date(2026, 11, 1), 1e6, 1e6, batch=False, tier="free") == 0.0


def _store(cache, model, arm, user, usage, mode="sync", tier="paid"):
    key = cache.register({"model": model, "arm": arm, "user": user})
    cache.store(key, 0, {"usage": usage, "mode": mode, "tier": tier, "requested_at": "2026-10-30T10:00:00+00:00",
                         "valid": True})  # fmt: skip


def test_spent_and_observed_output_come_from_the_cache(tmp_path):
    cache = ResponseCache(tmp_path)
    usage = {"prompt_token_count": 1_000_000, "candidates_token_count": 0, "thoughts_token_count": 1_000_000}
    _store(cache, "gemini-3.8-flash", "A1", "a", usage)
    _store(cache, "gemini-3.8-flash", "A1", "b", usage, mode="batch")
    _store(cache, "gemini-3.8-flash", "A1", "c", usage, tier="free")
    assert spent_usd(tmp_path) == pytest.approx(4.5 + 2.25)
    assert observed_output_tokens(tmp_path, "gemini-3.8-flash", "A1") == 1_000_000
    assert observed_output_tokens(tmp_path, "gemini-4-argon", "A1") is None


def test_check_budget_uses_the_margin():
    check_budget(spent=100, projected=130)  # 100 + 195 = 295
    with pytest.raises(BudgetExceeded, match="P22"):
        check_budget(spent=100, projected=140)  # 100 + 210 = 310
    assert projection_usd("gemini-4-argon", date(2026, 11, 1), [1000, 3000], 500, batch=True) == pytest.approx(
        0.5 * (4000 * 2 + 1000 * 10) / 1e6
    )


class _Counter:
    def __init__(self):
        self.calls = 0

    def count_tokens(self, tier, model, text):
        self.calls += 1
        assert model == "gemini-3.8-flash"  # nunca conta no Argon
        return len(text)


def test_token_counter_counts_once_and_persists(tmp_path):
    backend = _Counter()
    counter = TokenCounter(backend, tmp_path / "tokens.json", tier="paid")
    assert counter.count("k1", "abcd") == 4 and counter.count("k1", "abcd") == 4
    assert TokenCounter(backend, tmp_path / "tokens.json", tier="paid").count("k1", "abcd") == 4
    assert backend.calls == 1
