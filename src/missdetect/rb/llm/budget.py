"""Orçamento do LLM (pré-registro §4, emenda P22; passo 42, §5 e §6; 30/10/2026).

- **Preços:** US$ por milhão de tokens, com o raciocínio cobrado como saída, conferidos nas docs do Google em 29/10/2026
  (D5). Argon 2/10 até 30/11/2026 e 4/20 depois; 3.8 Flash 0,75/3,75 até 31/12/2026 e 1,50/7,50 depois; o lote custa
  metade, e o plano gratuito não cobra.
- **Gasto:** somado das respostas gravadas no caderno, pelos tokens que o provedor informou em cada uma.
- **Trava (P22):** antes de uma execução, gasto + 1,5 × projeção não pode passar do teto de US$ 300. A projeção usa a
  contagem de tokens de entrada (`countTokens` do Flash, gratuito, também para as perguntas do Argon) e a média de
  tokens de saída já observada para o modelo e o braço (ou uma suposição declarada, antes do piloto).
"""

import json
from datetime import date, datetime
from pathlib import Path

CAP_USD = 300.0
MARGIN = 1.5
BATCH_FACTOR = 0.5
COUNTING_MODEL = "gemini-3.8-flash"
DEFAULT_OUTPUT_TOKENS = 4_000  # suposição antes do piloto (D5): ≈ 4 mil tokens de raciocínio por chamada
PRICES: dict[str, tuple[tuple[date | None, float, float], ...]] = {
    "gemini-4-argon": ((date(2026, 11, 30), 2.0, 10.0), (None, 4.0, 20.0)),
    "gemini-3.8-flash": ((date(2026, 12, 31), 0.75, 3.75), (None, 1.50, 7.50)),
}


class BudgetExceeded(RuntimeError):
    pass


def price(model: str, when: date) -> tuple[float, float]:
    for until, price_in, price_out in PRICES[model]:
        if until is None or when <= until:
            return price_in, price_out
    raise ValueError(f"sem preço para {model} em {when}")


def call_cost(
    model: str, when: date, input_tokens: float, output_tokens: float, *, batch: bool, tier: str = "paid"
) -> float:
    if tier == "free":
        return 0.0
    price_in, price_out = price(model, when)
    factor = BATCH_FACTOR if batch else 1.0
    return factor * (input_tokens * price_in + output_tokens * price_out) / 1e6


def output_tokens(usage: dict) -> int:
    return int(usage.get("candidates_token_count") or 0) + int(usage.get("thoughts_token_count") or 0)


def response_cost(model: str, record: dict) -> float:
    usage = record.get("usage") or {}
    when = datetime.fromisoformat(record["requested_at"]).date()
    return call_cost(
        model, when, int(usage.get("prompt_token_count") or 0), output_tokens(usage),
        batch=record.get("mode") == "batch", tier=record.get("tier", "paid"),
    )  # fmt: skip


def spent_usd(cache_root: Path) -> float:
    """Gasto de todas as respostas do caderno (qualquer modelo e braço)."""
    total = 0.0
    for attempt in Path(cache_root).rglob("attempt-*.json"):
        request = json.loads((attempt.parent / "request.json").read_text(encoding="utf-8"))
        total += response_cost(request["model"], json.loads(attempt.read_text(encoding="utf-8")))
    return total


def observed_output_tokens(cache_root: Path, model: str, arm: str) -> float | None:
    """Média de tokens de saída (com raciocínio) das respostas já gravadas para o modelo e o braço."""
    values = []
    for attempt in Path(cache_root).rglob("attempt-*.json"):
        request = json.loads((attempt.parent / "request.json").read_text(encoding="utf-8"))
        if request["model"] == model and request["arm"] == arm:
            values.append(output_tokens(json.loads(attempt.read_text(encoding="utf-8")).get("usage") or {}))
    return sum(values) / len(values) if values else None


def projection_usd(
    model: str, when: date, input_tokens: list[int], mean_output_tokens: float, *, batch: bool
) -> float:  # fmt: skip
    return sum(call_cost(model, when, n, mean_output_tokens, batch=batch) for n in input_tokens)


def check_budget(spent: float, projected: float, cap: float = CAP_USD, margin: float = MARGIN) -> None:
    if spent + margin * projected > cap:
        raise BudgetExceeded(
            f"gasto US$ {spent:.2f} + {margin} × projeção US$ {projected:.2f} = US$ {spent + margin * projected:.2f} "
            f"passa do teto de US$ {cap:.0f} (P22: cortar A3, depois o modelo secundário, depois o A2)"
        )


class TokenCounter:
    """`countTokens` do Flash (gratuito), com as contagens guardadas em disco para não repetir chamadas."""

    def __init__(self, backend, store: Path, tier: str):
        self.backend, self.store, self.tier = backend, Path(store), tier
        self._counts: dict[str, int] = json.loads(self.store.read_text()) if self.store.exists() else {}

    def count(self, key: str, text: str) -> int:
        if key not in self._counts:
            self._counts[key] = self.backend.count_tokens(self.tier, COUNTING_MODEL, text)
            self.store.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._counts, sort_keys=True))
            tmp.replace(self.store)
        return self._counts[key]
