"""Projeção do orçamento do LLM (P22) sem nenhuma chamada: gasto do caderno + estimativas por lote, contra o teto global.

Refaz, de forma reproduzível, a conta de 30/10 (`data_rb/decisoes/orcamento_llm_com_h7_20261030.json`) com a H7
incluída pelo autor em 03/11/2026:
- **gasto:** todas as respostas gravadas em `results_rb/_llm_cache` (`budget.spent_usd`);
- **H7:** 30 bases × 18 instâncias × 2 braços = 1.080 chamadas por modelo, em lote. Entrada = média medida do S0 e do
  S7 nos dois exploratórios (Flash; as instâncias confirmatórias só existem depois da tag). Saída do Flash = média
  medida nesses braços. Saída do Argon: no cenário conservador, a mesma do Flash (o Argon só foi medido no A1, onde
  raciocinou bem menos); no cenário medido, a do teste A1 do Argon;
- **produção sem a H7** (A1, A2, A3 e sondas): as estimativas de 30/10 do adendo, §8 (o corpus confirmatório ainda
  não existe para recontar), e o A1 medido do Argon no cenário medido;
- **datas:** preços de `budget.PRICES` (o Argon dobra em 01/12/2026).

Uso: uv run --frozen python scripts/revisao_banca/llm/projecao_orcamento.py data_rb/decisoes/orcamento_llm_<data>.json
"""

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

from missdetect.rb.llm import budget

ARGON, FLASH = "gemini-4-argon", "gemini-3.8-flash"
CACHE = Path("results_rb/_llm_cache")
H7_CALLS_PER_ARM = 30 * 18
H7_ARMS = ("ctx-S0", "ctx-S7")
# Adendo LLM, §8 (projeção de 30/10, com os tokens do piloto; Argon com o raciocínio do Flash).
BASE_ARGON = {"A1": 26.0, "A2": 3.0, "A3": 27.0, "sondas": 12.0}
BASE_FLASH = 23.0
ARGON_A1_MEASURED_594 = 4.2021738  # A1 do Argon extrapolado do teste de 30 perguntas para 594 chamadas (30/10)
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing (conferido em 30/10/2026; reconferir antes da execução)"


def usage_by_arm(cache: Path) -> dict[tuple[str, str], dict]:
    """Médias de entrada e de saída (com raciocínio) por (modelo, braço), de todas as respostas do caderno."""
    values: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for attempt in cache.rglob("attempt-*.json"):
        request = json.loads((attempt.parent / "request.json").read_text(encoding="utf-8"))
        usage = json.loads(attempt.read_text(encoding="utf-8")).get("usage") or {}
        pair = (int(usage.get("prompt_token_count") or 0), budget.output_tokens(usage))
        values.setdefault((request["model"], request["arm"]), []).append(pair)
    return {
        key: {"n": len(v), "input": sum(a for a, _ in v) / len(v), "output": sum(b for _, b in v) / len(v)}
        for key, v in values.items()
    }


def h7_cost(model: str, when: date, inputs: dict[str, float], outputs: dict[str, float]) -> float:
    return sum(
        H7_CALLS_PER_ARM * budget.call_cost(model, when, inputs[arm], outputs[arm], batch=True) for arm in H7_ARMS
    )


def scenario(name: str, when: date, spent: float, argon: dict[str, float], flash: dict[str, float]) -> dict:
    future = sum(argon.values()) + sum(flash.values())
    total = spent + budget.MARGIN * future
    return {
        "cenario": name, "data_precos": when.isoformat(), "argon_futuro_usd": argon, "flash_futuro_usd": flash,
        "futuro_usd": future, "gasto_mais_margem_usd": total, "cabe_no_teto": total <= budget.CAP_USD,
    }  # fmt: skip


def project(cache: Path, today: date) -> dict:
    spent = budget.spent_usd(cache)
    usage = usage_by_arm(cache)
    inputs = {arm: _mean(usage, FLASH, arm, "input") for arm in H7_ARMS}
    flash_out = {arm: _mean(usage, FLASH, arm, "output") for arm in H7_ARMS}
    argon_a1_out = usage[(ARGON, "A1")]["output"]
    scenarios = []
    for when, label in ((date(2026, 11, 30), "ate_20261130"), (date(2026, 12, 1), "dezembro_2026")):
        doubling = budget.price(ARGON, when)[0] / budget.price(ARGON, date(2026, 11, 30))[0]
        base_argon = {k: v * doubling for k, v in BASE_ARGON.items()}
        scenarios.append(scenario(
            f"conservador_{label}", when, spent,
            base_argon | {"H7": h7_cost(ARGON, when, inputs, flash_out)},
            {"base": BASE_FLASH, "H7": h7_cost(FLASH, when, inputs, flash_out)},
        ))  # fmt: skip
        measured = dict.fromkeys(H7_ARMS, argon_a1_out)
        scenarios.append(scenario(
            f"argon_medido_{label}", when, spent,
            base_argon | {"A1": ARGON_A1_MEASURED_594 * doubling, "H7": h7_cost(ARGON, when, inputs, measured)},
            {"base": BASE_FLASH, "H7": h7_cost(FLASH, when, inputs, flash_out)},
        ))  # fmt: skip
    return {
        "data": today.isoformat(), "fonte_precos": PRICE_SOURCE, "precos_usd_por_milhao": budget.PRICES,
        "teto_usd": budget.CAP_USD, "margem": budget.MARGIN, "gasto_usd_todas_configs": spent,
        "h7": {"chamadas_por_modelo": 2 * H7_CALLS_PER_ARM, "entrada_media_tokens": inputs,
               "saida_media_flash_tokens": flash_out, "saida_media_argon_A1_medida_tokens": argon_a1_out},
        "cenarios": scenarios,
        "premissas": [
            "Sem chamadas: custo por tokens, não fatura.",
            "Produção sem a H7: estimativas de 30/10 do adendo, §8 (Argon 68 = A1 26 + A2 3 + A3 27 + sondas 12; "
            "Flash 23); recontar com o corpus e os prompts finais antes da execução.",
            "H7: entrada e saída do Flash medidas nos dois exploratórios (S0, S7); Argon conservador com a saída do "
            "Flash, medido com a saída do teste A1 do Argon.",
            "Em dezembro o Argon dobra de preço; o Flash não muda até 31/12/2026.",
            "Executar os lotes pagos em sequência: a trava não reserva o custo de jobs pendentes.",
            "Ordem de cortes já registrada (P22) se não couber: A3, depois o modelo secundário, depois o A2.",
        ],
    }  # fmt: skip


def _mean(usage: dict, model: str, arm: str, field: str) -> float:
    rows = [v for (m, a), v in usage.items() if m == model and a == arm]
    n = sum(r["n"] for r in rows)
    return sum(r[field] * r["n"] for r in rows) / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    result = project(args.cache, datetime.now(UTC).date())
    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
