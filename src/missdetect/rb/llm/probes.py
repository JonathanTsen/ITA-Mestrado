"""Sondas de memorização (passo 42, §3.4; pré-registro §4, emenda P21; 30/10/2026).

Adapta os testes do `tabmemcheck` 0.1.6 (Bordt et al., 2024, *Elephants never forget*) ao Gemini, pelo caderno de
respostas, com raciocínio `low`. A temperatura e o `max_tokens` dos testes são ignorados: os modelos não aceitam
temperatura, e um teto pequeno seria gasto no raciocínio e a resposta viria vazia.

**Bateria, por tabela e modelo** (números de perguntas fixados de antemão e reduzidos, passo 42 §6):
- **cabeçalho:** 4 cortes (linhas 2, 4, 6 e 8); mede quantos caracteres o modelo continua literalmente;
- **linhas:** 10 perguntas; conta as linhas completadas literalmente;
- **atributo:** 10 perguntas, no atributo com mais valores distintos; conta os valores exatos;
- **nome:** as 5 primeiras linhas, com o cabeçalho, e "qual é o nome desta base?" (sonda própria: a do pacote não devolve
  o resultado). Acerto = a resposta contém o nome do OpenML, normalizado; a resposta crua fica para conferência.

**Aplicação dupla (P21):** à tabela crua, com os nomes reais (covariável), e à renderização cega do A1 (critério de
exclusão da família na H4). Na renderização cega, as linhas são sorteadas e embaralhadas: completar a próxima linha é
impossível mesmo para quem memorizou a base, e o que conta ali é a sonda de nome.

**Duas passadas, para permitir o lote:** (1) *gravar*: os testes rodam com respostas vazias só para registrar as
perguntas no caderno (as sementes fixas garantem as mesmas perguntas na repetição); (2) as perguntas são respondidas,
no modo síncrono ou em lote; (3) *ler*: os testes rodam de novo e leem as respostas do caderno.

Os limiares de "reconhecida" (`THRESHOLDS`) são propostos aqui e congelados no adendo `prereg-v1-llm`.
"""

import contextlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tabmemcheck import functions as tm
from tabmemcheck.llm import LLM_Interface

from missdetect.rb.ids import child_seed
from missdetect.rb.llm.cache import ResponseCache
from missdetect.rb.llm.client import build_request
from missdetect.rb.llm.render import blind_table
from missdetect.rb.runs import read_observed

PROBES_VERSION = "1"
THINKING_LEVEL = "low"
MAX_OUTPUT_TOKENS = 16_384
HEADER_SPLIT_ROWS = [2, 4, 6, 8]
N_ROW_QUERIES = 10
N_FEATURE_QUERIES = 10
N_NAME_ROWS = 5
THRESHOLDS = {"header_chars": 50, "row_matches": 2, "feature_matches": 3}
# Sem resposta (passada de gravação ou pergunta que falhou): um caractere que nunca aparece numa tabela, para não contar
# acerto e para contornar o `header_test` do tabmemcheck 0.1.6, que falha quando todas as respostas vêm vazias.
NO_ANSWER = "\x00"
NAME_SYSTEM = "You are an expert on public machine learning datasets."
NAME_PROMPT = "Here are the first rows of a dataset in CSV format:\n\n{rows}\n\nWhat is the name of this dataset? Answer with the name only."


def text_validator(text: str | None) -> tuple[str | None, str | None]:
    return (text, None) if text and text.strip() else (None, "resposta vazia")


class ProbeLLM(LLM_Interface):
    """Interface de chat do `tabmemcheck` sobre o caderno de respostas: grava as perguntas e devolve as respostas."""

    chat_mode = True

    def __init__(self, cache: ResponseCache, model: str):
        self.cache, self.model = cache, model
        self.keys: list[str] = []

    def request(self, messages: list[dict]) -> dict:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        return build_request(
            arm="probe", model=self.model, system=system, user=turns, schema=None, thinking_level=THINKING_LEVEL,
            max_output_tokens=MAX_OUTPUT_TOKENS, renderer_version=PROBES_VERSION,
        )  # fmt: skip

    def chat_completion(self, messages, temperature, max_tokens):  # temperatura e max_tokens ignorados (docstring)
        key = self.cache.register(self.request(messages))
        self.keys.append(key)
        record = self.cache.first_valid(key)
        return record["parsed"] if record else NO_ANSWER


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def name_matches(response: str, dataset_name: str) -> bool:
    target = _normalize(dataset_name)
    return bool(target) and f" {target} " in f" {_normalize(response)} "


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def run_battery(csv_file: Path, llm: ProbeLLM, seed: int, dataset_name: str, log) -> dict:
    """Roda a bateria numa tabela. Com o caderno vazio, só grava as perguntas (as contagens saem zeradas)."""
    out: dict = {}
    with contextlib.redirect_stdout(log):
        _, header_true, header_answer = tm.header_test(
            str(csv_file), llm, split_rows=HEADER_SPLIT_ROWS, verbose=False, rng=np.random.default_rng(seed)
        )
        out["header_chars"] = _common_prefix(header_true or "", header_answer or "")
        suffixes, answers = tm.row_completion_test(
            str(csv_file), llm, num_queries=N_ROW_QUERIES, print_levenshtein=False, rng=np.random.default_rng(seed + 1)
        )
        out["row_matches"] = int(
            sum(bool(s.strip()) and s.strip() in a.strip() for s, a in zip(suffixes, answers, strict=True))
        )
        try:
            truth, guesses = tm.feature_completion_test(
                str(csv_file), llm, num_queries=N_FEATURE_QUERIES, rng=np.random.default_rng(seed + 2)
            )
            out["feature_matches"] = int(
                sum(str(t).strip() == str(g).strip() for t, g in zip(truth, guesses, strict=True))
            )
        except (ValueError, KeyError, IndexError):  # respostas vazias da passada de gravação não se deixam analisar
            out["feature_matches"] = 0
    rows = Path(csv_file).read_text(encoding="utf-8").splitlines()[: N_NAME_ROWS + 1]
    messages = [{"role": "system", "content": NAME_SYSTEM},
                {"role": "user", "content": NAME_PROMPT.format(rows="\n".join(rows))}]  # fmt: skip
    answer = llm.chat_completion(messages, temperature=0.0, max_tokens=0)
    out["name_answer"] = answer
    out["name_match"] = name_matches(answer, dataset_name)
    out["recognized"] = bool(
        out["name_match"]
        or out["header_chars"] >= THRESHOLDS["header_chars"]
        or out["row_matches"] >= THRESHOLDS["row_matches"]
        or out["feature_matches"] >= THRESHOLDS["feature_matches"]
    )
    return out


def blind_csv(table: pd.DataFrame, path: Path) -> Path:
    """Grava uma renderização cega do A1 como CSV (mesmo texto que o LLM recebe), para a bateria."""
    lines = [",".join(table.columns)]
    for row in table.to_numpy(dtype=float):
        lines.append(",".join("NA" if np.isnan(v) else str(int(v)) for v in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def raw_csv(source_csv: Path, default_target: str | None, path: Path) -> Path:
    """A tabela-fonte o mais perto possível da versão pública: sem a coluna de índice, inteiros sem ".0" e o alvo com o
    nome original do OpenML (o CSV do MechDetect chama o alvo de `target`)."""
    frame = pd.read_csv(source_csv, index_col=0)
    if default_target and "target" in frame.columns and default_target not in frame.columns:
        frame = frame.rename(columns={"target": default_target})
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_float_dtype(values) and values.dropna().apply(float.is_integer).all():
            frame[column] = values.astype("Int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def probe_tables(run_dir: Path, panel: pd.DataFrame, csv_dir: Path, metadata: pd.DataFrame, work: Path,
                 root_seed: int) -> pd.DataFrame:  # fmt: skip
    """Uma tabela crua e uma renderização cega por família do painel.

    A renderização cega é a da primeira instância (por `sample_id`) do contexto do painel com o maior k e a menor taxa,
    escolhida sem olhar o rótulo; é o texto que o A1 recebe."""
    contexts = pd.read_parquet(run_dir / "contexts.parquet").set_index("context_id")
    meta = metadata.set_index("openml_id")
    rows = []
    main = panel[panel.replicate == 0].join(contexts[["openml_id", "k", "rate"]], on="context_id")
    for family, group in main.groupby("family_id"):
        openml_id = int(group.openml_id.iloc[0])
        name = str(meta.loc[openml_id, "name"])
        target = meta.loc[openml_id, "default_target"]
        raw = raw_csv(csv_dir / f"{openml_id}.csv", None if pd.isna(target) else str(target),
                      work / "tables" / f"{family}-raw.csv")  # fmt: skip
        chosen = group.sort_values(["k", "rate", "sample_id"], ascending=[False, True, True]).iloc[0]
        seed = child_seed(root_seed, "llm_render", chosen.context_id)
        blind = blind_csv(
            blind_table(read_observed(run_dir, chosen.sample_id), seed), work / "tables" / f"{family}-blind.csv"
        )
        rows.append({"family_id": int(family), "openml_id": openml_id, "name": name, "table": "raw", "path": raw})
        rows.append({"family_id": int(family), "openml_id": openml_id, "name": name, "table": "blind", "path": blind})
    return pd.DataFrame(rows)
