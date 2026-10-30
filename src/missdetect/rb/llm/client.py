"""Chamadas ao Gemini pelo SDK oficial `google-genai` (passo 42, §3 e §6; pré-registro §4, emendas P20–P22; 30/10/2026).

- **Autorização (regra do autor, 29/10/2026):** o `gemini-3.8-flash` está autorizado. Qualquer outro modelo (hoje, o
  `gemini-4-argon`) só roda com a autorização explícita do autor, pedida antes de cada execução e passada em
  `authorization`; o texto vai para o log do run. A trava vale para toda chamada com esse modelo, inclusive `countTokens`.
- **Chaves:** lidas do `.env` do ITA-Mestrado: `GEMINI_API_KEY` (plano pago) e, opcional, `GEMINI_API_KEY_FREE` (plano
  gratuito). Nunca são impressas nem gravadas. O gratuito é usado primeiro, só no Flash e só no modo síncrono; quando a
  cota diária acaba, a execução passa para o pago e registra a troca. Cada resposta grava o plano que a deu. No plano
  gratuito, o Google pode usar o conteúdo enviado (dados públicos), o que é declarado na tese.
- **Geração (P20):** nenhum parâmetro de amostragem é enviado (os modelos 3.6+ e 4 ignoram temperatura, top_p e top_k e
  não têm semente). Vão só o nível de raciocínio, o esquema JSON e um teto de saída generoso. Sem ferramentas.
- **Retentativas:** erros transitórios (408, 429, 5xx, rede) são repetidos com espera e não contam como tentativa. Saída
  inválida conta como tentativa (no máximo 3, `cache.MAX_ATTEMPTS`). Erros permanentes (400, 403, 404...) param a
  execução, sem gravar nada.
- **Lote (−50%, só no plano pago):** o nome de cada job é gravado em disco antes de esperar o resultado. Uma nova
  execução recolhe os jobs pendentes antes de submeter outros, sem pagar de novo.
"""

import json
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
from dotenv import dotenv_values
from google import genai
from google.genai import errors, types

from missdetect.rb.llm.cache import ResponseCache, parse_probabilities

FREE, PAID = "free", "paid"
KEY_VARIABLES = {FREE: "GEMINI_API_KEY_FREE", PAID: "GEMINI_API_KEY"}
AUTHORIZED_MODELS = frozenset({"gemini-3.8-flash"})
FREE_TIER_MODELS = frozenset({"gemini-3.8-flash"})
TRANSIENT_CODES = frozenset({408, 429, 500, 502, 503, 504})
REQUEST_TIMEOUT_MS = 600_000
MAX_TRANSIENT_RETRIES = 6
MAX_FREE_WAITS = 10
BACKOFF_SECONDS = (5, 15, 30, 60, 120, 240)
TERMINAL_BATCH_STATES = frozenset(
    {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
     "JOB_STATE_EXPIRED"}
)  # fmt: skip
BATCH_MAX_REQUESTS = 500
BATCH_MAX_CHARS = 12_000_000  # o lote com pedidos embutidos aceita até 20 MB; folga para o JSON da requisição
USAGE_FIELDS = (
    "prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count",
    "cached_content_token_count",
)  # fmt: skip

Validator = Callable[[str | None], tuple[Any, str | None]]
# O esquema vai como `response_schema` (subconjunto OpenAPI): em 30/10/2026, o modo de lote ignorou o
# `response_json_schema` (só manteve o JSON) e respeitou o `response_schema`; no síncrono, os dois são aplicados
# (results_rb/_audit/20261030-schema-lote). O formato de envio faz parte da pergunta (e da chave do caderno).
SCHEMA_TRANSPORT = "openapi"


class AuthorizationRequired(PermissionError):
    pass


def require_authorization(model: str, authorization: str | None) -> None:
    if model not in AUTHORIZED_MODELS and not (authorization or "").strip():
        raise AuthorizationRequired(
            f"{model}: só roda com a autorização explícita do autor, pedida antes de cada execução (regra de 29/10/2026)"
        )


def load_api_keys(env_file: Path) -> dict[str, str]:
    """Chaves presentes no `.env`, por plano. Nunca imprima o retorno."""
    values = dotenv_values(env_file) if Path(env_file).exists() else {}
    return {tier: str(values[var]) for tier, var in KEY_VARIABLES.items() if values.get(var)}


def build_request(
    *, arm: str, model: str, system: str, user: str | list[dict], schema: dict | None, thinking_level: str,
    max_output_tokens: int, renderer_version: str, replicate: int = 0,
) -> dict:  # fmt: skip
    """A pergunta inteira, como entra na chave do caderno de respostas. O formato de envio do esquema só entra quando
    há esquema (as sondas não têm)."""
    request = {
        "arm": arm, "model": model, "system": system, "user": user, "schema": schema,
        "thinking_level": thinking_level, "max_output_tokens": max_output_tokens,
        "renderer_version": renderer_version, "replicate": replicate,
    }  # fmt: skip
    if schema is not None:
        request["schema_transport"] = SCHEMA_TRANSPORT
    return request


def contents_of(request: dict) -> Any:  # str ou list[types.Content]; as uniões do SDK não aceitam list[Content]
    """A mensagem do usuário, ou uma conversa (lista de {"role", "content"}, papéis user/assistant) nas sondas."""
    user = request["user"]
    if isinstance(user, str):
        return user
    return [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in user
    ]


def openapi_schema(schema: dict) -> types.Schema:
    """O subconjunto de JSON Schema que usamos (objeto, número, texto com enum, obrigatórios, faixa) em OpenAPI."""
    kind = {"object": types.Type.OBJECT, "number": types.Type.NUMBER, "integer": types.Type.INTEGER,
            "string": types.Type.STRING, "boolean": types.Type.BOOLEAN}[schema["type"]]  # fmt: skip
    fields: dict[str, Any] = {"type": kind}
    if "properties" in schema:
        fields["properties"] = {name: openapi_schema(sub) for name, sub in schema["properties"].items()}
        fields["property_ordering"] = list(schema["properties"])
    for key in ("required", "enum", "minimum", "maximum"):
        if key in schema:
            fields[key] = schema[key]
    return types.Schema(**fields)


def generation_config(request: dict) -> types.GenerateContentConfig:
    """Só raciocínio, esquema e teto de saída: nenhum parâmetro de amostragem nem ferramenta (P20)."""
    extra: dict[str, Any] = {}
    if request["schema"] is not None:
        if request.get("schema_transport") != SCHEMA_TRANSPORT:
            raise ValueError("pergunta sem o formato de envio do esquema (schema_transport); refaça o registro")
        extra = {"response_mime_type": "application/json", "response_schema": openapi_schema(request["schema"])}
    return types.GenerateContentConfig(
        system_instruction=request["system"],
        thinking_config=types.ThinkingConfig(thinking_level=request["thinking_level"]),
        max_output_tokens=request["max_output_tokens"],
        **extra,
    )


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def response_record(
    response: Any, *, tier: str, mode: str, requested_at: str, received_at: str, validator: Validator,
    batch_job: str | None = None,
) -> dict:  # fmt: skip
    """O que fica no caderno: texto cru, metadados do provedor (versão, id, tokens) e a validação."""
    candidate = (response.candidates or [None])[0]
    parts = candidate.content.parts if candidate and candidate.content and candidate.content.parts else []
    text = "".join(p.text for p in parts if p.text and not p.thought)
    usage = response.usage_metadata
    feedback = response.prompt_feedback
    parsed, reason = validator(text)
    return {
        "tier": tier, "mode": mode, "batch_job": batch_job, "requested_at": requested_at, "received_at": received_at,
        "model_version": response.model_version, "response_id": response.response_id,
        "finish_reason": candidate.finish_reason.value if candidate and candidate.finish_reason else None,
        "block_reason": feedback.block_reason.value if feedback and feedback.block_reason else None,
        "usage": {f: getattr(usage, f, None) for f in USAGE_FIELDS} if usage else {},
        "text": text, "parsed": list(parsed) if isinstance(parsed, tuple) else parsed,
        "valid": reason is None, "invalid_reason": reason,
    }  # fmt: skip


def _error_details(exc: BaseException) -> list[dict]:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        inner = details.get("error", details).get("details", [])
        return [d for d in inner if isinstance(d, dict)]
    return []


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in TRANSIENT_CODES
    return isinstance(exc, httpx.TransportError | TimeoutError | ConnectionError)


def is_daily_quota(exc: BaseException) -> bool:
    """429 por cota diária (não adianta esperar alguns segundos)."""
    for detail in _error_details(exc):
        for violation in detail.get("violations", []) or []:
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def retry_delay(exc: BaseException, default: float) -> float:
    for detail in _error_details(exc):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
        if match:
            return float(match.group(1)) + 1.0
    return default


class Backend(Protocol):
    def tiers(self) -> list[str]: ...
    def generate(self, tier: str, request: dict) -> Any: ...
    def count_tokens(self, tier: str, model: str, text: str) -> int: ...
    def batch_create(self, model: str, inlined: list[types.InlinedRequest], display_name: str) -> Any: ...
    def batch_get(self, name: str) -> Any: ...


class GeminiBackend:
    def __init__(self, keys: dict[str, str]):
        if not keys:
            raise ValueError("nenhuma chave do Gemini no .env (GEMINI_API_KEY ou GEMINI_API_KEY_FREE)")
        options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        self._clients = {tier: genai.Client(api_key=key, http_options=options) for tier, key in keys.items()}

    def tiers(self) -> list[str]:
        return sorted(self._clients)

    def generate(self, tier: str, request: dict) -> Any:
        return self._clients[tier].models.generate_content(
            model=request["model"], contents=contents_of(request), config=generation_config(request)
        )

    def count_tokens(self, tier: str, model: str, text: str) -> int:
        return int(self._clients[tier].models.count_tokens(model=model, contents=text).total_tokens or 0)

    def batch_create(self, model: str, inlined: list[types.InlinedRequest], display_name: str) -> Any:
        config = types.CreateBatchJobConfig(display_name=display_name)
        return self._clients[PAID].batches.create(model=model, src=inlined, config=config)

    def batch_get(self, name: str) -> Any:
        return self._clients[PAID].batches.get(name=name)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class Runner:
    """Resolve perguntas do caderno de respostas com um modelo, no modo síncrono ou em lote."""

    def __init__(
        self, backend: Backend, cache: ResponseCache, model: str, *, authorization: str | None = None,
        validator: Validator = parse_probabilities, max_workers: int = 4, events: Path | None = None,
        sleep: Callable[[float], None] = time.sleep, now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):  # fmt: skip
        require_authorization(model, authorization)
        self.backend, self.cache, self.model = backend, cache, model
        self.validator, self.max_workers, self.events = validator, max_workers, events
        self.sleep, self.now = sleep, now
        tiers = backend.tiers()
        self.tier = FREE if FREE in tiers and model in FREE_TIER_MODELS else PAID
        if self.tier == PAID and PAID not in tiers:
            raise ValueError(f"{model}: precisa da chave do plano pago (GEMINI_API_KEY)")
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _event(self, kind: str, **fields: Any) -> None:
        if self.events is not None:
            with self._lock:
                _append_jsonl(self.events, {"at": _iso(self.now()), "event": kind, "model": self.model, **fields})

    def _switch_to_paid(self, reason: str) -> None:
        with self._lock:
            if self.tier == PAID:
                return
            if PAID not in self.backend.tiers():
                raise RuntimeError(f"cota do plano gratuito acabou ({reason}) e não há chave do plano pago")
            self.tier = PAID
        self._event("tier_switch", to=PAID, reason=reason)

    def _call(self, request: dict) -> tuple[Any, str]:
        transient = free_waits = 0
        while True:
            if self._stop.is_set():
                raise RuntimeError("execução interrompida por erro em outra chamada")
            tier = self.tier
            try:
                return self.backend.generate(tier, request), tier
            except Exception as exc:
                if not is_transient(exc):
                    raise
                code = getattr(exc, "code", None)
                if tier == FREE and code == 429:
                    if is_daily_quota(exc) or free_waits >= MAX_FREE_WAITS:
                        self._switch_to_paid("cota diária" if is_daily_quota(exc) else f"{free_waits} esperas seguidas")
                        continue
                    free_waits += 1
                    self.sleep(retry_delay(exc, 30.0))
                    continue
                if transient >= MAX_TRANSIENT_RETRIES:
                    raise
                self._event("transient_error", code=code, tier=tier, error=type(exc).__name__)
                self.sleep(retry_delay(exc, BACKOFF_SECONDS[min(transient, len(BACKOFF_SECONDS) - 1)]))
                transient += 1

    def _resolve(self, key: str) -> None:
        request = self.cache.request(key)
        while (attempt := self.cache.next_attempt(key)) is not None:
            requested_at = _iso(self.now())
            response, tier = self._call(request)
            record = response_record(
                response, tier=tier, mode="sync", requested_at=requested_at, received_at=_iso(self.now()),
                validator=self.validator,
            )  # fmt: skip
            self.cache.store(key, attempt, record)

    def pending(self, keys: Iterable[str]) -> list[str]:
        return [k for k in keys if self.cache.next_attempt(k) is not None]

    def run_sync(self, keys: Iterable[str]) -> None:
        """Pergunta o que falta, em paralelo; o primeiro erro permanente para tudo (as respostas já gravadas ficam)."""
        todo = self.pending(keys)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._resolve, k) for k in todo]
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            failed = [f for f in done if f.exception() is not None]
            if failed:
                self._stop.set()
                for future in futures:
                    future.cancel()
                raise failed[0].exception()  # type: ignore[misc]

    # --- lote -------------------------------------------------------------------------------------------------------

    def _in_flight(self, jobs_file: Path) -> set[tuple[str, int]]:
        out: set[tuple[str, int]] = set()
        for job in _read_jsonl(jobs_file):
            if not (jobs_file.parent / f"{_safe_name(job['name'])}.collected.json").exists():
                out |= {(k, int(a)) for k, a in job["requests"]}
        return out

    def submit_batch(self, keys: Iterable[str], jobs_file: Path, display_name: str) -> list[str]:
        """Submete o que falta e não está num job pendente. O job é gravado antes de qualquer espera."""
        if PAID not in self.backend.tiers():
            raise ValueError("o modo de lote exige a chave do plano pago (GEMINI_API_KEY)")
        in_flight = self._in_flight(jobs_file)
        pending = [(k, a) for k in keys if (a := self.cache.next_attempt(k)) is not None and (k, a) not in in_flight]
        chunks: list[list[tuple[str, int]]] = []
        size = 0
        for item in pending:
            request = self.cache.request(item[0])
            chars = len(json.dumps(request["user"])) + len(request["system"])
            if not chunks or len(chunks[-1]) >= BATCH_MAX_REQUESTS or size + chars > BATCH_MAX_CHARS:
                chunks.append([])
                size = 0
            chunks[-1].append(item)
            size += chars
        names = []
        for chunk in chunks:
            inlined = []
            for key, attempt in chunk:
                request = self.cache.request(key)
                metadata = {"key": key, "attempt": str(attempt)}
                inlined.append(
                    types.InlinedRequest(
                        contents=contents_of(request), config=generation_config(request), metadata=metadata
                    )
                )
            job = self.backend.batch_create(self.model, inlined, display_name)
            _append_jsonl(
                jobs_file,
                {"name": job.name, "model": self.model, "display_name": display_name, "created_at": _iso(self.now()),
                 "requests": [[k, a] for k, a in chunk]},
            )  # fmt: skip
            self._event("batch_submitted", job=job.name, n_requests=len(chunk))
            names.append(job.name)
        return names

    def collect_batches(self, jobs_file: Path) -> dict[str, str]:
        """Recolhe os jobs terminados (uma vez cada) e devolve {job: estado} dos que ainda não terminaram."""
        running = {}
        for job_record in _read_jsonl(jobs_file):
            name = job_record["name"]
            marker = jobs_file.parent / f"{_safe_name(name)}.collected.json"
            if marker.exists():
                continue
            job = self.backend.batch_get(name)
            state = job.state.value if job.state is not None else "JOB_STATE_UNSPECIFIED"
            if state not in TERMINAL_BATCH_STATES:
                running[name] = state
                continue
            received_at = _iso(job.update_time or self.now())
            stored = failed = 0
            responses = (job.dest.inlined_responses if job.dest else None) or []
            for index, item in enumerate(responses):
                meta = item.metadata or {}
                key, attempt = (meta["key"], int(meta["attempt"])) if "key" in meta else job_record["requests"][index]
                if item.response is None or self.cache.has_attempt(key, int(attempt)):
                    failed += item.response is None
                    continue
                record = response_record(
                    item.response, tier=PAID, mode="batch", requested_at=job_record["created_at"],
                    received_at=received_at, validator=self.validator, batch_job=name,
                )  # fmt: skip
                self.cache.store(key, int(attempt), record)
                stored += 1
            summary = {"name": name, "state": state, "collected_at": _iso(self.now()), "stored": stored,
                       "request_errors": failed}  # fmt: skip
            marker.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            self._event("batch_collected", **{k: v for k, v in summary.items() if k != "name"}, job=name)
        return running

    def run_batch(
        self, keys: list[str], jobs_file: Path, display_name: str, poll_seconds: float = 60.0,
        max_rounds: int = 2 * 3,
    ) -> None:  # fmt: skip
        """Rodadas de lote até todas as perguntas terem resposta válida ou esgotarem as tentativas."""
        rounds = 0
        while True:
            running = self.collect_batches(jobs_file)
            if running:
                self.sleep(poll_seconds)
                continue
            if not self.pending(keys):
                return
            if rounds >= max_rounds:
                raise RuntimeError(
                    f"{len(self.pending(keys))} perguntas sem resposta depois de {rounds} rodadas de lote"
                )
            self.submit_batch(keys, jobs_file, display_name)
            rounds += 1
