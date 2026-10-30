"""Caderno de respostas do LLM em disco (passo 42, §3 e §6; pré-registro §4, emenda P20; 30/10/2026).

- **Chave:** SHA-256 do JSON canônico da pergunta inteira (modelo, instruções, mensagem, esquema, nível de raciocínio,
  versão do renderizador, repetição). O plano da chave (gratuito ou pago) não entra: é a mesma pergunta.
- **Leiaute:** `<raiz>/<chave[:2]>/<chave>/request.json` mais uma `attempt-<i>.json` por resposta recebida.
- **Nunca apagado nem sobrescrito:** cada arquivo é gravado uma vez, de forma atômica (temporário + `os.link`, que
  falha se o destino existir). Uma execução interrompida retoma sem pagar de novo.
- **Regra da resposta (P20):** vale a primeira tentativa válida; até 2 retentativas, só para saída inválida. Sem resposta
  válida em 3 tentativas, a instância é falha e conta como erro.
- O SHA-256 do caderno inteiro (`digest`) entra no registro externo.
"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

CLASSES = ("MCAR", "MAR", "MNAR")
MAX_ATTEMPTS = 3
SUM_TOLERANCE = 0.05


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def request_key(request: dict) -> str:
    return hashlib.sha256(_canonical(request)).hexdigest()


def parse_probabilities(text: str | None) -> tuple[tuple[float, float, float] | None, str | None]:
    """As três probabilidades, normalizadas para somar 1, ou o motivo da invalidez.

    Válida: um objeto JSON com exatamente as chaves MCAR, MAR e MNAR, números finitos em [0, 1] cuja soma difere de 1 em
    no máximo 0,05 (arredondamentos como 0,33 × 3)."""
    if not text or not text.strip():
        return None, "resposta vazia"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "não é json"
    if not isinstance(payload, dict) or set(payload) != set(CLASSES):
        return None, "chaves diferentes de MCAR, MAR e MNAR"
    values = [payload[c] for c in CLASSES]
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        return None, "valor que não é número"
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        return None, "valor fora da faixa [0, 1]"
    total = sum(values)
    if abs(total - 1) > SUM_TOLERANCE:
        return None, f"soma {total:.3f} longe de 1"
    a, b, c = (v / total for v in values)
    return (a, b, c), None


def _write_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} já existe; o caderno de respostas nunca é sobrescrito")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical(payload))
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


class ResponseCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, key: str) -> Path:
        return self.root / key[:2] / key

    def register(self, request: dict) -> str:
        """Grava a pergunta (uma vez) e devolve a chave."""
        key = request_key(request)
        path = self._dir(key) / "request.json"
        if not path.exists():
            _write_once(path, request)
        return key

    def request(self, key: str) -> dict:
        return json.loads((self._dir(key) / "request.json").read_text(encoding="utf-8"))

    def store(self, key: str, attempt: int, record: dict) -> None:
        if not 0 <= attempt < MAX_ATTEMPTS:
            raise ValueError(f"tentativa {attempt} fora de 0..{MAX_ATTEMPTS - 1}")
        _write_once(self._dir(key) / f"attempt-{attempt}.json", {**record, "key": key, "attempt": attempt})

    def attempts(self, key: str) -> list[dict]:
        out = []
        for attempt in range(MAX_ATTEMPTS):
            path = self._dir(key) / f"attempt-{attempt}.json"
            if not path.exists():
                break
            out.append(json.loads(path.read_text(encoding="utf-8")))
        return out

    def first_valid(self, key: str) -> dict | None:
        return next((a for a in self.attempts(key) if a.get("valid")), None)

    def next_attempt(self, key: str) -> int | None:
        """Índice da próxima tentativa a pedir, ou None se já há resposta válida ou as tentativas acabaram."""
        attempts = self.attempts(key)
        if any(a.get("valid") for a in attempts) or len(attempts) >= MAX_ATTEMPTS:
            return None
        return len(attempts)

    def digest(self) -> str:
        """SHA-256 do caderno inteiro (caminhos relativos e conteúdos, em ordem), para o registro externo."""
        h = hashlib.sha256()
        for path in sorted(p for p in self.root.rglob("*.json") if not p.name.startswith(".tmp-")):
            h.update(str(path.relative_to(self.root)).encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
        return h.hexdigest()
