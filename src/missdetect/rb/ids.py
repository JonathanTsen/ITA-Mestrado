"""Identificadores estáveis e sementes reprodutíveis (passo 30 do plano).

- `stable_id(spec)`: SHA-256 da especificação canônica (JSON com chaves ordenadas). Nunca use o índice de um
  DataFrame como identificador de instância (ANA-058).
- `child_seed(root, *keys)`: semente derivada por `numpy.random.SeedSequence` a partir de hashes estáveis das
  chaves. Nunca use o `hash()` nativo do Python, que muda entre processos (IND-11).
"""

import hashlib
import json

import numpy as np


def stable_id(spec: dict, length: int = 24) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def child_seed(root_seed: int, *keys: object) -> int:
    key_ints = [int(hashlib.sha256(str(k).encode("utf-8")).hexdigest()[:8], 16) for k in keys]
    sequence = np.random.SeedSequence([root_seed, *key_ints])
    return int(sequence.generate_state(1)[0])


def child_rng(root_seed: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(child_seed(root_seed, *keys))
