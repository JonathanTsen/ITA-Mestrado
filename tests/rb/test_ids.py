"""Testes do passo 30: identificadores e sementes estáveis entre processos."""

import subprocess
import sys

import numpy as np

from missdetect.rb.ids import child_rng, child_seed, stable_id

SCRIPT = (
    "from missdetect.rb.ids import child_rng, child_seed, stable_id;"
    "print(stable_id({'b': 2, 'a': [1, 'x']}), child_seed(7, 'fam', 3), child_rng(7, 'fam').integers(0, 10**9))"
)


def test_stable_id_ignores_key_order():
    assert stable_id({"a": 1, "b": [1, 2]}) == stable_id({"b": [1, 2], "a": 1})
    assert stable_id({"a": 1}) != stable_id({"a": 2})


def test_child_seed_depends_on_every_key():
    assert child_seed(1, "fam", 0) != child_seed(1, "fam", 1)
    assert child_seed(1, "fam", 0) != child_seed(2, "fam", 0)
    assert child_seed(1, "fam", 0) == child_seed(1, "fam", 0)


def test_seed_reproducible_across_processes():
    outputs = {
        subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True).stdout
        for _ in range(2)
    }
    assert len(outputs) == 1
    sid, seed, draw = outputs.pop().split()
    assert sid == stable_id({"a": [1, "x"], "b": 2})
    assert int(seed) == child_seed(7, "fam", 3)
    assert int(draw) == int(child_rng(7, "fam").integers(0, 10**9))


def test_child_rng_streams_are_independent():
    a = child_rng(0, "x").normal(size=1000)
    b = child_rng(0, "y").normal(size=1000)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.1
