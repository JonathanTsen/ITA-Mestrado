"""Partições por família de dados (passo 35 do plano).

CV externa: 5 folds formados por famílias inteiras, repetida 3 vezes com atribuições diferentes; CV interna: 4 folds
por família dentro da parte de treino. Nenhuma família aparece ao mesmo tempo em treino e teste.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from missdetect.rb.ids import child_seed


def family_folds(families: pd.Series, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds por família, com atribuição embaralhada e equilibrada pelo número de instâncias."""
    families = families.reset_index(drop=True)
    sizes = families.value_counts()
    if len(sizes) < n_splits:
        raise ValueError(f"{len(sizes)} famílias para {n_splits} folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.sort(sizes.index.to_numpy()))
    load = np.zeros(n_splits)
    fold_of = {}
    for family in sorted(order, key=lambda f: (-sizes[f], list(order).index(f))):
        k = int(np.argmin(load))
        fold_of[family] = k
        load[k] += sizes[family]
    fold = families.map(fold_of).to_numpy()
    idx = np.arange(len(families))
    return [(idx[fold != k], idx[fold == k]) for k in range(n_splits)]


def outer_splits(
    families: pd.Series, root_seed: int, n_splits: int = 5, n_repeats: int = 3
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """(repetição, fold, índices de treino, índices de teste) da CV externa."""
    for repeat in range(n_repeats):
        seed = child_seed(root_seed, "outer", repeat)
        for fold, (train, test) in enumerate(family_folds(families, n_splits, seed)):
            yield repeat, fold, train, test


def inner_splits(train_families: pd.Series, seed: int, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds internos por família, com índices relativos ao conjunto de treino recebido."""
    return family_folds(train_families.reset_index(drop=True), n_splits, seed)
