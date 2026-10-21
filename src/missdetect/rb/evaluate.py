"""Avaliador único: execução por fold, métricas por família, baselines triviais e auditoria de atalhos (passo 35).

Métricas (Cap2 `sec:evaluation_metrics`; Cap3 `sec:m_protocol`): acurácia, macro-recall (primária), macro-F1,
recall por classe, log-loss, Brier multiclasse, ECE por classe (10 bins de igual massa), AUC um-contra-o-resto.
Calculadas por família e resumidas por média e desvio-padrão entre famílias. Uma instância em que o método falha
conta como erro (probabilidades uniformes e classe prevista -1).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from missdetect.rb.models import CLASSES
from missdetect.rb.splits import outer_splits

N_CLASSES = 3
ECE_BINS = 10


def classwise_ece(proba: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """ECE por classe com bins de igual massa: média, sobre as classes, de Σ_b (n_b/n)|freq_b − conf_b|."""
    eces = []
    for c in range(proba.shape[1]):
        p = proba[:, c]
        target = (y == c).astype(float)
        order = np.argsort(p, kind="stable")
        ece = 0.0
        for bin_idx in np.array_split(order, n_bins):
            if len(bin_idx):
                ece += len(bin_idx) / len(p) * abs(target[bin_idx].mean() - p[bin_idx].mean())
        eces.append(ece)
    return float(np.mean(eces))


def metrics(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    """Métricas de um conjunto de instâncias (tipicamente uma família)."""
    out = {"n": len(y), "accuracy": float(np.mean(pred == y))}
    recalls = {}
    for c, name in enumerate(CLASSES):
        recalls[name] = float(np.mean(pred[y == c] == c)) if np.any(y == c) else np.nan
        out[f"recall_{name}"] = recalls[name]
    out["macro_recall"] = float(np.nanmean(list(recalls.values())))
    out["macro_f1"] = float(f1_score(y, pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0))
    p = np.clip(proba, 1e-12, 1.0)
    out["log_loss"] = float(-np.mean(np.log(p[np.arange(len(y)), y])))
    onehot = np.eye(N_CLASSES)[y]
    out["brier"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    out["ece_classwise"] = classwise_ece(proba, y)
    for c, name in enumerate(CLASSES):
        target = (y == c).astype(int)
        out[f"auc_ovr_{name}"] = float(roc_auc_score(target, proba[:, c])) if 0 < target.sum() < len(y) else np.nan
    return out


def summarize_by_family(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas por família (média das repetições) e resumo média ± dp entre famílias."""
    rows = []
    for (method, repeat, family), g in pred.groupby(["method", "repeat", "family_id"]):
        proba = g[[f"p_{c}" for c in CLASSES]].to_numpy()
        m = metrics(g.y_true.to_numpy(), proba, g.y_pred.to_numpy())
        rows.append({"method": method, "repeat": repeat, "family_id": family, **m})
    per_repeat = pd.DataFrame(rows)
    per_family = per_repeat.drop(columns="repeat").groupby(["method", "family_id"], as_index=False).mean()
    numeric = per_family.drop(columns=["family_id"]).groupby("method")
    summary = numeric.mean().add_suffix("_mean").join(numeric.std(ddof=1).add_suffix("_sd"))
    return per_family, summary


def run_method(
    name: str,
    factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    families: pd.Series,
    sample_ids: pd.Series,
    root_seed: int,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> tuple[pd.DataFrame, list[dict]]:
    """Executa um método na CV externa. Devolve as predições de todas as instâncias e as escolhas internas por fold."""
    families = families.reset_index(drop=True)
    rows, choices = [], []
    for repeat, fold, train, test in outer_splits(families, root_seed, n_splits, n_repeats):
        status = "ok"
        try:
            model = factory().fit(X[train], y[train], families.iloc[train])
            proba = model.predict_proba(X[test])
            pred = proba.argmax(axis=1)
            choices.append({"method": name, "repeat": repeat, "fold": fold, **getattr(model, "choices_", {})})
        except Exception as exc:  # falha conta como erro, nunca é descartada (teste 7 do passo 30)
            status = f"failed: {type(exc).__name__}"
            proba = np.full((len(test), N_CLASSES), 1 / N_CLASSES)
            pred = np.full(len(test), -1)
        for i, row in enumerate(test):
            rows.append(
                {
                    "method": name,
                    "repeat": repeat,
                    "fold": fold,
                    "sample_id": sample_ids.iloc[row],
                    "family_id": families.iloc[row],
                    "y_true": int(y[row]),
                    "y_pred": int(pred[i]),
                    **{f"p_{c}": float(proba[i, k]) for k, c in enumerate(CLASSES)},
                    "status": status,
                }
            )
    return pd.DataFrame(rows), choices


class _Trivial:
    def __init__(self, kind: str, seed: int):
        self.kind, self.seed = kind, seed

    def fit(self, X, y, families):
        self.prior_ = np.bincount(y, minlength=N_CLASSES) / len(y)
        return self

    def predict_proba(self, X):
        n = len(X)
        if self.kind == "uniform":
            return np.full((n, N_CLASSES), 1 / N_CLASSES)
        if self.kind == "majority":
            out = np.zeros((n, N_CLASSES))
            out[:, int(np.argmax(self.prior_))] = 1.0
            return out
        rng = np.random.default_rng(self.seed)  # "prior": sorteia classes com as prevalências do treino
        draws = rng.choice(N_CLASSES, size=n, p=self.prior_)
        return np.eye(N_CLASSES)[draws] * 0.999 + 0.001 / N_CLASSES


def trivial_factory(kind: str, seed: int) -> Callable[[], _Trivial]:
    return lambda: _Trivial(kind, seed)


def check_pairing(pred: pd.DataFrame) -> bool:
    """Teste 3 do passo 30: todos os métodos têm exatamente o mesmo conjunto de (sample_id, repeat, fold)."""
    keys = {m: set(map(tuple, g[["sample_id", "repeat", "fold"]].to_numpy())) for m, g in pred.groupby("method")}
    first = next(iter(keys.values()))
    return all(k == first for k in keys.values())
