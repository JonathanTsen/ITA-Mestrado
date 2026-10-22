"""Classificadores direto e hierárquico, calibração e seleção interna (passo 35 do plano).

Toda escolha feita a partir de dados (hiperparâmetros, limiar do roteamento por limiar, zonas do roteamento suave,
método de calibração e o próprio calibrador) usa só a parte de treino, por CV interna por família.
Classes na ordem (MCAR, MAR, MNAR) = (0, 1, 2).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from missdetect.rb.splits import inner_splits

CLASSES = ("MCAR", "MAR", "MNAR")
THRESHOLD_GRID = np.round(np.arange(0.20, 0.81, 0.05), 2)
ZONE_GRID = [(a, b) for a in (0.2, 0.3, 0.4) for b in (0.6, 0.7, 0.8)]


def make_learner(name: str, seed: int, **params) -> BaseEstimator:
    """Aprendizes disponíveis. Imputação e padronização ficam dentro do pipeline (ajustadas só no treino)."""
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params)
    prep = [SimpleImputer(strategy="median"), StandardScaler()]
    if name == "logreg":
        return make_pipeline(*prep, LogisticRegression(max_iter=2000, **params))
    if name == "rf":
        return make_pipeline(*prep, RandomForestClassifier(random_state=seed, n_jobs=1, **params))
    if name == "svm":
        return make_pipeline(*prep, SVC(kernel="rbf", probability=True, random_state=seed, **params))
    if name == "knn":
        return make_pipeline(*prep, KNeighborsClassifier(**params))
    if name == "mlp":
        return make_pipeline(*prep, MLPClassifier(max_iter=500, random_state=seed, **params))
    if name == "nb":
        return make_pipeline(*prep, GaussianNB(**params))
    raise ValueError(name)


def macro_recall(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    recalls = [np.mean(pred[y == c] == c) for c in range(n_classes) if np.any(y == c)]
    return float(np.mean(recalls))


class Calibrated:
    """Calibração por cross-fitting interno: Platt (logística no logit do escore) ou isotônica, um-contra-o-resto."""

    def __init__(self, method: str):
        self.method = method
        self.maps: list = []

    def fit(self, oof_proba: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "Calibrated":
        self.maps = []
        for c in range(oof_proba.shape[1]):
            target = (y == c).astype(int)
            score = np.clip(oof_proba[:, c], 1e-6, 1 - 1e-6)
            if self.method == "platt":
                model = LogisticRegression().fit(
                    np.log(score / (1 - score)).reshape(-1, 1), target, sample_weight=sample_weight
                )
            else:
                model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                    score, target, sample_weight=sample_weight
                )
            self.maps.append(model)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        cols = []
        for c, model in enumerate(self.maps):
            score = np.clip(proba[:, c], 1e-6, 1 - 1e-6)
            if self.method == "platt":
                cols.append(model.predict_proba(np.log(score / (1 - score)).reshape(-1, 1))[:, 1])
            else:
                cols.append(model.predict(score))
        out = np.column_stack(cols)
        if out.shape[1] == 1:
            return out
        out = np.clip(out, 1e-9, None)
        return out / out.sum(axis=1, keepdims=True)


def _weights(sample_weight: np.ndarray | None, rows) -> dict:
    """Argumento `sample_weight` do `fit`, só quando há pesos (sem pesos, a chamada é idêntica à original)."""
    return {} if sample_weight is None else {"sample_weight": sample_weight[rows]}


def _oof(
    learner: BaseEstimator, X: np.ndarray, y: np.ndarray, folds, sample_weight: np.ndarray | None = None
) -> np.ndarray:
    n_classes = int(y.max()) + 1
    oof = np.full((len(y), n_classes), np.nan)
    for train, test in folds:
        model = clone(learner).fit(X[train], y[train], **_weights(sample_weight, train))
        proba = np.zeros((len(test), n_classes))
        proba[:, model.classes_] = model.predict_proba(X[test])
        oof[test] = proba
    return oof


def _log_loss(proba: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    return float(-np.average(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, None)), weights=sample_weight))


@dataclass
class FittedProbabilistic:
    """Aprendiz binário ou multiclasse com calibrador escolhido na CV interna."""

    learner: BaseEstimator
    calibrator: Calibrated | None
    n_classes: int
    choice: dict = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), self.n_classes))
        proba[:, self.learner.classes_] = self.learner.predict_proba(X)
        if self.calibrator is not None:
            proba = self.calibrator.transform(proba)
        return proba


def fit_probabilistic(
    learner: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    families: pd.Series,
    seed: int,
    calibrations=("platt", "isotonic"),
    sample_weight: np.ndarray | None = None,
) -> tuple[FittedProbabilistic, np.ndarray]:
    """Ajusta o aprendiz no treino inteiro e escolhe o calibrador (ou nenhum) pelo log-loss nas predições OOF internas.

    Devolve também as probabilidades OOF internas calibradas (usadas para escolher limiares e zonas). Com
    `sample_weight` (mitigação por pesos da RQ3), os pesos entram em todas as etapas: OOF, calibradores, escolha pelo
    log-loss e modelo final."""
    folds = inner_splits(families, seed)
    oof = _oof(learner, X, y, folds, sample_weight)
    n_classes = oof.shape[1]
    candidates = {"none": (None, oof)}
    for method in calibrations:
        cal_oof = np.zeros_like(oof)
        for train, test in folds:  # cross-fitting: o calibrador de cada fold não vê esse fold
            w = None if sample_weight is None else sample_weight[train]
            cal_oof[test] = Calibrated(method).fit(oof[train], y[train], w).transform(oof[test])
        candidates[method] = (method, cal_oof)
    scores = {name: _log_loss(p, y, sample_weight) for name, (_, p) in candidates.items()}
    best = min(scores, key=scores.get)
    final = clone(learner).fit(X, y, **_weights(sample_weight, slice(None)))
    calibrator = Calibrated(best).fit(oof, y, sample_weight) if best != "none" else None
    return FittedProbabilistic(
        final, calibrator, n_classes, {"calibration": best, "inner_log_loss": scores}
    ), candidates[best][1]


def _route(p_mcar: np.ndarray, p_mar_given_not: np.ndarray, routing: str, params: dict) -> np.ndarray:
    """Probabilidades finais (MCAR, MAR, MNAR) para cada estratégia de roteamento."""
    full = np.column_stack([p_mcar, (1 - p_mcar) * p_mar_given_not, (1 - p_mcar) * (1 - p_mar_given_not)])
    if routing == "fullprob":
        return full
    if routing in ("hard", "threshold"):
        theta = 0.5 if routing == "hard" else params["theta"]
        to_l2 = p_mcar < theta
        out = np.zeros_like(full)
        out[~to_l2, 0] = 1.0
        out[to_l2, 1] = p_mar_given_not[to_l2]
        out[to_l2, 2] = 1 - p_mar_given_not[to_l2]
        return out
    if routing == "soft":
        a, b = params["zones"]
        out = full.copy()
        mcar = p_mcar > b
        l2 = p_mcar < a
        out[mcar] = [1.0, 0.0, 0.0]
        out[l2, 0] = 0.0
        out[l2, 1] = p_mar_given_not[l2]
        out[l2, 2] = 1 - p_mar_given_not[l2]
        return out
    raise ValueError(routing)


@dataclass
class DirectClassifier:
    learner_name: str
    learner_params: dict
    seed: int

    def fit(
        self, X: np.ndarray, y: np.ndarray, families: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "DirectClassifier":
        learner = make_learner(self.learner_name, self.seed, **self.learner_params)
        self.model_, _ = fit_probabilistic(learner, X, y, families, self.seed, sample_weight=sample_weight)
        self.choices_ = dict(self.model_.choice)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)


@dataclass
class HierarchicalClassifier:
    learner_name: str
    learner_params: dict
    routing: str
    seed: int

    def fit(self, X: np.ndarray, y: np.ndarray, families: pd.Series) -> "HierarchicalClassifier":
        families = families.reset_index(drop=True)
        y1 = (y != 0).astype(int)  # 1 = não-MCAR
        self.l1_, oof1 = fit_probabilistic(
            make_learner(self.learner_name, self.seed, **self.learner_params), X, y1, families, self.seed
        )
        not_mcar = np.flatnonzero(y != 0)
        y2 = (y[not_mcar] == 2).astype(int)  # 1 = MNAR
        self.l2_, _ = fit_probabilistic(
            make_learner(self.learner_name, self.seed, **self.learner_params),
            X[not_mcar],
            y2,
            families.iloc[not_mcar],
            self.seed,
        )
        self.params_: dict = {}
        if self.routing in ("threshold", "soft"):
            # Probabilidades OOF internas para todas as instâncias de treino: L1 já tem; L2 nas MCAR é previsto pelo
            # L2 de cada fold interno (cross-fit) para não usar o modelo final.
            p_mcar = oof1[:, 0]
            p_mar = self._l2_oof_all(X, y, families)
            grid = THRESHOLD_GRID if self.routing == "threshold" else ZONE_GRID
            key = "theta" if self.routing == "threshold" else "zones"
            scores = {}
            for value in grid:
                value = tuple(value) if key == "zones" else float(value)
                pred = _route(p_mcar, p_mar, self.routing, {key: value}).argmax(axis=1)
                scores[value] = macro_recall(y, pred, 3)
            self.params_[key] = max(sorted(scores), key=scores.get)
        self.choices_ = {
            "l1_calibration": self.l1_.choice["calibration"],
            "l2_calibration": self.l2_.choice["calibration"],
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params_.items()},
        }
        return self

    def _l2_oof_all(self, X: np.ndarray, y: np.ndarray, families: pd.Series) -> np.ndarray:
        out = np.zeros(len(y))
        for train, test in inner_splits(families, self.seed):
            tr = train[y[train] != 0]
            model = make_learner(self.learner_name, self.seed, **self.learner_params).fit(
                X[tr], (y[tr] == 2).astype(int)
            )
            proba = np.zeros((len(test), 2))
            proba[:, model.classes_] = model.predict_proba(X[test])
            out[test] = proba[:, 0]  # P(MAR | não-MCAR)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_mcar = self.l1_.predict_proba(X)[:, 0]
        p_mar = self.l2_.predict_proba(X)[:, 0]
        return _route(p_mcar, p_mar, self.routing, self.params_)
