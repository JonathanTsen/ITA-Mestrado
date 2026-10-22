"""RQ3: ruído de rótulo controlado no treino e mitigação com Confident Learning (passo 44; pré-registro, item 6).

- Só os rótulos de **treino** são corrompidos; o teste externo usa sempre os rótulos verdadeiros.
- A corrupção é **coerente por contexto** (as instâncias de um contexto trocam juntas, como um erro de rotulagem do
  conjunto de dados) e estratificada por família: em cada família, exatamente round(taxa × nº de contextos) contextos.
  O `round` do Python arredonda empates para o par (com 45 contextos, a taxa 0,10 vira 4/45 = 0,089; uma família com 5
  contextos não recebe ruído a 0,10); a taxa realizada de cada configuração vai para `settings.csv`.
  - simétrica: cada rótulo do contexto vai para uma das outras duas classes com igual probabilidade (um ciclo
    MCAR → MAR → MNAR → MCAR ou o inverso, sorteado por contexto);
  - MAR ↔ MNAR: o MAR e o MNAR do contexto trocam entre si; o MCAR não muda.
- Mitigações, dentro do treino externo, a partir de probabilidades OOF internas (CV por família, o mesmo aprendiz da
  configuração primária, sem calibração):
  - `none`: a configuração primária treinada com os rótulos ruidosos;
  - `weights`: pesos = qualidade do rótulo (`cleanlab.rank.get_label_quality_scores`), com piso de 0,1;
  - `prune`: remove as instâncias sinalizadas por `cleanlab.filter.find_label_issues`.
  A rerrotulagem não é feita (era circular; ANA-039).
- Análise: macro-recall no teste limpo por família (média das sementes de corrupção e das repetições); contrastes
  secundários pré-registrados na taxa de 0,20 (pesos × nenhuma e poda × nenhuma, nos dois tipos; Holm m = 4); curvas
  descritivas com IC por bootstrap de famílias; precisão e recall das flags; e, sem corrupção, a fração sinalizada por
  β e por R²(Y | Z) da base, que mede sobreposição entre classes e não ruído de rótulo (ORA-14).
"""

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from threadpoolctl import threadpool_limits

from missdetect.rb.analyses import bootstrap_mean_ci
from missdetect.rb.evaluate import N_CLASSES, metrics
from missdetect.rb.models import CLASSES, DirectClassifier, _oof, make_learner
from missdetect.rb.splits import inner_splits, outer_splits
from missdetect.rb.stats import compare_family

KINDS = ("symmetric", "mar_mnar")
RATES = (0.10, 0.20, 0.40)
N_SEEDS = 3
TEST_RATE = 0.20
MITIGATIONS = ("none", "weights", "prune")
QUALITY_FLOOR = 0.1
R2_BINS = ((-np.inf, 0.2, "<0,2"), (0.2, 0.5, "0,2–0,5"), (0.5, np.inf, "≥0,5"))
CELL = ["noise_kind", "noise_rate", "mitigation"]
MCAR, MAR, MNAR = 0, 1, 2


def noise_settings(n_seeds: int = N_SEEDS) -> list[tuple[str, float, int]]:
    """(tipo, taxa, semente): uma referência limpa, mais cada tipo × taxa > 0 × semente de corrupção."""
    return [("clean", 0.0, 0)] + [(kind, rate, seed) for kind in KINDS for rate in RATES for seed in range(n_seeds)]


def corrupt_labels(
    y: np.ndarray, contexts: pd.Series, families: pd.Series, rate: float, kind: str, rng: np.random.Generator
) -> np.ndarray:
    """Cópia de `y` com o ruído de rótulo aplicado a contextos inteiros, estratificado por família."""
    if kind not in KINDS:
        raise ValueError(f"tipo de ruído desconhecido: {kind}")
    y = np.asarray(y)
    noisy = y.copy()
    contexts = contexts.reset_index(drop=True)
    families = families.reset_index(drop=True)
    for family in sorted(families.unique()):
        family_contexts = np.sort(contexts[families == family].unique())
        n_hit = round(rate * len(family_contexts))
        if n_hit == 0:
            continue
        chosen = rng.choice(family_contexts, size=n_hit, replace=False)
        for context in sorted(chosen):
            rows = np.flatnonzero((contexts == context).to_numpy())
            if kind == "symmetric":
                noisy[rows] = (y[rows] + rng.integers(1, 3)) % 3
            else:
                noisy[rows] = np.where(y[rows] == MAR, MNAR, np.where(y[rows] == MNAR, MAR, y[rows]))
    return noisy


def check_test_labels(pred: pd.DataFrame, truth: pd.Series) -> None:
    """Confere, no processo principal, que o `y_true` de cada predição é o rótulo verdadeiro da instância."""
    expected = pred.sample_id.map(truth)
    wrong = expected.isna() | (expected.to_numpy() != pred.y_true.to_numpy())
    if wrong.any():
        raise RuntimeError(f"rótulo de teste diferente do verdadeiro em {int(wrong.sum())} predições")


def flag_quality(flagged: np.ndarray, corrupted: np.ndarray, families: pd.Series) -> pd.DataFrame:
    """Precisão e recall das flags contra os rótulos realmente corrompidos, por família (NA sem positivos)."""
    frame = pd.DataFrame(
        {
            "family_id": families.to_numpy(),
            "flagged": np.asarray(flagged, bool),
            "corrupted": np.asarray(corrupted, bool),
        }
    )
    rows = []
    for family, g in frame.groupby("family_id"):
        hits = int((g.flagged & g.corrupted).sum())
        n_flagged, n_corrupted = int(g.flagged.sum()), int(g.corrupted.sum())
        rows.append(
            {
                "family_id": family,
                "n": len(g),
                "n_flagged": n_flagged,
                "n_corrupted": n_corrupted,
                "flagged_fraction": n_flagged / len(g),
                "precision": hits / n_flagged if n_flagged else np.nan,
                "recall": hits / n_corrupted if n_corrupted else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cleanlab_scores(
    X: np.ndarray, y: np.ndarray, families: pd.Series, learner: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(sinalizadas, qualidade do rótulo) de cada instância de treino, a partir das probabilidades OOF internas."""
    oof = _oof(make_learner(learner, seed), X, y, inner_splits(families, seed))
    flagged = find_label_issues(labels=y, pred_probs=oof, n_jobs=1)
    quality = get_label_quality_scores(labels=y, pred_probs=oof)
    return np.asarray(flagged, bool), np.asarray(quality, float)


def fit_mitigated(
    mitigation: str,
    X: np.ndarray,
    y: np.ndarray,
    families: pd.Series,
    learner: str,
    seed: int,
    flagged: np.ndarray,
    quality: np.ndarray,
) -> DirectClassifier:
    """Configuração primária (direto, calibração interna) treinada com uma das mitigações."""
    families = families.reset_index(drop=True)
    model = DirectClassifier(learner, {}, seed)
    if mitigation == "none":
        return model.fit(X, y, families)
    if mitigation == "weights":
        return model.fit(X, y, families, sample_weight=np.maximum(quality, QUALITY_FLOOR))
    if mitigation == "prune":
        keep = ~np.asarray(flagged, bool)
        missing = sorted(set(range(N_CLASSES)) - set(np.unique(y[keep])))
        if missing:
            raise ValueError(f"a poda removeu a classe inteira: {[CLASSES[c] for c in missing]}")
        return model.fit(X[keep], y[keep], families[keep].reset_index(drop=True))
    raise ValueError(mitigation)


def run_setting(
    X: np.ndarray,
    y_true: np.ndarray,
    y_noisy: np.ndarray,
    families: pd.Series,
    sample_ids: pd.Series,
    root_seed: int,
    learner: str,
    model_seed: int,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma configuração de ruído na CV externa da avaliação primária: treino com `y_noisy`, teste com `y_true`.

    Devolve as predições no teste (uma linha por instância, repetição e mitigação; falha conta como erro) e as flags
    de cada instância de treino em cada fold."""
    families = families.reset_index(drop=True)
    sample_ids = sample_ids.reset_index(drop=True)
    preds, flags = [], []
    for repeat, fold, train, test in outer_splits(families, root_seed, n_splits, n_repeats):
        with threadpool_limits(1):
            flagged, quality = cleanlab_scores(X[train], y_noisy[train], families.iloc[train], learner, model_seed)
        flags.append(
            pd.DataFrame(
                {
                    "repeat": repeat, "fold": fold, "sample_id": sample_ids.iloc[train].to_numpy(),
                    "family_id": families.iloc[train].to_numpy(), "corrupted": y_noisy[train] != y_true[train],
                    "flagged": flagged, "quality": quality,
                }
            )
        )  # fmt: skip
        for mitigation in MITIGATIONS:
            status = "ok"
            try:
                with threadpool_limits(1):
                    model = fit_mitigated(
                        mitigation,
                        X[train],
                        y_noisy[train],
                        families.iloc[train],
                        learner,
                        model_seed,
                        flagged,
                        quality,
                    )
                    proba = model.predict_proba(X[test])
                pred = proba.argmax(axis=1)
            except Exception as exc:  # falha conta como erro, nunca é descartada
                status = f"failed: {type(exc).__name__}: {exc}"
                proba = np.full((len(test), N_CLASSES), 1 / N_CLASSES)
                pred = np.full(len(test), -1)
            preds.append(
                pd.DataFrame(
                    {
                        "mitigation": mitigation, "repeat": repeat, "fold": fold,
                        "sample_id": sample_ids.iloc[test].to_numpy(), "family_id": families.iloc[test].to_numpy(),
                        "y_true": y_true[test], "y_pred": pred,
                        **{f"p_{c}": proba[:, k] for k, c in enumerate(CLASSES)}, "status": status,
                    }
                )
            )  # fmt: skip
    return pd.concat(preds, ignore_index=True), pd.concat(flags, ignore_index=True)


def per_family_noise(pred: pd.DataFrame) -> pd.DataFrame:
    """Métricas no teste limpo por família e célula (tipo, taxa, mitigação), com média das sementes e repetições."""
    rows = []
    keys = [*CELL, "noise_seed", "repeat", "family_id"]
    for key, g in pred.groupby(keys):
        proba = g[[f"p_{c}" for c in CLASSES]].to_numpy()
        rows.append({**dict(zip(keys, key, strict=True)), **metrics(g.y_true.to_numpy(), proba, g.y_pred.to_numpy())})
    table = pd.DataFrame(rows).drop(columns=["noise_seed", "repeat"])
    return table.groupby([*CELL, "family_id"], as_index=False).mean()


def rq3_tests(per_family: pd.DataFrame) -> pd.DataFrame:
    """Secundária RQ3 (pré-registro, item 6): pesos × nenhuma e poda × nenhuma, ruído simétrico e MAR ↔ MNAR, na
    taxa de 0,20; macro-recall no teste limpo por família; Holm m = 4."""
    cell = per_family[np.isclose(per_family.noise_rate, TEST_RATE)]
    table = cell.assign(method=cell.noise_kind + "|" + cell.mitigation)
    hypotheses = {
        f"RQ3_{mitigation}_{kind}": (f"{kind}|{mitigation}", f"{kind}|none")
        for kind in KINDS
        for mitigation in ("weights", "prune")
    }
    return compare_family(table, hypotheses)


def degradation_curves(per_family: pd.DataFrame) -> pd.DataFrame:
    """Macro-recall médio por (tipo, taxa, mitigação), com IC por bootstrap de famílias (descritivo). A referência
    limpa é o ponto de taxa 0 das curvas dos dois tipos."""
    clean = per_family[per_family.noise_kind == "clean"]
    noisy = per_family[per_family.noise_kind != "clean"]
    table = pd.concat([noisy, *(clean.assign(noise_kind=kind) for kind in KINDS)], ignore_index=True)
    rows = []
    for key, g in table.groupby(CELL):
        low, high = bootstrap_mean_ci(g.macro_recall.to_numpy())
        rows.append({**dict(zip(CELL, key, strict=True)), "n_families": len(g),
                     "macro_recall_mean": g.macro_recall.mean(), "ci_low": low, "ci_high": high})  # fmt: skip
    return pd.DataFrame(rows)


def _mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    values = values.dropna()
    if values.empty:
        return np.nan, np.nan, np.nan, 0
    low, high = bootstrap_mean_ci(values.to_numpy())
    return float(values.mean()), low, high, len(values)


def flag_summary(flags: pd.DataFrame) -> pd.DataFrame:
    """Precisão, recall e fração sinalizada das flags por (tipo, taxa): por família (média dos folds, repetições e
    sementes em que a família estava no treino) e depois entre famílias, com IC por bootstrap."""
    parts = []
    for key, g in flags.groupby(["noise_kind", "noise_rate", "noise_seed", "repeat", "fold"]):
        q = flag_quality(g.flagged.to_numpy(), g.corrupted.to_numpy(), g.family_id)
        parts.append(q.assign(noise_kind=key[0], noise_rate=key[1]))
    per_family = (
        pd.concat(parts)
        .groupby(["noise_kind", "noise_rate", "family_id"], as_index=False)[["precision", "recall", "flagged_fraction"]]
        .mean()
    )
    rows = []
    for (kind, rate), g in per_family.groupby(["noise_kind", "noise_rate"]):
        row = {"noise_kind": kind, "noise_rate": rate}
        for col in ("precision", "recall", "flagged_fraction"):
            mean, low, high, n = _mean_ci(g[col])
            row |= {f"{col}_mean": mean, f"{col}_ci_low": low, f"{col}_ci_high": high, f"{col}_n_families": n}
        rows.append(row)
    return pd.DataFrame(rows)


def _r2_bin(r2: float) -> str:
    return next(label for low, high, label in R2_BINS if low <= r2 < high)


def overlap_flags(flags: pd.DataFrame, info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ORA-14: sem corrupção, fração de vezes que cada instância é sinalizada, por β e por faixa de R²(Y | Z) da base.

    `info` traz `sample_id`, `beta` e `r2_cv`. Média por família dentro de cada nível e depois entre famílias."""
    clean = flags[flags.noise_kind == "clean"]
    rate = clean.groupby(["sample_id", "family_id"], as_index=False).flagged.mean()
    rate = rate.merge(info, on="sample_id")
    rate["r2_bin"] = rate.r2_cv.map(_r2_bin)
    out = []
    for level in ("beta", "r2_bin"):
        per_family = rate.groupby([level, "family_id"], as_index=False).flagged.mean()
        rows = []
        for value, g in per_family.groupby(level):
            mean, low, high, n = _mean_ci(g.flagged)
            rows.append({level: value, "flagged_fraction_mean": mean, "ci_low": low, "ci_high": high,
                         "n_families": n})  # fmt: skip
        out.append(pd.DataFrame(rows))
    return out[0], out[1]
