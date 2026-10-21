"""MechDetect observed-only (MechDetect-O, adaptação nossa, pré-registrada) e painel oráculo (passo 39).

MechDetect-O usa só o que é observável (Z e a máscara M de Y):
- tarefa "complete" = M predito de [Z, Ŷ(Z)], onde Ŷ é ajustado por cross-fit só nas linhas observadas (o Y nas
  linhas ausentes não existe para o método);
- tarefa "shuffled" = a mesma, com a máscara permutada → teste 1 (não rejeitar → MCAR), como no artigo;
- tarefa "excluded" = M predito de Z sem Ŷ → teste 2. Como Ŷ é função de Z, "complete" ≈ "excluded" por construção:
  o teste 2 quase nunca rejeita e o método tende a decidir MAR entre os não-MCAR. Isso demonstra a
  não-identificabilidade de MAR × MNAR só com dados observados; não é falha de implementação.

Painel oráculo (informação privilegiada, tabela à parte, nunca ranqueado com o observed-only): o MechDetect oficial no
regime "clean", com o Y completo lido de `oracle_complete/` (inclusive nas linhas ausentes), como no artigo.

Em ambos: `HistGradientBoostingClassifier(max_iter=200)`, CV de 10 folds, AUC sobre `predict()`, Mann-Whitney
unilateral com alpha/2 por teste — exatamente a função `mechdetect_aucs` usada na reprodução do passo 36.
"""

import numpy as np
import pandas as pd

from missdetect.rb.baselines.mechdetect_official import (
    CASE_COMPLETE,
    CASE_EXCLUDED,
    CASE_SHUFFLED,
    DEFAULT_ALPHA,
    DEFAULT_CV_FOLDS,
    DEFAULT_SEED,
    MECHANISMS,
    mechdetect_aucs,
    mechdetect_decide,
    mechdetect_pvalues,
)
from missdetect.rb.features import crossfit_prediction
from missdetect.rb.inject import ObservedInstance

YHAT = "__yhat__"
Y_COMPLETE = "__y__"


def _summary(prefix: str, aucs: np.ndarray, alpha: float) -> dict:
    complete, shuffled, excluded = aucs[CASE_COMPLETE], aucs[CASE_SHUFFLED], aucs[CASE_EXCLUDED]
    p_shuffled, p_excluded = mechdetect_pvalues(complete, shuffled, excluded)
    return {
        f"{prefix}_decision": mechdetect_decide(complete, shuffled, excluded, alpha),
        f"{prefix}_p_shuffled": p_shuffled,
        f"{prefix}_p_excluded": p_excluded,
        f"{prefix}_auc_complete": float(np.mean(complete)),
        f"{prefix}_auc_shuffled": float(np.mean(shuffled)),
        f"{prefix}_auc_excluded": float(np.mean(excluded)),
    }


def observed_only(
    instance: ObservedInstance,
    alpha: float = DEFAULT_ALPHA,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict:
    y = np.asarray(instance.y, dtype=float)
    z = instance.z.to_numpy(dtype=float)
    mask = np.isnan(y)
    obs_idx = np.flatnonzero(~mask)
    y_hat, _ = crossfit_prediction(z[obs_idx], y[obs_idx], z, obs_idx)
    frame = pd.DataFrame(z, columns=[f"z{j}" for j in range(z.shape[1])])
    frame[YHAT] = y_hat
    aucs = mechdetect_aucs(frame, pd.Series(mask.astype(int)), YHAT, seed=seed, cv_folds=cv_folds)
    return _summary("mdo", aucs, alpha)


def oracle_clean(
    instance: ObservedInstance,
    y_complete: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """MechDetect oficial, regime clean: a tarefa "complete" vê o Y verdadeiro em todas as linhas."""
    mask = np.isnan(np.asarray(instance.y, dtype=float))
    observed = ~mask
    if not np.allclose(y_complete[observed], np.asarray(instance.y, dtype=float)[observed]):
        raise ValueError(f"{instance.sample_id}: Y completo não confere com o observado")
    frame = pd.DataFrame(instance.z.to_numpy(dtype=float), columns=[f"z{j}" for j in range(instance.z.shape[1])])
    frame[Y_COMPLETE] = y_complete
    aucs = mechdetect_aucs(frame, pd.Series(mask.astype(int)), Y_COMPLETE, seed=seed, cv_folds=cv_folds)
    return _summary("mdoracle", aucs, alpha)


def decision_proba(decisions: pd.Series) -> np.ndarray:
    """Decisões → probabilidades one-hot na ordem (MCAR, MAR, MNAR), para o avaliador único."""
    index = decisions.map({m: i for i, m in enumerate(MECHANISMS)})
    if index.isna().any():
        raise ValueError("decisão desconhecida")
    return np.eye(len(MECHANISMS))[index.to_numpy(dtype=int)]
