"""Descritores-oráculo do painel de informação privilegiada (pré-registro, item 6).

Calculados com o Y completo (lido de `oracle_complete/`), inclusive nas linhas ausentes, por isso só entram no painel
oráculo e nunca são comparados com os métodos observed-only:
- `orc_auc_mask_from_y`: AUC, por CV, de M predito do Y completo;
- `orc_auc_yz_minus_z`: AUC(M | Y, Z) − AUC(M | Z), por CV; quanto Y acrescenta a Z para prever M.

O classificador do painel usa os 28 descritores 34b mais estes dois. Os AUCs usam o mesmo HistGradientBoosting e os
mesmos folds estratificados (5, semente fixa) do descritor `g2_cv_auc_boosting`.
"""

import numpy as np

from missdetect.rb.features import _cv_auc
from missdetect.rb.inject import ObservedInstance

ORACLE_COLUMNS = ("orc_auc_mask_from_y", "orc_auc_yz_minus_z")


def compute(instance: ObservedInstance, y_complete: np.ndarray) -> dict:
    y_obs = np.asarray(instance.y, dtype=float)
    y_complete = np.asarray(y_complete, dtype=float)
    mask = np.isnan(y_obs)
    if not np.allclose(y_complete[~mask], y_obs[~mask]):
        raise ValueError(f"{instance.sample_id}: Y completo não confere com o observado")
    z = instance.z.to_numpy(dtype=float)
    auc_y = float(np.nanmean(_cv_auc(y_complete.reshape(-1, 1), mask, "boosting")))
    auc_yz = float(np.nanmean(_cv_auc(np.column_stack([y_complete, z]), mask, "boosting")))
    auc_z = float(np.nanmean(_cv_auc(z, mask, "boosting")))
    return {"sample_id": instance.sample_id, "orc_auc_mask_from_y": auc_y, "orc_auc_yz_minus_z": auc_yz - auc_z}
