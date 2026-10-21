"""Teste MCAR de Little (1988) para o caso do benchmark: ausência só em Y, auxiliares Z completas (passo 39).

Com um único padrão de ausência além do completo, o padrão é monótono e o estimador de máxima verossimilhança sob a
normal multivariada tem forma fechada (verossimilhança fatorada): μ_Z e Σ_ZZ com todas as linhas; a regressão de Y em
Z com as linhas completas. Não é preciso EM.

Estatística: d² = Σ_j n_j (x̄_j − μ̂_j)ᵀ Σ̂_j⁻¹ (x̄_j − μ̂_j), somando os dois padrões (completo, com as k + 1 variáveis;
Y ausente, com as k auxiliares); sob MCAR, d² ~ χ² com Σ_j p_j − p = k graus de liberdade.

Uso só na tarefa L1 (rejeita MCAR ou não), sem regra ad hoc de três classes (IND-28).
"""

import numpy as np
from scipy import stats

from missdetect.rb.inject import ObservedInstance


def little_mle(y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """μ̂ e Σ̂ (ordem: Y, Z) por verossimilhança fatorada; Σ̂ com denominador n (MLE)."""
    obs = ~np.isnan(y)
    mu_z = z.mean(axis=0)
    s_zz = np.cov(z, rowvar=False, bias=True).reshape(z.shape[1], z.shape[1])
    zc, yc = z[obs], y[obs]
    zc_mean, yc_mean = zc.mean(axis=0), yc.mean()
    s_zz_c = np.cov(zc, rowvar=False, bias=True).reshape(z.shape[1], z.shape[1])
    s_zy_c = ((zc - zc_mean) * (yc - yc_mean)[:, None]).mean(axis=0)
    beta = np.linalg.solve(s_zz_c, s_zy_c)
    resid_var = float(np.mean((yc - yc_mean - (zc - zc_mean) @ beta) ** 2))
    mu_y = yc_mean + (mu_z - zc_mean) @ beta
    s_yz = s_zz @ beta
    s_yy = resid_var + beta @ s_zz @ beta
    p = z.shape[1] + 1
    mu = np.concatenate([[mu_y], mu_z])
    sigma = np.empty((p, p))
    sigma[0, 0] = s_yy
    sigma[0, 1:] = sigma[1:, 0] = s_yz
    sigma[1:, 1:] = s_zz
    return mu, sigma


def little_test(y: np.ndarray, z: np.ndarray) -> tuple[float, int, float]:
    """(d², graus de liberdade, p-valor)."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    miss = np.isnan(y)
    if miss.sum() == 0 or (~miss).sum() <= z.shape[1] + 1:
        return np.nan, z.shape[1], np.nan
    mu, sigma = little_mle(y, z)
    full = np.column_stack([y, z])
    d2 = 0.0
    xa = full[~miss].mean(axis=0) - mu
    d2 += (~miss).sum() * float(xa @ np.linalg.solve(sigma, xa))
    xb = z[miss].mean(axis=0) - mu[1:]
    d2 += miss.sum() * float(xb @ np.linalg.solve(sigma[1:, 1:], xb))
    df = z.shape[1]
    return d2, df, float(stats.chi2.sf(d2, df))


def run(instance: ObservedInstance) -> dict:
    d2, df, p = little_test(instance.y, instance.z.to_numpy(dtype=float))
    return {"little_d2": d2, "little_df": df, "little_p": p}
