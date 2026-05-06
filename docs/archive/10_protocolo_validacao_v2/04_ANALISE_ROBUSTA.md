# Análise exaustiva — Calibração robusta (100/200) e diagnóstico de falhas

**Data:** 2026-05-04
**Arquivos de referência:**
- `data/calibration.json` (robusto), `data/calibration_smoke.json` (smoke)
- `results/v2_robust/real/validacao_rotulos_v2/validacao_v2.csv`
- `results/v2_smoke/real/validacao_rotulos_v2/validacao_v2.csv`
- `data/calibration_progress.csv` (300 sintéticos processados)

---

## 1. Resultados gerais

### Sintéticos

| Modo | Smoke (15/10) | Robusto (100/200) | Δ |
|------|:---:|:---:|:---:|
| Regras default | 51.1% | 50.0% | -1.1pp |
| Regras calibradas | 55.6% | 53.3% | -2.2pp |
| **Bayes** | **95.6%** | **78.3%** | **-17.3pp** |
| **Bayes 5-fold CV** | — | **59.0% ± 6.0%** | — |

### Recall Bayes por classe (sintéticos)

| Classe | Smoke | Robusto | Δ |
|--------|:---:|:---:|:---:|
| MCAR | 93.3% | 76.0% | -17.3pp |
| MAR | 93.3% | 87.0% | -6.3pp |
| MNAR | 100% | 72.0% | **-28.0pp** |

### Confusion matrix Bayes robusto (sintéticos)

```
Predito →      MCAR   MAR   MNAR
Verdadeiro ↓
MCAR            76     2     22
MAR              9    87      4
MNAR            27     1     72
```

**Padrão dominante:** MCAR e MNAR se confundem mutuamente (22 + 27 = 49 erros de 65 totais = 75% dos erros).

### Reais (29 datasets, Bayes robusto)

| | Correto | Errado | Ambíguo (conf < 0.4) |
|--|:--:|:--:|:--:|
| **MCAR (9)** | 2 | 4 | 3 |
| **MAR (11)** | 8 | 1 | 2 |
| **MNAR (9)** | 2 | 5 | 2 |
| **Total** | **12 (41.4%)** | **10 (34.5%)** | **7 (24.1%)** |

---

## 2. As 5 causas-raiz dos erros

### Causa 1 — MCAR ↔ MNAR são estatisticamente indistinguíveis (FUNDAMENTAL)

Os scores das camadas A-C produzem distribuições quase idênticas para MCAR e MNAR nos sintéticos:

| Score | MCAR (média±σ) | MNAR (média±σ) | AUC separação |
|-------|:---:|:---:|:---:|
| auc_obs | 0.483 ± 0.064 | 0.489 ± 0.069 | 0.536 |
| mi_max | 0.006 ± 0.005 | 0.006 ± 0.005 | ~0.50 |
| log10(little_p) | -0.38 | -0.38 | ~0.50 |
| caafe_cond_entropy | 0.28 | 0.29 | ~0.52 |

**Distância Euclidiana no espaço 10-dim:** MCAR↔MNAR = 2.07 ± 1.56 vs MCAR↔MAR = 9.93 ± 7.41.

**Implicação teórica:** Rubin (1976) e Molenberghs (2008) já dizem que MNAR é não-identificável a partir de dados observados. Isso confirma empiricamente: os scores calculados sobre dados observados não distinguem MCAR de MNAR. Isso é irredutível — nenhuma melhoria de bandwidth ou calibração resolve.

### Causa 2 — Dimensões 6-7 (CAAFE) são constantes (zero informação)

No vetor de 10 dimensões:
- **Dimensão 6** (`log1p(caafe_quantile_ratio)`): valor constante 3.045 em 100% dos 300 sintéticos
- **Dimensão 7** (`caafe_tail_asym`): valor constante 0.0 em 100% dos 300 sintéticos

Essas 2 de 10 dimensões não contribuem **nenhuma informação** para o KDE. O classificador opera efetivamente em 8 dimensões, e as 2 dimensões restantes que deveriam capturar MNAR (as únicas evidências positivas de auto-dependência) estão mortas.

Na calibração, `caafe_quantile_ratio` e `caafe_tail_asym` têm AUC = 0.500 (sem poder discriminativo).

**Fixável?** Sim, parcialmente — substituir por scores mais informativos (P5). Mas o sinal intrínseco de MNAR é fraco por definição.

### Causa 3 — Distribuição sintético→real diverge (shift de auc_obs)

| Mecanismo | auc_obs sintético | auc_obs real | Δ |
|-----------|:---:|:---:|:---:|
| MCAR | 0.483 | 0.580 | +0.097 |
| MAR | 0.737 | 0.659 | -0.078 |
| MNAR | 0.489 | 0.550 | +0.061 |

**O threshold calibrado (0.554) funciona nos sintéticos, mas:**
- MCAR reais têm auc_obs **acima** do threshold em 56% dos casos (vs 9% nos sintéticos)
- MNAR reais idem (56% vs 16%)
- Resultado: MCAR e MNAR reais são classificados como MAR

**mi_max tem o mesmo problema invertido:**
- MAR sintético: mi_max = 0.084 ± 0.109
- MAR real: mi_max = 0.036 ± 0.037 (2.3× mais fraco)
- MAR reais ficam abaixo do threshold calibrado em sintéticos

**Causa raiz:** Sintéticos são gerados com mecanismos puros (logístico, limiar, bloco). Reais têm:
- Relações não-lineares confundidas por covariáveis
- Missing rates e distribuições de base diferentes
- Estrutura de correlação entre features mais complexa

### Causa 4 — Treino = teste (sem cross-validation)

O Bayes avalia os KDEs nos **mesmos 300 sintéticos** usados para fittá-los. O 78.3% já inclui overfitting residual. O smoke (95.6% com n=15) era absurdamente otimista pelo mesmo motivo, amplificado pela escassez de dados.

**Resultado com 5-fold CV (2026-05-04):** 59.0% ± 6.0% (reduz 19.3pp do 78.3%). A estimativa anterior de 70–75% estava otimista; o CV confirmou que a confusão MCAR↔MNAR também aparece fora da amostra.

### Causa 5 — MNAR com confiança 1.0 errada (falha catastrófica)

4 datasets MNAR reais foram classificados como MCAR com confiança ≈ 1.0:

| Dataset | P(MCAR) | P(MNAR) | auc_obs | mi_max | caafe_cond_entropy |
|---------|:---:|:---:|:---:|:---:|:---:|
| adult_capitalgain | 1.00 | 2.6e-08 | 0.505 | 0.011 | 0.014 |
| kidney_pot | 1.00 | 5.4e-35 | 0.626 | 0.023 | 0.002 |
| kidney_sod | 1.00 | 7.0e-19 | 0.596 | 0.003 | 0.040 |
| pima_insulin | 0.94 | 0.029 | 0.478 | 0.015 | 0.087 |

**Todos têm scores virtualmente idênticos a MCAR sintético**, porque MNAR em dados observados se manifesta como... MCAR (a variável esconde de si mesma — não há evidência visível). O KDE vê esses pontos na região MCAR do espaço e atribui probabilidade ~1.0.

No smoke (n=15), esses mesmos datasets tinham P(MNAR)≈1.0 — o KDE com poucos pontos era mais "generoso" na zona MNAR. Com n=100, a distribuição MNAR ficou mais apertada e esses pontos caíram fora dela.

---

## 3. Análise dataset-a-dataset (29 reais)

### MCAR (9 datasets → 2 corretos, 22.2%)

| Dataset | Pred | Conf | P(MCAR) | P(MAR) | P(MNAR) | auc_obs | Nota |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| autompg_horsepower | MAR | 0.88 | 0.06 | 0.93 | 0.01 | 0.429 | auc_obs baixo mas MI patterns |
| **breastcancer_barenuclei** | MNAR | **0.02** | 0.23 | 0.38 | 0.39 | 0.641 | AMBÍGUO, trimodal |
| creditapproval_a14 | MAR | 1.00 | ~0 | 1.00 | ~0 | 0.783 | auc_obs alto demais para MCAR |
| cylinderbands_bladepressure | MAR | 1.00 | ~0 | 1.00 | ~0 | 0.760 | idem |
| **cylinderbands_esavoltage** | MNAR | **0.07** | 0.33 | 0.27 | 0.40 | 0.606 | AMBÍGUO |
| **echomonths_epss** | MCAR | **0.21** | 0.57 | 0.08 | 0.35 | 0.473 | correto mas ambíguo |
| **hepatitis_albumin** | MCAR | **0.21** | 0.58 | 0.06 | 0.36 | 0.471 | correto mas ambíguo |
| **hepatitis_alkphosphate** | MNAR | **0.02** | 0.24 | 0.38 | 0.61 | 0.339 | AMBÍGUO |
| hypothyroid_t4u | MAR | 1.00 | ~0 | 1.00 | ~0 | 0.718 | N=3772, Little hipersensível |

**Padrão:** 4/9 MCAR são classificados como MAR com confiança 1.0. Todos têm auc_obs > 0.70. Hipótese: esses "MCAR" têm estrutura de dependência residual nos dados reais que não existe nos MCAR sintéticos puros. Possível rótulo incorreto na literatura.

### MAR (11 datasets → 8 corretos, 72.7%)

| Dataset | Pred | Conf | P(MAR) | auc_obs | mi_max | Nota |
|---------|:---:|:---:|:---:|:---:|:---:|------|
| **airquality_ozone** | MAR | **0.08** | 0.43 | 0.660 | 0.050 | correto mas ambíguo |
| colic_resprate | MCAR | 0.08 | 0.07 | 0.521 | 0.018 | AMBÍGUO, auc_obs baixo |
| hearth_chol | MCAR | 0.66 | 0.001 | 0.414 | 0.0 | mi_max=0 → sem evidência MAR |
| **kidney_hemo** | MNAR | **0.17** | 0.07 | 0.590 | 0.022 | AMBÍGUO |
| mammographic_density | MAR | 1.00 | 1.00 | 0.584 | 0.002 | correto |
| oceanbuoys_airtemp | MAR | 1.00 | 1.00 | 0.916 | 0.105 | melhor score do dataset |
| oceanbuoys_humidity | MAR | 1.00 | 1.00 | 0.899 | 0.100 | idem |
| sick_t3 | MAR | 1.00 | 1.00 | 0.569 | 0.010 | correto |
| sick_tsh | MAR | 1.00 | 1.00 | 0.665 | 0.012 | correto |
| titanic_age | MAR | 1.00 | 1.00 | 0.735 | 0.032 | correto |
| titanic_age_v2 | MAR | 1.00 | 1.00 | 0.699 | 0.050 | correto |

**Padrão:** MAR é o mecanismo mais bem detectado. 7/11 com confiança 1.0. Os 3 erros têm auc_obs < 0.60 ou mi_max ≈ 0 (sinal MAR fraco nos dados reais).

### MNAR (9 datasets → 2 corretos, 22.2%)

| Dataset | Pred | Conf | P(MNAR) | auc_obs | caafe scores | Nota |
|---------|:---:|:---:|:---:|:---:|:---:|------|
| adult_capitalgain | MCAR | 1.00 | ~0 | 0.505 | cond_ent=0.014 | scores idênticos a MCAR |
| **colic_refluxph** | MNAR | **0.15** | 0.53 | 0.488 | cond_ent=0.313 | correto, cond_entropy salva |
| cylinderbands_varnishpct | MAR | 0.98 | 0.001 | 0.641 | tail_asym=1.0 | auc_obs alto → MAR |
| **hepatitis_protime** | MNAR | **0.02** | 0.49 | 0.426 | cond_ent=0.340 | correto, marginalmente |
| kidney_pot | MCAR | 1.00 | ~0 | 0.626 | cond_ent=0.002 | scores idênticos a MCAR |
| kidney_sod | MCAR | 1.00 | ~0 | 0.596 | cond_ent=0.040 | idem |
| **mroz_wages** | MCAR | **0.16** | 0.32 | 0.571 | cond_ent=0.179 | AMBÍGUO, Heckman selection |
| pima_insulin | MCAR | 0.94 | 0.029 | 0.478 | cond_ent=0.087 | invisível estatisticamente |
| pima_skinthickness | MAR | 1.00 | ~0 | 0.619 | cond_ent=0.172 | auc_obs moderado → MAR |

**Padrão:** MNAR é quase indetectável. Os únicos 2 corretos (colic_refluxph, hepatitis_protime) têm `caafe_cond_entropy` > 0.3, o único score C com sinal residual. Os 5 erros com confiança alta têm scores idênticos a MCAR — confirma não-identificabilidade.

---

## 4. Comparação de thresholds Youden's J

| Score | Smoke AUC | Robusto AUC | Δ | Nota |
|-------|:---:|:---:|:---:|------|
| auc_obs (MAR) | **0.916** | **0.895** | -0.021 | Estável, excelente |
| mi_max (MAR) | 0.756 | **0.804** | +0.048 | Melhorou com mais dados |
| little_p (MCAR) | 0.804 | 0.745 | -0.059 | Degradou ligeiramente |
| pklm_p (MCAR) | 0.702 | 0.659 | -0.043 | Idem |
| levene_p (MCAR) | 0.730 | 0.691 | -0.039 | Idem |
| caafe_kurt_excess (MNAR) | 0.540 | **0.629** | +0.089 | Melhorou significativamente |
| caafe_cond_entropy (MNAR) | 0.676 | 0.577 | -0.098 | Piorou |
| caafe_quantile_ratio (MNAR) | 0.500 | 0.500 | 0.000 | **Morto** |
| caafe_tail_asym (MNAR) | 0.500 | 0.500 | 0.000 | **Morto** |

---

## 5. Distribuição real vs sintética: o gap de 37pp

| Score | Sintético MCAR | Real MCAR | Sintético MAR | Real MAR | Sintético MNAR | Real MNAR |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| auc_obs | 0.483 | **0.580** | 0.737 | 0.659 | 0.489 | **0.550** |
| mi_max | 0.006 | **0.014** | 0.084 | **0.036** | 0.006 | **0.011** |
| cond_entropy | 0.28 | ~0.15 | 0.26 | ~0.20 | 0.29 | ~0.13 |

**auc_obs:** Reais MCAR e MNAR têm auc_obs ~20% mais alto que sintéticos → caem acima do threshold de MAR → classificados como MAR.

**mi_max:** Reais MAR têm mi_max 2.3× mais baixo que sintéticos → caem abaixo do threshold → classificados como MCAR.

**Causa raiz:** Sintéticos são gerados com mecanismos puros (missing depende limpa e diretamente de Xi). Reais têm confounders, relações não-lineares parciais, e estrutura de correlação que inflaciona auc_obs para MCAR/MNAR e deflaciona mi_max para MAR.

---

## 6. Classificação dos 29 datasets por tipo de erro

| Tipo de erro | Datasets | Fixável? |
|---|---|---|
| **CORRETO com confiança alta** (6) | mammographic, oceanbuoys×2, sick×2, titanic×2 | — |
| **CORRETO mas ambíguo** (6) | echomonths, hepatitis_albumin, airquality_ozone, colic_refluxph, hepatitis_protime | P6 (prior informativo) |
| **MCAR→MAR falso** (4) | autompg, creditapproval, cylinderbands_blade, hypothyroid | Rótulo duvidoso na literatura |
| **MNAR→MCAR falso** (4) | adult_capital, kidney_pot, kidney_sod, pima_insulin | Não-identificabilidade (fundamental) |
| **MNAR→MAR falso** (2) | cylinderbands_varnish, pima_skin | auc_obs inflado |
| **MAR→MCAR falso** (2) | colic_resprate, hearth_chol | Sinal MAR fraco no real |
| **Ambíguo trimodal** (5) | breastcancer, esavoltage, alkphosphate, kidney_hemo, mroz | Incerteza genuína |
