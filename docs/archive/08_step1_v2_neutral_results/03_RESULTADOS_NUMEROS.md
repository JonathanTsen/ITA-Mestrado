# Resultados Detalhados — Step 1 V2 Neutral

**Data:** 2026-04-25
**Fonte primária:** `Output/v2_improved/step1_v2_neutral/real/ml_com_llm/gemini-3-pro-preview/`

---

## 1. Performance global

### 1.1 Holdout (n=395 bootstraps de 8 datasets-test)

| Modelo | Acurácia | Precisão (macro) | Recall (macro) | F1 (macro) |
|--------|:--------:|:----------------:|:--------------:|:----------:|
| **NaiveBayes** | **55.19%** | **0.61** | **0.54** | **0.55** |
| LogisticRegression | 54.94% | 0.55 | 0.55 | 0.55 |
| SVM_RBF | 43.80% | 0.45 | 0.44 | 0.44 |
| MLP | 43.04% | 0.42 | 0.43 | 0.41 |
| RandomForest | 41.77% | 0.43 | 0.43 | 0.42 |
| GradientBoosting | 41.27% | 0.44 | 0.43 | 0.41 |
| KNN | 39.75% | 0.40 | 0.41 | 0.40 |

### 1.2 Cross-Validation (Group 5-Fold sobre 1.421 amostras)

| Modelo | CV avg | CV std | Mín | Máx |
|--------|:------:|:------:|:---:|:---:|
| **NaiveBayes** | **49.33%** | ±14.2 | 37.2% | 57.7% |
| LogisticRegression | 41.54% | ±23.5 | 22.8% | 53.9% |
| RandomForest | 38.97% | ±26.6 | 23.6% | 62.0% |
| GradientBoosting | 36.32% | ±18.7 | 29.3% | 54.6% |
| KNN | 35.02% | ±18.5 | 23.9% | 49.8% |
| MLP | 33.26% | ±24.3 | 24.8% | 57.3% |
| SVM_RBF | 32.85% | ±27.4 | 18.3% | 59.0% |

### 1.3 CV scores por fold (todos os modelos)

| Fold | RF | GBT | LogReg | SVM_RBF | KNN | MLP | NB |
|:----:|:--:|:---:|:------:|:-------:|:---:|:---:|:--:|
| 0 | 28.3% | 29.3% | 48.7% | 29.7% | 40.3% | 29.3% | **57.7%** |
| 1 | **62.0%** | 54.6% | 53.9% | **59.0%** | 49.8% | 57.3% | 54.9% |
| 2 | 40.1% | 33.2% | 32.9% | 18.3% | 23.9% | 26.0% | 47.1% |
| 3 | 40.8% | 34.5% | 49.5% | 29.3% | 33.4% | 28.9% | 49.8% |
| 4 | 23.6% | 30.0% | 22.8% | 28.0% | 27.6% | 24.8% | 37.2% |
| **avg** | 38.97% | 36.32% | 41.54% | 32.85% | 35.02% | 33.26% | **49.33%** |

**Observação crítica:** o fold 1 produz CV scores anormalmente altos (RF=62%, SVM=59%, NB=55%, MLP=57%), enquanto o fold 4 deprime todos os modelos para ~22-37%. Isso indica que **a heterogeneidade entre datasets é maior que entre folds** — confirmando que a variância CV ±14-27pp não é ruído estocástico mas sinal genuíno de domain shift entre grupos.

## 2. Análise por classe (NaiveBayes — best model)

### 2.1 Holdout

| Classe | Precisão | Recall | F1 | Suporte |
|--------|:--------:|:------:|:--:|:-------:|
| MCAR (0) | 0.79 | 0.44 | 0.57 | 95 |
| MAR (1) | 0.52 | 0.45 | 0.48 | 150 |
| MNAR (2) | 0.51 | **0.73** | **0.60** | 150 |

**Interpretação:**
- MCAR tem **alta precisão (79%)** mas baixo recall (44%): quando o modelo aposta MCAR, acerta na maioria das vezes, mas perde muitos casos genuínos.
- MNAR tem **alto recall (73%)** com precisão modesta (51%): o modelo pega quase todos os MNAR mas frequentemente confunde MAR como MNAR.
- MAR é a classe mais "confusa", com precisão e recall ambos abaixo de 53%.

### 2.2 Matriz de confusão (NaiveBayes holdout)

```
                 PREDITO
                 MCAR  MAR  MNAR
TRUE   MCAR    [  42    33    20 ]   95
       MAR     [   0    67    83 ]  150
       MNAR    [  11    30   109 ]  150
                 53   130   212    395
```

**Diagonais principais:**
- 42/95 MCAR corretos (44%)
- 67/150 MAR corretos (45%)
- 109/150 MNAR corretos (73%)

**Maior fonte de erro:** MAR → MNAR (83 casos = 55% dos MAR são preditos como MNAR). Isso ocorre porque os 3 datasets MAR no holdout (`airquality_ozone`, `sick_tsh`, `titanic_age`) têm features estatísticas que se sobrepõem com o padrão MNAR de truncamento — `sick_tsh` em particular tem 36% recall isolado e LLM majority MNAR.

### 2.3 Matrizes de confusão dos demais modelos

**LogisticRegression (Acc 54.94%):**
```
                 MCAR  MAR  MNAR
TRUE   MCAR    [  54    34     7 ]   95
       MAR     [  10    86    54 ]  150
       MNAR    [  32    41    77 ]  150
```
LogReg distribui melhor: tem 86/150 MAR e 77/150 MNAR. É um perfil mais equilibrado mas com acurácia ligeiramente menor que NB.

**RandomForest (Acc 41.77%):**
```
                 MCAR  MAR  MNAR
TRUE   MCAR    [  49    40     6 ]   95
       MAR     [  39    61    50 ]  150
       MNAR    [  73    22    55 ]  150
```
RF tem MNAR confundido com MCAR em 73 casos (49% dos MNAR), padrão diferente de NB. Indica que features estatísticas isoladas (sem regularização Bayesiana) não distinguem bem MNAR de MCAR.

## 3. Análise de `llm_ctx_domain_prior` solo

Cálculo direto da feature LLM (sem ML), discretizando o continuum em [-0.01, 0.25] → MCAR, (0.25, 0.75] → MAR, (0.75, 1.01] → MNAR:

### 3.1 Performance global (n=1.421)

- **Accuracy: 43.7%** (vs chance 33.3%; vs `forensic_neutral_v2` 63.1% → **regressão de −19.4pp**)
- Recall MCAR: **24.9%**
- Recall MAR: 67.6%
- Recall MNAR: **32.0%**

### 3.2 Matriz de confusão (domain_prior solo, n=1.421)

```
                 PREDITO LLM
                 MCAR   MAR    MNAR    Total
TRUE   MCAR    [  105   234     82 ]    421
       MAR     [   77   372    101 ]    550
       MNAR    [   58   248    144 ]    450
       Total      240   854    327     1421
```

**Tendências:**
- 854/1.421 (60%) das predições do LLM são MAR → indica MAR-bias residual significativo
- LLM rotula MCAR como MAR em 234/421 (55%) — recall MCAR cai para 25%
- LLM rotula MNAR como MAR em 248/450 (55%) — recall MNAR cai para 32%

### 3.3 Recall por parent-dataset (domain_prior solo)

**MCAR (9 datasets, ordenados por recall ascendente):**

| Dataset | Recall | LLM majority | Análise |
|---------|:------:|:------------:|---------|
| `MCAR_hypothyroid_t4u` | **0%** | MAR | Falha total |
| `MCAR_echomonths_epss` | 4% | MAR | Falha total |
| `MCAR_cylinderbands_esavoltage` | 10% | MNAR | Bias censura |
| `MCAR_hepatitis_alkphosphate` | 20% | MAR | MAR-bias |
| `MCAR_cylinderbands_bladepressure` | 28% | MAR | MAR-bias |
| `MCAR_hepatitis_albumin` | 32% | MAR | MAR-bias |
| `MCAR_breastcancer_barenuclei` | 36% | MNAR | Bias censura |
| `MCAR_autompg_horsepower` | 51% | MCAR | OK |
| `MCAR_creditapproval_a14` | 59% | MCAR | OK |

**MAR (11 datasets):**

| Dataset | Recall | LLM majority |
|---------|:------:|:------------:|
| `MAR_sick_tsh` | 36% | MNAR |
| `MAR_mammographic_density` | 38% | MCAR |
| `MAR_hearth_chol` | 52% | MAR |
| `MAR_airquality_ozone` | 54% | MAR |
| `MAR_kidney_hemo` | 56% | MAR |
| `MAR_colic_resprate` | 58% | MAR |
| `MAR_oceanbuoys_humidity` | 76% | MAR |
| `MAR_titanic_age` | 88% | MAR |
| `MAR_oceanbuoys_airtemp` | 88% | MAR |
| `MAR_titanic_age_v2` | 98% | MAR |
| `MAR_sick_t3` | **100%** | MAR |

**MNAR (9 datasets):**

| Dataset | Recall | LLM majority |
|---------|:------:|:------------:|
| `MNAR_pima_insulin` | **4%** | MAR | ⚠️ caso canônico |
| `MNAR_kidney_pot` | 4% | MAR |
| `MNAR_kidney_sod` | 8% | MAR |
| `MNAR_hepatitis_protime` | 12% | MAR |
| `MNAR_pima_skinthickness` | 18% | MAR |
| `MNAR_cylinderbands_varnishpct` | 28% | MAR |
| `MNAR_colic_refluxph` | 42% | MAR |
| `MNAR_adult_capitalgain` | 80% | MNAR |
| `MNAR_mroz_wages` | **92%** | MNAR |

### 3.4 Distribuição agregada

| Recall range | MCAR | MAR | MNAR | Total |
|:------------:|:----:|:---:|:----:|:-----:|
| 0–20% | 4 | 0 | 5 | 9 |
| 20–50% | 3 | 2 | 2 | 7 |
| 50–80% | 2 | 5 | 1 | 8 |
| 80–100% | 0 | 4 | 1 | 5 |

**Conclusão da seção 3:**
- 14 dos 29 datasets (48%) têm recall ≤ 50% → o LLM ainda erra pesadamente em quase metade do benchmark
- MAR é a classe mais bem reconhecida (5 datasets ≥ 80%, máximo 100%)
- MNAR tem distribuição bipolar: ou o LLM acerta canônicos (`mroz_wages`, `adult_capitalgain`) ou erra completamente (`pima_insulin`, `kidney_pot`)

## 4. Estatísticas das features LLM

| Feature | Mean | Std | Range observado |
|---------|:----:|:---:|:----------------:|
| `llm_ctx_domain_prior` | 0.51 | 0.34 | {0, 0.5, 1} discretizado pelo LLM |
| `llm_ctx_domain_confidence` | 0.62 | 0.18 | [0.1, 0.95] |
| `llm_ctx_stats_consistency` | 0.55 | 0.22 | [0, 1] |
| `llm_ctx_surprise` | 0.31 | 0.24 | [0, 0.85] |
| `llm_ctx_confidence_delta` | 0.04 | 0.21 | [-0.5, 0.5] |
| `llm_ctx_counter_strength` | 0.45 | 0.19 | [0, 1] |
| `llm_ctx_cause_type` | 0.51 | 0.32 | {0, 0.5, 1} (Tipo A/B/C) |
| `llm_ctx_n_causes` | 0.65 | 0.13 | [0, 1] (n_causas/5) |
| `llm_ctx_stats_agreement` | 0.65 | 0.30 | {0, 0.5, 1} |

**Variância saudável:** todas as features têm std > 0.1, indicando que o LLM está discriminando, não retornando default.

## 5. Sumário comparativo

| Métrica | Step 1 V2 (atual) | step10_flash | forensic_neutral_v2 | Δ vs ref |
|---------|:-----------------:|:------------:|:-------------------:|:--------:|
| Datasets | 29 | 29 | 23 | +6 |
| Bootstraps | 1.421 | 1.421 | 1.132 | +289 |
| Best Holdout | **55.19% (NB)** | 51.14% (LR) | — | — |
| Best CV | **49.33% (NB)** | 47.44% (NB) | 56.2% (NB) | **−7pp** |
| domain_prior solo | 43.7% | — | 63.1% | **−19.4pp** |
| Razão LLM/Total (RF) | 12.6% | 10.4% | — | +2.2pp |
| Recall MNAR (CV avg, NB) | ~50% | ~40% | — | +10pp |
| Tempo de extração | 1h33min | ~30min | ~25min | +63min |
| Custo API | ~$30-36 | ~$3-5 | ~$2-3 | +10x |

**Trade-off explícito:** Pro entrega +1.9pp CV sobre Flash mas custa **10x mais** ($30+ vs $3-5). Para experimentação iterativa (Step 2, Step 3), Flash continua sendo mais viável; Pro reservado para validações finais.

## 6. Arquivos gerados (`Output/v2_improved/step1_v2_neutral/.../`)

| Arquivo | Conteúdo |
|---------|----------|
| `X_features.csv` | 1.421 × 34 (features finais) |
| `y_labels.csv` | 1.421 × 1 (rótulos 0=MCAR, 1=MAR, 2=MNAR) |
| `groups.csv` | 1.421 × 1 (parent-dataset por amostra) |
| `predictions.csv` | 2.765 linhas (predições de cada modelo no holdout) |
| `metrics_per_class.csv` | 21 linhas (precision/recall/F1 por classe × 7 modelos) |
| `feature_importance.csv` | 34 linhas (RF feature importance) |
| `cv_scores.csv` | 35 linhas (CV scores por fold × modelo) |
| `confusion_matrices.json` | 7 modelos |
| `hyperparameters.json` | 7 modelos |
| `feature_selection_log.json` | log do feature selection (todas mantidas: 34/34) |
| `training_summary.json` | metadata completo do run |
| `relatorio.txt` | relatório textual completo |
| `resultados.png` | gráfico de barras de acurácia |
| `precisao_por_classe.png` | F1 por classe por modelo |
