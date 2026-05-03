# Inventário Completo de Resultados

**Data:** 2026-04-19
**Fonte dos dados:** `Output/v2_improved/` — verificados contra `cv_scores.csv`, `relatorio.txt`, `training_summary.json`, `forensic_summary.csv`

---

## 1. Ablação Principal (Dados Reais)

**Dataset:** 23 datasets reais, 1132 amostras (232 MCAR + 550 MAR + 350 MNAR)
**Validação:** GroupKFold-5 por dataset de origem + LODO
**Metadata:** MEDIUM-scope neutralizada (sem nomes canônicos, sem cutoffs clínicos)

| # | Cenário | n_features | Melhor Modelo | GroupKFold-5 | LODO | CI 95% (bootstrap) |
|---|---------|:----------:|---------------|:------------:|:----:|:-------------------:|
| A | Baseline estatístico | 21 | — | 40.5% | — | — |
| B | + CAAFE | 25 | NaiveBayes | 47.6% | 47.1% | [44.9%, 50.5%] |
| C | + LLM data-driven (sem domain_prior) | 30 | NaiveBayes | 50.5% | 50.4% | [47.7%, 53.4%] |
| D | + domain_prior (pipeline completo) | 31 | NaiveBayes | **56.2%** | **54.3%** | [53.3%, 59.1%] |
| E | domain_prior sozinho | 1 | 6 modelos empatados | **63.1%** | **63.1%** | [60.2%, 65.8%] |

**Chance level:** 33.3% (3 classes balanceadas → 1/3)

### Experimentos de referência

| Experimento | Diretório Output |
|-------------|-----------------|
| A (baseline) | `ctx_baseline/real/apenas_ml/baseline/` |
| B (CAAFE-only) | `step01_caafe_real/real/apenas_ml/baseline/` |
| C, D, E (ablação neutral) | `forensic_neutral_v2/` |

---

## 2. Contribuição Marginal de Cada Componente

Leitura sequencial da tabela de ablação (usando GroupKFold-5):

| Transição | O que foi adicionado | Δ Acurácia | Interpretação |
|-----------|---------------------|:----------:|---------------|
| Chance → A | 21 features estatísticas | +7.2 pp | Sinal real mas modesto a partir de padrões nos dados |
| A → B | 4 CAAFE-MNAR features (Python puro) | **+7.1 pp** | Maior contribuição do regime estatístico |
| B → C | 5 LLM data-driven features | +2.9 pp | Contribuição marginal do raciocínio LLM sobre estatísticas |
| C → D | 1 LLM domain_prior feature | **+5.7 pp** | Raciocínio de domínio genuíno via LLM |
| A → E | domain_prior sozinho vs baseline | **+22.6 pp** | Feature única captura raciocínio de domínio |

---

## 3. Desempenho por Classe

### 3.1. Distribuição do domain_prior (Cenário E)

| Classe Verdadeira | dp=0.0 (→MCAR) | dp=0.5 (→MAR) | dp=1.0 (→MNAR) | LLM Correto |
|-------------------|:---------------:|:--------------:|:---------------:|:-----------:|
| MCAR (n=232) | **27.6%** | 67.2% | 5.2% | 27.6% |
| MAR (n=550) | 1.3% | **96.5%** | 2.2% | 96.5% |
| MNAR (n=350) | 4.9% | 61.1% | **34.0%** | 34.0% |

**Observações:**
- MAR é a classe mais forte: 96.5% corretamente identificados
- MCAR é fraco: 67.2% são classificados como MAR (LLM assume posição moderada na ausência de evidência)
- MNAR é o mais difícil: 61.1% confundidos com MAR

### 3.2. Cenário D (Pipeline completo, NaiveBayes, LODO per-dataset)

Datasets com melhor desempenho:
- `MAR_oceanbuoys_airtemp`: 100% (todos os modelos)
- `MAR_titanic_age`: 94% (NaiveBayes)
- `MNAR_adult_capitalgain`: 92% (NaiveBayes)
- `MNAR_mroz_wages`: 90% (NaiveBayes)

Datasets com pior desempenho:
- `MCAR_hypothyroid_t4u`: 0% (todos os modelos)
- `MCAR_cylinderbands_esavoltage`: 2% (NaiveBayes)
- `MAR_sick_tsh`: 2% (NaiveBayes)
- `MNAR_pima_insulin`: 26% (NaiveBayes)

---

## 4. Classificadores: Ranking Geral

### Cenário D (31 features, GroupKFold-5)

| Rank | Modelo | Accuracy | F1 Macro | CI 95% |
|:----:|--------|:--------:|:--------:|:------:|
| 1 | **NaiveBayes** | **56.2%** | 0.501 | [53.3%, 59.1%] |
| 2 | LogisticRegression | 47.5% | 0.456 | [44.9%, 50.5%] |
| 3 | RandomForest | 40.7% | 0.361 | [38.0%, 43.6%] |
| 4 | GradientBoosting | 39.8% | 0.349 | [36.9%, 42.5%] |
| 5 | MLP | 39.5% | 0.356 | [36.7%, 42.4%] |
| 6 | SVM_RBF | 37.4% | 0.331 | [34.6%, 40.3%] |
| 7 | KNN | 37.1% | 0.351 | [34.4%, 39.8%] |

**Insight:** NaiveBayes domina porque a incerteza calibrada importa mais que capacidade do modelo neste problema com features de alta variância.

### Holdout vs Cross-Validation

| Classificador | Holdout (6 datasets) | GroupKFold-5 CV | Diferença |
|---------------|:--------------------:|:---------------:|:---------:|
| NaiveBayes | 41.4% | **55.5%** | +14.1pp |
| KNN | **48.1%** | 42.9% | -5.2pp |
| LogisticRegression | 47.8% | 47.8% | 0.0pp |

**Recomendação:** Reportar GroupKFold-5 como métrica principal (usa todos os dados, menos sensível ao split).

---

## 5. Dados Sintéticos (1200 amostras)

### Melhor resultado por abordagem

| Abordagem | Melhor Modelo | Accuracy | MCAR F1 | MAR F1 | MNAR F1 |
|-----------|---------------|:--------:|:-------:|:------:|:-------:|
| ML-only (21 features) | MLP | 76.7% | 0.569 | 0.908 | 0.710 |
| ML + LLM | RandomForest | 79.3% | 0.626 | 0.933 | 0.726 |
| Hierárquico V6 | RandomForest | 79.3% | — | — | 0.726 |

### Gap Sintético → Real

| Métrica | Sintético | Real | Gap |
|---------|:---------:|:----:|:---:|
| Melhor accuracy (direto) | 76.7% | 40.5% | -36.2pp |
| Melhor accuracy (pipeline completo) | 79.3% | 56.2% | -23.1pp |
| MNAR recall | 78-81% | 34% | -44pp |
| CAAFE importância SHAP | Rank 16-21 | **Rank 2-4** | Invertido |

---

## 6. Feature Importance (Cenário B, RandomForest)

Top-10 features no cenário CAAFE-only:

| Rank | Feature | Importance | Categoria |
|:----:|---------|:----------:|-----------|
| 1 | `X0_censoring_score` | 11.52% | Discriminativa |
| 2 | `X0_obs_vs_full_ratio` | 9.38% | Estatística resumo |
| 3 | `caafe_tail_asymmetry` | **9.31%** | **CAAFE** |
| 4 | `caafe_kurtosis_excess` | **8.72%** | **CAAFE** |
| 5 | `caafe_cond_entropy_X0_mask` | **7.37%** | **CAAFE** |
| 6 | `X0_obs_skew_diff` | 6.37% | Estatística |
| 7 | `X1_mean_diff` | 5.92% | Discriminativa |
| 8 | `X0_ks_obs_vs_imputed` | 5.92% | Discriminativa |
| 9 | `X0_mean_shift_X1_to_X4` | 5.03% | Discriminativa |
| 10 | `little_proxy_score` | 4.30% | MechDetect |

**4 CAAFE features = 28.3% da importância total** (com apenas 16% das features = 4/25).

---

## 7. Validação de Generalização (LODO ↔ GroupKFold)

| Cenário | GroupKFold-5 | LODO | Δ |
|---------|:------------:|:----:|:-:|
| B (CAAFE-only) | 47.6% | 47.1% | -0.5pp |
| C (sem domain_prior) | 50.5% | 50.4% | -0.1pp |
| D (pipeline completo) | 56.2% | 54.3% | -1.9pp |
| E (domain_prior sozinho) | 63.1% | 63.1% | 0.0pp |

**Conclusão:** Os ganhos do LLM generalizam para datasets não vistos. A diferença LODO-CV < 2pp confirma ausência de memorização.

---

## 8. Reprodutibilidade

Resultados com diferentes seeds (RandomForest, dados reais, holdout):

| Seed | Accuracy |
|:----:|:--------:|
| 42 | 39.3% |
| 123 | 53.7% |
| 456 | 52.0% |

**Alta variância** (14.4pp entre seeds) — reforça a necessidade de reportar CV em vez de holdout único.
