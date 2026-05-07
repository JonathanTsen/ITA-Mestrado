# Resultados — Gemini Pro v2b (32 datasets)

## Configuração

- **Modelo LLM**: `gemini-3-pro-preview`
- **Abordagem**: `context` (context-aware, metadata variant `neutral`)
- **Benchmark**: v2b — 32 datasets reais (6 MCAR + 13 MAR + 13 MNAR)
- **Execução**: em duas metades (part1: 16 datasets, part2: 16 datasets), depois merge
- **Amostras**: 1590 bootstraps (791 part1 + 799 part2)
- **Features**: 34 (25 estatísticas + 9 LLM context-aware)
- **Experimento**: `step12_pro_v2b_32datasets`

## Features LLM (9)

`llm_ctx_domain_prior`, `llm_ctx_domain_confidence`, `llm_ctx_stats_consistency`,
`llm_ctx_surprise`, `llm_ctx_confidence_delta`, `llm_ctx_counter_strength`,
`llm_ctx_cause_type`, `llm_ctx_n_causes`, `llm_ctx_stats_agreement`

## Resultados no teste (F1-macro)

| Modelo              | F1-macro | MCAR   | MAR    | MNAR   |
|---------------------|----------|--------|--------|--------|
| GradientBoosting    | 0.4883   | 0.5806 | 0.3568 | 0.5274 |
| RandomForest        | 0.4711   | 0.5342 | 0.3348 | 0.5442 |
| SVM_RBF             | 0.4548   | 0.6854 | 0.2387 | 0.4403 |
| MLP                 | 0.4054   | 0.5000 | 0.2879 | 0.4282 |
| KNN                 | 0.4056   | 0.5801 | 0.1961 | 0.4408 |
| LogisticRegression  | 0.3918   | 0.5784 | 0.1429 | 0.4541 |
| NaiveBayes          | 0.3709   | 0.5737 | 0.0148 | 0.5243 |

**Melhor modelo**: GradientBoosting (F1-macro = 0.4883)

## Cross-Validation (Group 5-Fold)

| Modelo              | F1-macro CV | std    |
|---------------------|-------------|--------|
| RandomForest        | 0.4677      | 0.1985 |
| GradientBoosting    | 0.4674      | 0.1400 |
| NaiveBayes          | 0.4295      | 0.1295 |
| MLP                 | 0.3961      | 0.1140 |
| KNN                 | 0.3787      | 0.1116 |
| SVM_RBF             | 0.3758      | 0.1270 |
| LogisticRegression  | 0.3038      | 0.0972 |

## Top 10 Features (importância RandomForest)

| Rank | Feature                      | Importância |
|------|------------------------------|-------------|
| 1    | caafe_cond_entropy_X0_mask   | 0.1006      |
| 2    | caafe_kurtosis_excess        | 0.0949      |
| 3    | X0_obs_skew_diff             | 0.0887      |
| 4    | X0_obs_vs_full_ratio         | 0.0834      |
| 5    | X0_censoring_score           | 0.0522      |
| 6    | caafe_kl_density             | 0.0446      |
| 7    | X1_mean_diff                 | 0.0374      |
| 8    | little_proxy_score           | 0.0356      |
| 9    | X0_mean_shift_X1_to_X4       | 0.0346      |
| 10   | llm_ctx_domain_confidence    | 0.0340      |

A feature LLM mais importante (`llm_ctx_domain_confidence`) ficou em 10o lugar —
abaixo de todas as features CAAFE e estatísticas principais. Padrão similar ao Flash.

## Distribuição dos dados

- MCAR: 298 (18.7%)
- MAR: 642 (40.4%)
- MNAR: 650 (40.9%)

Split teste: MCAR 100 | MAR 99 | MNAR 200

## Observações

- O GradientBoosting apresentou alta precisão em MCAR (0.82) mas recall baixo (0.45),
  indicando que quando prediz MCAR, acerta, mas perde muitos casos
- MAR é a classe mais difícil: F1 de 0.357 mesmo no melhor modelo
- Alta variância no CV do RandomForest (±0.20) indica instabilidade entre folds
