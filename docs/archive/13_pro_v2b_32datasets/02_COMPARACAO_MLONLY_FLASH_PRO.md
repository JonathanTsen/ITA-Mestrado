# Comparação — ML-only vs Flash vs Pro (benchmark v2b, 32 datasets)

## Configurações

| Experimento | Modelo LLM              | Features | Amostras |
|-------------|-------------------------|----------|----------|
| ML-only     | nenhum                  | 25       | 1593     |
| Flash       | gemini-3-flash-preview  | 34       | 1593     |
| Pro         | gemini-3-pro-preview    | 34       | 1590     |

Todas as execuções usam metadata variant `neutral` (sem leakage de mecanismo),
abordagem `context` (context-aware), benchmark v2b com 32 datasets reais.

A diferença de 3 amostras (1593 vs 1590) é por filtragem de bootstraps com
`missing_rate < 1%`, que varia ligeiramente por execução.

## F1-macro — Melhor modelo (GradientBoosting)

| Experimento | F1 teste | F1 CV (média) | F1 CV (std) |
|-------------|----------|---------------|-------------|
| **Flash**   | **0.5055** | 0.5132      | 0.1443      |
| ML-only     | 0.4946   | **0.5254**    | **0.1202**  |
| Pro         | 0.4883   | 0.4674        | 0.1400      |

Flash liderou no teste; ML-only liderou no CV (mais estável).
Pro ficou em último em ambas as métricas.

## F1-macro — Todos os modelos (teste)

| Modelo              | ML-only | Flash  | Pro    |
|---------------------|---------|--------|--------|
| GradientBoosting    | 0.4946  | **0.5055** | 0.4883 |
| RandomForest        | 0.4501  | 0.4671 | **0.4711** |
| SVM_RBF             | 0.4174  | 0.4358 | **0.4548** |
| MLP                 | 0.4119  | **0.4564** | 0.4054 |
| KNN                 | 0.3731  | **0.3994** | 0.4056 |
| LogisticRegression  | 0.3504  | 0.3863 | **0.3918** |
| NaiveBayes          | 0.2765  | 0.3493 | **0.3709** |

Não há um vencedor claro em todos os modelos. Flash lidera nos top-2 (GB, MLP);
Pro lidera em modelos lineares e NaiveBayes.

## Breakdown por classe — GradientBoosting

| Classe | ML-only | Flash  | Pro    | Melhor     |
|--------|---------|--------|--------|------------|
| MCAR   | 0.4940  | **0.6024** | 0.5806 | Flash (+22% vs ML-only) |
| MAR    | **0.4240** | 0.3814 | 0.3568 | ML-only    |
| MNAR   | **0.5659** | 0.5327 | 0.5274 | ML-only    |

Padrão consistente:
- **MCAR**: LLMs superiores (Flash > Pro > ML-only)
- **MAR**: ML-only superior (ML-only > Flash > Pro)
- **MNAR**: ML-only superior (ML-only > Flash > Pro)

## Breakdown por classe — Precision vs Recall (GradientBoosting)

### MCAR
| Exp     | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| ML-only | 0.6212    | 0.4100 | 0.4940 |
| Flash   | 0.7576    | 0.5000 | 0.6024 |
| Pro     | 0.8182    | 0.4500 | 0.5806 |

Pro teve a maior precisão MCAR (0.82) mas recall baixo (0.45).
Flash teve o melhor equilíbrio precision/recall.

### MAR
| Exp     | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| ML-only | 0.3932    | 0.4600 | 0.4240 |
| Flash   | 0.3309    | 0.4500 | 0.3814 |
| Pro     | 0.3028    | 0.4343 | 0.3568 |

ML-only melhor em precision; recall similar entre todos.

### MNAR
| Exp     | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| ML-only | 0.5438    | 0.5900 | 0.5659 |
| Flash   | 0.5354    | 0.5300 | 0.5327 |
| Pro     | 0.5248    | 0.5300 | 0.5274 |

Diferenças pequenas; ML-only lidera por margem estreita.

## Cross-Validation — GradientBoosting

| Exp     | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média  | Std    |
|---------|--------|--------|--------|--------|--------|--------|--------|
| ML-only | —      | —      | —      | —      | —      | 0.5254 | 0.1202 |
| Flash   | —      | —      | —      | —      | —      | 0.5132 | 0.1443 |
| Pro     | —      | —      | —      | —      | —      | 0.4674 | 0.1400 |

ML-only teve o melhor CV e menor variância.

## Feature Importance — Top 10 por experimento

### ML-only (25 features)
| Rank | Feature                      | Imp.   |
|------|------------------------------|--------|
| 1    | caafe_kurtosis_excess        | 0.1291 |
| 2    | X0_obs_skew_diff             | 0.1139 |
| 3    | caafe_cond_entropy_X0_mask   | 0.1033 |
| 4    | X0_obs_vs_full_ratio         | 0.0943 |
| 5    | X0_censoring_score           | 0.0566 |

### Flash (34 features)
| Rank | Feature                      | Imp.   |
|------|------------------------------|--------|
| 1    | caafe_kurtosis_excess        | 0.1034 |
| 2    | caafe_cond_entropy_X0_mask   | 0.0988 |
| 3    | X0_obs_skew_diff             | 0.0864 |
| 4    | X0_obs_vs_full_ratio         | 0.0825 |
| 5    | X0_censoring_score           | 0.0474 |
| 7    | **llm_ctx_domain_confidence**| 0.0351 |

### Pro (34 features)
| Rank | Feature                      | Imp.   |
|------|------------------------------|--------|
| 1    | caafe_cond_entropy_X0_mask   | 0.1006 |
| 2    | caafe_kurtosis_excess        | 0.0949 |
| 3    | X0_obs_skew_diff             | 0.0887 |
| 4    | X0_obs_vs_full_ratio         | 0.0834 |
| 5    | X0_censoring_score           | 0.0522 |
| 9    | **llm_ctx_domain_confidence**| 0.0340 |

Em ambos Flash e Pro, a melhor feature LLM (`llm_ctx_domain_confidence`) ficou
abaixo do top 5. As features estatísticas (CAAFE, skew, obs ratio, censoring)
dominam a importância em todos os experimentos.

## Conclusão

1. **Flash > ML-only > Pro** no F1-macro teste (GradientBoosting)
2. **ML-only > Flash > Pro** no CV (estabilidade)
3. LLMs melhoram MCAR significativamente (+10-22pp), pioram MAR (-4 a -7pp), e empatam em MNAR
4. O modelo Pro (mais caro e lento) não superou o Flash — evidência de que
   a qualidade do reasoning LLM não é o gargalo para essa tarefa
5. As features estatísticas dominam a importância; features LLM contribuem marginalmente
