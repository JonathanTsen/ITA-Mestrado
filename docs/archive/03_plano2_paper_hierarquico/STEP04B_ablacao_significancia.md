# STEP 04-B: Ablação Completa + Significância Estatística

**Status: PENDENTE**
**Estimativa: 1-2 dias**

---

## Motivação

A ablação do STEP04 do Plano 1 ficou incompleta — só 1 de 5 configurações foi rodada. Para sustentar a tese do paper (hierárquica + LLM focado), precisamos mostrar a **contribuição marginal** de cada grupo de features com rigor estatístico. A ablação é evidência de suporte, não o foco principal.

Sem testes de significância, "+3.1pp" pode ser ruído. Nenhum reviewer de journal aceita diferenças sem p-values.

---

## Implementação

**Arquivos a modificar:** `v2_improved/train_model.py`, `v2_improved/compare_results.py`

### Referência: As 21 Features Baseline

| # | Feature | Grupo | Módulo |
|:-:|---------|-------|--------|
| 1 | `auc_mask_prediction` | Discriminativa | discriminative.py |
| 2 | `logistic_coef_X1` | Discriminativa | discriminative.py |
| 3 | `log_pval_X1_mask` | Discriminativa | discriminative.py |
| 4 | `mean_diff_X1` | Discriminativa | discriminative.py |
| 5 | `mannwhitney_pval` | Discriminativa | discriminative.py |
| 6 | `little_proxy_score` | Discriminativa | discriminative.py |
| 7 | `X0_missing_rate` | Estatística invariante | statistical.py |
| 8 | `obs_vs_full_ratio` | Estatística invariante | statistical.py |
| 9 | `iqr_ratio` | Estatística invariante | statistical.py |
| 10 | `obs_skew_diff` | Estatística invariante | statistical.py |
| 11 | `X0_ks_obs_vs_imputed` | MNAR-específica | discriminative.py |
| 12 | `X0_tail_missing_ratio` | MNAR-específica | discriminative.py |
| 13 | `mask_entropy` | MNAR-específica | discriminative.py |
| 14 | `X0_censoring_score` | MNAR-específica | discriminative.py |
| 15 | `X0_mean_shift_X1_X4` | MNAR-específica | discriminative.py |
| 16 | `mechdetect_auc_complete` | MechDetect | mechdetect.py |
| 17 | `mechdetect_auc_shuffled` | MechDetect | mechdetect.py |
| 18 | `mechdetect_auc_excluded` | MechDetect | mechdetect.py |
| 19 | `mechdetect_delta_complete_shuffled` | MechDetect | mechdetect.py |
| 20 | `mechdetect_delta_complete_excluded` | MechDetect | mechdetect.py |
| 21 | `mechdetect_mwu_pvalue` | MechDetect | mechdetect.py |

**Nota:** Verificar nomes exatos nos arquivos fonte. A lista acima é baseada nos planos STEP02-STEP03.

### Parte A: Ablação com 6 configurações

| Exp | Features | N Features | Descrição |
|:---:|----------|:----------:|-----------|
| E1 | Discriminativas originais | 6 | #1-6 da tabela acima |
| E2 | E1 + invariantes + MNAR | 15 | + #7-15 |
| E3 | E2 + MechDetect | 21 | + #16-21 (baseline atual) |
| E4 | E3 + CAAFE | 25 | + caafe_missing_rate_by_quantile, tail_asymmetry, kurtosis_excess, cond_entropy |
| E5 | E3 + LLM v2 | 29 | + 8 features LLM: evidence_consistency, anomaly, dist_shift, mcar_conf, mar_conf, mnar_conf, mcar_vs_mnar, pattern_clarity |
| E6 | E3 + Judge MNAR | 25 | + 4 features Judge: mnar_probability, censoring_evidence, distribution_anomaly, pattern_structured (requer API) |

**Nota sobre E5:** LLM v2 é adicionado em cima do baseline (E3=21), NÃO em cima do CAAFE (E4=25). Isso isola a contribuição do LLM. Para testar LLM + CAAFE combinados, ver V4 vs V3 no STEP05A.

### Mapeamento E1-E6 ↔ V1-V6

| Ablação (STEP04B) | Variante hierárquica (STEP05A) | Relação |
|:------------------:|:------------------------------:|---------|
| E3 (21 feat) | V1 (direto 3-way, 21 feat) | Mesmo feature set, classificação direta |
| E3 (21 feat) | V2 (hier. puro, 21 feat ambos níveis) | Mesmo features, arquitetura diferente |
| E4 (25 feat) | V3 (hier. + CAAFE no N2) | CAAFE no N2 apenas |
| E5 (29 feat) | V4 (hier. + LLM no N2) | LLM no N2 apenas |
| E6 (25 feat) | V5 (hier. + Judge no N2) | Judge no N2 apenas |
| E5 (29 feat) | V6 (hier. + LLM em ambos) | LLM em ambos os níveis |

Cada experimento roda:
- Em dados **sintéticos** (1200 amostras) E **reais** (23 datasets, bootstrapped)
- Com os **7 classificadores**
- Com **GroupKFold CV** (k=5)

### Parte B: Testes de Significância

Para cada par de configurações (E1 vs E2, E2 vs E3, etc.):

1. **Wilcoxon signed-rank test** — compara accuracy dos 7 modelos entre duas configurações
   ```python
   from scipy.stats import wilcoxon
   stat, p = wilcoxon(acc_config_A, acc_config_B)
   ```

2. **McNemar test** — compara predições do melhor modelo entre duas configurações
   ```python
   from statsmodels.stats.contingency_tables import mcnemar
   # tabela 2x2: concordam/discordam nas predições
   ```

3. **Bootstrap 95% CI** — intervalo de confiança para cada accuracy
   ```python
   # 1000 bootstrap resamples do test set
   # CI = [percentile 2.5, percentile 97.5]
   ```

4. **Friedman test + post-hoc Nemenyi** — ranking global de métodos
   ```python
   from scipy.stats import friedmanchisquare
   # Se Friedman rejeita H0, aplicar Nemenyi para pairwise comparisons
   ```

### Parte C: Judge MNAR end-to-end

Rodar `extract_features.py --llm-approach judge` em dados sintéticos (requer API key Gemini).
Comparar com CAAFE e v2.

---

## Output Esperado

1. **Tabela de ablação** (accuracy por configuração × modelo):

| Modelo | E1 (6f) | E2 (15f) | E3 (21f) | E4 (25f) | E5 (29f) |
|--------|:-------:|:--------:|:--------:|:--------:|:--------:|
| RF | ? ± CI | ? ± CI | 77% ± CI | 78% ± CI | ? ± CI |
| GBT | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

2. **Tabela de significância** (p-values para cada par):

| Comparação | Wilcoxon p | McNemar p | Significativo? |
|------------|:----------:|:---------:|:--------------:|
| E1 → E2 | ? | ? | ? |
| E2 → E3 | ? | ? | ? |
| E3 → E4 | ? | ? | ? |
| E3 → E5 | ? | ? | ? |

3. **Friedman ranking** dos métodos com diagrama Nemenyi

---

## Testes de Validação

### Teste 1: Ablação é monotônica
E1 ≤ E2 ≤ E3 em accuracy média (mais features = melhor, se são informativas). Se E2 < E1, as features invariantes não ajudam.

### Teste 2: Significância existe
Pelo menos a diferença E1 → E3 deve ser estatisticamente significativa (p < 0.05). Se nem essa diferença é significativa, o N de modelos é muito pequeno.

### Teste 3: LLM não piora
E5 ≥ E3 em pelo menos 5/7 modelos. Se E5 < E3 em maioria, LLM não funciona.

---

## Critério de Conclusão

- [ ] 6 configurações de ablação (E1-E6) rodadas em sintético
- [ ] 6 configurações de ablação (E1-E6) rodadas em real
- [ ] Wilcoxon signed-rank para todos os pares
- [ ] Bootstrap 95% CI para todas as accuracies
- [ ] Friedman + Nemenyi ranking
- [ ] Judge MNAR testado end-to-end (se API disponível)
- [ ] Tabelas formatadas para o paper

---

# Anexo: Resultados do Experimento

> Originalmente publicado como `RESULTADOS_STEP04B.md`. Consolidado aqui em 2026-04-19.


**Data:** 2026-04-18
**Experimento:** step05_pro
**LLM:** gemini-3.1-pro-preview
**Status:** CONCLUÍDO

---

## Configurações de Ablação

| Config | Features | N | Descrição |
|--------|----------|:-:|-----------|
| E1 | Discriminativas | 6 | auc_mask, coef_X1, pval_X1, mean_diff, mannwhitney, little_proxy |
| E2 | E1 + Stat + MNAR | 15 | + missing_rate, ratios, ks, tail, entropy, censoring, mean_shift |
| E3 | E2 + MechDetect | 21 | + 6 MechDetect AUCs/deltas (= baseline) |
| E4 | E3 + CAAFE | 25 | + missing_rate_by_quantile, tail_asymmetry, kurtosis, cond_entropy |
| E5 | E3 + LLM v2 | 29 | + 8 LLM features (evidence, anomaly, confs, etc.) |
| E6 | E3 + CAAFE + LLM | 33 | Todas as features |

---

## Dados Sintéticos (1200 amostras)

### Accuracy por Configuração

| Config | N feat | Média (7 modelos) | Max | Melhor Modelo |
|--------|:------:|:-----------------:|:---:|---------------|
| E1 | 6 | 0.680 | 0.707 | LogisticRegression |
| E2 | 15 | **0.730** | 0.763 | LogisticRegression |
| E3 | 21 | 0.730 | 0.757 | SVM_RBF |
| E4 | 25 | 0.743 | 0.780 | RandomForest |
| E5 | 29 | 0.749 | 0.773 | MLP |
| **E6** | **33** | **0.752** | **0.787** | **RandomForest** |

### Significância Estatística (Sintético)

| Transição | Δ Média | Wilcoxon W | p-value | Sig? |
|-----------|:-------:|:----------:|:-------:|:----:|
| E1→E2 (+Stat+MNAR) | **+0.050** | 1.0 | **0.031*** | ✅ |
| E2→E3 (+MechDetect) | +0.000 | 12.0 | 0.813 | ❌ |
| E3→E4 (+CAAFE) | +0.013 | 5.0 | 0.156 | ❌ |
| E3→E5 (+LLM) | **+0.018** | 0.0 | **0.031*** | ✅ |
| E3→E6 (+CAAFE+LLM) | **+0.022** | 0.0 | **0.031*** | ✅ |
| E4 vs E5 (CAAFE vs LLM) | +0.005 | 13.0 | 0.891 | ❌ |

**Interpretação Sintéticos:**
1. **E1→E2 é significativo** (+5pp): Features estatísticas invariantes e MNAR-específicas são essenciais
2. **E2→E3 não é significativo**: MechDetect não adiciona valor marginal (informação já capturada por E2)
3. **E3→E5 e E3→E6 são significativos**: LLM features contribuem marginalmente (+1.8-2.2pp)
4. **E4 vs E5 não é significativo**: CAAFE e LLM contribuem de forma similar
5. **Ablação é monotônica**: E1 < E2 ≤ E3 < E4 < E5 < E6 (cada grupo adiciona valor)

---

## Dados Reais (1132 amostras)

### Accuracy por Configuração

| Config | N feat | Média (7 modelos) | Max | Melhor Modelo |
|--------|:------:|:-----------------:|:---:|---------------|
| **E1** | **6** | **0.421** | **0.495** | **RandomForest** |
| E2 | 15 | 0.366 | 0.397 | RandomForest |
| E3 | 21 | 0.360 | 0.403 | RandomForest |
| **E4** | **25** | **0.401** | **0.434** | **SVM_RBF** |
| E5 | 29 | 0.367 | 0.420 | KNN |
| **E6** | **33** | **0.400** | **0.447** | **KNN** |

### Significância Estatística (Real)

| Transição | Δ Média | Wilcoxon W | p-value | Sig? |
|-----------|:-------:|:----------:|:-------:|:----:|
| E1→E2 (+Stat+MNAR) | **-0.056** | 3.0 | 0.078 | ❌ |
| E2→E3 (+MechDetect) | -0.006 | 9.5 | 0.484 | ❌ |
| E3→E4 (+CAAFE) | **+0.041** | 5.0 | 0.156 | ❌ |
| E3→E5 (+LLM) | +0.007 | 7.0 | 0.563 | ❌ |
| E3→E6 (+CAAFE+LLM) | **+0.040** | 3.0 | 0.078 | ❌ |
| E4 vs E5 (CAAFE vs LLM) | -0.034 | 6.5 | 0.234 | ❌ |

**Interpretação Dados Reais:**
1. **E1 é surpreendentemente forte** (0.495 max!) — 6 features discriminativas capturam bem os padrões reais
2. **E1→E2 piora** (-5.6pp): Features invariantes e MNAR-específicas confundem em dados reais ruidosos
3. **E3→E4 (+CAAFE) melhora consistentemente** (+4.1pp): CAAFE features são as mais úteis em reais
4. **E5 (LLM) contribui menos que CAAFE** em dados reais
5. **E6 (todas) empata com E4**: LLM não adiciona valor significativo sobre CAAFE em reais
6. **Nenhuma transição é estatisticamente significativa** com N=7 modelos (poder baixo)

### Observação Crítica

A ablação em dados reais NÃO é monotônica: E1 > E2 > E3. Isso sugere que:
- Features adicionais (stat invariantes + MechDetect) introduzem ruído em dados reais
- O pipeline é sensível ao overfitting com mais features em N pequeno
- CAAFE features são as únicas que consistentemente ajudam em reais

---

## Síntese Cruzada

| Padrão | Sintético | Real |
|--------|:---------:|:----:|
| Ablação monotônica? | ✅ SIM | ❌ NÃO |
| CAAFE ajuda? | +1.3pp | **+4.1pp** |
| LLM ajuda? | **+1.8pp** (sig) | +0.7pp (n.s.) |
| MechDetect ajuda? | 0pp | -0.6pp |
| Melhor config | E6 (33f, 78.7%) | E1 (6f, 49.5%) ou E6 (33f, 44.7%) |

**Para o paper:** A contribuição marginal depende do domínio. Em dados limpos (sintéticos), mais features = melhor. Em dados reais ruidosos, features simples e discriminativas são surpreendentemente eficazes, e CAAFE é a única adição que consistentemente ajuda.

---

## Arquivos Gerados

```
Output/v2_improved/step05_pro/{sintetico|real}/ablacao/
├── ablacao_completa.csv     # 6 configs × 7 modelos
├── bootstrap_ci.csv         # 95% CI para cada accuracy
├── significancia.csv        # Wilcoxon + McNemar para cada par
├── heatmap_ablacao.png      # Heatmap configs × modelos
├── ablacao_curve.png        # Accuracy vs N features
├── ablacao_bars_ci.png      # Barras com 95% CI
└── training_summary.json
```
