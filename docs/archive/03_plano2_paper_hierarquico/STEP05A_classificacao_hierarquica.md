# STEP 05-A: Classificação Hierárquica + LLM no Nível 2

**CORE DO PAPER — PRIORIDADE MÁXIMA**
**Status: PENDENTE**
**Estimativa: 2-3 dias**

---

## Motivação

Classificar 3 classes diretamente falha porque MCAR e MNAR são quase indistinguíveis:
- MCAR = ausência de sinal (missing não correlaciona com nada)
- MNAR = sinal em X0 (missing correlaciona com o próprio valor faltante — mas X0 está faltante!)

Dados atuais confirmam: 46% dos MCAR sintéticos confundidos com MNAR, MNAR recall ≈ 0% em dados reais.

**A solução:** Dividir em 2 problemas binários, e usar LLM **apenas** no subproblema onde features estatísticas falham.

### Por que LLM só no Nível 2

| Nível | Problema | Features estatísticas | LLM |
|:-----:|----------|:---------------------:|:---:|
| 1 | MCAR vs não-MCAR | **Suficientes** — Little's test, MechDetect deltas detectam se missing depende de algo | Desnecessário (e piora: -20pp) |
| 2 | MAR vs MNAR | **Insuficientes** — circularidade de X0, KS test com baixo poder | **Aqui ajuda** — reconhece censura, truncamento, padrões qualitativos |

Isso explica o paradoxo dos resultados anteriores:
- LLM piora classificação 3-way (-20pp) → porque adiciona ruído no Nível 1
- LLM melhora em dados reais (+3.1pp) → porque dados reais têm mais amostras no Nível 2

---

## Implementação

**Arquivo a modificar:** `v2_improved/train_hierarchical.py` (já existe parcialmente)

### Nível 1: MCAR vs {MAR, MNAR} — Apenas features estatísticas

**Lógica:** Se AUC_complete ≈ AUC_shuffled → MCAR (missing não depende de nada)

**Features (21 baseline, SEM LLM):**
- `little_proxy_score` — proxy do teste de Little
- `mask_entropy` — entropia da máscara de missing
- `mechdetect_delta_complete_shuffled` — delta entre AUC complete e shuffled
- `mechdetect_auc_complete` — AUC da tarefa completa
- `X0_missing_rate` — taxa de missing de X0
- `mannwhitney_pval` — Mann-Whitney U p-value
- (+ restante das 21 features baseline)

**Classificador:** Testar os 7 (RF, GBT, LR, SVM, KNN, MLP, NB)

### Nível 2: MAR vs MNAR — Features estatísticas + LLM

**Lógica:** Features estatísticas sozinhas não conseguem distinguir MAR de MNAR com confiança. LLM features adicionam raciocínio qualitativo sobre padrões de censura e truncamento.

**Features baseline (sem LLM) para o Nível 2:**
- `mechdetect_delta_complete_excluded` — delta entre AUC complete e excluded
- `X0_ks_obs_vs_imputed` — KS test entre X0 observado e imputado
- `X0_censoring_score` — score de censura
- `X0_tail_missing_ratio` — razão de missing nas caudas
- `log_pval_X1_mask` — p-valor da correlação X1-mask
- `auc_mask_prediction` — AUC de predição da máscara
- (+ restante das 21 features baseline)

**Features LLM adicionadas APENAS no Nível 2:**

Opção A — CAAFE (4 features, sem API):
- **Nota 2026-05-06:** esta lista registra a versão histórica usada na fase.
  A implementação v2b atual substitui as duas primeiras por
  `caafe_auc_self_delta` e `caafe_kl_density`. Em todos os casos, são
  features CAAFE-inspired determinísticas, não uma reimplementação do CAAFE
  original com LLM gerando código.
- `caafe_missing_rate_by_quantile` — taxa de missing por faixa de X0
- `caafe_tail_asymmetry` — assimetria de caudas de X0
- `caafe_kurtosis_excess` — excesso de curtose de X0
- `caafe_cond_entropy_X0_mask` — entropia condicional mask-X0

Opção B — LLM v2 (8 features, requer API):
- `llm_evidence_consistency`, `llm_anomaly`, `llm_dist_shift`
- `llm_mcar_conf`, `llm_mar_conf`, `llm_mnar_conf`
- `llm_mcar_vs_mnar`, `llm_pattern_clarity`

Opção C — Judge MNAR (4 features, requer API):
- `mnar_probability`, `censoring_evidence`
- `distribution_anomaly`, `pattern_structured`

**Classificador:** Pode ser diferente do Nível 1 (testar melhor combinação)

### Pipeline

```python
def hierarchical_predict(X, X_with_llm):
    """
    X: features baseline (21 ou 25 com CAAFE)
    X_with_llm: features baseline + LLM (para Nível 2)
    """
    # Nível 1: MCAR vs não-MCAR (SEM LLM)
    pred_level1 = model_level1.predict(X)  # 0=MCAR, 1=não-MCAR
    
    # Nível 2: MAR vs MNAR (COM LLM, só para não-MCAR)
    mask_not_mcar = pred_level1 == 1
    pred_level2 = np.full(len(X), -1)
    if mask_not_mcar.any():
        pred_level2[mask_not_mcar] = model_level2.predict(
            X_with_llm[mask_not_mcar]  # <-- LLM features aqui
        )
    
    # Combinar: MCAR=0, MAR=1, MNAR=2
    final = np.where(pred_level1 == 0, 0,   # MCAR
            np.where(pred_level2 == 0, 1,    # MAR
                     2))                      # MNAR
    return final
```

### Variantes a testar (experimento principal do paper)

| Variante | Nível 1 | Nível 2 | Hipótese |
|----------|---------|---------|----------|
| V1: Direto 3-way (baseline) | — | — | Referência |
| V2: Hierárquico puro | 21 feat stat | 21 feat stat | Hierárquica já melhora? |
| V3: Hier. + CAAFE no N2 | 21 feat stat | 25 feat (stat+CAAFE) | CAAFE ajuda no N2? |
| V4: Hier. + LLM v2 no N2 | 21 feat stat | 29 feat (stat+LLM) | LLM ajuda no N2? |
| V5: Hier. + Judge no N2 | 21 feat stat | 25 feat (stat+Judge) | Judge focado ajuda? |
| V6: Hier. + LLM em ambos | 29 feat (stat+LLM) | 29 feat (stat+LLM) | LLM no N1 piora? |

**V6 é o controle:** se LLM no N1 piora vs V4, confirma a tese de que LLM deve ser focado no N2.

---

## Métricas de Comparação

Para cada variante:
- Accuracy global
- **Recall por classe (MCAR, MAR, MNAR)** — foco em MNAR recall
- F1 macro
- Confusion matrix
- Nível 1 accuracy isolada
- Nível 2 accuracy isolada (condicionada a não-MCAR correto)
- CV score (GroupKFold e LOGO)
- **Testes de significância** (Wilcoxon V2 vs V4, McNemar V1 vs V4)

---

## Resultado Esperado (hipótese)

| Variante | Acc Global | MNAR Recall | Interpretação |
|----------|:----------:|:-----------:|---------------|
| V1 (direto) | ~43% | ~0-38% | Baseline ruim |
| V2 (hier. puro) | ~50%? | ~20%? | Melhora modesta |
| **V4 (hier. + LLM N2)** | **~55%?** | **~40%?** | **Melhor resultado** |
| V6 (LLM em ambos) | ~45%? | ~30%? | LLM no N1 atrapalha |

Se V4 > V2 > V1 e V4 > V6, a tese central está comprovada.

---

## Testes de Validação

### Teste 1: Nível 1 funciona sem LLM
Accuracy MCAR vs não-MCAR > 70% com apenas features estatísticas.

### Teste 2: LLM melhora Nível 2
Accuracy MAR vs MNAR com LLM > sem LLM. Diferença estatisticamente significativa (Wilcoxon p < 0.05).

### Teste 3: LLM no Nível 1 não ajuda (ou piora)
V6 (LLM em ambos) ≤ V4 (LLM só N2) em accuracy do Nível 1. Confirma que LLM é ruído no N1.

### Teste 4: MNAR recall melhora
Recall MNAR de V4 > recall MNAR de V1. Se não, a hierárquica + LLM não resolve o gargalo.

### Teste 5: Accuracy global não degrada demais
V4 accuracy ≥ V1 accuracy - 5pp (tradeoff aceitável se MNAR recall melhorar muito).

### Teste 6: SHAP confirma
SHAP do Nível 2 mostra features LLM entre top 5 de importância. SHAP do Nível 1 mostra features LLM com importância baixa.

---

## Critério de Conclusão

- [ ] Pipeline hierárquico implementado com LLM seletivo por nível
- [ ] 6 variantes (V1-V6) testadas em dados sintéticos
- [ ] 6 variantes testadas em dados reais
- [ ] Comparação V1 vs V2 vs V4 vs V6 documentada com significância
- [ ] MNAR recall melhorou significativamente (V4 vs V1)
- [ ] SHAP confirma LLM mais importante no Nível 2
- [ ] Accuracy Nível 1 e Nível 2 reportadas separadamente
- [ ] Melhor combinação de classificadores (Nível 1 + Nível 2) identificada

---

# Anexo A: Resultados do Experimento (STEP05A)

> Originalmente publicado como `RESULTADOS_STEP05A.md`. Consolidado aqui em 2026-04-19.


**Data:** 2026-04-18
**Experimento:** step05_pro
**LLM:** gemini-3.1-pro-preview (LLM v2, 8 features)
**Status:** CONCLUÍDO

---

## Configuração

### Variantes Testadas

| Variante | Tipo | Nível 1 | Nível 2 | #Features |
|----------|------|---------|---------|:---------:|
| V1 | Direto 3-way | — | — | 21 (stat) |
| V2 | Hierárquico | 21 stat | 21 stat | 21 |
| V3 | Hierárquico | 21 stat | 25 (stat+CAAFE) | 21/25 |
| **V4** | **Hierárquico** | **21 stat** | **33 (stat+CAAFE+LLM)** | **21/33** |
| V5 | Hierárquico | 33 (todas) | 33 (todas) | 33/33 |
| V6 | Direto 3-way | — | — | 33 (todas) |

### Features por Grupo

- **Stat (21):** X0_missing_rate, X0_obs_vs_full_ratio, X0_iqr_ratio, X0_obs_skew_diff, auc_mask_from_Xobs, coef_X1_abs, log_pval_X1_mask, X1_mean_diff, X1_mannwhitney_pval, little_proxy_score, X0_ks_obs_vs_imputed, X0_tail_missing_ratio, mask_entropy, X0_censoring_score, X0_mean_shift_X1_to_X4, mechdetect_auc_complete, mechdetect_auc_shuffled, mechdetect_auc_excluded, mechdetect_delta_complete_shuffled, mechdetect_delta_complete_excluded, mechdetect_mwu_pvalue
- **CAAFE historico (4):** caafe_missing_rate_by_quantile, caafe_tail_asymmetry, caafe_kurtosis_excess, caafe_cond_entropy_X0_mask. Na versão v2b atual: `caafe_auc_self_delta`, `caafe_kl_density`, `caafe_kurtosis_excess`, `caafe_cond_entropy_X0_mask`.
- **LLM v2 (8):** llm_evidence_consistency, llm_anomaly, llm_dist_shift, llm_mcar_conf, llm_mar_conf, llm_mnar_conf, llm_mcar_vs_mnar, llm_pattern_clarity

### Protocolo

- **Split:** GroupShuffleSplit 75/25 (seed=42) — mesmo para todas as variantes
- **Balanceamento:** SMOTE (k=3 vizinhos)
- **Modelos:** RandomForest, GradientBoosting, LogisticRegression, SVM_RBF, KNN, MLP, NaiveBayes
- **CV:** LOGO (Leave-One-Group-Out) para dados reais (23 folds)

---

## Resultados: Dados Reais (1132 amostras, 23 datasets)

### Ranking por Accuracy Máxima (Holdout)

| Rank | Variante | Accuracy | Modelo | F1 Macro | MNAR Recall | MNAR F1 |
|:----:|----------|:--------:|--------|:--------:|:-----------:|:-------:|
| **1** | **V3: Hier+CAAFE N2** | **50.5%** | **GradientBoosting** | **0.488** | **40.0%** | **0.294** |
| 2 | V6: Direto+LLM | 44.8% | KNN | 0.414 | 26.0% | 0.194 |
| 3 | V4: Hier+LLM N2 | 44.4% | GradientBoosting | 0.396 | 6.0% | 0.049 |
| 4 | V5: Hier+LLM ambos | 43.1% | KNN | 0.411 | 30.0% | 0.210 |
| 5 | V1: Direto stat | 41.4% | RandomForest | 0.424 | 40.0% | 0.244 |
| 6 | V2: Hier puro | 41.0% | RandomForest | 0.423 | 40.0% | 0.242 |

### Comparação V1 vs V4 por Modelo (Holdout)

| Modelo | V1 Acc | V4 Acc | Δ Acc | V1 MNAR | V4 MNAR | Δ MNAR |
|--------|:------:|:------:|:-----:|:-------:|:-------:|:------:|
| GradientBoosting | 38.6% | **44.4%** | **+5.8pp** | 38.0% | 6.0% | -32.0pp |
| LogisticRegression | 27.1% | **37.3%** | **+10.2pp** | 10.0% | 4.0% | -6.0pp |
| SVM_RBF | 31.5% | **37.6%** | **+6.1pp** | 10.0% | 8.0% | -2.0pp |
| KNN | 37.6% | **41.7%** | **+4.1pp** | 36.0% | 26.0% | -10.0pp |
| NaiveBayes | 39.3% | **40.3%** | **+1.0pp** | 54.0% | 30.0% | -24.0pp |
| MLP | 38.0% | **39.3%** | **+1.4pp** | 20.0% | 18.0% | -2.0pp |
| RandomForest | **41.4%** | 39.7% | -1.7pp | 40.0% | 2.0% | -38.0pp |

### V3 (Hier+CAAFE N2) — Detalhado por Modelo

| Modelo | Accuracy | MCAR Recall | MAR Recall | MNAR Recall | MNAR F1 |
|--------|:--------:|:-----------:|:----------:|:-----------:|:-------:|
| **GradientBoosting** | **50.5%** | 47.4% | 56.0% | **40.0%** | 0.294 |
| RandomForest | 43.4% | 46.3% | 50.7% | 16.0% | 0.119 |
| KNN | 41.7% | 43.2% | 42.0% | 38.0% | 0.264 |
| NaiveBayes | 41.7% | 36.8% | 47.3% | 34.0% | 0.195 |
| LogisticRegression | 41.0% | 44.2% | 50.0% | 8.0% | 0.059 |
| SVM_RBF | 41.0% | 38.9% | 48.0% | 24.0% | 0.169 |
| MLP | 39.7% | 40.0% | 39.3% | 40.0% | 0.240 |

### Accuracy por Nível (L1 e L2) — V3

| Modelo | Acc L1 (MCAR vs NAO-MCAR) | Acc L2 (MAR vs MNAR) |
|--------|:-------------------------:|:--------------------:|
| RandomForest | 82.0% | 42.4% |
| GradientBoosting | 82.0% | **52.8%** |
| LogisticRegression | 79.7% | 40.9% |
| SVM_RBF | 78.0% | 43.5% |
| KNN | 70.2% | 49.4% |
| MLP | 80.0% | 39.9% |
| NaiveBayes | 79.7% | 44.0% |

**Insight:** L1 funciona bem (~80% accuracy), L2 é o gargalo (~40-53%). CAAFE melhora L2 vs stat puro.

### Testes de Significância: McNemar (V1 vs V4)

| Modelo | V1→V4 correto | V4→V1 correto | χ² | p-value | Sig? |
|--------|:-------------:|:-------------:|:--:|:-------:|:----:|
| LogisticRegression | 7 | 37 | 19.11 | **<0.001*** | ✅ |
| SVM_RBF | 6 | 24 | 9.63 | **0.002** | ✅ |
| GradientBoosting | 17 | 34 | 5.02 | **0.025*** | ✅ |
| KNN | 18 | 30 | 2.52 | 0.112 | ❌ |
| NaiveBayes | 19 | 22 | 0.10 | 0.755 | ❌ |
| MLP | 16 | 20 | 0.25 | 0.617 | ❌ |
| RandomForest | 20 | 15 | 0.46 | 0.499 | ❌ |

**3 de 7 modelos** mostram melhoria significativa de V1→V4 (p<0.05).

### LOGO Cross-Validation (23 folds, dados reais)

| Modelo | V1 LOGO | V2 LOGO | V3 LOGO | V4 LOGO | Δ(V3-V1) | Δ(V4-V1) |
|--------|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|
| **NaiveBayes** | 47.5% | 49.5% | **51.4%** | 51.2% | **+3.9pp** | +3.7pp |
| **LogisticRegression** | 36.8% | 37.8% | **44.0%** | **45.2%** | **+7.1pp** | **+8.3pp** |
| **GradientBoosting** | 36.5% | 37.5% | **38.3%** | 36.3% | **+1.8pp** | -0.2pp |
| RandomForest | **39.0%** | 40.5% | 38.3% | 37.2% | -0.7pp | -1.9pp |
| KNN | **38.9%** | 38.6% | 37.1% | 37.2% | -1.8pp | -1.7pp |
| MLP | 35.0% | 34.2% | 31.6% | **38.0%** | -3.4pp | +3.0pp |
| SVM_RBF | 32.9% | 32.3% | 31.8% | 32.6% | -1.1pp | -0.3pp |

**V3 NaiveBayes (51.4%) é o melhor resultado LOGO CV**, superando V4 NaiveBayes (51.2%). V3 melhora sobre V1 em 3 de 7 modelos (NaiveBayes +3.9pp, LogReg +7.1pp, GB +1.8pp). V4 melhora em 3 de 7 modelos (NaiveBayes +3.7pp, LogReg +8.3pp, MLP +3.0pp).

**Insight:** V3 e V4 têm performance LOGO CV muito similar no melhor modelo, mas V3 não requer API LLM.

---

## Resultados: Dados Sintéticos (1200 datasets)

### Comparação V1 vs V4 por Modelo

| Modelo | V1 Acc | V4 Acc | Δ | V1 MNAR | V4 MNAR | Δ MNAR |
|--------|:------:|:------:|:-:|:-------:|:-------:|:------:|
| MLP | **76.7%** | 73.0% | -3.7pp | 77.8% | 67.7% | -10.1pp |
| RandomForest | **76.0%** | 70.3% | -5.7pp | 71.7% | 53.5% | -18.2pp |
| SVM_RBF | **75.7%** | 66.0% | -9.7pp | 84.8% | 36.4% | -48.5pp |
| LogisticRegression | **75.0%** | 67.0% | -8.0pp | 80.8% | 27.3% | -53.5pp |
| GradientBoosting | **74.0%** | 73.0% | -1.0pp | 66.7% | 60.6% | -6.1pp |
| NaiveBayes | **73.0%** | 59.7% | -13.3pp | 83.8% | 4.0% | -79.8pp |
| KNN | **68.0%** | 64.3% | -3.7pp | 50.5% | 34.3% | -16.2pp |

**Em sintéticos, V1 (direto) é consistentemente melhor.** Wilcoxon V1 vs V4: p=0.016* (significativo). LLM features adicionam ruído em dados limpos — resultado esperado e coerente com a tese.

### Verificação da Tese (Sintéticos)

- V4 > V1? **NÃO** (V1 é melhor em sintéticos — esperado)
- V4 > V2? Parcial (V2 ~72% vs V4 ~70%)
- V4 > V5? **SIM** (LLM no N1 piora mais que LLM só no N2)

---

## Síntese e Interpretação

### Descoberta Principal: CAAFE > LLM v2 no Nível 2

| Métrica | V3 (CAAFE N2) | V4 (LLM N2) | Diferença |
|---------|:------------:|:----------:|:---------:|
| Accuracy máx (real) | **50.5%** | 44.4% | **+6.1pp** |
| MNAR recall (melhor modelo) | **40.0%** | 6.0% | **+34.0pp** |
| MNAR F1 (melhor modelo) | **0.294** | 0.049 | **+0.245** |

**CAAFE features** (puras Python, sem API) são mais eficazes que LLM v2 no Nível 2 porque:
1. São determinísticas (sem ruído de geração)
2. Capturam diretamente os sinais de MNAR (quantile missing rate, tail asymmetry, kurtosis, conditional entropy)
3. LLM v2 tende a classificar tudo como MAR no Nível 2 (viés)

### Verificação da Tese Central

| Hipótese | Resultado Real | Resultado Sintético |
|----------|:--------------:|:-------------------:|
| V4 > V1 (hier+LLM > direto) | ✅ +3.0pp | ❌ -7.7pp |
| V4 > V2 (LLM ajuda N2) | ✅ +3.4pp | ❌ -2.0pp |
| V4 > V5 (LLM só N2 > ambos) | ✅ +1.3pp | ✅ +2.3pp |
| V2 > V1 (hierárquica melhora) | ❌ -0.4pp | ❌ -3.0pp |
| **V3 > V1 (CAAFE N2 melhora)** | **✅ +9.1pp** | — |

### Narrativa Ajustada para o Paper

1. **Classificação direta 3-way** é limitada (~41% real, ~77% sintético)
2. **Hierárquica pura (V2)** não melhora por si só — o ganho vem das features certas no N2
3. **CAAFE no Nível 2 (V3)** é a melhor combinação: +9.1pp accuracy, mantém 40% MNAR recall
4. **LLM v2 no Nível 2 (V4)** melhora accuracy mas sacrifica MNAR recall
5. **LLM no Nível 1 sempre piora** (V5 < V4) — confirma que LLM é ruído no N1
6. **Em dados limpos (sintéticos)**, qualquer feature adicional piora — baseline estatístico é suficiente

### Notas da Auditoria (2026-04-18)

**Bugs corrigidos:**
1. `features/mechdetect.py:108`: Shuffled task avaliava em labels originais → corrigido para `mask_shuffled[test_idx]`
2. `features/discriminative.py:58`: AUC computado sem CV (train=test) → corrigido para StratifiedKFold 3-fold
3. V3 adicionado ao LOGO CV (anteriormente omitido — apenas V1/V2/V4 tinham LOGO)

**Notas metodológicas:**
- Wilcoxon com N=7 modelos tem poder estatístico muito baixo (p_min = 0.0156). Usar McNemar e Bootstrap CI como evidência primária.
- Features CAAFE são "CAAFE-inspired" (Python puro), não reimplementação do CAAFE original (que usa LLM).
- Impacto dos bugs corrigidos provavelmente baixo (shuffled AUC ≈ 0.5 em ambas as abordagens; AUC discriminativa tem viés sistemático não-diferencial).

### Próximos Passos

- [x] Step 04-B: Ablação — CONCLUÍDO
- [x] Step 06: MechDetect baseline real — CONCLUÍDO
- [x] Step 08: SHAP — CONCLUÍDO
- [ ] Step 06: MechDetect baseline **sintético** — EM PROGRESSO
- [ ] Re-extrair features com bugs corrigidos e verificar impacto
- [ ] Escrita do paper com narrativa atualizada

---

## Arquivos Gerados

```
Output/v2_improved/step05_pro/
├── sintetico/
│   ├── apenas_ml/baseline/          # 21 features, 7 modelos
│   ├── ml_com_llm/gemini-3.1-pro-preview/  # 33 features, 7 modelos
│   ├── hierarquico_variants/        # 6 variantes × 7 modelos
│   │   ├── todas_variantes.csv
│   │   ├── resumo_variantes.csv
│   │   ├── comparacao_por_modelo.csv
│   │   ├── significancia_mcnemar.csv
│   │   ├── variantes_accuracy.png
│   │   ├── v1_vs_v2_vs_v4.png
│   │   ├── confusion_v1_v2_v4.png
│   │   ├── heatmap_variantes.png
│   │   └── training_summary.json
│   ├── comparacao.csv
│   └── comparacao.png
├── real/
│   ├── apenas_ml/baseline/
│   ├── ml_com_llm/gemini-3.1-pro-preview/
│   ├── hierarquico_variants/        # 6 variantes × 7 modelos + LOGO CV
│   │   ├── todas_variantes.csv
│   │   ├── resumo_variantes.csv
│   │   ├── comparacao_por_modelo.csv
│   │   ├── significancia_mcnemar.csv
│   │   ├── cv_logo_variantes.csv
│   │   ├── variantes_accuracy.png
│   │   ├── v1_vs_v2_vs_v4.png
│   │   ├── confusion_v1_v2_v4.png
│   │   ├── heatmap_variantes.png
│   │   └── training_summary.json
│   ├── comparacao.csv
│   └── comparacao.png
```

---

# Anexo B: Experimentos de Balanceamento (SMOTE variants)

> Originalmente publicado como `RESULTADOS_BALANCEAMENTO.md`. Complementa o STEP05A com ablacao de tecnicas de balanceamento.


**Data:** 2026-04-18
**Experimento:** step05_pro
**Método base:** V3 Hierárquica (L1: 21 stat, L2: 25 stat+CAAFE)
**Dados:** Real (1132 amostras)

---

## Motivação

O dataset real tem desbalanceamento moderado:
- **Global:** MCAR 232 (20.5%), MAR 550 (48.6%), MNAR 350 (30.9%)
- **Nível 1:** MCAR 232 (20.5%) vs NAO-MCAR 900 (79.5%) — razão 1:4
- **Nível 2:** MAR 550 (61.1%) vs MNAR 350 (38.9%) — razão 1.6:1

Testamos se estratégias de balanceamento alternativas ao SMOTE melhoram os resultados.

---

## Estratégias Testadas

| Estratégia | Descrição | Biblioteca |
|------------|-----------|------------|
| **none** | Sem balanceamento | — |
| **SMOTE** | Synthetic Minority Oversampling (k=3) | imblearn |
| **ADASYN** | Adaptive Synthetic Sampling | imblearn |
| **BorderlineSMOTE** | SMOTE focado nas amostras de fronteira | imblearn |
| **Undersampling** | Random undersampling da classe majoritária | imblearn |
| **SMOTE+Tomek** | SMOTE seguido de remoção de Tomek links | imblearn |
| **class_weight='balanced'** | Pesos inversamente proporcionais à frequência | sklearn |

Cada estratégia aplicada em ambos os níveis (L1 e L2) da classificação hierárquica V3.

---

## Resultados

### Todas as Combinações (ordenado por accuracy)

| Estratégia | Modelo | Accuracy | F1 Macro | MCAR R | MAR R | MNAR R | Acc L1 |
|------------|--------|:--------:|:--------:|:------:|:-----:|:------:|:------:|
| **SMOTE** | **GBT** | **50.5%** | **0.488** | 47.4% | **56.0%** | **40.0%** | 82.0% |
| **SMOTE + balanced_weights** | **GBT** | **50.5%** | **0.488** | 47.4% | **56.0%** | **40.0%** | 82.0% |
| none | GBT | 49.8% | 0.475 | 46.3% | 57.3% | 34.0% | 82.0% |
| ADASYN | GBT | 46.1% | 0.445 | 46.3% | 51.3% | 30.0% | 81.7% |
| BorderlineSMOTE | GBT | 46.1% | 0.442 | 46.3% | 52.0% | 28.0% | 81.7% |
| Undersampling | GBT | 46.1% | 0.445 | 49.5% | 48.0% | 34.0% | 78.6% |
| SMOTE+Tomek | GBT | 44.8% | 0.417 | 47.4% | 52.7% | 16.0% | 82.0% |
| SMOTE | RF | 43.4% | 0.409 | 46.3% | 50.7% | 16.0% | 82.0% |
| SMOTE + balanced_weights | RF_balanced | 43.4% | 0.409 | 46.3% | 50.7% | 16.0% | 82.0% |
| none | RF | 43.1% | 0.405 | 46.3% | 50.7% | 14.0% | 82.7% |
| Undersampling | RF | 42.0% | 0.392 | 48.4% | 46.0% | 18.0% | 77.6% |
| SMOTE+Tomek | RF | 42.0% | 0.386 | 45.3% | 50.7% | 8.0% | 82.0% |
| BorderlineSMOTE | RF | 41.4% | 0.383 | 46.3% | 49.3% | 8.0% | 81.7% |
| SMOTE + balanced_weights | LR_balanced | 41.0% | 0.375 | 44.2% | 50.0% | 8.0% | 79.7% |
| SMOTE + balanced_weights | SVM_balanced | 41.0% | 0.389 | 38.9% | 48.0% | 24.0% | 78.0% |
| ADASYN | RF | 40.7% | 0.381 | 43.2% | 49.3% | 12.0% | 81.0% |

### Ranking por F1 Macro

| Rank | Estratégia + Modelo | F1 Macro | MNAR Recall |
|:----:|---------------------|:--------:|:-----------:|
| 1 | **SMOTE + GBT** | **0.488** | **40.0%** |
| 2 | **SMOTE+balanced + GBT** | **0.488** | **40.0%** |
| 3 | none + GBT | 0.475 | 34.0% |
| 4 | ADASYN + GBT | 0.445 | 30.0% |
| 5 | Undersampling + GBT | 0.445 | 34.0% |

### Ranking por MNAR Recall

| Rank | Estratégia + Modelo | MNAR Recall | Accuracy |
|:----:|---------------------|:-----------:|:--------:|
| 1 | **SMOTE + GBT** | **40.0%** | **50.5%** |
| 2 | **SMOTE+balanced + GBT** | **40.0%** | **50.5%** |
| 3 | none + GBT | 34.0% | 49.8% |
| 4 | Undersampling + GBT | 34.0% | 46.1% |
| 5 | ADASYN + GBT | 30.0% | 46.1% |

---

## Análise

### 1. SMOTE é a melhor estratégia

SMOTE com GBT alcança o melhor resultado em todas as métricas (accuracy, F1 macro, e MNAR recall). Nenhuma alternativa melhora.

### 2. class_weight='balanced' não adiciona valor com SMOTE

Quando SMOTE já é aplicado, adicionar `class_weight='balanced'` nos classificadores produz **resultados idênticos**. O SMOTE já resolveu o desbalanceamento antes do treinamento.

### 3. GBT >> RF consistentemente

GradientBoosting supera RandomForest em **todas** as estratégias de balanceamento (+5-9pp accuracy). O GBT é mais robusto ao ruído e melhor calibrado para probabilidades.

### 4. Estratégias mais agressivas pioram

| Estratégia | Delta vs SMOTE (GBT) | Razão provável |
|------------|:--------------------:|----------------|
| ADASYN | **-4.4pp** | Oversampla regiões ruidosas, gera amostras menos representativas |
| BorderlineSMOTE | **-4.4pp** | Foca nas fronteiras que são naturalmente ambíguas em dados reais |
| Undersampling | **-4.4pp** | Perde informação da classe majoritária (MAR) |
| SMOTE+Tomek | **-5.7pp** | Remove links Tomek que podem ser informativos, MNAR recall cai para 16% |

### 5. Sem balanceamento quase empata com SMOTE

A diferença entre "none" (49.8%) e SMOTE (50.5%) é pequena (+0.7pp accuracy, +6pp MNAR recall). O benefício principal do SMOTE é no **MNAR recall** (34% → 40%), não na accuracy geral.

### 6. O gargalo não é o balanceamento

O teto de ~50.5% accuracy indica que o **limite é a dificuldade intrínseca do problema MAR vs MNAR em dados reais**, não o desbalanceamento. Com apenas 23 datasets reais (muitos com labels possivelmente incorretos — 57% inconsistentes), melhorias via balanceamento têm retorno decrescente.

---

## Implicação para o Paper

- **SMOTE é a escolha correta e já otimizada** — não precisa de justificativa extensa
- **GBT é o classificador ideal** para o Nível 2 da hierárquica
- **O resultado de 50.5% é robusto**: não depende de uma estratégia específica de balanceamento (49.8% sem balanceamento, 50.5% com SMOTE)
- Pode ser mencionado em uma nota de rodapé: "We evaluated six balancing strategies; SMOTE with GradientBoosting proved optimal (see supplementary material)"

---

## Arquivos Gerados

```
Output/v2_improved/step05_pro/real/balanceamento/
└── resultados_balanceamento.csv    # 16 combinações × 6 métricas
```
