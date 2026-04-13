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
