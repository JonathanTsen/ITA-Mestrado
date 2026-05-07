# STEP 08: Análise SHAP + Error Analysis

**Status: PENDENTE**
**Estimativa: 1-2 dias**
**Dependência: Steps 04-B, 05-A, 05-B, 06, 07 (precisa dos resultados de todos os métodos incluindo LOGO CV)**

---

## Motivação

SHAP é essencial para a tese central do paper: **provar que features LLM são mais importantes no Nível 2 (MAR vs MNAR) do que no Nível 1 (MCAR vs não-MCAR)**. RF importance mostra quanto cada feature importa, mas SHAP mostra a direção e interação — crítico para argumentar que LLM deve ser usado cirurgicamente.

Error analysis identifica onde a hierárquica melhora e onde ainda falha — informando limitações e future work.

---

## Parte A: Análise SHAP

**Arquivo a criar:** `v2_improved/analyze_shap.py`

### Implementação

```python
import shap

# Para o melhor modelo (provavelmente RF ou GBT)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

### Figuras a gerar (foco no argumento hierárquico + LLM)

1. **SHAP Nível 1 (MCAR vs não-MCAR)** — beeswarm mostrando que features estatísticas dominam, LLM features com baixa importância
2. **SHAP Nível 2 (MAR vs MNAR)** — beeswarm mostrando features LLM entre top rankings ← **figura chave do paper**
3. **SHAP comparison Nível 1 vs Nível 2** — gráfico lado a lado mostrando a mudança de importância das features LLM
4. **Dependence plots** — top 3 features LLM no Nível 2, como o valor afeta predição MAR vs MNAR
5. **SHAP por classe (3-way direto)** — para comparação: mostra LLM diluído no problema 3-way

### Insights esperados (hipóteses a confirmar)

- **Nível 1:** `mechdetect_delta_complete_shuffled` e `little_proxy_score` dominam. Features LLM < 5% importância.
- **Nível 2:** `X0_censoring_score` + features LLM (`llm_mcar_vs_mnar`, `llm_pattern_clarity`) entre top 5. Features LLM > 15% importância.
- **3-way direto:** Features LLM diluídas (~6-14% importância) — confirma que aplicar LLM no geral é ineficiente.
- Isso prova a tese: **LLM deve ser focado no subproblema onde features estatísticas falham.**

---

## Parte B: Error Analysis

### 1. Análise por variante sintética

As 12 variantes (gerador_v2.py): 3 MCAR + 5 MAR + 4 MNAR, cada uma com 4 distribuições base (uniform, normal, exponential, beta) × 100 amostras.

| Variante | Mecanismo | N | Accuracy | Erros Mais Comuns |
|----------|:---------:|:-:|:--------:|-------------------|
| MCAR_random | MCAR | 300 | ? | ? confundido com ? |
| MCAR_bernoulli | MCAR | 300 | ? | ... |
| MCAR_block | MCAR | 300 | ? | ... |
| MAR_linear_X1 | MAR | 200 | ? | ... |
| MAR_threshold_X1 | MAR | 200 | ? | ... |
| MAR_logistic_X1 | MAR | 200 | ? | ... |
| MAR_quantile_X1 | MAR | 200 | ? | ... |
| MAR_interaction | MAR | 200 | ? | ... |
| MNAR_censura_X0 | MNAR | 300 | ? | ... |
| MNAR_self_mask_X0 | MNAR | 300 | ? | ... |
| MNAR_threshold_X0 | MNAR | 300 | ? | ... |
| MNAR_logistic_X0 | MNAR | 300 | ? | ... |

**Nota:** Nomes das variantes são aproximados — verificar em `gerador_v2.py` os nomes exatos.

**Perguntas a responder:**
- Quais variantes são mais fáceis/difíceis?
- MNAR_censura é mais detectável que MNAR_self_mask?
- MAR_linear vs MAR_threshold — algum é mais difícil?

### 2. Análise por dataset real

| Dataset | Mecanismo | Label Validado | Accuracy | Notas |
|---------|:---------:|:--------------:|:--------:|-------|
| autompg | MAR | ✅ | ? | ... |
| breastcancer | MAR | ✅ | ? | ... |
| oceanbuoys | MAR (reclassificado) | ✅ | ? | ... |
| ... | ... | ... | ... | ... |

**Perguntas a responder:**
- Datasets com labels inconsistentes são mais difíceis? (esperamos que sim)
- Correlação entre taxa de missing e difficulty?
- Correlação entre N amostras e difficulty?
- Algum domínio (médico, financeiro, etc.) é sistematicamente mais difícil?

### 3. Confusion Analysis

- **Heatmap de confusão** antes/depois da hierárquica
- **Análise de erros MCAR↔MNAR**: quais features diferem entre MCAR corretamente classificado vs MCAR confundido com MNAR?
- **t-SNE/UMAP** do espaço de features colorido por: (a) classe real, (b) acerto/erro

### 4. Tabela Comparativa Final (para o paper)

| Método | Acc Sint | Acc Real | MCAR R | MAR R | MNAR R | F1 Macro | N Feat | LLM? | Tempo |
|--------|:--------:|:--------:|:------:|:-----:|:------:|:--------:|:------:|:----:|:-----:|
| MechDetect original | ? | ? | ? | ? | ? | ? | 5 | Não | ? |
| PKLM + heurísticas | ? | ? | ? | ? | ? | ? | 1+N | Não | ? |
| Direto 3-way (21f) | 77% | 43% | ? | ? | ? | ? | 21 | Não | ? |
| Direto + LLM (29f) | ? | 46% | ? | ? | ? | ? | 29 | Sim | ? |
| Hier. puro (V2) | ? | ? | ? | ? | ? | ? | 21 | Não | ? |
| Hier. + CAAFE N2 (V3) | ? | ? | ? | ? | ? | ? | 21+4 | Não* | ? |
| **Hier. + LLM N2 (V4)** | **?** | **?** | **?** | **?** | **?** | **?** | **21+8** | **Sim** | **?** |
| Hier. + LLM ambos (V6) | ? | ? | ? | ? | ? | ? | 29 | Sim | ? |

*Atualização terminológica:* as features CAAFE usadas no código atual são
CAAFE-inspired e determinísticas (Python puro, sem API em runtime). Elas não
são geradas por LLM; o CAAFE original de Hollmann et al. usa LLM para gerar
código de features, mas este projeto implementa uma adaptação manual focada em
MNAR. Ver `docs/caafe_mnar.md`.

---

## Testes de Validação

### Teste 1: SHAP é consistente com RF importance
Top 5 features em SHAP devem ser similares ao top 5 em RF importance (não necessariamente idêntica, mas sobreposição > 60%).

### Teste 2: Error analysis revela padrões
Deve haver pelo menos 1 insight não-óbvio sobre quais variantes/datasets são mais difíceis e por quê.

### Teste 3: Tabela comparativa completa
Todas as células da tabela final preenchidas com valores reais.

---

## Critério de Conclusão

- [ ] SHAP Nível 1 gerado (mostrando LLM irrelevante)
- [ ] SHAP Nível 2 gerado (mostrando LLM importante) ← **figura chave**
- [ ] SHAP comparison Nível 1 vs Nível 2 lado a lado
- [ ] SHAP dependence plots top 3 features LLM no Nível 2
- [ ] SHAP 3-way direto (para comparação)
- [ ] Error analysis por variante sintética
- [ ] Error analysis por dataset real
- [ ] Confusion analysis com visualizações
- [ ] t-SNE/UMAP plot
- [ ] Tabela comparativa final completa

---

# Anexo: Resultados do Experimento

> Originalmente publicado como `RESULTADOS_STEP08.md`. Consolidado aqui em 2026-04-19.


**Data:** 2026-04-18
**Experimento:** step05_pro
**Status:** CONCLUÍDO

---

## Descoberta Principal: CAAFE Features Dominam o Nível 2 em Dados Reais

### SHAP Importance — Dados Reais

**Nível 1 (MCAR vs NAO-MCAR) — 21 stat features:**

| Rank | Feature | Mean |SHAP| | Grupo |
|:----:|---------|:----------:|-------|
| 1 | X0_censoring_score | alto | MNAR-específica |
| 2 | X1_mean_diff | alto | Discriminativa |
| 3 | X0_ks_obs_vs_imputed | alto | MNAR-específica |
| 4 | X0_obs_skew_diff | alto | Estatística |
| 5 | X0_obs_vs_full_ratio | alto | Estatística |

→ Features estatísticas e MNAR-específicas dominam. Nenhuma CAAFE.

**Nível 2 (MAR vs MNAR) — 25 features (stat + CAAFE):**

| Rank | Feature | Mean |SHAP| | Grupo |
|:----:|---------|:----------:|-------|
| 1 | X0_obs_vs_full_ratio | 0.0767 | Estatística |
| **2** | **caafe_cond_entropy_X0_mask** | **0.0721** | **CAAFE** |
| **3** | **caafe_kurtosis_excess** | **0.0637** | **CAAFE** |
| **4** | **caafe_tail_asymmetry** | **0.0541** | **CAAFE** |
| 5 | X0_obs_skew_diff | — | Estatística |

→ **3 das top 5 features são CAAFE!** Isto é a evidência central do paper.

### SHAP Importance — Dados Sintéticos (contraste)

**Nível 2 (MAR vs MNAR) — 25 features (stat + CAAFE):**

| Rank | Feature | Grupo |
|:----:|---------|-------|
| 1 | X1_mean_diff | Discriminativa |
| 2 | X1_mannwhitney_pval | Discriminativa |
| 3 | log_pval_X1_mask | Discriminativa |
| 4 | coef_X1_abs | Discriminativa |
| 5 | mechdetect_auc_excluded | MechDetect |
| ... | ... | ... |
| **16** | **caafe_kurtosis_excess** | **CAAFE** |
| **21** | **caafe_cond_entropy_X0_mask** | **CAAFE** |

→ Em sintéticos, CAAFE features são **irrelevantes** (bottom do ranking). Features discriminativas baseadas em X1 bastam.

### Interpretação: Por que CAAFE Importa em Reais mas Não em Sintéticos?

1. **Dados sintéticos** têm relações limpas e controladas → features discriminativas (correlação X1-mask) detectam MAR vs MNAR diretamente
2. **Dados reais** têm ruído, relações não-lineares, e distribuições complexas → features discriminativas perdem poder
3. **CAAFE features** capturam **padrões de segunda ordem** da distribuição de X0:
   - `caafe_cond_entropy_X0_mask`: informação mútua entre X0 e a máscara de missing — alta para MNAR
   - `caafe_kurtosis_excess`: excesso de curtose — sinal de truncamento (MNAR)
   - `caafe_tail_asymmetry`: assimetria nas caudas — sinal de censura (MNAR)
4. Esses padrões são **mais robustos ao ruído** que correlações simples X1-mask

---

## SHAP: Nível 2 com TODAS as Features (33f, incluindo LLM)

Quando LLM features são incluídas no N2 junto com CAAFE:

**Dados Reais — Top features no N2 (33f):**
- CAAFE features continuam no top
- LLM features tipicamente ficam no meio do ranking
- LLM não adiciona valor discriminativo além do que CAAFE já captura

Isso confirma: **CAAFE é suficiente no N2, LLM é redundante.**

---

## Error Analysis — Dados Reais

### Accuracy por Classe (V3 Hier+CAAFE, GBT no L2)

| Classe | Accuracy |
|--------|:--------:|
| MCAR | ~47% |
| MAR | ~56% |
| MNAR | ~40% |

### Datasets Mais Difíceis

| Dataset | Classe | Accuracy | Possível Razão |
|---------|:------:|:--------:|----------------|
| MCAR_hypothyroid_t4u | MCAR | 0% | Confundido com MNAR (distribuição de tiróide é assimétrica) |
| MAR_sick_tsh | MAR | 4% | TSH tem distribuição muito skewed |
| MNAR_colic_refluxph | MNAR | 42% | MNAR Diffuse (pH reflete condição clínica) |

### Datasets Mais Fáceis

| Dataset | Classe | Accuracy | Possível Razão |
|---------|:------:|:--------:|----------------|
| MAR_titanic_age | MAR | 98% | Padrão MAR clássico (idade depende de classe/sexo) |
| MCAR_breastcancer_barenuclei | MCAR | 98% | Missing genuinamente aleatório |

---

## Figuras Geradas (para o Paper)

### Figuras Principais

1. **`shap_l1_vs_l2_comparison.png`** — **FIGURA CENTRAL DO PAPER**
   - Side-by-side: L1 (features stat dominam) vs L2 (CAAFE features entram no top 5)
   - Mostra visualmente que CAAFE é essencial no N2 mas dispensável no N1

2. **`shap_nivel2_caafe.png`** — Beeswarm do N2 com CAAFE features destacadas

3. **`shap_dependence_caafe.png`** — Como cada CAAFE feature influencia a predição MNAR

### Figuras de Suporte

4. **`shap_direto_3way.png`** — SHAP da classificação direta (para comparação)
5. **`shap_nivel1_stat.png`** — SHAP do L1 (MCAR vs NAO-MCAR)
6. **`shap_nivel2_all.png`** — SHAP do L2 com todas as 33 features
7. **`tsne_features.png`** — t-SNE colorido por classe real e por acerto/erro
8. **`umap_features.png`** — UMAP idem
9. **`confusion_v3_hier.png`** — Confusion matrix do V3

---

## Síntese para o Paper

### Argumento SHAP (Section 4.4 do paper)

> "SHAP analysis reveals a striking asymmetry between the two hierarchical levels.
> At Level 1 (MCAR vs non-MCAR), statistical and discriminative features dominate,
> with CAAFE features contributing negligible importance. At Level 2 (MAR vs MNAR),
> three of the four CAAFE features rank in the top 5 — `cond_entropy_X0_mask` (#2),
> `kurtosis_excess` (#3), and `tail_asymmetry` (#4) — collectively accounting for
> a substantial portion of the model's discriminative power.
>
> This pattern is absent in synthetic data, where discriminative features based on
> X1-mask correlation suffice for MAR-MNAR separation. The differential importance
> of CAAFE features in real vs. synthetic data suggests that these features capture
> higher-order distributional patterns (conditional entropy, tail behavior, kurtosis)
> that are essential for mechanism classification in noisy, real-world settings but
> redundant in controlled synthetic environments."

### Tese Confirmada

| Hipótese | SHAP Resultado |
|----------|:--------------:|
| CAAFE features importam mais no N2 que N1 | ✅ Top 5 no N2, ausente no N1 |
| Features stat dominam o N1 | ✅ Top 5 são todas stat |
| CAAFE > LLM no N2 (dados reais) | ✅ CAAFE no top, LLM no meio |
| CAAFE irrelevante em sintéticos | ✅ #16 e #21 no ranking |
| Diferentes features para diferentes níveis | ✅ Assimetria clara L1 vs L2 |

---

## Arquivos Gerados

```
Output/v2_improved/step05_pro/{sintetico|real}/shap_analysis/
├── shap_direto_3way.png              # SHAP beeswarm 3-way (3 classes)
├── shap_nivel1_stat.png              # SHAP L1: MCAR vs NAO-MCAR
├── shap_nivel2_caafe.png             # SHAP L2: MAR vs MNAR (stat+CAAFE)
├── shap_nivel2_all.png               # SHAP L2: MAR vs MNAR (todas 33f)
├── shap_l1_vs_l2_comparison.png      # Side-by-side L1 vs L2 [FIGURA CENTRAL]
├── shap_dependence_caafe.png         # Dependence plots CAAFE no L2
├── shap_importance_direto.csv        # Ranking features 3-way
├── shap_importance_nivel1.csv        # Ranking features L1
├── shap_importance_nivel2_caafe.csv  # Ranking features L2 (stat+CAAFE)
├── shap_importance_nivel2_all.csv    # Ranking features L2 (todas)
├── shap_comparison_l1_vs_l2.csv      # Delta importance L1 vs L2
├── tsne_features.png                 # t-SNE visualização
├── umap_features.png                 # UMAP visualização
├── confusion_v3_hier.png             # Confusion matrix V3
├── error_analysis.csv                # Predições + erros por amostra
├── error_by_dataset.csv              # Accuracy por dataset
└── shap_summary.json                 # Resumo com rankings
```
