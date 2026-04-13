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

*CAAFE features foram geradas por LLM mas executam como Python puro (sem API em runtime)

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
