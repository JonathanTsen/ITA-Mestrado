# Resultados: Impacto das Fases 1 e 2 nos Dados Reais

> **AVISO SUPERSEDED:** Os resultados deste documento sofriam de **data leakage via bootstrap**
> (bootstraps do mesmo dataset apareciam em treino E teste). A correcao veio na Fase 3 com
> `GroupShuffleSplit`. **Use [RESULTADOS_FASE3.md](RESULTADOS_FASE3.md) como referencia.**
> Este arquivo e mantido apenas como registro historico da evolucao metodologica.

**Data:** 2026-04-11
**Execucao:** Pipeline completo com bootstrap (300 amostras) + todas as melhorias das Fases 1 e 2
**Experimento:** `fase12_bootstrap300` (resultados sofriam de data leakage — ver RESULTADOS_FASE3.md)
**Output:** `Output/v2_improved/fase12_bootstrap300/` (nao preservado — sobrescrito pela Fase 3)

---

## 1. Mudanca mais impactante: Bootstrap (43 -> 300 amostras)

| Metrica | Antes (v1) | Depois (v2 + Fases 1-2) |
|---------|:-:|:-:|
| Amostras totais | 43 | **300** |
| Amostras por classe | 16/11/16 (desbalanceado) | **100/100/100** (perfeito) |
| Razao amostras/features (baseline) | 1.8:1 | **30:1** |
| Razao amostras/features (+LLM) | 1.8:1 | **16.7:1** |
| Amostras de treino | 32 | **225** |
| Amostras de teste | 11 | **75** |

---

## 2. Resultados: Baseline (apenas ML)

### 2.1 Accuracy por Modelo

| Modelo | Antes (v1) | Depois (v2) | Delta |
|--------|:-:|:-:|:-:|
| GradientBoosting | 90.9% | **98.7%** | **+7.8%** |
| RandomForest | 81.8% | **98.7%** | **+16.9%** |
| SVM_RBF | 72.7% | **98.7%** | **+26.0%** |
| MLP | 72.7% | **97.3%** | **+24.6%** |
| KNN | 72.7% | **93.3%** | **+20.6%** |
| LogisticRegression | 63.6% | **90.7%** | **+27.1%** |
| NaiveBayes | 54.5% | **84.0%** | **+29.5%** |

### 2.2 Cross-Validation (Repeated Stratified 5-Fold, 3x)

| Modelo | CV Accuracy | Desvio (+/-) |
|--------|:-:|:-:|
| GradientBoosting | 99.6% | 1.5% |
| RandomForest | 99.2% | 2.4% |
| SVM_RBF | 98.8% | 2.6% |
| MLP | 98.6% | 2.4% |
| KNN | 95.3% | 3.7% |
| LogisticRegression | 91.6% | 7.6% |
| NaiveBayes | 85.3% | 7.8% |

**Variancia CV caiu de +/-40% para +/-1.5-7.8%.** Meta do documento era < 15%. Atingida com folga.

### 2.3 Feature Importance (RandomForest baseline)

| Rank | Feature | Importancia |
|:----:|---------|:-:|
| 1 | `log_pval_X1_mask` | 17.5% |
| 2 | `little_proxy_score` | 16.6% |
| 3 | `X0_q75` | 13.7% |
| 4 | `X1_mannwhitney_pval` | 12.6% |
| 5 | `X0_q25` | 8.8% |

**Mudanca vs v1:** Features discriminativas (`log_pval_X1_mask`, `little_proxy_score`) agora dominam tambem nos dados reais (antes eram os quantis de X0). Isso indica que o bootstrap gerou amostras com sinais discriminativos mais claros.

### 2.4 Confusao entre Classes

Baseline v2 (RF): Apenas **1 erro** em 75 amostras (1 MCAR classificado como MNAR).
- MCAR: 24/25 corretos (96% recall)
- MAR: 25/25 corretos (100% recall)
- MNAR: 25/25 corretos (100% recall)

---

## 3. Resultados: ML + LLM (gemini-3-flash-preview)

### 3.1 Accuracy por Modelo

| Modelo | Antes (v1) | Depois (v2) | Delta |
|--------|:-:|:-:|:-:|
| RandomForest | 81.8% | **100.0%** | **+18.2%** |
| GradientBoosting | 72.7% | **98.7%** | **+26.0%** |
| SVM_RBF | 63.6% | **92.0%** | **+28.4%** |
| MLP | 63.6% | **92.0%** | **+28.4%** |
| LogisticRegression | 63.6% | **89.3%** | **+25.7%** |
| KNN | 54.5% | **85.3%** | **+30.8%** |
| NaiveBayes | 45.5% | **85.3%** | **+39.8%** |

### 3.2 Cross-Validation

| Modelo | CV Accuracy | Desvio (+/-) |
|--------|:-:|:-:|
| RandomForest | 99.3% | 2.0% |
| GradientBoosting | 99.2% | 1.7% |
| MLP | 95.1% | 10.3% |
| SVM_RBF | 94.8% | 5.4% |
| LogisticRegression | 92.3% | 6.5% |
| KNN | 88.7% | 10.2% |
| NaiveBayes | 84.1% | 7.9% |

### 3.3 Feature Importance (RandomForest +LLM)

| Rank | Feature | Importancia | Tipo |
|:----:|---------|:-:|:-:|
| 1 | `log_pval_X1_mask` | 15.1% | Estatistica |
| 2 | `little_proxy_score` | 14.1% | Estatistica |
| 3 | `X0_q75` | 13.3% | Estatistica |
| 4 | `X1_mannwhitney_pval` | 9.3% | Estatistica |
| 5 | `X0_q25` | 7.9% | Estatistica |
| ... | | | |
| 10 | **`llm_mar_conf`** | **4.0%** | LLM |
| 12 | **`llm_mnar_conf`** | **2.3%** | LLM |

**Importancia total LLM: 11.3%** (antes era ~0% por ser ruido puro). A feature LLM mais util e `llm_mar_conf` (4.0%), seguida de `llm_mnar_conf` (2.3%).

---

## 4. Comparacao: Baseline vs LLM (v2)

| Modelo | Baseline v2 | +LLM v2 | Delta v2 | Delta v1 (referencia) |
|--------|:-:|:-:|:-:|:-:|
| RandomForest | 98.7% | **100.0%** | **+1.3%** | 0.0% |
| GradientBoosting | 98.7% | 98.7% | 0.0% | -18.2% |
| SVM_RBF | 98.7% | 92.0% | -6.7% | -9.1% |
| MLP | 97.3% | 92.0% | -5.3% | -9.1% |
| KNN | 93.3% | 85.3% | -8.0% | -18.2% |
| LogisticRegression | 90.7% | 89.3% | -1.3% | 0.0% |
| NaiveBayes | 84.0% | 85.3% | **+1.3%** | -9.1% |

### 4.1 Interpretacao

**O LLM ainda nao supera consistentemente o baseline**, mas a situacao melhorou drasticamente:

- **v1:** LLM piorou em 5/7 modelos, com quedas de ate -18.2%
- **v2:** LLM piorou em 4/7 modelos, mas quedas menores (max -8.0%)
- **v2:** LLM melhorou em 2/7 modelos (RF: +1.3%, NB: +1.3%) e empatou em 1 (GB)
- **v2:** RandomForest + LLM atingiu **100% de accuracy** (vs 98.7% baseline)

**O problema fundamental mudou:** Antes, o LLM era destrutivo (adicionava ruido). Agora, o LLM e neutro-a-levemente-positivo para modelos robustos (RF, GB, NB), mas ainda prejudica modelos sensiveis a dimensionalidade (SVM, KNN, MLP).

---

## 5. Validacao dos Criterios de Sucesso

| Criterio (do ANALISE_RESULTADOS_REAIS.md) | Meta | Resultado | Status |
|------|:-:|:-:|:-:|
| Accuracy media LLM > baseline em 5+ modelos | 5/7 | 2/7 | NAO ATINGIDO |
| Variancia CV < 15% | < 15% | **1.5-10.3%** | ATINGIDO |
| Features LLM importancia > 15% e variancia > 0.05 | > 15% | 11.3% | PARCIAL |
| Resultado reprodutivel (CV estavel) | Sim | **Sim (CV +/- 2%)** | ATINGIDO |

---

## 6. Impacto por Mudanca Implementada

| Mudanca | Impacto Medido |
|---------|---------------|
| **Bootstrap (n=300)** | **MASSIVO.** Causa raiz resolvida: de 43 para 300 amostras, classes balanceadas, CV estavel. Baseline saltou de 54-91% para 84-99%. |
| **Shuffle** | Incorporado ao bootstrap (reamostragem ja e aleatoria). |
| **Fallback NaN + imputacao mediana** | Positivo. Eliminado vies em ~28% das amostras. Features LLM agora tem importancia > 0. |
| **Prompt recalibrado** | Positivo. `llm_mar_conf` e a 10a feature mais importante (4.0%). Antes era ruido puro. |
| **Feature selection** | Nao ativada (n=300, max_features=30 > 18). Correto — so atua em datasets pequenos. |
| **Hiperparametros adaptativos** | Nao ativados (n=300 > 100). Correto — regime grande selecionado. |
| **SMOTE** | Classes ja balanceadas (100/100/100), SMOTE nao alterou distribuicao. Util se coletar mais dados desbalanceados. |
| **CV adaptativa** | Repeated Stratified 5-Fold (3x). Variancia de +/-40% caiu para +/-1.5-7.8%. |

---

## 7. Diagnostico: Por que o LLM Ainda Prejudica SVM/KNN/MLP

Com 300 amostras e 18 features (vs 10 baseline), a razao amostras/features e 16.7:1 — aceitavel, mas nao folgada. Modelos sensiveis a dimensionalidade (SVM com kernel RBF, KNN, MLP) sofrem com as 8 features LLM adicionais mesmo quando elas tem algum sinal.

**Evidencia:** Na CV, SVM cai de 98.8% (baseline) para 94.8% (+LLM), e KNN de 95.3% para 88.7%. Isso e consistente com a maldicao da dimensionalidade residual.

**Solucao sugerida (Fase 3):**
- PCA antes de SVM/KNN/MLP para reduzir 18 features a ~8 componentes
- Ou forcar feature selection (`SelectKBest(k=10)`) quando LLM features estao presentes e n < 500

---

## 8. Proximos Passos Recomendados

1. **PCA para modelos sensiveis** (Fase 3, acao 11) — deve resolver a queda em SVM/KNN/MLP
2. **Coletar mais datasets reais** (Fase 3, acao 9) — validar se os resultados generalizam alem dos 6 arquivos atuais (bootstraps sao do mesmo dado)
3. **Validar rotulos com MissMecha** (Fase 3, acao 12) — confirmar que os mecanismos atribuidos estao corretos
4. **Ensemble adaptativo** (Fase 3, acao 10) — usar LLM so para amostras de baixa confianca do baseline

---

## 9. Conclusao

As Fases 1 e 2 transformaram o cenario dos dados reais:

| Metrica | Antes (v1) | Depois (v2) |
|---------|:-:|:-:|
| Melhor accuracy baseline | 90.9% | **98.7%** |
| Melhor accuracy +LLM | 81.8% | **100.0%** |
| Variancia CV | +/-40% | **+/-1.5%** |
| Modelos onde LLM melhora | 0/7 | **2/7** |
| Modelos onde LLM piora >5% | 3/7 | **3/7** |
| Pior queda do LLM | -18.2% | **-8.0%** |

**O bootstrap foi a mudanca mais impactante**, responsavel pela maior parte da melhoria. O prompt recalibrado e o fallback NaN contribuiram para que as features LLM passassem de "ruido puro" para "levemente informativas". O LLM ainda nao cumpre o criterio de sucesso principal (superar baseline em 5+ modelos), mas a lacuna e muito menor e potencialmente resolvivel com PCA e mais dados.
