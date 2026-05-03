# Resultados: Fase 3 — Melhorias Estruturais (com correcao de data leakage)

**Data:** 2026-04-11
**Execucao:** Pipeline completo com Fase 3 + GroupShuffleSplit (sem data leakage)
**Experimento:** `fase3_groupsplit`
**Output:** `Output/v2_improved/fase3_groupsplit/`

---

## 0. Correcao Critica: Data Leakage via Bootstrap

### O Problema

A execucao inicial da Fase 3 (e todas as anteriores com dados reais bootstrapped) apresentou **data leakage**:

- O `subdividir_dados_reais.py` gera 50 bootstraps por arquivo original. Cada bootstrap compartilha linhas do mesmo dataset.
- O `train_model.py` usava `train_test_split` simples, colocando bootstraps do **mesmo arquivo** em treino E teste.
- O modelo aprendia a identificar **de qual dataset veio a amostra** (via X0_q75, X0_mean, etc.) em vez de aprender o mecanismo de missing.
- **Resultado inflado:** 100% de accuracy (RF, GB, SVM) era puro overfitting.

### A Correcao

1. `extract_features.py` agora salva `groups.csv` com o dataset de origem de cada amostra (ex: `MCAR_breastcancer_barenuclei`)
2. `train_model.py` usa `GroupShuffleSplit`: bootstraps do mesmo dataset ficam **todos no treino OU todos no teste**
3. Cross-validation usa `GroupKFold`: cada fold exclui todos os bootstraps de um dataset inteiro
4. `ensemble_model.py` tambem usa `GroupShuffleSplit`

### Verificacao

```
Grupos treino: [MAR_airquality_ozone, MAR_titanic_age, MCAR_breastcancer_barenuclei,
                MCAR_oceanbuoys_airtemp, MNAR_adult_capitalgain, MNAR_mroz_wages, ...]
Grupos teste:  [MAR_mammographic_density, MCAR_oceanbuoys_humidity, MNAR_pima_insulin]
Sem leakage: 0 grupos compartilhados
```

---

## 1. Configuracao do Experimento

| Metrica | Valor |
|---------|:-----:|
| Datasets originais | 9 (3 por mecanismo) |
| Amostras bootstrap | 445 |
| MCAR | 145 (breastcancer, oceanbuoys_airtemp, oceanbuoys_humidity) |
| MAR | 150 (airquality, mammographic, titanic) |
| MNAR | 150 (adult, mroz, pima) |
| Split | GroupShuffleSplit (25% teste, por dataset) |
| CV | GroupKFold (k=5, por dataset) |
| Treino | ~295 amostras (6-7 datasets) |
| Teste | ~150 amostras (2-3 datasets) |

---

## 2. Resultados: Baseline (apenas ML, 10 features)

### 2.1 Accuracy por Modelo

| Modelo | Accuracy | Obs |
|--------|:--------:|-----|
| **MLP** | **60.7%** | Melhor modelo — detecta MCAR (84%) e MAR (98%) |
| LogisticRegression | 52.7% | Detecta MCAR (58%) e MAR (100%) |
| KNN | 47.3% | |
| RandomForest | 33.3% | Classifica tudo como MCAR |
| GradientBoosting | 33.3% | Classifica tudo como MCAR |
| SVM_RBF | 33.3% | Classifica tudo como MCAR |
| NaiveBayes | 33.3% | Classifica tudo como MCAR |

### 2.2 Cross-Validation (Group 5-Fold)

| Modelo | CV Accuracy | Desvio (+/-) |
|--------|:-----------:|:------------:|
| RandomForest | 51.9% | 63.8% |
| LogisticRegression | 48.7% | 60.5% |
| MLP | 46.0% | 62.5% |
| NaiveBayes | 38.2% | 74.1% |
| GradientBoosting | 34.6% | 60.9% |
| KNN | 28.6% | 60.8% |
| SVM_RBF | 27.2% | 55.3% |

**Variancia extrema** (+/- 55-74%) indica que o desempenho varia drasticamente entre folds — depende de quais datasets ficam no treino vs teste.

### 2.3 Feature Importance (RandomForest)

| Rank | Feature | Importancia |
|:----:|---------|:-----------:|
| 1 | `X0_q75` | 28.0% |
| 2 | `X0_mean` | 24.2% |
| 3 | `X0_q50` | 15.2% |
| 4 | `X0_q25` | 11.3% |
| 5 | `little_proxy_score` | 9.9% |

Features de X0 dominam (78.7%) — mas agora sabemos que isso nao e util, pois reflete a distribuicao do dataset (fingerprint), nao o mecanismo.

### 2.4 Matriz de Confusao (MLP — melhor modelo)

```
           MCAR  MAR  MNAR
MCAR (50)  [42    8    0]
MAR  (50)  [ 1   49    0]
MNAR (50)  [ 0   50    0]
```

**Problema principal:** MNAR tem 0% de recall. O modelo nao consegue distinguir MNAR dos outros mecanismos. Confunde MNAR com MAR sistematicamente.

---

## 3. Resultados: ML + LLM (gemini-3-flash-preview, 18 features + PCA)

### 3.1 Accuracy por Modelo

| Modelo | Accuracy | Delta vs BL |
|--------|:--------:|:-----------:|
| LogisticRegression | 51.3% | -1.3% |
| KNN | 50.0% | +2.7% |
| MLP | 50.0% | -10.7% |
| SVM_RBF | 38.7% | +5.3% |
| RandomForest | 33.3% | 0.0% |
| GradientBoosting | 33.3% | 0.0% |
| NaiveBayes | 33.3% | 0.0% |

### 3.2 Cross-Validation (Group 5-Fold)

| Modelo | CV Accuracy | Desvio (+/-) |
|--------|:-----------:|:------------:|
| RandomForest | 49.7% | 63.6% |
| LogisticRegression | 48.8% | 47.5% |
| NaiveBayes | 37.0% | 74.2% |
| MLP | 36.8% | 45.4% |
| GradientBoosting | 34.1% | 60.1% |
| SVM_RBF | 29.4% | 51.1% |
| KNN | 27.3% | 48.8% |

### 3.3 Feature Importance LLM

| Rank | Feature | Importancia | Tipo |
|:----:|---------|:-----------:|:----:|
| 8 | `llm_mnar_conf` | 1.6% | LLM |
| 10 | `llm_mcar_vs_mnar` | 1.2% | LLM |
| 12 | `llm_pattern_clarity` | 1.1% | LLM |

**Importancia total LLM: 6.6%** — features LLM tem contribuicao marginal.

---

## 4. Comparacao: Baseline vs LLM vs Ensemble

| Modelo | Baseline | +LLM | Ensemble | ENS vs BL |
|--------|:--------:|:----:|:--------:|:---------:|
| **MLP** | **60.7%** | 50.0% | **58.7%** | -2.0% |
| LogisticRegression | 52.7% | 51.3% | **54.0%** | **+1.3%** |
| KNN | 47.3% | 50.0% | 47.3% | 0.0% |
| SVM_RBF | 33.3% | 38.7% | 33.3% | 0.0% |
| RandomForest | 33.3% | 33.3% | 33.3% | 0.0% |
| GradientBoosting | 33.3% | 33.3% | 33.3% | 0.0% |
| NaiveBayes | 33.3% | 33.3% | 33.3% | 0.0% |

**O ensemble nao traz ganho significativo** porque ambos os modelos (baseline e LLM) falham nos mesmos casos.

---

## 5. Validacao de Rotulos

| Dataset | Rotulo | MCAR Consistente | Evidencia MAR | Diagnostico |
|---------|:------:|:----------------:|:-------------:|:-----------:|
| breastcancer_barenuclei | MCAR | SIM | NAO | CONSISTENTE |
| oceanbuoys_airtemp | MCAR | **NAO** | **SIM** | **INCONSISTENTE** |
| oceanbuoys_humidity | MCAR | **NAO** | **SIM** | **INCONSISTENTE** |
| airquality_ozone | MAR | SIM* | NAO | **FRACO** |
| mammographic_density | MAR | SIM* | SIM | CONSISTENTE |
| titanic_age | MAR | SIM* | SIM | CONSISTENTE |
| adult_capitalgain | MNAR | SIM | NAO | nao testavel |
| mroz_wages | MNAR | SIM | NAO | nao testavel |
| pima_insulin | MNAR | SIM | NAO | nao testavel |

**3/9 rotulos consistentes.** Os 2 datasets oceanbuoys (69% das amostras MCAR) provavelmente sao MAR.

---

## 6. Diagnostico: Por que o Modelo Falha

### 6.1 O problema fundamental: 9 datasets nao sao suficientes

Com GroupShuffleSplit, o teste contem **2-3 datasets nunca vistos**. O modelo precisa generalizar de 6-7 datasets de treino para datasets com distribuicoes completamente diferentes. Com apenas 3 datasets por mecanismo, cada um com distribuicao unica de X0, nao ha como aprender um sinal generalizado.

### 6.2 Features estatisticas sao "fingerprints" de dataset, nao de mecanismo

As features dominantes (X0_q75, X0_mean, X0_q50, X0_q25) refletem a **distribuicao original** de cada dataset, nao o mecanismo de missing. Por exemplo:
- `adult_capitalgain` (MNAR): X0_mean ~ 0.09
- `breastcancer` (MCAR): X0_mean ~ 0.28
- `titanic` (MAR): X0_mean ~ 0.37

Quando o modelo ve um dataset de teste com X0_mean ~ 0.09, ele nao sabe que "isso e MNAR" — ele so sabe que "isso parece com adult_capitalgain". Se adult_capitalgain estiver no teste (e nao no treino), o modelo nao tem referencia.

### 6.3 MNAR e indistinguivel com features atuais

Todos os modelos tem recall 0% para MNAR. As features discriminativas (baseadas em X1) nao capturam o sinal de MNAR porque:
- MNAR depende do **proprio valor** de X0 (nao observavel quando missing)
- As features atuais medem correlacao de missingness com X1-X4, que e sinal de MAR
- Little's proxy e as features de regressao logistica nao distinguem MAR de MNAR

### 6.4 Rotulos inconsistentes poluem o treino

Oceanbuoys (rotulado MCAR) mostra forte evidencia de MAR. Isso ensina ao modelo que "quando ha correlacao mask-X1, pode ser MCAR" — confundindo o classificador.

---

## 7. Comparacao com Resultados Anteriores (Corrigidos)

| Metrica | Pre-Fases (v1) | Fases 1+2 | Fase 3 (sem leakage) |
|---------|:--------------:|:---------:|:--------------------:|
| Amostras | 43 | 300 | 445 |
| Split | train_test | train_test | **GroupShuffleSplit** |
| Melhor accuracy BL | 90.9%* | 98.7%* | **60.7%** |
| Variancia CV | +/-40%* | +/-1.5%* | **+/-55-74%** |
| Data leakage | Nao (chunking) | **SIM (bootstrap)** | **NAO** |

*Valores marcados com * estavam inflados por data leakage em Fases 1+2 (train_test_split sem GroupShuffleSplit com bootstraps do mesmo dataset).

**Nota:** Os resultados das Fases 1-2 com 300 amostras tambem sofriam de leakage. O resultado de v1 (43 amostras sem bootstrap) era chunking sequencial — cada chunk vinha do mesmo arquivo, entao tambem havia leakage (mas menos severo que bootstrap).

---

## 8. Proximos Passos Recomendados

### 8.1 Prioridade Alta: Mais Datasets

O problema raiz e ter **apenas 3 datasets por mecanismo**. A meta de 10+ por mecanismo do documento original e essencial. Com 10 datasets/mecanismo:
- GroupKFold teria 10 folds por classe
- O modelo veria distribuicoes mais diversas no treino
- CV seria mais estavel

**Datasets sugeridos (ja no ANALISE_RESULTADOS_REAIS.md):**
- MCAR: BRFSS, Heart Disease Cleveland, Wisconsin Diagnostic
- MAR: NHANES, German Credit, Hepatitis
- MNAR: Chronic Kidney Disease, BRFSS (income), Student Performance

### 8.2 Prioridade Alta: Features Invariantes ao Dataset

As features atuais (quantis de X0, media) sao especificas ao dataset. Precisamos de features que capturem o **mecanismo** independentemente da distribuicao:
- **Correlacao normalizada** entre mask e X1 (em vez de p-valor absoluto)
- **Razao de quantis** (q75/q25 dos observados vs missing) em vez de valores absolutos
- **Testes de permutacao** para calcular significancia relativa
- **Features de segunda ordem**: interacao entre little_proxy_score e correlacoes

### 8.3 Prioridade Media: Corrigir Rotulos

- Reclassificar ou remover oceanbuoys (MCAR → MAR)
- Validar MNAR com testes mais sofisticados (MissMecha)

### 8.4 Prioridade Media: Melhorar Prompt LLM

As features LLM contribuem apenas 6.6%. O prompt pode ser melhorado para:
- Focar em sinais do **mecanismo** em vez de descricao da distribuicao
- Comparar explicitamente MNAR vs MAR (atualmente confunde ambos)

---

## 9. Conclusao

A correcao do data leakage revelou que **os modelos nao generalizam para datasets nunca vistos**. O melhor resultado (MLP: 60.7%) e apenas marginalmente melhor que chute aleatorio (33.3%), e nenhum modelo detecta MNAR.

O LLM nao melhora o cenario: accuracy similar ao baseline, e features LLM contribuem apenas 6.6% de importancia. O ensemble nao ajuda porque ambos os modelos falham nos mesmos casos.

**A causa raiz nao e o pipeline ou o modelo — e a quantidade insuficiente de datasets reais (3 por mecanismo) e features que refletem a distribuicao do dataset em vez do mecanismo de missing.**

Para que esta abordagem funcione em dados reais, sao necessarios:
1. **10+ datasets por mecanismo** com distribuicoes diversas
2. **Features invariantes** que capturem o mecanismo independentemente da distribuicao de X0
3. **Rotulos validados** estatisticamente
