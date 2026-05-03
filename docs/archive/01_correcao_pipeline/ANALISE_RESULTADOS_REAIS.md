# Analise dos Resultados: Dados Reais vs Sinteticos

> **AVISO SUPERSEDED:** Analise feita antes da correcao de data leakage (Fase 3). Os resultados
> dos dados reais aqui estavam **inflados por overfitting ao dataset de origem**. As hipoteses
> diagnosticas continuam validas (pequenez da amostra, ruido de labels, LLM over-confidence),
> mas os numeros especificos nao. Ver [RESULTADOS_FASE3.md](RESULTADOS_FASE3.md) para os
> resultados corrigidos que guiaram os planos 1-3.

## 1. Contexto do Problema

Este projeto classifica mecanismos de dados faltantes (MCAR, MAR, MNAR) usando Machine Learning, com a hipotese central de que features extraidas por LLM melhoram a acuracia em relacao ao baseline puramente estatistico.

**Nos dados sinteticos, a hipotese se confirmou** -- o LLM adicionou ate +3.3% de accuracy (KNN: 64.8% -> 68.1%; SVM: 68.5% -> 71.6%). Porem, **nos dados reais, o LLM piorou o desempenho** em todos os modelos exceto RandomForest (empate) e LogisticRegression (empate).

Este documento analisa as causas raiz, formula hipoteses testáveis e propoe melhorias concretas para que o LLM supere o baseline tambem em dados reais.

---

## 2. Resultados Observados

### 2.1 Comparativo de Accuracy

| Modelo | Sint. Baseline | Sint. +LLM | Delta Sint. | Real Baseline | Real +LLM | Delta Real |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| GradientBoosting | 64.5% | 65.2% | +0.6% | **90.9%** | 72.7% | **-18.2%** |
| RandomForest | 64.8% | 67.3% | +2.5% | 81.8% | 81.8% | 0.0% |
| SVM_RBF | 68.5% | **71.6%** | **+3.1%** | 72.7% | 63.6% | -9.1% |
| KNN | 64.8% | 68.1% | **+3.3%** | 72.7% | 54.5% | **-18.2%** |
| MLP | 65.6% | 62.7% | -2.9% | 72.7% | 63.6% | -9.1% |
| LogisticRegression | 69.9% | 70.2% | +0.4% | 63.6% | 63.6% | 0.0% |
| NaiveBayes | 69.9% | 69.7% | -0.2% | 54.5% | 45.5% | -9.1% |

### 2.2 Estabilidade (Cross-Validation 5-fold)

| Metrica | Sintetico | Real |
|---------|:-:|:-:|
| Melhor CV (baseline) | 71.4% +/- 3.5% | 76.4% +/- **39.7%** |
| Melhor CV (+LLM) | 72.1% +/- 3.5% | 74.4% +/- **45.3%** |
| Variancia media | **2-5%** | **22-45%** |

A variancia de +/-40% nos dados reais indica que **os resultados individuais nao sao estatisticamente confiaveis** -- a diferenca entre 90.9% e 72.7% pode ser puramente aleatoria.

### 2.3 Padroes de Confusao entre Classes

**Sintetico (n=375 teste):** Confusao sistematica entre Classe 0 (MCAR) e Classe 2 (MNAR). MCAR tem recall de apenas 51-59%, enquanto MAR tem 84-88%. O LLM ajuda especificamente na distincao MCAR vs MNAR.

**Real (n=11 teste):** Sem padrao de confusao dominante. Erros distribuidos entre todas as classes. Com LLM, GradientBoosting perde recall de MCAR (100% -> 50%) e ganha confusao entre MCAR e MAR/MNAR.

### 2.4 Features Dominantes

| Rank | Sintetico (Baseline) | Real (Baseline) |
|:----:|----------------------|-----------------|
| 1 | `X1_mean_diff` (19.7%) | `X0_q50` (21.7%) |
| 2 | `coef_X1_abs` (12.9%) | `X0_q25` (18.5%) |
| 3 | `X1_mannwhitney_pval` (12.6%) | `X0_q75` (14.6%) |
| 4 | `log_pval_X1_mask` (11.5%) | `X0_mean` (14.2%) |
| 5 | `X0_mean` (8.7%) | `X1_mean_diff` (7.4%) |

**Observacao critica:** Nos dados sinteticos, as features discriminativas (baseadas em X1) dominam. Nos dados reais, as features estatisticas de X0 (quantis) dominam. Isso sugere que os mecanismos de missing nos dados reais sao capturados por sinais fundamentalmente diferentes dos sinteticos.

---

## 3. Diagnostico: Por que o LLM Piora nos Dados Reais

### 3.1 Causa Raiz 1: Maldicao da Dimensionalidade (CRITICO)

Com apenas **43 amostras** (32 treino, 11 teste), adicionar 8 features LLM aumenta a dimensionalidade em 80% (de 10 para 18 features) sem aumento correspondente de amostras.

| Metrica | Sintetico | Real | Limiar Recomendado |
|---------|:-:|:-:|:-:|
| Amostras de treino | ~1.125 | 32 | - |
| Features (baseline) | 10 | 10 | - |
| Features (+LLM) | 18 | 18 | - |
| **Razao amostras/features** | **62:1** | **1.8:1** | **>10:1 (minimo)** |

A literatura recomenda no minimo 10-20 amostras por feature para evitar overfitting. Com razao de 1.8:1, o modelo memoriza ruido em vez de aprender padroes, e as features LLM se tornam vetores de ruido adicional.

### 3.2 Causa Raiz 2: Fallback do LLM com Vies Sistematico (CRITICO)

Quando o LLM falha (timeout, erro de parsing), o `extractor_v2.py` retorna valores default do Pydantic:

```python
# LLMAnalysisV2() defaults (linha 25-48 de extractor_v2.py):
evidence_consistency = 0.5   # "incerto"
anomaly_detected     = 0.0   # "sem anomalia"
distribution_shift   = 0.0   # "sem desvio"
mcar_confidence      = 0.33  # uniforme
mar_confidence       = 0.33  # uniforme
mnar_confidence      = 0.34  # leve vies MNAR
mcar_vs_mnar         = 0.5   # "incerto"
pattern_clarity      = 0.5   # "medio"
```

**Impacto medido:** Aproximadamente 12 de 43 amostras (~28%) receberam esses valores identicos. Essas amostras sao predominantemente MCAR (com estatisticas proximas de zero), mas recebem features LLM que nao distinguem nenhum mecanismo. Isso injeta ruido sistematico e confunde o classificador.

**Evidencia:** As features `llm_mcar_conf` e `llm_mcar_vs_mnar` tem desvio padrao de 0.034 e 0.038 nos dados reais (vs 0.198 e 0.311 nos sinteticos) -- sao praticamente constantes, sem capacidade discriminativa.

### 3.3 Causa Raiz 3: Distribution Shift entre Sintetico e Real

Os dados sinteticos e reais diferem fundamentalmente:

**Distribuicao de X0 (variavel com missing):**
- Sintetico: Uniforme [0, 1], media ~ 0.50, sem skew
- Real MCAR (oceanbuoys): Concentrada [0.61, 0.99], media ~ 0.81
- Real MAR (airquality): Skew positivo forte (~1.22), media ~ 0.25
- Real MNAR (wages): Vies alto, media ~ 0.62

**Distribuicao de X1-X4 (preditores):**
- Sintetico: Sempre uniforme [0, 1]
- Real: Distribuicoes variaveis, com X3 tendo 70% de valores zero em MNAR (wages)

**Consequencia para o LLM:** O prompt usa limiares calibrados para dados sinteticos:
```
Se mar_combined_evidence < 0.1 E mnar_combined_evidence < 0.1 → MCAR
Se mar_combined_evidence > 0.3 → MAR
Se mnar_combined_evidence > 0.2 E mnar_internal_consistency = 1 → MNAR
```

Nos dados reais, `mnar_combined_evidence = abs(X0_mean_dev) * 50 + abs(X0_skew) * 5`. Com X0_mean real de 0.81 (MCAR oceanbuoys), a desvio da media 0.5 esperada e 0.31, gerando `mnar_combined_evidence = 0.31 * 50 = 15.5` (saturado em 1.0). Isso faz o LLM classificar dados MCAR reais como MNAR -- um erro sistematico causado pela premissa de distribuicao uniforme.

### 3.4 Causa Raiz 4: Chunking Sequencial Cria Amostras Nao-Representativas

O script `subdividir_dados_reais.py` corta os arquivos em blocos sequenciais de 100 linhas. Como o missing nos dados reais tem clustering temporal (equipamento falha por periodos):

- MCAR chunk01: 0% missing (primeiras 100 linhas limpas)
- MCAR chunk02: ~69% missing (periodo de falha do equipamento)
- MNAR chunks 01-03: 0% missing (mulheres que trabalham listadas primeiro)

**Problema:** Chunks sem nenhum dado faltante geram features estatisticas e discriminativas degeneradas (AUC=0.5, coeficientes=0, p-valores=1). Esses chunks sao indistinguiveis entre mecanismos e adicionam ruido puro ao dataset.

### 3.5 Causa Raiz 5: Ausencia de Feature Selection

O `train_model.py` importa `SelectKBest` (linha 30) mas **nunca o utiliza**. Todas as 18 features sao passadas diretamente aos classificadores. Com n=32 amostras de treino, mesmo features com importancia de 0.5% (como `llm_mcar_vs_mnar` com 0.497%) sao usadas, adicionando dimensoes de ruido.

### 3.6 Causa Raiz 6: Hiperparametros Calibrados para Datasets Grandes

Os hiperparametros dos modelos foram escolhidos para os 3000 arquivos sinteticos:

- RandomForest: `n_estimators=400` -- 400 arvores para 32 amostras e extremo overfitting
- GradientBoosting: `n_estimators=300` -- idem
- MLP: `hidden_layer_sizes=(128, 64, 32)` -- 128 neuronios na primeira camada para 32 amostras
- KNN: `n_neighbors=5` -- com 32 amostras de treino e 3 classes, vizinhanca de 5 pode ser excessiva

---

## 4. Hipoteses para Melhorar o Desempenho do LLM em Dados Reais

### Hipotese 1: Aumentar o Numero de Amostras via Chunking Aleatorio

**Tese:** O chunking sequencial cria amostras com distribuicao de missing extrema (0% ou >>10%). Embaralhar as linhas antes de dividir distribuira o missing uniformemente entre chunks, criando amostras mais representativas.

**Teste proposto:**
```python
# Em subdividir_dados_reais.py, antes de dividir:
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

**Resultado esperado:** Cada chunk tera taxa de missing proxima da media do arquivo original (~6-10%), eliminando chunks sem missing e reduzindo variancia.

**Metrica de sucesso:** Reducao da variancia no CV de +/-40% para +/-15%.

### Hipotese 2: Bootstrap Augmentation para Multiplicar Amostras

**Tese:** Com apenas 6 arquivos originais, mesmo com chunking nao teremos amostras suficientes. Bootstrap com reamostragem de linhas pode gerar centenas de amostras sinteticas a partir dos dados reais.

**Teste proposto:**
- Para cada arquivo original, gerar 50 amostras bootstrap (sampling com reposicao de N linhas)
- Total: 6 arquivos * 50 bootstraps = 300 amostras
- Manter proporcao de missing por amostragem estratificada das linhas com/sem NaN

**Resultado esperado:** Razao amostras/features passa de 1.8:1 para ~16:1, dentro do limiar minimo.

**Metrica de sucesso:** LLM features passam a ter variancia > 0.1 e importancia > 5% individual.

### Hipotese 3: Feature Selection Adaptativa por Tamanho do Dataset

**Tese:** Com n=43 (ou n=300 apos bootstrap), deve-se limitar o numero de features a sqrt(n) ou n/10 para evitar overfitting.

**Teste proposto:**
```python
# Em train_model.py, antes do split:
from sklearn.feature_selection import SelectKBest, f_classif
k = min(int(len(X) / 10), X.shape[1])  # n/10, max todas
selector = SelectKBest(f_classif, k=k)
X = pd.DataFrame(selector.fit_transform(X, y), 
                  columns=X.columns[selector.get_support()])
```

Com n=43: k=4 features. Com n=300: k=18 (todas). Isso automaticamente descartaria features LLM ruidosas em datasets pequenos.

**Resultado esperado:** Features LLM so seriam mantidas se tiverem poder discriminativo real (F-statistic significativo).

**Metrica de sucesso:** Accuracy com LLM >= accuracy sem LLM para todos os modelos.

### Hipotese 4: Corrigir o Fallback do LLM

**Tese:** Retornar `NaN` em vez de valores default quando o LLM falha, e imputar com a mediana das amostras que tiveram resposta valida. Isso elimina o vies sistematico dos defaults.

**Teste proposto:**
```python
# Em extractor_v2.py, na funcao _call_llm_with_retry:
except Exception as e:
    if attempt == max_retries - 1:
        # Retorna NaN em vez de defaults
        return {k: float('nan') for k in LLMAnalysisV2().to_feature_dict()}
```

```python
# Em extract_features.py, apos extracao:
# Imputar NaN com mediana das amostras validas
from sklearn.impute import SimpleImputer
llm_cols = [c for c in X.columns if c.startswith('llm_')]
imputer = SimpleImputer(strategy='median')
X[llm_cols] = imputer.fit_transform(X[llm_cols])
```

**Resultado esperado:** Eliminacao do vies MNAR nos ~28% de amostras com fallback.

**Metrica de sucesso:** Variancia das features `llm_mcar_conf` e `llm_mcar_vs_mnar` aumenta de <0.04 para >0.1.

### Hipotese 5: Recalibrar o Prompt do LLM para Dados Reais

**Tese:** Os limiares no prompt (`mar_combined_evidence < 0.1`, `mnar_combined_evidence > 0.2`) e a formula de `mnar_combined_evidence` (que usa `abs(X0_mean_dev) * 50`) assumem X0 uniforme [0,1] com media 0.5. Dados reais tem medias de 0.25 a 0.81, saturando as metricas.

**Teste proposto:**
1. Remover limiares fixos do prompt
2. Substituir `X0_mean_deviation` (desvio de 0.5) por desvio relativo ao dataset: `(X0_mean - dataset_mean) / dataset_std`
3. Adicionar ao prompt: "A distribuicao de X0 NAO e uniforme. Considere a distribuicao real ao avaliar desvios."
4. Reduzir o multiplicador de 50 para 5: `mnar_combined_evidence = abs(X0_mean_dev) * 5 + abs(X0_skew) * 2`

**Resultado esperado:** LLM gera confidencias mais calibradas para dados com distribuicoes nao-uniformes.

**Metrica de sucesso:** Correlacao entre `llm_mcar_conf` e label MCAR > 0.3 nos dados reais.

### Hipotese 6: Normalizar Features LLM por Z-score antes do Treino

**Tese:** As features LLM tem escala e variancia muito diferentes entre sintetico e real. Normalizar por z-score antes do treino pode ajudar modelos que sao sensiveis a escala (KNN, SVM, MLP).

**Teste proposto:** Ja implementado para LogReg, SVM, KNN, MLP (usam `StandardScaler` no pipeline). Porem RandomForest e GradientBoosting nao usam. Adicionar normalizacao tambem para esses.

**Observacao:** Arvores de decisao sao invariantes a escala, entao o ganho sera marginal para RF/GB. O foco deve ser nos modelos com pipeline.

### Hipotese 7: Usar SMOTE para Balancear Classes nos Dados Reais

**Tese:** Os dados reais tem desbalanceamento (16 MCAR, 11 MAR, 16 MNAR). MAR e sub-representada por ter menos dados originais (airquality: 153 linhas, mammographic: 886 linhas vs 700+ para outros). SMOTE pode equalizar.

**Teste proposto:**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=min(3, min_class_count-1))
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

**Resultado esperado:** Melhora na precisao da classe MAR (atualmente com precision de 0.60 no baseline).

**Metrica de sucesso:** F1-macro > 0.85 para dados reais com LLM.

### Hipotese 8: Usar Leave-One-Out Cross-Validation

**Tese:** Com n=43, o split 75/25 gera conjunto de teste com apenas 11 amostras. LOOCV usa n-1 para treino e 1 para teste, repetido n vezes, dando estimativa mais estavel.

**Teste proposto:**
```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(modelo, X, y, cv=loo, scoring='accuracy')
```

**Resultado esperado:** Estimativa de accuracy mais confiavel, com intervalo de confianca mais estreito.

**Metrica de sucesso:** Diferenca entre accuracy LOOCV e accuracy teste < 5%.

### Hipotese 9: Ensemble Adaptativo (Baseline + LLM como Segundo Estagio)

**Tese:** Em vez de usar todas as features juntas, treinar o modelo baseline primeiro e usar features LLM apenas para amostras onde o modelo base tem baixa confianca.

**Teste proposto:**
1. Treinar modelo baseline com 10 features
2. Para amostras com probabilidade maxima < 0.6 (baixa confianca), usar modelo LLM com 18 features
3. Combinar predicoes

**Resultado esperado:** LLM corrige erros do baseline sem introduzir ruido nas amostras que o baseline ja classifica bem.

**Metrica de sucesso:** Accuracy do ensemble > max(baseline, LLM) em pelo menos 3 dos 7 modelos.

### Hipotese 10: PCA antes do Classificador

**Tese:** PCA reduz as 18 features a um numero menor de componentes ortogonais, eliminando correlacoes entre features LLM e estatisticas. Isso pode extrair o sinal util das features LLM sem o ruido.

**Teste proposto:**
```python
from sklearn.decomposition import PCA
# Manter componentes que explicam 95% da variancia
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
```

**Resultado esperado:** Reducao para 5-8 componentes, preservando >95% da informacao.

**Metrica de sucesso:** Accuracy com PCA+LLM > accuracy baseline sem PCA.

### Hipotese 11: Coletar Mais Datasets Reais

**Tese:** O problema fundamental e que 6 arquivos (2 por mecanismo) sao insuficientes. A literatura em missing data oferece dezenas de datasets com mecanismos documentados.

**Datasets sugeridos:**

| Dataset | Mecanismo | Fonte | Linhas |
|---------|-----------|-------|--------|
| NHANES (saude) | MAR | CDC | ~10.000 |
| BRFSS (saude publica) | MCAR | CDC | ~400.000 |
| German Credit | MAR | UCI | 1.000 |
| Adult Income | MNAR | UCI | 48.842 |
| Titanic | MAR | Kaggle | 891 |
| Wisconsin Breast Cancer | MCAR | UCI | 699 |
| Heart Disease (Cleveland) | MAR | UCI | 303 |
| Chronic Kidney Disease | MNAR | UCI | 400 |

**Meta:** 10+ datasets por mecanismo, gerando 100+ chunks.

**Metrica de sucesso:** Variancia CV < 10%.

### Hipotese 12: Usar MissMecha para Validacao Estatistica

**Tese:** O pacote MissMecha (2025) oferece testes estatisticos para verificar se os mecanismos atribuidos aos dados reais estao corretos. Se um dataset rotulado como MCAR na verdade e MAR, o classificador aprende sinais errados.

**Teste proposto:**
```bash
pip install missmecha
```
```python
from missmecha.tests import littles_mcar_test
# Validar cada dataset real
for file in real_files:
    df = pd.read_csv(file, sep='\t')
    result = littles_mcar_test(df)
    print(f"{file}: p-value={result.pvalue:.4f}")
    # p > 0.05 -> nao rejeita MCAR
```

**Resultado esperado:** Confirmacao (ou correcao) dos rotulos de mecanismo nos dados reais.

**Metrica de sucesso:** Todos os datasets MCAR tem p > 0.05 no teste de Little.

---

## 5. Plano de Acao Priorizado

### Fase 1: Quick Wins (impacto alto, esforco baixo) -- IMPLEMENTADA (2026-04-11)

| # | Acao | Arquivo | Impacto Esperado | Status |
|:-:|------|---------|:---------------:|:------:|
| 1 | Shuffle antes de chunking | `subdividir_dados_reais.py` | Elimina chunks sem missing | DONE |
| 2 | Substituir fallback por NaN + imputacao mediana | `extractor_v2.py`, `extract_features.py` | Elimina vies em 28% das amostras | DONE |
| 3 | Adicionar SelectKBest com k=n/10 | `train_model.py` | Descarta features ruidosas | DONE |
| 4 | Trocar CV fixa por LOOCV para n<50 | `train_model.py` | Estimativa mais confiavel | DONE |

**Detalhes da implementacao:**
- **Acao 1:** `df.sample(frac=1, random_state=42).reset_index(drop=True)` adicionado antes do loop de chunking
- **Acao 2a:** Fallback em `extractor_v2.py` agora retorna `{k: float('nan') ...}` em vez de `LLMAnalysisV2().to_feature_dict()`
- **Acao 2b:** `extract_features.py` agora faz imputacao diferenciada: `fillna(0)` para features estatisticas, `SimpleImputer(strategy='median')` para features LLM
- **Acao 3:** `SelectKBest(f_classif, k=max(5, n//10))` adicionado apos carregar dados, antes da definicao de modelos
- **Acao 4:** CV adaptativa: `LeaveOneOut()` para n<50, `RepeatedStratifiedKFold(5, 3x)` para n>=50

### Fase 2: Melhorias Moderadas (impacto alto, esforco medio) -- IMPLEMENTADA (2026-04-11)

| # | Acao | Arquivo | Impacto Esperado | Status |
|:-:|------|---------|:---------------:|:------:|
| 5 | Bootstrap augmentation (50x por arquivo) | `subdividir_dados_reais.py` | n=43 -> n=300 | DONE |
| 6 | Recalibrar prompt LLM (remover limiares fixos) | `extractor_v2.py` | Features LLM mais informativas | DONE |
| 7 | Hiperparametros adaptativos por tamanho de dataset | `train_model.py` | Menos overfitting | DONE |
| 8 | SMOTE para classe MAR | `train_model.py` | Melhor recall MAR | DONE |

**Detalhes da implementacao:**
- **Acao 5:** `subdividir_dados_reais.py` reescrito para bootstrap: 50 amostras por arquivo via `df.sample(replace=True)`, filtro de missing >= 1%, gera ~300 amostras totais. `run_all.py` agora chama este script antes de processar dados reais.
- **Acao 6a:** `X0_mean_dev` agora usa desvio relativo ao range `(mediana - media) / range` em vez de desvio absoluto de 0.5
- **Acao 6b:** Multiplicador `mnar_evidence` reduzido de 50 para 5, skew de 5 para 2
- **Acao 6c:** Instrucoes de raciocinio no prompt substituidas por comparacao de magnitude relativa; adicionada nota sobre distribuicao nao-uniforme de X0; `X0_obs_range` adicionado ao stats_dict
- **Acao 7:** `get_modelos(n_samples)` retorna modelos mais simples para n<100: RF(100 arvores, depth=5), GB(50, depth=3), SVM(C=1), KNN(k=3), MLP(32,16)
- **Acao 8:** SMOTE aplicado apos train_test_split com `k_neighbors=min(3, min_class-1)`. Fallback gracioso se imbalanced-learn nao estiver instalado.

### Fase 3: Melhorias Estruturais (impacto muito alto, esforco alto)

| # | Acao | Arquivo | Impacto Esperado |
|:-:|------|---------|:---------------:|
| 9 | Coletar 10+ datasets reais por mecanismo | Novos scripts | Resolver o problema raiz |
| 10 | Ensemble adaptativo (baseline + LLM) | Novo script | Combinar o melhor de ambos |
| 11 | PCA como pre-processamento | `train_model.py` | Reducao de dimensionalidade |
| 12 | Validar rotulos com MissMecha | Novo script | Garantir ground truth |

---

## 6. Experimentos Sugeridos (Matriz de Testes)

Para validar as hipoteses de forma sistematica, recomenda-se a seguinte matriz de experimentos:

```
Experimento 1: Baseline atual (controle)
  -> Chunking sequencial, sem feature selection, split 75/25
  -> Resultado: 90.9% baseline, 72.7% LLM (GradientBoosting)

Experimento 2: Shuffle + LOOCV
  -> Shuffle antes de chunking, LOOCV em vez de split
  -> Hipoteses testadas: H1, H8

Experimento 3: Shuffle + LOOCV + SelectKBest
  -> Adicionar feature selection k=n/10
  -> Hipoteses testadas: H1, H3, H8

Experimento 4: Shuffle + LOOCV + Fallback NaN
  -> Corrigir fallback do LLM
  -> Hipoteses testadas: H1, H4, H8

Experimento 5: Bootstrap (300 amostras) + SelectKBest
  -> Aumentar dataset via bootstrap
  -> Hipoteses testadas: H2, H3

Experimento 6: Bootstrap + Prompt Recalibrado
  -> Combinar aumento de dados com melhoria do prompt
  -> Hipoteses testadas: H2, H5

Experimento 7: Full Stack (todas as melhorias)
  -> Bootstrap + Shuffle + SelectKBest + Fallback NaN + Prompt Recalibrado + LOOCV
  -> Hipoteses testadas: H1-H6, H8

Experimento 8: Mais dados reais (se disponivel)
  -> 10+ datasets por mecanismo
  -> Hipotese testada: H11
```

### Criterio de Sucesso Global

O LLM sera considerado eficaz em dados reais quando:

1. **Accuracy media (LLM) > Accuracy media (baseline)** para pelo menos 5 dos 7 modelos
2. **Variancia CV < 15%** (atualmente 28-45%)
3. **Features LLM com importancia total > 15%** e variancia individual > 0.05
4. **Resultado reprodutivel** em 3+ execucoes com seeds diferentes

---

## 7. Analise Teorica: Por que o LLM Funciona no Sintetico

Para entender como fazer o LLM funcionar no real, e crucial entender por que ele funciona no sintetico:

1. **Dados sinteticos tem distribuicao uniforme** -- os limiares do prompt estao calibrados para isso
2. **3000 amostras** -- razao de 62:1 (amostras/features) permite ao modelo aprender quais features LLM sao informativas
3. **Mecanismos puros** -- cada arquivo sintetico implementa exatamente um mecanismo via funcao matematica (sigmoid), sem ambiguidade
4. **Features LLM tem alta variancia** -- std de 0.2-0.3 em todas as features, fornecendo sinal discriminativo real
5. **LLM features contribuem 23.9% da importancia** -- sao genuinamente uteis, especialmente `llm_evidence_consistency` (3.9%) e `llm_mar_conf` (3.7%)

**Conclusao:** O LLM funciona como um "meta-analista" que sintetiza estatisticas de primeira ordem em features de segunda ordem. Quando as estatisticas de primeira ordem sao claras (sintetico), o LLM adiciona valor. Quando sao ruidosas ou fora do range esperado (real), o LLM adiciona ruido.

A chave para fazer funcionar em dados reais e: **(a)** ter amostras suficientes para o modelo filtrar ruido, **(b)** calibrar o LLM para distribuicoes nao-uniformes, e **(c)** usar feature selection para descartar features LLM quando nao sao informativas.

---

## 8. Proposta de Implementacao (v3 do Pipeline)

A proposta abaixo combina as hipoteses H1-H8 em mudancas concretas nos arquivos existentes, sem criar scripts novos (exceto o bootstrap que substitui o chunking). A estrategia e atacar os 3 eixos simultaneamente: **(1) mais dados, (2) melhor LLM, (3) treino adaptativo**.

### 8.1 Visao Geral das Mudancas

```
Arquivos modificados:
  Scripts/subdividir_dados_reais.py     -> Shuffle + Bootstrap (substituicao completa)
  Scripts/v2_improved/llm/extractor_v2.py -> Prompt recalibrado + Fallback NaN + Metricas relativas
  Scripts/v2_improved/extract_features.py -> Imputacao de NaN nas features LLM
  Scripts/v2_improved/train_model.py     -> Feature selection + LOOCV + Hiperparametros adaptativos
  Scripts/v2_improved/run_all.py         -> Chamar subdividir antes do pipeline real

Nenhum arquivo novo. Nenhuma dependencia nova (exceto imbalanced-learn opcional para SMOTE).
```

### 8.2 Mudanca 1: `subdividir_dados_reais.py` — Shuffle + Bootstrap

**Objetivo:** Transformar 6 arquivos em ~300 amostras com distribuicao de missing representativa.

**Estrategia:** Para cada arquivo original, gerar N amostras bootstrap. Cada amostra e criada por reamostragem com reposicao de `CHUNK_SIZE` linhas, embaralhando a ordem. Isso garante que cada chunk tem taxa de missing proxima da media original.

```python
"""
Gera amostras bootstrap dos dados reais processados.

Para cada arquivo original (ex: 736 linhas), gera N_BOOTSTRAP amostras
de CHUNK_SIZE linhas via reamostragem com reposicao. Shuffle automatico.
"""
import os
import pandas as pd
import shutil

CHUNK_SIZE = 100       # linhas por amostra
N_BOOTSTRAP = 50       # amostras por arquivo original
MIN_MISSING_RATE = 0.01  # descarta amostras com <1% missing

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSADO = os.path.join(BASE, "..", "Dataset", "real_data", "processado")
OUTPUT = os.path.join(BASE, "..", "Dataset", "real_data", "processado_chunks")
MECANISMOS = ["MCAR", "MAR", "MNAR"]


def gerar_bootstrap():
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)

    total = 0
    for mec in MECANISMOS:
        input_dir = os.path.join(PROCESSADO, mec)
        output_dir = os.path.join(OUTPUT, mec)
        os.makedirs(output_dir, exist_ok=True)

        arquivos = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])

        for arq in arquivos:
            df = pd.read_csv(os.path.join(input_dir, arq), sep="\t")
            base_name = arq.replace(".txt", "")
            gerados = 0

            for i in range(N_BOOTSTRAP):
                # Reamostragem com reposicao (bootstrap)
                amostra = df.sample(n=min(CHUNK_SIZE, len(df)),
                                    replace=True,
                                    random_state=42 + i)

                # Verifica taxa de missing minima
                missing_rate = amostra["X0"].isna().mean()
                if missing_rate < MIN_MISSING_RATE:
                    continue

                out_name = f"{base_name}_boot{i+1:03d}.txt"
                amostra.to_csv(os.path.join(output_dir, out_name),
                               sep="\t", index=False)
                gerados += 1
                total += 1

            print(f"  {mec}/{arq}: {len(df)} linhas -> {gerados} bootstraps")

    print(f"\nTotal: {total} arquivos em {OUTPUT}")
    for mec in MECANISMOS:
        d = os.path.join(OUTPUT, mec)
        n = len([f for f in os.listdir(d) if f.endswith(".txt")])
        print(f"   {mec}: {n} arquivos")


if __name__ == "__main__":
    gerar_bootstrap()
```

**Resultado esperado:**
- ~300 amostras (50 bootstraps * 6 arquivos, menos as descartadas por missing < 1%)
- Cada amostra com taxa de missing proxima da media do arquivo original
- Razao amostras/features: ~300/18 = 16.7:1 (vs 1.8:1 atual)
- Classes balanceadas: ~100 por mecanismo

### 8.3 Mudanca 2: `extractor_v2.py` — 3 correcoes

#### 2a. Metricas relativas em vez de absolutas

**Problema atual (linha 173):**
```python
X0_mean_dev = round(0.5 - float(np.mean(X0_obs)), 4)  # Assume media 0.5
```
Com oceanbuoys (media 0.81), isso gera desvio de -0.31, saturando `mnar_combined_evidence` em 1.0.

**Correcao:** Usar desvio relativo ao intervalo dos dados, nao a 0.5 fixo.

```python
# ANTES (linha 170-175):
X0_mean_dev = round(0.5 - float(np.mean(X0_obs)), 4)

# DEPOIS:
# Desvio relativo: quanto a media dos observados desvia da media total esperada
# Para dados normalizados [0,1], a media esperada e a media de TODOS os X0 (obs + estimados)
# Usamos a mediana como estimador robusto do centro
X0_range = float(np.max(X0_obs) - np.min(X0_obs)) if len(X0_obs) > 1 else 1.0
X0_center = float(np.median(X0_obs))
X0_mean_dev = round((X0_center - float(np.mean(X0_obs))) / max(X0_range, 0.01), 4)
```

**Correcao da formula de evidencia MNAR (linha 185):**

```python
# ANTES:
mnar_evidence = abs(X0_mean_dev) * 50 + abs(X0_skew) * 5  # Multiplicador 50 satura

# DEPOIS:
mnar_evidence = abs(X0_mean_dev) * 5 + abs(X0_skew) * 2  # Escala reduzida
```

#### 2b. Fallback retorna NaN em vez de defaults

**Problema atual (linha 350-352):**
```python
return LLMAnalysisV2().to_feature_dict()  # Valores fixos que enviesam
```

**Correcao:**
```python
# ANTES:
return LLMAnalysisV2().to_feature_dict()

# DEPOIS:
print(f"  LLM v2 falhou apos {max_retries} tentativas: {e}")
return {k: float('nan') for k in LLMAnalysisV2().to_feature_dict()}
```

Mesma correcao na linha 354 (fallback final).

#### 2c. Prompt sem limiares fixos

**Problema atual (linhas 283-286):**
```
1. Se mar_combined_evidence < 0.1 E mnar_combined_evidence < 0.1 → MCAR
2. Se mar_combined_evidence > 0.3 → MAR
...
```

**Correcao:** Substituir por instrucoes relativas.

```python
# ANTES (linhas 281-286):
## INSTRUCOES DE RACIOCINIO
1. Se mar_combined_evidence < 0.1 E mnar_combined_evidence < 0.1 → provavelmente MCAR
2. Se mar_combined_evidence > 0.3 → provavelmente MAR
3. Se mnar_combined_evidence > 0.2 E mnar_internal_consistency = 1 → provavelmente MNAR
4. Se ambas evidencias sao altas → caso ambiguo, reduza confianca

# DEPOIS:
## INSTRUCOES DE RACIOCINIO
1. Compare a MAGNITUDE RELATIVA de mar_combined_evidence vs mnar_combined_evidence
2. Se ambas sao baixas relativas uma a outra → provavelmente MCAR
3. Se mar_combined_evidence >> mnar_combined_evidence → provavelmente MAR
4. Se mnar_combined_evidence >> mar_combined_evidence E mnar_internal_consistency = 1 → provavelmente MNAR
5. Se ambas sao altas e proximas → caso ambiguo, reduza confianca e pattern_clarity
6. IMPORTANTE: A distribuicao de X0 pode NAO ser uniforme [0,1]. Avalie os desvios
   relativamente ao X0_obs_std, nao em termos absolutos.
```

Tambem adicionar ao prompt o `X0_obs_std` e `X0_obs_range` para dar contexto:

```python
# Adicionar ao stats_dict (apos linha 153):
stats_dict["X0_obs_range"] = round(float(np.max(X0_obs) - np.min(X0_obs)), 4)
```

### 8.4 Mudanca 3: `extract_features.py` — Imputacao de NaN

**Objetivo:** Apos extrair features de todos os arquivos, imputar os NaN das features LLM com a mediana das amostras validas (em vez de manter os 0.0 que o `fillna(0)` atual produz).

**Correcao (linhas 233-234):**

```python
# ANTES:
X = pd.DataFrame(X_all).replace([np.inf, -np.inf], 0).fillna(0)

# DEPOIS:
X = pd.DataFrame(X_all).replace([np.inf, -np.inf], np.nan)

# Imputar features LLM com mediana (preserva distribuicao)
llm_cols = [c for c in X.columns if c.startswith("llm_")]
stat_cols = [c for c in X.columns if not c.startswith("llm_")]

# Features estatisticas: NaN -> 0 (ausencia de sinal)
X[stat_cols] = X[stat_cols].fillna(0)

# Features LLM: NaN -> mediana das amostras com resposta valida
if llm_cols:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X[llm_cols] = pd.DataFrame(
        imputer.fit_transform(X[llm_cols]),
        columns=llm_cols, index=X.index
    )
```

### 8.5 Mudanca 4: `train_model.py` — Treino Adaptativo

#### 4a. Feature Selection por tamanho do dataset

**Adicionar apos carregar dados (apos linha 78):**

```python
# Feature selection adaptativa: k = max(5, n_amostras / 10)
# Garante que nao usamos mais features do que o dataset suporta
n_samples = len(X)
max_features = max(5, n_samples // 10)

if X.shape[1] > max_features:
    print(f"\n Feature selection: {X.shape[1]} -> {max_features} features (n={n_samples})")
    selector = SelectKBest(f_classif, k=max_features)
    X_selected = pd.DataFrame(
        selector.fit_transform(X, y),
        columns=X.columns[selector.get_support()]
    )
    removed = set(X.columns) - set(X_selected.columns)
    print(f"   Removidas: {removed}")
    X = X_selected
```

Com n=300 (bootstrap): max_features = 30 -> usa todas as 18. OK.
Com n=43 (chunking atual): max_features = 5 -> seleciona top 5.

#### 4b. Hiperparametros adaptativos

**Substituir definicao de modelos (linhas 83-106) por funcao adaptativa:**

```python
def get_modelos(n_samples: int) -> dict:
    """Retorna modelos com hiperparametros adaptados ao tamanho do dataset."""
    if n_samples < 100:
        # Dataset pequeno: modelos simples, menos overfitting
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, C=0.5, random_state=42))
            ]),
            "SVM_RBF": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1, random_state=42))
            ]),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=3))
            ]),
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(32, 16),
                                      max_iter=2000, random_state=42))
            ]),
            "NaiveBayes": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB())
            ]),
        }
    else:
        # Dataset grande: modelos originais
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=400, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=300, random_state=42),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=42))
            ]),
            "SVM_RBF": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=3, random_state=42))
            ]),
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5))
            ]),
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                      max_iter=2000, random_state=42))
            ]),
            "NaiveBayes": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB())
            ]),
        }

modelos = get_modelos(len(X))
```

**Justificativa dos hiperparametros para n<100:**
- RF/GB: Menos arvores (100/50) e profundidade limitada (5/3) reduzem overfitting
- SVM: C=1 (vs C=3) para margem mais larga com menos dados
- KNN: k=3 (vs k=5) pois com ~30 amostras de treino, 5 vizinhos pode ser excessivo
- MLP: 2 camadas (32,16) vs 3 camadas (128,64,32) -- rede menor para dataset menor

#### 4c. Cross-validation adaptativa

**Substituir CV section (linhas 169-177):**

```python
from sklearn.model_selection import (
    cross_val_score, RepeatedStratifiedKFold, LeaveOneOut
)

# CV adaptativa: LOOCV para n<50, RepeatedStratifiedKFold para n>=50
n_samples = len(X)
if n_samples < 50:
    cv_strategy = LeaveOneOut()
    cv_name = "Leave-One-Out"
else:
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    cv_name = "Repeated Stratified 5-Fold (3x)"

print(f"\n Estrategia CV: {cv_name} (n={n_samples})")

cv_results = {}
for nome, modelo in tqdm(modelos.items(), desc="Cross-validation"):
    scores = cross_val_score(modelo, X, y, cv=cv_strategy, scoring="accuracy")
    cv_results[nome] = {"mean": scores.mean(), "std": scores.std()}
    relatorio_lines.append(f"{nome}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### 8.6 Mudanca 5: `run_all.py` — Integrar bootstrap

**Adicionar chamada ao bootstrap antes de processar dados reais (apos linha 53, dentro do loop de data_types):**

```python
for data_type in DATA_TYPES:
    # Se dados reais, regenerar bootstrap
    if data_type == "real":
        print(f"\n Gerando amostras bootstrap dos dados reais...")
        bootstrap_script = os.path.join(
            SCRIPT_DIR, "..", "subdividir_dados_reais.py")
        subprocess.run([sys.executable, bootstrap_script], cwd=SCRIPT_DIR)
    # ... resto do loop
```

### 8.7 Resumo das Mudancas por Arquivo

| Arquivo | Linhas afetadas | Natureza da mudanca |
|---------|:-:|---|
| `subdividir_dados_reais.py` | Reescrita | Chunking sequencial -> Bootstrap com shuffle |
| `llm/extractor_v2.py` L170-175 | Edit | `X0_mean_dev` relativo ao range, nao a 0.5 |
| `llm/extractor_v2.py` L185 | Edit | Multiplicador de 50 -> 5 em `mnar_evidence` |
| `llm/extractor_v2.py` L283-286 | Edit | Limiares fixos -> instrucoes relativas no prompt |
| `llm/extractor_v2.py` L350-354 | Edit | Fallback: defaults -> NaN |
| `extract_features.py` L233-234 | Edit | `fillna(0)` -> imputacao mediana para features LLM |
| `train_model.py` L83-106 | Edit | Modelos fixos -> `get_modelos(n)` adaptativo |
| `train_model.py` L108+ | Insert | Feature selection com `SelectKBest(k=n/10)` |
| `train_model.py` L169-177 | Edit | CV fixa -> LOOCV (n<50) / RepeatedStratKFold |
| `run_all.py` L54 | Insert | Chamar bootstrap antes de pipeline real |

### 8.8 Fluxo Esperado apos Implementacao

```
run_all.py --data real
  |
  +--> subdividir_dados_reais.py  (gera ~300 bootstraps)
  |
  +--> extract_features.py --model none --data real
  |      |-> 300 amostras * 10 features = X(300, 10)
  |      |-> fillna(0) para features estatisticas
  |
  +--> train_model.py --model none --data real
  |      |-> n=300 >= 100 -> hiperparametros grandes
  |      |-> max_features = 30 >= 10 -> sem feature selection
  |      |-> RepeatedStratifiedKFold (n>=50)
  |
  +--> extract_features.py --model gemini-3-flash-preview --data real
  |      |-> 300 amostras * 18 features = X(300, 18)
  |      |-> Prompt recalibrado (metricas relativas, sem limiares fixos)
  |      |-> ~5% NaN (fallback) -> imputacao mediana
  |
  +--> train_model.py --model gemini-3-flash-preview --data real
  |      |-> n=300 >= 100 -> hiperparametros grandes
  |      |-> max_features = 30 >= 18 -> sem feature selection
  |      |-> RepeatedStratifiedKFold
  |
  +--> compare_results.py --data real
```

### 8.9 Impacto Esperado por Mudanca

| Mudanca | Problema que resolve | Impacto estimado |
|---------|---------------------|:---:|
| Bootstrap (n=300) | Dimensionalidade, variancia CV | +++++ |
| Prompt relativo | LLM classifica MCAR real como MNAR | ++++ |
| Fallback NaN | 28% das amostras com vies | +++ |
| Imputacao mediana | Features LLM zeradas pelo fillna(0) | ++ |
| Feature selection | Curse of dimensionality | ++ |
| Hiperparametros adaptativos | Overfitting em dataset pequeno | ++ |
| CV adaptativa | Estimativa nao-confiavel | + (metrica, nao accuracy) |

### 8.10 Riscos e Mitigacoes

| Risco | Probabilidade | Mitigacao |
|-------|:---:|---|
| Bootstrap cria amostras correlacionadas (data leakage no split) | Alta | Usar GroupKFold com grupo = arquivo original para que chunks do mesmo arquivo nao aparecam em treino e teste |
| LLM cache invalido (mesmas stats -> mesmo hash) | Media | Com bootstrap, cada amostra tem stats diferentes -> cache e util sem problemas |
| Feature selection descarta features LLM uteis | Baixa | Com n=300, k=30 > 18 -> nao descarta nada |
| Prompt recalibrado piora nos sinteticos | Media | Manter o prompt original para `--data sintetico`, usar novo para `--data real` |

**Risco critico -- Data Leakage no Bootstrap:**

Bootstraps do mesmo arquivo original compartilham linhas (sampling com reposicao). Se um bootstrap do arquivo A esta no treino e outro do mesmo arquivo A esta no teste, ha data leakage.

**Solucao:** Adicionar `GroupKFold` ou `GroupShuffleSplit` usando o nome do arquivo original como grupo:

```python
# Em train_model.py, salvar grupo ao lado das features:
# Extrair nome do arquivo original do nome do chunk
# Ex: "MCAR_oceanbuoys_airtemp_boot023.txt" -> "MCAR_oceanbuoys_airtemp"
groups = [nome_chunk.rsplit("_boot", 1)[0] for nome_chunk in file_names]
```

```python
# No split:
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

Isso garante que todos os bootstraps de um mesmo arquivo original ficam no mesmo lado do split (treino OU teste, nunca ambos).

---

## 9. Referencias

- [Curse of Dimensionality - DataCamp](https://www.datacamp.com/blog/curse-of-dimensionality-machine-learning)
- [Feature Extraction for Small Datasets - Nature](https://www.nature.com/articles/s41598-025-07725-9)
- [Classification for HDLSS Data - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320322003090)
- [Cross-Validation Unreliable in Small Samples - ResearchGate](https://www.researchgate.net/publication/222650078_Cross-validation_and_bootstrapping_are_unreliable_in_small_sample_classification)
- [Stop Using Little's MCAR Test - PubMed](https://pubmed.ncbi.nlm.nih.gov/39251529/)
- [MissMecha Python Package - arXiv](https://arxiv.org/html/2508.04740)
- [LLM Calibration - LearnPrompting](https://learnprompting.org/docs/reliability/calibration)
- [SMOTE for Tabular Data - TowardsDataScience](https://towardsdatascience.com/smote-synthetic-data-augmentation-for-tabular-data-1ce28090debc/)
- [LLM-Based Tabular Augmentation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417424027192)
- [Domain-Aware Tabular Augmentation with LLMs - OpenReview](https://openreview.net/forum?id=Iw6lOEIBoX)
- [Hybrid Feature Selection for High-Dimensional Data - Nature](https://www.nature.com/articles/s41598-025-08699-4)
- [Review: Handling Missing Data Mechanisms - arXiv](https://arxiv.org/abs/2404.04905)
- [Missing Data Concepts MCAR MAR MNAR - Van Buuren](https://stefvanbuuren.name/fimd/sec-MCAR.html)
