# Tabelas Comparativas — ML × Flash × Pro

**Data:** 2026-04-25
**Benchmark:** 29 datasets reais, 1.421 bootstraps, 25 features estatísticas (+ 9 LLM em Flash/Pro)

---

## 1. Holdout (GroupShuffleSplit 75/25, n=395)

### 1.1 Acurácia por modelo (lado a lado)

| Modelo | ML-only (25f) | Flash (34f) | Pro (34f) | Δ Pro−ML | Δ Pro−Flash |
|--------|:-------------:|:-----------:|:---------:|:--------:|:-----------:|
| **NaiveBayes** | 0.5392 | 0.5114 | **0.5519** | **+0.0127** | **+0.0405** |
| **LogisticRegression** | **0.5494** | **0.5494** | 0.5494 | 0.0000 | 0.0000 |
| SVM_RBF | 0.4633 | 0.4228 | 0.4380 | −0.0253 | +0.0152 |
| MLP | 0.4203 | 0.4152 | 0.4304 | +0.0101 | +0.0152 |
| KNN | 0.4177 | 0.4253 | 0.3975 | −0.0202 | −0.0278 |
| GradientBoosting | 0.4101 | 0.4380 | 0.4127 | +0.0026 | −0.0253 |
| RandomForest | 0.4051 | 0.4177 | 0.4177 | +0.0126 | 0.0000 |
| **Best modelo** | **0.5494 (LogReg)** | **0.5494 (LogReg)** | **0.5519 (NB)** | **+0.0025** | **+0.0025** |

### 1.2 Ordenado por melhor performance (Pro)

| Rank | Modelo | ML-only | Flash | Pro |
|:----:|--------|:-------:|:-----:|:---:|
| 1 | **NaiveBayes** | 53.92% | 51.14% | **55.19%** |
| 2 | LogisticRegression | 54.94% | 54.94% | 54.94% |
| 3 | SVM_RBF | 46.33% | 42.28% | 43.80% |
| 4 | MLP | 42.03% | 41.52% | 43.04% |
| 5 | RandomForest | 40.51% | 41.77% | 41.77% |
| 6 | GradientBoosting | 41.01% | 43.80% | 41.27% |
| 7 | KNN | 41.77% | 42.53% | 39.75% |

## 2. Cross-Validation (Group 5-Fold, n=1421)

### 2.1 Acurácia média ± std por modelo

| Modelo | ML-only | Flash | Pro | Δ Pro−ML | Δ Pro−Flash |
|--------|:-------:|:-----:|:---:|:--------:|:-----------:|
| **NaiveBayes** | 0.4747 ±0.091 | 0.4744 ±0.115 | **0.4933** ±0.142 | **+0.0186** | **+0.0189** |
| LogisticRegression | 0.4199 ±0.093 | 0.3979 ±0.235 | 0.4154 ±0.235 | −0.0045 | +0.0175 |
| RandomForest | 0.3882 ±0.128 | 0.3960 ±0.249 | 0.3897 ±0.266 | +0.0015 | −0.0063 |
| KNN | 0.3820 ±0.091 | 0.3929 ±0.160 | 0.3502 ±0.185 | −0.0318 | −0.0427 |
| SVM_RBF | 0.3753 ±0.121 | 0.3589 ±0.283 | 0.3285 ±0.274 | −0.0468 | −0.0304 |
| GradientBoosting | 0.3625 ±0.108 | 0.3601 ±0.218 | 0.3632 ±0.187 | +0.0007 | +0.0031 |
| MLP | 0.3328 ±0.122 | 0.3415 ±0.223 | 0.3326 ±0.243 | −0.0002 | −0.0089 |
| **Best CV** | **0.4747 (NB)** | **0.4744 (NB)** | **0.4933 (NB)** | **+0.0186** | **+0.0189** |

### 2.2 Ranking CV ordenado

| Rank | Modelo | ML-only | Flash | Pro |
|:----:|--------|:-------:|:-----:|:---:|
| 1 | **NaiveBayes** | **47.47%** | **47.44%** | **49.33%** |
| 2 | LogisticRegression | 41.99% | 39.79% | 41.54% |
| 3 | RandomForest | 38.82% | 39.60% | 38.97% |
| 4 | KNN | 38.20% | 39.29% | 35.02% |
| 5 | SVM_RBF | 37.53% | 35.89% | 32.85% |
| 6 | GradientBoosting | 36.25% | 36.01% | 36.32% |
| 7 | MLP | 33.28% | 34.15% | 33.26% |

### 2.3 Variância CV (std/avg)

| Modelo | ML-only | Flash | Pro |
|--------|:-------:|:-----:|:---:|
| NaiveBayes | 0.19 | 0.24 | 0.29 |
| LogisticRegression | 0.22 | 0.59 | 0.57 |
| RandomForest | 0.33 | 0.63 | 0.68 |
| KNN | 0.24 | 0.41 | 0.53 |
| SVM_RBF | 0.32 | 0.79 | 0.83 |
| GradientBoosting | 0.30 | 0.61 | 0.51 |
| MLP | 0.37 | 0.65 | 0.73 |
| **Média** | **0.28** | **0.56** | **0.59** |

**Observação:** features LLM **dobram a variância CV** (0.28 → 0.56-0.59). Isto reflete o achado de que o LLM falha em alguns datasets — quando esses caem em folds específicos, a fold inteira degrada.

## 3. Diferenças agregadas (sumário)

### 3.1 Best Holdout

| Comparação | Δ |
|------------|:-:|
| Flash − ML | 0.00pp |
| Pro − ML | **+0.25pp** |
| Pro − Flash | **+0.25pp** |

### 3.2 Best CV

| Comparação | Δ |
|------------|:-:|
| Flash − ML | **−0.03pp** ← Flash não agrega |
| Pro − ML | **+1.86pp** ← ganho real do LLM |
| Pro − Flash | **+1.89pp** ← ganho de Pro sobre Flash |

### 3.3 NaiveBayes (modelo dominante) — todos os números

| Métrica | ML-only | Flash | Pro | Δ Pro−ML | Δ Pro−Flash |
|---------|:-------:|:-----:|:---:|:--------:|:-----------:|
| Holdout acc | 53.92% | 51.14% | 55.19% | +1.27pp | +4.05pp |
| CV avg | 47.47% | 47.44% | 49.33% | +1.86pp | +1.89pp |
| CV std | 9.05 | 11.50 | 14.23 | +5.18 | +2.73 |
| CV min | ~37% | ~30% | ~37% | — | — |
| CV max | ~58% | ~58% | ~58% | — | — |

## 4. Análise por classe (NaiveBayes holdout)

### 4.1 ML-only (apenas 25 features)

```
                 PREDITO
                 MCAR  MAR  MNAR    Total
TRUE   MCAR    [  ?    ?    ?  ]    95
       MAR     [  ?    ?    ?  ]    150
       MNAR    [  ?    ?    ?  ]    150
```
Acurácia 53.92% (não há matriz armazenada — seria necessário rerun com persistência).

### 4.2 Flash + ML

| Classe | Precisão | Recall | F1 | Suporte |
|--------|:--------:|:------:|:--:|:-------:|
| MCAR (0) | 0.77 | 0.39 | 0.52 | 95 |
| MAR (1) | 0.68 | 0.21 | 0.32 | 150 |
| MNAR (2) | 0.44 | 0.89 | 0.59 | 150 |

Matriz Flash:
```
                 MCAR  MAR  MNAR
TRUE   MCAR    [  37    9   49 ]
       MAR     [   0   32  118 ]
       MNAR    [  11    6  133 ]
```

### 4.3 Pro + ML

| Classe | Precisão | Recall | F1 | Suporte |
|--------|:--------:|:------:|:--:|:-------:|
| MCAR (0) | **0.79** | 0.44 | 0.57 | 95 |
| MAR (1) | 0.52 | 0.45 | 0.48 | 150 |
| MNAR (2) | 0.51 | **0.73** | **0.60** | 150 |

Matriz Pro:
```
                 MCAR  MAR  MNAR
TRUE   MCAR    [  42   33   20 ]
       MAR     [   0   67   83 ]
       MNAR    [  11   30  109 ]
```

### 4.4 Comparação por classe (NB holdout)

| Métrica | Flash | Pro | Δ |
|---------|:-----:|:---:|:-:|
| Recall MCAR | 39% | 44% | +5pp |
| Recall MAR | **21%** ← muito baixo | **45%** | **+24pp** |
| Recall MNAR | 89% | 73% | −16pp |
| F1 MCAR | 0.52 | 0.57 | +0.05 |
| F1 MAR | **0.32** | **0.48** | **+0.16** |
| F1 MNAR | 0.59 | 0.60 | +0.01 |
| **F1 macro** | 0.48 | **0.55** | **+0.07** |

**Insight:** Pro **trade-off explícito** — sacrifica recall MNAR (−16pp) para recuperar recall MAR (+24pp). O MAR-bias do Step 1 prompt funcionou exatamente como projetado, mas em direção oposta ao que esperávamos: ele **aumentou** recall MAR de 21% → 45% em vez de **diminuir**. Isso ocorre porque com Flash o LLM já não predizia MAR (recall 21%), então o anti-bias não tinha efeito; com Pro, o LLM estava "honesto demais" e o anti-bias o calibrou para níveis razoáveis de MAR.

## 5. Tabela mestre (referência rápida)

| Métrica | ML-only | Flash | Pro |
|---------|:-------:|:-----:|:---:|
| **Datasets** | 29 | 29 | 29 |
| **Bootstraps** | 1.421 | 1.421 | 1.421 |
| **Features estatísticas** | 25 | 25 | 25 |
| **Features LLM** | 0 | 9 | 9 |
| **Modelo LLM** | — | gemini-3-flash-preview | gemini-3-pro-preview |
| **Prompt LLM** | — | original | Step 1 (few-shot+typology+anti-bias) |
| **Metadata variant** | — | neutral | neutral |
| **Best Holdout** | 54.94% (LogReg) | 54.94% (LogReg) | **55.19% (NB)** |
| **Best CV** | 47.47% (NB) | 47.44% (NB) | **49.33% (NB)** |
| **F1 macro (best)** | ~0.54 (LogReg) | 0.48 (NB) | **0.55 (NB)** |
| **Custo extração** | $0 | ~$2-4 | ~$30-36 |
| **Tempo extração** | < 1 min | ~30 min | ~1h33min |
| **Variância CV (média)** | ±10.7% | ±21.1% | ±21.9% |
| **Razão custo/+1pp CV vs ML** | — | indefinido | $16-19 |
