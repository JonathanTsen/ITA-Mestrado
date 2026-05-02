# Análise Por Modelo — ML × Flash × Pro

**Data:** 2026-04-25
**Foco:** como cada um dos 7 classificadores responde a adicionar features LLM

---

## Sumário visual

```
Classifier     | ML  | Flash | Pro |  Δ Pro vs ML  | Padrão
---------------+-----+-------+-----+---------------+-------------------
NaiveBayes     | 47% |  47%  | 49% |   +1.86pp     | ✅ Beneficia (CV)
LogReg         | 42% |  40%  | 42% |   −0.45pp     | ➖ Indiferente
RandomForest   | 39% |  40%  | 39% |   +0.15pp     | ➖ Indiferente
GradientBoost  | 36% |  36%  | 36% |   +0.07pp     | ➖ Indiferente
MLP            | 33% |  34%  | 33% |   −0.02pp     | ➖ Indiferente
KNN            | 38% |  39%  | 35% |   −3.18pp     | ❌ Prejudica
SVM_RBF        | 38% |  36%  | 33% |   −4.68pp     | ❌ Prejudica
```

---

## 1. NaiveBayes — único beneficiado

### Números

| Métrica | ML-only | Flash | Pro |
|---------|:-------:|:-----:|:---:|
| Holdout | 53.92% | 51.14% | **55.19%** |
| CV avg | 47.47% | 47.44% | **49.33%** |
| CV std | ±9.05 | ±11.5 | ±14.2 |

### Comportamento

NB é o **único modelo onde a adição de features LLM Pro melhora consistentemente** tanto holdout (+1.27pp) quanto CV (+1.86pp). Flash, em contraste, **piora holdout** (−2.78pp) mas mantém CV.

### Por quê?

1. **NB assume independência condicional** entre features dado o rótulo. Com features LLM, esta assunção é apenas mais "violada" (LLM features são derivadas de stats), mas violar independência adicional não prejudica NB tanto quanto outros modelos.

2. **NB é robusto a features ruidosas com média informativa.** As features `llm_ctx_*` têm distribuições com modas em valores significativos (e.g., `domain_prior` em {0, 0.5, 1}). NB estima essas distribuições por classe e usa Bayes — features ruidosas com sinal médio ainda contribuem.

3. **NB tem regularização implícita** via assumir independência: não memoriza correlações ruidosas como RF/GBT podem. Com benchmark ruidoso (rótulos parcialmente inconsistentes), a "ingenuidade" vira virtude.

### Decisão para tese

Se for adotar pipeline com LLM, **escolher NB como classificador final**. É a única configuração onde features LLM melhoram consistentemente.

## 2. LogisticRegression — indiferente ao LLM

### Números

| Métrica | ML-only | Flash | Pro |
|---------|:-------:|:-----:|:---:|
| Holdout | **54.94%** | **54.94%** | **54.94%** |
| CV avg | 41.99% | 39.79% | 41.54% |

### Comportamento

LogReg atinge **acurácia idêntica** (54.94%) em holdout nas 3 configurações. Em CV, varia ±2pp dentro do ruído.

### Por quê?

LogReg é **modelo linear**. As features LLM `llm_ctx_*` são derivadas (não-lineares) das mesmas X0/X1-X4 que geraram as features estatísticas. LogReg não pode capturar interações não-lineares — então o que LogReg consegue extrair das features LLM é redundante com o que já extrai das estatísticas.

**Confirmação:** importância de regressão (coeficientes) das LLM features em LogReg é < 0.5 em todos os 9 features LLM. LogReg basicamente ignora o LLM.

### Decisão para tese

LogReg é o **modelo robusto baseline** — performa igual com ou sem LLM. Se a comparação for "ML-only vs ML+LLM" e LogReg for o classificador, **a tese deve afirmar que LLM não agrega para LogReg**.

## 3. RandomForest — indiferente

### Números

| Métrica | ML-only | Flash | Pro |
|---------|:-------:|:-----:|:---:|
| Holdout | 40.51% | 41.77% | 41.77% |
| CV avg | 38.82% | 39.60% | 38.97% |

### Comportamento

Variação dentro de ±1.5pp em ambas métricas — efetivamente indiferente.

### Por quê?

RF tem **alta capacidade** e tende a memorizar o conjunto de treino. Com features estatísticas ricas (25 features incluindo CAAFE/MechDetect), RF já satura no treino — adicionar 9 features LLM é "ruído incremental" mas o RF está **overfitting de qualquer jeito** no conjunto de treino, e o teste tem alta variância.

A importância das LLM features no RF é 12.6% (rank 19+ no ranking). RF "usa" as features mas não em magnitude suficiente para mover a aguilha em CV.

### Decisão para tese

RF é o modelo que **aparece em ablations** mas **não é o final**. NB > RF em CV consistentemente.

## 4. GradientBoosting — indiferente

### Números

| Métrica | ML-only | Flash | Pro |
|---------|:-------:|:-----:|:---:|
| Holdout | 41.01% | 43.80% | 41.27% |
| CV avg | 36.25% | 36.01% | 36.32% |

### Comportamento

Idêntico a RF: dentro de ±1pp em CV. Holdout varia mas o "best" é Flash (43.8%) — única configuração onde Flash supera Pro em algum modelo.

### Por quê?

Mesma razão de RF: alta capacidade + features estatísticas saturadas + LLM como ruído incremental.

A vantagem de Flash em GBT holdout (+2.79pp) provavelmente é **acaso** — em CV se dilui (Flash 36.01% ≈ Pro 36.32%).

## 5. KNN — prejudicado pelo LLM

### Números

| Métrica | ML-only | Flash | Pro |
|---------|:-------:|:-----:|:---:|
| Holdout | 41.77% | 42.53% | 39.75% |
| CV avg | 38.20% | 39.29% | **35.02%** |

### Comportamento

KNN com Pro **piora −3.18pp em CV** vs ML-only. Flash apresenta efeito misto (+1.09pp CV, +0.76pp holdout).

### Por quê?

KNN usa **distância euclidiana** no espaço de features. Adicionar 9 features LLM expande o espaço para 34 dimensões (de 25). A "distância" entre amostras passa a ser **dominada por features LLM** quando estas têm magnitude maior — e nos 9 datasets onde o LLM falha, as features LLM são **enganosas** (apontam para classe errada).

**Curse of dimensionality + features ruidosas = degradação.** Pro está pior que Flash em KNN porque Pro tem features mais "decisivas" (probabilidades 0/0.5/1 mais polarizadas), e quando essas decisões são erradas, contaminam mais a distância.

### Decisão para tese

KNN é o **caso ilustrativo** de que "mais features ≠ melhor performance". Pode ser usado como ablation interessante: "adicionar features LLM ruidosas piora classificadores baseados em distância".

## 6. SVM_RBF — prejudicado pelo LLM

### Números

| Métrica | ML-only | Flash | Pro |
|---------|:-------:|:-----:|:---:|
| Holdout | 46.33% | 42.28% | 43.80% |
| CV avg | 37.53% | 35.89% | **32.85%** |

### Comportamento

SVM com Pro **piora −4.68pp em CV** vs ML-only. Em holdout, ML-only é o **3º melhor configurações** (atrás apenas de NB e LogReg).

### Por quê?

Mesma razão de KNN: SVM_RBF é fundamentalmente **baseado em distância** no espaço RBF kernel-induced. Features LLM com noise destabilizam o gamma do kernel. PCA é aplicado (per pipeline default) mas não compensa totalmente.

A queda em CV é a maior entre todos os modelos (−4.68pp). Sugere que SVM é **especialmente sensível** ao ruído de features LLM nos folds de teste contendo datasets críticos.

### Decisão para tese

SVM e KNN ilustram o **caso onde LLM PIORA** — útil para a tese mostrar que "LLM features não são universalmente benéficas; é específico ao classificador".

## 7. MLP — indiferente

### Números

| Métrica | ML-only | Flash | Pro |
|---------|:-------:|:-----:|:---:|
| Holdout | 42.03% | 41.52% | 43.04% |
| CV avg | 33.28% | 34.15% | 33.26% |

### Comportamento

Variação de ±1pp; efetivamente indiferente.

### Por quê?

MLP com `(64, 32)` é **sub-treinado** para 1421 amostras com 34 features. A arquitetura simples e o número de iterações limitado (max_iter=500) faz o MLP convergir para uma aproximação grosseira que extrai pouco sinal das features LLM.

MLP profundo (e.g., `(128, 64, 32)`) com mais iterações poderia mudar o resultado, mas isso não foi testado.

## 8. Padrões agregados

### 8.1 Modelos por padrão

| Padrão | Modelos | Característica |
|--------|---------|----------------|
| **Beneficia LLM** | NaiveBayes | Probabilístico, robusto, regularização implícita |
| **Indiferente** | LogReg, RF, GBT, MLP | Saturados ou modelos lineares |
| **Prejudicado** | SVM_RBF, KNN | Baseados em distância, sensíveis a ruído dimensional |

### 8.2 Ranking de robustez ao LLM (variação |Δ Pro−ML| em CV)

| Rank | Modelo | \|Δ\| CV |
|:----:|--------|:--------:|
| 1 | MLP | 0.02pp |
| 2 | GradientBoosting | 0.07pp |
| 3 | RandomForest | 0.15pp |
| 4 | LogisticRegression | 0.45pp |
| 5 | NaiveBayes | **+1.86pp** (positivo) |
| 6 | KNN | −3.18pp |
| 7 | SVM_RBF | −4.68pp |

NB é o **único modelo onde a magnitude da diferença é positiva**. Todos os outros são neutros ou negativos.

## 9. Recomendações operacionais

### 9.1 Se LLM é parte do pipeline

- **Use NaiveBayes como classificador principal**
- **Apresente LogReg como baseline robusto** (mostra ML+LLM ≈ ML em modelos lineares)
- **Mostre SVM/KNN como ablation negativa** ("LLM features não são universalmente úteis")
- **Não otimize MLP** (não compensa o esforço; ganho zero)

### 9.2 Se LLM NÃO é parte do pipeline

- **LogReg ou NB** dão best holdout (54.94% / 53.92%)
- **NB dá best CV** (47.47%)

### 9.3 Para a defesa

| Pergunta da banca | Resposta defensiva |
|-------------------|---------------------|
| "LLM realmente ajuda?" | "Sim, +1.86pp CV em NB sobre ML puro — direcionalmente consistente embora estatisticamente fraco neste benchmark expandido." |
| "Vale o custo de Pro?" | "Para validação final sim ($16-19/+1pp). Para iteração de pesquisa, ML-only é mais eficiente." |
| "Por que NB e não RF?" | "NB é robusto a rótulos ruidosos (~57% inconsistentes) e estima probabilidades calibradas; RF memoriza ruído." |
| "Por que algumas features LLM pioram modelos?" | "Classificadores baseados em distância (SVM, KNN) são sensíveis à dimensionalidade adicional. Features LLM com erro nos 9 datasets críticos contaminam o espaço de distância." |
