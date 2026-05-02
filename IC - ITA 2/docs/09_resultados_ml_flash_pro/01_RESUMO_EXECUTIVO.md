# Resumo Executivo — Comparação ML × Flash × Pro

**Data:** 2026-04-25

---

## Pergunta de pesquisa

> Qual é o ganho de performance **isolado** ao adicionar features LLM (gemini-3-flash-preview ou gemini-3-pro-preview com extrator `context_aware` e Step 1 prompt) sobre o pipeline de classificação puramente baseado em 25 features estatísticas, mantendo benchmark, splits, SMOTE, e hiperparâmetros fixados?

## Resposta encontrada

**Pro adiciona +1.86pp em CV** (NaiveBayes, Group 5-Fold) sobre ML puro. **Flash não adiciona nada** (−0.03pp, dentro do ruído). Conclusão: o LLM **funciona** mas o ganho é incremental, e apenas o modelo Pro é grande o suficiente para extrair sinal incremental sobre as 25 features estatísticas existentes neste benchmark expandido.

---

## Tabela final

| Configuração | Holdout best | CV best (NB) | Custo | Tempo |
|--------------|:------------:|:------------:|:-----:|:-----:|
| **ML-only (25 features)** | 54.94% (LogReg) | 47.47% ± 9.1 | $0 | < 1 min |
| **Flash + ML (34 features)** | 54.94% (LogReg) | 47.44% ± 11.5 | ~$2-4 | ~30 min |
| **Pro + ML (34 features)** | **55.19% (NB)** | **49.33% ± 14.2** | ~$30-36 | ~1h33min |

### Deltas

| Comparação | Holdout best | CV best | Custo |
|------------|:------------:|:-------:|:-----:|
| Flash − ML | 0.00pp | **−0.03pp** | +$2-4 |
| Pro − ML | +0.25pp | **+1.86pp** | +$30-36 |
| Pro − Flash | +0.25pp | **+1.89pp** | +10× |

---

## Veredicto em 3 frases

1. **Flash é dominado por ML-only** — mesmo CV, mesmo holdout, custo positivo. **Não há razão racional para usar Flash neste benchmark.**

2. **Pro entrega ganho real mas marginal** — +1.86pp CV é estatisticamente fraco (dentro de std ±9-14pp) e custa **$16-19 USD por +1pp**. Justifica-se para validação final na tese, **não** para iteração rápida de pesquisa.

3. **NaiveBayes é o único modelo que se beneficia consistentemente do LLM** — RF, GBT, MLP saturam com features estatísticas; SVM/KNN sofrem com a dimensionalidade extra. Para usar LLM, **commit a NB.**

---

## 5 achados principais

### 1. Flash + ML ≈ ML-only

| Modelo | ML-only CV | Flash CV | Δ |
|--------|:----------:|:--------:|:-:|
| NaiveBayes | 47.47% | 47.44% | **−0.03pp** |
| RandomForest | 38.82% | 39.60% | +0.78pp |
| LogisticRegression | 41.99% | 39.79% | −2.20pp |

Em **5 dos 7 modelos**, Flash performou **igual ou pior** que ML-only. Conclusão: as features Flash não adicionam sinal estatisticamente distinguível neste benchmark.

### 2. Pro + ML melhora consistentemente apenas em NaiveBayes

NB é o único com ganho robusto:
- **Pro vs ML:** +1.86pp CV / +1.27pp holdout
- **Pro vs Flash:** +1.89pp CV / +4.05pp holdout

Outros modelos: ganho marginal (RF +0.15pp), nulo (LogReg, MLP) ou negativo (SVM −4.68pp, KNN −3.18pp).

### 3. SVM e KNN PIORAM com features LLM

| Modelo | ML-only CV | Pro CV | Δ |
|--------|:----------:|:------:|:-:|
| SVM_RBF | 37.53% | 32.85% | **−4.68pp** |
| KNN | 38.20% | 35.02% | **−3.18pp** |

Classificadores baseados em distância são prejudicados pela dimensionalidade adicional + ruído das features LLM (especialmente nos 9 datasets críticos onde LLM falha).

### 4. LogReg holdout é estável: 54.94% nas 3 configurações

Sugere que LogReg captura o sinal principal apenas via features estatísticas robustas. Adicionar features LLM (boas ou ruidosas) não move a aguilha em LogReg.

### 5. Variância CV aumenta com features LLM

| Configuração | Std médio (7 modelos) |
|--------------|:---------------------:|
| ML-only | ±10.7% |
| Flash + ML | ±21.1% |
| Pro + ML | ±21.9% |

Features LLM **dobram a variância** entre folds. Isto é coerente com o achado de que o LLM falha em alguns datasets — quando esses caem em um fold de teste, a fold inteira tem performance ruim, ampliando std.

---

## Implicações imediatas

### Para o uso operacional

| Cenário | Recomendação |
|---------|--------------|
| Iteração de pesquisa (Steps 2/3, fine-tuning de prompts) | **Use ML-only ou Pro** — Flash é desperdício |
| Validação final para tese/paper | **Pro** — única configuração que justifica número final superior |
| Demonstração rápida de pipeline | **ML-only** — gratuito, instantâneo, ~95% do desempenho |
| Análise per-classe (recall MNAR) | **Pro com NB** — único onde MNAR recall sobe consistentemente |

### Para a narrativa da tese

A linha narrativa deve ser:

> "O LLM context-aware adiciona ganho **incremental** (+1.86pp CV em NaiveBayes) sobre o pipeline de 25 features estatísticas no benchmark de 29 datasets reais. O efeito é estatisticamente fraco mas direcionalmente consistente, limitado por 9 datasets onde o LLM falha em superar seu MAR-bias prior. Para benchmarks menores ou domínios onde o LLM tem maior afinidade, o ganho histórico foi maior (+5-9pp). A conclusão é que features LLM são **úteis mas não essenciais** — o ganho marginal versus o custo (~$30+ por execução com Pro) sugere que pesquisas futuras devem priorizar **decomposição estruturada do raciocínio LLM** (Steps 2/3) sobre prompt engineering simples."

---

## Limitações desta comparação

1. **Single seed (=42):** sem múltiplas execuções, deltas de ±2pp são tecnicamente dentro do ruído. Confirmação requereria 3+ runs (~$90+).

2. **ML-only foi "extraída" do dataset com LLM:** as imputações de mediana usadas no `X_features.csv` foram calculadas com features LLM presentes (afetando apenas estatísticas de mediana, não as estatísticas em si). É controle de quase-experimento, não experimento RCT.

3. **Mesmo prompt para Flash e Pro (Step 1):** não temos comparação com "prompt original" pré-Step 1. Possível que Flash com prompt ainda mais cuidadoso tivesse números melhores.

4. **Apenas 1 estratégia de CV:** Group 5-Fold foi comparada; LODO daria números diferentes (provavelmente mais favoráveis a Pro nos datasets onde tem confiança).
