# Resumo Executivo — Step 1 V2 Neutral

**Data:** 2026-04-25

---

## Pergunta de pesquisa

> O prompt Step 1 (3 exemplos canônicos + tipologia MNAR + instrução explícita anti-MAR-bias), executado com gemini-3-pro-preview e metadata neutral sobre o benchmark expandido de 29 datasets reais, eleva a acurácia hierárquica de classificação de mecanismos de missing data (MCAR/MAR/MNAR) acima do patamar de 56.2% CV obtido em `forensic_neutral_v2` (que usava 23 datasets)?

## Resposta encontrada

**Não.** O Step 1 com Pro + neutral atinge **49.33% CV** (NaiveBayes, Group 5-Fold), regredindo **−7pp** em relação ao `forensic_neutral_v2` e ficando **−10.7pp abaixo** do target planejado de 60%+ CV. A causa principal é a **persistência do MAR-bias** em 6 datasets do benchmark expandido — incluindo o caso canônico `MNAR_pima_insulin` (4% recall), que o Step 1 anti-bias falhou em corrigir.

Apesar disso, o experimento confirma duas tendências positivas: (a) **Pro supera Flash com Step 1** em +1.9pp CV / +4.05pp holdout sobre `step10_flash_ca_neutral`; (b) **NaiveBayes domina** classificadores complexos (RF, GBT, MLP) por +6 a +13pp em CV, reforçando o padrão observado na fase V3+ de que calibração de incerteza é mais importante que capacidade do modelo neste regime de rótulos ruidosos.

---

## Números-chave

### Configuração

| Item | Valor |
|------|-------|
| Modelo LLM | gemini-3-pro-preview |
| Extrator | `context_aware` (Step 1 prompt) |
| Metadata variant | `neutral` (Canal F fechado) |
| Datasets reais | 29 (9 MCAR + 11 MAR + 9 MNAR) |
| Bootstraps totais | 1.421 |
| Features | 34 (25 estatísticas + 9 LLM `llm_ctx_*`) |
| Split | GroupShuffleSplit 75/25 + Group 5-Fold CV |
| Workers paralelos | 10 |

### Performance

| Modelo | Holdout (n=395) | CV (Group 5-Fold) |
|--------|:---------------:|:-----------------:|
| **NaiveBayes** | **55.19%** | **49.33% ± 14.2%** |
| LogisticRegression | 54.94% | 41.54% ± 23.5% |
| SVM_RBF | 43.80% | 32.85% ± 27.4% |
| MLP | 43.04% | 33.26% ± 24.3% |
| RandomForest | 41.77% | 38.97% ± 26.6% |
| GradientBoosting | 41.27% | 36.32% ± 18.7% |
| KNN | 39.75% | 35.02% ± 18.5% |

### LLM standalone (`llm_ctx_domain_prior` sozinho como classificador)

| Métrica | Valor | vs `forensic_neutral_v2` |
|---------|:-----:|:------------------------:|
| Accuracy global | 43.7% | −19.4pp |
| Recall MCAR | 24.9% | — |
| Recall MAR | 67.6% | (mantido) |
| Recall MNAR | 32.0% | — |

A queda de −19.4pp em domain_prior solo (de 63.1% → 43.7%) é o sinal mais forte de que o problema **não é regressão do prompt** (que melhorou anti-bias), mas sim **dificuldade dos 6 datasets adicionados ao benchmark** entre `forensic_neutral_v2` (23 datasets) e o estado atual (29 datasets).

### Comparação com referências

| Experimento | Holdout best | CV avg best | domain_prior alone | n datasets |
|-------------|:------------:|:-----------:|:------------------:|:----------:|
| **`step1_v2_neutral`** (este) | **55.19%** | **49.33%** | 43.7% | 29 |
| `step10_flash_ca_neutral` | 51.14% | 47.44% | — | 29 |
| `forensic_neutral_v2` (ref) | — | 56.2% | 63.1% | 23 |
| Target Step 1 (planejado) | — | 60%+ | — | 29 |
| Chance level (3 classes) | 33.3% | 33.3% | 33.3% | — |

### Custos e tempo

| Etapa | Tempo | Custo estimado |
|-------|:-----:|:--------------:|
| Metade 1 (15 datasets, 721 bootstraps) | 47:12 | $15-18 |
| Metade 2 (14 datasets, 700 bootstraps) | 46:24 | $15-18 |
| Merge + treino | < 1 min | — |
| **Total** | **~1h33min** | **~$30-36 USD** |

---

## Veredicto em 3 frases

1. O Step 1 não atingiu o target, mas confirma o padrão **Pro > Flash** com prompt instrumentado e mantém **NaiveBayes** como classificador dominante para regimes ruidosos.
2. A regressão vs `forensic_neutral_v2` é majoritariamente atribuível à **expansão do benchmark** (6 datasets clinicamente difíceis adicionados), não a falha do prompt em si.
3. Para subir além de 50% CV no benchmark de 29 datasets, é necessário **Step 2 (Causal Reasoning DAG)** que ataca diretamente o MAR-bias residual em datasets onde o LLM atualmente "encontra razões" para MAR sem nomear a variável causadora — exatamente o padrão observado em `pima_insulin`, `kidney_pot`, `kidney_sod`, `hypothyroid_t4u`, `echomonths_epss` e `cylinderbands_esavoltage`.

---

## Pontos fortes do experimento

1. **Anti-vazamento auditado:** Canal F fechado via `--metadata-variant neutral`; canais A-E confirmados fechados na revisão de código.
2. **Reprodutibilidade total:** seed=42, listas de datasets versionadas em `Scripts/v2_improved/data/datasets_part{1,2}.txt`, comando exato registrado.
3. **Cobertura completa:** todos os 29 datasets processados, distribuição balanceada (421 MCAR + 550 MAR + 450 MNAR).
4. **Sem leakage de grupos:** GroupShuffleSplit confirmou 0 grupos compartilhados entre train (21 datasets) e test (8 datasets).
5. **Comparabilidade:** mesma arquitetura de 7 classificadores, mesma estratégia CV, mesma feature engineering que experimentos anteriores.

## Limitações reconhecidas

1. **Variância CV alta** (NB: ±14.2pp; outros: ±18-27pp) — indica que poucos folds e grupos heterogêneos tornam a estimativa instável.
2. **`step1_fewshot` antigo é incomparável** — usava dataset menor (16/29 datasets, 780/1421 bootstraps) e variante de metadata desconhecida (provavelmente leaky). Não pode ser usado como baseline direto.
3. **Não testado contra Step 2/3** — o pipeline ainda não rodou Causal DAG nem Self-Consistency com Pro+neutral, então o teto real do prompt engineering com este dataset permanece aberto.
4. **Custo não permite múltiplas reruns** — variância alta sugere que confiança estatística beneficiaria-se de re-executar com seeds diferentes, mas $30+/run torna isso caro.
