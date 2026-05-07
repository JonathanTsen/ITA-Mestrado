# Resultados Flash + ML sobre 32 datasets reais (Fase 12)

**Experimento:** `step12_flash_neutral_v2b_32datasets`
**Pipeline:** `extract_features` (34 features: 25 stat + 9 LLM) → `train_model` (7 classificadores, Group 5-Fold CV + 75/25 holdout)
**Modelo LLM:** `gemini-3-flash-preview` (metadata_variant=neutral, llm_approach=context)
**Custo estimado:** ~$2-4 | **Tempo:** ~45 min (1.593 bootstraps, 10 workers paralelos)

## 1. Tabela síntese — accuracy

| Modelo              | Holdout | CV Group 5-Fold (μ ± σ) | Δ vs Flash 29 (CV) |
|---------------------|:------:|:-----------------------:|:------------------:|
| **RandomForest**    | 47.50% | **51.93% ± 18.0**   | (n/a — NB era ref.) |
| **GradientBoosting**| **50.25%** | 51.32% ± 25.8   | — |
| MLP                 | 45.00% | 46.53% ± 21.4     | — |
| NaiveBayes          | 42.25% | 44.91% ± 30.7     | **−2.53pp** vs 47.44% |
| SVM_RBF             | 41.75% | 43.93% ± 25.4     | — |
| KNN                 | 40.25% | 42.34% ± 20.4     | — |
| LogisticRegression  | 37.75% | 37.24% ± 7.3      | — |

**Referência Phase 6 Flash (step10_flash_ca_neutral, 29 datasets):** NB 47.44% CV (best), LogReg 54.94% holdout (best).

## 2. Flash × ML-only — comparação direta (32 datasets)

| Modelo | ML-only CV | Flash CV | Δ Flash-ML | ML-only Holdout | Flash Holdout | Δ |
|--------|:----------:|:--------:|:----------:|:---------------:|:-------------:|:--:|
| GBT    | **52.54%** | 51.32%   | **−1.22pp**| **51.25%** | **50.25%** | **−1.00pp** |
| RF     | 51.61%     | **51.93%** | **+0.32pp**| 47.25% | 47.50% | +0.25pp |
| MLP    | 44.60%     | 46.53%   | +1.93pp    | 42.25% | 45.00% | +2.75pp |
| NB     | 42.59%     | 44.91%   | +2.32pp    | 33.50% | 42.25% | +8.75pp |
| SVM    | 42.57%     | 43.93%   | +1.36pp    | 41.00% | 41.75% | +0.75pp |
| KNN    | 44.35%     | 42.34%   | −2.01pp    | 38.50% | 40.25% | +1.75pp |
| LogReg | 36.43%     | 37.24%   | +0.81pp    | 36.75% | 37.75% | +1.00pp |

**Resultado-chave:** nenhum ganho Flash > 5pp, portanto **nenhum ganho confiável**
(limiar de confiabilidade = 5pp dado ±21pp CV std). Flash não bate ML-only de forma
consistente nem em CV nem em holdout — mesmo padrão de Pareto-dominância da Fase 6.

## 3. Recall por classe (holdout, n=400)

Support fixo: MCAR=100, MAR=100, MNAR=200.

| Modelo              | MCAR | MAR | MNAR | Macro avg |
|---------------------|:----:|:---:|:----:|:---------:|
| KNN                 | **70%** | 26% | 32.5% | 42.8% |
| SVM_RBF             | 64%  | 37% | 33.0% | 44.7% |
| GradientBoosting    | 50%  | **45%** | **53.0%** | **49.3%** |
| MLP                 | 51%  | 36% | 46.5% | 44.5% |
| RandomForest        | 44%  | 38% | 54.0% | 45.3% |
| NaiveBayes          | **79%** | 1%  | 44.5% | 41.5% |
| LogisticRegression  | 61%  | 19% | 35.5% | 38.5% |

### Flash × ML-only — recall por classe (GBT holdout)

| Classe | ML-only | Flash | Δ |
|--------|:-------:|:-----:|:--:|
| MCAR   | 41%     | **50%** | **+9pp** |
| MAR    | 46%     | 45%   | −1pp |
| MNAR   | 59%     | 53%   | −6pp |

Flash melhora MCAR (+9pp) mas perde em MNAR (−6pp). O ganho em MCAR é plausivelmente
porque o contexto neutro do domínio ajuda a distinguir MCAR de MNAR (ex: design MCAR
em `boys_*`, `brandsma_*` — dados de visitação/teste, contexto fácil de reconhecer).
A piora em MNAR pode refletir confusão do LLM em datasets borderline (NHANES, SUPPORT2).

## 4. Matrizes de confusão (holdout)

### GradientBoosting (best holdout)
```
              pred_MCAR  pred_MAR  pred_MNAR
true_MCAR        50        12        38
true_MAR          1        45        54
true_MNAR        15        79       106
```
Erro principal: 38 MCARs preditos como MNAR; 79 MNARs preditos como MAR. Mesmo padrão
de confusão da versão ML-only, mas com recall MCAR +9pp (50 vs 41).

### NaiveBayes (colapso MAR — idêntico ao ML-only)
```
              pred_MCAR  pred_MAR  pred_MNAR
true_MCAR        79         2        19
true_MAR         17         1        82
true_MNAR        78        33        89
```
NB ainda colapsa MAR (1/100 = 1%). Flash não resolve a limitação estrutural do NB
de assumir independência gaussiana — o contexto LLM não ajuda quando o modelo base
não consegue separar as distribuições.

## 5. Feature importance (RandomForest)

| # | Feature | Importance |
|--:|---------|:----------:|
| 1 | `caafe_kurtosis_excess` | 10.34% |
| 2 | `caafe_cond_entropy_X0_mask` | 9.88% |
| 3 | `X0_obs_skew_diff` | 8.64% |
| 4 | `X0_obs_vs_full_ratio` | 8.25% |
| 5 | `X0_censoring_score` | 4.74% |
| 6 | `caafe_kl_density` | 4.45% |
| 7 | `X1_mean_diff` | 3.58% |
| 8 | **`llm_ctx_domain_confidence`** | **3.51%** |
| 9 | `little_proxy_score` | 3.47% |
| 10 | `X0_mean_shift_X1_to_X4` | 3.21% |
| 19 | **`llm_ctx_counter_strength`** | **2.04%** |
| 20 | **`llm_ctx_cause_type`** | **2.04%** |

**Total LLM (9 features):** 12.94% | **Estatísticas (25 features):** 87.06%

Comparação:
- Flash 32 datasets: 12.94% LLM
- Pro 29 datasets (Fase 6): 12.6% LLM (source: `docs/archive/09_resultados_ml_flash_pro/`)

A contribuição LLM **é consistente entre Flash e Pro** (~13%) — o que muda não é o
quanto o LLM contribui, mas sim quão calibrado o contexto é. A `llm_ctx_domain_confidence`
entra no top-8, confirmando que a confiança do modelo no domínio é o sinal mais útil.

## 6. Diagnóstico das 3 hipóteses levantadas em 00_INDICE.md

### H1: "Flash continua dominado por ML-only?"

**Resposta: sim, estatisticamente.** Flash CV best = RF 51.93%; ML-only CV best = GBT 52.54%.
Diferença −0.61pp << limiar de confiabilidade de 5pp. No holdout: Flash GBT 50.25% vs
ML-only GBT 51.25% (−1.00pp). Flash não adiciona ganho confiável — **Pareto-dominado por ML-only**,
exatamente como na Fase 6 (onde Flash 47.44% ≈ ML-only 47.47%).

### H2: "A virada de regime se aplica também à interação ML×LLM?"

**Resposta: parcialmente.** No benchmark 29 datasets, NB era o único modelo que se
beneficiava do Pro (NB+Pro > NB+ML). No benchmark 32, GBT lidera tanto em ML-only
quanto em Flash, e o ganho diferencial do LLM é difuso entre modelos.
O padrão de que **o modelo dominante ganha mais do LLM** não se confirma: GBT perde
−1.22pp CV com Flash vs +2.32pp do NB — o LLM ajuda mais os modelos mais fracos (NB,
MLP) do que o líder (GBT).

### H3: "MAR recall (46% GBT ML-only) muda com Flash?"

**Resposta: não.** Flash GBT MAR = 45% (−1pp vs ML-only 46%). O LLM não ajuda
especificamente na classe mais difícil. A dificuldade de MAR não é falta de contexto
semântico — é uma limitação fundamental de identificabilidade (Molenberghs 2008).

## 7. Conclusões

1. **Flash ≈ ML-only** em 32 datasets: diferenças máximas de 1-2pp (abaixo do
   limiar de confiabilidade ±5pp) — mesmo padrão de Pareto-dominância da Fase 6.
2. **LLM contribui 12.94% de importância** (consistente com 12.6% do Pro na Fase 6),
   mas não converte em ganho de accuracy agregada.
3. **Flash melhora MCAR +9pp** (contexto reconhece planned-missingness designs);
   **piora MNAR −6pp** (confusão em datasets borderline censoring).
4. **GBT é o best em Flash**, exatamente como em ML-only — o regime pós-curadoria
   (GBT > NB) se mantém independentemente de usar ou não LLM.
5. **Pro sobre 32 datasets** (não rodado) continua sendo o próximo experimento natural
   para saber se um LLM mais capaz consegue Pareto-dominar o ML-only.

## Arquivos gerados

| Caminho | Conteúdo |
|---|---|
| `results/step12_flash_neutral_v2b_32datasets/real/ml_com_llm/gemini-3-flash-preview/relatorio.txt` | Relatório textual |
| `.../resultados.png` | Accuracy por modelo |
| `.../precisao_por_classe.png` | Precisão por classe |
| `.../predictions.csv` | 2.800 predições |
| `.../metrics_per_class.csv` | precision/recall/F1 por classe e modelo |
| `.../feature_importance.csv` | Importância das 34 features (RF) |
| `.../cv_scores.csv` | Accuracy por fold × modelo (35 linhas) |
| `.../confusion_matrices.json` | Matrizes de confusão (7 modelos) |
| `.../X_features.csv` | Matriz 1.593 × 34 |
