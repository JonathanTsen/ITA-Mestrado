# Resultados ML-only sobre 32 datasets reais (Fase 12)

**Experimento:** `step12_ml_only_v2b_32datasets`
**Pipeline:** `extract_features` (25 features, sem LLM) → `train_model` (7 classificadores, Group 5-Fold CV + 75/25 holdout)
**Comando:**
```bash
uv run python -m missdetect.extract_features --model none --llm-approach caafe \
  --data real --metadata-variant neutral \
  --datasets-include src/missdetect/metadata/datasets_v2b_32.txt \
  --experiment step12_ml_only_v2b_32datasets

uv run python -m missdetect.train_model --model none --data real \
  --experiment step12_ml_only_v2b_32datasets
```

**Saída em:** `results/step12_ml_only_v2b_32datasets/real/apenas_ml/baseline/`

## 1. Tabela síntese — accuracy

| Modelo              | Holdout | CV Group 5-Fold (μ ± σ) | Δ vs ML 29 (CV) |
|---------------------|:------:|:-----------------------:|:----------------:|
| **GradientBoosting**| **51.25%** | **52.54% ± 21.5** | **+5.07pp** ✅ |
| RandomForest        | 47.25% | 51.61% ± 15.2     | +12.79pp ✅      |
| MLP                 | 42.25% | 44.60% ± 29.4     | (n/a)            |
| KNN                 | 38.50% | 44.35% ± 15.3     | +6.15pp          |
| NaiveBayes          | 33.50% | 42.59% ± 34.2     | **−4.88pp** ❌   |
| SVM_RBF             | 41.00% | 42.57% ± 22.9     | +5.04pp          |
| LogisticRegression  | 36.75% | 36.43% ± 18.6     | **−5.56pp** ❌   |

**Referência da Fase 6 (ML 29 datasets):** NaiveBayes 47.47% CV (best),
LogisticRegression 54.94% holdout (best). O modelo dominante mudou de NB
para GBT.

## 2. Recall por classe (holdout, n=400)

Com **support fixo: MCAR=100, MAR=100, MNAR=200** (split GroupShuffleSplit 75/25, seed=42).

| Modelo              |  MCAR | MAR | MNAR | Macro avg |
|---------------------|:-----:|:---:|:----:|:---------:|
| **GradientBoosting**| 41.0% | **46.0%** | **59.0%** | **48.7%** |
| RandomForest        | 36.0% | 40.0% | 56.5% | 44.2%     |
| SVM_RBF             | 59.0% | 36.0% | 34.5% | 43.2%     |
| MLP                 | 44.0% | 32.0% | 46.5% | 40.8%     |
| KNN                 | 63.0% | 23.0% | 34.0% | 40.0%     |
| LogisticRegression  | 54.0% | 15.0% | 39.0% | 36.0%     |
| NaiveBayes          | **75.0%** | **1.0%** | 29.0% | 35.0% |

### Quem é melhor em cada classe

- **MCAR:** NaiveBayes 75% (mas custo: MAR 1% — modelo apenas chuta MCAR).
  Excluindo NB, **KNN 63%** e **SVM 59%** são os melhores honestos.
- **MAR:** GBT 46% — a classe mais difícil; nenhum modelo passa de 50%.
- **MNAR:** GBT 59% — coerente com o ganho das CAAFE-MNAR features
  (kurtosis_excess, cond_entropy, kl_density), que dominam o feature importance.

### Padrão por classe

- **MCAR (100 amostras):** spread enorme entre modelos (36–75%). Modelos de
  capacidade alta (RF, GBT) confundem MCAR com MNAR; modelos simples (NB, KNN,
  LogReg) viciam-se em prever MCAR — o que aumenta o recall MCAR à custa de MAR.
- **MAR (100 amostras):** classe mais difícil para todos. Recall vai de 1% (NB)
  a 46% (GBT). Confunde-se principalmente com MNAR — coerente com Molenberghs
  et al. (2008): MAR e MNAR não são separáveis somente pelos dados observados.
- **MNAR (200 amostras, classe majoritária):** classe mais fácil porque tem o
  dobro de amostras E porque as features CAAFE-MNAR foram desenhadas para
  detectá-la. Recall 29–59%.

### Por que GBT lidera

GBT é o **único modelo com recall ≥ 41% em todas as três classes**. Os outros
sacrificam pelo menos uma:

- NB sacrifica MAR (1%) e MNAR (29%)
- LogReg/KNN sacrificam MAR (15%, 23%)
- SVM/KNN sacrificam MNAR (34–35%)

GBT não é o melhor em **nenhuma** classe individual, mas é o **único equilibrado**
— e isso basta para liderar o agregado tanto em holdout quanto em CV.

## 3. Matrizes de confusão (holdout)

### GradientBoosting (best)
```
              pred_MCAR  pred_MAR  pred_MNAR
true_MCAR        41        14        45
true_MAR          0        46        54
true_MNAR        25        57       118
```
Erro principal: 45 MCARs preditos como MNAR; 57 MNARs preditos como MAR.

### NaiveBayes (colapso da classe MAR)
```
              pred_MCAR  pred_MAR  pred_MNAR
true_MCAR        75         2        23
true_MAR         19         1        80
true_MNAR       110        32        58
```
NB praticamente desistiu da classe MAR (1/100). Está prevendo MCAR ou MNAR
para quase tudo. **110 MNARs preditos como MCAR** — sinal de que a distribuição
gaussiana assumida pelo NB não consegue mais separar essas duas classes com os
4 novos MCARs (`boys_*`, `brandsma_*`) que têm distribuições muito diferentes
dos antigos (`hepatitis_*`).

### LogisticRegression (segunda pior)
```
              pred_MCAR  pred_MAR  pred_MNAR
true_MCAR        54        40         6
true_MAR          6        15        79
true_MNAR        65        57        78
```
LogReg tinha sido o **best holdout no benchmark de 29** (54.94%); agora está em
36.75%. Confunde MAR com MNAR de forma sistemática (79/100 dos MARs vão para MNAR).

## 4. Feature importance (RandomForest)

| # | Feature | Importance |
|--:|--|:--:|
| 1 | `caafe_kurtosis_excess` | 12.91% |
| 2 | `X0_obs_skew_diff` | 11.39% |
| 3 | `caafe_cond_entropy_X0_mask` | 10.33% |
| 4 | `X0_obs_vs_full_ratio` | 9.43% |
| 5 | `X0_censoring_score` | 5.66% |
| 6 | `caafe_kl_density` | 5.37% |
| 7 | `X1_mean_diff` | 4.36% |
| 8 | `X0_mean_shift_X1_to_X4` | 3.79% |
| 9 | `little_proxy_score` | 3.65% |
| 10 | `caafe_auc_self_delta` | 3.46% |

**Total CAAFE-MNAR (4 features):** 26.3% — quase 1/4 da importância total.

Confirma a hipótese da Fase 8 (protocolo v2b): as features CAAFE-MNAR
adicionadas para detectar mecanismos não-aleatórios estão de fato
contribuindo. Nesta execução **sem LLM**, a importância LLM é zero por
construção; a comparação relevante (LLM marginal aporta ou não?) só será
possível quando o Flash terminar.

## 5. Variância CV alta — sinal de heterogeneidade do benchmark

Std médio dos 7 modelos: **±21.6pp**, muito acima do habitual (±9-15pp na
Fase 6). Isso significa que **a performance depende fortemente de quais
datasets caem em qual fold** — alguns são fáceis (NHANES MNAR, Pima MNAR),
outros são MCAR-puzzle (boys/brandsma) que confundem o classificador.

Implicação prática: as comparações "X bate Y por +Δpp" só são confiáveis se
Δ > 5pp. Diferenças menores (ex: NB 42.59 vs SVM 42.57) estão dentro do
ruído de fold.

## 6. Conclusões

1. **GBT é o novo best ML-only** sobre o benchmark v2b: 52.54% CV / 51.25% holdout.
2. **NB caiu de líder a penúltimo** (47.47% → 42.59% CV). Hipótese: a curadoria removeu boa parte do ruído de labels que justificava o ganho de NB na Fase 6.
3. **CAAFE-MNAR features dominam o ranking de importância** (3 das top 6, 26% do total) — investimento em CAAFE pagou.
4. **MAR continua a classe mais difícil** (best recall 46%, Molenberghs limit em ação).
5. **Variância CV ±21pp** indica que o benchmark ainda é heterogêneo demais para conclusões finas. Diferenças <5pp não são confiáveis.

## Arquivos gerados

| Caminho | Conteúdo |
|---|---|
| `results/step12_ml_only_v2b_32datasets/real/apenas_ml/baseline/relatorio.txt` | Relatório textual detalhado |
| `.../resultados.png` | Gráfico de accuracy por modelo |
| `.../precisao_por_classe.png` | Gráfico de precisão por classe |
| `.../predictions.csv` | Predições por amostra (2.800 linhas: 400 holdout + 5 folds × ~480) |
| `.../metrics_per_class.csv` | precision/recall/F1 por classe e modelo |
| `.../feature_importance.csv` | Importância das 25 features (RandomForest) |
| `.../cv_scores.csv` | Acurácia por fold por modelo (35 linhas: 7×5) |
| `.../confusion_matrices.json` | Matrizes de confusão por modelo (holdout) |
| `.../hyperparameters.json` | Hiperparâmetros usados |
| `.../X_features.csv` | Matriz de features (1.593 × 25) |
| `.../y_labels.csv` | Labels (1.593) |
| `.../groups.csv` | Grupo de origem por amostra (1.593, 32 únicos) |
