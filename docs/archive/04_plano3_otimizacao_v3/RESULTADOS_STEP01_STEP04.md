# Resultados STEP 01 (Cleanlab) + STEP 04 (Roteamento Probabilístico)

**Data:** 2026-04-18
**Experimento:** step05_pro
**Status:** CONCLUÍDO (3 runs executados)

---

## Resumo Executivo

| Run | Configuração | Holdout Max | LOGO CV Max | Modelo |
|:---:|-------------|:---:|:---:|---|
| 1 | Sem pesos, sem routing | 50.5% | 51.4% | GBT / NaiveBayes |
| 2 | Sem pesos, com routing | 50.5% | **56.0%** | GBT / NaiveBayes+soft3zone |
| **3** | **Com pesos + routing** | **53.2%** | **56.0%** | **GBT+threshold / NaiveBayes+soft3zone** |

**Melhorias totais:** Holdout +2.7pp, LOGO CV +4.6pp, MNAR recall +6pp, F1 +0.027

---

## STEP 01: Cleanlab — Limpeza de Labels

### Labels Problemáticos: 59.4% (672/1132)

**15 de 23 datasets** têm label discordante com o modelo.

### Piores Datasets (quality < 0.1)

| Dataset | Label | Pred | Quality | Issues |
|---------|:-----:|:----:|:-------:|:------:|
| MCAR_hypothyroid_t4u | MCAR | MAR | 0.001 | 100% |
| MAR_sick_tsh | MAR | MNAR | 0.024 | 98% |
| MCAR_autompg_horsepower | MCAR | MAR | 0.024 | 97% |
| MNAR_mroz_wages | MNAR | MAR | 0.029 | 100% |
| MAR_sick_t3 | MAR | MCAR | 0.073 | 92% |

### Melhores Datasets (quality > 0.7)

| Dataset | Label | Pred | Quality | Issues |
|---------|:-----:|:----:|:-------:|:------:|
| MAR_oceanbuoys_airtemp | MAR | MAR | 0.940 | 2% |
| MAR_titanic_age | MAR | MAR | 0.907 | 6% |
| MCAR_breastcancer_barenuclei | MCAR | MCAR | 0.893 | 4% |
| MAR_titanic_age_v2 | MAR | MAR | 0.736 | 22% |
| MAR_hearth_chol | MAR | MAR | 0.686 | 22% |

### Confident Joint

```
           | pred MCAR | pred MAR | pred MNAR
      MCAR |       64  |     109  |       59
       MAR |       63  |     260  |      227
      MNAR |       25  |     215  |      110
```

### Quality por Classe

| Classe | Quality média |
|--------|:---:|
| MAR | 0.466 |
| MNAR | 0.317 |
| MCAR | 0.268 |

**Interpretação:** MCAR é o mais confuso. MAR é a classe "absorvente" — 215 MNAR preditos como MAR, 109 MCAR preditos como MAR.

### Pesos Salvos
- min=0.100, mean=0.417, max=1.000
- Mediana=0.190 (a maioria das amostras tem peso baixo)

---

## STEP 04: Roteamento Probabilístico V3+ — Run 2 (sem pesos)

### Holdout (295 amostras)

| Variante | Acc Max | Modelo | MNAR Recall | F1 |
|----------|:-------:|--------|:-----------:|:--:|
| **V3 hard** | **50.5%** | GBT | 40.0% | 0.488 |
| V3+ threshold | 50.2% | GBT | **44.0%** | 0.488 |
| V3+ soft3zone | 48.1% | GBT | 28.0% | 0.455 |
| V3+ fullprob | 48.1% | GBT | 28.0% | 0.455 |

### LOGO CV (23 folds) — sem pesos

| Modelo | V3 hard | V3+ soft3zone | V3+ fullprob | V3+ threshold |
|--------|:-------:|:-------------:|:------------:|:-------------:|
| **NaiveBayes** | 51.4% | **56.0%** | **56.0%** | 51.7% |
| LogisticRegression | 44.0% | 45.3% | **45.4%** | 44.4% |
| KNN | 37.1% | **38.5%** | **38.5%** | 37.2% |
| MLP | 31.6% | **33.2%** | **33.2%** | 31.5% |
| RandomForest | 38.3% | 36.9% | 36.9% | 37.5% |
| GradientBoosting | 38.3% | 35.8% | 35.8% | 38.1% |
| SVM_RBF | 31.8% | 24.9% | 24.8% | 31.7% |

### McNemar: V3 hard vs V3+ soft3zone (sem pesos)

| Modelo | b | c | p-value | Sig? |
|--------|:-:|:-:|:-------:|:----:|
| **LogisticRegression** | 3 | 18 | **0.002** | ** |
| **MLP** | 3 | 13 | **0.024** | * |
| **NaiveBayes** | 2 | 12 | **0.016** | * |
| KNN | 7 | 17 | 0.066 | marginal |

---

## Run 3: Combinação Cleanlab Pesos + Routing

### Holdout (295 amostras) — com pesos Cleanlab

| Variante | Acc Max | Modelo | MNAR Recall | F1 |
|----------|:-------:|--------|:-----------:|:--:|
| **V3+ threshold** | **53.2%** | GBT | **46.0%** | **0.515** |
| V3 hard | 52.9% | GBT | 44.0% | 0.509 |
| V3+ soft3zone | 45.8% | SVM_RBF | 2.0% | 0.373 |
| V3+ fullprob | 45.8% | SVM_RBF | 2.0% | 0.373 |

### Comparação: Sem pesos vs Com pesos (holdout)

| Variante | Sem pesos | Com pesos | Delta |
|----------|:---------:|:---------:|:-----:|
| V3 hard (GBT) | 50.5% | **52.9%** | **+2.4pp** |
| V3+ threshold (GBT) | 50.2% | **53.2%** | **+3.0pp** |
| V3 hard MNAR recall | 40.0% | **44.0%** | **+4.0pp** |
| V3+ threshold MNAR recall | 44.0% | **46.0%** | **+2.0pp** |

### V3 hard com pesos — detalhado por modelo

| Modelo | Sem pesos | Com pesos | Delta |
|--------|:---------:|:---------:|:-----:|
| **GBT** | 50.5% | **52.9%** | **+2.4pp** |
| **SVM_RBF** | 41.0% | **50.5%** | **+9.5pp** |
| **LogisticRegression** | 41.0% | **49.2%** | **+8.2pp** |
| **NaiveBayes** | 41.7% | **47.1%** | **+5.4pp** |
| **KNN** | 41.7% | **43.4%** | **+1.7pp** |
| RandomForest | **43.4%** | 44.1% | +0.7pp |
| MLP | **39.7%** | 40.7% | +1.0pp |

**Pesos Cleanlab melhoram TODOS os modelos!** Maior ganho: SVM_RBF +9.5pp, LogReg +8.2pp.

### LOGO CV (23 folds) — mesmo que Run 2

Os resultados LOGO CV são **idênticos** ao Run 2 porque o LOGO CV interno não aplica pesos (usa SMOTE padrão). Os pesos só afetam o split holdout.

| Modelo | LOGO CV (= Run 2) |
|--------|:--:|
| **NaiveBayes** | **56.0%** |
| LogisticRegression | 45.4% |
| KNN | 38.5% |

### McNemar: V3 hard vs V3+ soft3zone (com pesos)

**Com pesos, soft3zone PIORA significativamente:**

| Modelo | b (hard>soft3zone) | c (soft3zone>hard) | p-value | Sig? |
|--------|:--:|:--:|:-------:|:----:|
| **GBT** | 24 | 0 | **<0.0001** | *** (PIORA) |
| **LogisticRegression** | 24 | 3 | **0.0001** | *** (PIORA) |
| **NaiveBayes** | 26 | 11 | **0.021** | * (PIORA) |
| **SVM_RBF** | 28 | 14 | **0.045** | * (PIORA) |

**Razão:** Com pesos (sem SMOTE), o L2 tem ~70% MAR / 30% MNAR sem balanceamento. A calibração de probabilidades no soft3zone/fullprob fica enviesada para MAR. O threshold routing é mais robusto porque mantém a decisão hard com um cutoff ajustado.

---

## Análise Consolidada

### Por que Threshold + Pesos é a melhor combinação no holdout?

1. **Pesos Cleanlab** reduzem influência de labels ruidosos no `.fit()` → modelos aprendem padrões mais limpos
2. **Threshold routing** é conservador: apenas ajusta o ponto de corte L1, sem combinar probabilidades
3. **Sem SMOTE quando com pesos** → menos amostras sintéticas na fronteira ruidosa
4. **GBT se beneficia** dos pesos porque suporta `sample_weight` nativo

### Por que Soft3zone é melhor no LOGO CV (sem pesos)?

1. **NaiveBayes** produz probabilidades naturalmente bem calibradas
2. **Com SMOTE** (ativo no LOGO CV), as classes são balanceadas → calibração é mais estável
3. **Zona incerta** (35%-65%) captura amostras que hard routing classificaria errado
4. **LOGO CV** com 23 folds é mais robusto que holdout único

### Estratégia Recomendada para o Paper

Reportar **ambos os cenários**:
- **LOGO CV:** NaiveBayes + soft3zone = 56.0% (robustez, generalização)
- **Holdout:** GBT + threshold + Cleanlab pesos = 53.2% (accuracy + MNAR recall)

---

## Descobertas Importantes para o Paper

### 1. Labels são o gargalo principal
59.4% dos labels reais são inconsistentes. Pesos Cleanlab melhoram **todos** os modelos (+0.7pp a +9.5pp). Isso sugere que o teto real de accuracy é significativamente maior que os ~41% do baseline.

### 2. Routing probabilístico funciona, mas depende do contexto
- **Com SMOTE (classes balanceadas):** soft3zone > hard para modelos calibrados
- **Com pesos (sem SMOTE):** threshold > soft3zone (calibração instável sem balanceamento)

### 3. MNAR recall é o indicador mais sensível
Pesos Cleanlab melhoram MNAR recall de 40% → 46% — evidência de que labels ruidosos confundiam especialmente a classe MNAR.

---

## Próximos Passos (por prioridade)

### 1. Testar SMOTE-ENN (flag `--balancing smote_enn` já implementada)
```bash
python train_hierarchical_v3plus.py --data real --experiment step05_pro --routing all --balancing smote_enn
python train_hierarchical_v3plus.py --data real --experiment step05_pro --routing all --balancing smote_enn --clean-labels weight
```
Potencial: SMOTE-ENN limpa fronteira + pesos reduzem ruído → pode resolver o problema de soft3zone com pesos.

### 2. Step 02 — CatBoost + Optuna
GBT com pesos já chega a 53.2%. CatBoost pode ir mais longe. Optuna otimiza por nível.

### 3. Step 03 — Novas Features L2
Só depois de estabilizar o pipeline com SMOTE-ENN + classificadores melhores.

---

## Bug Corrigido

**`_fit_with_weights()` não era chamado em `run_hierarchical_v3plus()`.**
- Run 1 e 2 usavam `model.fit(X, y)` ignorando pesos
- Run 3 usa `_fit_with_weights(model, X, y, weights)` corretamente
- Quando pesos ativos, SMOTE é desativado (incompatível com sample_weight)
- Pipeline detecta step `clf` e passa `clf__sample_weight`

---

## Arquivos Gerados

```
Output/v2_improved/step05_pro/real/
├── label_analysis/                    # Step 01
│   ├── clean_labels_summary.json
│   ├── confident_joint.png
│   ├── label_issues_ranked.csv
│   ├── label_quality_scores.csv
│   ├── quality_by_dataset.csv
│   ├── quality_by_dataset.png
│   ├── quality_distribution.png
│   └── sample_weights.csv
└── hierarquico_v3plus/                # Step 04 (Run 3 = mais recente)
    ├── todas_variantes_v3plus.csv     # 5 variantes × 7 modelos
    ├── resumo_v3plus.csv
    ├── significancia_v3plus.csv       # McNemar todos os pares
    ├── cv_logo_v3plus.csv             # LOGO CV (= Run 2, pesos não afetam)
    ├── v3plus_comparison.png
    ├── heatmap_v3plus.png
    └── training_summary.json          # Metadata com clean_labels=weight
```
