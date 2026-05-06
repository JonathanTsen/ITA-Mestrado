# Protocolo v2 de validação de rótulos — Arquitetura e resultados

**Data:** 2026-05-03
**Arquivos principais:** `src/missdetect/validar_rotulos_v2.py`, `src/missdetect/calibrar_protocolo.py`

---

## 1. Princípio de design

Em vez de tomar decisões binárias em cascata (v1), o protocolo v2 **acumula evidências** por mecanismo em camadas independentes e reconcilia via probabilidades Bayesianas.

```
Camada A (MCAR)    →  3 p-values + decisão ≥2/3
Camada B (MAR)     →  AUC + permutation p-value + MI
Camada C (MNAR)    →  4 scores CAAFE-MNAR
         ↓                 ↓                 ↓
     ┌─────────────────────────────────────────┐
     │  Camada D — reconciliação (regras OU Bayes)  │
     └─────────────────────────────────────────┘
                         ↑
     ┌─────────────────────────────────────────┐
     │  Camada E — calibração contra sintéticos     │
     └─────────────────────────────────────────┘
```

---

## 2. Camada A — Detecção de MCAR

Três testes complementares, voto majoritário (≥2 de 3 rejeitam → rejeita MCAR):

| # | Teste | Tipo | Vantagem sobre v1 |
|---|-------|------|---------------------|
| A.1 | Little's MCAR test | Paramétrico (normalidade) | Mantido como referência histórica |
| A.2 | **PKLM** (Spohn 2024) | Não-paramétrico (RandomForest + permutação) | Robusto a N grande; não assume normalidade |
| A.3 | **Levene-stratified** | Heterocedasticidade (variância), Bonferroni | Complementa Little que vê médias |

**Ganho principal:** PKLM reduz falsos positivos de Little em datasets com N > 1.000 (ex.: `hypothyroid_t4u`, `sick_*`). O voto majoritário estabiliza a decisão.

---

## 3. Camada B — Evidência de MAR

Substitui a correlação ponto-biserial por métodos que capturam **não-linearidade e interações**:

| # | Score | Método |
|---|-------|--------|
| B.1 | `auc_obs` | AUC de RandomForest 5-fold CV prevendo `mask` a partir de `X1..X4` (sem X0) |
| B.2 | `auc_p` | p-valor via 200 permutações de mask |
| B.3 | `auc_z` | Z-score: (AUC_obs - mean(AUC_perm)) / std(AUC_perm) |
| B.4 | `mi_max` | `sklearn.feature_selection.mutual_info_classif` — máxima mutual information entre mask e cada Xi |
| B.5 | `mi_mean` | Média da MI |

**Decisão (regras):** evidência MAR se `auc_obs > threshold` **e** `auc_p < 0.05`, ou `mi_max > threshold`.

**Ganho principal:** captura relações não-lineares (ex.: missing condicional a quartis de X1) e interações (X1 × X2) que a correlação ponto-biserial não detecta.

---

## 4. Camada C — Evidência indireta de MNAR

Substitui o KS obs-vs-mediana (tautológico) por 4 features CAAFE-MNAR em `features/caafe_mnar.py`.

**Versão v2b (atual — P5):** as features C.1 e C.2 originais (`caafe_quantile_ratio` e `caafe_tail_asym`) foram substituídas por features com poder discriminativo real (AUC original = 0.5 = chance):

| # | Score | O que mede | Por que sugere MNAR |
|---|-------|------------|---------------------|
| C.1 | `caafe_auc_self_delta` | Delta AUC ao incluir X0_imputado como feature para prever a própria máscara | MNAR: X0 prediz sua ausência → delta > 0; MCAR: delta ≈ 0 |
| C.2 | `caafe_kl_density` | KL(dist X0_imp onde observado ∥ dist X0_imp onde faltante) | MNAR: distribuições diferem → KL > 0; MCAR: similares → KL ≈ 0 |
| C.3 | `caafe_kurt_excess` | Curtose de X0_obs (Fisher) | Truncamento → platicúrtica |
| C.4 | `caafe_cond_entropy` | I(mask; X0_disc) / H(mask) — informação mútua normalizada | MNAR → mask depende de X0 → MI > 0 |

**Por que as originais falharam:**
- `caafe_quantile_ratio`: usava mediana dos observados para imputar → quartis colapsavam → razão ≈ constante em todos os mecanismos.
- `caafe_tail_asym`: calculava percentis de X0_obs e contava obs abaixo/acima → por definição ~10% em cada cauda → abs(n_lower - n_upper) ≈ 0 sempre.

**Decisão (regras):** evidência MNAR se ≥3 de 4 scores acima do limiar calibrado.

---

## 5. Camada D — Reconciliação

### Modo "regras" (default, sem calibração)

Árvore de decisão simples:
1. Camada A não rejeita MCAR → prediz **MCAR** (conf=0.7)
2. Rejeita MCAR + evidência MAR (B) + ≤1 score MNAR (C) → **MAR** (conf=0.6)
3. Rejeita MCAR + ≥3 scores MNAR (C) + sem evidência MAR → **MNAR** (conf=0.6)
4. Evidências mistas → decisão com confiança baixa (0.3–0.4)

### Modo "Bayes" (com `--bayes-scores calibration_scores.npz`)

1. Transforma scores das 3 camadas em vetor de 10 dimensões (`scores_to_vec`)
2. Avalia a likelihood do vetor sob 3 KDEs Gaussianos (um por mecanismo), ajustados nos sintéticos
3. Aplica Bayes: P(M | scores) ∝ P(scores | M) × P(M)
4. Prior: uniforme (1/3) por default; configurável via `--prior-mnar` para usar conhecimento de domínio
5. Confiança = P_max − P_segundo; se < 0.4 → "AMBÍGUO"
6. **Bandwidth ótimo** (P7): selecionado via GridSearchCV 5-fold se `--auto-bandwidth` for passado na calibração

**Vetor de 10 dimensões (v2b):**

```
log10(little_p), log10(pklm_p), log10(levene_p),
auc_obs, auc_z, mi_max,
caafe_auc_self_delta, caafe_kl_density, kurt_excess, cond_entropy
```

---

## 6. Camada E — Calibração contra sintéticos

Script: `calibrar_protocolo.py`

1. Amostra `n_per_class` datasets de cada mecanismo dos 1.200 sintéticos (com ground truth)
2. Roda Camadas A-C em cada, coleta os 10 scores
3. Para cada score, calcula curva ROC e threshold via Youden's J
4. Avalia accuracy do protocolo nos mesmos sintéticos sob 3 modos (default/calibrado/Bayes)
5. Salva `calibration.json` (thresholds) e `calibration_scores.npz` (matrizes para KDE)

---

## 7. Resultados — Calibração

### Histórico de versões

| Versão | Amostras | Permutações | Bayes (treino=teste) | **Bayes (5-fold CV)** |
|--------|:--------:|:-----------:|:--------------------:|:---------------------:|
| Smoke v2a | 15/classe | 10 | 95,6% | — |
| Robusto v2a | 100/classe | 200 | 78,3% | **59,0% ± 6,0%** |
| **Robusto v2b** (atual) | 100/classe | 200 | Em andamento | Em andamento |

**Nota sobre interpretação:** o resultado de 78,3% era treino=teste (estimativa otimista).
**A métrica honesta para a dissertação é a CV: 59,0% ± 6,0%.** Ver [07_CROSS_VALIDATION_BAYES.md](07_CROSS_VALIDATION_BAYES.md).

A versão v2b incorpora novos scores CAAFE (P5), bandwidth ótimo via GridSearchCV (P7) e prior informativo opcional (P6). Os resultados da v2b substituirão esta tabela após a conclusão da calibração.

### Thresholds calibrados v2a (referência — Youden's J, 100/classe)

| Score | Threshold | Youden's J | AUC da ROC | Interpretação |
|-------|:---------:|:----------:|:----------:|---------------|
| `little_p` (MCAR vs não) | — | — | ~0.80 | Bom discriminador |
| `pklm_p` | — | — | ~0.70 | Moderado |
| `levene_p` | — | — | ~0.73 | Moderado |
| `auc_obs` (MAR vs não) | — | — | **~0.92** | **Excelente** — feature mais discriminativa |
| `mi_max` (MAR) | — | — | ~0.76 | Bom |
| `caafe_quantile_ratio` (MNAR) | ∞ | 0.000 | **0.500** | **Substituído em v2b** (AUC = chance) |
| `caafe_tail_asym` (MNAR) | ∞ | 0.000 | **0.500** | **Substituído em v2b** (AUC = chance) |
| `caafe_auc_self_delta` (MNAR) | — | — | A medir | Novo em v2b |
| `caafe_kl_density` (MNAR) | — | — | A medir | Novo em v2b |
| `caafe_kurt_excess_abs` (MNAR) | — | — | ~0.54 | Fraco |
| `caafe_cond_entropy` (MNAR) | — | — | ~0.68 | Moderado |

**Insight chave:** `auc_obs` é o score mais forte (AUC ROC ~0.92 para separar MAR). Os scores CAAFE-MNAR individuais eram fracos (2 deles AUC=0.5), mas no Bayes multivariado (10-dim) a combinação compensava. Com os novos scores v2b, espera-se melhora principalmente na recall de MNAR.

---

## 8. Resultados — Dados reais (29 datasets, modo Bayes)

### Confusion matrix

```
             pred_MAR  pred_MCAR  pred_MNAR
true_MAR         5         1          5
true_MCAR        4         2          3
true_MNAR        2         2          5
```

**Accuracy vs rótulo literário: 41,4% (12/29).**

### Datasets confirmados pelo Bayes (alta confiança > 0,9)

| Dataset | Rótulo literário | Predição v2 | Confiança |
|---------|:----------------:|:-----------:|:---------:|
| `mammographic_density` | MAR | MAR | 1.00 |
| `oceanbuoys_airtemp` | MAR | MAR | 1.00 |
| `oceanbuoys_humidity` | MAR | MAR | 1.00 |
| `titanic_age` | MAR | MAR | 1.00 |
| `titanic_age_v2` | MAR | MAR | 1.00 |
| `adult_capitalgain` | MNAR | MNAR | 1.00 |
| `kidney_pot` | MNAR | MNAR | 1.00 |
| `kidney_sod` | MNAR | MNAR | 1.00 |
| `pima_insulin` | MNAR | MNAR | 1.00 |

### Datasets ambíguos (confiança < 0,4)

| Dataset | Rótulo | Predição v2 | Conf | Interpretação |
|---------|:------:|:-----------:|:----:|---------------|
| `echomonths_epss` | MCAR | MCAR | 0.26 | Pode ser MAR fraco |
| `hepatitis_albumin` | MCAR | MCAR | 0.32 | Idem |
| `cylinderbands_esavoltage` | MCAR | MAR | 0.06 | Quase 50/50 MCAR/MAR |
| `colic_resprate` | MAR | MCAR | 0.12 | Evidência de MCAR, não MAR |
| `kidney_hemo` | MAR | MNAR | 0.29 | Pode ser MNAR diffuse |
| `colic_refluxph` | MNAR | MNAR | 0.08 | Quase 50/50 MNAR/MCAR |
| `hepatitis_protime` | MNAR | MCAR | 0.33 | Não detecta MNAR estatisticamente |
| `mroz_wages` | MNAR | MCAR | 0.36 | Heckman selection não detectável por features |

**Interpretação:** os 7 ambíguos são candidatos a sensitivity analysis na dissertação. A baixa confiança é o resultado honesto — melhor que o v1 que forçava "INCONSISTENTE" sem quantificar incerteza.

---

## 9. Comparação v1 vs v2

| Aspecto | v1 | v2 |
|---------|:---:|:---:|
| Testes para MCAR | 1 (Little) | 3 (Little + PKLM + Levene), voto ≥2/3 |
| Evidência MAR | Correlação linear | AUC RF não-linear + MI + permutation p |
| Evidência MNAR | KS tautológico | 4 scores CAAFE thresholdados |
| Reconciliação | Árvore if/else | Bayes via KDE 10-dim |
| Calibração | Nenhuma | Youden's J em 1.200 sintéticos |
| Output | CONSISTENTE/INCONSISTENTE | P(MCAR), P(MAR), P(MNAR) + confiança |
| Accuracy nos sintéticos | ~55% (estimado) | 95,6% (Bayes) |
| Accuracy nos reais | ~30% (8-10/29) | 41,4% (12/29) |
| Quantifica incerteza | Não | Sim (confiança 0–1) |

---

## 10. Limitações do protocolo v2

1. **MNAR fundamentalmente não-identificável.** Por definição, o mecanismo MNAR depende de X0 não-observado. Nenhum teste baseado apenas em X_obs pode distinguir MNAR de MCAR com certeza — o protocolo v2 quantifica a incerteza, não a elimina.

2. **Accuracy honesta: 59% ± 6% (5-fold CV).** O 78,3% (treino=teste) e 95,6% (smoke) são estimativas otimistas. A CV é a métrica correta. Confusão dominante: MCAR ↔ MNAR (46–49% dos erros).

3. **Gap sintético→real (78% → 41%).** Parte é label noise (rótulos literatura são curadoria, não ground truth), parte é distribuição shift entre sintéticos e dados reais.

4. **Scores CAAFE-MNAR individualmente fracos.** Até v2a: 2 de 4 com AUC=0.5. Em v2b (`caafe_auc_self_delta`, `caafe_kl_density`): espera-se melhora, mas ainda limitados pelo problema de não-identificabilidade.

5. **Prior uniforme vs prior de domínio.** Disponível via `--prior-mnar` desde v2b. Para dissertação: reportar sensibilidade do diagnóstico ao prior (P_MNAR = 0.2, 0.35, 0.5).

6. **Custo computacional:** com paralelismo v2b: ~10 min para 300 sintéticos (antes: 9h sequencial). 29 reais: ~5 min.

7. **Benchmark expandido (2026-05-05).** 4 novos datasets MCAR adicionados com evidência publicada, totalizando 33 datasets (13 MCAR, 11 MAR, 9 MNAR):
   - `boys_hc` e `boys_hgt`: `mice::boys` (Fourth Dutch Growth Study). Van Buuren (2018) FIMD Ch. 9 — missing por gaps aleatórios de agendamento clínico.
   - `brandsma_lpr` e `brandsma_apr`: `mice::brandsma` (educação primária holandesa). Alunos ausentes no dia do teste. Evidência quantitativa: correlação mask~ses p=0.72/0.75, mask~iqv p=0.15.
   - **Motivação:** pesquisa exaustiva na literatura mostrou que MCAR confirmado em dados reais é extremamente raro. Planned missingness designs (Graham 2006) são o único MCAR garantido, mas datasets públicos nesse formato são escassos. Os novos datasets são os mais bem-documentados da literatura.
   - **Re-calibração pendente** para incluir os 33 datasets.

Ver também: [09_LIMITACOES_ARTIGO.md](09_LIMITACOES_ARTIGO.md) para formulação específica para o texto da dissertação.
