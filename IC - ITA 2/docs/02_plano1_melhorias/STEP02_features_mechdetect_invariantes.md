# STEP 02: Features MechDetect + Invariantes

**Fase 4B — Resolver a causa raiz: features que identificam o mecanismo, nao o dataset**
**Status: CONCLUIDO (2026-04-12)**

---

## Problema

As features atuais de `statistical.py` (X0_mean, X0_q25, X0_q50, X0_q75) representam 78.7% da importancia no RF, mas sao **fingerprints do dataset** — identificam de qual dataset veio a amostra, nao qual mecanismo de missing esta ativo. MNAR tem 0% de recall porque nenhuma feature captura dependencia do proprio X0.

---

## Parte A: Implementar Features MechDetect

### Fonte
Paper: MechDetect (Jung et al., 2024) — 89.14% accuracy em 101 datasets reais.

### Logica

Criar novo modulo `features/mechdetect.py` com funcao que recebe um DataFrame e retorna features baseadas em 3 tarefas de classificacao:

**Tarefa 1 — Complete:** Treinar classificador (HistGradientBoosting) usando X inteiro (X0 observado + X1-X4) para prever a mascara de missing de X0. Calcular AUC-ROC via 10-fold CV. Se AUC >> 0.5, o missing depende dos dados (nao e MCAR).

**Tarefa 2 — Shuffled:** Mesmo que Complete, mas o target e a mascara permutada aleatoriamente. O AUC deve ser ~0.5 (baseline). Serve como referencia.

**Tarefa 3 — Excluded:** Treinar classificador usando apenas X1-X4 (sem X0) para prever a mascara. Se AUC alto, o missing depende das variaveis observadas (MAR). Se AUC baixo comparado com Complete, o missing depende de X0 (MNAR).

**Features resultantes (6):**
| Feature | Significado |
|---------|------------|
| `mechdetect_auc_complete` | AUC media da tarefa Complete |
| `mechdetect_auc_shuffled` | AUC media da tarefa Shuffled (baseline ~0.5) |
| `mechdetect_auc_excluded` | AUC media da tarefa Excluded |
| `mechdetect_delta_complete_shuffled` | Complete - Shuffled (sinal de nao-MCAR) |
| `mechdetect_delta_complete_excluded` | Complete - Excluded (sinal de MNAR) |
| `mechdetect_mwu_pvalue` | p-valor do Mann-Whitney entre Complete e Shuffled |

**Por que funciona:** Estas features sao invariantes a distribuicao de X0 porque medem a *capacidade preditiva* do missing, nao a *distribuicao* dos valores.

### Decisao de design

MechDetect original usa 10-fold CV para gerar 10 AUC-ROC por tarefa. Com nossos bootstraps de 100 linhas e 1-10% missing, pode haver poucos missing para 10 folds. Usar 5-fold ou stratified split adaptativo.

### Decisao de implementacao (2026-04-12)

- **Modelo**: LogisticRegression em vez de HistGradientBoostingClassifier. HistGBT levava ~1.3s/arquivo (65 min para 3000 arquivos), LogReg leva ~0.017s/arquivo (< 1 min total). O sinal discriminativo esta nos deltas entre tarefas, nao no AUC absoluto.
- **Folds**: 3-fold adaptativo (minimo 2). Suficiente para dados pequenos.
- **Passada unica**: Complete, Shuffled e Excluded treinados nos mesmos folds em uma unica passada de CV, evitando treino duplicado para o Mann-Whitney.

---

## Parte B: Substituir Features Estatisticas Absolutas

### Em `features/statistical.py`

**Remover:** `X0_mean`, `X0_q25`, `X0_q50`, `X0_q75`

**Adicionar features invariantes:**

| Feature | Logica | Por que detecta mecanismo |
|---------|--------|--------------------------|
| `X0_missing_rate` | Proporcao de NaN em X0 | Basico mas informativo; varia entre mecanismos |
| `X0_obs_vs_full_ratio` | Imputar X0 (mediana/KNN), calcular ratio media_observados / media_imputados | MNAR: observados tem media diferente do esperado (ex: so valores baixos ficam). MCAR: ratio ~1.0 |
| `X0_iqr_ratio` | IQR dos observados / IQR dos imputados | MNAR: missing nas caudas comprime o IQR dos observados. MCAR: ratio ~1.0 |
| `X0_obs_skew_diff` | skew(observados) - skew(imputados) | MNAR: missing assimetrico altera skew. MCAR: diff ~0 |

**Nota sobre imputacao:** Para calcular "full" ou "imputados", usar imputacao simples (mediana) temporariamente. Nao e imputacao para uso downstream — e apenas para comparar distribuicoes observadas vs esperadas.

### Em `features/discriminative.py`

**Manter todas as 6 features atuais** (auc_mask_from_Xobs, coef_X1_abs, log_pval_X1_mask, X1_mean_diff, X1_mannwhitney_pval, little_proxy_score).

**Adicionar 5 features para detectar MNAR:**

| Feature | Logica | Detecta |
|---------|--------|---------|
| `X0_ks_obs_vs_imputed` | Teste KS entre X0 observado e X0 imputado (mediana). Retorna estatistica KS | MNAR: distribuicoes divergem significativamente |
| `X0_tail_missing_ratio` | Dividir X0 imputado em quartis. Calcular taxa de missing no Q4 (cauda) / taxa no Q2 (centro) | MNAR: missing concentrado nas caudas → ratio >> 1. MCAR: ratio ~1 |
| `mask_entropy` | Calcular entropia de Shannon dos comprimentos de runs consecutivos de missing/nao-missing | MCAR: alta entropia (padrao aleatorio). MNAR: baixa entropia (missing em blocos ou concentrado) |
| `X0_censoring_score` | Correlacao de Spearman entre rank(X0_imputado) e mascara de missing | MNAR: alta correlacao (valores extremos sao missing). MCAR: correlacao ~0 |
| `X0_mean_shift_X1_to_X4` | Media do valor absoluto de (media_Xi_quando_missing - media_Xi_quando_observado) para i=1..4 | MAR: shift alto (missing depende de Xi). MCAR e MNAR: shift baixo |

### Contagem final de features

| Grupo | Antes | Depois |
|-------|:-----:|:------:|
| Estatisticas (statistical.py) | 4 | 4 |
| Discriminativas (discriminative.py) | 6 | 11 |
| MechDetect (novo) | 0 | 6 |
| **Total baseline** | **10** | **21** |
| LLM (se ativo) | 8 | 8 |
| **Total com LLM** | **18** | **29** |

---

## Parte C: Atualizar `extract_features.py`

- Importar e chamar o novo modulo `features/mechdetect.py`
- Atualizar a lista de features no checkpoint system
- Manter imputacao: NaN → 0 para stat/disc, NaN → mediana para LLM, NaN → 0 para MechDetect (se falhar por falta de missing)

---

## Testes de Validacao

### Teste 1: Sanidade no sintetico atual
Rodar pipeline sintetico com features novas (sem LLM). Comparar accuracy com o baseline antigo. As features invariantes devem manter ou melhorar accuracy porque os dados sinteticos tem distribuicao uniforme — as features antigas e novas capturam sinais similares nesse caso.

### Teste 2: Sanidade no sintetico com GroupShuffleSplit
Gerar 30 datasets sinteticos (10 por mecanismo) com distribuicoes nao-uniformes (normal, exponencial). Aplicar GroupShuffleSplit. As features invariantes devem ter accuracy significativamente maior que as features antigas, porque as antigas falham quando a distribuicao muda.

### Teste 3: Feature importance redistribuida
No RF, as features MechDetect e invariantes devem ter importancia mais distribuida. X0_mean/q25/q50/q75 nao devem mais dominar (>70%). Se mechdetect_delta_complete_excluded tiver importancia alta, e sinal de que a deteccao MNAR melhorou.

### Teste 4: Recall de MNAR
Com as novas features, MNAR recall deve sair de 0% para algo positivo (meta: >20% no sintetico com GroupShuffleSplit). Features como X0_ks_obs_vs_imputed, X0_tail_missing_ratio e X0_censoring_score sao desenhadas para isso.

### Teste 5: Invariancia ao dataset
Calcular as novas features para 2 datasets com distribuicoes muito diferentes (ex: uniform [0,1] vs exponential). Se o mecanismo for MCAR em ambos, os valores das features invariantes devem ser similares (ratio ~1, skew_diff ~0, censoring ~0). As features antigas (X0_mean) seriam completamente diferentes.

---

## Criterio de Conclusao

- [x] `features/mechdetect.py` criado e integrado ao pipeline
- [x] `statistical.py` sem features absolutas, com 4 invariantes
- [x] `discriminative.py` com 5 novas features MNAR
- [x] Total de 21 features baseline (vs 10 anteriores)
- [ ] Testes 2-5 pendentes (requerem STEP03 com mais datasets)
- [ ] MNAR recall > 0% no sintetico com GroupShuffleSplit (pendente)

---

## Resultados (2026-04-12) — Experimento `step02_eval`

### Dados sinteticos — Apenas ML (baseline)

| Modelo | Fase 3 (10 feat) | Step02 (21 feat) | Melhoria |
|--------|:---:|:---:|:---:|
| LogisticRegression | 69.9% | **87.2%** | +17.3% |
| RandomForest | 64.8% | **87.2%** | +22.4% |
| SVM_RBF | 68.5% | **86.1%** | +17.6% |
| GradientBoosting | 64.5% | 85.3% | +20.8% |
| NaiveBayes | 69.9% | 85.1% | +15.2% |
| KNN | 64.8% | 84.3% | +19.5% |
| MLP | 65.6% | 82.4% | +16.8% |

CV estabilidade: desvio padrao de ±4.3% (vs ±5-7% antes).

### Dados sinteticos — ML + LLM (gemini-3-flash-preview)

| Modelo | Apenas ML (21 feat) | ML + LLM (29 feat) | Diferenca |
|--------|:---:|:---:|:---:|
| LogisticRegression | **87.2%** | 66.9% | **-20.3%** |
| RandomForest | **87.2%** | 63.5% | **-23.7%** |
| SVM_RBF | **86.1%** | 64.3% | **-21.8%** |
| GradientBoosting | 85.3% | 61.6% | -23.7% |
| NaiveBayes | 85.1% | 59.7% | -25.4% |
| KNN | 84.3% | 57.9% | -26.4% |
| MLP | 82.4% | 60.8% | -21.6% |

### Conclusoes

1. **Features invariantes + MechDetect funcionam**: ganho de ~17-22pp em accuracy sem LLM
2. **LLM prejudica significativamente**: -20 a -26pp. As 8 features LLM atuais adicionam ruido que confunde os classificadores. Confirma diagnostico de que o prompt gera classificacao redundante, nao features novas
3. **STEP04 (reformulacao LLM) e critico**: sem reformular o LLM, a tese nao consegue mostrar contribuicao positiva do LLM
4. **Dados reais pendentes**: rodar para validar se o ganho se mantem fora do sintetico
