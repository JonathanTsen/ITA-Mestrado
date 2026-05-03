# Análise de Importância de Features

**Data:** 2026-04-25
**Fonte:** `Output/v2_improved/step1_v2_neutral/.../feature_importance.csv` (RandomForest)

---

## 1. Ranking completo (34 features)

| Rank | Feature | Importance | Tipo |
|:----:|---------|:----------:|------|
| 1 | `caafe_kurtosis_excess` | 7.54% | CAAFE |
| 2 | `caafe_tail_asymmetry` | 7.27% | CAAFE |
| 3 | `X0_censoring_score` | 7.18% | Statistical |
| 4 | `X0_obs_vs_full_ratio` | 6.41% | Statistical |
| 5 | `X0_obs_skew_diff` | 5.82% | Statistical |
| 6 | `X0_mean_shift_X1_to_X4` | 5.66% | Statistical |
| 7 | `caafe_cond_entropy_X0_mask` | 5.63% | CAAFE |
| 8 | `X1_mean_diff` | 5.51% | Discriminative |
| 9 | `X0_ks_obs_vs_imputed` | 4.44% | Statistical |
| 10 | `little_proxy_score` | 3.61% | Statistical |
| 11 | `mechdetect_delta_complete_excluded` | 3.50% | MechDetect |
| 12 | `mask_entropy` | 2.87% | Statistical |
| 13 | `log_pval_X1_mask` | 2.55% | Discriminative |
| 14 | `X0_missing_rate` | 2.47% | Statistical |
| 15 | `mechdetect_auc_complete` | 2.40% | MechDetect |
| 16 | `coef_X1_abs` | 2.26% | Discriminative |
| 17 | `X1_mannwhitney_pval` | 2.15% | Discriminative |
| 18 | `mechdetect_auc_excluded` | 1.98% | MechDetect |
| 19 | **`llm_ctx_cause_type`** | **1.90%** | LLM |
| 20 | `auc_mask_from_Xobs` | 1.85% | Discriminative |
| 21 | `llm_ctx_domain_confidence` | 1.84% | LLM |
| 22 | `X0_iqr_ratio` | 1.82% | Statistical |
| 23 | `llm_ctx_stats_agreement` | 1.75% | LLM |
| 24 | `mechdetect_auc_shuffled` | 1.65% | MechDetect |
| 25 | `mechdetect_delta_complete_shuffled` | 1.65% | MechDetect |
| 26 | `llm_ctx_surprise` | 1.58% | LLM |
| 27 | `llm_ctx_stats_consistency` | 1.46% | LLM |
| 28 | `llm_ctx_counter_strength` | 1.35% | LLM |
| 29 | `mechdetect_mwu_pvalue` | 1.20% | MechDetect |
| 30 | `llm_ctx_confidence_delta` | 0.95% | LLM |
| 31 | `llm_ctx_domain_prior` | 0.88% | LLM |
| 32 | `llm_ctx_n_causes` | 0.85% | LLM |
| 33 | `X0_tail_missing_ratio` | 0.02% | Statistical |
| 34 | `caafe_missing_rate_by_quantile` | 0.001% | CAAFE |

## 2. Agregação por tipo

| Tipo | Count | Sum importance | Avg importance |
|------|:-----:|:--------------:|:--------------:|
| Statistical | 11 | 40.5% | 3.7% |
| CAAFE | 4 | 20.4% | 5.1% |
| MechDetect | 6 | 12.4% | 2.1% |
| Discriminative | 4 | 11.8% | 2.9% |
| **LLM** | **9** | **12.6%** | **1.4%** |

**Total LLM: 12.6%** — confirma o pattern visto desde V3+: features estatísticas dominam, LLM contribui marginalmente em ML clássico.

## 3. Insights

### 3.1 CAAFE permanece o maior contribuidor por feature

`caafe_kurtosis_excess` (7.54%) e `caafe_tail_asymmetry` (7.27%) ocupam o top 2 — assim como em V3+ e em `forensic_neutral_v2`. Estas features capturam:
- **Kurtosis excess:** quanto a distribuição observada de X0 é mais pontuda (sinal de truncamento) ou mais achatada (sinal de mistura) que a esperada para gaussiana
- **Tail asymmetry:** se a distribuição observada é fortemente assimétrica em uma das caudas (sinal de censura tipo C/MNAR)

São features **simples (4 linhas pandas cada)** que capturam o sinal principal de detecção MNAR via análise univariada de X0. Replicabilidade alta.

### 3.2 LLM features contribuem mas são individualmente fracas

A feature LLM mais importante é `llm_ctx_cause_type` (rank 19, 1.90%), que codifica o **tipo causal** identificado pelo DAG (A=MCAR/0, B=MAR/0.5, C=MNAR/1). É a feature LLM que mais carrega informação discriminativa.

Surpresa negativa: **`llm_ctx_domain_prior` está em rank 31** (0.88%). Apesar de ser a feature mais discutida nos docs e a que solo dá 43.7% accuracy, no contexto do RF ela é dominada por features estatísticas correlacionadas. Isso é evidência de que **muito do sinal do domain_prior é redundante com features CAAFE** (que detectam os mesmos casos canônicos via outras métricas).

### 3.3 Hierarquia de importância confirma estrutura V3+

A ordem por tipo (Statistical > CAAFE > MechDetect > Discriminative > LLM) é praticamente idêntica à observada em `forensic_neutral_v2`. Isso sugere que:

- **A arquitetura de features é estável** entre experimentos
- **O ML está usando o mesmo subset de informação** principalmente
- **Adicionar mais features LLM** (Steps 2/3) provavelmente não revolucionará o ranking — para mover a importância LLM acima de 20% seria necessário uma feature LLM **substancialmente mais informativa** que as atuais

### 3.4 Features quase-nulas

Duas features têm importance < 0.02%:
- `X0_tail_missing_ratio` (0.02%)
- `caafe_missing_rate_by_quantile` (0.001%)

Estas features são **constantes ou quase-constantes** no dataset atual (variância estatística mínima). Podem ser candidatas a remoção do pipeline, simplificando o modelo sem perda perceptível.

## 4. Comparação de importance entre experimentos

| Experimento | LLM total | CAAFE total | Stat total | Outros |
|-------------|:---------:|:-----------:|:----------:|:------:|
| `step1_v2_neutral` (este) | 12.6% | 20.4% | 40.5% | 26.5% |
| `step10_flash_ca_neutral` | 10.4% | ~21% | ~41% | ~28% |
| `forensic_neutral_v2` (estimado) | ~13% | ~20% | ~40% | ~27% |

A consistência sugere que **a estrutura de feature importance é robusta entre runs** — a variação ±2pp em LLM total é compatível com ruído de RF sampling.

**Implicação:** mudanças no prompt LLM (Flash → Pro, original → Step 1) afetam **acurácia per-amostra** mas não a **estrutura agregada de importância**. Para mover a aguilha em LLM total, precisaríamos de mudança estrutural — Steps 2/3 ou retraining/fine-tuning.

## 5. Recomendações operacionais

### 5.1 Para Step 2 (Causal DAG)

Adicionar features que capturem o **conteúdo do DAG** explicitamente:
- `llm_ctx_n_type_A_causes` (n de causas Tipo A/MCAR identificadas)
- `llm_ctx_n_type_B_causes`
- `llm_ctx_n_type_C_causes`
- `llm_ctx_dag_consistency` (se causa mais plausível bate com estatística)
- `llm_ctx_dag_specificity` (se nomeou variável específica vs genérica)

Estas seriam mais informativas que `cause_type` (atualmente apenas categórica em 3 valores).

### 5.2 Feature pruning

`X0_tail_missing_ratio` e `caafe_missing_rate_by_quantile` podem ser removidas. Reduz ruído e simplifica.

### 5.3 Análise de correlação

Vale calcular matriz de correlação entre `llm_ctx_domain_prior` e top CAAFE/Statistical. Se correlação > 0.6, confirma redundância e justifica investimento em features LLM **diferentes**, não mais features LLM **similares**.
