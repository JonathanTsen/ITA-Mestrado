# Plano de Reescrita da Dissertação — Consolidado

**Data de consolidação:** 2026-04-19
**Arquivo alvo:** `ModeloTesePPGPO/tese.tex` (compilado: 83 paginas, 0 erros)
**Status global:** Etapas 1-6 CONCLUIDAS; Etapas 7-9 parcialmente executadas (pendencias administrativas externas)

> Este arquivo substitui o antigo `plano_reescrita_tese.md` (1312 linhas).
> Conteudo removido durante consolidacao: 14 analises criticas de um plano anterior
> ja corrigido + listas exaustivas de problemas ja resolvidos.
> Historico detalhado da execucao esta em [PENDENCIAS.md](PENDENCIAS.md).

---

## 1. Narrativa Final Adotada

**Tese central:**
> A classificacao automatica de mecanismos de missing data e viavel via pipeline
> hierarquico com features estatisticas especializadas, mas enfrenta um **teto
> teorico fundamental** em dados reais devido a indistinguibilidade MAR/MNAR
> (Rubin, 1976) e ruido nos rotulos (~60% de labels problematicos via Cleanlab).

**Arco narrativo em 5 atos:**

1. **Problema:** Classificar MCAR/MAR/MNAR automaticamente e necessario mas dificil.
   Classificacao direta 3-way atinge apenas **41.36%** em dados reais.
2. **Diagnostico:** MAR e separavel (AUC > 0.8), mas MCAR vs MNAR e quase
   indistinguivel por features estatisticas convencionais.
3. **Solucao:** Pipeline hierarquico (L1: detectar MAR ? routing probabilistico
   ? L2: MCAR vs MNAR com features CAAFE-MNAR especializadas).
4. **Resultado:** **55.97% LOGO CV** (+14.6pp vs direto 3-way), com recall
   balanceado nas 3 classes. Descoberta: **NaiveBayes calibrado supera
   XGBoost+Optuna** — problema e calibracao, nao capacidade.
5. **Limite:** 59.4% rotulos problematicos + Rubin (1976) = teto pratico ~60-65%.
   Estamos a 4-9pp do maximo teorico.

---

## 2. Decisoes Metodologicas Consolidadas

### 2.1 Titulo

**Adotado:**
> "Hierarchical Classification of Missing Data Mechanisms: A Statistical Feature
> Engineering Approach with Real-World Validation"

Pendente de aprovacao do orientador (Prof. Vitor Curtis). Ver
[PENDENCIAS.md](PENDENCIAS.md) Step 8.

### 2.2 Research Questions (reformuladas)

| # | Pergunta | Resposta |
|:-:|----------|----------|
| RQ1 | A decomposicao hierarquica (L1 MAR ? L2 MCAR vs MNAR) melhora a classificacao vs direto 3-way? | **Sim.** +14.6pp LOGO CV (41.4% ? 56.0%) |
| RQ2 | Features CAAFE-MNAR capturam sinais que features convencionais e LLM-derived nao capturam? | **Sim para CAAFE, nao para LLM.** CAAFE rank 2-4 SHAP em real; LLM nao discrimina (Cohen's d < 0.4) |
| RQ3 | Qual o impacto de label noise e como mitigar? | **59.4% labels problematicos.** Sample weighting (Cleanlab) traz +2.7pp em holdout |

### 2.3 Terminologia de Features (CANONICA)

| Nome | Contagem | Composicao |
|------|:--------:|------------|
| **Statistical features** | 21 | 4 stat + 11 discrim + 6 mechdetect |
| **Full pipeline features** | **25** | 21 + 4 CAAFE-MNAR ? pipeline final |
| **LLM-augmented** | 33 | 25 + 8 LLM ? resultado negativo em real |

### 2.4 Numeros-chave verificados (fonte: CSVs em `Output/v2_improved/`)

**Sintetico:**

| Metrica | Valor | Modelo |
|---------|:-----:|:------:|
| Baseline 21f | 76.67% | MLP |
| Com CAAFE 25f | 77.67% | RF |
| Com LLM 33f | **79.33%** | RF |
| LLM delta vs 21f | **+3.33pp** | RF |

**Real:**

| Metrica | Valor | Modelo |
|---------|:-----:|:------:|
| Direto 3-way | 41.36% | RF |
| V3 hard LOGO CV | 51.42% | NaiveBayes |
| **V3+ soft3zone LOGO CV** | **55.97%** | NaiveBayes |
| V3+ threshold holdout | 53.22% | GBT+Cleanlab |
| L1 accuracy | 82.0-82.7% (GBT/RF), 61% (NB) | varia |
| L2 accuracy (melhor) | 56.5% | GBT+threshold+weights |
| MNAR recall | 46.0% | GBT+threshold+weights |
| F1 macro | 0.515 | GBT+threshold+weights |
| Cleanlab issues | **59.36%** (672/1132) | — |
| McNemar significante | 3/7 (p<0.05) | LogReg, SVM, GBT |

**Contagens:**

| Item | Valor |
|------|:-----:|
| Variantes sinteticas | 12 (3 MCAR + 5 MAR + 4 MNAR) |
| Datasets sinteticos | 1,200 |
| Datasets reais | 23 (5 MCAR + 11 MAR + 7 MNAR) |
| Bootstraps reais | 1,132 |

### 2.5 Resultados Negativos Documentados

Quatro resultados negativos importantes integrados como contribuicoes:

1. **XGBoost/CatBoost + Optuna:** sem ganho (38.7%, 36.6% LOGO CV vs NB 51.4%).
   Modelo simples calibrado > modelo complexo otimizado.
2. **SMOTE variants:** SMOTE regular domina; SMOTE-ENN remove amostras uteis
   em N pequeno.
3. **LLM features no L2:** 8 features com Cohen's d < 0.4, medianas identicas
   entre classes, multicolinearidade 6 pares |r| > 0.5 ? nao discriminam MAR vs MNAR.
4. **Features avancadas L2:** -2pp accuracy, MNAR recall ? 0% (ruido > sinal).

---

## 3. Estrutura Final dos Capitulos

| Cap | Nome | Status | Arquivos |
|:---:|------|:------:|----------|
| 1 | Introduction | REESCRITO | `tese.tex` + secoes RQ/objetivos |
| 2 | Theoretical Foundations | MODIFICADO | `Cap2/cap2.tex` (+7 secoes novas) |
| 3 | Methodology | REESCRITO | `Cap3/cap3.tex` |
| 4 | Results | REESCRITO | `Cap4/cap4.tex` |
| 5 | Conclusion | REESCRITO | `tese.tex` |
| Ap B | Real Datasets Catalog | CRIADO | `ApeB/apendiceB.tex` |

### 3.1 Cap 1 — Introduction (reescrito ~70%)

- Motivacao: prevalencia de missing data, Rubin 1976, impacto pratico da escolha do mecanismo
- Related Work: tabela 7 colunas com 6 trabalhos (Little, Jamshidian, Rouzinov, PKLM, MechDetect, CAAFE, Cleanlab)
- 3 posicionamentos "first to..." (hierarquica + real + label noise)
- 3 RQs reformuladas (secao 2.2 acima)
- 6 objetivos especificos
- Removido: discussao de time series e deep learning (contexto excessivo)

### 3.2 Cap 2 — Theoretical Foundations (+7 secoes, ~117 linhas novas)

Secoes novas adicionadas em `Cap2/cap2.tex`:

- **Rubin's Impossibility Result** (CRITICO para narrativa — teto teorico)
- **Hierarchical Classification** (Silla & Freitas 2011, 4 routing strategies)
- **Oversampling and Class Imbalance** (SMOTE + 3 variantes)
- **Confident Learning and Label Noise** (Northcutt 2021, confident joint, 3 estrategias)
- **Hyperparameter Optimization** (Optuna, TPE, GroupKFold)
- **SHAP Values and Interpretability** (Lundberg & Lee 2017)
- **Probability Calibration** (motiva por que NaiveBayes domina)

Secao de series temporais reduzida drasticamente (54 ? 10 linhas).

### 3.3 Cap 3 — Methodology (reescrito do zero)

- Pipeline Overview (2 fases: sintetica + real hierarquica)
- Gerador v2 (12 variantes × 4 distribuicoes base)
- Curadoria de dados reais (23 datasets, bootstrap, 3 testes de validacao de rotulos)
- Feature extraction (21 baseline + 4 CAAFE + 8 LLM como experimento)
- Pipeline hierarquico (L1 + routing × 4 + L2)
- Label noise mitigation (Cleanlab diagnostico + sample weighting)
- Avaliacao (7+2 classifiers, LOGO CV com 23 folds, McNemar, SHAP)

### 3.4 Cap 4 — Results (reescrito, organizado por RQ)

| Secao | Conteudo |
|-------|----------|
| 4.1 | Sintetico baseline (21f/25f/33f, LLM positivo em sintetico) |
| 4.2 | Real direto (41.36% RF, gap -36pp) |
| 4.3 | Hierarquico V3+ (55.97% LOGO NB, +14.6pp, McNemar) |
| 4.4 | Features (ablacao E1-E5, SHAP por nivel, CAAFE vs LLM) |
| 4.5 | Label noise (59.4%, confident joint, impacto pesos) |
| 4.6 | Resultados negativos (XGBoost, SMOTE, LLM L2, features ADV, PKLM) |
| 4.7 | Comparativo (evolucao Stage 0-6, gap por classe, head-to-head, teto teorico) |
| 4.8 | Discussao (6 temas) |

### 3.5 Cap 5 — Conclusion (reescrito)

- Sintese por RQ com valores exatos
- 5 contribuicoes (coerentes com resultados)
- 6 limitacoes (honestas, numeradas)
- 5 trabalhos futuros

### 3.6 Apendice B — Real Datasets Catalog (criado)

Integrado ao PDF (pagina 71). 4 secoes + descricoes por dataset:

1. Dataset Sources and Characteristics (`catalog_sources`, 23 datasets)
2. Label Validation Results (`catalog_validation`, Little's p + correlacao + KS)
3. Cleanlab Quality Scores (`catalog_cleanlab`)
4. MNAR Subtype Classification (`catalog_mnar_subtype` — todos Diffuse)
5. Individual Dataset Descriptions (paragrafo domain-specific por dataset)

Fontes: `validar_rotulos.py`, `classificar_mnar.py`, `clean_labels_summary.json`.

---

## 4. Registro de Execucao

### 2026-04-19 — Reescrita Completa (Etapas 1-6)

**Etapa 1 — Capitulo 2 (Fundamentacao):**
7 novas secoes em `Cap2/cap2.tex`; ~15 referencias novas em `referencias.bib`.

**Etapa 2 — Capitulo 3 (Metodologia):**
Conteudo antigo removido de `tese.tex`; `Cap3/cap3.tex` reescrito do zero.

**Etapa 3 — Capitulo 4 (Resultados):**
Conteudo antigo removido; `Cap4/cap4.tex` reescrito do zero, organizado por RQ.

**Etapa 4 — Capitulo 1 (Introducao):**
Motivacao reescrita; related work expandido (tabela 7 colunas); 3 RQs reformuladas;
6 objetivos; estrutura atualizada.

**Etapa 5 — Capitulo 5 (Conclusao):**
Sintese por RQ; 5 contribuicoes; 6 limitacoes; 5 trabalhos futuros.

**Etapa 6 — Elementos Globais:**
Titulo atualizado (hierarquico, nao LLM); abstract reescrito (~250 palavras);
resumo em portugues atualizado; keywords atualizadas (5 termos); ~15 referencias novas.

### 2026-04-19 — Figuras e Compilacao (Steps 3, 5, 6)

12 figuras geradas via `Scripts/v2_improved/gerar_figuras_tese.py` em
`ModeloTesePPGPO/figuras/`. Principais:

- `fluxograma_pipeline.png` (Cap 3)
- `confusion_matrices_comparison.png`, `confusion_v3plus.png` (Cap 4.3)
- `shap_3way.png`, `shap_l1_vs_l2.png` (Cap 4.4)
- `evolucao_accuracy.png` (Cap 4.7 — Stage 0?6)
- `label_quality_distribution.png`, `confident_joint_heatmap.png` (Cap 4.5)
- `gap_sintetico_vs_real.png`, `cohens_d_caafe_vs_llm.png` (Cap 4)
- `logo_cv_models.png` (NB vs XGBoost)

Compilacao LaTeX (`pdflatex + bibtex + pdflatex x3`):
83 paginas, 0 erros, 0 refs/cites undefined.

### 2026-04-19 — Auditoria de Coerencia (Step 9)

Verificado:
- Todos os numeros do Cap 1 batem com Cap 4 (auditados: 55.97%, 41.36%, +14.6pp,
  +10.1pp, +4.6pp, +3.33pp, 76.00?79.33%, d=0.84, 59.4%, +2.7pp, 672/1132)
- Todas as 3 RQs respondidas explicitamente no Cap 5
- 6 objetivos abordados em Cap 3 e 4
- 5 contribuicoes suportadas por evidencia no Cap 4
- 6 limitacoes coerentes com Cap 3/4/Apendice B
- Abstract (56.0%) e Resumo (56,0%) consistentes

Correcoes aplicadas:
- Padronizacao "Naive Bayes" ? "Na\\\"{\\i}ve Bayes" em `tese.tex:506,508,534`
- `tab:routing_strategies`: 56.0% ? 55.97% (consistencia precisao)
- `Cap4/cap4.tex:117`: "V1 vs V4" ? "direct three-way vs hierarchical" (nomes corrigidos)
- `tab:head_to_head`: 41.4% ? 41.36%, 56.0% ? 55.97% (precisao)

---

## 5. Pendencias

Ver [PENDENCIAS.md](PENDENCIAS.md) para o detalhamento atual. Resumo:

| Prioridade | Item | Status |
|:----------:|------|:------:|
| BAIXA | Step 7 — placeholders administrativos (banca, Pro-Reitor, datas FRD, dedicatoria) | Pendente (inputs externos) |
| BAIXA | Step 8 — aprovacao do titulo com orientador | Pendente (inputs externos) |

Todos os steps criticos (1-6, 9) concluidos. Tese compilada com sucesso.

---

## 6. Referencias a Outros Documentos

- [PENDENCIAS.md](PENDENCIAS.md) — Status atual dos 9 steps da reescrita
- [../../HISTORICO.md](../../HISTORICO.md) — Linha do tempo completa do projeto
- [../03_plano2_paper_hierarquico/ACHADOS_CONSOLIDADOS.md](../03_plano2_paper_hierarquico/ACHADOS_CONSOLIDADOS.md) — Base experimental consolidada
- [../04_plano3_otimizacao_v3/VISAO_GERAL.md](../04_plano3_otimizacao_v3/VISAO_GERAL.md) — Otimizacoes finais
