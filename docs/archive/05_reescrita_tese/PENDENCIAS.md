# Pendências da Reescrita da Tese

**Data:** 2026-04-19
**Última atualização:** 2026-04-19 (pós-auditoria de coerência)
**Status:** 7 de 9 steps concluídos. Tese compilada (83 páginas, 0 erros, 0 undefined refs), auditoria de coerência aplicada. Restam apenas Steps 7 (placeholders administrativos) e 8 (título com orientador), ambos dependentes de inputs externos.

---

## ~~Step 1: Tabela dos 23 Datasets Reais (Cap 3)~~ ✅ CONCLUÍDO

**Arquivo:** `ModeloTesePPGPO/Cap3/cap3.tex` (linhas 72-116)
**Data conclusão:** 2026-04-19

Tabela inserida após Seção 3.3.1 com 7 colunas (#, Dataset, Variable X0, Source, N, Missing Rate, Bootstraps), agrupada por mecanismo (MCAR/MAR/MNAR). Dados extraídos de `real_datasets_metadata.json`. Totais: 54.638 observações originais, 1.132 bootstraps. Referência `\ref{tab:real_datasets}` adicionada ao texto.

**Nota:** oceanbuoys já consta na pasta MAR nos dados atuais — reclassificação já aplicada, asterisco desnecessário.

---

## ~~Step 2: Apêndice B — Catálogo Completo dos Datasets Reais~~ ✅ CONCLUÍDO

**Arquivo:** `ModeloTesePPGPO/ApeB/apendiceB.tex` (243 linhas)
**Registrado em:** `tese.tex` linhas 653-655 (entre Apêndice A e `\bibliography`)
**Data conclusão:** 2026-04-19

Apêndice B integrado ao PDF (página 71, label `app:real_datasets`) com 4 seções:

1. **Dataset Sources and Characteristics** — tabela `catalog_sources` (23 datasets, agrupados MCAR/MAR/MNAR, com source, X0, N, missing rate)
2. **Label Validation Results** — tabela `catalog_validation` (Little's p, max |r| correlação, KS p, diagnóstico C/I)
3. **Cleanlab Quality Scores** — tabela `catalog_cleanlab` (avg quality, issues %, agreement, suggested label)
4. **MNAR Subtype Classification** — tabela `catalog_mnar_subtype` (AUC complete/excluded, Δ, Focused/Diffuse — todos Diffuse)
5. **Individual Dataset Descriptions** — parágrafo domínio-específico por dataset

**Fonte dos dados utilizada:**
- `Scripts/v2_improved/validar_rotulos.py --data real`
- `Scripts/v2_improved/classificar_mnar.py --data real`
- `Scripts/v2_improved/clean_labels_summary.json`

---

## ~~Step 3: Gerar Figuras Novas~~ ✅ CONCLUÍDO

**Data conclusão:** 2026-04-19

12 figuras geradas via `Scripts/v2_improved/gerar_figuras_tese.py` e salvas em `ModeloTesePPGPO/figuras/`. **Nomes finais consolidados** (diferem da lista original; estes são os que de fato existem no disco e são referenciados em `cap4.tex`):

- **3a.** `fluxograma_pipeline.png` — Fluxograma do pipeline hierárquico (Cap 3)
- **3b.** `confusion_matrices_comparison.png` (4 CMs em grid), `confusion_v3plus.png` (CM detalhada V3+) — consolidadas em vez de 4 arquivos separados
- **3c.** `shap_3way.png` (top 15 overall), `shap_l1_vs_l2.png` — SHAP summary
- **3d.** `evolucao_accuracy.png` — Stage 0→6
- **3e.** `label_quality_distribution.png` — histograma Cleanlab
- **3f.** `confident_joint_heatmap.png` — heatmap 3×3
- **3g.** `gap_sintetico_vs_real.png` — gap por classe
- **3h.** `cohens_d_caafe_vs_llm.png` — Cohen's d CAAFE vs LLM
- **3i.** `logo_cv_models.png` — NB vs XGBoost LOGO CV
- Extra: `quality_distribution_existing.png` (pré-existente, não referenciada)

---

## ~~Step 4: Decidir sobre Figuras v1~~ ✅ CONCLUÍDO

**Data conclusão:** 2026-04-19

Opção 2 aplicada: 10 figuras v1 removidas (`MCAR_Estatistica.png`, `MAR_Estatistica.png`, `MNAR_Estatistica.png`, `Media_Estatistica.png`, `MCAR_LLM.png`, `MAR_LLM.png`, `MNAR_LLM.png`, `Media_LLM.png`, `Sample.png`, `Ganhos.png`). Nenhuma era referenciada no LaTeX reescrito. Apenas `Fluxograma.png` permanece (será substituído no Step 3a).

---

## ~~Step 5: Referências de Figuras no Cap 4~~ ✅ CONCLUÍDO

**Data conclusão:** 2026-04-19

10 figuras inseridas em `ModeloTesePPGPO/Cap4/cap4.tex` com `\begin{figure}[H]...\end{figure}`, `\caption`, `\label` e referências cruzadas `\ref`:

- **Seção 4.1:** Referência `\ref{fig:cm_comparison}` adicionada ao texto da confusion matrix sintética
- **Seção 4.3:** `confusion_matrices_comparison.png` (4 CMs comparativas), `confusion_v3plus.png` (CM detalhada V3+)
- **Seção 4.3.4:** `logo_cv_models.png` (NB vs XGBoost LOGO CV)
- **Seção 4.4:** `shap_3way.png` (top 15 overall), `shap_l1_vs_l2.png` (L1 vs L2), `cohens_d_caafe_vs_llm.png` (Cohen's d)
- **Seção 4.5:** `label_quality_distribution.png` (histograma Cleanlab), `confident_joint_heatmap.png` (heatmap 3×3)
- **Seção 4.7:** `evolucao_accuracy.png` (Stage 0→6), `gap_sintetico_vs_real.png` (gap por classe)

---

## ~~Step 6: Compilar LaTeX e Corrigir Erros~~ ✅ CONCLUÍDO

**Data conclusão:** 2026-04-19

TeX Live 2026 instalado em `~/texlive/2026/`. Compilação: `pdflatex` + `bibtex` + `pdflatex` ×3. Resultado: 83 páginas, 0 erros, 0 refs/cites undefined. Correção aplicada: `Cap3/cap3.tex` atualizado de `Fluxograma.png` para `figuras/fluxograma_pipeline.png`. Warnings residuais cosméticos: font shape `TS1/aer/m/n` e lastpage/hyperref rerun loop (não afetam o PDF).

Recompilação pós-Step 9 (2026-04-19 16:55): mesmos 83 páginas, 0 erros, 0 undefined refs.

Comando usado:

```bash
export PATH="$HOME/texlive/2026/bin/universal-darwin:$PATH"
cd ModeloTesePPGPO
pdflatex -interaction=nonstopmode tese.tex
bibtex tese
pdflatex -interaction=nonstopmode tese.tex
pdflatex -interaction=nonstopmode tese.tex
```

---

## Step 7: Verificar Formatação ITA ⏳ PARCIAL

**Itens já OK (verificados no PDF compilado):**
- [x] `\onehalfspacing` aplicado em `tese.tex:57`
- [x] `ita.cls` e `abnt-alf.bst` em uso (formato ABNT)
- [x] Numeração de capítulos/seções, sumário, cabeçalhos de tabelas/figuras OK
- [x] Folha de rosto compila com novo título

**Placeholders administrativos pendentes em `tese.tex`** (dependem de inputs externos — secretaria/orientador):
- [ ] Linha 87: `\boss{Prof.~Dr.}{Nome}` — nome do Pró-Reitor da Pós-Graduação
- [ ] Linha 93: `\bosscourse{Prof.~Dr.}{Nome}` — nome do coordenador do curso
- [ ] Linhas 105-106: `\examiner{Prof. Dr.}{Nome}{}{UXXX}` × 2 — banca examinadora
- [ ] Linha 113: `\date{XX}{MONTH}{2026}` — data da defesa
- [ ] Linha 129: `Aos amigos ...` — dedicatória pessoal
- [ ] Linha 668: `\FRDitadata{XX de XXXX de 2026}` — data FRD
- [ ] Linha 669: `\FRDitadocnro{DCTA/ITA/DM-018/2017}` — número de registro (solicitar à biblioteca)
- [ ] Linha 678: `\FRDitapalavraapresentacao{... Defesa em XX/XX/2026. Publicada em XX/XX/2026.}` — datas FRD

---

## Step 8: Discutir Título com Orientador

**Título atual (proposto):**
> "Hierarchical Classification of Missing Data Mechanisms: A Statistical Feature Engineering Approach with Real-World Validation"

**Título alternativo (mantendo LLM no subtítulo):**
> "Automatic Classification of Missing Data Mechanisms via Hierarchical Pipeline: Statistical Features, LLM Analysis, and Label Noise Assessment"

**Pontos para discutir:**
- LLMs são resultado negativo em dados reais — manter no título?
- "Hierarchical" é a contribuição principal — deve ser a palavra-chave no título?
- Comprimento do título (padrão ITA?)

---

## ~~Step 9: Revisão Final de Coerência~~ ✅ CONCLUÍDO

**Data conclusão:** 2026-04-19

- [x] Todos os números no Cap 1 (intro) são consistentes com Cap 4 (resultados) — auditados 55.97%, 41.36%, +14.6pp, +10.1pp, +4.6pp, +3.33pp, 76.00→79.33%, d=0.84, d<0.4, 59.4%, +2.7pp, +6pp, 82%, 672/1132; todos batem via cálculo
- [x] Todas as 3 RQs do Cap 1 são respondidas explicitamente no Cap 5 — tese.tex:606-610 (RQ1 hierárquico, RQ2 CAAFE vs LLM, RQ3 label noise)
- [x] Todos os 6 objetivos do Cap 1 são abordados em Cap 3 (sec:hierarchical_strategy, sec:feature_extraction, sec:synthetic_data, sec:real_data) e Cap 4 (sec:results_features, sec:results_label_noise)
- [x] As 5 contribuições do Cap 5 são suportadas por evidência no Cap 4 — tab:hierarchical_v3plus, tab:caafe_vs_llm, tab:confident_joint, sec:Why Naive Bayes Dominates, sec:results_negative
- [x] As 6 limitações do Cap 5 não contradizem claims — "10 de 23 validados" bate com Cap 3 sec:label_validation; "todos 7 MNAR reais são Diffuse" bate com Apêndice B tab:catalog_mnar_subtype
- [x] Abstract (56.0%) e Resumo (56,0%) refletem os resultados finais — valores arredondados consistentes; sem intermediários
- [x] Terminologia consistente: "25 features" explicada como "21 baseline + 4 CAAFE-MNAR" em Cap 3 sec:feature_extraction; tab:feature_summary mostra Total = 25

### Correções aplicadas nos .tex:

1. **`tese.tex:506,508,534`** — padronizado "Naive Bayes" → "Na\\\"{\\i}ve Bayes" (trema) para bater com o uso em Cap 2/3/4
2. **`Cap3/cap3.tex:282`** — tab:routing_strategies "56.0\\%" → "55.97\\%" para consistência com o valor preciso
3. **`Cap4/cap4.tex:117`** — "comparing direct (V1) versus hierarchical (V4)" → "comparing direct three-way classification versus the hierarchical pipeline" (V1 e V4 não existem como configurações nomeadas)
4. **`Cap4/cap4.tex:404`** — tab:head_to_head "41.4\\%" → "41.36\\%" para consistência com o valor preciso
5. **`Cap4/cap4.tex:406`** — tab:head_to_head "56.0\\%" → "55.97\\%" para consistência

---

## Resumo de Prioridades

| Prioridade | Step | Esforço | Bloqueador? | Status |
|:----------:|:----:|:-------:|:-----------:|:------:|
| CRÍTICA | 6 | 30min | Sim — valida todo o trabalho | ✅ |
| ALTA | 1 | 1h | Sim — tabela essencial no Cap 3 | ✅ |
| ALTA | 3a | 1h | Sim — fluxograma é figura central | ✅ |
| ALTA | 3c | 1h | Sim — SHAP é evidência chave | ✅ |
| MÉDIA | 2 | 2h | Não — pode ir no final | ✅ |
| MÉDIA | 3b-3i | 3h | Não — enriquecem mas texto funciona sem | ✅ |
| MÉDIA | 5 | 1h | Depende do Step 3 | ✅ |
| BAIXA | 4 | 15min | Não | ✅ |
| BAIXA | 7 | 1h | Após Step 6 | ⏳ Placeholders administrativos pendentes (banca, Pró-Reitor, coord., data defesa, FRD, dedicatória) — dependem de inputs externos |
| BAIXA | 8 | — | Externo (orientador) | ⏳ Pendente (título) |
| BAIXA | 9 | 1h | Após tudo | ✅ |

**Resumo final:** 7 de 9 steps concluídos. Steps 7 e 8 aguardam inputs externos (secretaria/orientador) e podem ser resolvidos em uma única iteração antes da defesa.
