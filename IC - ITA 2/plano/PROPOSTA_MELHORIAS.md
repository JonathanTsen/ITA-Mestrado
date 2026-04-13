# Proposta de Melhorias: Classificacao de Mecanismos de Missing Data

**Data:** 2026-04-12 (atualizado)
**Status:** STEP01-03 concluidos. STEP04-05 pendentes. Todas as 11 propostas estao detalhadas nos STEPs abaixo.

---

## Diagnostico (atualizado pos-STEP03)

| Problema | Gravidade | Status | Causa Raiz |
|----------|:---------:|:------:|------------|
| ~~3 datasets por mecanismo~~ | ~~CRITICA~~ | **RESOLVIDO** | 23 datasets (5 MCAR + 11 MAR + 7 MNAR) |
| ~~Features X0 sao fingerprints~~ | ~~CRITICA~~ | **RESOLVIDO** | STEP02: ratios/diffs substituiram fingerprints |
| MCAR vs MNAR confusao | **CRITICA** | ABERTO | 46% MCAR classificados como MNAR (sintetico). Features nao capturam auto-dependencia de X0 |
| Rotulos reais inconsistentes | **CRITICA** | PARCIAL | 13/23 falham validacao. MNAR: 6/7 testam como MCAR |
| ~~Outputs insuficientes~~ | ~~ALTA~~ | **RESOLVIDO** | STEP01: CSV/JSON para tudo |
| LLM precisa de foco | **ALTA** | EM PROGRESSO | LLM melhora +3.1pp em reais mas piora -20pp em sinteticos. Precisa focar em MCAR/MNAR |
| ~~Dados sinteticos pouco diversos~~ | ~~MEDIA~~ | **RESOLVIDO** | STEP03: 12 variantes, 1200 datasets |
| MNAR diffuse indetectavel | MEDIA | DOCUMENTADO | 7/7 MNAR reais = Diffuse. Limitacao teorica |

---

## Propostas (11 no total)

| # | Proposta | Status | Coberta por |
|:-:|----------|:------:|-------------|
| 1 | Features invariantes ao dataset | **FEITO** | STEP02 — features de ratio/diff substituiram fingerprints |
| 2 | Expandir para 10+ datasets reais/mecanismo | **PARCIAL** | STEP03 — 23 coletados, 10 validados. Meta nao atingida para MCAR (3) e MNAR (1) |
| 3 | Reformular LLM (3 alternativas) | PENDENTE | STEP04 — CAAFE, embeddings, prompt + **MCAR vs MNAR** |
| 4 | Corrigir rotulos dos datasets | **FEITO** | STEP03 — validar_rotulos.py com 3 testes |
| 5 | Estrategia de treinamento revisada | PENDENTE | STEP05 — hierarquica, LOGO CV |
| 6 | Pipeline de validacao experimental | PENDENTE | STEP05 — ablacao, metricas de sucesso |
| 7 | Enriquecer outputs do pipeline | **FEITO** | STEP01 — CSV/JSON para todas as metricas |
| 8 | Adotar MechDetect | **FEITO** | STEP02 — 6 features MechDetect (3 AUCs + deltas) |
| 9 | MissMecha para dados diversos | **FEITO** | STEP03 — gerador_v2.py com 12 variantes |
| 10 | CAAFE-style LLM | PENDENTE | STEP04 — com foco extra em MCAR vs MNAR |
| 11 | Limitacao MNAR diffuse | **FEITO** | STEP03 — classificar_mnar.py (Focused vs Diffuse) |

---

## Ordem de Execucao

| Fase | Step | Descricao | Status | Arquivo de detalhes |
|:----:|:----:|-----------|:------:|---------------------|
| 4A | 1 | Outputs enriquecidos | FEITO (fase3) | [STEP01](STEP01_outputs_enriquecidos.md) |
| 4B | 2 | Features MechDetect + invariantes + MNAR | **FEITO** | [STEP02](STEP02_features_mechdetect_invariantes.md) |
| 4C | 3 | MissMecha + rotulos + novos datasets | **FEITO (parcial)** | [STEP03](STEP03_dados_missmecha_rotulos.md) |
| 4D | 4 | LLM: CAAFE, embeddings, prompt | **CRITICO** | [STEP04](STEP04_llm_caafe_embeddings.md) |
| 4E | 5 | Otimizacao + documentacao tese | PENDENTE | [STEP05](STEP05_otimizacao_tese.md) |

---

## Metas de Sucesso

| Metrica | Fase 3 | Step02 (sint) | Step03 (sint) | Step03_final (real) | Meta Min | Meta Ideal |
|---------|:------:|:-------------:|:-------------:|:-------------------:|:--------:|:----------:|
| Accuracy (melhor modelo) | 69.9% | **87.2%** ✅ | **76.7%** ⚠️ | **43.4%** ❌ | > 70% | > 85% |
| CV variancia | 55-74% | **±4.3%** ✅ | **±3.3%** ✅ | **±30%** ❌ | < 20% | < 10% |
| LLM delta vs baseline | -10.7% | **-20.3%** ❌ | pendente | **+3.1pp** ✅ | >= 0% | > +5% |
| Outputs estruturados | ~30% | 100% ✅ | 100% ✅ | 100% ✅ | 100% | 100% |

**Notas Step03:**
- Accuracy sintetico caiu de 87.2% → 76.7% (mais variantes = problema mais dificil, ESPERADO)
- Accuracy real baixa (43%) reflete rotulos inconsistentes (13/23 falharam validacao)
- **LLM contribui positivamente pela primeira vez**: +3.1pp medio, melhora em 5/7 modelos. Maior ganho em SVM (+8.5pp). Hipotese: com rotulos ruidosos, o raciocinio qualitativo do LLM ajuda a compensar

---

## Artigos-Chave dos `Artigos_Relevantes/`

| Artigo | Pasta | Contribuicao para o projeto |
|--------|-------|-----------------------------|
| MechDetect (Jung et al., 2024) | 08 | Abordagem de 3 tarefas AUC-ROC — 89% em 101 datasets |
| PKLM (Spohn et al., 2024) | 08 | Teste MCAR nao-parametrico via Random Forest + KL |
| MissMecha (Zhou et al., 2025) | 08 | Toolkit Python com 17 variantes + validacao estatistica |
| CAAFE (Hollmann et al., NeurIPS 2023) | 09 | LLM gera codigo para features, loop iterativo |
| Enriching Tabular (Kasneci, 2024) | 09 | LLM embeddings + PCA + RF selection |
| Comprehensive Review (Zhou et al., 2024) | 08 | Focused vs Diffuse MNAR — limitacao teorica |
