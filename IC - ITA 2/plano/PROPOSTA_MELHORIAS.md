# Proposta de Melhorias: Classificacao de Mecanismos de Missing Data

**Data:** 2026-04-12
**Status:** STEP02 concluido. Sintetico baseline subiu de 69.9% → 87.2% (LogReg). LLM continua prejudicando (-20pp). Dados reais pendente.

---

## Diagnostico

| Problema | Gravidade | Causa Raiz |
|----------|:---------:|------------|
| 3 datasets por mecanismo | CRITICA | Impossivel generalizar com GroupShuffleSplit |
| Features X0 sao fingerprints | CRITICA | X0_mean/q25/q50/q75 identificam dataset, nao mecanismo |
| MNAR indetectavel | CRITICA | Nenhuma feature captura dependencia do proprio X0 |
| Rotulos inconsistentes | ALTA | Oceanbuoys (MCAR) provavelmente e MAR |
| Outputs insuficientes | ALTA | Metricas em texto/PNG, sem CSV/JSON estruturado |
| LLM contribui 6.6% | MEDIA | Prompt gera classificacao redundante, nao features novas |
| Dados sinteticos pouco diversos | MEDIA | So 1 variante por mecanismo (de 17 possiveis) |
| MNAR diffuse e teoricamente indetectavel | MEDIA | Limitacao fundamental (literature) |

---

## Propostas (11 no total)

| # | Proposta | Prioridade | Detalhes |
|:-:|----------|:----------:|----------|
| 1 | Features invariantes ao dataset | MAXIMA | Substituir X0_mean/q25/q50/q75 por ratios e diffs |
| 2 | Expandir para 10+ datasets reais/mecanismo | MAXIMA | Resolver o N insuficiente |
| 3 | Reformular LLM (3 alternativas) | ALTA | CAAFE, embeddings, ou prompt novo |
| 4 | Corrigir rotulos dos datasets | ALTA | Remover oceanbuoys, validar com testes estatisticos |
| 5 | Estrategia de treinamento revisada | MEDIA | Hierarquica, LOGO CV, normalizacao |
| 6 | Pipeline de validacao experimental | MEDIA | Sanidade sintetico, ablacao, metricas de sucesso |
| 7 | Enriquecer outputs do pipeline | ALTA | CSV/JSON para todas as metricas |
| 8 | **Adotar MechDetect** (do paper) | ALTA | 3 tarefas AUC-ROC, 89% em 101 datasets |
| 9 | **MissMecha para dados diversos** (do paper) | ALTA | 17 variantes de mecanismos |
| 10 | **CAAFE-style LLM** (do paper) | ALTA | LLM gera codigo Python, nao scores |
| 11 | **Limitacao MNAR diffuse** (do paper) | MEDIA | Reconhecer na tese, separar focused vs diffuse |

---

## Ordem de Execucao

| Fase | Step | Descricao | Status | Arquivo de detalhes |
|:----:|:----:|-----------|:------:|---------------------|
| 4A | 1 | Outputs enriquecidos | FEITO (fase3) | [STEP01](STEP01_outputs_enriquecidos.md) |
| 4B | 2 | Features MechDetect + invariantes + MNAR | **FEITO** | [STEP02](STEP02_features_mechdetect_invariantes.md) |
| 4C | 3 | MissMecha + rotulos + novos datasets | PENDENTE | [STEP03](STEP03_dados_missmecha_rotulos.md) |
| 4D | 4 | LLM: CAAFE, embeddings, prompt | **CRITICO** | [STEP04](STEP04_llm_caafe_embeddings.md) |
| 4E | 5 | Otimizacao + documentacao tese | PENDENTE | [STEP05](STEP05_otimizacao_tese.md) |

---

## Metas de Sucesso

| Metrica | Fase 3 | Step02 (sintetico) | Meta Minima | Meta Ideal |
|---------|:------:|:------------------:|:-----------:|:----------:|
| Accuracy (melhor modelo, sintetico) | 69.9% | **87.2%** ✅ | > 70% | > 85% |
| Accuracy (melhor modelo, real) | 60.7% | pendente | > 70% | > 85% |
| CV variancia | 55-74% | **±4.3%** ✅ | < 20% | < 10% |
| LLM delta vs baseline | -10.7% | **-20.3%** ❌ | >= 0% | > +5% |
| Outputs estruturados | ~30% | 100% ✅ | 100% | 100% |

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
