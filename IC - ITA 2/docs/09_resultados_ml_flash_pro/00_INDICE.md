# 09 — Comparação Head-to-Head: ML × Flash × Pro

**Data:** 2026-04-25
**Benchmark:** 29 datasets reais, 1.421 bootstraps, 34 features (25 estatísticas + 9 LLM)
**Pergunta central:** qual é o ganho **isolado** do LLM (Flash e Pro) sobre o pipeline ML puro, mantendo todas as outras variáveis controladas?

---

## Contexto da pasta

Esta pasta consolida a **comparação direta head-to-head** entre três configurações do pipeline de classificação de mecanismos de missing data, todas avaliadas sobre o **mesmo benchmark de 29 datasets reais**:

1. **ML-only** — apenas 25 features estatísticas (descritivas, discriminativas, MechDetect, CAAFE) sem qualquer chamada a LLM
2. **Flash + ML** — pipeline completo com `gemini-3-flash-preview` + extrator `context_aware` (experimento `step10_flash_ca_neutral`)
3. **Pro + ML** — pipeline completo com `gemini-3-pro-preview` + extrator `context_aware` com Step 1 prompt (experimento `step1_v2_neutral`)

A novidade deste documento sobre os anteriores (`06`, `07`, `08`) é que **fixa o benchmark e o pipeline ML**, isolando o efeito puro do modelo LLM. Permite responder: "vale a pena rodar Flash? E Pro?"

A configuração ML-only foi obtida **reaproveitando o `X_features.csv` do `step1_v2_neutral`** e filtrando as 9 colunas `llm_*`, sem necessidade de re-extração — garantindo head-to-head perfeito (mesmo split, mesmo SMOTE, mesmas hyperparams, mesma seed).

---

## Arquivos nesta pasta

| # | Arquivo | Conteúdo |
|---|---------|----------|
| 01 | [01_RESUMO_EXECUTIVO.md](01_RESUMO_EXECUTIVO.md) | Síntese de 1 página com tabela final e veredicto |
| 02 | [02_METODOLOGIA.md](02_METODOLOGIA.md) | Como foi feita a comparação; controle de variáveis; reprodutibilidade |
| 03 | [03_TABELAS_COMPARATIVAS.md](03_TABELAS_COMPARATIVAS.md) | Todas as tabelas (holdout, CV, std, deltas) lado a lado |
| 04 | [04_ANALISE_POR_MODELO.md](04_ANALISE_POR_MODELO.md) | Análise individual: como cada modelo (NB, LogReg, RF, etc.) responde ao LLM |
| 05 | [05_ANALISE_CUSTOS.md](05_ANALISE_CUSTOS.md) | Custo, tempo, eficiência por dollar gasto, breakeven |
| 06 | [06_INSIGHTS_E_NARRATIVA.md](06_INSIGHTS_E_NARRATIVA.md) | Implicações para tese, paper, e iteração de pesquisa |

---

## TL;DR

**Ranking de Best CV (Group 5-Fold):**

| Posição | Configuração | Best CV | Custo | Cust/+1pp |
|:-------:|--------------|:-------:|:-----:|:---------:|
| 1º | **Pro + ML** | **49.33%** (NB) | $30-36 | $16-19 USD/+1pp |
| 2º | ML-only | 47.47% (NB) | $0 | — (gratuito) |
| 3º | Flash + ML | 47.44% (NB) | $2-4 | dominado |

**Achado central:** Flash + ML é **estatisticamente equivalente** a ML-only em CV (diferença −0.03pp; std ±9-12pp). **Flash não agrega valor neste benchmark.** Pro + ML entrega ganho marginal real (+1.86pp CV) sobre ML, justificável apenas para validação final.

**Para a tese:** o pipeline LLM context_aware contribui **incrementalmente** (+1.86pp), não revolucionariamente, sobre features estatísticas + CAAFE no benchmark expandido. O ganho do LLM **diminui com a expansão do benchmark** porque os 6 datasets adicionados (`hypothyroid_t4u`, `echomonths_epss`, `kidney_pot`, `kidney_sod`, `pima_insulin`, `pima_skinthickness`) são exatamente onde o LLM falha (recall ≤ 20% — ver `08_step1_v2_neutral_results/05_DATASETS_PROBLEMATICOS.md`).

---

## Como usar esta documentação

- **Para defesa rápida de "vale a pena LLM?":** ler `01_RESUMO_EXECUTIVO.md` e `05_ANALISE_CUSTOS.md`
- **Para reportar números na tese:** copiar tabelas de `03_TABELAS_COMPARATIVAS.md`
- **Para entender comportamento por modelo:** `04_ANALISE_POR_MODELO.md` (especialmente NB vs SVM/KNN)
- **Para narrativa do paper:** `06_INSIGHTS_E_NARRATIVA.md`
