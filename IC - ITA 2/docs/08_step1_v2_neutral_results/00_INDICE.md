# 08 — Step 1 V2 Neutral: Resultados e Análise Exaustiva

**Data:** 2026-04-25
**Experimento:** `step1_v2_neutral` (`Output/v2_improved/step1_v2_neutral/`)
**Modelo LLM:** gemini-3-pro-preview com extrator `context_aware`
**Metadata:** variante neutral (anti-vazamento Canal F)
**Dados:** 29 datasets reais, 1421 bootstraps, 34 features (25 estatísticas + 9 LLM)

---

## Contexto da pasta

Este conjunto de documentos detalha a re-execução do **Step 1** do plano de melhorias de domain reasoning (originalmente proposto em `07_next_steps_domain_reasoning/01_step1_fewshot_tipologia.md`) sobre o benchmark expandido de 29 datasets reais. O objetivo era validar se o prompt revisado (3 exemplos canônicos + tipologia MNAR + instrução anti-MAR-bias, todos já implementados em `llm/context_aware.py`) elevaria a acurácia hierárquica acima do patamar de 56.2% CV obtido em `forensic_neutral_v2`.

A execução foi planejada em duas metades por restrição de custo (~$15-18 cada metade Pro), com 15 parent-datasets na Metade 1 e 14 na Metade 2, mantendo balanceamento de mecanismos (MCAR/MAR/MNAR) em ambas. Os resultados foram consolidados via `merge_halves.py` antes do treinamento final.

---

## Arquivos nesta pasta

| # | Arquivo | Conteúdo |
|---|---------|----------|
| 01 | [01_RESUMO_EXECUTIVO.md](01_RESUMO_EXECUTIVO.md) | Síntese de uma página: pergunta, resposta, números-chave, veredicto |
| 02 | [02_METODOLOGIA.md](02_METODOLOGIA.md) | Pipeline, split em halves, proteções anti-vazamento, custos, tempos |
| 03 | [03_RESULTADOS_NUMEROS.md](03_RESULTADOS_NUMEROS.md) | Holdout, CV, matrizes de confusão, métricas por classe e por modelo |
| 04 | [04_ANALISE_REGRESSAO.md](04_ANALISE_REGRESSAO.md) | Por que houve regressão de −7pp vs `forensic_neutral_v2` |
| 05 | [05_DATASETS_PROBLEMATICOS.md](05_DATASETS_PROBLEMATICOS.md) | Os 6 datasets onde o LLM ainda falha (MAR-bias residual) |
| 06 | [06_FEATURE_IMPORTANCE.md](06_FEATURE_IMPORTANCE.md) | Ranking de importância, razão LLM/estatísticas |
| 07 | [07_DISCUSSAO_LIMITACOES.md](07_DISCUSSAO_LIMITACOES.md) | Discussão dos achados, limitações, validade externa |
| 08 | [08_PROXIMOS_PASSOS.md](08_PROXIMOS_PASSOS.md) | Direções: Step 2 (DAG), Step 3 (SC com Pro), audit de regressão |

---

## TL;DR

**Score final (Group 5-Fold CV, NaiveBayes):** **49.33% ± 14.2%**

**Score holdout (GroupShuffleSplit 75/25):** **55.19%** (NB) / **54.94%** (LogReg)

**Comparação:**
- vs `step10_flash_ca_neutral` (referência Flash, mesmo benchmark): **+1.9pp CV / +4.05pp holdout** ✅
- vs `forensic_neutral_v2` (Pro neutral, benchmark antigo 23 datasets): **−7pp CV / −19pp domain_prior só** ❌
- vs target Step 1 (60% CV planejado): **−10.7pp** ❌

**Veredicto:** Step 1 entregou ganho marginal sobre Flash, mas **não atingiu o target** e regrediu vs a referência publicável. A causa raiz é MAR-bias residual em 6 datasets (incluindo o canônico `MNAR_pima_insulin` com apenas 4% recall). Step 2 (Causal Reasoning DAG) é o próximo passo natural.

---

## Como usar esta documentação

- **Para defesa rápida:** ler `01_RESUMO_EXECUTIVO.md` e tabela final de `03_RESULTADOS_NUMEROS.md`
- **Para auditoria reproducível:** `02_METODOLOGIA.md` + comandos exatos em `merge_halves.py`
- **Para entender a regressão:** `04_ANALISE_REGRESSAO.md` (comparação por dataset com `forensic_neutral_v2`)
- **Para escrever discussão da tese/paper:** `05_DATASETS_PROBLEMATICOS.md` + `07_DISCUSSAO_LIMITACOES.md`
- **Para planejar o próximo experimento:** `08_PROXIMOS_PASSOS.md`

---

## Documento relacionado

🔗 **[09_resultados_ml_flash_pro/](../09_resultados_ml_flash_pro/00_INDICE.md)** — Comparação head-to-head ML-only × Flash × Pro sobre o **mesmo benchmark** desta pasta. Resposta direta a "vale a pena LLM?":

| Configuração | Best CV | Custo |
|--------------|:-------:|:-----:|
| ML-only (25 features) | 47.47% | $0 |
| Flash + ML | 47.44% | $3 |
| **Pro + ML (este experimento)** | **49.33%** | **$33** |

**Achado chave:** Flash é Pareto-dominado por ML; Pro agrega +1.86pp CV (incremental, não revolucionário). Para análise detalhada de custo-benefício, narrativa para tese e implicações por classificador, ver pasta 09.
