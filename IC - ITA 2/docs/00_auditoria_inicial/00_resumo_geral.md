# Resumo Geral da Analise de Codigo

> **Contexto historico (Fase 0 do projeto):** Auditoria de codigo realizada em 05/04/2026, antes de qualquer
> experimento. Identificou 6 bugs CRITICOS + 7 ALTOS, que foram corrigidos nas fases seguintes.
> Ver [HISTORICO.md](../../HISTORICO.md) para a linha do tempo completa. Este arquivo e preservado
> como registro historico — alguns problemas citados ja foram resolvidos nos planos 1/2/3.

**Data:** 2026-04-05
**Escopo:** `IC - ITA 2/Scripts/` (todos os arquivos Python)

## Contagem de Problemas por Severidade

| Severidade | Quantidade |
|------------|-----------|
| CRITICO    | 6         |
| ALTO       | 7         |
| MEDIO      | 18        |
| BAIXO      | 12        |

## Top 10 Problemas Mais Graves

| # | Arquivo | Problema | Severidade |
|---|---------|----------|------------|
| 1 | `extract_features.py` | Resume de checkpoint corrompe alinhamento features-labels | CRITICO |
| 2 | `extract_features.py` | Modo `--test` so amostra classe MCAR (50 primeiros arquivos) | CRITICO |
| 3 | `train_model.py` | Crash na geracao de graficos per-class (string vs int keys no classification_report) | CRITICO |
| 4 | `analyze_feature_relevance.py` | Diretorio hardcoded `gemini-3-pro-preview` nao existe no pipeline | CRITICO |
| 5 | `analyze_feature_relevance.py` | Permutation importance calculada nos dados de treino (inflada) | CRITICO |
| 6 | `extractor_v2.py` | Thread-safety do cache com 100 workers concorrentes | ALTO |
| 7 | `extractor_v2.py` | 100 chamadas LLM simultaneas causam rate limiting massivo | ALTO |
| 8 | `extractor_v2.py` | Fallback silencioso mascara falha total da API | ALTO |
| 9 | `gerador.py` | Excecao silenciosa esconde se mdatagen funciona ou nao | ALTO |
| 10 | `discriminative.py` | AUC calculada nos dados de treino (overfitting) | ALTO |

## Arquivos de Analise Detalhada

- [01_gerador.md](01_gerador.md) -- Gerador de datasets sinteticos
- [02_extract_features.md](02_extract_features.md) -- Pipeline de extracao de features (inclui statistical.py e discriminative.py)
- [03_llm_extractor.md](03_llm_extractor.md) -- Extrator de features via LLM
- [04_train_model.md](04_train_model.md) -- Treinamento de modelos ML
- [05_compare_results.md](05_compare_results.md) -- Comparacao de resultados
- [06_analyze_feature_relevance.md](06_analyze_feature_relevance.md) -- Analise de importancia de features
- [07_run_all.md](07_run_all.md) -- Orquestrador do pipeline
- [08_problemas_cross_file.md](08_problemas_cross_file.md) -- Problemas entre arquivos
