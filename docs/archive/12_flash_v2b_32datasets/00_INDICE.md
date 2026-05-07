# Fase 12 — Re-execução sobre benchmark v2b (32 datasets)

**Data:** 2026-05-06
**Pergunta-piloto:** A curadoria do benchmark (29 → 32 datasets, Fase 11) muda
qualitativamente as conclusões da Fase 6 sobre quais classificadores e quais
features melhor detectam o mecanismo de missing em dados reais?

## Composição do benchmark

- **6 MCAR + 13 MAR + 13 MNAR = 32 colunas** (de 21 source datasets)
- **1.593 bootstraps** (~50 por coluna; alguns com perdas pelo filtro `missing_rate ≥ 1%`)
- 7 datasets removidos por classificação duvidosa, 6 reclassificados MCAR→MAR,
  4 MCAR adicionados (`boys_*`, `brandsma_*`), 6 MNAR adicionados (`nhanes_*`,
  `support2_*`). Detalhes em `data/real/sources.md` e `docs/HISTORICO.md` Fase 11.

## Status das execuções

| Pipeline | Status | Best CV | Best holdout |
|---|---|---|---|
| **ML-only** (25 features, $0) | ✅ concluído | **GBT 52.54%** | **GBT 51.25%** |
| **Flash + ML** (34 features, ~$3) | ✅ concluído | **RF 51.93%** | **GBT 50.25%** |
| Pro + ML | (não rodado nesta fase) | — | — |

## Comparação rápida vs Fase 6 (29 datasets)

| Métrica | ML-only 29 (Fase 6) | ML-only 32 (Fase 12) | Flash 32 (Fase 12) |
|---|---|---|---|
| Best CV | NB 47.47% | **GBT 52.54%** | RF 51.93% |
| Best holdout | LogReg 54.94% | GBT 51.25% | GBT 50.25% |
| Modelo dominante | NaiveBayes | GradientBoosting | GradientBoosting |
| Importância LLM (RF) | 12.6% (Pro Fase 6) | — | **12.94%** |
| Flash Pareto-domina ML? | ≈ empate (Fase 6) | — | **não** (−0.61pp CV) |

## Mapa de arquivos desta pasta

- [`00_INDICE.md`](00_INDICE.md) — este arquivo (TL;DR e mapa)
- [`01_RESULTADOS_ML_ONLY.md`](01_RESULTADOS_ML_ONLY.md) — resultados detalhados ML-only (32) + matrizes de confusão por classe + feature importance
- [`02_COMPARACAO_29_VS_32.md`](02_COMPARACAO_29_VS_32.md) — comparação ML 29 vs ML 32 lado-a-lado, hipóteses para a virada de regime
- [`03_RESULTADOS_FLASH.md`](03_RESULTADOS_FLASH.md) — resultados Flash sobre 32 datasets: tabelas, recall por classe, feature importance, diagnóstico de 3 hipóteses
- [`04_INFRAESTRUTURA.md`](04_INFRAESTRUTURA.md) — bugs corrigidos para destravar a re-execução (paths.py, subdivisão, metadata neutral)
- [`05_DIAGNOSTICO_MEMORIZACAO.md`](05_DIAGNOSTICO_MEMORIZACAO.md) — teste de memorização: Flash anônimo vs neutral nos 6 MCAR; diagnóstico do ganho +9pp

## Achado-chave (FINAL — Flash concluído)

A Fase 6 concluiu que **NaiveBayes domina porque os labels têm 59% de ruído** — esse
achado **não se replica** com o benchmark v2b. A curadoria removeu boa parte do ruído
e o ranking dos modelos virou: GBT passa NB em +10pp CV. Isso confirma que a vitória
do NB era um sintoma de labels ruins, não de uma propriedade fundamental do problema.

O Flash **não Pareto-domina o ML-only** (−0.61pp CV, dentro do ruído de ±21pp),
exatamente como na Fase 6. O LLM contribui 12.94% de importância mas não converte em
ganho de accuracy agregada. Flash **melhora MCAR +9pp** (reconhece planned-missingness)
mas **piora MNAR −6pp** (confunde borderline censoring). MAR continua irresolvida (45%).

**Próximo passo natural:** Pro sobre 32 datasets — saber se um LLM mais capaz consegue
superar o ML-only em benchmark com ruído reduzido.
