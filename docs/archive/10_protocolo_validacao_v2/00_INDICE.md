# 10 — Protocolo v2 de Validação de Rótulos

**Data:** 2026-05-03
**Branch:** cairo
**Motivação:** O protocolo v1 (`validar_rotulos.py`) classificava ~57% dos rótulos como inconsistentes, mas possuía fragilidades conceituais nos 3 testes (Little, correlação ponto-biserial, KS com mediana). O protocolo v2 substitui por camadas robustas + calibração com ground truth sintético.

## Documentos nesta pasta

| Documento | Conteúdo |
|-----------|----------|
| [01_DIAGNOSTICO_V1.md](01_DIAGNOSTICO_V1.md) | 5 problemas concretos do protocolo antigo |
| [02_PROTOCOLO_V2.md](02_PROTOCOLO_V2.md) | Arquitetura em camadas, implementação, resultados |
| [03_PENDENCIAS.md](03_PENDENCIAS.md) | O que falta fazer para produção |
| [04_ANALISE_ROBUSTA.md](04_ANALISE_ROBUSTA.md) | Análise exaustiva da calibração robusta (100/200): causas-raiz dos erros, dataset-a-dataset, distribuição real vs sintética |
| [05_PLANO_PARALELISMO.md](05_PLANO_PARALELISMO.md) | Plano de paralelização: 4 níveis, alocação de cores, mudanças por arquivo, reprodutibilidade, checkpoint |
| [06_PLANO_NEXT_STEPS.md](06_PLANO_NEXT_STEPS.md) | Plano de correção de falhas: 6 passos priorizados (CV, scores MNAR, bandwidth, prior, documentação, pipeline) |
| [07_CROSS_VALIDATION_BAYES.md](07_CROSS_VALIDATION_BAYES.md) | Resultado da validação cruzada do Bayes/KDE e o que isso ensina |
| [08_DECISAO_METODOLOGICA_ROTULOS_REAIS.md](08_DECISAO_METODOLOGICA_ROTULOS_REAIS.md) | Decisão para artigo: como usar rótulos da literatura e v2 juntos, com incerteza explícita |

## Resumo em 30 segundos

1. **Protocolo v1** usava Little (sensível a N), correlação ponto-biserial (só linear), KS obs-vs-mediana (tautológico). 57% de inconsistência; PKLM e CAAFE existiam no codebase mas não eram usados.

2. **Protocolo v2** organiza evidências em 5 camadas:
   - **A** (MCAR): Little + PKLM + Levene, voto ≥2/3
   - **B** (MAR): AUC RF mask~Xobs + permutation p-value + MI
   - **C** (MNAR): 4 scores CAAFE-MNAR thresholdados
   - **D** (Reconciliação): Bayes via KDE ajustado nos sintéticos
   - **E** (Calibração): Youden's J + accuracy de sanity check nos 1.200 sintéticos

3. **Resultados smoke test (15/classe, 10 permutações):**
   - Sintéticos: **95,6%** accuracy Bayes (treino=teste, otimista)
   - Reais (29 datasets): **41,4%** accuracy vs rótulo literário (v1 era ~30%)
   - 7 datasets ambíguos (confiança < 0,4)

4. **Resultados calibração robusta (100/classe, 200 permutações):**
   - Sintéticos: **78,3%** accuracy Bayes (treino=teste, abaixo do critério de 85% — achado negativo honesto)
   - Sintéticos com 5-fold CV: **59,0% ± 6,0%** accuracy Bayes (estimativa honesta fora da amostra)
   - Reais (29 datasets): **41,4%** accuracy (mesma, mas distribuição diferente: MAR melhorou 5→8, MNAR piorou 5→2)
   - 11 datasets ambíguos (confiança < 0,4)
   - Análise completa em [04_ANALISE_ROBUSTA.md](04_ANALISE_ROBUSTA.md) e [07_CROSS_VALIDATION_BAYES.md](07_CROSS_VALIDATION_BAYES.md)

5. **Decisão metodológica para escrita do artigo:**
   - Entre rótulo intuitivo e v2, usar **v2** é mais científico (reprodutível e auditável).
   - Entre “v2 sozinho” e “literatura + v2”, usar **literatura como referência principal** e **v2 como evidência auxiliar** (com incerteza).
   - Documento: [08_DECISAO_METODOLOGICA_ROTULOS_REAIS.md](08_DECISAO_METODOLOGICA_ROTULOS_REAIS.md).

## Arquivos de código

| Arquivo | Função |
|---------|--------|
| `src/missdetect/validar_rotulos_v2.py` | Camadas A-D, funções públicas `validate_one()`, `diagnose_rules()`, `diagnose_bayes()` |
| `src/missdetect/calibrar_protocolo.py` | Camada E, calibra thresholds e salva `.json` + `.npz` |
| `tests/test_validar_rotulos_v2.py` | 17 testes (inclui reprodutibilidade paralela e CV Bayes) |
| `data/calibration.json` | Thresholds calibrados (artefato gerado) |
| `data/calibration_scores.npz` | Vetores 10-dim por classe (artefato gerado) |
| `data/calibration_progress.csv` | Checkpoint incremental da calibração (1 linha por dataset; permite retomada) |
| `data/calibration_smoke.json` | Backup dos thresholds do smoke test (15/10) para comparação |
| `data/calibration_scores_smoke.npz` | Backup dos vetores do smoke test |

## Comandos

```bash
# Calibrar com checkpoint (retoma após crash/sleep)
# Sequencial: ~9h | Com paralelismo (ver 05_PLANO_PARALELISMO.md): ~10 min
caffeinate -i uv run python -m missdetect.calibrar_protocolo \
    --n-per-class 100 --n-permutations 200 \
    --checkpoint data/calibration_progress.csv

# Calibrar sem checkpoint (~5 min com params mínimos / smoke)
uv run python -m missdetect.calibrar_protocolo --n-per-class 15 --n-permutations 10

# Monitorar progresso da calibração
wc -l data/calibration_progress.csv   # target: 301 (header + 300 datasets)

# Validar dados reais (modo Bayes)
uv run python -m missdetect.validar_rotulos_v2 --data real --experiment v2 \
    --calibration data/calibration.json --bayes-scores data/calibration_scores.npz

# Validar dados reais (modo regras, sem calibração)
uv run python -m missdetect.validar_rotulos_v2 --data real --experiment v2

# Testes
uv run --extra dev python -m pytest tests/test_validar_rotulos_v2.py
```
