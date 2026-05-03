# Plano 2: Paper Hierárquico + LLM para Journal

**Data:** 2026-04-18 (atualizado)
**Objetivo:** Paper propondo classificação hierárquica de mecanismos de missing data com LLM-augmented features focadas na desambiguação MCAR-MNAR
**Target:** Journal (Statistical Analysis and Data Mining, DAMI, ML Journal)

---

## Contexto

O Plano 1 (pasta `plano/`) construiu o pipeline: features invariantes, MechDetect, dados diversos, LLM reformulado. Os resultados revelaram:

- **Novelty forte**: ninguém combinou LLMs com classificação de mecanismos de missing data
- **Gargalo identificado**: MCAR vs MNAR é o problema central — 46% de confusão em sintéticos, 0% MNAR recall em reais
- **Insight chave**: LLM piora em dados limpos (-20pp sintético) mas ajuda em dados ruidosos (+3.1pp real)
- **Descobertas secundárias**: 57% dos labels de benchmark inconsistentes, MNAR Diffuse fundamentalmente difícil

---

## Ângulo do Paper: Hierárquico + LLM Focado

**Tese central:** A classificação 3-way direta de mecanismos de missing data falha porque MCAR e MNAR são quase indistinguíveis por features estatísticas. A solução proposta:

1. **Classificação hierárquica em 2 níveis:**
   - **Nível 1:** MCAR vs não-MCAR (features estatísticas bastam — problema mais fácil)
   - **Nível 2:** MAR vs MNAR (aqui features estatísticas falham → **LLM entra como diferencial**)

2. **Por que LLM só no Nível 2:**
   - MCAR vs não-MCAR é resolvível com features estatísticas (Little's test, MechDetect deltas)
   - MAR vs MNAR é onde a circularidade aparece (MNAR depende de X0, mas X0 está faltante)
   - LLM captura padrões qualitativos (censura, truncamento, assimetria) que escapam a métricas pontuais
   - Isso explica o resultado contra-intuitivo: LLM piora no geral (-20pp) mas ajuda no subproblema difícil

3. **Contribuições secundárias integradas:**
   - Validação de labels: 57% inconsistentes → questiona benchmarks existentes
   - MNAR Focused vs Diffuse: limitação teórica documentada
   - Ablação mostrando contribuição marginal de cada grupo de features

**Narrativa do paper:**
> Features estatísticas → classificação 3-way falha (MCAR≈MNAR) → diagnóstico: circularidade de X0 → solução: hierárquica + LLM focado no gargalo → resultado: LLM só ajuda onde features estatísticas falham

---

## STEPs

| Ordem | Step | Descrição | Estimativa | Dependência | Status |
|:-----:|:----:|-----------|:----------:|:-----------:|:------:|
| 1 | 05-A | **Classificação hierárquica + LLM no Nível 2** (CORE) | 2-3 dias | Nenhuma | ✅ CONCLUÍDO (2026-04-18) |
| 2 | 04-B | Ablação completa + significância estatística | 1-2 dias | Nenhuma | ✅ CONCLUÍDO (2026-04-18) |
| 3 | 06 | MechDetect como baseline de comparação | 1-2 dias | Nenhuma | ✅ CONCLUÍDO (real: 2026-04-18, sintético: pendente) |
| 4 | 07 | PKLM como baseline / feature | 1-2 dias | Nenhuma | ✅ CONCLUÍDO (2026-04-18) |
| 5 | 05-B | LOGO Cross-Validation | 1 dia | Steps 1-4 | ✅ CONCLUÍDO (integrado no 05-A) |
| 6 | 08 | Análise SHAP + Error Analysis | 1-2 dias | Steps 1-5 | ✅ CONCLUÍDO (2026-04-18) |
| 7 | — | Investigação V4 MNAR recall 6% | — | Step 1 | ✅ CONCLUÍDO (2026-04-18) |
| 8 | 09 | Escrita do paper | 3-5 dias | Steps 1-7 | PENDENTE |

**Tempo total estimado:** 2-3 semanas (inclui ~30% buffer para debugging e iterações)
- Steps 1-4 paralelizáveis: ~2-3 dias (gargalo: hierárquica)
- Steps 5-6 sequenciais: ~2-3 dias
- Step 7 (escrita): ~3-5 dias
**Paralelizável:** Steps 1, 2, 3, 4 (sem dependências entre eles)

**Nota:** Steps 3 e 4 (MechDetect/PKLM) são baselines de comparação, não o foco. Escopo reduzido vs plano anterior (Survey/Benchmark).

---

## Diagnóstico: O que existe vs O que falta

### Já temos (do Plano 1):
- Pipeline LLM-augmented completo (21 baseline + CAAFE + embeddings + judge)
- 1200 datasets sintéticos (12 variantes) + 23 reais
- 7 classificadores testados
- Validação de labels (57% inconsistentes)
- Classificação MNAR Focused/Diffuse
- Features inspiradas em MechDetect (6 features)
- Evidência de que LLM ajuda em real (+3.1pp) mas não em sintético (-20pp)

### Falta para este paper:
- **Classificação hierárquica com LLM no Nível 2** (contribuição principal)
- MechDetect e PKLM como baselines (escopo reduzido — comparação, não benchmark exaustivo)
- Ablação completa (6 configurações + hierárquica)
- Testes de significância estatística
- LOGO Cross-Validation
- Análise SHAP mostrando que LLM importa mais no Nível 2

---

## Critérios de Sucesso para o Paper

| Critério | Meta Mínima | Meta Ideal |
|----------|:-----------:|:----------:|
| Hierárquica melhora MNAR recall | > 30% (vs 0% atual) | > 50% |
| LLM no Nível 2 supera Nível 2 sem LLM | Delta ≥ 0% | Delta > +5pp |
| Hierárquica accuracy global | ≥ accuracy direta - 5pp | > accuracy direta |
| Testes de significância | Wilcoxon + CI 95% | + McNemar + Friedman |
| SHAP confirma: LLM mais importante no Nível 2 | Importância LLM Nível 2 > Nível 1 | LLM top 5 features no Nível 2 |

**Plano B:** Se hierárquica + LLM NÃO melhorar, o paper vira "Feature Engineering para Missing Data" (Opção 5) — a história de fingerprints → invariantes → MechDetect continua forte.

---

## Journals Candidatos

| Journal | Escopo | Impact Factor | Fit |
|---------|--------|:-------------:|:---:|
| **Statistical Analysis and Data Mining** | Estatística + ML | ~1.5 | Muito alto — foco perfeito |
| **Data Mining and Knowledge Discovery (DAMI)** | ML + Data Mining | ~4.0 | Alto — contribuição metodológica |
| **Machine Learning Journal** | ML teórico e empírico | ~4.5 | Alto — se análise SHAP for profunda |
| **Pattern Recognition Letters** | Short paper | ~3.9 | Backup — se resultados modestos |

---

## Estrutura Proposta do Paper

1. **Introduction** — Problema, gargalo MCAR-MNAR, proposta hierárquica + LLM
2. **Related Work** — Missing data theory, MechDetect, PKLM, LLM features (CAAFE)
3. **Methodology**
   - 3.1 Datasets (sintético + real + validação de labels)
   - 3.2 Feature extraction (baseline 21 + CAAFE + LLM)
   - 3.3 Classificação hierárquica (Nível 1 + Nível 2 + LLM no Nível 2)
   - 3.4 Baselines (direto 3-way, MechDetect, PKLM)
   - 3.5 Evaluation protocol (LOGO CV, significância)
4. **Results**
   - 4.1 Classificação direta vs hierárquica (contribuição principal)
   - 4.2 Impacto do LLM no Nível 2 (contribuição principal)
   - 4.3 Ablação de features
   - 4.4 SHAP: por que LLM importa mais no Nível 2
   - 4.5 Comparação com baselines (MechDetect, PKLM)
5. **Discussion**
   - 5.1 MCAR-MNAR: barreira fundamental e como hierárquica contorna
   - 5.2 Label quality: 57% inconsistentes
   - 5.3 MNAR Focused vs Diffuse: limitação teórica
   - 5.4 Quando e onde LLMs ajudam
6. **Conclusion + Future Work**

---

## Arquivos de Detalhe

> **Nota (2026-04-19):** Os antigos arquivos `RESULTADOS_STEP04B.md`, `RESULTADOS_STEP05A.md`,
> `RESULTADOS_BALANCEAMENTO.md`, `RESULTADOS_STEP06.md`, `RESULTADOS_STEP07.md` e
> `RESULTADOS_STEP08.md` foram fundidos como secao "Anexo" nos respectivos arquivos de plano.

- [STEP05A_classificacao_hierarquica.md](STEP05A_classificacao_hierarquica.md) — **CORE do paper** (plano + resultados + balanceamento)
- [STEP04B_ablacao_significancia.md](STEP04B_ablacao_significancia.md) — plano + resultados
- [STEP05B_logo_cv.md](STEP05B_logo_cv.md)
- [STEP06_mechdetect_original.md](STEP06_mechdetect_original.md) — plano + resultados
- [STEP07_pklm.md](STEP07_pklm.md) — plano + resultados (PKLM não detecta MNAR, poder 5.8%)
- [STEP08_shap_error_analysis.md](STEP08_shap_error_analysis.md) — plano + resultados
- [INVESTIGACAO_V4_MNAR.md](INVESTIGACAO_V4_MNAR.md) — Por que V4 tem MNAR recall 6% (LLM features não discriminam)
- [ACHADOS_CONSOLIDADOS.md](ACHADOS_CONSOLIDADOS.md) — Síntese de todos os achados experimentais (2026-04-18)
- [STEP09_escrita_paper.md](STEP09_escrita_paper.md)
