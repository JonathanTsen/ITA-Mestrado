# Plano 3: Otimização da V3 (Hierárquica + CAAFE)

**Data início:** 2026-04-18
**Última atualização:** 2026-04-18
**Objetivo:** Maximizar accuracy da V3 em dados reais através de melhorias complementares
**Meta:** 58-65% accuracy em dados reais mantendo MNAR recall >= 30%

---

## Estado Atual

| Métrica | Antes (plano_2) | Agora (plano_3) | Delta |
|---------|:---------------:|:---------------:|:-----:|
| **Melhor LOGO CV** | 51.4% (V3 hard, NaiveBayes) | **56.0% (V3+ soft3zone, NaiveBayes)** | **+4.6pp** |
| **Melhor holdout** | 50.5% (V3 hard, GBT) | **53.2% (V3+ threshold + Cleanlab pesos, GBT)** | **+2.7pp** |
| **MNAR recall** (holdout) | 40.0% | **46.0%** | **+6.0pp** |
| **F1 macro** (holdout) | 0.488 | **0.515** | **+0.027** |
| Labels problemáticos | 57% (estimativa) | **59.4% (confirmado Cleanlab)** | — |

---

## Configurações Ótimas Descobertas

| Cenário | Configuração | Accuracy |
|---------|-------------|:--------:|
| **Melhor LOGO CV (robustez)** | NaiveBayes + soft3zone + sem pesos | **56.0%** |
| **Melhor holdout (absoluto)** | GBT + threshold + pesos Cleanlab | **53.2%** |
| **Melhor MNAR recall** | GBT + threshold + pesos Cleanlab | **46.0%** |

**Insight chave:** Pesos Cleanlab melhoram holdout (+2.7pp) mas não afetam LOGO CV (pesos só aplicados no holdout, LOGO CV usa SMOTE). soft3zone é melhor para modelos bem calibrados (NaiveBayes), threshold é melhor para GBT.

---

## Diagnóstico: Por que V3 não passa de 50.5%?

### Gargalos Identificados (por ordem de impacto)

| # | Gargalo | Evidência | Status |
|:-:|---------|-----------|:------:|
| 1 | **Labels ruidosos** (59.4%) | Cleanlab: 672/1132 issues, 15/23 datasets discordantes | Parcial (pesos +2.7pp holdout) |
| 2 | **L2 accuracy baixa** (53%) | MAR vs MNAR é teoricamente difícil (Rubin 1976) | Aberto |
| 3 | ~~Classificador subótimo~~ | XGB/CatBoost + Optuna: sem ganho LOGO CV (38.7% vs 38.3%) | ✅ Descartado |
| 4 | ~~Propagação hard L1→L2~~ | soft3zone +4.6pp LOGO CV, threshold +2.7pp holdout | ✅ Resolvido |
| 5 | **Features redundantes** | Ablação: E1(6f)=49.5% > E3(21f)=40.3% | Aberto |
| 6 | **Poucas features L2** | CAAFE 4f, precisa de mais sinais MAR-vs-MNAR | Aberto |
| 7 | ~~SMOTE básico~~ | SMOTE-ENN/Tomek/Borderline testados, nenhum superou SMOTE k=3 | ✅ Descartado |

### Teto Teórico

**MAR e MNAR não são distinguíveis apenas por dados observados** (Rubin, 1976). Com 59.4% labels inconsistentes, teto prático estimado em **60-65%**. Estamos a 56.0% LOGO CV — a **~4-9pp do teto**.

---

## STEPs — Status

| Ordem | Step | Descrição | Resultado | Status |
|:-----:|:----:|-----------|:---------:|:------:|
| 1 | 01 | **Cleanlab: Limpeza de Labels** | 59.4% issues, pesos +2.7pp holdout | ✅ CONCLUÍDO |
| 2 | 04 | **Roteamento Probabilístico** | soft3zone +4.6pp LOGO, threshold +2.7pp holdout | ✅ CONCLUÍDO |
| 3 | 01+04 | **Combinar Cleanlab + Routing** | threshold+pesos = **53.2% holdout** | ✅ CONCLUÍDO |
| 4 | 07 | **SMOTE-ENN / Borderline** | SMOTE regular vence todos (ENN -1pp, Tomek -0.1pp) | ✅ CONCLUÍDO (sem ganho) |
| 5 | 02 | **XGBoost/CatBoost + Optuna** | Sem ganho: XGB 38.2% / CatB 37.5% LOGO CV vs NB 51.4% | ✅ CONCLUÍDO (sem ganho) |
| 6 | 03 | **Novas Features para L2** | 7f ADV pioram: -2pp acc, MNAR recall=0% | ✅ CONCLUÍDO (sem ganho) |
| 7 | 05 | **Feature Selection Adaptativa** | — | DESCARTADO (03 sem ganho) |
| 8 | 06 | **Stacking Ensemble no L2** | — | DESCARTADO (02+03 sem ganho) |

---

## Conclusão do Plano 3

**Todos os 7 steps experimentais foram executados.** Apenas Steps 01 e 04 produziram melhorias reais.

### O que funcionou
- **Step 01 (Cleanlab pesos):** +2.7pp holdout — reduzir influência de labels ruidosos
- **Step 04 (Soft routing):** +4.6pp LOGO CV — combinar probabilidades L1×L2 para modelos calibrados

### O que não funcionou
- **Step 02 (XGBoost/CatBoost):** NaiveBayes supera todos os classificadores avançados (+12pp LOGO CV)
- **Step 03 (Features ADV L2):** Piora accuracy (-2pp) e mata MNAR recall (0%) — ruído > sinal
- **Step 07 (SMOTE-ENN):** SMOTE regular é melhor — ENN remove amostras úteis em N pequeno
- **Steps 05, 06:** Descartados (dependências sem ganho)

### Próximo passo: Escrita do paper
Os resultados experimentais do plano_3 estão completos. Integrar com achados do plano_2 (SHAP, ablação, PKLM, MechDetect) para a escrita do paper (Step 09 do plano_2).

### Insight principal para o paper
> NaiveBayes com probabilidades calibradas + soft routing hierárquico supera todos os classificadores avançados (GBT, XGBoost, CatBoost, SVM, MLP). Isso sugere que o problema é fundamentalmente de **calibração de incerteza**, não de capacidade do classificador. Com 59.4% de labels ruidosos, modelos simples que estimam probabilidades honestamente vencem modelos complexos que memorizam ruído.

---

## Scripts Criados

| Script | Descrição | Status |
|--------|-----------|:------:|
| `v2_improved/clean_labels.py` | Cleanlab: report, prune, weight, relabel | ✅ Funcional |
| `v2_improved/train_hierarchical_v3plus.py` | V3+ com routing + Cleanlab + balancing | ✅ Funcional |

### Flags disponíveis em `train_hierarchical_v3plus.py`:
- `--routing {hard,threshold,soft3zone,fullprob,all}` — Estratégia de roteamento
- `--clean-labels {none,weight,prune,relabel}` — Integração Cleanlab
- `--balancing {smote,smote_enn,borderline,smote_tomek,none}` — Balanceamento de classes
- `--calibrate / --no-calibrate` — Calibração Platt scaling
- `--optimize` — Otimizar XGBoost/CatBoost com Optuna
- `--n-trials <N>` — Número de trials Optuna (default: 100)

---

## Critérios de Sucesso

| Critério | Meta Mínima | Meta Ideal | Atual | Status |
|----------|:-----------:|:----------:|:-----:|:------:|
| Accuracy real (LOGO CV) | >= 52% | >= 55% | **56.0%** | ✅ Superou |
| Accuracy real (holdout) | >= 55% | >= 60% | **53.2%** | Parcial |
| MNAR recall | >= 30% | >= 45% | **46.0%** | ✅ Superou |
| F1 macro | >= 0.50 | >= 0.55 | **0.515** | ✅ Atingiu |
| McNemar sig (p<0.05) | >= 3/7 | >= 5/7 | 3/7 | ✅ Atingiu |

**3 de 5 metas mínimas já atingidas!** Foco: holdout >= 55%.

---

## Bug Corrigido (2026-04-18)

**`_fit_with_weights` não era chamado no `run_hierarchical_v3plus`.**
Pesos do Cleanlab eram carregados mas ignorados no `.fit()`. Corrigido: agora `_fit_with_weights()` é usado, e quando pesos estão ativos, SMOTE é desativado (pesos já lidam com desbalanceamento).

---

## Referências

- **Cleanlab:** Northcutt et al. (2021). JAIR.
- **MechDetect:** Jung et al. (2024). arxiv:2512.04138
- **PKLM:** Spohn et al. (2021). arxiv:2109.10150
- **MissMecha:** Python package (2025). arxiv:2508.04740
- **CatBoost:** Prokhorenkova et al. (2018).
- **Distance Correlation:** Székely et al. (2007).
- **TabPFN:** Hollmann et al. (2023).

---

## Arquivos de Detalhe

> **Nota (2026-04-19):** Os antigos arquivos `RESULTADOS_STEP02.md`, `RESULTADOS_STEP03.md`
> e `RESULTADOS_STEP07.md` foram fundidos nos respectivos planos como secao "Anexo".
> O `RESULTADOS_STEP01_STEP04.md` permanece separado pois cobre dois steps em conjunto.

### Planos + Resultados (cada arquivo contem plano no inicio e resultados em anexo ao final)
- [STEP01_cleanlab.md](STEP01_cleanlab.md) — ✅ Concluído (resultados em [RESULTADOS_STEP01_STEP04.md](RESULTADOS_STEP01_STEP04.md))
- [STEP02_classificadores_otimizados.md](STEP02_classificadores_otimizados.md) — ✅ Concluído (sem ganho)
- [STEP03_novas_features_l2.md](STEP03_novas_features_l2.md) — ✅ Concluído (sem ganho)
- [STEP04_roteamento_probabilistico.md](STEP04_roteamento_probabilistico.md) — ✅ Concluído (resultados em [RESULTADOS_STEP01_STEP04.md](RESULTADOS_STEP01_STEP04.md))
- [STEP07_smote_avancado.md](STEP07_smote_avancado.md) — ✅ Concluído (sem ganho)

### Resultados combinados
- [RESULTADOS_STEP01_STEP04.md](RESULTADOS_STEP01_STEP04.md) — Cleanlab + Routing (melhorias reais, +2.7pp holdout / +4.6pp LOGO CV)

### Descartados
- [descartados/STEP05_feature_selection.md](descartados/STEP05_feature_selection.md)
- [descartados/STEP06_stacking_ensemble.md](descartados/STEP06_stacking_ensemble.md)
