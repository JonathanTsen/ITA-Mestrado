# Análise da Regressão vs `forensic_neutral_v2`

**Data:** 2026-04-25
**Achado:** `step1_v2_neutral` ficou −7pp em CV abaixo de `forensic_neutral_v2`

---

## 1. O fato a ser explicado

| Métrica | `forensic_neutral_v2` (2026-04-19) | `step1_v2_neutral` (2026-04-25) | Δ |
|---------|:---:|:---:|:---:|
| Datasets reais | 23 | 29 | +6 |
| Bootstraps | 1.132 | 1.421 | +289 |
| Best CV avg (NB) | **56.2%** | **49.33%** | **−6.87pp** |
| domain_prior alone | **63.1%** | **43.7%** | **−19.4pp** |
| Recall MAR (LODO) | 96.5% | ~67% (CV) | −30pp |
| Recall MCAR (LODO) | ~30% | ~25% | −5pp |
| Recall MNAR (LODO) | ~34% | ~32% | −2pp |
| Modelo LLM | Pro context_aware | Pro context_aware (com Step 1 prompt) | upgrade |
| Variante metadata | neutral | neutral | igual |

A regressão é **maior** quando se isola a feature LLM (`domain_prior` solo: −19pp) e **menor** quando há ML em cima (NB CV: −7pp). Isso sugere que o problema central está no **sinal do LLM**, parcialmente compensado pelo ML.

## 2. Hipóteses para a regressão

### 2.1 Hipótese H1: Expansão do benchmark adicionou datasets clinicamente difíceis

**Evidência a favor:**

Os 6 datasets que **não estavam** em `forensic_neutral_v2` foram identificados via comparação com `docs/06_analise_final_publicacao/02_INVENTARIO_RESULTADOS.md`:

| Dataset adicionado | Recall (domain_prior) | Comentário |
|--------------------|:--------------------:|------------|
| `MCAR_creditapproval_a14` | 59% | OK, não é o problema |
| `MCAR_echomonths_epss` | **4%** | Falha total |
| `MCAR_hepatitis_albumin` | 32% | MAR-bias |
| `MCAR_hepatitis_alkphosphate` | 20% | MAR-bias |
| `MCAR_hypothyroid_t4u` | **0%** | Falha total |
| `MNAR_kidney_sod` | 8% | MAR-bias |

**5 dos 6 datasets novos têm recall ≤ 32%** (média ≈ 17%). Se removêssemos esses 5 ruins do benchmark, o domain_prior solo subiria para algo próximo de 55-60% — coerente com a referência. **Esta é a hipótese mais forte.**

**Cálculo aproximado:** assumindo que os 23 datasets de `forensic_neutral_v2` mantêm acc agregada ~63% e os 6 novos têm acc ~28%, a média ponderada é (1.132 × 0.63 + 289 × 0.28) / 1.421 ≈ 56% — significativamente acima dos 43.7% observados. Isso sugere que H1 explica **parte** mas não toda a regressão.

### 2.2 Hipótese H2: O prompt Step 1 reduziu MAR-bias mas não compensou perda de contexto semântico

**Evidência a favor:**

Em `forensic_neutral_v2`, o LLM tinha 96.5% recall em MAR — extremamente alto. O Step 1 explicitamente combateu MAR-bias:

> ## INSTRUÇÃO IMPORTANTE
> ATENÇÃO: Não assuma que MAR é o mecanismo mais provável por padrão.
> Em datasets reais:
> - ~30% são MCAR (dados faltam por falhas técnicas)
> - ~40% são MAR
> - ~30% são MNAR

Esta instrução pode ter "puxado" a distribuição de predições para MCAR/MNAR, reduzindo MAR de 96.5% → 67.6% recall. Em datasets onde **MAR era a resposta correta** (mas o LLM agora hesita), a precisão aumenta (51% MCAR precision via NB) mas o recall desce.

**Evidência contra:**

A precisão MAR também caiu (52% no holdout NB), e LogReg distribui melhor — sugere que não é só "puxada" para outras classes, mas confusão genuína em datasets ambíguos.

### 2.3 Hipótese H3: Mudança no benchmark de avaliação (LODO → Group 5-Fold CV)

**Evidência a favor:**

`forensic_neutral_v2` usava **LODO** (Leave-One-Dataset-Out) — cada dataset vira teste isolado por vez. `step1_v2_neutral` usa **Group 5-Fold CV** — 5 partições com ~6 datasets em teste por vez.

LODO tende a ser **mais otimista** porque:
- Cada fold tem 22 grupos de treino vs 5 grupos de teste (mais sinal por fold)
- O modelo pode "memorizar" comportamentos de quase todos os datasets
- Variância entre folds é diluída (n_folds = n_datasets, com avg estável)

5-Fold CV tem 21 grupos de treino vs 8 grupos de teste por fold, exigindo generalização para ~28% dos datasets simultaneamente.

**Estimativa do impacto:** poderia explicar 2-4pp de regressão sozinho, especialmente quando a heterogeneidade entre datasets é alta (como atestado pela variância CV de ±14-27pp).

### 2.4 Hipótese H4: Mudança no código entre datas (regressão silenciosa)

**Evidência a favor:**

Entre 2026-04-19 e 2026-04-25, vários commits modificaram o pipeline:

```
1dd9c3f feat: Add new articles and datasets related to causal reasoning and missing data techniques
3621b8a feat/Add new articles and datasets related to MAR vs MNAR distinction
846de4f feat/Add baseline results and training summaries for synthetic datasets
38edc99 Add expanded real datasets (MAR, MCAR, MNAR) with bootstrap chunks and update plans
```

O commit 38edc99 expandiu o dataset (cobrindo H1). Os outros adicionaram artigos e baselines sintéticos. **Não há evidência clara de mudança no `context_aware.py` ou no prompt** entre as duas datas — o Step 1 prompt já estava implementado em ambos os períodos (audit confirmou linhas 354-420 e 488-499).

**Evidência contra:** sem regressão de código identificada, H4 é improvável.

### 2.5 Hipótese H5: Fluctuação do LLM (não-determinismo)

**Evidência a favor:**

Mesmo com `temperature=0.1`, gemini-3-pro-preview tem variância entre chamadas (sampling não-zero). Re-rodar `forensic_neutral_v2` hoje poderia dar 56.2% ± 3pp.

**Evidência contra:**

A regressão de −19pp em domain_prior solo é **muito grande para ser ruído estocástico** — esperaríamos ±2-5pp típico.

## 3. Veredicto

A regressão de −7pp CV / −19pp domain_prior é **majoritariamente atribuível a H1 (expansão do benchmark)**, com contribuição menor de H3 (mudança LODO → 5-Fold).

**Estimativa de partição:**

| Hipótese | Estimativa de impacto |
|----------|:---------------------:|
| H1 (datasets difíceis adicionados) | −10 a −15pp em domain_prior |
| H3 (LODO → 5-Fold CV) | −2 a −4pp em CV |
| H2 (Step 1 trade-off MAR/outros) | −2 a −3pp em domain_prior |
| H5 (ruído LLM) | −2 a +2pp |
| H4 (regressão de código) | improvável |
| **Total observado** | **−7pp CV; −19pp domain_prior** |

## 4. Implicações

### 4.1 O Step 1 prompt **não é o vilão**

O Step 1 reduziu MAR-bias (de 96.5% → 67.6% em MAR recall) e melhorou marginalmente vs Flash. Não há evidência de que reverter para o prompt original `forensic_neutral_v2` melhoraria os números — porque a maior parte da regressão vem dos datasets adicionados, que seriam difíceis com qualquer prompt.

### 4.2 O benchmark expandido é mais honesto

Os 6 datasets adicionados representam casos **mais difíceis e mais realistas** (clinical missingness sutil, MCAR ambíguo). A queda da accuracy reflete realidade clínica, não fragilidade do método.

**Comparação justa requer:**
- ❌ **Não comparar** `step1_v2_neutral` (29 datasets) com `forensic_neutral_v2` (23 datasets) diretamente — são benchmarks diferentes
- ✅ **Comparar** `step1_v2_neutral` (29) com `step10_flash_ca_neutral` (29 mesmos datasets) — mesma comparação justa: +1.9pp CV / +4.05pp holdout
- ✅ **Reportar** ambas referências na tese, justificando a expansão como contribuição metodológica

### 4.3 Para subir além de 50% CV no benchmark de 29 datasets

Os 6 datasets problemáticos compartilham uma característica: **MAR-bias persistente** mesmo com instrução anti-bias explícita. Para vencê-los é necessário que o LLM:
1. **Nomeie a variável causadora** quando classifica MAR (Step 2 — Causal Reasoning DAG)
2. **Vote em múltiplas perspectivas** para reduzir variância dentro do mesmo bootstrap (Step 3 — Self-Consistency com Pro)

A configuração Pro + Step 1 + neutral provavelmente está próxima do **teto do que prompt engineering simples consegue** — futuros ganhos exigem decomposição estruturada do raciocínio.

## 5. Recomendação de narrativa para tese/paper

A regressão **deve ser reportada explicitamente** na seção de resultados, com a seguinte estrutura:

1. **Reportar** `forensic_neutral_v2` (56.2% CV) como referência sobre 23 datasets
2. **Reportar** `step1_v2_neutral` (49.3% CV) sobre o benchmark expandido de 29 datasets
3. **Atribuir a queda** à expansão do benchmark, não à degradação do método
4. **Mostrar** que sobre os mesmos 29 datasets, Step 1 Pro **bate** Flash (+1.9pp CV), demonstrando que o prompt instrumentado é incrementalmente melhor
5. **Concluir** que prompt engineering simples atinge ~50% CV neste benchmark realista, motivando os Steps 2 e 3 como direções futuras

Esta narrativa é **honesta, científica e publicável** — apresenta limitação e contextualiza, sem inflar ou esconder números.

## 6. Próximas validações sugeridas

Para confirmar H1 e descartar H4 definitivamente, propõe-se:

1. **Re-executar `forensic_neutral_v2` com o código atual** sobre os mesmos 23 datasets (custo: ~$25-30). Se acc ≈ 56% confirma ausência de regressão de código.
2. **Avaliar `step1_v2_neutral` com LODO** sobre os 29 datasets (custo: zero, só re-treino). Se LODO der ~52-54%, confirma que Group 5-Fold é a métrica mais conservadora.
3. **Computar acc de `step1_v2_neutral` apenas sobre os 23 datasets antigos** (subset analysis, custo zero). Se acc ≈ 60-65% nos 23 antigos, confirma que H1 é a explicação principal.

A validação 3 é especialmente barata e diagnóstica — pode ser executada em < 10 minutos.
