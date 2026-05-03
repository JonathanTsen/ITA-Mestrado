# Análise de Custos — ML × Flash × Pro

**Data:** 2026-04-25
**Foco:** quanto custa cada incremento de performance e qual configuração é Pareto-eficiente

---

## 1. Custos diretos (USD por 1 execução completa)

| Item | ML-only | Flash | Pro |
|------|:-------:|:-----:|:---:|
| Chamadas LLM | 0 | 2.842 | 2.842 |
| Tokens input avg/call | 0 | ~3.5K | ~3.5K |
| Tokens output avg/call | 0 | ~700 | ~700 |
| Total tokens input | 0 | ~10M | ~10M |
| Total tokens output | 0 | ~2M | ~2M |
| Pricing input ($/M) | $0 | $0.075 | $1.25 |
| Pricing output ($/M) | $0 | $0.30 | $5.00 |
| **Custo extração** | **$0** | **~$1.35** | **~$22.50** |
| Cost overhead (retry, cache miss) | $0 | ~$0.50-2 | ~$10-15 |
| **Total estimado** | **$0** | **~$2-4** | **~$30-36** |

## 2. Custos indiretos (tempo)

| Etapa | ML-only | Flash | Pro |
|-------|:-------:|:-----:|:---:|
| Extração de features estatísticas | < 1 min | ~30 min* | ~1h33min |
| Treino (7 modelos) | ~1 min | ~1 min | ~1 min |
| Merge (se 2 metades) | n/a | n/a | < 5s |
| **Total wall-clock** | **~1 min** | **~30 min** | **~1h35min** |

*Flash extraído anteriormente (`step10_flash_ca_neutral`); tempo estimado.

## 3. Performance × custo (Pareto)

### 3.1 Tabela de eficiência

| Configuração | Best CV | Δ vs ML | Custo | Custo/+1pp | Eficiência |
|--------------|:-------:|:-------:|:-----:|:----------:|:----------:|
| ML-only | 47.47% | — | $0 | — | ∞ (gratuito) |
| Flash + ML | 47.44% | **−0.03pp** | ~$3 | indefinido (gain ≈ 0) | **0** (Pareto-dominado) |
| Pro + ML | 49.33% | **+1.86pp** | ~$33 | **$17.74** | finito mas alto |

### 3.2 Visualização Pareto

```
Best CV
  ▲
50%│                                  Pro ★ ($33)
  │                              
49%│                              
  │                              
48%│                              
  │  ML ★ ($0)         Flash X ($3) ← dominado por ML
47%│
   └────┬─────┬─────┬─────┬─────► Custo USD
        $0    $10   $20   $30
```

**Conclusão:** Flash é **Pareto-dominado** por ML (mesmo CV, custo positivo). Apenas duas configurações ficam na fronteira de Pareto: **ML-only** (eficiência infinita) e **Pro** (eficiência limitada mas direcionalmente boa).

## 4. Análise de breakeven

**Pergunta:** quantas runs precisaríamos antes de Pro pagar pelo seu custo (em valor de "+1pp confiabilidade")?

### Suposições

- 1 publicação aceita "vale" $X em valor científico
- Ganho de Pro sobre ML: +1.86pp CV / +0.25pp holdout
- Probabilidade de aceitação aumenta marginalmente com cada +1pp accuracy

### Cálculo back-of-envelope

Se cada +1pp de CV aumenta probabilidade de aceitação em journal Q2 em ~5% (estimativa otimista), e uma aceitação em Q2 vale digamos $10K (em prestígio + carreira + fellowship $$):

```
Expected value de Pro vs ML = 0.05 × 1.86 × $10K = $930
Custo de Pro: $33
Razão valor/custo: 28×
```

**Conclusão:** Pro é **claramente cost-effective para validação final em tese/paper**. Não é cost-effective para iteração rápida de pesquisa (onde o ganho cumulativo é o que importa, não cada run individual).

## 5. Custo por modelo de uso

### 5.1 Iteração de pesquisa (várias runs)

**Cenário:** rodar 10 variantes de prompt para encontrar a configuração ótima.

| Configuração | Custo total (10 runs) | Tempo total | Best CV final esperado |
|--------------|:---------------------:|:-----------:|:----------------------:|
| ML-only (10× sklearn) | $0 | ~10 min | 47.47% (sem variação) |
| Flash (10× ~$3) | $30 | ~5h | 47.44% (variação 47-48%) |
| Pro (10× ~$33) | **$330** | **~16h** | 49-51% (variação 49-51%) |

**Conclusão:** para iteração, **ML-only é uma baseline de fronteira**. Flash é desperdício. Pro é viável apenas com orçamento dedicado.

### 5.2 Validação final única

**Cenário:** uma execução final para reportar números na tese/paper.

| Configuração | Custo | Tempo | Best CV |
|--------------|:-----:|:-----:|:-------:|
| ML-only | $0 | 1 min | 47.47% |
| Flash | $3 | 30 min | 47.44% |
| **Pro** | **$33** | **1h35min** | **49.33%** |

**Conclusão:** **Pro é a escolha clara** para reporting final.

### 5.3 Baseline de comparação

**Cenário:** mostrar que LLM agrega valor (ablation para reviewers).

| Configuração | Justificativa |
|--------------|---------------|
| **ML-only** | Baseline sem LLM (essencial para tese mostrar ganho do LLM) |
| **Pro** | Configuração final com LLM |
| ~~Flash~~ | ~~Não justifica reporting~~ — confunde sem agregar (já que ≈ ML-only) |

## 6. Custos amortizados (re-uso de features)

Uma vez extraído o `X_features.csv` de Pro (custo $33), pode-se:

1. **Treinar variantes ML diferentes:** 0 custo adicional (basta retreinar)
2. **Testar Cleanlab + stacking:** 0 custo adicional (apenas pós-processamento)
3. **Análise per-classe / SHAP:** 0 custo adicional (apenas análise)

**Custo amortizado por análise:** se fizermos 5 análises diferentes a partir do mesmo X_features, o custo efetivo é $33/5 = **$6.60 por análise** — competitivo.

## 7. Comparação histórica de custos

| Experimento | Modelo | Custo estimado | CV avg |
|-------------|--------|:--------------:|:------:|
| `forensic_neutral_v2` (23 datasets) | Pro neutral | ~$25-30 | 56.2% |
| `step10_flash_ca_neutral` (29) | Flash neutral | ~$3-5 | 47.44% |
| `step1_v2_neutral` (29) | Pro Step1 neutral | $30-36 | 49.33% |
| `step08_flash_sc` (LEAKED, descartado) | Flash SC | ~$8 | 92.3% (inválido) |
| `step09_flash_sc_neutral` (29) | Flash SC | ~$7 | 38.4% (regrediu) |

**Insight histórico:** Pro com benchmark menor (23 datasets) entrega 56% CV por $30. Pro com benchmark expandido (29) entrega 49% por mesmos $30. **Custo escala linearmente com bootstraps; performance não escala.**

## 8. Recomendação financeira

### 8.1 Para a defesa de mestrado

**Investir em Pro x 1 run final** ($33) para garantir números reportáveis. Iterar Steps 2/3 com ML-only (custo $0) para protótipos.

### 8.2 Para escrita do paper

**1× Pro Step1 (já temos)** + **1× Pro Step2 quando implementado** ($33-36) = total $33-66 para validação completa.

### 8.3 Para iteração contínua

**Investir em fine-tuning de modelo menor** (e.g., Llama 7B local) — eliminaria custo per-run após investimento inicial em hardware. Estimativa: $500-1000 hardware + 0 custo per-run.

## 9. Análise de sensibilidade

### 9.1 E se pricing dos modelos mudar?

| Cenário | Flash custo | Pro custo | Decisão |
|---------|:-----------:|:---------:|---------|
| Pricing atual | $3 | $33 | Pro só para final |
| Flash 50% off | $1.50 | $33 | Flash continua dominado por ML-only ($0) |
| Pro 50% off | $3 | $16.50 | Pro torna-se viável para iteração |
| Pro 80% off | $3 | $6.60 | Pro torna-se default para tudo |

### 9.2 E se Pro melhorasse +5pp ao invés de +1.86pp?

| Δ Pro vs ML | Custo/+1pp | Decisão |
|-------------|:----------:|---------|
| +1.86pp (atual) | $17.74 | Apenas final |
| +3pp | $11.00 | Viável para iteração |
| +5pp | $6.60 | Default |
| +10pp | $3.30 | Sempre |

**Implicação:** se Step 2 (Causal DAG) elevar Pro de 49.3% → ~55%, o custo/+1pp cai para ~$5 — passando a ser viável para uso operacional contínuo.

## 10. Síntese

| Decisão | Recomendação |
|---------|--------------|
| Configuração para defesa | **Pro Step 1** ($33) — única que justifica número reportado |
| Configuração para iteração | **ML-only** ($0) ou **Pro com cache de features** ($33 amortizado) |
| Configuração para descartar | **Flash** (Pareto-dominado por ML) |
| Próximo investimento | **Step 2 com Pro** ($33-36) — se elevar CV >55%, torna Pro econômico |
| Investimento de longo prazo | **Fine-tuning de modelo menor** ($500-1K hardware, $0 per-run) |
