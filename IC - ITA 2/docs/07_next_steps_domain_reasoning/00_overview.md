# Next Steps: Melhorar Domain Reasoning para Classificacao de Missing Data

**Data:** 2026-04-19
**Contexto:** Experimento forensic_neutral_v2 atingiu 56.2% (GroupKFold-5) e 63.1% (domain_prior alone). Analise forense confirmou ausencia de data leakage classico, mas revelou fraquezas sistematicas no domain reasoning da LLM.

## Diagnostico Atual

### Resultados forensic_neutral_v2 (NaiveBayes, GroupKFold-5)

| Cenario | Features | Accuracy | F1-macro |
|---------|----------|----------|----------|
| Baseline (estatistico) | 21 | 40.5% | — |
| + CAAFE | 25 | 47.6% | — |
| + LLM data-driven | 30 | 50.5% | 43.5% |
| + domain_prior | 31 | **56.2%** | **50.1%** |
| domain_prior sozinho | 1 | **63.1%** | **53.8%** |

### Fraquezas Identificadas

| Problema | Evidencia | Causa Raiz |
|----------|-----------|------------|
| **MAR bias** | 96.5% acuracia MAR, 30% MCAR, 34% MNAR | LLM defaults para MAR em qualquer dominio clinico |
| **MNAR_pima_insulin = 0%** | Caso classico MNAR completamente errado | Metadados neutros removeram contexto essencial |
| **MCAR quase nunca predito** | cylinderbands 2-6%, hypothyroid 0% | LLM "encontra razoes" para MAR em qualquer contexto |
| **Single-call instability** | Varia entre bootstraps do mesmo dataset | Uma chamada com temp=0.1 nao captura incerteza |
| **Paradoxo full < prior** | 31 features (56.2%) < 1 feature (63.1%) | Features estatisticas adicionam ruido que dilui o sinal |

### Oportunidade de Melhoria

O gap entre o resultado atual e o potencial e grande:
- **Accuracy atual:** 56.2% (full) / 63.1% (prior only)
- **F1-macro atual:** 50.1% / 53.8%
- **Teto teorico:** ~80-85% (limitado pela ambiguidade intrinseca MAR/MNAR)
- **Principal gargalo:** LLM nao distingue bem MCAR e MNAR

## Plano de Melhorias: 3 Steps Incrementais

Cada step e **independente** e **testavel** com o pipeline existente:

```
Step 1: Few-Shot + Tipologia MNAR ──> calibracao da distribuicao a priori
Step 2: Causal Reasoning Prompt    ──> raciocinio estruturado step-by-step
Step 3: Self-Consistency Voting    ──> multiplas perspectivas + agregacao
```

### Ordem de Execucao Recomendada

1. **Step 1** primeiro (menor esforco, maior impacto esperado na reducao do MAR bias)
2. **Step 2** segundo (reescreve o prompt core, complementa Step 1)
3. **Step 3** terceiro (multiplica chamadas, maior custo de API, maior robustez)

### Metricas de Sucesso

| Metrica | Atual | Target Step 1 | Target Step 2 | Target Step 3 |
|---------|-------|---------------|---------------|---------------|
| Accuracy GroupKFold-5 | 56.2% | 60%+ | 65%+ | 68%+ |
| F1-macro | 50.1% | 55%+ | 60%+ | 63%+ |
| MCAR accuracy LODO | ~30% | 45%+ | 50%+ | 55%+ |
| MNAR accuracy LODO | ~34% | 45%+ | 50%+ | 55%+ |
| MAR accuracy LODO | 96.5% | >90% | >90% | >90% |

## Documentos Detalhados

- [Step 1: Few-Shot + Tipologia MNAR](./01_step1_fewshot_tipologia.md)
- [Step 2: Causal Reasoning Prompt](./02_step2_causal_reasoning.md)
- [Step 3: Self-Consistency Voting](./03_step3_self_consistency.md)

## Referencias da Pesquisa

### Self-Consistency e Confidence
- Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
- "Confidence Improves Self-Consistency in LLMs" — ACL Findings 2025
- "Reasoning Aware Self-Consistency" — NAACL 2025

### Causal Reasoning com LLMs
- "Can LLMs Assist Expert Elicitation for Probabilistic Causal Modeling?" — arxiv 2504.10397
- "CARE: Turning LLMs Into Causal Reasoning Expert" — arxiv 2511.16016
- "LLM-Driven Causal Discovery via Harmonized Prior" — IEEE TKDE 2025

### Multi-Agent Debate
- Du et al. (2023) "Improving Factuality and Reasoning through Multiagent Debate" — ICML 2024
- "Adaptive Heterogeneous Multi-Agent Debate" — JKSU 2025
- "iMAD: Intelligent Multi-Agent Debate for Efficient LLM Inference" — arxiv 2511.11306

### Missing Data e LLMs
- "A Context-Aware Approach for Enhancing Data Imputation with PLMs" — COLING 2025
- "Data Imputation Based on Retrieval-Augmented Generation" — Applied Sciences 2025

### Few-Shot Prompting
- "The Few-shot Dilemma: Over-prompting Large Language Models" — arxiv 2509.13196
- "Fairness-guided Few-shot Prompting for LLMs" — Tencent AI Lab
