# Avaliação de Publicabilidade

**Data:** 2026-04-19

---

## 1. Para Dissertação de Mestrado: CLARAMENTE SUFICIENTE

Uma dissertação de mestrado no ITA (PPGPO) não exige resultados state-of-the-art. Exige demonstração de competência em pesquisa. Este trabalho atende todos os critérios:

| Critério | Status | Evidência |
|----------|:------:|-----------|
| Problema bem definido e relevante | ✅ | Classificação 3-way de mecanismos de missing data — lacuna na literatura |
| Revisão de literatura adequada | ✅ | MechDetect, PKLM, CAAFE, Little's test, Rubin 1976 |
| Metodologia rigorosa e reproduzível | ✅ | GroupKFold, LODO, bootstrap CIs, auditoria forense, metadata neutralizada |
| Resultados honestos com análise crítica | ✅ | Data leakage detectado e corrigido; resultados negativos documentados |
| Contribuição original | ✅ | LLM domain reasoning para missing data; CAAFE features; auditoria de labels |
| Resultados negativos documentados | ✅ | LLM features estatísticas falham; mecanismo de falha identificado |
| Reprodutibilidade | ✅ | Código, dados sintéticos, checkpoint system, features salvas |

### Diferencial para a banca

A **jornada de descoberta** (pipeline inflado → correção honesta → features melhoradas → domain reasoning) demonstra maturidade científica excepcional para mestrado. A capacidade de identificar e corrigir os próprios erros (data leakage) é mais valiosa que alcançar accuracy alta.

---

## 2. Para Journal: PUBLICÁVEL COM A NARRATIVA CORRETA

### Narrativa que NÃO funciona
- ❌ "Alcançamos 56% de accuracy na classificação de mecanismos"
- ❌ "LLMs melhoram classificação de missing data"
- ❌ "Resolvemos o problema de classificação automática"
- ❌ "Nosso método é state-of-the-art"

### Narrativa que FUNCIONA
- ✅ "LLMs demonstram domain reasoning genuíno para classificação de mecanismos de missing data"
- ✅ "Features determinísticas especializadas (CAAFE) superam LLM features para este problema"
- ✅ "A classificação tem um teto fundamental, e o domain reasoning é a chave para superá-lo"
- ✅ "Benchmarks de missing data têm 57% de labels inconsistentes"
- ✅ "Resultado negativo: LLM features via análise estatística de segunda ordem não funcionam"

### Ângulos de publicação possíveis

**Ângulo 1: Domain Reasoning (melhor para journal de ML/AI)**
> "Can LLMs Reason About Missing Data? Domain Knowledge vs Statistical Analysis for Mechanism Classification"
- Foco: LLM como domain expert, ablação domain_prior vs features estatísticas
- Resultado negativo sobre features LLM como complemento forte

**Ângulo 2: Benchmark + Auditoria (melhor para journal de Estatística)**
> "How Reliable Are Missing Data Mechanism Labels? An Empirical Audit of 23 Benchmark Datasets"
- Foco: 57% de inconsistência, implicações para pesquisa, teto teórico
- Método de validação automática como contribuição

**Ângulo 3: Método Completo (melhor para journal aplicado)**
> "Automatic Classification of Missing Data Mechanisms Using Statistical and LLM-Derived Features"
- Foco: pipeline completo, CAAFE features, comparação com baselines

---

## 3. Journals Alvo (em ordem de adequação)

### Tier 1 — Melhor encaixe

| Journal | IF | Foco | Por que encaixa |
|---------|:--:|------|----------------|
| **Statistics and Computing** (Springer) | ~2.0 | Métodos estatísticos computacionais | Classificação de mecanismos + ML + LLM. Aceita contribuições metodológicas |
| **Computational Statistics & Data Analysis** (Elsevier) | ~1.8 | Métodos computacionais para análise de dados | Features para missing data + comparação rigorosa |

### Tier 2 — Mais competitivo, maior impacto

| Journal | IF | Foco | Por que encaixa |
|---------|:--:|------|----------------|
| **Machine Learning** (Springer) | ~4.9 | ML puro | LLM como feature extractor + resultado negativo bem documentado |
| **JMLR** | ~6.0 | ML metodológico | Aceita resultados negativos rigorosos; contribuição timely sobre LLMs |
| **Statistical Science** | ~5.7 | Discussão estatística | Artigo de review/discussion sobre classificação de mecanismos |

### Tier 3 — Mais acessível

| Journal | IF | Foco | Por que encaixa |
|---------|:--:|------|----------------|
| **Statistical Analysis and Data Mining** (Wiley) | ~1.5 | Aplicado | Aceita contribuições modestas mas bem fundamentadas |
| **Journal of Statistical Software** | ~5.4 | Software + método | Se o pipeline for empacotado como biblioteca Python |
| **Data Mining and Knowledge Discovery** | ~4.0 | Aplicações de DM | Pipeline aplicado com validação rigorosa |

### Conferências

| Conferência | Tipo | Por que encaixa |
|-------------|------|----------------|
| **AISTATS** | Top venue ML+Stats | Interseção ML-Estatística, aceita contribuições metodológicas |
| **NeurIPS Workshop** (Missing Data / Tabular) | Workshop | Resultado negativo sobre LLMs é timely |
| **ICML Workshop** (LLMs for Tabular Data) | Workshop | Domain reasoning como caso de uso de LLM |
| **AAAI** | Top venue AI | Aplicação de LLM com resultado nuanced |

---

## 4. Elementos que Fortalecem a Submissão

### Para qualquer journal

1. **Ablação rigorosa** com 5 cenários e 2 métricas de validação (GroupKFold + LODO)
2. **Comparação com baselines publicados** (MechDetect, PKLM) nos mesmos dados
3. **Resultado negativo** bem documentado — journals valorizam honestidade
4. **Auditoria forense** de leakage — demonstra rigor incomum
5. **Reprodutibilidade** — código, dados, checkpoint system

### Elementos faltantes (para fortalecer)

1. **Teste de significância estatística** entre cenários (paired t-test ou Wilcoxon entre folds)
2. **Mais datasets** — 23 é aceitável mas 50+ seria mais convincente
3. **Análise de sensibilidade** ao modelo LLM (testar com GPT-4, Claude, Llama)
4. **Empacotamento como biblioteca** — aumenta impacto e citações

---

## 5. Timeline Sugerida

| Fase | Ação | Prioridade |
|------|------|:----------:|
| 1 | Escrever dissertação (Caps 1-6) | **Alta** |
| 2 | Extrair artigo da dissertação (Ângulo 1 ou 3) | Alta |
| 3 | Submeter a Statistics and Computing ou CSDA | Média |
| 4 | Se rejeitado, resubmeter a Statistical Analysis and Data Mining | Backup |
| 5 | Considerar workshop NeurIPS/ICML para resultado negativo | Opcional |
