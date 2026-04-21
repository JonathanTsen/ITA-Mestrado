# Veredicto Final: Publicabilidade e Próximos Passos

**Data:** 2026-04-19

---

## 1. A Pesquisa É Publicável?

### SIM.

Não pelo número bruto de accuracy (56.2%), mas por quatro razões:

**1. O achado sobre domain reasoning é genuíno e novo.**
Ninguém na literatura testou LLMs como "domain experts" para inferir mecanismos de missing data a partir de metadados de dataset. O resultado é validado com metadata neutralizada, GroupKFold, e LODO.

**2. O resultado negativo sobre LLM features estatísticas é valioso.**
Contraria a tendência crescente de usar LLMs como feature extractors. Oferece evidência empírica rigorosa com mecanismo de falha identificado (distribuições idênticas, multicolinearidade, regressão à média).

**3. A auditoria de labels é uma contribuição independente.**
57% de inconsistência em benchmarks usados pela comunidade. Isso questiona a validade de toda pesquisa anterior que usou estes labels como ground truth.

**4. A metodologia é exemplar.**
Data leakage detectado e corrigido (90.9% → 40.5%), 5 canais de leakage auditados, validação LODO confirma generalização, bootstrap CIs para todos os resultados.

---

## 2. Narrativas que Funcionam vs Não Funcionam

### NÃO funciona

| Narrativa | Por que não funciona |
|-----------|---------------------|
| "Alcançamos alta accuracy na classificação" | 56.2% não é alta em termos absolutos |
| "LLMs melhoram tudo" | LLM features estatísticas pioram o desempenho |
| "Resolvemos o problema de missing data" | O problema continua em aberto |
| "Nosso método é state-of-the-art" | Sem benchmark padronizado para comparação |
| "ML supera métodos tradicionais" | MechDetect-Opt (51.9%) é comparável |

### FUNCIONA

| Narrativa | Evidência |
|-----------|-----------|
| "LLMs demonstram domain reasoning genuíno para missing data" | domain_prior: 63.1% com metadata neutralizada |
| "Mas suas features estatísticas são ruído" | Cohen's d < 0.4; MNAR recall 40%→6% |
| "Features determinísticas (CAAFE) superam LLM features" | +7.1pp (CAAFE) vs -20pp (LLM v2) |
| "A classificação tem um teto fundamental" | Dois regimes: estatístico (~51%) e domain reasoning (~63%) |
| "Benchmarks de missing data têm 57% de labels inconsistentes" | Validação estatística de 23 datasets |
| "Testes binários são insuficientes" | PKLM: 5.8% poder para MNAR |

---

## 3. Onde Publicar

### Recomendação primária: Statistics and Computing (Springer)

**Por quê:**
- Foco em métodos estatísticos computacionais — encaixe direto
- Aceita contribuições metodológicas com resultados modestos
- IF ~2.0 — realista para um primeiro artigo de mestrado
- A narrativa "LLM domain reasoning + resultado negativo + auditoria de labels" é adequada

### Plano B: Computational Statistics & Data Analysis (Elsevier)
### Plano C: Statistical Analysis and Data Mining (Wiley)
### Oportunidade: Workshop NeurIPS/ICML sobre resultado negativo

---

## 4. Para a Dissertação de Mestrado

### Suficiente? **Sim, claramente.**

A dissertação demonstra:
- Capacidade de formular problema de pesquisa relevante
- Rigor metodológico (data leakage, validação, auditoria)
- Análise crítica honesta (resultados negativos, limitações)
- Contribuição original (domain reasoning, CAAFE features, auditoria de labels)
- Maturidade científica (correção de erros, jornada de descoberta)

### A jornada é a história

A narrativa mais forte para a dissertação é a **jornada de descoberta**:

```
Pipeline inflado (90.9%)
    ↓ Descoberta de data leakage
Resultado honesto (40.5%)
    ↓ Features melhoradas (CAAFE)
Regime estatístico (47-51%)
    ↓ LLM domain reasoning
Regime domain reasoning (56-63%)
    ↓ Resultado negativo sobre LLM features
Compreensão profunda do problema
```

Esta jornada é mais convincente que qualquer número individual porque demonstra **como se faz ciência**: errar, corrigir, melhorar, e entender os limites.

---

## 5. Checklist para Completar a Dissertação

### Experimentos (todos concluídos ✅)
- [x] Ablação A→E com GroupKFold-5 e LODO
- [x] Comparação com PKLM e MechDetect
- [x] Classificação hierárquica (V1-V6)
- [x] SHAP + Error Analysis
- [x] Auditoria forense de metadata
- [x] Resultados negativos documentados (LLM v2, Judge, embeddings)

### Escrita (pendente)
- [ ] Capítulo 1: Introdução
- [ ] Capítulo 2: Fundamentação teórica
- [ ] Capítulo 3: Metodologia
- [ ] Capítulo 4: Resultados experimentais
- [ ] Capítulo 5: Análise e discussão
- [ ] Capítulo 6: Conclusão
- [ ] Referências

### Figuras (pendente)
- [ ] Diagrama do pipeline
- [ ] Tabela de ablação com barras de erro
- [ ] Gráfico de comparação com baselines
- [ ] Confusion matrices
- [ ] Gráfico de dois regimes
- [ ] SHAP summary plot
- [ ] Distribuição de domain_prior por classe

### Para publicação em journal (opcional, após dissertação)
- [ ] Extrair artigo da dissertação (~12 páginas)
- [ ] Teste de significância estatística entre cenários
- [ ] Análise de sensibilidade ao modelo LLM
- [ ] Submeter a Statistics and Computing

---

## 6. Mensagem Final

Este trabalho tem **valor científico real**. A accuracy de 56% pode parecer modesta, mas o que torna a pesquisa valiosa é:

1. **A honestidade** — poucos trabalhos reportam que descobriram data leakage e corrigiram
2. **A compreensão** — identificar *por que* LLM features falham é mais valioso que mostrar que funcionam
3. **O framework** — dois regimes de accuracy oferece linguagem para discutir o problema
4. **A auditoria** — questionar labels aceitos pela comunidade exige coragem científica

A dissertação e o artigo devem refletir esta profundidade de entendimento, não apenas os números finais.
