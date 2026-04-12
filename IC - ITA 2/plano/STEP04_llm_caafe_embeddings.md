# STEP 04: LLM Reformulado (CAAFE, Embeddings, Prompt)

**Fase 4D — Fazer o LLM contribuir de verdade**
**Status: CRITICO — prioridade elevada apos resultados do STEP02**

---

## Problema

O LLM atual recebe estatisticas pre-computadas e retorna 8 scores de confianca. Isso e redundante — as estatisticas ja contem o sinal, e o LLM nao adiciona informacao nova. Contribuicao: apenas 6.6% de importancia no RF.

**Agravamento pos-STEP02 (2026-04-12):** Com as 21 features invariantes, o LLM nao so nao ajuda como **piora drasticamente**: -20 a -26pp em accuracy (87% → 61-67%). As 8 features LLM adicionam ruido que confunde os classificadores. Sem reformulacao, a tese nao consegue demonstrar contribuicao positiva do LLM.

A literatura (CAAFE, Enriching Tabular Data) mostra abordagens fundamentalmente diferentes que geram valor real.

---

## Abordagem 1: CAAFE-style — LLM Gera Codigo Python

### Fonte
CAAFE (Hollmann et al., NeurIPS 2023): LLM gera codigo Python para novas features, avaliadas iterativamente.

### Logica

Em vez de pedir ao LLM "classifique este padrao", pedir "gere uma funcao Python que cria uma nova feature".

**Input para o LLM:**
- Descricao do problema (classificar mecanismo de missing)
- Lista de features atuais e seus nomes
- Accuracy atual do melhor modelo
- Algumas linhas de exemplo do DataFrame

**Output do LLM:**
- Codigo Python de uma funcao `def create_feature(df) -> Series`
- Explicacao do por que a feature ajudaria

**Loop iterativo:**
1. Pedir ao LLM para gerar 1 feature
2. Executar o codigo no DataFrame
3. Treinar modelo com feature adicionada
4. Se accuracy (CV) melhorou → manter feature
5. Se nao melhorou → descartar
6. Repetir ate N iteracoes ou sem melhoria

### Por que e melhor que a abordagem atual

- O LLM contribui **creatividade** (transformacoes nao-obvias) em vez de **classificacao** (redundante)
- Cada feature e testada empiricamente — so fica se melhora
- As features geradas sao interpretaveis (tem codigo e explicacao)
- Escala naturalmente: mais iteracoes = mais features candidatas

### Riscos e mitigacao

- LLM pode gerar codigo com erros → executar em sandbox com try/except
- Features geradas podem ser redundantes → verificar correlacao com features existentes
- Codigo pode ser computacionalmente caro → timeout por feature

---

## Abordagem 2: LLM Embeddings

### Fonte
Enriching Tabular Data with LLM Embeddings (Kasneci, 2024): usar embeddings de LLM como features adicionais.

### Logica

1. **Serializar** o resumo estatistico de cada amostra como texto (ex: "missing_rate: 0.05, X0_mean: 0.42, corr_X1_mask: 0.31, ...")
2. **Gerar embedding** passando o texto por um LLM (Gemini/GPT) e extraindo o vetor de representacao interna
3. **Reduzir dimensionalidade** com PCA (768 dims → 10-20 componentes)
4. **Selecionar** as componentes mais informativas via RF importance
5. **Concatenar** com features estatisticas

### Por que e melhor que a abordagem atual

- O embedding captura relacoes nao-lineares entre as estatisticas que nao sao explicitas nos 8 scores
- Nao depende de um prompt bem calibrado — o LLM "entende" o texto naturalmente
- PCA + selection garante que so componentes informativas ficam

### Decisoes de design

- Qual LLM usar? Gemini tem API de embeddings? Se nao, usar sentence-transformers local (all-MiniLM-L6-v2) como alternativa gratuita
- Quantas componentes PCA? Comecar com variancia explicada 95%, depois testar fixo (10, 15, 20)
- Caching: embeddings sao determinísticos para mesmo input → cachear agressivamente

---

## Abordagem 3: Prompt Reformulado (fallback)

### Logica

Se CAAFE e embeddings nao funcionarem, reformular o prompt atual para gerar features **complementares** em vez de **classificatorias**:

**Atual (redundante):** "Qual a confianca MCAR/MAR/MNAR?" → gera llm_mcar_conf, llm_mar_conf, llm_mnar_conf

**Novo (complementar):** Perguntas sobre **padroes** que as estatisticas nao capturam diretamente:
- "Os dados faltantes parecem concentrados em uma regiao especifica?" (0-1)
- "O padrao de missing parece aleatorio ou estruturado?" (0-1)  
- "Ha evidencia de censura (valores extremos omitidos)?" (0-1)
- "A relacao entre missing e X1 parece causal ou espuria?" (0-1)

**Por que seria melhor:** Features sobre padroes qualitativos vs features que sao a resposta final.

---

## Ordem de execucao

1. **Primeiro testar CAAFE-style** — maior potencial de impacto, gera features verificavelmente uteis
2. **Se CAAFE nao funcionar, testar embeddings** — menor esforco de prompt engineering
3. **Se nenhum funcionar, testar prompt reformulado** — menor mudanca no pipeline
4. **Se nenhum funcionar, remover LLM** — focar em features estatisticas (que ja devem ser melhores apos Steps 2-3)

---

## Ablacao Final Comparativa

Apos implementar a melhor abordagem LLM, rodar 5 experimentos:

| Experimento | Features | Objetivo |
|-------------|----------|----------|
| E1 | Apenas discriminativas originais (6) | Baseline minimo |
| E2 | + invariantes + MNAR (15) | Impacto das novas features |
| E3 | + MechDetect (21) | Impacto do MechDetect |
| E4 | + LLM (21 + N_llm) | Impacto do LLM |
| E5 | Full stack | Tudo junto |

Para cada experimento, salvar outputs completos (STEP01) e comparar:
- Accuracy por modelo
- Recall MNAR
- Feature importance das features LLM
- CV variancia

---

## Testes de Validacao

### Teste 1: CAAFE — Feature gerada e valida
Rodar 1 iteracao CAAFE. Verificar que o codigo Python gerado executa sem erro e produz uma Series com mesmo indice que o DataFrame. A feature nao deve ser constante (std > 0).

### Teste 2: CAAFE — Feature melhora accuracy
Das features geradas em 5 iteracoes, pelo menos 1 deve melhorar accuracy em CV (senao o LLM nao esta gerando features uteis para este dominio).

### Teste 3: Embeddings — Dimensionalidade reduzida
Apos PCA, verificar que 10-20 componentes explicam >90% da variancia. Se nao, o embedding nao e informativo para este dominio.

### Teste 4: Embeddings — Melhora vs baseline
Accuracy com embeddings deve ser >= accuracy sem embeddings para pelo menos 4/7 modelos.

### Teste 5: Ablacao — LLM nao piora
Comparar E3 (sem LLM) vs E4 (com LLM). LLM nao deve piorar accuracy em mais de 2/7 modelos. Se piorar, a abordagem LLM nao funciona e deve ser descartada.

### Teste 6: Importancia LLM
No experimento E4, features LLM devem ter importancia total > 10% no RF. Se < 5%, nao estao contribuindo.

---

## Criterio de Conclusao

- [ ] Pelo menos 1 das 3 abordagens testada end-to-end
- [ ] Ablacao com 5 experimentos completa
- [ ] Decisao documentada: qual abordagem LLM usar (ou nenhuma)
- [ ] Se LLM escolhido: delta vs baseline >= 0% em 5+/7 modelos
- [ ] Testes relevantes passam
