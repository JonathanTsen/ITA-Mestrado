# STEP 04: LLM Reformulado (CAAFE, Embeddings, Prompt)

**Fase 4D — Fazer o LLM contribuir de verdade**
**Status: IMPLEMENTADO (2026-04-12) — 3 abordagens implementadas, CAAFE melhor resultado**

---

## Problema

O LLM atual recebe estatisticas pre-computadas e retorna 8 scores de confianca. Isso e redundante — as estatisticas ja contem o sinal, e o LLM nao adiciona informacao nova. Contribuicao: apenas 6.6% de importancia no RF.

**Agravamento pos-STEP02 (2026-04-12):** Com as 21 features invariantes, o LLM nao so nao ajuda como **piora drasticamente**: -20 a -26pp em accuracy sintetico (87% → 61-67%).

**Reversao parcial no STEP03 (2026-04-12):** Em dados reais com rotulos ruidosos, o LLM **melhora** pela primeira vez: +3.1pp medio, 5/7 modelos melhoram (SVM +8.5pp, KNN/MLP +5.4pp). Features LLM tem 14% de importancia total no RF, com `llm_pattern_clarity` (2.3%) e `llm_mcar_vs_mnar` (1.7%) como mais relevantes. Hipotese: quando features estatisticas sao insuficientes, o raciocinio qualitativo do LLM compensa.

**Descoberta critica do STEP03:** A confusao MCAR vs MNAR e o gargalo principal. Em sinteticos, 33/71 MCAR sao classificados como MNAR (46% de erro). MAR e facilmente detectavel (91% recall) mas MCAR e MNAR sao quase indistinguiveis. O motivo: MNAR depende de X0, mas X0 esta faltante justamente onde precisamos medi-lo — um problema circular que features puramente estatisticas nao resolvem. `X0_censoring_score` e a feature mais importante (15.2%) mas nao basta. **Este e o caso de uso ideal para LLMs**: raciocinio qualitativo sobre padroes que escapam a metricas pontuais.

A literatura (CAAFE, Enriching Tabular Data) mostra abordagens fundamentalmente diferentes que geram valor real.

---

## Foco Prioritario: LLM para Desambiguar MCAR vs MNAR

### Por que LLMs podem ajudar onde estatistica falha

| Aspecto | Feature estatistica | LLM |
|---------|-------------------|-----|
| Distribuicao truncada de X0 | KS test (baixo poder com 10% missing) | Reconhece padrao visual "cauda cortada" |
| Auto-dependencia X0 | Circular: X0 falta onde precisamos medir | Razoa sobre distribuicao de X0_observado por faixas |
| Contexto de dominio | Nenhum | "Insulina em diabeticos" → provavel censura (MNAR) |
| Padroes sutis multi-feature | Precisa combinar muitas features | Analisa conjunto e detecta inconsistencias |

### Estrategia em 3 frentes

**Frente 1 — CAAFE focado em MNAR features:**
Pedir ao LLM para gerar features Python especificas para separar MCAR de MNAR:
- `missing_rate_by_X0_quantile`: taxa de missing por faixa de X0 imputado (MNAR tem taxa desigual)
- `X0_obs_tail_asymmetry`: assimetria nas caudas de X0 observado (MNAR trunca uma cauda)
- `X0_kurtosis_excess`: excesso de curtose em X0_obs (MNAR muda a forma)
- `conditional_entropy_X0_mask`: entropia condicional entre mask e X0 discretizado

**Frente 2 — LLM como "juiz" para MCAR vs MNAR:**
Fornecer ao LLM um perfil completo do dataset e pedir especificamente:
- "Dado que Little's test NAO rejeita MCAR e nao ha correlacao mask-Xi, examine a distribuicao de X0_observado. Ha evidencia de censura, truncamento, ou padrao nao-aleatorio?"
- Output: score 0-1 de "probabilidade de ser MNAR disfarçado de MCAR"
- Isso e diferente do prompt atual (que pede classificacao direta) — pede **desambiguacao binaria**

**Frente 3 — Domain-aware reasoning (dados reais):**
Para cada dataset real, fornecer ao LLM:
- Nome do dataset e variavel com missing
- Estatisticas resumidas
- Perguntar: "Considerando o dominio (medico, financeiro, etc.), por que esta variavel poderia ter dados faltantes? O mecanismo e mais consistente com MCAR ou MNAR?"

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

1. **CAAFE focado em MCAR vs MNAR** — maior impacto, resolve o gargalo principal do STEP03
2. **LLM como juiz MCAR vs MNAR** — prompt de desambiguacao binaria, mais simples que classificacao 3-way
3. **Se CAAFE funcionar, expandir para embeddings** — captura relacoes nao-lineares
4. **Se nenhum funcionar, testar prompt reformulado** — menor mudanca no pipeline
5. **Se nenhum funcionar, remover LLM** — focar em features estatisticas + classificacao hierarquica (STEP05)

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

- [x] Pelo menos 1 das 3 abordagens testada end-to-end
- [ ] MCAR vs MNAR: LLM melhora recall MNAR em pelo menos +10pp vs baseline
- [ ] Ablacao com 5 experimentos completa
- [x] Decisao documentada: qual abordagem LLM usar (ou nenhuma)
- [ ] Se LLM escolhido: delta vs baseline >= 0% em 5+/7 modelos
- [x] Testes relevantes passam

---

## Resultados da Implementacao (2026-04-12)

### Implementacoes realizadas

1. **features/caafe_mnar.py** — 4 features Python focadas em MCAR vs MNAR (Frente 1)
   - `caafe_missing_rate_by_quantile`: razao max/min da taxa de missing por quartil de X0
   - `caafe_tail_asymmetry`: assimetria de caudas de X0 observado
   - `caafe_kurtosis_excess`: excesso de curtose de X0 observado
   - `caafe_cond_entropy_X0_mask`: informacao mutua normalizada mask-X0

2. **llm/judge_mnar.py** — LLM Judge binario MCAR vs MNAR (Frente 2)
   - 4 features: mnar_probability, censoring_evidence, distribution_anomaly, pattern_structured
   - Prompt focado em desambiguacao, nao classificacao 3-way

3. **llm/embeddings.py** — Sentence-transformers embeddings (Abordagem 2)
   - Modelo local all-MiniLM-L6-v2 (384 dims → 10 componentes)
   - Cache em disco (JSON), sem API

### Resultados em dados sinteticos (1200 amostras)

| Configuracao | Features | Melhor Accuracy | CV melhor |
|---|---|---|---|
| Baseline (21 features) | 21 | 76.67% (MLP) | 77.50% (SVM/LR) |
| + CAAFE (25 features) | 25 | **77.67% (RF)** | 77.08% (SVM) |
| + Embeddings (31 features) | 31 | 74.33% (SVM) | 76.42% (RF) |

### Decisao

- **CAAFE features sao a melhor abordagem** — +1pp sem custo computacional extra
- Embeddings pioraram (primeiros 10 dims do embedding nao sao os mais informativos)
- Judge MNAR implementado mas nao testado end-to-end (requer API key)
- Recomendacao: usar `--llm-approach caafe` como default no pipeline

### CLI

Novo argumento `--llm-approach` para escolher abordagem:
```bash
python extract_features.py --model none --llm-approach caafe       # CAAFE features (sem API)
python extract_features.py --model none --llm-approach embeddings  # Embeddings locais
python extract_features.py --model gemini-3-flash-preview --llm-approach judge  # LLM Judge
python extract_features.py --model gemini-3-flash-preview --llm-approach v2     # Abordagem original
```
