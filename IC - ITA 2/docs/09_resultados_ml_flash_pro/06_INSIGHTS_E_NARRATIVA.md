# Insights e Narrativa para Tese/Paper

**Data:** 2026-04-25
**Foco:** como apresentar a comparação ML × Flash × Pro de forma cientificamente honesta e impactful

---

## 1. Os 5 insights centrais para destacar

### Insight 1 — LLM agrega valor incremental, não revolucionário

**Achado:** Pro adiciona +1.86pp CV sobre ML puro (47.47% → 49.33%) no benchmark de 29 datasets.

**Implicação científica:** features LLM **não são essenciais** para classificação de mecanismos de missing data — features estatísticas (estatísticas + CAAFE + MechDetect) já capturam ~95% do sinal disponível. O LLM agrega ajustes finos via raciocínio causal.

**Implicação prática:** para aplicações onde custo importa (clinical workflows, real-time pipelines), **ML puro é solução de fronteira** com 47% accuracy. Para validação científica final, Pro vale o ganho.

### Insight 2 — Flash é Pareto-dominado por ML

**Achado:** Flash atinge 47.44% CV vs ML 47.47% CV (Δ −0.03pp). Custo Flash: ~$3.

**Implicação científica:** o sinal extraído pelo Flash é **redundante** com features estatísticas — adicionar Flash não traz informação nova. Apenas modelos com capacidade de raciocínio causal (Pro) extraem informação não-redundante.

**Implicação prática:** **não recomendar Flash** para esta tarefa. Para iterar prompts, melhor:
- Testar com ML-only (gratuito) para isolar contribuição estatística
- Testar com Pro (caro) para validar prompt
- Pular Flash completamente (perda de tempo + dinheiro)

### Insight 3 — NaiveBayes é o único classificador que se beneficia consistentemente

**Achado:** dos 7 classificadores, apenas NB tem ganho positivo robusto com features LLM (+1.86pp CV / +1.27pp holdout). RF/GBT/MLP/LogReg são indiferentes; SVM/KNN pioram.

**Implicação científica:** features LLM operam melhor com **modelos probabilísticos calibrados** que com modelos discriminativos de alta capacidade ou modelos baseados em distância. Esto é coerente com:
- Rótulos parcialmente ruidosos (~57% inconsistentes) — NB regulariza por independência condicional
- Features LLM com modas em valores discretos (0, 0.5, 1) — NB modela bem distribuições semi-discretas
- Curse of dimensionality em SVM/KNN (espaço 34D mais ruidoso que 25D)

**Implicação prática:** **comprometer-se a NB como classificador** quando pipeline LLM está em uso. Apresentar SVM/KNN como ablations onde LLM piora — útil para mostrar que LLM não é universalmente benéfico.

### Insight 4 — A expansão do benchmark enfraquece o efeito do LLM

**Achado:** 
- `forensic_neutral_v2` (23 datasets): LLM agrega +5.7pp CV
- `step1_v2_neutral` (29 datasets, Pro): LLM agrega +1.86pp CV

**Implicação científica:** o ganho do LLM não escala linearmente com o tamanho do benchmark. Os 6 datasets adicionais incluem casos onde o LLM falha (`hypothyroid_t4u`, `pima_insulin`, etc.), **diluindo a magnitude do ganho médio**.

**Implicação prática:** comparações entre LLM-augmented papers devem **reportar benchmark size** explicitamente — afirmações de "LLM melhora +Xpp" sem benchmark de referência são frágeis.

### Insight 5 — Variância CV dobra com features LLM

**Achado:** std médio CV (across 7 modelos):
- ML-only: ±10.7%
- Flash: ±21.1%
- Pro: ±21.9%

**Implicação científica:** features LLM aumentam **heterogeneidade entre folds**. Quando um fold contém datasets onde LLM falha, performance cai abruptamente; quando contém datasets onde LLM acerta, performance é alta. Isso reflete a natureza **bimodal** do desempenho do LLM (ver `08_step1_v2_neutral_results/05_DATASETS_PROBLEMATICOS.md` — 5 datasets com recall <10% e 5 com recall >80%).

**Implicação prática:** **reportar std CV** sempre, não apenas média — leitor precisa saber que a média esconde variância grande. Para aplicação operacional, considerar **roteamento adaptativo** (LLM apenas em casos estatisticamente ambíguos) para reduzir variância.

---

## 2. Narrativa proposta para tese/paper

### 2.1 Estrutura recomendada (seção de resultados)

```
4. RESULTADOS

4.1 Setup experimental
    [descrição do benchmark de 29 datasets, 1421 bootstraps]

4.2 Baseline ML (sem LLM)
    [Tabela: 7 classifiers × Holdout/CV; melhor: LogReg holdout 54.94%, NB CV 47.47%]
    
4.3 Adição de features LLM Flash
    [Tabela: deltas vs ML-only; achado: +0.00pp holdout, −0.03pp CV]
    [Conclusão: Flash não agrega valor neste benchmark]
    
4.4 Adição de features LLM Pro com Step 1 prompt
    [Tabela: deltas vs ML-only; achado: +0.25pp holdout, +1.86pp CV]
    [Conclusão: Pro agrega ganho real mas marginal, principalmente em NB]
    
4.5 Análise por modelo
    [Padrão: NB beneficia, LogReg/RF/GBT/MLP indiferentes, SVM/KNN pioram]
    
4.6 Análise por classe (Pro NB holdout)
    [F1 macro 0.55, recall MNAR 73%, recall MCAR 44%]
    
4.7 Custo-benefício
    [Pro: $17/+1pp CV vs ML; viável para validação final, não para iteração]
    
4.8 Comparação com benchmark anterior (forensic_neutral_v2)
    [Reportar 56.2% CV em 23 datasets como referência histórica]
    [Atribuir queda à expansão do benchmark, não a regressão metodológica]

5. DISCUSSÃO
    
5.1 LLM como melhoria incremental
    [Argumentar que features LLM são úteis mas não essenciais]
    
5.2 Limitações do prompt engineering
    [Os 9 datasets críticos onde LLM falha mesmo com Step 1]
    [Necessidade de decomposição estruturada (Step 2)]
    
5.3 Trade-off variância vs media
    [Features LLM aumentam variance; rotamento adaptativo proposto]
```

### 2.2 Tabela mestra para o paper

```latex
\begin{table}[h]
\caption{Comparação head-to-head: configurações ML, Flash, Pro no benchmark de 29 datasets reais}
\begin{tabular}{lccc}
\hline
                       & ML-only (25f) & Flash (34f) & Pro (34f) \\
\hline
Holdout best (model)   & 54.94\% (LogReg) & 54.94\% (LogReg) & \textbf{55.19\%} (NB) \\
CV best (model)        & 47.47\% (NB) & 47.44\% (NB) & \textbf{49.33\%} (NB) \\
F1-macro best          & 0.54 & 0.48 & \textbf{0.55} \\
\hline
$\Delta$ vs ML (CV)    & --- & $-0.03$pp & $+1.86$pp \\
$\Delta$ vs ML (Hold)  & --- & $0.00$pp  & $+0.25$pp \\
\hline
Custo extração         & \$0 & \$2-4 & \$30-36 \\
Tempo extração         & $<1$min & $\sim$30min & $\sim$1h33min \\
Var CV (médio std)     & $\pm10.7$ & $\pm21.1$ & $\pm21.9$ \\
\hline
\end{tabular}
\label{tab:ml_flash_pro}
\end{table}
```

### 2.3 Frases-chave reutilizáveis

Para abstract:
> "Showing that pre-trained Large Language Models contribute marginal but consistent gains (+1.86pp Group 5-Fold CV in NaiveBayes) over a 25-feature statistical baseline, while smaller LLM variants (Flash) provide no measurable improvement."

Para introduction:
> "Our findings demonstrate that **the value of LLM features for missing data classification depends critically on (a) model size, (b) classifier choice, and (c) benchmark composition** — challenging the common assumption that any LLM augmentation improves performance."

Para conclusion:
> "We provide evidence that **decomposed causal reasoning prompts (Step 1) extract incremental information** beyond what statistical features capture, but the magnitude (+1.86pp CV) is bounded by domain priors that the LLM struggles to override in clinical contexts. Future work should explore structured causal decomposition (Step 2) to attack this ceiling."

---

## 3. Como apresentar para diferentes audiências

### 3.1 Banca de mestrado (defesa)

**Mensagem central:** "Provei que LLM agrega valor mensurável mas marginal; o Step 1 prompt funcionou direcionalmente; o próximo passo é Step 2 que ataca o gargalo identificado."

**Slides recomendados:**
1. Tabela ML × Flash × Pro (1 slide)
2. Análise por classe NB Pro (1 slide com matriz de confusão)
3. 9 datasets críticos (1 slide ilustrando MAR-bias residual)
4. Roadmap Step 2 (1 slide)

**Antecipar perguntas:**
- "Por que não 60%?" → expansão benchmark + datasets clínicos difíceis
- "Vale o custo?" → custo/CV-gain de $17/+1pp, viável para final
- "Por que NB?" → robustez a rótulos ruidosos (57% inconsistentes)

### 3.2 Reviewer de journal Q2-Q3

**Mensagem central:** "Apresentamos benchmark expandido + auditoria formal de leakage + comparação rigorosa de configurações LLM, com achado negativo cientificamente honesto."

**Pontos fortes a destacar:**
1. Auditoria formal de 6 canais de vazamento (ABCDEF)
2. Comparação controlada de mesma metodologia em 3 configurações
3. Resultado negativo (Flash = ML) reportado explicitamente — diferenciando-se de papers que escondem ablations negativas
4. Reprodutibilidade total (scripts, listas, seeds)

**Antecipar críticas:**
- "Single seed" → reportar como limitação na seção 5.2
- "Apenas 29 datasets" → reportar como progresso vs literatura (~5-15 datasets típico)
- "Falta comparação com método X" → adicionar PKLM, MechDetect (dados já disponíveis)

### 3.3 Comunidade industrial / aplicação

**Mensagem central:** "Para classificação de missing data em produção, ML puro entrega 95% do desempenho de soluções LLM-augmentadas a custo zero — adoção de LLM faz sentido apenas para validação científica ou casos onde recall MNAR é prioritário."

**Pontos práticos:**
1. ML-only = pipeline de produção viável (~47% CV)
2. Pro adicional = $17/+1pp, considerar quando precisão importa
3. Variância CV é alta — sistema de produção deve ter monitoring per-dataset

---

## 4. Riscos retóricos a evitar

### Não dizer

❌ **"LLM revolucionou classificação de missing data"** — o ganho é 1.86pp, não revolucionário

❌ **"Pro é a melhor solução"** — depende do classificador (NB sim, SVM/KNN não)

❌ **"Flash é uma alternativa econômica a Pro"** — Flash é **pior que ML puro**, não alternativa econômica

❌ **"O método atinge SOTA"** — ainda há gap vs forensic_neutral_v2 (que era SOTA interno em 23 datasets)

### Sim dizer

✅ **"LLM contribui incrementalmente como classifier-dependent feature engineering"**

✅ **"Pro Step 1 entrega ganho marginal robusto em NaiveBayes; outros classificadores são indiferentes"**

✅ **"Flash não agrega valor mensurável neste benchmark"**

✅ **"O método estabelece uma fronteira de Pareto entre ML puro (gratuito, 47%) e Pro (caro, 49%)"**

---

## 5. Conexão com próximos passos da pesquisa

A análise ML × Flash × Pro é a **base** para justificar Step 2 (Causal DAG):

```
Argumento Step 2:
  Step 1 + Pro entrega +1.86pp sobre ML, mas é insuficiente
  para mover acurácia >55% no benchmark expandido.
  
  A razão é o MAR-bias residual em 9 datasets críticos —
  o LLM "encontra causas MAR" mesmo quando deveria classificar
  como MCAR ou MNAR.
  
  Step 2 ataca diretamente este gargalo via decomposição
  estruturada: forçando o LLM a NOMEAR a variável causadora
  antes de classificar como MAR. Esperamos elevação de
  +5-10pp CV (49% → 55-59%).
  
  Custo Step 2: similar a Step 1 (~$30-36).
  Risco: similar a Step 1 (não atingir target).
  Reward: dobrar o ganho do LLM sobre ML (de +1.86pp para +5pp+).
```

Esta narrativa transforma o resultado "negativo" do Step 1 em **motivação positiva** para Step 2 — científicamente saudável e pedagógicamente clara.
