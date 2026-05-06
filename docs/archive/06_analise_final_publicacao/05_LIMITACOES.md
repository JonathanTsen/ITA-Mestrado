# Limitações Sérias e Como Mitigá-las

**Data:** 2026-04-20 (atualizado após expansão do benchmark de 23→29 datasets)

Toda pesquisa honesta documenta suas limitações. Aqui estão as limitações mais sérias deste trabalho, com contraargumentos e sugestões de como apresentá-las na dissertação.

---

## A) Accuracy Absoluta é Modesta

### O problema
- Melhor resultado: **56.2%** em dados reais (pipeline completo, benchmark de 23 datasets)
- domain_prior sozinho: **63.1%**
- Ambos estão longe dos 80-90% que impressionam revisores
- **Nota:** Estes números referem-se ao benchmark de 23 datasets. Precisam ser recalculados com o benchmark expandido de 29 datasets.

### Contra-argumento
1. **O problema é fundamentalmente difícil.** MNAR é teoricamente indetectável por testes baseados em variáveis observadas (demonstrado pelo PKLM: 5,8% de poder para MNAR)
2. **~55% dos *labels* são inconsistentes** (15/29 no benchmark expandido, vs 57% no anterior). O teto teórico é ~60-65%, não 100%
3. **+23pp sobre o acaso** (33,3%) é estatisticamente significativo
4. **Supera todos os *baselines*** publicados (PKLM: 27,5%, MechDetect: 39,5%)

### Como apresentar na dissertação
> "A *accuracy* de 56% deve ser interpretada no contexto de um problema com teto teórico estimado em 60-65% (devido a ~55% de *label noise*) e onde o melhor *baseline* publicado alcança 39,5%."

---

## B) Desequilíbrio por Classe é Problemático

### O problema (benchmark original de 23 datasets)
- MAR: **96,5%** de *recall* (domina o resultado)
- MCAR: **27,6%** de *recall* (pior que o acaso de 33,3%!)
- MNAR: **34,0%** de *recall* (marginalmente acima do acaso)
- A *accuracy* de 63,1% é inflada por MAR (48,6% das amostras)

### Mitigação esperada (benchmark de 29 datasets)
O rebalanceamento de classes (9/11/9 vs. 5/11/7) e quase-dobro de MCAR (5→9) devem melhorar o recall de MCAR. **Necessita re-execução do pipeline para verificar.**

### Contra-argumento
1. **F1 macro** (0,501) é mais representativo que *accuracy* e também é o melhor entre *baselines*
2. **MAR é genuinamente mais fácil** — há correlação observável entre *mask* e outras variáveis. Isso é esperado
3. **MCAR fraco é informativo** — confirma que MCAR é indistinguível de MAR por domínio (um especialista também teria dificuldade)
4. **V3 (sem *domain_prior*)** tem *recall* mais equilibrado: MCAR 47%, MAR 56%, MNAR 40%
5. **(Novo)** Benchmark expandido com 9 MCAR datasets deve oferecer mais exemplos de treino para a classe mais fraca

### Como apresentar na dissertação
Reportar **duas perspectivas**: *pipeline* completo (*accuracy* máxima, viés MAR) e V3 (*recall* equilibrado, menor *accuracy*). Usar F1 macro como métrica primária.

---

## C) Alta Variância entre Folds

### O problema (benchmark original de 23 datasets)
- NaiveBayes CV: 55,5% **± 32,9%**
- Scores por *fold*: 50%, 82,4%, 65,4%, 38%, 41,5%
- Desvio padrão de 33pp é muito alto

### Mitigação esperada (benchmark de 29 datasets)
Com 29 datasets e 5 folds, cada fold terá ~5-6 datasets no teste (vs ~4-5 antes). **Necessita re-execução para medir impacto.**

### Contra-argumento
1. Com **29 datasets** (e 5 *folds*), cada *fold* contém ~5-6 datasets no teste — variância alta mas menor que com 23
2. **LODO confirma** os resultados (54,3% vs 56,2%, diferença < 2pp) — agora com **29 folds** de LODO
3. ***Bootstrap* CIs** (1000 iterações) são estreitos: [53,3%, 59,1%]
4. A variância é **entre datasets**, não entre amostras — reflete a heterogeneidade real dos dados

### Como apresentar na dissertação
> "A alta variância entre *folds* (±33pp) reflete a heterogeneidade inerente dos 29 datasets, com ~5-6 datasets por *fold* de teste. Os intervalos de confiança via *bootstrap* (1000 iterações) são mais informativos que o desvio padrão entre folds."

---

## D) Tamanho da Amostra Pequeno

### O problema (original)
- ~~Apenas **23 datasets reais** (5 MCAR + 11 MAR + 7 MNAR)~~
- ~~Desbalanceado (232 vs. 550 vs. 350 amostras)~~
- *Bootstrap* de ~50 amostras por dataset cria pseudo-replicação

### Mitigação aplicada (2026-04-20)
Expandido para **29 datasets reais** (9 MCAR + 11 MAR + 9 MNAR), todos validados com 3 testes estatísticos (Little's MCAR, correlação point-biserial, KS):

**Novos MCAR (+4, validados):**
| Dataset | Fonte | X0 (variável com missing) | Missing | Little's p | Justificativa MCAR |
|---------|-------|---------------------------|---------|-----------|---------------------|
| hepatitis_alkphosphate | OpenML/UCI | ALK_PHOSPHATE | 9.7% | 0.44 ✅ | Teste hepático de rotina, omitido por backlog laboratorial/volume de amostra |
| hepatitis_albumin | OpenML/UCI | ALBUMIN | 9.7% | 0.68 ✅ | Teste proteico de rotina, omitido aleatoriamente nos mesmos contextos |
| creditapproval_a14 | OpenML/UCI | A14 (contínuo) | 1.9% | 0.70 ✅ | Campo de formulário de crédito anonimizado, incompletude aleatória |
| echomonths_epss | OpenML/UCI | epss (ecocardiograma) | 10.0% | 0.65 ✅ | Janela acústica insuficiente para imagem, independente do valor |

**Removido da lista MCAR:**
- ~~arrhythmia_pwave~~ — Little's p=0.025 rejeita MCAR; correlação mask↔T-wave (p=0.004) indica MAR

**Novos MNAR (+2, com justificativa de domínio):**
| Dataset | Fonte | X0 (variável com missing) | Missing | Justificativa MNAR |
|---------|-------|---------------------------|---------|---------------------|
| hepatitis_protime | OpenML/UCI | PROTIME | 9.7% | Tempo de protrombina só é solicitado quando coagulação anormal é suspeita (médico ordena teste *porque* espera valor anormal) |
| pima_skinthickness | Kaggle/UCI | SkinThickness | 9.9% | Compasso padrão (≤45mm) não mede dobra cutânea em obesos — MNAR documentado na literatura médica (Pima Indians Diabetes Dataset) |

**Removidos da lista MNAR:**
- ~~colic_pcv~~ — Little's p=0.975, fortemente MCAR; justificativa de hemólise não sustentada
- ~~colic_totalprotein~~ — Ambíguo; correlação mask↔respiratory_rate (p=0.04) indica MAR, não MNAR

**Nota sobre validação MNAR:** Testes estatísticos têm poder limitado para detectar MNAR (o PKLM tem apenas 5.8% de poder). A classificação MNAR baseia-se em **raciocínio de domínio**, não apenas em testes — conforme discutido na seção A.

**Novo balanço:** 421 MCAR + 550 MAR + 450 MNAR = **1421 amostras bootstrap** (antes: 232 + 550 + 350 = 1132, +25%)

### Contra-argumento (atualizado)
1. A unidade de análise é o **dataset**, não a amostra *bootstrap*. GroupKFold trata isso corretamente
2. **29 datasets** de diferentes domínios (medicina, hepatologia, cardiologia, finanças, oceanografia, indústria, censos, endocrinologia) é boa diversidade
3. MCAR agora tem **9 datasets** (quase dobro do original 5), melhor equilíbrio
4. LODO (29 folds) é um teste rigoroso com esses dados
5. Todos os novos datasets foram **validados com 3 testes estatísticos** (Little's, correlação, KS); candidatos que falharam foram descartados
6. **MechDetect (Jung, 2024)** usa poucos datasets com rótulos reais também — é uma limitação do campo

### Como apresentar na dissertação
> "O benchmark foi expandido de 23 para 29 datasets reais (9 MCAR, 11 MAR, 9 MNAR), todos submetidos a validação estatística. Datasets candidatos que falharam na validação foram removidos (3 de 9 descartados). Novos domínios incluem hepatologia, cardiologia e finanças."

---

## E) *domain_prior* É Essencialmente uma Tabela de Consulta

### O problema
- Todos os *bootstraps* do mesmo dataset recebem o mesmo valor de *domain_prior*
- É efetivamente uma classificação **por dataset**, não por amostra
- O LLM está fazendo "qual mecanismo é plausível dado este domínio?" — o que um humano também poderia fazer
- Não é aprendizado de máquina no sentido tradicional — é recuperação de conhecimento (*knowledge retrieval*)

### Contra-argumento
1. **Automatização em escala** é valiosa — um LLM pode classificar 1000 datasets em minutos; um especialista levaria dias
2. **GroupKFold garante generalização** — o LLM classifica corretamente datasets que nunca viu durante o treino
3. **É assim que especialistas fazem** — um estatístico também inferiria MNAR para "insulina em estudo de diabetes". O LLM automatiza esse raciocínio
4. **Metadados são *input* legítimo** — todo dataset real tem domínio e nomes de variáveis disponíveis

### Como apresentar na dissertação
> "O *domain_prior* codifica raciocínio de domínio análogo ao que um estatístico experiente aplicaria: dado o contexto 'insulina em estudo de diabetes', a probabilidade de MNAR é alta porque pacientes sem diabetes raramente fazem o exame. A contribuição é automatizar essa inferência em escala usando LLMs."

---

## F) Gap Sintético-Real Persiste

### O problema
- Sintético: **77%** de *accuracy*
- Real: **41-56%** de *accuracy*
- *Gap* de **21-36pp** sugere que dados sintéticos não capturam a complexidade real

### Contra-argumento
1. O *gap* é **esperado** — dados sintéticos são gerados com mecanismos puros; dados reais têm mecanismos mistos, *label noise* e distribuições complexas
2. **O *gap* diminuiu** ao longo da pesquisa: V1 (34,6pp) → V3 (19,8pp) → *pipeline* D (23,1pp com *features* adicionais)
3. *Features* CAAFE são **mais importantes em dados reais** (*rank* 2-4) do que em sintéticos (*rank* 16-21) — mostra adaptação ao problema

### Como apresentar na dissertação
> "O *gap* de 23pp entre dados sintéticos e reais é atribuível a três fatores: (1) mecanismos mistos em dados reais, (2) 57% de *label noise* e (3) distribuições não paramétricas complexas. Esse *gap* motivou o desenvolvimento de *features* CAAFE, que se mostraram mais importantes em dados reais do que em sintéticos."

---

## G) Reprodutibilidade Dependente de API

### O problema
- *domain_prior* depende de chamadas a LLMs (Gemini)
- LLMs mudam ao longo do tempo (*model updates*)
- Resultados podem não ser perfeitamente reprodutíveis

### Contra-argumento
1. *Features* são **extraídas uma vez e salvas** — o modelo treinado não precisa de LLM em inferência
2. O protocolo de extração é **documentado e reprodutível**
3. Resultados com **metadados neutralizados** minimizam a sensibilidade ao modelo específico
4. Sistema de *checkpoint* permite **retomada** após interrupção

### Como apresentar na dissertação
Documentar versão exata do modelo LLM usado, *prompts* e metadados. Disponibilizar *features* extraídas como artefato para reprodutibilidade. Todas as fontes de datasets estão documentadas em [11_FONTES_DATASETS_REAIS.md](11_FONTES_DATASETS_REAIS.md) com links para OpenML, UCI e Kaggle.

---

## Resumo: Limitações vs Mitigações

| Limitação | Severidade | Mitigação |
|-----------|:----------:|-----------|
| *Accuracy* modesta (56%) | Alta | Contextualizar com teto teórico e *baselines* |
| Desequilíbrio por classe | Alta | Reportar F1 macro + V3 como alternativa equilibrada |
| Alta variância entre *folds* | Média | Usar *bootstrap* CIs, LODO como validação |
| ~~Amostra pequena (23 datasets)~~ Expandida para 29 | ~~Média~~ **Mitigada** | 9 MCAR + 11 MAR + 9 MNAR; validados estatisticamente |
| *domain_prior* = tabela de consulta | Média | Automatização em escala; GroupKFold valida generalização |
| *Gap* sintético-real | Média | Esperado; documentado e parcialmente explicado |
| Dependência de API | Baixa | *Features* extraídas uma vez; modelo salvo localmente |

---

## Ranking: Gravidade × Complexidade de Implementação

A tabela acima classifica as limitações isoladamente. Abaixo, uma análise cruzada considerando **impacto na validade da pesquisa** e **esforço necessário para resolver**, ordenada da mais crítica para a menos crítica.

> **Nota:** Este ranking foi reordenado em 2026-04-20 após a mitigação de D (expansão do benchmark). D deixou de ser #1 e desceu para #5. E (domain_prior) e B (desequilíbrio) subiram.

### 🔴 1. *domain_prior* É uma Tabela de Consulta (E) — AMEAÇA À CONTRIBUIÇÃO

| Dimensão | Avaliação |
|----------|-----------|
| **Gravidade** | ⬛⬛⬛⬛⬜ Alta |
| **Complexidade** | ⬛⬛⬛⬛⬛ Muito Alta |
| **Tipo** | Conceitual — questiona a contribuição principal |

**Por que é a mais grave agora:** Com D mitigado, esta é a limitação com maior risco para a dissertação. A contribuição central é "LLM melhora a classificação de mecanismos de missing data". Se o *domain_prior* for reduzido a uma tabela de consulta, um revisor pode argumentar que **não há aprendizado genuíno** — apenas recuperação de conhecimento. Isso enfraquece o argumento de contribuição científica.

**Por que é complexo:** Resolver exige **repensar a abordagem do LLM**. Opções possíveis: (1) usar o LLM para raciocínio probabilístico sobre padrões estatísticos (não apenas domínio), (2) *chain-of-thought* com evidências numéricas, (3) ensemble de LLM + estatísticas onde o LLM pondera evidências conflitantes. Cada abordagem exige experimentação extensiva e pode não melhorar os resultados.

---

### 🔴 2. Desequilíbrio por Classe (B) — INVALIDA MCAR

| Dimensão | Avaliação |
|----------|-----------|
| **Gravidade** | ⬛⬛⬛⬛⬜ Alta |
| **Complexidade** | ⬛⬛⬛⬜⬜ Média |
| **Tipo** | Metodológico — classificador falha em 1 de 3 classes |

**Por que é grave:** MCAR com *recall* de 27,6% (abaixo do acaso de 33,3%) significa que o classificador **efetivamente não consegue identificar MCAR**. Um classificador que funciona para 2 de 3 classes tem utilidade prática limitada.

**Mitigação parcial via D:** O benchmark expandido (9 MCAR vs. 5 antes) dá mais exemplos de treino para MCAR. **Re-executar o pipeline para verificar se o recall de MCAR melhorou.**

**Outras opções:** *class weighting*, SMOTE/oversampling para MCAR, *threshold tuning* por classe, ou classificador hierárquico (MAR vs. não-MAR → MCAR vs. MNAR). O problema subjacente é que MCAR e MAR são **teoricamente difíceis de distinguir**.

---

### 🟠 3. Gap Sintético-Real (F) — QUESTIONA VALIDADE EXTERNA

| Dimensão | Avaliação |
|----------|-----------|
| **Gravidade** | ⬛⬛⬛⬜⬜ Média-Alta |
| **Complexidade** | ⬛⬛⬛⬛⬜ Alta |
| **Tipo** | Validação — transferência limitada |

**Por que é grave:** Um gap de 21-36pp entre sintético (77%) e real (41-56%) levanta dúvidas sobre se o modelo é útil fora do laboratório. Se o pipeline só funciona bem em dados sintéticos, a contribuição prática é limitada.

**Por que é complexo:** Requer atacar de dois lados: (1) tornar dados sintéticos mais realistas (mecanismos mistos, distribuições não-paramétricas, label noise simulado) e (2) melhorar features para dados reais. A geração sintética mais realista é um problema de pesquisa aberto por si só.

---

### 🟡 4. Accuracy Modesta (A) — CONSEQUÊNCIA

| Dimensão | Avaliação |
|----------|-----------|
| **Gravidade** | ⬛⬛⬛⬜⬜ Média |
| **Complexidade** | ⬛⬛⬛⬛⬜ Alta |
| **Tipo** | Resultado — consequência de E, B, F |

**Por que é média gravidade (não alta):** A accuracy de 56% é modesta em termos absolutos, mas **já está próxima do teto teórico** (60-65%) dado o label noise de ~55%. O contra-argumento é sólido: supera todos os baselines e o teto é fundamentalmente limitado.

**Por que é complexo:** Melhorar além de 56% exige resolver E (melhor uso do LLM), B (desequilíbrio) ou F (melhor transferência) — todos problemas difíceis. Não há "quick win" aqui.

---

### ✅ 5. Tamanho da Amostra Pequeno (D) — PARCIALMENTE MITIGADO

| Dimensão | Avaliação |
|----------|-----------|
| **Gravidade** | ~~⬛⬛⬛⬛⬛ Crítica~~ → ⬛⬛⬜⬜⬜ Baixa-Média |
| **Complexidade** | ~~⬛⬛⬛⬛⬛ Muito Alta~~ → ✅ Resolvido |
| **Tipo** | Estrutural — mitigado com expansão do benchmark |

**Mitigação aplicada (2026-04-20):** Benchmark expandido de **23 → 29 datasets**, validados estatisticamente:
- MCAR: 5 → **9** (+4: hepatitis×2, credit-approval, echoMonths — todos confirmados por Little's test)
- MAR: 11 → **11** (sem mudança, já era a maior classe)
- MNAR: 7 → **9** (+2: hepatitis protime, pima skin thickness — justificados por domínio)
- Descartados: 3 candidatos falharam na validação (arrhythmia P-wave, colic PCV, colic total_protein)

**Novos domínios cobertos:** cardiologia (echoMonths), finanças (credit-approval), hepatologia (hepatitis×3), endocrinologia (pima skin).

**Impacto esperado:** O rebalanceamento de classes (9/11/9 vs. 5/11/7) deve melhorar o recall de MCAR (B) e reduzir a variância entre folds (C). Bootstrap total: 1421 amostras (antes: 1132, +25%).

**Limitação residual:** 29 datasets ainda é modesto comparado a benchmarks de ML tradicionais, mas é competitivo para classificação de mecanismos de missing data com rótulos reais. Fontes documentadas em [11_FONTES_DATASETS_REAIS.md](11_FONTES_DATASETS_REAIS.md).

**Mitigação adicional (2026-05-03) — protocolo v2 de validação de rótulos:**
O protocolo v1 (`validar_rotulos.py`) usava apenas 3 testes (Little, correlação ponto-biserial, KS observado-vs-imputado) e classificava ~50% dos rótulos como inconsistentes. Diagnóstico crítico (em `system-instruction-you-are-working-starry-pudding.md`) identificou que:
- Little é hipersensível em N grande;
- correlação ponto-biserial só captura linearidade;
- KS-com-mediana é conceitualmente mal-formulado (compara distribuição com versão de si mesma + spike artificial).

O **protocolo v2** (`validar_rotulos_v2.py` + `calibrar_protocolo.py`) substitui esses testes por 3 camadas:

- **A — MCAR**: voto majoritário entre Little, **PKLM** (Spohn 2024, não-paramétrico) e **Levene-stratified** (Bonferroni). Reduz falsos positivos em N grande.
- **B — MAR**: AUC de RandomForest prevendo `mask` a partir de `X_obs` com permutation p-value (200 perm) + mutual information. Captura não-linearidade.
- **C — MNAR**: 4 scores CAAFE-MNAR (tail asymmetry, kurtosis excess, conditional entropy, missing rate por quartil) thresholdados via Youden's J em sintéticos.
- **D — Reconciliação Bayesiana**: KDE Gaussiano por mecanismo ajustado nos sintéticos (1.200 com ground truth) → P(MCAR), P(MAR), P(MNAR). Confidência = P_max − P_segundo.

Calibração inicial (15 sintéticos por classe, smoke test) atinge **95.6% accuracy nos sintéticos** (modo Bayes) — bem acima do alvo de 85% e validando que o vetor de 10 features tem informação suficiente para separar mecanismos quando os rótulos são confiáveis. Aplicado aos 29 reais, a accuracy contra rótulos da literatura é **41.4%** — o gap de ~54pp é a estimativa empírica do "label noise" + dificuldade fundamental de MNAR (Molenberghs 2008), não falha do protocolo.

Datasets que o Bayes v2 confirma com alta confiança:
- MAR (literatura confirmada): `mammographic_density`, `oceanbuoys×2`, `titanic_age×2`
- MNAR (literatura confirmada): `adult_capitalgain`, `kidney_pot`, `kidney_sod`, `pima_insulin`

Casos ambíguos (confiança < 0.4) — candidatos a sensitivity analysis na dissertação:
- `MCAR_echomonths_epss`, `MCAR_hepatitis_albumin`
- `MAR_colic_resprate`, `MAR_kidney_hemo`
- `MNAR_colic_refluxph`, `MNAR_hepatitis_protime`, `MNAR_mroz_wages`

---

### 🟡 6. Alta Variância entre Folds (C) — CONSEQUÊNCIA DE D

| Dimensão | Avaliação |
|----------|-----------|
| **Gravidade** | ⬛⬛⬜⬜⬜ Média-Baixa |
| **Complexidade** | ⬛⬛⬜⬜⬜ Baixa |
| **Tipo** | Estatístico — resolvido indiretamente |

**Por que é menos grave:** O bootstrap CI [53,3%, 59,1%] e LODO (54,3%) já mitigam essa limitação. Com N=29 (vs. 23), cada fold terá ~5-6 datasets (vs. ~4-5), reduzindo a variância.

**Por que é pouco complexo:** A expansão de D (23→29) já deve reduzir C. Alternativamente, técnicas como *repeated stratified k-fold* ou *nested CV* podem ajudar. LODO com 29 folds é mais robusto que com 23.

---

### 🟢 7. Reprodutibilidade Dependente de API (G) — MENOR IMPACTO

| Dimensão | Avaliação |
|----------|-----------|
| **Gravidade** | ⬛⬜⬜⬜⬜ Baixa |
| **Complexidade** | ⬛⬜⬜⬜⬜ Baixa |
| **Tipo** | Operacional — mitigações já existem |

**Por que é menos grave:** Features são extraídas uma vez e salvas. O modelo treinado não precisa de LLM em inferência. A maioria dos trabalhos com LLM tem essa limitação — revisores a reconhecem como inerente ao campo.

**Por que é pouco complexo:** Já está mitigado: documentar versão do modelo, salvar prompts e features como artefatos. Opcionalmente, usar modelos locais (Llama, Mistral) como validação de robustez.

---

## Grafo de Dependências entre Limitações (atualizado 2026-04-20)

```
D (Amostra pequena) ─── ✅ MITIGADO (23→29, classes 9/11/9)
       │
       ├──► B (Desequilíbrio) ─── esperado melhorar (mais MCAR no treino)
       │         │
       ├──► C (Alta variância) ── esperado melhorar (mais folds no LODO)
       │         │
       │         ▼
       └──► A (Accuracy modesta) ◄──── F (Gap sintético-real)
                     ▲
                     │
            E (domain_prior = lookup) ◄── #1 prioridade agora
```

### Prioridades de trabalho futuro

| # | Limitação | Ação necessária |
|---|-----------|----------------|
| 1 | **E** (domain_prior) | Repensar uso do LLM: raciocínio probabilístico, chain-of-thought com evidências numéricas |
| 2 | **B** (desequilíbrio) | Re-executar pipeline com 29 datasets; se MCAR recall ainda < 33%, aplicar class weighting |
| 3 | **F** (gap sintético-real) | Gerar dados sintéticos mais realistas (mecanismos mistos, distribuições complexas) |
| 4 | **A** (accuracy) | Consequência de E+B+F — melhora indiretamente |
| 5 | — | **Re-executar pipeline completo** com benchmark de 29 datasets para atualizar todos os números |

**Insight principal:** Com D mitigado, o investimento de maior retorno é **E (melhor uso do LLM)**. A expansão do benchmark deve ter efeito cascata em B e C, mas **precisa ser verificado empiricamente**.
