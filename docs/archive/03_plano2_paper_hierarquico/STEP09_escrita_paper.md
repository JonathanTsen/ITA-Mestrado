# STEP 09: Escrita do Paper

**Status: PENDENTE**
**Estimativa: 3-5 dias**
**Dependência: Todos os steps anteriores**

---

## Título (proposta)

"Hierarchical Classification of Missing Data Mechanisms with LLM-Augmented Features for MCAR-MNAR Disambiguation"

Alternativas:
- "Where Do LLMs Help? Hierarchical Missing Data Mechanism Classification with Targeted LLM Features"
- "Breaking the MCAR-MNAR Barrier: Hierarchical Classification with LLM-Augmented Disambiguation"

---

## Estrutura do Paper

### 1. Introduction (~1.5 páginas)

- **Problema:** Missing data mechanisms (MCAR, MAR, MNAR) determinam quais métodos são válidos. Escolha errada → viés.
- **Estado atual:** Testes clássicos (Little) só detectam MCAR. Classificação automática 3-way (MechDetect) alcança ~89% em sintético, mas MCAR e MNAR são confundidos.
- **Gap:** Ninguém investigou (a) por que MCAR≈MNAR é tão difícil, (b) se a abordagem pode ser decomposta, (c) se LLMs podem ajudar no subproblema difícil.
- **Contribuições:**
  1. **Classificação hierárquica** em 2 níveis (MCAR vs não-MCAR → MAR vs MNAR) que isola o gargalo
  2. **LLM-augmented features focadas** no Nível 2 (onde features estatísticas falham)
  3. **Evidência empírica** de que LLMs ajudam APENAS no subproblema MAR vs MNAR, não no geral
  4. **Validação de labels**: 57% dos datasets reais com labels inconsistentes
  5. **Análise de limitações**: MNAR Diffuse é fundamentalmente difícil de detectar

### 2. Related Work (~1.5 páginas)

- 2.1 Missing data theory (Rubin 1976, Little 1988)
- 2.2 Statistical tests for MCAR (Little's test, Jamshidian, PKLM)
- 2.3 ML-based mechanism classification (MechDetect — Jung 2024)
- 2.4 LLMs for feature engineering (CAAFE — Hollmann 2023, embeddings — Kasneci 2024)
- 2.5 **Posicionamento:** Primeiro trabalho a (a) usar classificação hierárquica para mecanismos, (b) usar LLM features focadas no subproblema difícil

### 3. Methodology (~3 páginas)

#### 3.1 Problem Formulation
- Classificação 3-way de mecanismo de missing data
- Por que é difícil: circularidade de X0 (MNAR depende de X0, que está faltante)
- Motivação para decomposição hierárquica

#### 3.2 Datasets
- **Sintéticos:** 1200 datasets (12 variantes × 100), 5 colunas, 1-10% missing rate
  - 3 MCAR + 5 MAR + 4 MNAR variants
  - 4 distribuições base (uniform, normal, exponential, beta)
- **Reais:** 23 datasets de OpenML e literatura
  - Validação de labels com 3 testes (Little's MCAR, correlação mask-Xi, KS obs vs imputado)
  - 57% inconsistentes — impacto discutido
- Tabela descritiva de todos os datasets

#### 3.3 Feature Extraction
- **Baseline (21 features):** 4 statistical + 6 discriminative + 5 MNAR-specific + 6 MechDetect
- **CAAFE (4 features):** missing_rate_by_quantile, tail_asymmetry, kurtosis_excess, cond_entropy
- **LLM v2 (8 features):** evidence_consistency, anomaly, dist_shift, confs, mcar_vs_mnar, pattern_clarity
- **Judge MNAR (4 features):** mnar_probability, censoring_evidence, distribution_anomaly, pattern_structured

#### 3.4 Proposed Approach: Hierarchical + LLM
- **Nível 1:** MCAR vs {MAR, MNAR} — features baseline APENAS (sem LLM)
  - Justificativa: features estatísticas são suficientes para detectar se missing depende de algo
- **Nível 2:** MAR vs MNAR — features baseline + LLM
  - Justificativa: circularidade de X0 requer raciocínio qualitativo que LLM fornece
- 7 classificadores testados em cada nível (RF, GBT, LR, SVM, KNN, MLP, NB)
- Diagrama do pipeline (Fig 1)

#### 3.5 Baselines
- **Direto 3-way:** 7 classificadores × 21 features (baseline puro)
- **MechDetect original:** HistGBT + 10-fold + regras de threshold (Jung 2024)
- **PKLM:** RF + KL divergence como teste de MCAR (Spohn 2024)

#### 3.6 Evaluation Protocol
- LOGO CV (Leave-One-Group-Out) para dados reais
- GroupKFold (k=5) como comparação
- Wilcoxon signed-rank para pares de métodos
- McNemar test para predições do melhor modelo
- Bootstrap 95% CI
- Friedman + Nemenyi para ranking global

### 4. Results (~3 páginas)

#### 4.1 Hierárquico vs Direto (contribuição principal)
- Tabela: V1 (direto) vs V2 (hier. puro) vs V4 (hier.+LLM) vs V6 (LLM em ambos)
- Accuracy global, recall por classe, F1 macro
- **Foco:** MNAR recall melhora? LLM no N2 > sem LLM no N2?

#### 4.2 Onde o LLM Ajuda (contribuição principal)
- Accuracy Nível 1 isolada (com vs sem LLM)
- Accuracy Nível 2 isolada (com vs sem LLM)
- **Evidência:** LLM melhora Nível 2 mas não Nível 1

#### 4.3 SHAP: Por que LLM importa no Nível 2
- SHAP summary plots separados por nível
- Features LLM ranking no Nível 1 vs Nível 2
- Dependence plots das top features LLM no Nível 2

#### 4.4 Ablação de Features
- 6 configurações (E1-E6): contribuição marginal de cada grupo
- Significância estatística (Wilcoxon p-values)

#### 4.5 Comparação com Baselines
- MechDetect original vs nosso pipeline vs hierárquico+LLM
- PKLM como feature vs como primeiro estágio
- Tabela comparativa final

#### 4.6 Label Quality (contribuição secundária)
- 57% inconsistentes: impacto na accuracy
- Accuracy com TODOS os datasets vs apenas validados

### 5. Discussion (~1.5 páginas)

- 5.1 **A barreira MCAR-MNAR:** circularidade de X0 e como a hierárquica contorna
- 5.2 **Quando LLMs ajudam:** só no subproblema MAR vs MNAR, não no geral. Implicação: LLMs para tabular data devem ser focados, não genéricos
- 5.3 **MNAR Focused vs Diffuse:** 100% dos MNAR reais são Diffuse — limitação fundamental
- 5.4 **Label quality:** 57% inconsistentes questiona benchmarks anteriores (incluindo MechDetect)
- 5.5 **Limitações:** N datasets, estrutura fixa (5 colunas), single LLM, missing só em X0
- 5.6 **Implicações práticas:** recomendações para praticantes

### 6. Conclusion (~0.5 página)

- Resumo: hierárquica + LLM focado supera abordagem direta
- Insight principal: LLMs devem ser aplicados cirurgicamente, não universalmente
- Future work: mais datasets, mais LLMs, detecção online, multi-mechanism, multi-column missing

---

## Figuras e Tabelas Chave

### Tabelas
1. **Tabela 1:** Descrição dos datasets (nome, N, features, missing rate, mecanismo, validação)
2. **Tabela 2:** Hierárquico vs Direto — accuracy e recall por classe (6 variantes V1-V6) ← **tabela principal**
3. **Tabela 3:** Accuracy Nível 1 vs Nível 2 isoladas (com e sem LLM)
4. **Tabela 4:** Ablação de features (E1-E6) com p-values
5. **Tabela 5:** Comparação com baselines (MechDetect, PKLM)
6. **Tabela 6:** Validação de labels (23 datasets × 3 testes)

### Figuras
1. **Fig 1:** Diagrama do pipeline hierárquico (Nível 1 stat → Nível 2 stat+LLM) ← **figura principal**
2. **Fig 2:** Confusion matrices (direto vs hierárquico+LLM) lado a lado
3. **Fig 3:** SHAP beeswarm do Nível 2 (mostrando features LLM em destaque)
4. **Fig 4:** SHAP comparison: importância LLM no Nível 1 vs Nível 2
5. **Fig 5:** Nemenyi diagram (critical difference) entre todos os métodos
6. **Fig 6:** Accuracy por variante sintética (quais MNAR são mais difíceis?)

---

## Journals Target (em ordem de preferência)

1. **Statistical Analysis and Data Mining** — foco perfeito (estatística + ML + missing data), IF ~1.5
2. **Data Mining and Knowledge Discovery (DAMI)** — contribuição metodológica, IF ~4.0
3. **Machine Learning Journal** — se SHAP analysis for profunda, IF ~4.5
4. **Pattern Recognition Letters** — backup como short paper se resultados modestos, IF ~3.9

---

## Critério de Conclusão

- [ ] Todas as 6 tabelas geradas com dados reais
- [ ] Todas as 6 figuras geradas
- [ ] Draft completo em LaTeX
- [ ] Abstract escrito (250 palavras)
- [ ] Revisão interna (orientador)
- [ ] Código público preparado (GitHub)
- [ ] Submissão
