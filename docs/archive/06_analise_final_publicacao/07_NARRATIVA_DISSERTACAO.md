# Narrativa e Estrutura da Dissertação

**Data:** 2026-04-19

---

## Título Sugerido

**"Classificação Automática de Mecanismos de Dados Faltantes: Combinando Features Estatísticas e Raciocínio de Domínio via Large Language Models"**

**Título alternativo (mais curto):**
"LLMs como Especialistas de Domínio para Classificação de Mecanismos de Dados Faltantes"

**Título em inglês (para o artigo):**
"Can LLMs Reason About Missing Data? Combining Statistical Features and Domain Knowledge for Mechanism Classification"

---

## Narrativa Central

A dissertação deve ser contada como uma **jornada de descoberta científica**, não como uma apresentação de resultados finais. Esta jornada é mais convincente que qualquer número individual:

1. **O problema:** Classificar mecanismos de missing data é fundamental mas não resolvido
2. **A tentativa inicial:** Features + ML → accuracy inflada por data leakage
3. **O diagnóstico:** Correção honesta revela o problema real (40.5%)
4. **A solução estatística:** CAAFE features +7pp, LLM data-driven +3pp → teto ~51%
5. **A descoberta:** LLM domain reasoning adiciona +6pp genuínos
6. **O resultado negativo:** LLM features estatísticas são ruído
7. **A síntese:** Dois regimes de accuracy; domain reasoning é a chave

---

## Estrutura Proposta (7 Capítulos)

### Capítulo 1 — Introdução (~15 páginas)

**Conteúdo:**
- Importância de dados faltantes em ciência e indústria
- Os três mecanismos de Rubin (1976): MCAR, MAR, MNAR
- Por que identificar o mecanismo importa (impacta escolha de método de imputação)
- Limitações dos testes existentes:
  - Little's MCAR test: apenas MCAR vs não-MCAR (binário)
  - PKLM (Spohn 2024): não detecta MNAR (5.8% de poder)
- **Lacuna:** Não existe método automático para classificação 3-way
- **Pergunta de pesquisa:** LLMs podem melhorar esta classificação?
- Contribuições listadas (6 contribuições formais)
- Organização da dissertação

**Referências-chave:** Rubin 1976, Little & Rubin 2002, Spohn et al. 2024, Jung et al. 2024

### Capítulo 2 — Fundamentação Teórica (~25 páginas)

**2.1. Dados Faltantes**
- Taxonomia de Rubin: MCAR, MAR, MNAR
- Definições formais e exemplos
- Consequências de cada mecanismo para análise estatística
- Métodos de imputação e sua dependência do mecanismo

**2.2. Testes Existentes para Mecanismos**
- Little's MCAR Test (1988): formulação, limitações
- PKLM (Spohn et al. 2024): Random Forest + JSD + permutação
- Limitação teórica: MNAR é indetectável por variáveis observadas

**2.3. Machine Learning para Dados Tabulares**
- Random Forest, GradientBoosting, NaiveBayes, SVM, KNN, MLP
- Por que NaiveBayes funciona bem com features de alta variância
- Cross-validation com grupos (GroupKFold)

**2.4. LLMs para Dados Tabulares**
- CAAFE (Hollmann et al., NeurIPS 2023): LLM gera features em código
- Embeddings para dados tabulares
- LLMs como "domain experts" — estado da arte

**2.5. MechDetect**
- Jung et al. (2024): threshold-based, features AUC
- Limitações: calibração fixa, viés MNAR

### Capítulo 3 — Metodologia (~30 páginas)

**3.1. Geração de Dados Sintéticos**
- 12 variantes via MissMecha (3 MCAR + 5 MAR + 4 MNAR)
- 4 distribuições base × variantes = 1200 amostras
- Convenção: 5 colunas, X0 com missing, 1000 linhas, 1-10% missing rate

**3.2. Coleta de Dados Reais**
- 23 datasets de OpenML, UCI, literatura
- Processo de seleção e validação
- Bootstrap: ~50 amostras por dataset = 1132 total
- Validação de labels (Little's MCAR + correlação + KS)

**3.3. Pipeline de Features**
- 21 features baseline:
  - 4 estatísticas resumo
  - 11 discriminativas
  - 6 MechDetect
- 4 CAAFE features (Python puro)
- 6 LLM features (5 data-driven + 1 domain_prior)

**3.4. Abordagens LLM Testadas**
- v2: análise de segunda ordem (8 features)
- Judge MNAR: classificação binária (4 features)
- Embeddings: sentence-transformers local (10 features)
- CAAFE automático: geração de código via LLM
- Context-aware: domain reasoning com metadata neutralizada

**3.5. Protocolo de Validação**
- GroupKFold-5 (por dataset de origem)
- LODO (Leave-One-Dataset-Out, 23 folds)
- Bootstrap CIs (1000 iterações)
- Holdout GroupShuffleSplit 75/25

**3.6. Protocolo Anti-Leakage**
- Auditoria de 5 canais de leakage
- Metadata MEDIUM-scope neutralizada
- Verificação de infraestrutura (sentinel fields, filename isolation)

### Capítulo 4 — Resultados Experimentais (~25 páginas)

**4.1. Resultados em Dados Sintéticos**
- Baseline: 76.7% (MLP)
- Com LLM: 79.3% (RF)
- Hierárquico V6: 79.3%

**4.2. Ablação Completa em Dados Reais**
- Cenários A→E com tabela e gráficos
- Contribuição marginal de cada componente
- Desempenho por classe (MAR >> MNAR > MCAR)

**4.3. Comparação com Baselines**
- PKLM: 27.5% accuracy, bom para MCAR (93%), falha em MNAR (4.3%)
- MechDetect: 39.5%, bom para MNAR (93%), falha em MAR (0%)
- V3 proposto: 50.5%, recall equilibrado

**4.4. Classificação Hierárquica (V1-V6)**
- Level 1: MCAR vs {MAR,MNAR} (~80%)
- Level 2: MAR vs MNAR (gargalo)
- V3 (CAAFE) vs V4 (CAAFE+LLM): V3 vence

**4.5. Análise Per-Dataset (LODO)**
- Quais datasets são fáceis/difíceis
- Correlação com características do dataset

**4.6. Feature Importance**
- SHAP values
- RandomForest importance
- CAAFE features dominam rank 2-4

### Capítulo 5 — Análise e Discussão (~20 páginas)

**5.1. Dois Regimes de Accuracy**
- Regime estatístico (40-51%): teto com features puras
- Regime domain reasoning (56-63%): LLM adiciona knowledge

**5.2. Por que LLM Features Estatísticas Falham**
- Distribuições idênticas entre classes
- Multicolinearidade
- Regressão à média nas confidências
- Mecanismo de falha detalhado

**5.3. Por que CAAFE Funciona**
- Captura propriedades de X0 mesmo com missing
- tail_asymmetry tem Cohen's d = 0.84 (forte)
- Determinísticas > estocásticas para este problema

**5.4. Limitação Fundamental do MNAR**
- MNAR depende de X0 (faltante) → não-observável
- PKLM demonstra: poder = 5.8% para MNAR
- Implicação: sem domain knowledge, MNAR é indetectável

**5.5. Implicações para "LLM como Feature Extractor"**
- Valor do LLM está no raciocínio, não na computação
- Features determinísticas > features LLM quando o conhecimento de domínio pode ser codificado
- LLM é valioso quando domain reasoning não pode ser codificado em regras

**5.6. Labels de Benchmark: Problema Sistêmico**
- 57% inconsistentes
- Implicações para toda pesquisa em missing data mechanisms
- Necessidade de validação antes de usar como ground truth

### Capítulo 6 — Conclusão (~5 páginas)

**6.1. Contribuições Confirmadas**
1. Framework 3-way com ML
2. CAAFE features (+7.1pp)
3. LLM domain reasoning genuíno (+5.7pp, +22.6pp standalone)
4. Resultado negativo sobre LLM features estatísticas
5. Auditoria de labels (57% inconsistentes)
6. Benchmark de 23 datasets

**6.2. Limitações**
- 23 datasets (pequeno mas diverso)
- Accuracy modesta (contextualizada com teto teórico)
- domain_prior é per-dataset (mas automatiza raciocínio)

**6.3. Trabalhos Futuros**
- Expandir benchmark (50+ datasets)
- Testar com outros LLMs (GPT-4, Claude, Llama)
- Empacotar como biblioteca Python
- Explorar domain reasoning com metadata mais rica
- Transfer learning sintético → real

### Capítulo 7 — Referências

---

## Contribuições Formais (para listar na Introdução)

1. **Framework de classificação 3-way** para mecanismos de missing data usando ML, superando abordagens binárias (Little's, PKLM) e baseadas em thresholds (MechDetect)

2. **Quatro features CAAFE** para detecção de MNAR (tail_asymmetry, kurtosis_excess, cond_entropy_X0_mask, missing_rate_by_quantile), contribuindo +7.1pp de accuracy — a maior contribuição individual do regime estatístico

3. **Evidência empírica de LLM domain reasoning** genuíno para classificação de mecanismos: +22.6pp sobre baseline usando apenas domínio e nome da variável, validado com metadata neutralizada e LODO

4. **Resultado negativo rigoroso** sobre LLM features via análise estatística de segunda ordem: distribuições idênticas entre classes, Cohen's d < 0.4, mecanismo de falha identificado

5. **Auditoria de labels** em 23 datasets benchmark revelando 57% de inconsistência entre labels de mecanismo e testes estatísticos

6. **Benchmark reproduzível** de 23 datasets reais com validação estatística de labels, código aberto, e features pré-extraídas

---

## Figuras-Chave para a Dissertação

1. **Diagrama do pipeline** (dados → features → classificador → predição)
2. **Tabela de ablação** (Cenários A→E com barras de erro)
3. **Gráfico de barras** comparando baselines (PKLM, MechDetect, V3, D)
4. **Heatmap** de confusion matrix por cenário
5. **Gráfico de decomposição** mostrando os dois regimes (estatístico vs domain reasoning)
6. **SHAP summary plot** mostrando importância de features
7. **Distribuição de domain_prior** por classe verdadeira (3 histogramas)
8. **Tabela LODO per-dataset** mostrando variabilidade por dataset
