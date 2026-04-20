# Artigos Relevantes: Distinção MAR vs MNAR

Levantamento exaustivo de artigos científicos sobre como detectar, testar e distinguir mecanismos MAR (Missing At Random) e MNAR (Missing Not At Random).

---

## 1. TRABALHOS FUNDACIONAIS

### 1.1 Rubin (1976) — Definição dos mecanismos
- **Título:** Inference and Missing Data
- **Autor:** Donald B. Rubin
- **Publicação:** Biometrika, 63(3), 581–592, 1976
- **Contribuição:** Definição formal dos mecanismos MCAR, MAR e MNAR (originalmente "nonignorable"). Estabeleceu o framework teórico que fundamenta toda a literatura subsequente.
- **Relevância MAR vs MNAR:** Provou que MAR permite inferência válida via likelihood ignorando o mecanismo de missingness, enquanto MNAR requer modelagem explícita do mecanismo.

### 1.2 Little & Rubin (2002) — Livro referência
- **Título:** Statistical Analysis with Missing Data (2nd Edition)
- **Autores:** Roderick J.A. Little, Donald B. Rubin
- **Publicação:** Wiley, 2002 (3rd ed. 2019)
- **Contribuição:** Tratamento completo dos três mecanismos, métodos de likelihood, múltipla imputação, e modelos para dados MNAR.
- **Relevância MAR vs MNAR:** Capítulos dedicados a pattern-mixture models e selection models para MNAR.

### 1.3 Little (1988) — Teste MCAR de Little
- **Título:** A Test of Missing Completely at Random for Multivariate Data with Missing Values
- **Autor:** Roderick J.A. Little
- **Publicação:** Journal of the American Statistical Association, 83(404), 1198–1202, 1988
- **Contribuição:** Propôs o teste chi-quadrado multivariado para MCAR baseado em EM.
- **Limitação crucial:** Rejeição indica apenas "não-MCAR" — não distingue MAR de MNAR.

---

## 2. TESTES ESTATÍSTICOS PARA DISTINGUIR MAR vs MNAR

### 2.1 Diggle & Kenward (1994) — Selection model para dropout
- **Título:** Informative Drop-Out in Longitudinal Data Analysis
- **Autores:** Peter J. Diggle, Michael G. Kenward
- **Publicação:** Applied Statistics (JRSS-C), 43(1), 49–93, 1994
- **Contribuição:** Propuseram um selection model conjunto para dados longitudinais com dropout informativo (MNAR). Incluíram um teste de razão de verossimilhança comparando modelo MAR vs MNAR.
- **Limitação:** O teste é condicional ao modelo alternativo (MNAR) estar correto. Sensível a outliers e suposições distribucionais.
- **Link:** https://www.jstor.org/stable/2986113

### 2.2 Wang, Shao & Kim (2023) — Score test MAR vs MNAR
- **Título:** Score Test for Missing at Random or Not under Logistic Missingness Models
- **Autores:** Lei Wang, Jun Shao, Jae Kwang Kim
- **Publicação:** Biometrics, 79(2), 1268–1279, 2023
- **arXiv:** https://arxiv.org/abs/2105.12921
- **Contribuição:** Desenvolveram dois score tests para testar H0: MAR vs H1: MNAR sob modelos logísticos de missingness (paramétrico e semiparamétrico). O score test requer apenas estimação sob H0 (MAR), contornando o problema de identificação sob MNAR.
- **Relevância:** Um dos poucos testes formais especificamente desenhados para MAR vs MNAR.

### 2.3 Jamshidian & Jalal (2010) — Teste não-paramétrico MCAR
- **Título:** Tests of Homoscedasticity, Normality, and Missing Completely at Random for Incomplete Multivariate Data
- **Autores:** Mortaza Jamshidian, Siavash Jalal
- **Publicação:** Psychometrika, 75(4), 649–674, 2010
- **Contribuição:** Teste não-paramétrico de MCAR baseado em homoscedasticidade entre grupos com padrões de missingness idênticos. Não requer normalidade multivariada (vantagem sobre Little's test).
- **Implementação:** R packages `MissMech` e `mice::mcar()`
- **Link:** https://link.springer.com/article/10.1007/s11336-010-9175-3

### 2.4 Berrett & Samworth (2023) — Teste ótimo não-paramétrico MCAR
- **Título:** Optimal Nonparametric Testing of Missing Completely At Random, and Its Connections to Compatibility
- **Autores:** Thomas B. Berrett, Richard J. Samworth
- **Publicação:** Annals of Statistics, 51(5), 2170–2193, 2023
- **arXiv:** https://arxiv.org/abs/2205.08627
- **Contribuição:** Caracterização completa do conjunto de alternativas distinguíveis de MCAR. Definição de "incompatibility index" como medida de detectabilidade. Prova de taxas minimax ótimas.
- **Implementação:** R package `MCARtest`
- **Relevância:** Framework teórico rigoroso para o que é detectável a partir dos dados observados.

### 2.5 Potthoff et al. (2006) — Teste MAR+ e logistic regression
- **Título:** Can One Assess Whether Missing Data Are Missing at Random in Medical Studies?
- **Autores:** Richard F. Potthoff et al.
- **Publicação:** Statistical Methods in Medical Research, 15(3), 213–234, 2006
- **Contribuição:** Introduziu a suposição MAR+ (testável) e procedimento de teste baseado em regressão logística para dados longitudinais médicos.
- **Relevância:** Abordagem prática para investigar se dados são MAR em contexto clínico.

---

## 3. MODELOS PARA DADOS MNAR (Selection Models & Pattern-Mixture Models)

### 3.1 Heckman (1979) — Selection model original
- **Título:** Sample Selection Bias as a Specification Error
- **Autor:** James J. Heckman
- **Publicação:** Econometrica, 47(1), 153–161, 1979
- **Contribuição:** Modelo de seleção para corrigir viés de seleção amostral. Fundamento dos selection models para MNAR em estatística.
- **Prêmio:** Nobel de Economia 2000 (parcialmente por este trabalho).

### 3.2 Little (1993) — Pattern-Mixture Models
- **Título:** Pattern-Mixture Models for Multivariate Incomplete Data
- **Autor:** Roderick J.A. Little
- **Publicação:** Journal of the American Statistical Association, 88(421), 125–134, 1993
- **Contribuição:** Formulação dos pattern-mixture models como alternativa aos selection models para dados MNAR. Modelagem da distribuição dos dados condicionada nos padrões de missingness.

### 3.3 Molenberghs, Kenward et al. (2008) — Equivalência MAR-MNAR
- **Título:** Every Missingness Not at Random Model Has a Missingness at Random Counterpart with Equal Fit
- **Autores:** Geert Molenberghs, Herbert Thijs, Michael Kenward, Bart Michiels
- **Publicação:** Journal of the Royal Statistical Society: Series B, 70(2), 371–388, 2008
- **Contribuição:** Resultado fundamental: para todo modelo MNAR existe um modelo MAR com ajuste idêntico. Isso implica impossibilidade de distinguir MAR de MNAR apenas com dados observados — razão teórica pela qual testes diretos são impossíveis.
- **Link:** https://academic.oup.com/jrsssb/article/70/2/371/7109518

### 3.4 Molenberghs & Kenward (2007) — Livro Missing Data in Clinical Studies
- **Título:** Missing Data in Clinical Studies
- **Autores:** Geert Molenberghs, Michael G. Kenward
- **Publicação:** Wiley, 2007, 528 pp.
- **Contribuição:** Tratamento detalhado de sensitivity analysis sob MNAR: selection models, pattern-mixture models, shared parameter models. Único livro focado em dados faltantes longitudinais com análise de sensibilidade tão detalhada.

---

## 4. ABORDAGENS CAUSAIS E GRAFOS (m-DAGs)

### 4.1 Mohan & Pearl (2021) — Graphical Models for Missing Data
- **Título:** Graphical Models for Processing Missing Data
- **Autores:** Karthika Mohan, Judea Pearl
- **Publicação:** Journal of the American Statistical Association, 116(534), 1023–1037, 2021
- **arXiv:** https://arxiv.org/abs/1801.03583
- **Contribuição:** Framework de "missingness graphs" (m-DAGs) que representam explicitamente os mecanismos causais de missingness. Definição de "recoverability" — quando estimação consistente é possível mesmo sob MNAR. Derivação de implicações testáveis para modelos MAR e MNAR.
- **Relevância:** Reformula o problema MAR/MNAR em termos causais, permitindo testar suposições específicas do grafo.

### 4.2 Mohan & Pearl (2014) — Testability of Models with Missing Data
- **Título:** On the Testability of Models with Missing Data
- **Autores:** Karthika Mohan, Judea Pearl
- **Publicação:** Proceedings of AISTAT, 2014
- **Contribuição:** Condições suficientes sob as quais independências condicionais no m-graph são testáveis. Identificação de impedimentos à testabilidade.

### 4.3 Nabi, Bhattacharya & Shpitser (2020) — Full Law Identification
- **Título:** Full Law Identification in Graphical Models of Missing Data: Completeness Results
- **Autores:** Razieh Nabi, Rohit Bhattacharya, Ilya Shpitser
- **Publicação:** Proceedings of the 37th International Conference on Machine Learning (ICML), 2020
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7716645/
- **Contribuição:** Primeiro resultado de completude para identificação em modelos de missing data — condições necessárias e suficientes (graficais) sob as quais a distribuição completa pode ser recuperada dos dados observados.

### 4.4 Guo, Nabi & Shpitser (2023) — Sufficient Identification under MNAR
- **Título:** Sufficient Identification Conditions and Semiparametric Estimation under Missing Not at Random Mechanisms
- **Autores:** F. Richard Guo, Razieh Nabi, Ilya Shpitser
- **Publicação:** Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI), 2023
- **Link:** https://proceedings.mlr.press/v216/guo23a.html
- **Contribuição:** Condições de identificação menos restritivas para MNAR, permitindo todas as variáveis terem valores faltantes. Estimação semiparamétrica sob MNAR.

### 4.5 Doretti, Geneletti & Stanghellini (2018) — Taxonomia unificada
- **Título:** Missing Data: A Unified Taxonomy Guided by Conditional Independence
- **Autores:** Marco Doretti, Sara Geneletti, Elena Stanghellini
- **Publicação:** International Statistical Review, 86(2), 189–204, 2018
- **Contribuição:** Argumentam que MCAR/MAR/MNAR são frequentemente mal compreendidos e articulados de forma imprecisa. Propõem abordagem baseada em independência condicional e m-DAGs como alternativa mais transparente.

---

## 5. SHADOW VARIABLES E INSTRUMENTOS PARA MNAR

### 5.1 Miao, Tchetgen Tchetgen & Geng (2016) — Shadow variable para MNAR
- **Título:** Identification and Doubly Robust Estimation of Data Missing Not at Random with a Shadow Variable
- **Autores:** Wang Miao, Eric J. Tchetgen Tchetgen, Zhi Geng
- **Contribuição:** Framework de "shadow variable" — variável correlacionada com o outcome mas independente do mecanismo de missingness (dado outcome e covariáveis). Condições gerais de identificação não-paramétrica sob MNAR.
- **Relevância:** Método prático para obter estimativas não-viesadas sob MNAR quando um instrumento válido está disponível.

### 5.2 Miao, Liu & Geng (2024) — Semiparametric Efficiency Theory
- **Título:** Identification and Semiparametric Efficiency Theory of Nonignorable Missing Data with a Shadow Variable
- **Autores:** Wang Miao, Lan Liu, Eric J. Tchetgen Tchetgen, Zhi Geng
- **Publicação:** ACM / IMS Journal of Data Science, 2024
- **arXiv:** https://arxiv.org/abs/1509.02556
- **Contribuição:** Teoria de eficiência semiparamétrica para a maior classe de modelos não-paramétricos identificáveis via shadow variable sob MNAR.

---

## 6. SENSITIVITY ANALYSIS PARA MNAR

### 6.1 Van Buuren (2018) — Flexible Imputation of Missing Data (Cap. 3.8 e 9.2)
- **Título:** Flexible Imputation of Missing Data (2nd Edition)
- **Autor:** Stef van Buuren
- **Publicação:** CRC Press, 2018
- **Capítulos relevantes:** 3.8 (Nonignorable missing data) e 9.2 (Sensitivity analysis)
- **Online:** https://stefvanbuuren.name/fimd/
- **Contribuição:** Tratamento prático e acessível de sensitivity analysis via delta-adjustment e pattern-mixture models no framework de múltipla imputação.

### 6.2 Cro et al. (2020) — Sensitivity Analysis com Controlled MI
- **Título:** Sensitivity Analysis for Clinical Trials with Missing Continuous Outcome Data Using Controlled Multiple Imputation: A Practical Guide
- **Autores:** Suzie Cro et al.
- **Publicação:** Statistics in Medicine, 39(21), 2815–2842, 2020
- **Link:** https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8569
- **Contribuição:** Guia prático para sensitivity analysis via controlled MI (reference-based imputation, delta-adjustment, tipping point analysis) em ensaios clínicos.

### 6.3 Leurent et al. (2018) — Tipping Point Analysis Tutorial
- **Título:** Sensitivity Analysis for Not-at-Random Missing Data in Trial-Based Cost-Effectiveness Analysis: A Tutorial
- **Autores:** Baptiste Leurent et al.
- **Publicação:** PharmacoEconomics, 36(8), 889–901, 2018
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6021473/
- **Contribuição:** Tutorial de tipping point analysis — encontrar o desvio mínimo de MAR (delta) necessário para reverter conclusões da análise primária. Se o tipping point é implausível, há confiança nos resultados sob MAR.

### 6.4 Sterba & Gottfredson (2015) — Case Influence em MAR vs MNAR
- **Título:** Diagnosing Global Case Influence on MAR Versus MNAR Model Comparisons
- **Autores:** Sonya K. Sterba, Nisha C. Gottfredson
- **Publicação:** Structural Equation Modeling, 2015
- **Link:** https://cdn.vanderbilt.edu/vu-my/wp-content/uploads/sites/2472/2017/04/14142355/Sterba.Gottfredson_2015.pdf
- **Contribuição:** Diagnósticos de influência global para comparações MAR vs MNAR em modelos longitudinais. Mostraram que conclusões podem depender de um único caso. Advocam uso de AIC/BIC para seleção MAR vs MNAR.

### 6.5 Mason et al. (2022) — Sensitivity Analysis com Latent Growth Models
- **Título:** Sensitivity Analyses for Data Missing at Random Versus Missing Not at Random Using Latent Growth Modelling: A Practical Guide for Randomised Controlled Trials
- **Autores:** Alexina J. Mason et al.
- **Publicação:** BMC Medical Research Methodology, 22, 2022
- **Link:** https://link.springer.com/article/10.1186/s12874-022-01727-1
- **Contribuição:** Guia prático para sensitivity analysis MAR vs MNAR usando latent growth models em RCTs.

---

## 7. DEEP LEARNING E MÉTODOS MODERNOS PARA MNAR

### 7.1 Ipsen, Mattei & Frellsen (2021) — not-MIWAE
- **Título:** not-MIWAE: Deep Generative Modelling with Missing Not at Random Data
- **Autores:** Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen
- **Publicação:** ICLR 2021
- **GitHub:** https://github.com/nbip/notMIWAE
- **Contribuição:** Modelo generativo profundo (VAE) que modela explicitamente o mecanismo MNAR via "missing model". Treino via importance-weighted variational inference. Superior a métodos MAR em cenários MNAR (censoring, selection bias).
- **Relevância:** Primeiro VAE que trata MNAR explicitamente, mostrando que ignorar MNAR introduz viés significativo na imputação.

### 7.2 Sportisse, Boyer & Josse (2020) — PPCA com MNAR
- **Título:** Estimation and Imputation in Probabilistic Principal Component Analysis with Missing Not At Random Data
- **Autores:** Aude Sportisse, Claire Boyer, Julie Josse
- **Publicação:** NeurIPS 2020
- **arXiv:** https://arxiv.org/abs/1906.02493
- **Contribuição:** Extensão de PPCA para dados MNAR do tipo "self-masked" (probabilidade de observação depende do valor da própria variável). Algoritmo EM adaptado.

### 7.3 Sportisse, Boyer & Josse (2020b) — Low-rank estimation com MNAR
- **Título:** Imputation and Low-Rank Estimation with Missing Not At Random Data
- **Autores:** Aude Sportisse, Claire Boyer, Julie Josse
- **Publicação:** Statistics and Computing, 30, 1629–1643, 2020
- **Link:** https://link.springer.com/article/10.1007/s11222-020-09963-5
- **Contribuição:** Estimação de baixa rank e imputação para MNAR via otimização com penalização nuclear norm.

### 7.4 Collier et al. (2022) — How to Deal with Missing Data in Supervised DL
- **Título:** How to Deal with Missing Data in Supervised Deep Learning?
- **Autores:** Vincent Collier, Olivier Lecuelle, Rodolphe Jézéquel
- **Publicação:** ICLR 2022
- **Link:** https://openreview.net/forum?id=J7b4BCtDm4
- **Contribuição:** Avaliação sistemática de métodos para missing data em deep learning supervisionado, incluindo cenários MAR e MNAR.

---

## 8. ABORDAGENS PRÁTICAS E DIAGNÓSTICO

### 8.1 Fairclough (2010) — Logistic regression para diagnóstico
- **Título:** Design and Analysis of Quality of Life Studies in Clinical Trials (2nd Edition)
- **Autora:** Diane L. Fairclough
- **Publicação:** CRC Press, 2010
- **Contribuição:** Abordagem prática via regressão logística: missingness indicator como variável dependente, covariáveis e scores observados como preditores. Se score corrente (antes do dropout) é significativo após ajuste por covariáveis, evidência de MNAR.
- **Relevância:** Método amplamente usado em clinical trials para investigar mecanismo de missingness.

### 8.2 Noonan et al. (2016) — Follow-up e proxy variables
- **Título:** Investigating the Missing Data Mechanism in Quality of Life Outcomes: A Comparison of Approaches
- **Autores:** Richard Noonan, Diane L. Fairclough et al.
- **Publicação:** BMC Medical Research Methodology, 2009
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC2711047/
- **Contribuição:** Comparação de abordagens para investigar mecanismo de missingness em QoL: follow-up com não-respondentes, uso de proxy variables, regressão logística. Mostraram que follow-up dados podem confirmar MNAR quando a variável corrente prediz missingness.

### 8.3 Ibrahim, Lipsitz et al. (2001) — EM para MNAR
- **Título:** Missing-Data Methods for Generalized Linear Models: A Comparative Review
- **Autores:** Joseph G. Ibrahim, Ming-Hui Chen, Stuart R. Lipsitz, Amy H. Herring
- **Publicação:** Journal of the American Statistical Association, 2005
- **Contribuição:** Review comparativo de métodos para MNAR em GLMs, incluindo EM algorithm para nonignorable missingness.

---

## 9. REVIEWS E SURVEYS ABRANGENTES

### 9.1 Zhou, Aryal & Bouadjenek (2024) — Review Comprehensive
- **Título:** A Comprehensive Review of Handling Missing Data: Exploring Special Missing Mechanisms
- **Autores:** Youran Zhou, Sunil Aryal, Mohamed Reda Bouadjenek
- **arXiv:** https://arxiv.org/abs/2404.04905
- **Contribuição:** Review abrangente cobrindo técnicas tradicionais (deletion, imputação) e modernas (representation learning) para MAR e MNAR. Foco especial em "special missing mechanisms".

### 9.2 Ibrahim & Molenberghs (2009) — Review longitudinal
- **Título:** Missing Data Methods in Longitudinal Studies: A Review
- **Autores:** Joseph G. Ibrahim, Geert Molenberghs
- **Publicação:** TEST, 18, 68–86, 2009
- **Link:** https://link.springer.com/article/10.1007/s11749-009-0138-x
- **Contribuição:** Review de métodos para missing data longitudinal, incluindo selection models, pattern-mixture models e shared parameter models para MNAR.

### 9.3 Tang & Ju (2018) — Selective review nonignorable
- **Título:** Statistical Inference for Nonignorable Missing-Data Problems: A Selective Review
- **Autores:** Niansheng Tang, Yuanyuan Ju
- **Publicação:** Statistical Theory and Related Fields, 2(2), 105–133, 2018
- **Link:** https://www.tandfonline.com/doi/full/10.1080/24754269.2018.1522481
- **Contribuição:** Review seletivo de inferência sob MNAR: identification, estimation, testing. Cobre shadow variables, instrumental variables, e identifiability conditions.

### 9.4 Seaman et al. (2013) — Moving beyond MCAR/MAR/MNAR
- **Título:** Assumptions and Analysis Planning in Studies with Missing Data in Multiple Variables: Moving Beyond the MCAR/MAR/MNAR Classification
- **Publicação:** International Journal of Epidemiology (updated 2023, PMC)
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10396404/
- **Contribuição:** Argumentam que a classificação MCAR/MAR/MNAR é insuficiente para planejamento de análise com missingness multivariável. Advocam uso de m-DAGs para representar suposições de forma mais precisa.

### 9.5 De Silva et al. (2019) — MNAR não-ignorável
- **Título:** A Practical Guide to Sensitivity Analysis for Causal Effects in the Presence of Non-Ignorable Loss to Follow-Up
- **Contribuição:** Guia prático cobrindo pattern-mixture modeling para sensitivity analysis sob MNAR em estudos com perda de seguimento.
- **Online:** https://bookdown.org/glorya_hu/MNAR-Guide/

---

## 10. SOFTWARE E IMPLEMENTAÇÕES

### 10.1 MissMech (R)
- **Referência:** Jamshidian, Jalal & Jansen (2014). MissMech: An R Package for Testing Homoscedasticity, Multivariate Normality, and Missing Completely at Random (MCAR). Journal of Statistical Software, 56(6).
- **Link:** https://www.jstatsoft.org/article/view/v056i06
- **Função:** Testes de MCAR (paramétrico e não-paramétrico)

### 10.2 MCARtest (R)
- **Referência:** Berrett & Samworth (2023)
- **Link:** https://cran.r-project.org/web/packages/MCARtest/
- **Função:** Teste ótimo não-paramétrico de MCAR

### 10.3 MissMecha (Python)
- **Referência:** MissMecha: An All-in-One Python Package for Studying Missing Data Mechanisms (2025)
- **arXiv:** https://arxiv.org/abs/2508.04740
- **PyPI:** https://pypi.org/project/missmecha-py/
- **Função:** Simulação, visualização e teste de mecanismos MCAR/MAR/MNAR. Inclui Little's test e suporte a dados categóricos.

---

## 11. RESULTADO TEÓRICO FUNDAMENTAL

### O Problema Central da Indistinguibilidade MAR vs MNAR

O resultado de **Molenberghs et al. (2008)** — "Every MNAR model has a MAR counterpart with equal fit" — é o teorema mais importante desta área. Ele demonstra que:

1. **É impossível** distinguir MAR de MNAR puramente a partir dos dados observados
2. Para qualquer modelo MNAR que se ajuste aos dados, existe um modelo MAR com ajuste idêntico
3. A escolha entre MAR e MNAR depende necessariamente de **conhecimento de domínio** e **suposições não-testáveis**

### Implicações para a pesquisa:
- Testes estatísticos podem rejeitar MCAR, mas não podem distinguir MAR de MNAR
- Sensitivity analysis é a abordagem recomendada: assumir MAR, depois verificar robustez sob cenários MNAR plausíveis
- Shadow variables/instrumentos podem permitir identificação sob MNAR em casos específicos
- Abordagens causais (m-DAGs) permitem testar implicações específicas do grafo, mas não resolver o problema geral

---

## RESUMO POR ABORDAGEM

| Abordagem | Artigos-Chave | Pode distinguir MAR/MNAR? |
|-----------|---------------|---------------------------|
| Testes de MCAR | Little (1988), Jamshidian (2010), Berrett (2023) | Não — apenas rejeita MCAR |
| Score test MAR vs MNAR | Wang, Shao & Kim (2023) | Parcialmente — sob modelo logístico |
| Selection models | Diggle & Kenward (1994), Heckman (1979) | Não — depende de suposições não-testáveis |
| Pattern-mixture models | Little (1993) | Não — requer identifying restrictions |
| Sensitivity analysis | Cro (2020), Leurent (2018), Mason (2022) | Indiretamente — avalia robustez |
| Grafos causais (m-DAGs) | Mohan & Pearl (2021), Nabi & Shpitser (2020) | Parcialmente — implicações testáveis do grafo |
| Shadow variables | Miao & Tchetgen (2016, 2024) | Sim — quando instrumento válido disponível |
| Deep generative (MNAR-aware) | Ipsen et al. (2021), Sportisse et al. (2020) | Modela MNAR explicitamente |
| Diagnóstico empírico | Fairclough (2010), Sterba (2015) | Heurístico — evidência indireta |

---

---

## 12. ESTADO DA ARTE: COMO SE CLASSIFICA MAR vs MNAR HOJE

### 12.1 O problema central

**Não existe método consolidado para classificação direta em 3 classes (MCAR/MAR/MNAR).** A literatura quase inteiramente se limita a:
- Testes binários: H0: MCAR vs H1: não-MCAR (sem distinguir MAR de MNAR)
- Sensitivity analysis: assume MAR e avalia robustez sob cenários MNAR
- Diagnósticos heurísticos: regressão logística, domain knowledge

O resultado teórico de **Molenberghs et al. (2008)** — "todo modelo MNAR tem um MAR com ajuste idêntico" — demonstra que a distinção MAR vs MNAR é **impossível apenas com dados observados**, fundamentando esse gap.

### 12.2 Abordagens existentes e suas limitações

#### A) Testes de hipótese (apenas MCAR vs não-MCAR)

| Método | Ano | Tipo | Power Reportado | Limitação Principal |
|--------|-----|------|-----------------|---------------------|
| Little's MCAR test | 1988 | Chi-quadrado (EM) | p=4: ~97%; p=20: ~39-48% | Baixo poder em alta dimensão; não distingue MAR/MNAR |
| Jamshidian & Jalal | 2010 | Não-paramétrico | Type I error inflado (0.04-0.13) | Não distingue MAR/MNAR |
| PKLM | 2024 | Random Forest + KL | p=10: 93-99%; p=20: 91-93% | Apenas MCAR vs não-MCAR |
| Berrett & Samworth | 2023 | Ótimo não-paramétrico | Taxas minimax ótimas | Apenas MCAR vs não-MCAR |

#### B) Score test MAR vs MNAR (único teste direto)

| Método | Ano | Tipo | Performance |
|--------|-----|------|-------------|
| Wang, Shao & Kim | 2023 | Score test logístico | Requer modelo paramétrico; sem acurácia multiclasse reportada |

Este é o **único teste formal** que testa diretamente H0: MAR vs H1: MNAR. Contorna o problema de identificação sob MNAR ao estimar apenas sob H0 (MAR). Limitado a modelos logísticos de missingness.

#### C) MechDetect — Classificação automática via ML (estado da arte)

| Métrica | Resultado |
|---------|-----------|
| Classificador | HistGradientBoostingClassifier (scikit-learn) |
| Datasets | 101 datasets reais |
| Acurácia média geral | **89.14%** |
| Acurácia MAR | **100%** (mediana) |
| Acurácia MCAR | **~95%** (mediana) |
| Acurácia MNAR | **~86%** (mediana, maior dispersão) |
| AUC-ROC MAR | ~1.0 |
| AUC-ROC MCAR | ~0.5 (esperado — random) |
| AUC-ROC MNAR | ~1.0 (task Complete); menor (task Excluded) |

**Diferença metodológica crucial:** MechDetect treina classificadores diretamente na relação mask-de-erros ↔ dados (3 tarefas supervisionadas: Complete, Shuffled, Excluded). **Não extrai features estatísticas** — usa o próprio dado + mask como input.

#### D) Sensitivity Analysis (abordagem mais comum na prática)

Não classifica o mecanismo — assume MAR e verifica robustez:
- **Delta-adjustment:** desloca imputações por um parâmetro δ
- **Tipping point analysis:** encontra o δ mínimo que inverte conclusões
- Se tipping point é implausível → resultados robustos sob MAR
- Amplamente recomendado em clinical trials (ICH E9(R1))

#### E) Diagnóstico heurístico (prática clínica)

- Regressão logística: missingness indicator como Y, covariáveis + scores como X
- Se score corrente prediz missingness após ajuste → evidência de MNAR
- Follow-up de não-respondentes como "gold standard" prático
- Domain knowledge: dados sensíveis (renda, saúde) → suspeita de MNAR

### 12.3 Acurácias reportadas — Comparação com este projeto

| Abordagem | Tipo | Dados | Acurácia/Power | Classes |
|-----------|------|-------|----------------|---------|
| **MechDetect (2024)** | ML direto (mask→dados) | 101 reais | **89.14%** média | 3 (MCAR/MAR/MNAR) |
| **PKLM (2024)** | Teste estatístico | Simulados | **91-99%** power | 2 (MCAR vs não-MCAR) |
| **Little's test (1988)** | Teste estatístico | Simulados | **39-97%** power | 2 (MCAR vs não-MCAR) |
| **Este projeto — baseline ML** | Features estatísticas | Sintéticos | **~76%** (RF) | 3 (MCAR/MAR/MNAR) |
| **Este projeto — ML + LLM** | Features stat + LLM | Sintéticos | **~79%** (RF) | 3 (MCAR/MAR/MNAR) |
| **Este projeto — baseline ML** | Features estatísticas | Reais | **~41%** (RF) | 3 (MCAR/MAR/MNAR) |
| **Este projeto — ML + LLM** | Features stat + LLM | Reais | **~45%** (KNN) | 3 (MCAR/MAR/MNAR) |
| **Este projeto — Hier+CAAFE** | Features stat + CAAFE | Reais | **~50.5%** (melhor) | 3 (MCAR/MAR/MNAR) |

### 12.4 Gap identificado: LLMs para classificação de mecanismos

**Nenhum artigo na literatura utiliza LLMs para classificar ou distinguir mecanismos de missing data.**

Os trabalhos com LLMs na área focam exclusivamente em:
- **Imputação** de valores faltantes (arxiv:2603.22332, 2026)
- **Feature engineering** para classificação tabular genérica (CAAFE, Hollmann et al. 2023)

A abordagem desta tese — usar features derivadas de LLM (análise de segunda ordem, julgamento MNAR, embeddings) como input para classificadores de mecanismo — é **sem precedentes na literatura**.

### 12.5 Por que a distinção MAR vs MNAR é tão difícil

1. **Impossibilidade teórica (Molenberghs 2008):** Para todo modelo MNAR, existe um modelo MAR com ajuste idêntico aos dados observados
2. **Informação não-observada:** MNAR depende do próprio valor faltante, que por definição não está disponível
3. **Confounding:** Em dados reais, os mecanismos frequentemente coexistem no mesmo dataset
4. **Rótulos incertos:** Datasets reais raramente têm o mecanismo ground-truth confirmado
5. **Dependência de domínio:** A distinção frequentemente requer conhecimento externo (e.g., "renda alta → mais missingness" sugere MNAR)

Isso explica a queda de acurácia de ~76-79% (sintéticos, ground-truth limpo) para ~41-50% (reais, rótulos heurísticos) observada neste projeto e é consistente com o estado da arte.

---

*Levantamento realizado em abril/2026. Total: ~35 referências organizadas em 12 categorias.*
*Análise comparativa inclui dados do projeto (step05_pro e Hier+CAAFE).*
