# Annotated bibliography

Curated references that informed the design and analysis of missdetect,
grouped by topic. Citations follow author-year-and-venue format. PDFs of
the referenced works are kept locally outside this repository (under
`_local_references/`, gitignored) for copyright reasons.

The single most important read is **Rubin (1976)** — every other entry in
this bibliography either refines, generalises, tests, or pushes back against
that paper.

---

## 1. Foundational theory of missing data

- **Rubin, D. B. (1976).** *Inference and Missing Data*. **Biometrika** 63(3),
  581–592. — Defines MCAR / MAR / MNAR. Paper-of-origin.
- **Little, R. J. A. (1988).** *A Test of Missing Completely at Random for
  Multivariate Data with Missing Values*. **JASA** 83(404), 1198–1202. —
  Standard MCAR test, our `little_proxy_score` feature is its pragmatic
  approximation.
- **Little, R. J. A., & Rubin, D. B. (2019).** *Statistical Analysis with
  Missing Data* (3rd ed.). Wiley. — Textbook reference for the field.
- **Schafer, J. L., & Graham, J. W. (2002).** *Missing Data: Our View of the
  State of the Art*. **Psychological Methods** 7(2), 147–177.

## 2. MAR vs MNAR distinction (the core problem)

- **Molenberghs, G., Beunckens, C., Sotto, C., & Kenward, M. G. (2008).*
  *Every missingness not at random model has a missingness at random
  counterpart with equal fit*. **JRSS-B** 70(2), 371–388. — **The
  impossibility theorem.** Defines the ceiling our work brushes against.
- **Wang, Z., Shao, J., & Kim, J. K. (2023).** *Score Test for MAR vs MNAR*.
  Recent practical test exploiting moment conditions.
- **Jamshidian, M., & Jalal, S. (2010).** *Tests of Homoscedasticity,
  Normality, and Missing Completely at Random for Incomplete Multivariate
  Data*. **Psychometrika** 75(4), 649–674.
- **Berrett, T. B., & Samworth, R. J. (2023).** *Optimal Nonparametric
  Testing of MCAR with Unspecified Alternatives*. **Annals of Statistics**.
- **Diggle, P., & Kenward, M. G. (1994).** *Informative Drop-out in
  Longitudinal Data Analysis*. **Applied Statistics** 43(1), 49–93.
- **Potthoff, R. F., Tudor, G. E., Pieper, K. S., & Hasselblad, V. (2006).
  *Can one assess whether missing data are missing at random in medical
  studies?*. **Stat. Methods Med. Res.** 15(3), 213–234.

## 3. Causal / graphical perspective on missingness

- **Mohan, K., & Pearl, J. (2021).** *Graphical Models for Processing
  Missing Data*. **JASA** 116(534), 1023–1037.
- **Nabi, R., & Shpitser, I. (2020).** *Full-law identification in
  graphical models of missing data*. **Biometrika** 107(2), 485–501.
- **Guo, F. R., Nabi, R., & Shpitser, I. (2023).** *Sufficient Identification
  Conditions and Semiparametric Estimation under Missing Not At Random
  Mechanisms*. **JMLR**.
- **Miao, W., & Tchetgen Tchetgen, E. J. (2016).** *On varieties of
  doubly robust estimators under missingness not at random with a shadow
  variable*. **Biometrika** 103(2), 475–482.

## 4. Tests for MCAR by classification

- **Spohn, R., & Bühlmann, P. — Sportisse, A. (2024).** *PKLM: A Flexible
  MCAR Test Using Classification*. **NeurIPS / arXiv 2302.10902**. —
  Treats MCAR testing as a binary classification problem; cannot detect
  MNAR by design.
- **Le, T., et al. (2024).** *MechDetect: Detecting Data-Dependent Errors*.
  — Closest comparable baseline; reports 89% on 101 real datasets but with
  systematic MNAR bias in our reproduction.

## 5. Imputation under missingness mechanisms

- **van Buuren, S. (2018).** *Flexible Imputation of Missing Data* (2nd
  ed.). CRC Press. — Authoritative reference for multiple imputation.
- **Ipsen, N. B., et al. (2021).** *not-MIWAE: Deep Generative Modelling
  with Missing Not At Random Data*. **ICLR**.
- **Sportisse, A., Boyer, C., & Josse, J. (2020).** *Imputation and
  low-rank estimation with Missing Non At Random data*. **Stat.
  Comput.** 30, 1629–1643.
- **Mattei, P.-A., & Frellsen, J. (2019).** *MIWAE: Deep Generative
  Modelling and Imputation of Incomplete Data Sets*. **ICML**.

## 6. LLMs in tabular / missing-data contexts

- **Hollmann, N., Müller, S., & Hutter, F. (2023).** *Large Language
  Models for Automated Data Science: Introducing CAAFE for Context-Aware
  Automated Feature Engineering*. **NeurIPS**. — Inspiration for our
  deterministic CAAFE-MNAR features. We do not reimplement the original
  CAAFE LLM code-generation loop; see [`caafe_mnar.md`](caafe_mnar.md).
- **Wei, J., et al. (2022).** *Chain-of-Thought Prompting Elicits Reasoning
  in Large Language Models*. **NeurIPS**.
- **Wang, X., et al. (2022).** *Self-Consistency Improves Chain of Thought
  Reasoning*. **ICLR 2023**.
- **Time-LLM (Jin et al., 2024).** *Time Series Forecasting by Reprogramming
  LLMs*. **ICLR**.
- **Chronos (Ansari et al., 2024).** *Learning the Language of Time Series*.
  **TMLR**.

## 7. Robust learning under label noise

- **Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021).** *Confident
  Learning: Estimating Uncertainty in Dataset Labels*. **JAIR** 70, 1373–1411. —
  Cleanlab paper. Drives our V3+ Level-2 weighting.
- **Patrini, G., et al. (2017).** *Making Deep Neural Networks Robust to
  Label Noise: a Loss Correction Approach*. **CVPR**.

## 8. Classifiers / methods used

- **Breiman, L. (2001).** *Random Forests*. **Machine Learning** 45(1), 5–32.
- **Friedman, J. H. (2001).** *Greedy Function Approximation: A Gradient
  Boosting Machine*. **Annals of Statistics** 29(5), 1189–1232.
- **Cortes, C., & Vapnik, V. (1995).** *Support-Vector Networks*. **Machine
  Learning** 20, 273–297.
- **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to
  Interpreting Model Predictions*. **NeurIPS**. — SHAP.

## 9. Software and benchmarks

- **mdatagen** — Python package for synthetic missing-data generation.
  https://pypi.org/project/mdatagen/
- **missmecha-py** — Python port of MissMech R package, used for label
  validation. https://github.com/missmecha/missmecha-py
- **MissMech (R)** — Jamshidian & Jalal's R package, the original
  implementation of their 2010 test.
- **OpenML** (Vanschoren et al. 2014) — `sklearn.datasets.fetch_openml()`
  is our canonical access path for 12 of the 29 real datasets.
- **UCI Machine Learning Repository** (Dua & Graff 2017) — 14 of the 29
  real datasets.

## 10. Surveys

- **Zhou, B., et al. (2024).** *A Comprehensive Review of Handling Missing
  Data: Imputation Methods and Beyond*. — Most recent comprehensive review.
- **Emmanuel, T., et al. (2021).** *A survey on missing data in machine
  learning*. **J. Big Data** 8(140).

---

For the chronological reading list (which paper informed which decision in
which phase) see [`HISTORICO.md`](HISTORICO.md), and the per-phase notes in
[`archive/15_MAR_vs_MNAR_Distincao/`](archive/) — these contain longer
free-form annotations.
