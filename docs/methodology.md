# Methodology

This document gives a self-contained overview of the missdetect methodology.
For the chronological narrative, see [`HISTORICO.md`](HISTORICO.md). For
detailed phase-by-phase decisions, see [`archive/`](archive/).

## 1. Problem statement

Given a tabular dataset and a target column with missing values, decide
whether the missingness mechanism is **MCAR**, **MAR**, or **MNAR** in the
sense of Rubin (1976):

- **MCAR** (Missing Completely At Random): `P(R | Y, X) = P(R)`
- **MAR** (Missing At Random): `P(R | Y, X) = P(R | X_obs)`
- **MNAR** (Missing Not At Random): `P(R | Y, X)` depends on `Y_mis`

where `R` is the missingness indicator, `Y` the variable of interest, `X` the
covariates, and `X_obs` the observed part of `X`.

Following Molenberghs et al. (2008), MAR and MNAR are not separable from the
observed-data law alone — every MNAR model has an MAR counterpart with
identical observed-data fit. The classification problem is therefore not a
test of a hypothesis but a *plausibility ranking* informed by domain priors
encoded as features.

## 2. Feature engineering

We extract three groups of features per dataset / column:

### 2.1 Statistical features (4 per column, 25 total after expansion)

Capture distribution and temporal-pattern signatures of the masked column:
`missing_rate`, `X0_mean`, `X0_std`, `X0_skew`, `X0_kurt`,
`mask_autocorr_{1,2,3}`, `runs_z_score`, `avg_burst_size`, `max_burst_size`,
`corr_mask_{X1..X4}`.

### 2.2 Discriminative features (11)

Designed to separate mechanisms statistically:

- `auc_mask_from_Xobs` — AUC of predicting the missingness indicator from
  the observed covariates. Elevated values suggest **MAR**.
- `little_proxy_score` — proxy for Little's MCAR test (1988).
- `missing_rate_extremes` — proportion of missing among the 10% extreme
  observed values. Elevated values suggest **MNAR**.
- `tail_asymmetry` — CAAFE-MNAR feature; very strong on real data
  (Cohen's d = -0.84).
- `corr_X1_mask`, `X1_mean_diff`, etc.

### 2.3 LLM features (8–9, optional)

Five extraction strategies are implemented in [`src/missdetect/llm/`](../src/missdetect/llm/):

| Strategy | Output features | Cost (per dataset) |
|:--|:-:|:-:|
| `extractor_v2` (original prompt) | 8 | $ |
| `judge_mnar` (binary MCAR vs MNAR) | 4 | $ |
| `embeddings` (sentence-transformers, local) | 10 | free |
| `context_aware` (DAG + counter-arguments — **Step 1**) | 9 | $$ |
| `self_consistency` (5 perspectives, CISC voting) | 8 | $$$ |

Each strategy returns a confidence vector for MCAR / MAR / MNAR plus
auxiliary scores (e.g. `llm_ctx_self_censoring`, `llm_ctx_domain_prior`).
The `context_aware` extractor uses dataset metadata (domain description,
column semantics) drawn from `src/missdetect/metadata/`. Two metadata
variants are available:

- `default` — includes mechanism-suggestive context (used in earlier
  experiments, mildly leaky).
- `neutral` — strips identifying domain hints to close information channel
  F. **Used in Step 1 V2 Neutral, the canonical experiment.**

## 3. Hierarchical classification

```mermaid
flowchart LR
  X((feature<br/>vector)) --> L1{Level 1<br/>RandomForest<br/>n_estimators=100}
  L1 -->|p_MCAR ≥ τ_high| outMCAR[predict MCAR]
  L1 -->|p_MCAR ≤ τ_low| L2[Level 2]
  L1 -->|τ_low < p_MCAR < τ_high| Lroute{soft3zone routing}
  Lroute --> L2
  L2 -->|NaiveBayes<br/>+ Cleanlab weights| outBin{MAR or MNAR}
  outBin --> outMAR[predict MAR]
  outBin --> outMNAR[predict MNAR]
```

- **Level 1 (MCAR vs non-MCAR)**: Random Forest, 100 trees, defaults. The
  binary task is easier than 3-way and statistical features dominate.
- **Level 2 (MAR vs MNAR)**: NaiveBayes with sample weights derived from
  Cleanlab's per-sample noise probability (`1 − P(label is correct)`).
- **soft3zone routing**: when L1's confidence falls in the ambiguous middle
  band, the sample is routed through L2 with a probabilistic blend of L1
  and L2 outputs rather than a hard cut.

NaiveBayes was chosen over RandomForest, GradientBoosting, MLP, SVM,
XGBoost-tuned-by-Optuna and CatBoost-tuned-by-Optuna at Level 2 because it
**dominated all of them by +6–13pp under Group LOGO CV** in the V3+
experiments. The diagnosis: under 59.4% measured label noise (Cleanlab),
calibrated probabilistic models beat high-capacity discriminative models
that memorise noisy boundaries.

## 4. Evaluation

We report two metrics:

1. **Group 5-Fold CV** (referred to as "CV" throughout): 5-fold
   cross-validation with `GroupKFold` on the dataset-of-origin grouping
   variable. Bootstraps from the same source dataset are kept entirely in
   train or entirely in test.
2. **Holdout** (75/25 GroupShuffleSplit, fixed seed 42): a single split
   used for sanity checking and confusion matrices.

The Group split fixes the data-leakage issue identified in Phase 1 of the
project — earlier results that used random K-Fold across bootstraps
inflated accuracy from ~43% to a misleading 100% by memorising dataset
fingerprints. See [`archive/01_correcao_pipeline/RESULTADOS_FASE3.md`](archive/01_correcao_pipeline/RESULTADOS_FASE3.md).

## 5. Datasets

### 5.1 Synthetic (1,200)

Generated with [`mdatagen`](https://pypi.org/project/mdatagen/) and our own
extensions. 12 mechanism variants × 100 datasets each, balanced across
MCAR / MAR / MNAR. See `src/missdetect/metadata/synthetic_variants_metadata.json`
for the per-variant configuration (seed, missingness rate, parameter ranges).

### 5.2 Real (32 columns from 21 source datasets)

Curated from UCI MLR, OpenML, Kaggle, R packages (`mice`, `naniar`,
`Ecdat`, `datasets`), NHANES CDC, and SUPPORT2 UCI.
Audited 2026-05-06: 7 datasets with doubtful classification removed;
6 reclassified from MCAR to MAR after v2b protocol verification and
domain review. Total: 6 MCAR, 13 MAR, 13 MNAR.
Per-column provenance, licence and mechanism-labelling justification is in
[`../data/real/sources.md`](../data/real/sources.md). Each column was
bootstrapped to ~50 series of length 500, totalling 1,421 bootstrap
samples. Mechanism labels are domain-expert assignments cross-checked with
two validation protocols:

- **v1** — three independent tests: Little's MCAR (`src/missdetect/validar_rotulos.py`),
  point-biserial correlation `mask × X_i`, and KS observed-vs-imputed.
  57% of expert labels disagree with at least one of the three (see
  limitations).

- **v2** — layered protocol introduced in
  `src/missdetect/validar_rotulos_v2.py`. Aggregates evidence in three
  layers (Camadas A/B/C) plus a Bayesian reconciliation calibrated on the
  1,200 synthetic datasets:

  - **Camada A — MCAR**: majority vote across Little (paramétrico),
    PKLM (Spohn 2024, non-parametric, in `baselines/pklm.py`) and
    a Bonferroni-corrected Levene-stratified test.
  - **Camada B — MAR**: AUC of a Random Forest predicting `mask` from
    `X_obs` with a 200-permutation p-value, plus mutual information.
    Captures non-linear dependencies the point-biserial test misses.
  - **Camada C — MNAR**: four CAAFE-MNAR scores (tail asymmetry,
    kurtosis excess, conditional entropy, missing rate by quartile)
    thresholded at the Youden-optimal cut calibrated against synthetic
    ground truth.
  - **Camada D — Reconciliation**: Bayesian aggregation via
    Gaussian-kernel KDEs fitted per mechanism on the synthetic scores
    (artefacts: `data/calibration.json`, `data/calibration_scores.npz`).

  Calibration is performed by `src/missdetect/calibrar_protocolo.py` and
  reports the protocol's accuracy on synthetic ground truth as a sanity
  check before applying it to real data.

## 6. Reproducibility caveats

- Random seeds are pinned (`seed=42` for splits; per-dataset seeds for
  bootstrap recorded in metadata).
- LLM outputs are non-deterministic. We cache responses to keep cost down,
  but exact replication of Pro/Flash runs requires either the cache or
  budget for a fresh extraction. Statistical-only experiments are fully
  deterministic.
- Cleanlab is run on the L2 training fold only, with a fixed seed.

## 7. References

See [`bibliography.md`](bibliography.md) for the full annotated bibliography.
The minimal essential reads:

- Rubin (1976) — defines MCAR / MAR / MNAR formally.
- Little & Rubin (2019, 3rd ed.) — *Statistical Analysis with Missing Data*.
- Little (1988) — MCAR test.
- Molenberghs et al. (2008) — MNAR-MAR equivalence theorem (the ceiling).
- Mohan & Pearl (2021) — graphical models for missing data.
- Le et al. (2024) — MechDetect, the closest comparable baseline.
- Sportisse et al. (2024) — PKLM, MCAR test based on classification.
