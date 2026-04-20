# Ablation Study Results — Context-Aware LLM Features (Neutral Metadata)

**Purpose.** Canonical source of numbers for the ablation experiment reported in the thesis/article. All values here are cross-verified against the raw `cv_scores.csv`, `relatorio.txt`, and `training_summary.json` files under `Output/v2_improved/`.

**Date of last run:** 2026-04-19.
**Dataset:** 23 real datasets from `processado/{MCAR,MAR,MNAR}/`, bootstrapped to 1132 samples (232 MCAR, 550 MAR, 350 MNAR).
**Validation:** GroupKFold-5 by source dataset + LODO (23 folds, one dataset per fold).
**Metadata:** MEDIUM-scope neutralized (`real_datasets_metadata_neutral.json`) — generic source, no clinical cutoffs, no canonical dataset names. Domain + variable name preserved as legitimate modelling inputs.

---

## Main Ablation Table

All values are **best-model cross-validation accuracy** (best of 7 classifiers: RandomForest, GradientBoosting, LogisticRegression, SVM-RBF, KNN, MLP, NaiveBayes). Bootstrap 95% CIs available in `forensic_summary.csv` where applicable.

| # | Scenario | Features | Best model | GroupKFold-5 | LODO | Experiment ID |
|---|----------|----------|------------|--------------|------|---------------|
| A | Baseline (statistical only) | 21 | — | **40.5%** | — | `ctx_baseline` |
| B | Baseline + CAAFE | 25 | NaiveBayes | **47.6%** | 47.1% | `step01_caafe_real` |
| C | Baseline + CAAFE + LLM data-driven (no `domain_prior`) | 30 | NaiveBayes | 50.5% | 50.4% | `forensic_neutral_v2` (C_no_prior) |
| D | Baseline + CAAFE + all LLM features (with `domain_prior`) | 31 | NaiveBayes | **56.2%** | **54.3%** | `forensic_neutral_v2` (C_full) |
| E | `llm_ctx_domain_prior` alone | 1 | 6 models tied | **63.1%** | **63.1%** | `forensic_neutral_v2` (C_only_prior) |

Chance level for 3-class stratified: 33.3%.

---

## Marginal Contribution of Each Component

Reading the table as successive feature-group additions (using GroupKFold-5):

| Transition | What was added | Marginal Δ | Interpretation |
|------------|----------------|------------|----------------|
| A → B | 4 CAAFE-MNAR features (pure Python) | **+7.1 pp** | Statistical MNAR features provide real signal. |
| B → C | 5 LLM data-driven features (surprise, stats_consistency, counter_strength, confidence_delta, domain_confidence) | +2.9 pp | Marginal contribution from LLM statistical reasoning. |
| C → D | `llm_ctx_domain_prior` (1 feature) | **+5.7 pp** | LLM domain reasoning from variable name + domain context. |
| — → E | `llm_ctx_domain_prior` alone vs baseline (A) | +22.6 pp | A single LLM feature captures domain reasoning about mechanism. |

---

## Validation of Generalization (LODO ↔ GroupKFold)

The LODO (Leave-One-Dataset-Out) validation confirms that the observed gains are not dataset memorization.

| Scenario | GroupKFold-5 | LODO | Δ |
|----------|--------------|------|---|
| B (`step01_caafe_real`) | 47.6% | 47.1% | −0.5 pp |
| C (`C_no_prior`) | 50.5% | 50.4% | −0.1 pp |
| D (`C_full`) | 56.2% | 54.3% | −1.9 pp |
| E (`C_only_prior`) | 63.1% | 63.1% | 0.0 pp |

**Conclusion:** The LLM's behavior generalizes across datasets. The domain reasoning from variable name + domain transfers to unseen domains.

---

## Per-Class Breakdown: domain_prior Distribution (Neutral Metadata)

| True Class | dp=0.0 (→MCAR) | dp=0.5 (→MAR) | dp=1.0 (→MNAR) | LLM Correct |
|------------|-----------------|----------------|-----------------|-------------|
| MCAR (n=232) | **27.6%** | 67.2% | 5.2% | 27.6% |
| MAR (n=550) | 1.3% | **96.5%** | 2.2% | 96.5% |
| MNAR (n=350) | 4.9% | 61.1% | **34.0%** | 34.0% |

**Observations:**
- **MAR is the strongest class:** 96.5% correctly identified — the LLM reliably infers MAR from domain + variable context.
- **MCAR is weak:** Only 27.6% correctly tagged; 67.2% default to MAR (dp=0.5). Without mechanism-revealing context, the LLM defaults to moderate position.
- **MNAR is the hardest class:** Only 34.0% correctly tagged; 61.1% confused as MAR. The LLM cannot reliably distinguish MNAR from MAR using only domain + variable name.

**Naive accuracy** (domain_prior thresholds alone): **63.1%** — driven primarily by MAR's high correct rate (96.5%).

---

## Feature Importance (CAAFE-only, Scenario B, RandomForest)

Top-10 features from `step01_caafe_real/real/apenas_ml/baseline/relatorio.txt`:

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `X0_censoring_score` | 11.52% | Statistical (discriminative) |
| 2 | `X0_obs_vs_full_ratio` | 9.38% | Statistical (summary) |
| 3 | **`caafe_tail_asymmetry`** | **9.31%** | **CAAFE** |
| 4 | **`caafe_kurtosis_excess`** | **8.72%** | **CAAFE** |
| 5 | **`caafe_cond_entropy_X0_mask`** | **7.37%** | **CAAFE** |
| 6 | `X0_obs_skew_diff` | 6.37% | Statistical |
| 7 | `X1_mean_diff` | 5.92% | Discriminative |
| 8 | `X0_ks_obs_vs_imputed` | 5.92% | Discriminative |
| 9 | `X0_mean_shift_X1_to_X4` | 5.03% | Discriminative |
| 10 | `little_proxy_score` | 4.30% | MechDetect |

The 4 CAAFE features together account for **28.3%** of feature importance, despite being only 4/25 = 16% of features. They are disproportionately important for the statistical part of the pipeline.

---

## Decomposition: Where Does Each Component's Signal Come From?

| Source of accuracy | Accuracy | Marginal contribution |
|-------------------|----------|----------------------|
| Chance (3 classes) | 33.3% | — |
| Statistical features alone (baseline) | 40.5% | +7.2pp from data patterns |
| + CAAFE features | 47.6% | +7.1pp from MNAR-targeted stats |
| + 5 data-driven LLM features | ~50.5% | +2.9pp from LLM reasoning on stats |
| + domain_prior (neutral metadata) | **56.2%** | +5.7pp from domain+variable reasoning |

**Naive accuracy** (domain_prior alone): **63.1%** — higher than the full model because the ML pipeline blends domain_prior with noisier statistical features.

---

## Two Accuracy Regimes

1. **Statistical regime (40-51%).** Pure data patterns. Statistical features (baseline + CAAFE + data-driven LLM) capture weak but real signal. Ceiling: ~51%.

2. **Domain reasoning regime (56-63%).** The LLM uses `domain` + `x0_variable` to infer mechanisms from world knowledge, without mechanism-revealing metadata. This adds +5-13pp over statistical features alone. This is **genuine and publishable**.

---

## Interpretation for the Thesis

The ablation reveals two regimes of performance on real-world missingness data:

1. **Purely statistical (A → B → C, 40.5% → 47.6% → 50.5%).** Adding CAAFE features yields **+7.1 pp** and data-driven LLM features add another **+2.9 pp**. This is the upper bound achievable without domain knowledge.

2. **Statistical + LLM domain reasoning (C → D, 50.5% → 56.2%).** Adding `domain_prior` — which encodes the LLM's inference about the mechanism given only domain and variable name — yields **+5.7 pp**. This represents genuine domain reasoning.

The claim *"LLM domain reasoning improves statistical ML classification"* is **supported** by this ablation: the domain_prior adds +5.7pp through legitimate inference from domain + variable name. The claim *"LLM features enhance ML classification via data-driven reasoning"* is **modestly supported** — the 5 data-driven features contribute +2.9pp.

The **honest upper bound** for LLM-enhanced classification with neutral metadata is **~56%** (full model) to **63%** (domain_prior alone) on this dataset collection.

---

## Artifacts and Reproducibility

All raw results are stored under `Output/v2_improved/`:

| Scenario | Path |
|----------|------|
| A (baseline) | `ctx_baseline/real/apenas_ml/baseline/` |
| B (CAAFE-only) | `step01_caafe_real/real/apenas_ml/baseline/` |
| C, D, E (neutral ablation) | `forensic_neutral_v2/` |

Key files:
- `cv_scores.csv` — per-fold cross-validation scores per model
- `relatorio.txt` — formatted summary with confusion matrices
- `feature_importance.csv` — per-feature RandomForest importance
- `training_summary.json` — configuration and feature list
- `forensic_summary.csv` — accuracy, F1-macro and 95% bootstrap CIs (1000 iterations)
- `lodo_per_dataset.csv` — per-dataset LODO accuracy for scenarios C/D/E

---

## Status of Ablation Plan

| Step | Description | Status | Output |
|------|-------------|--------|--------|
| 1 | CAAFE-only baseline on real data | Done | `step01_caafe_real/` |
| 2 | Ablation removing `domain_prior` | Done | `forensic_neutral_v2/` |
| 3 | Neutral metadata experiment | Done | `forensic_neutral_v2/` |
| 4 | Leave-One-Dataset-Out validation | Done | `forensic_neutral_v2/lodo_per_dataset.csv` |
| 5 | Document results | Done | This file + `forensic_analysis_context_aware.md` |
