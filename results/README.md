# Results

Reproducible experiment outputs. Two canonical experiments are kept here;
the remaining 25 historical experiments are archived under
`_archive/` (gitignored — available on request as a tarball or via Zenodo).

## Canonical experiments

### `step1_v2_neutral/` — 29 datasets, Pro + neutral metadata

The headline LLM-augmented experiment. Run on the expanded benchmark of 29
real datasets with `gemini-3-pro-preview` and the `context_aware`
extractor (Step 1 prompt: 3 canonical examples + MNAR typology + explicit
anti-MAR-bias instruction) under the `neutral` metadata variant.

| Best classifier | Holdout (n=395) | CV (Group 5-Fold) |
|:--|:-:|:-:|
| **NaiveBayes** | **55.19%** | **49.33% ± 14.2%** |
| LogisticRegression | 54.94% | 41.54% ± 23.5% |

Cost: ~$30 for 1,421 bootstraps × ~2 LLM calls each. Wall-clock: 1h33min
on a single laptop with 10 worker threads.

### `step05_pro/` — 23 datasets, V3+ peak

The peak result of the V3+ phase: hierarchical classification with
Cleanlab-weighted Level 2 NaiveBayes and soft3zone routing, on the
original 23-dataset benchmark. **No LLM features.**

| Configuration | LOGO CV | Holdout |
|:--|:-:|:-:|
| **V3+ (NaiveBayes + Cleanlab + soft3zone)** | **55.97%** | 50.5% |
| V3 plain | 51.4% | 50.5% |
| Direct 3-way (no hierarchy) | 41.4% | — |

This is the highest accuracy ever achieved on the 23-dataset benchmark.
The −7pp gap between this and `step1_v2_neutral` is attributable
predominantly to the +6 datasets in the expanded benchmark (clinically
hard cases like `pima_insulin`, `kidney_pot`, `hypothyroid_t4u`) — see
`docs/archive/08_step1_v2_neutral_results/04_ANALISE_REGRESSAO.md` for
the full diagnosis.

## Layout

Each experiment directory contains:

```
<experiment>/
├── apenas_ml/            # Statistical features only
│   └── baseline/
│       ├── X_features.csv      # n × p feature matrix
│       ├── y_labels.csv        # 0=MCAR, 1=MAR, 2=MNAR
│       ├── groups.csv          # Source-dataset grouping for GroupKFold
│       ├── confusion_matrices.json
│       ├── cv_scores.csv
│       ├── feature_importance.csv
│       ├── metrics_per_class.csv
│       ├── relatorio.txt       # Human-readable report
│       └── *.png               # Diagnostic plots
├── ml_com_llm/           # ML + LLM features (when applicable)
│   └── gemini-*/
├── ensemble/             # Adaptive ensemble (when applicable)
├── hierarquico/          # Hierarchical classification outputs
└── experiment_config.json
```

`relatorio.txt` is the place to start for any experiment — it contains
per-classifier numbers and the confusion matrix in plain text.

## Archive

The historical experiments (`_archive/`) include intermediate runs from
each of the 7 project phases — feature ablations, alternative LLM prompts,
self-consistency runs, etc. They are not strictly needed to reproduce the
headline numbers but are useful for auditing. They are gitignored to keep
the repository slim and will be deposited on Zenodo with a DOI for the
final paper submission.
