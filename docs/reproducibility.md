# Reproducibility guide

This document is the operational replay of the experiments reported in the
README and the thesis. Every command assumes you are at the repository
root with the package installed (`uv pip install -e ".[boosting,llm]"`).

## 0. Prerequisites

- Python 3.11+
- `uv` (or pip) with the repo installed in editable mode
- For LLM experiments: `GEMINI_API_KEY` (and optionally `OPENAI_API_KEY`)
  exported in the environment or a `.env` file at the repo root
- ~5 GB free disk for synthetic + real datasets and outputs

## 1. Verify the install

```bash
python -c "import missdetect; print(missdetect.__version__)"
# expected: 0.1.0

missdetect-extract --help
# expected: argparse usage banner
```

## 2. The headline experiments

### 2.1 Synthetic baseline (≈5 min, no LLM)

```bash
missdetect-extract --model none --data synthetic
missdetect-train  --model none --data synthetic --experiment synthetic_baseline
```

Output goes to `results/synthetic_baseline/`. Expected best CV: ~75% (NB)
on synthetic, easier than real because synthetic mechanisms are clean.

### 2.2 V3+ peak on 23 real datasets (≈15 min, no LLM)

```bash
missdetect-extract --model none --data real --metadata-variant neutral
python -m missdetect.train_hierarchical_v3plus \
  --model none --data real \
  --experiment step05_pro
```

Expected: **55.97% LOGO CV (NaiveBayes)**. Outputs in `results/step05_pro/`
match the committed reference outputs bit-for-bit (modulo plotting font
warnings). Diff against the reference:

```bash
diff <(python -m missdetect.train_hierarchical_v3plus \
        --model none --data real --experiment step05_pro_repro \
        --quiet) \
     results/step05_pro/relatorio.txt
```

### 2.3 Step 1 V2 Neutral on 29 real datasets (≈90 min, ~$30 USD)

This is the most expensive run and the canonical Pro+LLM experiment.

```bash
# Half 1 — 15 datasets (~47 min, ~$15)
missdetect-extract \
  --model gemini-3-pro-preview \
  --llm-approach context_aware \
  --metadata-variant neutral \
  --data real \
  --datasets-include src/missdetect/metadata/datasets_part1.txt \
  --workers 10 \
  --output-suffix part1

# Half 2 — 14 datasets (~46 min, ~$15)
missdetect-extract \
  --model gemini-3-pro-preview \
  --llm-approach context_aware \
  --metadata-variant neutral \
  --data real \
  --datasets-include src/missdetect/metadata/datasets_part2.txt \
  --workers 10 \
  --output-suffix part2

# Merge halves
python -m missdetect.merge_halves \
  --part1 results/step1_v2_neutral_part1 \
  --part2 results/step1_v2_neutral_part2 \
  --output results/step1_v2_neutral

# Train on the merged feature set
missdetect-train \
  --model gemini-3-pro-preview \
  --data real \
  --experiment step1_v2_neutral
```

Expected: **49.33% Group 5-Fold CV (NaiveBayes)**, **55.19% holdout**.

LLM responses are cached on disk; subsequent runs skip API calls when the
cache is warm. To force a fresh extraction, delete
`~/.cache/missdetect/llm/`.

## 3. Ablations and supporting experiments

| Goal | Command | Wall-clock | Cost |
|:--|:--|:-:|:-:|
| Flash (cheap) head-to-head vs Pro | `missdetect-extract --model gemini-3-flash-preview --llm-approach context_aware --metadata-variant neutral --data real` | 30 min | ~$3 |
| Self-consistency 5-perspective | `missdetect-extract --model gemini-3-pro-preview --llm-approach self_consistency --workers 5` | 4–5 hr | ~$150 |
| SHAP feature importance | `python -m missdetect.analyze_shap --experiment step1_v2_neutral` | 5 min | $0 |
| Cleanlab label-noise diagnosis | `python -m missdetect.clean_labels --experiment step1_v2_neutral` | 2 min | $0 |
| Forensic ablation per feature group | `python -m missdetect.forensic_ablation --experiment step1_v2_neutral` | 10 min | $0 |
| MechDetect baseline | `python -m missdetect.baselines.mechdetect_original --data real` | 15 min | $0 |
| PKLM baseline | `python -m missdetect.baselines.pklm --data real` | 8 min | $0 |

## 4. Regenerating data from scratch

### 4.1 Synthetic data

```bash
python -m missdetect.data_generation.gerador_v2 \
  --output data/synthetic \
  --seed 42 \
  --n-per-variant 100
```

Produces 1,200 series (12 variants × 100). Deterministic given the seed.

### 4.2 Real data

```bash
python -m missdetect.data_generation.preparar_dados_reais  \
  --output data/real/raw         # 5 datasets from local CSVs / R sources

python -m missdetect.data_generation.expandir_dados_reais  \
  --output data/real/raw         # 24 datasets via OpenML / URL

python -m missdetect.data_generation.subdividir_dados_reais \
  --input data/real/raw \
  --output data/real/processed \
  --bootstraps 50 \
  --seed 42
```

Network access required for the OpenML / Kaggle / GitHub URLs. Hashes of
the expected processed bootstrap files are recorded in
`data/real/processed/MANIFEST.sha256` (generated on first run).

## 5. Hardware and timing baselines

The numbers above are measured on:

- Apple MacBook Pro M2 Max, 32 GB RAM, macOS 14.5
- Python 3.11.9
- numpy 1.26, pandas 2.2, scikit-learn 1.4

Linux x86 with comparable RAM is ~10–15% faster on CPU-bound steps; LLM
extraction is API-bound and identical.

## 6. Troubleshooting

- **"GEMINI_API_KEY not set"** — export it or create `.env`. The `none`
  model never needs credentials.
- **"OutOfMemory in train_hierarchical_v3plus"** — Cleanlab's K-fold
  ensemble is memory-hungry; pass `--cleanlab-folds 3` instead of the
  default 5.
- **Reproducing Pro results gives slightly different numbers** — Gemini
  Pro is non-deterministic at temperature 0; expect ±0.5pp drift between
  runs. Use the cached responses bundled in
  `results/step1_v2_neutral/llm_cache.json` for exact reproduction.
