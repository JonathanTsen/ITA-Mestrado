# Data

Two collections accompany missdetect:

| Folder | Contents | Size | Provenance |
|:--|:--|:-:|:--|
| `synthetic/` | 1,200 generated time-series with controlled mechanisms | ~80 MB | `mdatagen` + custom generator (`src/missdetect/data_generation/gerador_v2.py`) |
| `real/` | 29 columns from 18 source datasets, with bootstraps | ~50 MB | UCI / OpenML / Kaggle / R packages — see `real/sources.md` |

## Synthetic data

```
synthetic/
├── MCAR/   # 300 series
├── MAR/    # 500 series
└── MNAR/   # 400 series
```

Each filename encodes the mechanism, variant, seed and missing rate:
`{mechanism}_{variant}_seed{seed}_mr{missing_rate}.txt`. Example:
`MAR_logistic_seed42_mr0.3.txt`.

The full per-variant configuration (seed range, parameter ranges,
generator class) is in `src/missdetect/metadata/synthetic_variants_metadata.json`.

To regenerate from scratch:

```bash
python -m missdetect.data_generation.gerador_v2 \
  --output data/synthetic --seed 42 --n-per-variant 100
```

## Real data

```
real/
├── raw/                  # Original CSVs as obtained
│   ├── MCAR/
│   ├── MAR/
│   └── MNAR/
├── processed/            # Bootstrapped univariate series used in experiments
│   ├── MCAR/             # 9 source columns × ~50 bootstraps
│   ├── MAR/              # 11 × ~50
│   └── MNAR/             # 9 × ~50
├── processed_chunks/     # Same as processed/, sharded for parallel processing
└── sources.md            # Per-column provenance and licence
```

The 29 mechanism-labelled columns come from 18 source datasets. Mechanism
labels are domain-expert judgements cross-checked with three statistical
tests (Little's MCAR, correlation tests, Kolmogorov–Smirnov) — note that
57% of expert labels disagree with at least one of the three statistical
tests, illustrating the difficulty of the problem (see
`docs/archive/02_plano1_melhorias/STEP03_dados_missmecha_rotulos.md`).

To regenerate the bootstraps from sources:

```bash
python -m missdetect.data_generation.preparar_dados_reais  --output data/real/raw
python -m missdetect.data_generation.expandir_dados_reais  --output data/real/raw
python -m missdetect.data_generation.subdividir_dados_reais \
  --input data/real/raw --output data/real/processed --bootstraps 50 --seed 42
```

This step requires network access to OpenML / Kaggle / GitHub.

## Licensing

The bundled real datasets are **redistributed under their original licences**
— UCI MLR (CC BY 4.0 for most datasets), OpenML (CC BY 4.0), Kaggle (per
dataset; the two we use are public-domain Titanic and CC0 Pima Indians
Diabetes), R packages (MIT or GPL-2 depending on the source). Per-dataset
licence and citation are in `real/sources.md`. Synthetic data is
generated from random seeds and is released under MIT.

When citing this package, cite both the software and the original sources of
the real datasets — see CITATION.cff at the repo root and `real/sources.md`
for the BibTeX entries.
