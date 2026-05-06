# Real datasets — sources, licences and labelling rationale

Each row below corresponds to one mechanism-labelled column from a source
dataset. There are 32 columns in total drawn from 21 source datasets:
MCAR (6), MAR (13) and MNAR (13).

> **Auditoria 2026-05-06:** 7 datasets com classificação duvidosa foram removidos
> (creditapproval_a14, echomonths_epss, autompg_horsepower, hearth_chol,
> kidney_hemo, colic_resprate, cylinderbands_varnishpct). 6 datasets foram
> reclassificados de MCAR para MAR após verificação pelo protocolo v2b e
> revisão de domínio. Ver `docs/archive/10_protocolo_validacao_v2/11_AUDITORIA_BENCHMARK.md`.

When citing missdetect please also cite the original source of any real
dataset you use. BibTeX entries for source datasets are at the bottom.

## Per-column v2 diagnoses

The mechanism column in each table below is the **literature/domain
label** — the rationale a domain expert would give for that column being
MCAR/MAR/MNAR. This is the label used as ground truth for training and
evaluation. The v2 protocol (`src/missdetect/validar_rotulos_v2.py`)
produces **independent diagnoses** that are stored alongside, not as
replacements:

- Run `uv run python -m missdetect.calibrar_protocolo` once to produce
  `data/calibration.json` and `data/calibration_scores.npz`.
- Run `uv run python -m missdetect.validar_rotulos_v2 --data real
  --calibration data/calibration.json --bayes-scores data/calibration_scores.npz`
  to obtain per-column v2 diagnoses in
  `results/v2_protocol/real/validacao_rotulos_v2/validacao_v2.csv`.

Disagreement between the literature label and the v2 diagnosis is
expected for MNAR columns — MNAR is theoretically non-identifiable from
observed data alone (Molenberghs et al. 2008). v2 disagreement is a
**flag for sensitivity analysis**, not an automatic re-label.

## Summary

| Source | Datasets | Columns | Access path | Licence |
|:--|:-:|:-:|:--|:--|
| UCI ML Repository | 8 | 14 | `sklearn.datasets.fetch_openml()` mirror | CC BY 4.0 |
| OpenML | mirror of UCI | 12 | `fetch_openml()` direct | CC BY 4.0 |
| Kaggle | 2 (Titanic, Pima) | 4 | `pandas.read_csv(URL)` | Public domain / CC0 |
| R packages | 3 (`naniar`, `datasets`, `Ecdat`) | 3 | bundled CSVs in `raw/` | MIT / GPL-2 |

## MCAR (6 columns)

| Column | Source dataset | Mechanism rationale |
|:--|:--|:--|
| `hepatitis_alkphosphate` | OpenML 55 / UCI Hepatitis | 29/155 missing; routine test omitted, Little's p=0.44 |
| `hepatitis_albumin` | OpenML 55 / UCI Hepatitis | 16/155; routine, Little's p=0.68 |
| `boys_hc` | `mice::boys` (Fourth Dutch Growth Study) | 46/748 (6.1%); random scheduling gaps in clinic visits. Van Buuren (2018) FIMD Ch. 9 |
| `boys_hgt` | `mice::boys` (Fourth Dutch Growth Study) | 20/748 (2.7%); same — random visit scheduling |
| `brandsma_lpr` | `mice::brandsma` (Dutch primary education) | 320/4106 (7.8%); student absent on test day. Correlation mask~ses: p=0.72, mask~iqv: p=0.15 |
| `brandsma_apr` | `mice::brandsma` (Dutch primary education) | 309/4106 (7.5%); student absent on test day. Correlation mask~ses: p=0.75 |

## MAR (13 columns)

| Column | Source dataset | Mechanism rationale |
|:--|:--|:--|
| `airquality_ozone` | R `datasets::airquality` (NYC 1973) | Missing correlates with Wind and Temp |
| `mammographic_density` | UCI Mammographic Mass | Density depends on BIRADS and Age |
| `sick_t3` | OpenML 38 / UCI Thyroid (sick) | T3 ordered based on other clinical signs |
| `sick_tsh` | OpenML 38 / UCI Thyroid (sick) | TSH ordered based on other tests |
| `titanic_age` | British Board of Trade 1912 / R datasets | Age missing correlates with passenger class |
| `titanic_age_v2` | Kaggle Titanic | Same source, completer version |
| `oceanbuoys_humidity` | `naniar::oceanbuoys` (NOAA TAO/TRITON) | Sensor failure correlates with environmental conditions; v2b confirms MAR (conf=1.0) |
| `oceanbuoys_airtemp` | `naniar::oceanbuoys` | Same — sensor failure correlated with conditions |
| `hypothyroid_t4u` | OpenML 57 / UCI Thyroid | T4U ordered based on TSH/TT4/FTI results; v2b confirms MAR (conf=1.0) |
| `breastcancer_barenuclei` | UCI Breast Cancer Wisconsin Original | 16/699 missing; missingness predictable by other features; v2b confirms MAR (conf=1.0) |
| `cylinderbands_bladepressure` | OpenML 6332 / UCI Cylinder Bands | Sensor failure correlates with printing conditions (press_speed, ink_temperature); v2b confirms MAR (conf=1.0) |
| `cylinderbands_esavoltage` | OpenML 6332 / UCI Cylinder Bands | Same — voltage sensor failure correlates with conditions; v2b confirms MAR (conf=0.77) |
| `support2_pafi` | SUPPORT2 (Knaus et al. 1995) | PaO2/FiO2 requer ABG invasivo; forte correlação com sinais observados: corr(mask, hrt)=−0.19, corr(mask, temp)=−0.18 p<0.001 → predominantemente MAR |

## MNAR (13 columns)

| Column | Source dataset | Mechanism rationale |
|:--|:--|:--|
| `pima_insulin` | UCI Pima Indians Diabetes | Zero coded as missing; test only when normal expected |
| `pima_skinthickness` | Kaggle Pima Indians Diabetes | Caliper has 45mm cap; obese patients out of range |
| `mroz_wages` | R `Ecdat::Mroz` (econometric classic) | Wage missing for women out of labour force — Heckman selection MNAR |
| `adult_capitalgain` | UCI Adult / Census Income | Capital gain = 0 for non-investors; value determines absence |
| `colic_refluxph` | OpenML 25 / UCI Horse Colic | pH unmeasurable at extremes |
| `kidney_pot` | OpenML 42972 / UCI CKD | Extreme potassium values not reported |
| `kidney_sod` | OpenML 42972 / UCI CKD | Same logic for sodium |
| `hepatitis_protime` | OpenML 55 / UCI Hepatitis | Protime ordered only when coagulation suspected — domain MNAR |
| `nhanes_cadmium` | NHANES 2017-18 (CDC PBCD_J) | Blood cadmium below LOD (0.100 μg/L) → **left-censored MNAR** (valor < limiar físico). Tellez-Plaza et al. (2012) EHP |
| `nhanes_mercury` | NHANES 2017-18 (CDC PBCD_J) | Blood mercury below LOD (0.28 μg/L) → **left-censored MNAR** |
| `nhanes_cotinine` | NHANES 2017-18 (CDC COT_J) | Serum cotinine below LOD (0.015 ng/mL) → **left-censored MNAR**. Bernert et al. (2011) |
| `support2_albumin` | SUPPORT2 (Knaus et al. 1995) | **MNAR misto**: albumina não-rotineira em UTI; corr(mask, hrt)=−0.07. 37% missing |
| `support2_bilirubin` | SUPPORT2 (Knaus et al. 1995) | **MNAR misto**: bilirrubina ordenada quando disfunção hepática esperada. corr(mask, age)=0.08. 28.6% missing |

> **Nota sobre subtipos de MNAR:** os datasets NHANES são **MNAR puro** (left-censoring por LOD — mecanismo físico indiscutível). Os datasets SUPPORT2 e os originais de test-ordering (hepatitis_protime, kidney_pot/sod) são **MNAR misto** — a decisão de medir depende tanto de sinais observados (MAR) quanto do valor esperado (MNAR).

## BibTeX for source datasets

```bibtex
@misc{uci-mlr,
  author = {Dua, D. and Graff, C.},
  title  = {UCI Machine Learning Repository},
  year   = {2017},
  url    = {http://archive.ics.uci.edu/ml}
}

@article{openml,
  author = {Vanschoren, J. and van Rijn, J. N. and Bischl, B. and Torgo, L.},
  title  = {OpenML: networked science in machine learning},
  journal= {SIGKDD Explorations},
  volume = {15},
  number = {2},
  year   = {2014},
  pages  = {49--60}
}

@misc{naniar,
  author = {Tierney, N. and Cook, D.},
  title  = {naniar: Data Structures, Summaries, and Visualisations for
            Missing Data},
  year   = {2023},
  url    = {https://CRAN.R-project.org/package=naniar}
}

@article{mroz1987,
  author  = {Mroz, T. A.},
  title   = {The Sensitivity of an Empirical Model of Married Women's Hours
             of Work to Economic and Statistical Assumptions},
  journal = {Econometrica},
  volume  = {55},
  number  = {4},
  year    = {1987},
  pages   = {765--799}
}

@article{smith1988pima,
  author  = {Smith, J. W. and Everhart, J. E. and Dickson, W. C. and
             Knowler, W. C. and Johannes, R. S.},
  title   = {Using the {ADAP} learning algorithm to forecast the onset of
             diabetes mellitus},
  journal = {Proc. of the Annual Symposium on Computer Application in
             Medical Care},
  year    = {1988}
}

@article{detrano1989heart,
  author  = {Detrano, R. and Janosi, A. and Steinbrunn, W. and others},
  title   = {International application of a new probability algorithm for
             the diagnosis of coronary artery disease},
  journal = {American Journal of Cardiology},
  volume  = {64},
  year    = {1989},
  pages   = {304--310}
}

@inproceedings{elter2007,
  author    = {Elter, M. and Schulz-Wendtland, R. and Wittenberg, T.},
  title     = {The prediction of breast cancer biopsy outcomes using two CAD
               approaches},
  booktitle = {Medical Physics},
  year      = {2007}
}

@book{vanbuuren2018,
  author    = {Van Buuren, S.},
  title     = {Flexible Imputation of Missing Data},
  edition   = {2nd},
  publisher = {CRC Press},
  year      = {2018},
  url       = {https://stefvanbuuren.name/fimd/}
}

@article{brandsma1989,
  author  = {Brandsma, H. P. and Knuver, J. W. M.},
  title   = {Effects of school and classroom characteristics on pupil progress
             in language and arithmetic},
  journal = {International Journal of Educational Research},
  volume  = {13},
  year    = {1989},
  pages   = {777--788}
}

@article{fredriks2000,
  author  = {Fredriks, A. M. and van Buuren, S. and Burgmeijer, R. J. F. and others},
  title   = {Continuing positive secular growth change in the {Netherlands} 1955--1997},
  journal = {Pediatric Research},
  volume  = {47},
  year    = {2000},
  pages   = {316--323}
}

@article{tellezplaza2012,
  author  = {Tellez-Plaza, M. and Navas-Acien, A. and Menke, A. and Crainiceanu, C. M.
             and Pastor-Barriuso, R. and Guallar, E.},
  title   = {Cadmium exposure and all-cause and cardiovascular mortality in the
             {U.S.} general population},
  journal = {Environmental Health Perspectives},
  volume  = {120},
  number  = {7},
  year    = {2012},
  pages   = {1017--1022}
}

@article{bernert2011,
  author  = {Bernert, J. T. and Gordon, S. M. and Jain, R. B. and others},
  title   = {Toward improved statistical methods for analyzing cotinine-biomarker
             health association data},
  journal = {Tobacco Induced Diseases},
  volume  = {9},
  number  = {11},
  year    = {2011}
}

@article{knaus1995,
  author  = {Knaus, W. A. and Harrell, F. E. and Lynn, J. and others},
  title   = {The {SUPPORT} prognostic model: objective estimates of survival
             for seriously ill hospitalized adults},
  journal = {Annals of Internal Medicine},
  volume  = {122},
  number  = {3},
  year    = {1995},
  pages   = {191--203}
}

@misc{nhanes2018,
  author = {{Centers for Disease Control and Prevention (CDC)}},
  title  = {National Health and Nutrition Examination Survey ({NHANES}) 2017--2018},
  year   = {2020},
  url    = {https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017}
}
```

For details on each access path (`fetch_openml` ID, GitHub URL, R package
version) see `src/missdetect/data_generation/expandir_dados_reais.py` —
the canonical source of truth for how each column is materialised.
