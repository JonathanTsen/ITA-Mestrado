# Real datasets — sources, licences and labelling rationale

Each row below corresponds to one mechanism-labelled column from a source
dataset. There are 29 columns in total drawn from 18 source datasets,
balanced across MCAR (9), MAR (11) and MNAR (9).

When citing missdetect please also cite the original source of any real
dataset you use. BibTeX entries for source datasets are at the bottom.

## Summary

| Source | Datasets | Columns | Access path | Licence |
|:--|:-:|:-:|:--|:--|
| UCI ML Repository | 8 | 14 | `sklearn.datasets.fetch_openml()` mirror | CC BY 4.0 |
| OpenML | mirror of UCI | 12 | `fetch_openml()` direct | CC BY 4.0 |
| Kaggle | 2 (Titanic, Pima) | 4 | `pandas.read_csv(URL)` | Public domain / CC0 |
| R packages | 3 (`naniar`, `datasets`, `Ecdat`) | 3 | bundled CSVs in `raw/` | MIT / GPL-2 |

## MCAR (9 columns)

| Column | Source dataset | Mechanism rationale |
|:--|:--|:--|
| `oceanbuoys_humidity` | `naniar::oceanbuoys` (NOAA TAO/TRITON) | Sensor / transmission failure independent of value |
| `oceanbuoys_airtemp` | `naniar::oceanbuoys` | Same — sensor failure |
| `breastcancer_barenuclei` | UCI Breast Cancer Wisconsin Original | 16 missing in 699 (2.3%); clerical record gap |
| `cylinderbands_bladepressure` | OpenML 6332 / UCI Cylinder Bands | Sensor failure during printing |
| `cylinderbands_esavoltage` | OpenML 6332 / UCI Cylinder Bands | Voltage sensor failure |
| `hypothyroid_t4u` | OpenML 57 / UCI Thyroid | T4U not routinely ordered |
| `autompg_horsepower` | OpenML 196 / UCI Auto MPG | 6 missing in 398 (1.5%); no apparent pattern |
| `hepatitis_alkphosphate` | OpenML 55 / UCI Hepatitis | 29/155 missing; routine test omitted, Little's p=0.44 |
| `hepatitis_albumin` | OpenML 55 / UCI Hepatitis | 16/155; routine, Little's p=0.68 |
| `creditapproval_a14` | OpenML 29 / UCI Credit Approval | 13/690; anonymised continuous field |
| `echomonths_epss` | OpenML 222 / UCI Echocardiogram | 14/130; insufficient acoustic window |

## MAR (11 columns)

| Column | Source dataset | Mechanism rationale |
|:--|:--|:--|
| `airquality_ozone` | R `datasets::airquality` (NYC 1973) | Missing correlates with Wind and Temp |
| `mammographic_density` | UCI Mammographic Mass | Density depends on BIRADS and Age |
| `sick_t3` | OpenML 38 / UCI Thyroid (sick) | T3 ordered based on other clinical signs |
| `sick_tsh` | OpenML 38 / UCI Thyroid (sick) | TSH ordered based on other tests |
| `kidney_hemo` | OpenML 42972 / UCI Chronic Kidney Disease | Haemoglobin depends on case severity |
| `hearth_chol` | OpenML 51 / UCI Heart-h (Hungarian) | Cholesterol omission depends on observed clinical state |
| `titanic_age` | British Board of Trade 1912 / R datasets | Age missing correlates with passenger class |
| `titanic_age_v2` | Kaggle Titanic | Same source, completer version |
| `colic_resprate` | OpenML 25 / UCI Horse Colic | Respiratory rate measurement depends on severity |

## MNAR (9 columns)

| Column | Source dataset | Mechanism rationale |
|:--|:--|:--|
| `pima_insulin` | UCI Pima Indians Diabetes | Zero coded as missing; test only when normal expected |
| `pima_skinthickness` | Kaggle Pima Indians Diabetes | Caliper has 45mm cap; obese patients out of range |
| `mroz_wages` | R `Ecdat::Mroz` (econometric classic) | Wage missing for women out of labour force — Heckman selection MNAR |
| `adult_capitalgain` | UCI Adult / Census Income | Capital gain = 0 for non-investors; value determines absence |
| `colic_refluxph` | OpenML 25 / UCI Horse Colic | pH unmeasurable at extremes |
| `cylinderbands_varnishpct` | OpenML 6332 / UCI Cylinder Bands | Quality-dependent measurement |
| `kidney_pot` | OpenML 42972 / UCI CKD | Extreme potassium values not reported |
| `kidney_sod` | OpenML 42972 / UCI CKD | Same logic for sodium |
| `hepatitis_protime` | OpenML 55 / UCI Hepatitis | Protime ordered only when coagulation suspected — domain MNAR |

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
```

For details on each access path (`fetch_openml` ID, GitHub URL, R package
version) see `src/missdetect/data_generation/expandir_dados_reais.py` —
the canonical source of truth for how each column is materialised.
