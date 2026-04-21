# Fontes dos Datasets Reais

**Data:** 2026-04-20

Documentação completa das fontes, links e justificativas de classificação para todos os 29 datasets reais utilizados no benchmark.

---

## Datasets via `preparar_dados_reais.py` (arquivos locais)

### MCAR

| Dataset | Fonte Original | Link | Referência |
|---------|---------------|------|------------|
| oceanbuoys_humidity | TAO/TRITON Buoy Array (NOAA/PMEL) | [R naniar package](https://cran.r-project.org/package=naniar) / [PMEL TAO](https://www.pmel.noaa.gov/tao/) | Tierney & Cook (2023). naniar R package. Dados de boias oceanográficas do Pacífico tropical. |
| oceanbuoys_airtemp | TAO/TRITON Buoy Array (NOAA/PMEL) | [R naniar package](https://cran.r-project.org/package=naniar) / [PMEL TAO](https://www.pmel.noaa.gov/tao/) | Mesma fonte. Missing por falha de sensor/transmissão (MCAR). |

### MAR

| Dataset | Fonte Original | Link | Referência |
|---------|---------------|------|------------|
| airquality_ozone | New York Air Quality (R datasets) | [R datasets::airquality](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/airquality.html) / [EPA](https://www.epa.gov/) | Chambers et al. (1983). Dados meteorológicos NYC maio-setembro 1973. Missing correlaciona com Wind e Temp (MAR). |
| mammographic_density | UCI Mammographic Mass | [UCI Repository](https://archive.ics.uci.edu/dataset/161/mammographic+mass) / [OpenML](https://www.openml.org/d/310) | Elter et al. (2007). BI-RADS attributes of mammographic masses. Density missing depende de BIRADS e Age (MAR). |

### MNAR

| Dataset | Fonte Original | Link | Referência |
|---------|---------------|------|------------|
| pima_insulin | Pima Indians Diabetes (UCI/NIDDK) | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / [UCI](https://archive.ics.uci.edu/dataset/34/diabetes) | Smith et al. (1988). NIDDK. Insulina: zeros = missing (biologicamente impossível). MNAR: teste não realizado quando resultado esperado é normal. |
| mroz_wages | Mroz Female Labor Supply | [Wooldridge Datasets](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/) / [R Ecdat package](https://cran.r-project.org/package=Ecdat) | Mroz (1987). Econometrica. Salário (lwg) missing para mulheres fora da força de trabalho — MNAR clássico em econometria (Heckman selection). |

---

## Datasets via `expandir_dados_reais.py` (OpenML / URLs)

### MCAR

| Dataset | OpenML Name | OpenML ID | Link OpenML | Link UCI | X0 (variável) | Referência Original |
|---------|-------------|-----------|-------------|----------|----------------|---------------------|
| breastcancer_barenuclei | breast-cancer-wisconsin | — | — | [UCI](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) | Bare Nuclei | Mangasarian & Wolberg (1990). 16 missing em 699 (2.3%), falha de registro clínico. |
| cylinderbands_bladepressure | cylinder-bands | [6332](https://www.openml.org/d/6332) | [OpenML](https://www.openml.org/d/6332) | [UCI](https://archive.ics.uci.edu/dataset/32/cylinder+bands) | blade_pressure | Aha (1990s). Dados de manufatura de impressão rotogravura. Missing por falha de sensor. |
| cylinderbands_esavoltage | cylinder-bands | [6332](https://www.openml.org/d/6332) | [OpenML](https://www.openml.org/d/6332) | [UCI](https://archive.ics.uci.edu/dataset/32/cylinder+bands) | ESA_Voltage | Mesma fonte. Missing por falha de sensor de voltagem. |
| hypothyroid_t4u | hypothyroid | [57](https://www.openml.org/d/57) | [OpenML](https://www.openml.org/d/57) | [UCI](https://archive.ics.uci.edu/dataset/102/thyroid+disease) | T4U | Quinlan (1987). Garavan Institute. Exame tireoidiano não solicitado rotineiramente. |
| autompg_horsepower | autoMpg | [196](https://www.openml.org/d/196) | [OpenML](https://www.openml.org/d/196) | [UCI](https://archive.ics.uci.edu/dataset/9/auto+mpg) | horsepower | Quinlan (1993). Auto MPG. 6 valores missing em 398 (1.5%). |
| hepatitis_alkphosphate | hepatitis | [55](https://www.openml.org/d/55) | [OpenML](https://www.openml.org/d/55) | [UCI](https://archive.ics.uci.edu/dataset/46/hepatitis) | ALK_PHOSPHATE | Dua & Graff (2017). 29/155 missing (18.7%). Teste hepático de rotina omitido por logística laboratorial. Little's p=0.44. |
| hepatitis_albumin | hepatitis | [55](https://www.openml.org/d/55) | [OpenML](https://www.openml.org/d/55) | [UCI](https://archive.ics.uci.edu/dataset/46/hepatitis) | ALBUMIN | Mesma fonte. 16/155 missing (10.3%). Teste proteico de rotina. Little's p=0.68. |
| creditapproval_a14 | credit-approval | [29](https://www.openml.org/d/29) | [OpenML](https://www.openml.org/d/29) | [UCI](https://archive.ics.uci.edu/dataset/27/credit+approval) | A14 | Quinlan (1987). 13/690 missing (1.9%). Campo contínuo anonimizado de aplicação de crédito. Little's p=0.70. |
| echomonths_epss | echoMonths | [222](https://www.openml.org/d/222) | [OpenML](https://www.openml.org/d/222) | [UCI](https://archive.ics.uci.edu/dataset/38/echocardiogram) | epss | Salzberg (1988). 14/130 missing (10.8%). EPSS: medida ecocardiográfica, missing por janela acústica insuficiente. Little's p=0.65. |

### MAR

| Dataset | OpenML Name | OpenML ID | Link OpenML | Link UCI | X0 (variável) | Referência Original |
|---------|-------------|-----------|-------------|----------|----------------|---------------------|
| sick_t3 | sick | [38](https://www.openml.org/d/38) | [OpenML](https://www.openml.org/d/38) | [UCI](https://archive.ics.uci.edu/dataset/102/thyroid+disease) | T3 | Quinlan (1987). Garavan Institute. Teste T3 solicitado com base em outros sintomas clínicos (MAR). |
| sick_tsh | sick | [38](https://www.openml.org/d/38) | [OpenML](https://www.openml.org/d/38) | [UCI](https://archive.ics.uci.edu/dataset/102/thyroid+disease) | TSH | Mesma fonte. TSH solicitado baseado em outros exames. |
| kidney_hemo | chronic-kidney-disease | [42972](https://www.openml.org/d/42972) | [OpenML](https://www.openml.org/d/42972) | [UCI](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | hemo | Rubini (2015). Hemoglobina: exame depende da severidade do caso (MAR). |
| hearth_chol | heart-h (Hungarian) | [51](https://www.openml.org/d/51) | [OpenML](https://www.openml.org/d/51) | [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) | chol | Detrano et al. (1989). Cleveland/Hungarian heart disease. Colesterol não medido depende de fatores clínicos. |
| titanic_age | Titanic | — | — | — | Age | British Board of Trade (1912). Passageiros do RMS Titanic. Idade missing correlaciona com classe (Pclass) — MAR. |
| titanic_age_v2 | Titanic (Kaggle) | — | [Kaggle](https://www.kaggle.com/c/titanic) / [GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) | — | Age | Mesma fonte, versão completa do Kaggle. |
| colic_resprate | colic | [25](https://www.openml.org/d/25) | [OpenML](https://www.openml.org/d/25) | [UCI](https://archive.ics.uci.edu/dataset/47/horse+colic) | respiratory_rate | McLeish & Cecile (1989). Frequência respiratória de cavalos: exame depende da severidade. |

### MNAR

| Dataset | OpenML Name | OpenML ID | Link OpenML | Link UCI | X0 (variável) | Referência Original |
|---------|-------------|-----------|-------------|----------|----------------|---------------------|
| adult_capitalgain | adult | — | — | [UCI](https://archive.ics.uci.edu/dataset/2/adult) | capital-gain | Becker & Kohavi (1996). Census Income. Capital gain=0 para quem não investe (MNAR: valor determina ausência). |
| colic_refluxph | colic | [25](https://www.openml.org/d/25) | [OpenML](https://www.openml.org/d/25) | [UCI](https://archive.ics.uci.edu/dataset/47/horse+colic) | nasogastric_reflux_PH | McLeish & Cecile (1989). pH do refluxo: medição impossível em extremos. |
| cylinderbands_varnishpct | cylinder-bands | [6332](https://www.openml.org/d/6332) | [OpenML](https://www.openml.org/d/6332) | [UCI](https://archive.ics.uci.edu/dataset/32/cylinder+bands) | varnish_pct | Aha (1990s). Percentual de verniz: qualidade dependente. |
| kidney_pot | chronic-kidney-disease | [42972](https://www.openml.org/d/42972) | [OpenML](https://www.openml.org/d/42972) | [UCI](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | pot | Rubini (2015). Potássio: valores extremos não reportados (MNAR). |
| kidney_sod | chronic-kidney-disease | [42972](https://www.openml.org/d/42972) | [OpenML](https://www.openml.org/d/42972) | [UCI](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | sod | Rubini (2015). Sódio: mesma lógica do potássio. |
| mroz_wages | Mroz (1987) | — | — | — | lwg | (ver seção preparar_dados_reais.py acima) |
| pima_insulin | diabetes (Pima) | — | — | — | Insulin | (ver seção preparar_dados_reais.py acima) |
| hepatitis_protime | hepatitis | [55](https://www.openml.org/d/55) | [OpenML](https://www.openml.org/d/55) | [UCI](https://archive.ics.uci.edu/dataset/46/hepatitis) | PROTIME | Dua & Graff (2017). 67/155 missing (43.2%). Tempo de protrombina: solicitado apenas quando coagulação anormal é suspeita (MNAR por domínio). |
| pima_skinthickness | diabetes (Pima) | — | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / [GitHub](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) | [UCI](https://archive.ics.uci.edu/dataset/34/diabetes) | SkinThickness | Smith et al. (1988). NIDDK. 227/768 zeros (29.6%) → NaN. Compasso padrão (≤45mm) não mede pacientes obesas. MNAR documentado em: Pham et al. (2015) PMC4594849; Bray et al. (1978). |

---

## Resumo de Fontes

| Repositório | Datasets | IDs |
|-------------|----------|-----|
| **UCI Machine Learning Repository** | 14 | hepatitis, credit-approval, autoMpg, cylinder-bands, hypothyroid, sick, chronic-kidney-disease, heart-h, colic, echoMonths, adult, mammographic-mass, diabetes/pima |
| **OpenML** (mirror UCI) | 12 | IDs: 25, 29, 38, 51, 55, 57, 196, 222, 6332, 42972 |
| **Kaggle** | 2 | Titanic, Pima Indians Diabetes |
| **R packages** | 3 | naniar (oceanbuoys), datasets (airquality), Ecdat (mroz) |
| **GitHub (URL direta)** | 2 | Titanic v2, Pima SkinThickness |

## Protocolo de Acesso

Todos os datasets são acessados programaticamente via:
1. `sklearn.datasets.fetch_openml()` — para datasets do OpenML (12 datasets)
2. `pandas.read_csv(url)` — para URLs diretas do GitHub/Kaggle (2 datasets)
3. Arquivos CSV locais em `Dataset/real_data/{MCAR,MAR,MNAR}/` — para datasets do R (5 datasets)

Scripts: `preparar_dados_reais.py` (5 datasets locais) e `expandir_dados_reais.py` (24 datasets via API/URL).
