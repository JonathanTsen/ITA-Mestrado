# Estratégia de Validação com Dados Reais

## 1. Objetivo

Validar se o classificador de mecanismos de missing data (treinado em dados sintéticos) é capaz de classificar corretamente datasets reais onde o mecanismo de missing é conhecido por **conhecimento de domínio** (não por rótulo explícito).

---

## 2. Datasets Coletados

### 2.1 MCAR — Oceanbuoys / TAO (Tropical Atmosphere Ocean)

| Atributo | Valor |
|----------|-------|
| **Fonte** | NOAA TAO Project, via pacote R `naniar` |
| **Observações** | 736 |
| **Variáveis originais** | 8 (year, latitude, longitude, sea.surface.temp, air.temp, humidity, uwind, vwind) |
| **Variável com missing** | `humidity` (93 NaN, 12.6%) e `air.temp` (81 NaN, 11.0%) |
| **Por que MCAR?** | Missing causado por falhas aleatórias de equipamento em boias oceânicas. A falha do sensor não depende do valor da medição nem de outras variáveis — é puramente mecânica/eletrônica. |
| **Referência** | Tierney, N.J. & Cook, D.H. (2023). "Expanding Tidy Data Principles to Facilitate Missing Data Exploration." *JSS* 105(7). |
| **Limitação** | Taxa de missing (11-12%) está no limite superior do nosso range sintético (1-10%). |

### 2.2 MAR — Airquality (New York, 1973)

| Atributo | Valor |
|----------|-------|
| **Fonte** | NY State Dept. of Conservation / National Weather Service |
| **Observações** | 153 |
| **Variáveis originais** | 6 (Ozone, Solar.R, Wind, Temp, Month, Day) |
| **Variável com missing** | `Ozone` (37 NaN, 24.2%) |
| **Por que MAR?** | Medições de ozônio faltam mais em dias frios e ventosos (relacionado a Wind e Temp, que são observados). Em dias com baixa temperatura, a formação fotoquímica de ozônio é menor e os equipamentos da época eram menos confiáveis em condições adversas. |
| **Referência** | Chambers, J.M. et al. (1983). *Graphical Methods for Data Analysis*. |
| **Limitação** | Taxa de missing alta (24.2%) e poucas observações (153). Pode não ser "puro" MAR — existe debate se há componente MNAR (ozônio faltante quando nível é muito baixo). |

### 2.3 MAR — Mammographic Mass (UCI)

| Atributo | Valor |
|----------|-------|
| **Fonte** | UCI Machine Learning Repository (ID 161) |
| **Observações** | 886 (após limpeza) |
| **Variáveis originais** | 6 (BI-RADS, Age, Shape, Margin, Density, Severity) |
| **Variável com missing** | `Density` (56 NaN, 6.3%) |
| **Por que MAR?** | Radiologistas tendem a omitir a avaliação de densidade mamográfica quando a classificação BI-RADS e a idade já indicam claramente benignidade ou malignidade. A decisão de não registrar depende de variáveis observadas (BI-RADS, Age), não do próprio valor de Density. |
| **Referência** | Elter, M. et al. (2007). "The prediction of breast cancer biopsy outcomes using two CAD approaches." |
| **Limitação** | Variáveis são discretas/ordinais (1-5), não contínuas como nossos dados sintéticos. |

### 2.4 MNAR — Pima Indians Diabetes (Insulin)

| Atributo | Valor |
|----------|-------|
| **Fonte** | National Institute of Diabetes, via UCI Repository |
| **Observações** | 768 |
| **Variáveis originais** | 9 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age, Outcome) |
| **Variável com missing** | `Insulin` (374 zeros→NaN, 48.7%) |
| **Por que MNAR?** | Zeros no campo Insulin representam exames não realizados. Médicos tendem a não solicitar o teste de insulina quando não suspeitam de diabetes — e essa suspeita está correlacionada com o próprio nível de insulina (pacientes com insulina normal são menos propensos a ter o exame solicitado). |
| **Referência** | Smith, J.W. et al. (1988). "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus." |
| **Limitação** | Taxa de missing muito alta (48.7%) — muito acima do range sintético (1-10%). Os zeros como código de missing é uma convenção, não um missing "natural". |

### 2.5 MNAR — Mroz Wages (Labor Force Participation)

| Atributo | Valor |
|----------|-------|
| **Fonte** | Panel Study of Income Dynamics (PSID), 1975 |
| **Observações** | 753 |
| **Variáveis originais** | 8 (lfp, k5, k618, age, wc, hc, lwg, inc) |
| **Variável com missing** | `lwg` (log wage) — 325 NaN (43.2%) para mulheres que não participam da força de trabalho |
| **Por que MNAR?** | Este é o exemplo canônico de MNAR na econometria (modelo de seleção de Heckman). Salários são observados apenas para mulheres que trabalham. A decisão de trabalhar depende do próprio salário potencial — mulheres com baixo salário potencial tendem a não trabalhar, tornando o missing dependente do valor não observado. |
| **Referência** | Mroz, T.A. (1987). "The sensitivity of an empirical model of married women's hours of work." *Econometrica* 55, 765-799. Heckman, J.J. (1979). "Sample selection bias as a specification error." *Econometrica* 47(1), 153-161. |
| **Limitação** | Taxa de missing alta (43.2%). O mecanismo MNAR é forte e bem documentado, mas o missing não é "sutil" — é uma seleção binária (trabalha/não trabalha). |

---

## 3. Dificuldades e Desafios

### 3.1 Ausência de Ground Truth

**O problema fundamental:** Em dados reais, **nunca sabemos com certeza** qual é o mecanismo de missing. O que temos são argumentos de domínio que sugerem fortemente um mecanismo. Isso significa que:

- Se o classificador errar, pode ser porque: (a) o classificador falhou, ou (b) o mecanismo não é exatamente o que assumimos.
- Não podemos calcular "acurácia" no sentido estrito — apenas avaliar **consistência** entre a predição e o conhecimento de domínio.

### 3.2 Taxas de Missing Incompatíveis

| Dataset | Missing Rate | Range Sintético |
|---------|-------------|-----------------|
| Oceanbuoys (humidity) | 12.6% | 1-10% |
| Airquality (Ozone) | 24.2% | 1-10% |
| Mammographic (Density) | 6.3% | 1-10% ✅ |
| Pima (Insulin) | 48.7% | 1-10% |
| Mroz (Wages) | 43.2% | 1-10% |

**Problema:** Apenas 1 dos 6 datasets tem taxa de missing dentro do range de treinamento. O classificador foi treinado com 1-10% de missing e será testado em datasets com até 48.7%.

**Estratégias de mitigação:**
1. **Subsampling do missing**: Para datasets com taxa alta, remover aleatoriamente alguns NaN (substituindo pelo valor original, se disponível, ou pela média) para trazer a taxa para ~5-10%.
2. **Retreinar com range expandido**: Gerar dados sintéticos com taxas de 1-50% e retreinar.
3. **Aceitar a limitação**: Reportar os resultados como estão e discutir a sensibilidade à taxa de missing.

### 3.3 Distribuições Diferentes

Os dados sintéticos são gerados como **Uniform[0,1]**, mas os dados reais têm distribuições variadas (normal, skewed, discreta, etc.). Mesmo após normalização min-max para [0,1], as distribuições internas são diferentes.

**Impacto:** Features como `X0_mean`, `X0_q25`, `X0_q50`, `X0_q75` terão valores que o modelo nunca viu no treinamento.

**Estratégias de mitigação:**
1. **Rank-transform**: Converter valores para ranks antes de calcular features (torna qualquer distribuição aproximadamente uniforme).
2. **Normalizar features, não dados**: Aplicar StandardScaler nas features extraídas, não nos dados brutos.
3. **Treinar com distribuições mistas**: Gerar dados sintéticos com Normal, Log-normal, Exponencial além de Uniform.

### 3.4 Poucas Amostras por Mecanismo

| Mecanismo | Datasets | Total no treino sintético |
|-----------|----------|--------------------------|
| MCAR | 2 | 1000 |
| MAR | 2 | 1000 |
| MNAR | 2 | 1000 |

**Problema:** Temos 2 datasets por mecanismo contra 1000 no treino. Não dá para calcular métricas robustas (acurácia, F1) com 2 amostras.

**Estratégias de mitigação:**
1. **Usar como teste qualitativo**: Apresentar como "estudo de caso" e não como "teste estatístico".
2. **Bootstrap**: Gerar múltiplas amostras de cada dataset real (subsampling com reposição) e classificar cada amostra.
3. **Abordagem semi-sintética** (ver seção 5): Usar `pyampute` para criar centenas de variantes com mecanismo controlado.

### 3.5 Variáveis Discretas vs Contínuas

Mammographic Mass tem variáveis ordinais (1-5). Mroz tem variáveis binárias (wc, hc). Os dados sintéticos são puramente contínuos.

**Impacto:** Features discriminativas (AUC, correlação, Mann-Whitney) podem se comportar diferentemente com dados discretos.

### 3.6 Ambiguidade dos Mecanismos

Na realidade, os mecanismos raramente são "puros":
- **Airquality**: Provavelmente MAR com componente MNAR (ozônio falta quando nível é baixo E quando tempo está ruim).
- **Pima Diabetes**: MNAR para insulina, mas a decisão médica também depende de glicose (variável observada) → componente MAR.
- **Oceanbuoys**: Principalmente MCAR, mas falhas de sensor podem ser mais comuns em condições extremas → componente MAR.

---

## 4. Estratégia de Validação Proposta

### Fase 1: Teste Direto (Baseline)

```
Dados reais processados → extract_features.py → modelo treinado → predição
```

1. Rodar `extract_features.py` nos 6 datasets reais processados.
2. Usar o modelo treinado nos dados sintéticos para predizer o mecanismo.
3. Comparar predição com o mecanismo esperado por domínio.
4. Reportar: predição, confiança (probabilidades), features mais influentes.

**Resultado esperado:** Acurácia parcial. Datasets com missing rate dentro do range (Mammographic) devem ter melhor resultado.

### Fase 2: Análise de Features

1. Extrair features dos datasets reais e compará-las com a distribuição das features dos dados sintéticos.
2. Identificar quais features ficam fora da distribuição de treinamento (OOD — out-of-distribution).
3. Isso ajuda a explicar erros de classificação e guiar melhorias.

### Fase 3: Validação Semi-Sintética (Mais Rigorosa)

```
Dataset completo real → pyampute (MCAR/MAR/MNAR controlado) → pipeline → acurácia real
```

1. Pegar datasets completos (Iris, Wine, Boston Housing, etc.) com ~5 variáveis.
2. Usar `pyampute` para injetar missing com mecanismo CONHECIDO em cada variável.
3. Gerar 100+ variantes por mecanismo.
4. Rodar o pipeline completo e calcular acurácia com ground truth.

**Vantagem:** Ground truth perfeito + distribuições reais (não uniformes).

### Fase 4: Análise de Sensibilidade

Testar como o classificador se comporta variando:
- Taxa de missing (1%, 5%, 10%, 20%, 50%)
- Distribuição dos dados (uniform, normal, skewed)
- Número de observações (100, 500, 1000)

---

## 5. Como Reportar no Artigo

### Seção sugerida: "Validation on Real-World Data"

> To assess the practical applicability of the trained classifier, we evaluated it on six real-world datasets with domain-knowledge-characterized missing data mechanisms. These datasets span three domains (environmental monitoring, medical imaging, labor economics) and exhibit missing rates from 6.3% to 48.7%.

**Tabela de resultados sugerida:**

| Dataset | Domain | Expected | Missing% | Predicted | Confidence | Match |
|---------|--------|----------|----------|-----------|------------|-------|
| Oceanbuoys (humidity) | Environmental | MCAR | 12.6% | ? | ?% | ? |
| Oceanbuoys (air temp) | Environmental | MCAR | 11.0% | ? | ?% | ? |
| Airquality (ozone) | Environmental | MAR | 24.2% | ? | ?% | ? |
| Mammographic (density) | Medical | MAR | 6.3% | ? | ?% | ? |
| Pima (insulin) | Medical | MNAR | 48.7% | ? | ?% | ? |
| Mroz (wages) | Economics | MNAR | 43.2% | ? | ?% | ? |

**Tom adequado para o artigo:**
- Não afirmar que o classificador "acertou" ou "errou" — discutir consistência.
- Reconhecer limitações (taxa de missing, distribuição, ambiguidade).
- Usar como evidência complementar, não como prova definitiva.

---

## 6. Estrutura de Pastas

```
Dataset/real/
├── MCAR/
│   └── oceanbuoys_tao.csv           # Dataset original
├── MAR/
│   ├── airquality.csv               # Dataset original
│   └── mammographic_mass_raw.csv    # Dataset original
├── MNAR/
│   ├── mroz_wages.csv               # Dataset original
│   └── pima_diabetes_raw.csv        # Dataset original
├── processado/                       # Formato padronizado (X0-X4, tab-separated)
│   ├── MCAR/
│   │   ├── MCAR_oceanbuoys_humidity.txt
│   │   └── MCAR_oceanbuoys_airtemp.txt
│   ├── MAR/
│   │   ├── MAR_airquality_ozone.txt
│   │   └── MAR_mammographic_density.txt
│   └── MNAR/
│       ├── MNAR_pima_insulin.txt
│       └── MNAR_mroz_wages.txt
├── semi_sintetico/                   # (futuro) Datasets via pyampute
└── ESTRATEGIA_VALIDACAO_DADOS_REAIS.md  # Este documento
```

---

## 7. Próximos Passos

- [ ] Rodar Fase 1 (teste direto) com o modelo atual
- [ ] Analisar distribuição das features (Fase 2)
- [ ] Instalar `pyampute` e gerar datasets semi-sintéticos (Fase 3)
- [ ] Avaliar se é necessário retreinar com taxas de missing mais altas
- [ ] Considerar transformação rank para lidar com distribuições não-uniformes
- [ ] Escrever seção do artigo com resultados

---

## 8. Referências Bibliográficas

Referências completas dos datasets e trabalhos utilizados, em formato textual e BibTeX para inclusão direta no artigo LaTeX.

### 8.1 Referências por Dataset

#### Oceanbuoys / TAO (MCAR)

**Dados originais:**
- McPhaden, M.J. et al. (1998). "The Tropical Ocean-Global Atmosphere observing system: A decade of progress." *Journal of Geophysical Research: Oceans*, 103(C7), 14169–14240. DOI: 10.1029/97JC02906

**Pacote R (naniar) que disponibiliza o dataset:**
- Tierney, N.J. & Cook, D.H. (2023). "Expanding Tidy Data Principles to Facilitate Missing Data Exploration, Visualization and Assessment of Imputations." *Journal of Statistical Software*, 105(7), 1–31. DOI: 10.18637/jss.v105.i07

#### Airquality (MAR)

**Dados originais:**
- Chambers, J.M., Cleveland, W.S., Kleiner, B. & Tukey, P.A. (1983). *Graphical Methods for Data Analysis*. Wadsworth & Brooks/Cole, Pacific Grove, CA.

**Disponibilização como dataset R:**
- R Core Team (2024). *R: A Language and Environment for Statistical Computing*. R Foundation for Statistical Computing, Vienna, Austria. URL: https://www.R-project.org/

#### Mammographic Mass (MAR)

**Artigo introdutório do dataset:**
- Elter, M., Schulz-Wendtland, R. & Wittenberg, T. (2007). "The prediction of breast cancer biopsy outcomes using two CAD approaches that both emphasize an intelligible decision process." *Medical Physics*, 34(11), 4164–4172. DOI: 10.1118/1.2786864

**Repositório UCI:**
- Elter, M. (2007). Mammographic Mass [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C53K6Z

#### Pima Indians Diabetes (MNAR)

**Artigo introdutório do dataset:**
- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C. & Johannes, R.S. (1988). "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus." In *Proceedings of the Annual Symposium on Computer Application in Medical Care*, pp. 261–265. American Medical Informatics Association.

**Repositório UCI:**
- National Institute of Diabetes and Digestive and Kidney Diseases (1990). Pima Indians Diabetes Database [Dataset]. UCI Machine Learning Repository. URL: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

#### Mroz Wages (MNAR)

**Artigo original dos dados:**
- Mroz, T.A. (1987). "The sensitivity of an empirical model of married women's hours of work to economic and statistical assumptions." *Econometrica*, 55(4), 765–799. DOI: 10.2307/1911029

**Modelo teórico de seleção (justificativa MNAR):**
- Heckman, J.J. (1979). "Sample selection bias as a specification error." *Econometrica*, 47(1), 153–161. DOI: 10.2307/1912352

### 8.2 Referências Metodológicas (Missing Data)

**Taxonomia MCAR/MAR/MNAR:**
- Rubin, D.B. (1976). "Inference and missing data." *Biometrika*, 63(3), 581–592. DOI: 10.1093/biomet/63.3.581

- Little, R.J.A. & Rubin, D.B. (2002). *Statistical Analysis with Missing Data*. 2nd ed. Wiley-Interscience, Hoboken, NJ. ISBN: 978-0-471-18386-0

**Ferramenta pyampute (validação semi-sintética):**
- Schouten, R.M., Lugtig, P. & Vink, G. (2018). "Generating missing values for simulation purposes: a multivariate amputation procedure." *Journal of Statistical Computation and Simulation*, 88(15), 2909–2930. DOI: 10.1080/00949655.2018.1491577

**Imputação múltipla (pacote mice):**
- van Buuren, S. & Groothuis-Oudshoorn, K. (2011). "mice: Multivariate Imputation by Chained Equations in R." *Journal of Statistical Software*, 45(3), 1–67. DOI: 10.18637/jss.v045.i03

**Benchmark de imputação (Jenga):**
- Jäger, S., Allhorn, A. & Bießmann, F. (2021). "A Benchmark for Data Imputation Methods." *Frontiers in Big Data*, 4, 693674. DOI: 10.3389/fdata.2021.693674

### 8.3 Entradas BibTeX (copiar para o .bib do artigo)

```bibtex
% ==============================================================
% DATASETS
% ==============================================================

@article{mcphaden1998tao,
  author    = {McPhaden, Michael J. and Busalacchi, Antonio J. and Cheney, Robert
               and Donguy, Jean-René and Gage, Kenneth S. and Halpern, David
               and Ji, Ming and Julian, Paul and Meyers, Gary and Mitchum, Gary T.
               and Niiler, Pearn P. and Picaut, Joel and Reynolds, Richard W.
               and Smith, Neville and Takeuchi, Kunio},
  title     = {The {Tropical Ocean-Global Atmosphere} observing system: A decade of progress},
  journal   = {Journal of Geophysical Research: Oceans},
  year      = {1998},
  volume    = {103},
  number    = {C7},
  pages     = {14169--14240},
  doi       = {10.1029/97JC02906}
}

@article{tierney2023naniar,
  author    = {Tierney, Nicholas J. and Cook, Dianne H.},
  title     = {Expanding Tidy Data Principles to Facilitate Missing Data Exploration,
               Visualization and Assessment of Imputations},
  journal   = {Journal of Statistical Software},
  year      = {2023},
  volume    = {105},
  number    = {7},
  pages     = {1--31},
  doi       = {10.18637/jss.v105.i07}
}

@book{chambers1983graphical,
  author    = {Chambers, John M. and Cleveland, William S. and Kleiner, Beat
               and Tukey, Paul A.},
  title     = {Graphical Methods for Data Analysis},
  publisher = {Wadsworth \& Brooks/Cole},
  address   = {Pacific Grove, CA},
  year      = {1983},
  isbn      = {978-0-534-98052-8}
}

@article{elter2007mammographic,
  author    = {Elter, Matthias and Schulz-Wendtland, Rüdiger and Wittenberg, Thomas},
  title     = {The prediction of breast cancer biopsy outcomes using two {CAD} approaches
               that both emphasize an intelligible decision process},
  journal   = {Medical Physics},
  year      = {2007},
  volume    = {34},
  number    = {11},
  pages     = {4164--4172},
  doi       = {10.1118/1.2786864}
}

@misc{elter2007mammographic_uci,
  author    = {Elter, Matthias},
  title     = {Mammographic Mass},
  year      = {2007},
  howpublished = {UCI Machine Learning Repository},
  doi       = {10.24432/C53K6Z}
}

@inproceedings{smith1988adap,
  author    = {Smith, Jack W. and Everhart, James E. and Dickson, W.C.
               and Knowler, William C. and Johannes, Robert S.},
  title     = {Using the {ADAP} learning algorithm to forecast the onset of diabetes mellitus},
  booktitle = {Proceedings of the Annual Symposium on Computer Application in Medical Care},
  year      = {1988},
  pages     = {261--265},
  organization = {American Medical Informatics Association}
}

@article{mroz1987sensitivity,
  author    = {Mroz, Thomas A.},
  title     = {The sensitivity of an empirical model of married women's hours of work
               to economic and statistical assumptions},
  journal   = {Econometrica},
  year      = {1987},
  volume    = {55},
  number    = {4},
  pages     = {765--799},
  doi       = {10.2307/1911029}
}

@article{heckman1979sample,
  author    = {Heckman, James J.},
  title     = {Sample selection bias as a specification error},
  journal   = {Econometrica},
  year      = {1979},
  volume    = {47},
  number    = {1},
  pages     = {153--161},
  doi       = {10.2307/1912352}
}

% ==============================================================
% METODOLOGIA (MISSING DATA)
% ==============================================================

@article{rubin1976inference,
  author    = {Rubin, Donald B.},
  title     = {Inference and missing data},
  journal   = {Biometrika},
  year      = {1976},
  volume    = {63},
  number    = {3},
  pages     = {581--592},
  doi       = {10.1093/biomet/63.3.581}
}

@book{little2002statistical,
  author    = {Little, Roderick J.A. and Rubin, Donald B.},
  title     = {Statistical Analysis with Missing Data},
  edition   = {2},
  publisher = {Wiley-Interscience},
  address   = {Hoboken, NJ},
  year      = {2002},
  isbn      = {978-0-471-18386-0}
}

@article{schouten2018pyampute,
  author    = {Schouten, Rianne Margaretha and Lugtig, Peter and Vink, Gerko},
  title     = {Generating missing values for simulation purposes: a multivariate
               amputation procedure},
  journal   = {Journal of Statistical Computation and Simulation},
  year      = {2018},
  volume    = {88},
  number    = {15},
  pages     = {2909--2930},
  doi       = {10.1080/00949655.2018.1491577}
}

@article{vanbuuren2011mice,
  author    = {van Buuren, Stef and Groothuis-Oudshoorn, Karin},
  title     = {{mice}: Multivariate Imputation by Chained Equations in {R}},
  journal   = {Journal of Statistical Software},
  year      = {2011},
  volume    = {45},
  number    = {3},
  pages     = {1--67},
  doi       = {10.18637/jss.v045.i03}
}

@article{jager2021benchmark,
  author    = {Jäger, Sebastian and Allhorn, Arndt and Bießmann, Felix},
  title     = {A Benchmark for Data Imputation Methods},
  journal   = {Frontiers in Big Data},
  year      = {2021},
  volume    = {4},
  pages     = {693674},
  doi       = {10.3389/fdata.2021.693674}
}
```

### 8.4 URLs de Download dos Datasets

| Dataset | URL |
|---------|-----|
| Oceanbuoys/TAO | https://github.com/chxy/MissingDataGUI (arquivo `tao.rda`) |
| Airquality | https://vincentarelbundock.github.io/Rdatasets/csv/datasets/airquality.csv |
| Mammographic Mass | https://archive.ics.uci.edu/dataset/161/mammographic+mass |
| Pima Indians Diabetes | https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv |
| Mroz Wages | https://vincentarelbundock.github.io/Rdatasets/csv/carData/Mroz.csv |
