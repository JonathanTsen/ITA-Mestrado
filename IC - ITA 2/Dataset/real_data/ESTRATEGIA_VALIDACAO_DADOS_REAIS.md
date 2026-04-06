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

## 3. Pipeline de Pré-processamento (raw → processado)

O script `Scripts/preparar_dados_reais.py` converte cada dataset bruto para o formato esperado pelo pipeline de classificação. As etapas abaixo são aplicadas a todos os datasets.

### 3.1 Visão geral do fluxo

```
Dataset/real_data/{MECANISMO}/{arquivo_original}
        │
        ▼
   [1] Seleção da variável com missing → X0
   [2] Seleção das variáveis preditoras → X1, X2, X3, X4
   [3] Preenchimento de missing nas preditoras (amostragem da distribuição observada)
   [4] Cap da taxa de missing em X0 para ≤10% (subsampling aleatório dos NaN)
   [5] Normalização min-max para [0, 1]
   [6] Jitter gaussiano em variáveis ordinais (apenas Mammographic)
   [7] Salvamento como .txt tab-separated
        │
        ▼
Dataset/real_data/processado/{MECANISMO}/{arquivo_processado}.txt
```

### 3.2 Etapas detalhadas

#### Etapa 1 — Seleção da variável alvo (X0)

Para cada dataset, é escolhida a variável que concentra os valores missing e que tem justificativa de domínio para o mecanismo de interesse. Ela se torna **X0** no arquivo de saída e é a única coluna que mantém os `NaN`.

| Dataset | Variável original → X0 | Motivo |
|---------|------------------------|--------|
| Oceanbuoys (variante 1) | `humidity` | 93 NaN por falha de sensor (MCAR) |
| Oceanbuoys (variante 2) | `air.temp` | 81 NaN por falha de sensor (MCAR) |
| Airquality | `Ozone` | 37 NaN correlacionados com Wind/Temp (MAR) |
| Mammographic Mass | `Density` | 56 NaN dependentes de BI-RADS e Age (MAR) |
| Pima Diabetes | `Insulin` | 374 zeros convertidos em NaN (MNAR) |
| Mroz Wages | `lwg` (log wage) | 325 NaN para mulheres fora da força de trabalho (MNAR) |

#### Etapa 2 — Seleção das variáveis preditoras (X1–X4)

São escolhidas as variáveis observadas mais relevantes para o mecanismo de cada dataset. A ordem importa: **X1 é sempre a variável mais associada ao mecanismo MAR** (seguindo a convenção do pipeline sintético onde MAR depende de X1).

| Dataset | X1 | X2 | X3 | X4 |
|---------|----|----|----|----|
| Oceanbuoys (humidity) | sea.surface.temp | air.temp | uwind | vwind |
| Oceanbuoys (air.temp) | sea.surface.temp | humidity | uwind | vwind |
| Airquality | Wind | Temp | Solar.R | Month |
| Mammographic Mass | BI-RADS | Age | Shape | Margin |
| Pima Diabetes | Glucose | BloodPressure | BMI | Age |
| Mroz Wages | age | inc | k5 | wc (0/1) |

#### Etapa 3 — Tratamento de missing nas preditoras

As colunas X1–X4 devem estar completamente preenchidas (mesma convenção dos dados sintéticos). NaN residuais são preenchidos por **amostragem aleatória da distribuição observada** da coluna (em vez de média simples, para preservar a variância original). Casos específicos:

- **Oceanbuoys**: `sea.surface.temp` (3 NaN) e `air.temp` (81 NaN) — imputados com amostras da distribuição observada.
- **Airquality**: `Solar.R` (7 NaN) — imputado com amostras.
- **Pima Diabetes**: `Glucose` (5 zeros), `BloodPressure` (35 zeros) e `BMI` (11 zeros) — zeros biologicamente impossíveis convertidos em NaN, depois imputados com amostras.
- **Mammographic Mass**: linhas com NaN nas preditoras (BI-RADS, Age, Shape, Margin) são **removidas** (não imputadas), pois a remoção é mais conservadora quando há poucas ausências.
- **Mroz Wages**: variável categórica `wc` (college education: yes/no) convertida para binária (1/0).

> **Justificativa (amostragem vs média):** Imputar com a média cria um pico artificial na distribuição da variável, achatando a variância. Amostrar da distribuição observada preserva a forma original da distribuição e produz valores mais realistas para o classificador.

#### Etapa 4 — Cap da taxa de missing (≤10%)

Datasets com taxa de missing acima do range de treinamento sintético (1–10%) têm seus NaN em X0 reduzidos por **subsampling aleatório**:

1. Calcula quantos NaN manter: `n_keep = 10% × n_total`
2. Seleciona aleatoriamente quais posições NaN manter
3. Imputa os NaN excedentes com amostras da distribuição observada de X0

| Dataset | Taxa original | Após cap | NaN mantidos / originais |
|---------|--------------|----------|--------------------------|
| Oceanbuoys humidity | 12.6% (93) | 9.9% (73) | 73/93 (78%) |
| Oceanbuoys airtemp | 11.0% (81) | 9.9% (73) | 73/81 (90%) |
| Airquality ozone | 24.2% (37) | 9.8% (15) | 15/37 (41%) |
| Mammographic density | 6.3% (56) | 6.3% (56) | sem alteração |
| Pima insulin | 48.7% (374) | 9.9% (76) | 76/374 (20%) |
| Mroz wages | 43.2% (325) | 10.0% (75) | 75/325 (23%) |

> **Preservação do mecanismo:** O subsampling é puramente aleatório (uniforme sobre as posições NaN), portanto o subconjunto retido preserva o padrão original do mecanismo. Para MNAR: se P(NaN|X0) = f(X0), então P(NaN retido|X0) ∝ f(X0). Para MAR e MCAR, a mesma lógica se aplica.

#### Etapa 5 — Normalização min-max para [0, 1]

Aplicada separadamente em X0 e em X1–X4 para manter compatibilidade com os dados sintéticos (gerados como Uniform[0,1]):

- **X1–X4**: normalizadas com min/max da coluna completa.
- **X0**: normalizada com min/max calculado **apenas sobre os valores observados** (os NaN são preservados e não participam do cálculo).

```
X_normalizado = (X - min(X_observado)) / (max(X_observado) - min(X_observado))
```

#### Etapa 6 — Jitter gaussiano em variáveis ordinais

Aplicado **apenas ao Mammographic Mass**, cujas variáveis são ordinais com poucos valores distintos (ex: Density ∈ {1,2,3,4} → normalizado para {0, 0.33, 0.67, 1.0}).

Ruído gaussiano N(0, σ=0.02) é adicionado aos valores observados, com clipping em [0, 1]:

```
X_jittered = clip(X_normalizado + N(0, 0.02), 0, 1)
```

Colunas com jitter: X0 (Density), X1 (BIRADS), X3 (Shape), X4 (Margin).
X2 (Age) é contínua por natureza e não recebe jitter.

> **Justificativa:** O classificador foi treinado com dados contínuos Uniform[0,1]. Sem jitter, X0 com apenas 4 valores distintos gera features estatísticas (X0_mean, quartis) atípicas. O jitter (σ=0.02, ~6% do gap entre níveis ordinais) cria variação contínua sem alterar a semântica ordinal.

#### Etapa 7 — Formato de saída

Arquivo `.txt` tab-separated com cabeçalho, 5 colunas (`X0` a `X4`), mesmo formato gerado por `gerador.py`. Compatível diretamente com `extract_features.py`.

### 3.3 Resumo dos arquivos gerados

| Arquivo de saída | Obs | Missing X0 (original → final) | Valores únicos X0 | No range? |
|------------------|-----|-------------------------------|--------------------|----|
| `MCAR_oceanbuoys_humidity.txt` | 736 | 12.6% → 9.9% | 166 | **Sim** |
| `MCAR_oceanbuoys_airtemp.txt` | 736 | 11.0% → 9.9% | 307 | **Sim** |
| `MAR_airquality_ozone.txt` | 153 | 24.2% → 9.8% | 67 | **Sim** |
| `MAR_mammographic_density.txt` | 886 | 6.3% (sem cap) | 823 (jitter) | **Sim** |
| `MNAR_pima_insulin.txt` | 768 | 48.7% → 9.9% | 185 | **Sim** |
| `MNAR_mroz_wages.txt` | 753 | 43.2% → 10.0% | 374 | **Sim** |

> **Nota:** Todos os datasets agora estão dentro do range de treinamento (1–10%). O cap de missing preserva o padrão do mecanismo original (ver Etapa 4). O Mammographic ganhou valores contínuos via jitter (de 4 valores ordinais para 823 valores únicos).

### 3.4 Detalhamento das Transformações por Dataset

Abaixo, a transformação concreta aplicada a cada arquivo bruto pelo script `preparar_dados_reais.py`.

#### 3.4.1 MCAR — Oceanbuoys / TAO → 2 arquivos

**Arquivo de entrada:** `real_data/MCAR/oceanbuoys_tao.csv` (736 linhas, 8 colunas)

```
year, latitude, longitude, sea.surface.temp, air.temp, humidity, uwind, vwind
```

**Variante 1 — humidity como X0:**

| Passo | Operação |
|-------|----------|
| 1 | `humidity` → X0 (93 NaN originais) |
| 2 | Imputa NaN de `air.temp` (81) e `sea.surface.temp` (3) com amostras da distribuição observada |
| 3 | Seleciona X1=`sea.surface.temp`, X2=`air.temp`, X3=`uwind`, X4=`vwind` |
| 4 | Cap missing: 93 → 73 NaN (12.6% → 9.9%), 20 NaN imputados com amostras |
| 5 | Normaliza X1–X4 com min-max; X0 com min-max sobre observados |
| 6 | Salva como `processado/MCAR/MCAR_oceanbuoys_humidity.txt` (736 linhas, 9.9% missing) |

**Variante 2 — air.temp como X0:**

| Passo | Operação |
|-------|----------|
| 1 | `air.temp` → X0 (81 NaN originais) |
| 2 | Imputa NaN de `humidity` (93) e `sea.surface.temp` (3) com amostras |
| 3 | Seleciona X1=`sea.surface.temp`, X2=`humidity`, X3=`uwind`, X4=`vwind` |
| 4 | Cap missing: 81 → 73 NaN (11.0% → 9.9%) |
| 5–6 | Normalização min-max + salvamento (736 linhas, 9.9% missing) |

#### 3.4.2 MAR — Airquality → 1 arquivo

**Arquivo de entrada:** `real_data/MAR/airquality.csv` (153 linhas, 7 colunas incluindo `rownames`)

```
rownames, Ozone, Solar.R, Wind, Temp, Month, Day
```

| Passo | Operação |
|-------|----------|
| 1 | `Ozone` → X0 (37 NaN originais) |
| 2 | Imputa NaN de `Solar.R` (7) com amostras da distribuição observada |
| 3 | Seleciona X1=`Wind`, X2=`Temp`, X3=`Solar.R`, X4=`Month` |
| 4 | Cap missing: 37 → 15 NaN (24.2% → 9.8%) |
| 5 | Normalização min-max |
| 6 | Salva como `processado/MAR/MAR_airquality_ozone.txt` (153 linhas, 9.8% missing) |

> **Nota:** `Day` e `rownames` são descartados (não informativos para o mecanismo).

#### 3.4.3 MAR — Mammographic Mass → 1 arquivo

**Arquivo de entrada:** `real_data/MAR/mammographic_mass_raw.csv` (961 linhas, 6 colunas, sem cabeçalho)

```
BIRADS, Age, Shape, Margin, Density, Severity
```

| Passo | Operação |
|-------|----------|
| 1 | Leitura com `na_values="?"` (valores `?` no CSV tratados como NaN) |
| 2 | Converte colunas para numérico (`pd.to_numeric`) |
| 3 | **Remove** linhas com NaN em `BIRADS`, `Age`, `Shape` ou `Margin` (886 restam) |
| 4 | `Density` → X0 (56 NaN, 6.3% — já dentro do range, sem cap) |
| 5 | Seleciona X1=`BIRADS`, X2=`Age`, X3=`Shape`, X4=`Margin` |
| 6 | Normalização min-max |
| 7 | **Jitter gaussiano** N(0, 0.02) em X0, X1, X3, X4 (ordinais). X2 (Age) é contínua, sem jitter |
| 8 | Salva como `processado/MAR/MAR_mammographic_density.txt` (886 linhas, 6.3% missing, 823 valores únicos em X0) |

> **Nota:** Sem jitter, Density teria apenas 4 valores distintos {0, 0.33, 0.67, 1.0}. O jitter cria variação contínua (823 únicos) compatível com o treino sintético. Única operação de remoção de linhas — nos demais datasets, missing nas preditoras é imputado por amostragem.

#### 3.4.4 MNAR — Pima Indians Diabetes → 1 arquivo

**Arquivo de entrada:** `real_data/MNAR/pima_diabetes_raw.csv` (768 linhas, 9 colunas, sem cabeçalho)

```
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age, Outcome
```

| Passo | Operação |
|-------|----------|
| 1 | Converte zeros de `Insulin` para NaN → X0 (374 NaN, 48.7%) |
| 2 | Converte zeros biologicamente impossíveis para NaN: `Glucose` (5), `BloodPressure` (35), `BMI` (11) |
| 3 | Imputa NaN de `Glucose`, `BloodPressure` e `BMI` com amostras da distribuição observada |
| 4 | Seleciona X1=`Glucose`, X2=`BloodPressure`, X3=`BMI`, X4=`Age` |
| 5 | Cap missing: 374 → 76 NaN (48.7% → 9.9%) |
| 6 | Normalização min-max |
| 7 | Salva como `processado/MNAR/MNAR_pima_insulin.txt` (768 linhas, 9.9% missing) |

> **Nota:** Os zeros no Pima são uma convenção do dataset — valores 0 para Glucose, BloodPressure, BMI e Insulin são biologicamente impossíveis e representam dados não coletados. O subsampling retém 76 dos 374 NaN originais, preservando o padrão MNAR (exames não solicitados quando médico não suspeita diabetes).

#### 3.4.5 MNAR — Mroz Wages → 1 arquivo

**Arquivo de entrada:** `real_data/MNAR/mroz_wages.csv` (753 linhas, 9 colunas)

```
rownames, lfp, k5, k618, age, wc, hc, lwg, inc
```

| Passo | Operação |
|-------|----------|
| 1 | Define `lwg` como NaN onde `lfp == "no"` (mulheres fora da força de trabalho) → X0 (325 NaN, 43.2%) |
| 2 | Converte `wc` ("yes"/"no") para numérico (1/0) |
| 3 | Seleciona X1=`age`, X2=`inc`, X3=`k5`, X4=`wc_num` |
| 4 | Cap missing: 325 → 75 NaN (43.2% → 10.0%) |
| 5 | Normalização min-max |
| 6 | Salva como `processado/MNAR/MNAR_mroz_wages.txt` (753 linhas, 10.0% missing) |

> **Nota:** O CSV original contém valores estimados de `lwg` (salários do modelo de Heckman) para `lfp == "no"`. O script marca esses como NaN porque o salário real não foi observado. O subsampling retém 75 dos 325 NaN, preservando o padrão MNAR (seleção de Heckman).

### 3.5 Reprodução

Para regenerar todos os arquivos processados a partir dos originais:

```bash
cd "IC - ITA 2/Scripts"
python preparar_dados_reais.py
```

---

## 4. Dificuldades e Desafios

### 4.1 Ausência de Ground Truth

**O problema fundamental:** Em dados reais, **nunca sabemos com certeza** qual é o mecanismo de missing. O que temos são argumentos de domínio que sugerem fortemente um mecanismo. Isso significa que:

- Se o classificador errar, pode ser porque: (a) o classificador falhou, ou (b) o mecanismo não é exatamente o que assumimos.
- Não podemos calcular "acurácia" no sentido estrito — apenas avaliar **consistência** entre a predição e o conhecimento de domínio.

### 4.2 Taxas de Missing Incompatíveis

| Dataset | Missing original | Após cap | Range sintético |
|---------|-----------------|----------|-----------------|
| Oceanbuoys (humidity) | 12.6% | 9.9% ✅ | 1-10% |
| Oceanbuoys (airtemp) | 11.0% | 9.9% ✅ | 1-10% |
| Airquality (Ozone) | 24.2% | 9.8% ✅ | 1-10% |
| Mammographic (Density) | 6.3% | 6.3% ✅ | 1-10% |
| Pima (Insulin) | 48.7% | 9.9% ✅ | 1-10% |
| Mroz (Wages) | 43.2% | 10.0% ✅ | 1-10% |

**Mitigação aplicada:** Subsampling aleatório dos NaN em X0 (Etapa 4 do pipeline). Os NaN excedentes são imputados com amostras da distribuição observada. O subconjunto retido preserva o padrão do mecanismo original (ver justificativa na Seção 3.2, Etapa 4).

**Limitação residual:** Para datasets com cap agressivo (Pima: 374→76, Mroz: 325→75), a quantidade absoluta de NaN é pequena, o que pode reduzir o poder estatístico das features discriminativas.

### 4.3 Distribuições Diferentes

Os dados sintéticos são gerados como **Uniform[0,1]**, mas os dados reais têm distribuições variadas (normal, skewed, discreta, etc.). Mesmo após normalização min-max para [0,1], as distribuições internas são diferentes.

**Impacto:** Features como `X0_mean`, `X0_q25`, `X0_q50`, `X0_q75` terão valores que o modelo nunca viu no treinamento.

**Estratégias de mitigação:**
1. **Rank-transform**: Converter valores para ranks antes de calcular features (torna qualquer distribuição aproximadamente uniforme).
2. **Normalizar features, não dados**: Aplicar StandardScaler nas features extraídas, não nos dados brutos.
3. **Treinar com distribuições mistas**: Gerar dados sintéticos com Normal, Log-normal, Exponencial além de Uniform.

### 4.4 Poucas Amostras por Mecanismo

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

### 4.5 Variáveis Discretas vs Contínuas

Mammographic Mass tem variáveis ordinais (1-5). Mroz tem variáveis binárias (wc, hc). Os dados sintéticos são puramente contínuos.

**Impacto:** Features discriminativas (AUC, correlação, Mann-Whitney) podem se comportar diferentemente com dados discretos.

### 4.6 Ambiguidade dos Mecanismos

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
Dataset/real_data/
├── MCAR/
│   └── oceanbuoys_tao.csv           # Dataset original (736 linhas, 8 cols)
├── MAR/
│   ├── airquality.csv               # Dataset original (153 linhas, 7 cols)
│   └── mammographic_mass_raw.csv    # Dataset original (961 linhas, 6 cols, sem cabeçalho)
├── MNAR/
│   ├── pima_diabetes_raw.csv        # Dataset original (768 linhas, 9 cols, sem cabeçalho)
│   └── mroz_wages.csv               # Dataset original (753 linhas, 9 cols)
├── processado/                       # Formato padronizado (X0-X4, tab-separated)
│   ├── MCAR/
│   │   ├── MCAR_oceanbuoys_humidity.txt   # 736 linhas, 9.9% missing ✅
│   │   └── MCAR_oceanbuoys_airtemp.txt    # 736 linhas, 9.9% missing ✅
│   ├── MAR/
│   │   ├── MAR_airquality_ozone.txt       # 153 linhas, 9.8% missing ✅
│   │   └── MAR_mammographic_density.txt   # 886 linhas, 6.3% missing ✅ (com jitter)
│   └── MNAR/
│       ├── MNAR_pima_insulin.txt          # 768 linhas, 9.9% missing ✅
│       └── MNAR_mroz_wages.txt            # 753 linhas, 10.0% missing ✅
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
