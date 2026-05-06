# Verificação dos novos datasets MNAR — Evidência e reclassificação

**Data:** 2026-05-05
**Contexto:** Após expansão do benchmark com 6 novos datasets candidatos a MNAR, cada um foi verificado rigorosamente quanto ao mecanismo de missingness.

---

## Método de verificação

Para cada dataset, testamos:
1. **Mecanismo de domínio**: a justificativa publicada para MNAR é sólida?
2. **Correlação mask~covariáveis**: point-biserial entre missingness e variáveis observadas. Se forte → componente MAR significativa.
3. **Dependência no valor não-observado**: o valor de X0 causa sua própria ausência?

Critério de reclassificação: se as correlações mask~covariáveis são fortes (|r| > 0.15, p < 0.001) e o mecanismo de domínio é consistente com MAR, o dataset é reclassificado.

---

## Resultados

### NHANES 2017-18 — Left-censoring por LOD

| Dataset | LOD | % abaixo LOD | Valor below-LOD | Veredicto |
|:--|:--|:--:|:--|:--|
| `nhanes_cadmium` | 0.100 μg/L | 18.6% | Todos = 0.07 (LLOD/√2) | **MNAR puro** ✅ |
| `nhanes_mercury` | 0.28 μg/L | 26.4% | Todos = 0.20 (LLOD/√2) | **MNAR puro** ✅ |
| `nhanes_cotinine` | 0.015 ng/mL | 34.2% | Todos = 0.011 (LLOD/√2) | **MNAR puro** ✅ |

**Justificativa**: left-censoring por limite de detecção (LOD) é MNAR por definição — o valor está ausente *porque é baixo demais para o instrumento medir*. O limiar é uma propriedade física do instrumento, não depende de covariáveis. Nenhuma variável observada pode fazer um valor abaixo do LOD aparecer.

**Nota**: a correlação `below-LOD ~ idade` é significativa para cadmium (r=−0.44, p≈0) porque jovens têm menor exposição ambiental. Isso NÃO anula o MNAR: o LOD é um limiar fixo; a idade influencia o *nível de exposição* (e portanto a probabilidade de estar abaixo do LOD), mas o mecanismo de censura é físico.

**Referências**:
- Tellez-Plaza et al. (2012). "Cadmium exposure and all-cause and cardiovascular mortality." *Environmental Health Perspectives*, 120(7), 1017–1022.
- Bernert et al. (2011). "Toward improved statistical methods for analyzing cotinine-biomarker health association data." *Tobacco Induced Diseases*, 9(11).
- Helsel, D. R. (2012). *Statistics for Censored Environmental Data Using Minitab and R*. Wiley. (Cap. 1: left-censoring como MNAR.)

---

### SUPPORT2 — Test-ordering clínico

| Dataset | % missing | Correlações mask~covariáveis | Veredicto |
|:--|:--:|:--|:--|
| `support2_albumin` | 37.0% | mask~age: r=0.03 p=0.01; mask~hrt: r=−0.07 p<0.001 | **MNAR misto** ✅ |
| `support2_bilirubin` | 28.6% | mask~age: r=0.08 p<0.001; mask~hrt: r=−0.07 p<0.001 | **MNAR misto** ✅ |
| `support2_pafi` | 25.5% | mask~hrt: **r=−0.19** p<0.001; mask~temp: **r=−0.18** p<0.001; mask~resp: r=−0.09 p<0.001 | **Reclassificado → MAR** ⚠️ |

**MNAR mistos (albumin, bilirubin)**: a correlação mask~covariáveis é estatisticamente significativa mas fraca (|r| < 0.08). Isso é esperado: a decisão de ordenar o teste depende parcialmente de sinais clínicos observados (componente MAR), mas também do valor esperado do exame (componente MNAR). Na taxonomia de Rubin, qualquer dependência no valor não-observado classifica como MNAR. Os datasets são mantidos como MNAR com a ressalva de que têm componente MAR.

**Reclassificado (pafi)**: as correlações mask~hrt (r=−0.19) e mask~temp (r=−0.18) são moderadas-fortes. A decisão de realizar gasometria arterial (ABG) é predominantemente baseada em deterioração clínica observável: taquicardia, febre e taquipneia. A componente MNAR (suspeita sobre o valor de PaO2/FiO2 em si) é secundária. Reclassificado como MAR.

**Referência**: Knaus, W. A., Harrell, F. E., et al. (1995). "The SUPPORT prognostic model." *Annals of Internal Medicine*, 122(3), 191–203.

---

## Comparação com datasets MNAR existentes

| Dataset existente | Mecanismo | Força MNAR |
|:--|:--|:--|
| `pima_insulin` | Zeros estruturais (não-medido = normal esperado) | Forte |
| `pima_skinthickness` | Truncamento físico (caliper cap 45mm) | Forte |
| `mroz_wages` | Heckman selection (fora da força de trabalho) | Forte |
| `adult_capitalgain` | Zeros estruturais (não-investidores) | Forte |
| `hepatitis_protime` | Test-ordering (coagulação suspeita) | Misto (como SUPPORT2) |
| `kidney_pot`/`kidney_sod` | Valores extremos não reportados | Misto |
| **nhanes_cadmium/mercury/cotinine** | **LOD left-censoring** | **Puro — mais forte do benchmark** |
| **support2_albumin/bilirubin** | **Test-ordering ICU** | **Misto (como hepatitis_protime)** |

Os datasets NHANES são o tipo mais forte e indiscutível de MNAR no benchmark. Adicionam um subtipo (left-censoring) que não existia anteriormente.

---

## Resumo final

| Ação | Datasets | Justificativa |
|:--|:--|:--|
| MNAR confirmado (LOD) | nhanes_cadmium, nhanes_mercury, nhanes_cotinine | Mecanismo físico de left-censoring |
| MNAR confirmado (misto) | support2_albumin, support2_bilirubin | Test-ordering com componente MAR fraca (|r|<0.08) |
| Reclassificado → MAR | support2_pafi | Correlações mask~covariáveis fortes (|r|=0.18-0.19) |

**Benchmark final: 39 datasets (13 MCAR, 12 MAR, 14 MNAR)**
