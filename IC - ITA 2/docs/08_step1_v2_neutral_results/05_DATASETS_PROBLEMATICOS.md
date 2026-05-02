# Datasets Problemáticos — Análise Caso a Caso

**Data:** 2026-04-25
**Foco:** os 9 datasets onde `llm_ctx_domain_prior` falha (recall ≤ 20%) e os padrões que emergem

---

## 1. Os 9 datasets críticos

Ordenados por gravidade da falha:

| # | Dataset | Classe real | Recall LLM | LLM majority diz | Padrão |
|:-:|---------|:-----------:|:----------:|:----------------:|---------|
| 1 | `MCAR_hypothyroid_t4u` | MCAR | **0%** | MAR | Defaulting para MAR em domínio clínico |
| 2 | `MCAR_echomonths_epss` | MCAR | 4% | MAR | Defaulting para MAR em domínio clínico |
| 3 | `MNAR_kidney_pot` | MNAR | 4% | MAR | Falha em detectar selection bias |
| 4 | `MNAR_pima_insulin` | MNAR | **4%** | MAR | ⚠️ **caso canônico** falhado |
| 5 | `MNAR_kidney_sod` | MNAR | 8% | MAR | Falha em detectar selection bias |
| 6 | `MCAR_cylinderbands_esavoltage` | MCAR | 10% | MNAR | Detecta truncamento espúrio |
| 7 | `MNAR_hepatitis_protime` | MNAR | 12% | MAR | Falha em detectar selection bias |
| 8 | `MNAR_pima_skinthickness` | MNAR | 18% | MAR | Falha em detectar selection bias |
| 9 | `MCAR_hepatitis_alkphosphate` | MCAR | 20% | MAR | Defaulting para MAR em domínio clínico |

### 1.1 Distribuição de erros por padrão

| Padrão de erro | Datasets | Tendência |
|----------------|:--------:|-----------|
| **MNAR → MAR** (LLM "encontra causa" em outras variáveis) | 5 | Maior fraqueza |
| **MCAR → MAR** (LLM "encontra causa" em domínio clínico) | 3 | Domínios clínicos |
| **MCAR → MNAR** (LLM detecta truncamento que não existe) | 1 | Manufatura/medição |

**Conclusão imediata:** o erro dominante é a tendência do LLM em **inventar causas MAR** quando os dados estatísticos são ambíguos — exatamente o que o Step 1 anti-bias deveria ter combatido, mas não foi efetivo o suficiente.

## 2. Análise individual

### 2.1 `MNAR_pima_insulin` — o fracasso emblemático

**Contexto do dataset:**
- **Origem:** Pima Indians Diabetes (UCI, 1988)
- **Variável (X0):** Insulina sérica (μU/ml)
- **Mecanismo de missing:** O exame de insulina **não era pedido** quando a glicose estava dentro do range normal. Logo, a probabilidade de missing depende do **valor latente de insulina** (que se correlacionaria com glicose normal) — definição clássica de MNAR por seleção.

**O que o LLM fez:**
- Recall: **4%** (apenas 2 de 50 bootstraps classificados como MNAR)
- LLM majority: **MAR**
- Provável raciocínio interno: "missing depende de glicose (X1), que é observável" → MAR

**Por que é um fracasso emblemático:**
1. É o exemplo MNAR mais frequentemente citado em livros-texto de missing data (Little & Rubin 2002; van Buuren 2018)
2. Está **explicitamente nos few-shot examples** do Step 1 (mroz_wages como MNAR de truncamento, mas conceitualmente similar)
3. Era target específico de validação no plano original (`docs/07_next_steps_domain_reasoning/00_overview.md`):
   > **MNAR_pima_insulin = 0%** | Caso clássico MNAR completamente errado | Metadados neutros removeram contexto essencial

**O que o Step 1 deveria ter feito:**
- Reconhecer "insulina" como caso de teste seletivo
- Aplicar Tipo C ("MNAR por seleção") da tipologia
- Anti-bias: recusar inferência MAR sem variável claramente causadora

**Por que falhou:**
- Sem o nome do domínio ("diabetes") explicitamente no metadata neutral, o LLM trata como variável genérica numérica
- Métricas estatísticas (correlação X0 vs X1, taxa missing por quartil) não são suficientes para distinguir MNAR de MAR neste caso
- O Step 2 (Causal DAG) **forçaria** o LLM a nomear a variável que causa o missing, exposing a fragilidade

### 2.2 `MCAR_hypothyroid_t4u` — recall 0%

**Contexto:**
- **Origem:** Thyroid Disease Records (UCI/Garvan Institute)
- **Variável (X0):** T4U (thyroxine uptake), exame laboratorial
- **Mecanismo:** MCAR genuíno — o exame T4U falha por problemas técnicos no equipamento, sem correlação com qualquer condição clínica.

**O que o LLM fez:**
- Recall: **0%** (nenhum dos 50 bootstraps classificado como MCAR)
- LLM majority: MAR
- Provável raciocínio: "exame clínico → médico decide pedir → MAR baseado em outras variáveis"

**Por que é um problema:**
- O LLM tem **forte prior** de que dados clínicos são MAR (médico decide com base em outros sintomas/exames)
- Mesmo com instrução anti-bias dizendo "considere MCAR" para falhas técnicas, o LLM não ativa essa rota para "exame laboratorial"
- O dataset tem menos contexto semântico no metadata neutral do que `pima_insulin` — mais difícil ainda

### 2.3 `MNAR_kidney_pot` e `MNAR_kidney_sod` — par de falhas

**Contexto:**
- Mesmo dataset original (Chronic Kidney Disease, UCI 2015)
- X0 são valores eletrolíticos (potássio e sódio séricos)
- MNAR de seleção: exames de eletrólitos só são pedidos quando há suspeita clínica de desequilíbrio

**Recalls:**
- `MNAR_kidney_pot`: 4%
- `MNAR_kidney_sod`: 8%

**Padrão de erro idêntico:** ambos os datasets têm a mesma estrutura clínica e ambos são classificados como MAR pelo LLM. Sugere que a falha não é do dataset específico mas do **padrão "exame laboratorial em paciente com doença renal"**.

**Implicação:** o LLM não consegue distinguir "MAR (exame pedido com base em outros sintomas)" de "MNAR (exame pedido com base no valor esperado do próprio analito)". A diferença é sutil para um humano e impossível para o LLM sem decomposição causal explícita.

### 2.4 `MCAR_cylinderbands_esavoltage` — falso positivo MNAR

**Contexto:**
- **Origem:** Cylinder Bands (UCI, 1993) — banding em prensa de impressão
- **Variável (X0):** ESA Voltage (Electrostatic Assist)
- **Mecanismo:** MCAR — falhas no sensor de voltagem, sem padrão sistemático

**O que o LLM fez:**
- Recall: **10%** (5 de 50 corretos)
- LLM majority: **MNAR**
- Raciocínio provável: "missing concentrado em valores baixos de X0 → censura → MNAR"

**Por que é um falso positivo:**
- Por chance amostral nos bootstraps, alguns podem ter missing concentrado em uma cauda
- O LLM detecta esse padrão e infere "censura" sem considerar que pode ser ruído estocástico
- A instrução anti-bias **não cobria detecção de falsos positivos MNAR** — só anti-MAR

**Implicação para Step 2:** o DAG causal precisaria de uma instrução tipo:
> Se há missing concentrado em uma cauda, considere se há equipamento que NUNCA mediria valores desse range. Se sim → MNAR (censura). Se a falha é estatisticamente plausível, considere MCAR como hipótese nula.

## 3. Padrões compartilhados

Os 9 datasets críticos compartilham 3 propriedades:

### 3.1 **Domínio clínico ou médico-laboratorial** (7 de 9)

`hypothyroid`, `echomonths`, `kidney_pot/sod`, `pima_insulin/skinthickness`, `hepatitis_*` — todos contextos onde o LLM tem prior fortíssimo de "MAR baseado em decisão médica".

**Implicação:** o anti-bias do Step 1 (genérico) não consegue desfazer o prior dominial. Step 2 precisaria explicitamente listar contraexemplos para cada subdomínio.

### 3.2 **Variável X0 com baixo R² estatístico contra X1-X4** (6 de 9)

Quando a regressão `X0 ~ X1+X2+X3+X4` tem R² < 0.1, as estatísticas de Q-rates de X0 caem para o fallback uniforme (corrigido em 2026-04-20). Isso significa que o LLM tem **menos informação estatística diferencial** disponível, e cai no domínio prior (MAR para clínicos).

**Datasets com R² baixo entre os críticos:** `pima_insulin`, `hypothyroid_t4u`, `echomonths_epss`, `kidney_pot`, `kidney_sod`, `hepatitis_protime`.

**Implicação:** features estatísticas não bastam para esses casos. A solução é **mais raciocínio causal**, não mais features numéricas.

### 3.3 **Bootstraps consistentes na falha** (8 de 9)

Quando o LLM erra um dataset, erra **a maioria dos seus bootstraps de forma similar**. Isso é evidência de que a falha é **sistemática (no prompt/modelo)** e não estocástica (variabilidade entre amostras).

**Verificação:** stdev de `llm_ctx_domain_prior` dentro de cada parent-dataset é tipicamente < 0.15, vs stdev global de 0.34. Significa que o LLM "convergiu" em uma classificação errada para o dataset inteiro.

**Implicação:** Self-Consistency (Step 3) com 5 perspectivas pode ajudar **se as perspectivas ativarem priors diferentes**. Mas se todas seguirem o prior dominial, a votação só consolida o erro. Decomposição causal (Step 2) é mais promissora porque força o LLM a explicitar o caminho do raciocínio.

## 4. O grupo de datasets onde o LLM acerta

Para contraste, os 13 datasets com recall > 80%:

| Dataset | Classe | Recall | Por que funcionou? |
|---------|:------:|:------:|--------------------|
| `MAR_sick_t3` | MAR | 100% | T3 é exame de tiroide pedido com base em sintomas observáveis (X1-X4) — caso MAR canônico |
| `MAR_titanic_age_v2` | MAR | 98% | Idade é fortemente correlacionada com classe de passageiro (observável) |
| `MNAR_mroz_wages` | MNAR | 92% | Salários ausentes para mulheres não-trabalhadoras (truncation por valor latente) — exemplo livro-texto |
| `MAR_titanic_age` | MAR | 88% | Idem `titanic_age_v2` |
| `MAR_oceanbuoys_airtemp` | MAR | 88% | Falha de sensor correlacionada com umidade (X1) — caso oceanográfico canônico |
| `MNAR_adult_capitalgain` | MNAR | 80% | Capital gain frequentemente missing para baixa renda — selection clara |
| `MAR_oceanbuoys_humidity` | MAR | 76% | Idem airtemp |

**Padrão:** o LLM acerta **datasets com correspondência clara aos exemplos canônicos do prompt** (oceanbuoys ≈ exemplo MAR no prompt; mroz_wages ≈ exemplo MNAR no prompt). Falha em datasets onde precisaria **generalizar** o conceito sem casar com um padrão dado.

## 5. Recomendações cirúrgicas

Para o **Step 2 (Causal DAG)**, as seguintes adições direcionam exatamente os datasets problemáticos:

### 5.1 Subtipos de MNAR clínico explícitos

```
TIPO C2: MNAR por SELEÇÃO MÉDICA
Quando um exame laboratorial só é pedido após suspeita de anormalidade,
o valor missing depende implicitamente do valor latente esperado.
Exemplos:
- Insulina (pima): só medida se glicose suspeita
- Eletrólitos (kidney_pot, kidney_sod): só medidos se sintomas renais
- T4U/T3 (hypothyroid): se TSH alterado
```

### 5.2 Instrução de calibração para "tudo parece MAR"

```
ATENÇÃO ESPECIAL: Em domínios clínicos, NÃO presuma MAR mesmo
quando há correlação estatística com outras variáveis.
Pergunte: "A decisão de MEDIR esta variável depende do valor
esperado dela mesma?" Se sim → MNAR. Se a decisão é independente
do valor esperado → MAR.
```

### 5.3 Hipótese nula MCAR para domínios não-clínicos

```
Em domínios técnicos (manufatura, medição automatizada), considere
MCAR como hipótese nula a ser refutada. Falha de equipamento aleatória
é o cenário default em sensores industriais.
```

Essas três adições no prompt do Step 2 atacam diretamente os 3 padrões identificados na seção 3 acima — clínico (7 datasets), R² baixo (6 datasets), e falsos positivos MNAR (1 dataset).

## 6. Datasets como casos de teste para Step 2

Para validar o Step 2, propõe-se um **conjunto-alvo** de 6 datasets:

| Dataset | Classe | Recall atual (Step 1) | Target (Step 2) |
|---------|:------:|:--------------------:|:---------------:|
| `MNAR_pima_insulin` | MNAR | 4% | **>50%** |
| `MNAR_kidney_pot` | MNAR | 4% | >40% |
| `MCAR_hypothyroid_t4u` | MCAR | 0% | >30% |
| `MCAR_echomonths_epss` | MCAR | 4% | >30% |
| `MNAR_kidney_sod` | MNAR | 8% | >40% |
| `MCAR_cylinderbands_esavoltage` | MCAR | 10% | >30% |

Se o Step 2 elevar recall **médio destes 6** de ~5% → 35%, isso geraria ganho agregado estimado de:
- 6 datasets × ~50 bootstraps × 30pp recall improvement / 1.421 total ≈ **+6.3pp na accuracy global**

Suficiente para empurrar CV avg de 49.3% → ~55%, alinhando com a referência publicável e justificando o investimento adicional em Pro + DAG prompting.
