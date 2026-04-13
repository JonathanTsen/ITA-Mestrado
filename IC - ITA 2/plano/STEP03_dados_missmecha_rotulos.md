# STEP 03: Dados Diversos com MissMecha + Rotulos Validados

**Fase 4C — Mais dados, mais variantes, rotulos confiaveis**

---

## Problema

Com apenas 3 datasets por mecanismo e GroupShuffleSplit, o teste contem 2-3 datasets nunca vistos. O modelo nao consegue generalizar. Alem disso, 2/9 rotulos sao inconsistentes (oceanbuoys) e os dados sinteticos so implementam 1 variante por mecanismo.

---

## Parte A: Dados Sinteticos Diversos com MissMecha

### Motivacao

Nosso `gerador.py` gera:
- MCAR: masking uniforme
- MAR: logistic model baseado em X1
- MNAR: self-censoring baseado em X0

MissMecha implementa **17 variantes** (3 MCAR, 8 MAR, 6 MNAR). Treinar so em 1 variante por mecanismo e como ensinar alguem a reconhecer "cachorro" mostrando so golden retrievers — nao generaliza.

### O que fazer

1. Instalar MissMecha: `pip install missmecha-py`
2. Modificar `gerador.py` (ou criar `gerador_v2.py`) para gerar datasets usando multiplas variantes de cada mecanismo
3. Para cada variante, gerar N datasets com distribuicoes de base variadas (uniforme, normal, exponencial, bimodal)
4. Manter o formato padrao (5 colunas, 1000 linhas, tab-separated)

### Variantes a incluir

**MCAR (3 tipos):**
- Tipo 1: Uniform masking (cada celula independente)
- Tipo 2: Fixed selection (N celulas fixas aleatorias)
- Tipo 3: Column-balanced (missing distribuido uniformemente entre colunas)

**MAR (selecionar 4-5 dos 8 tipos):**
- Logistic model (missing depende de X1 via funcao logistica)
- Point-biserial (correlacao direta com variavel binaria)
- Correlation ranking (missing nos ranks mais altos de X1)
- Rank-based masking (baseado em ranks de variavel de controle)
- Binary grouping (split por mediana de X1)

**MNAR (selecionar 3-4 dos 6 tipos):**
- Quantile thresholding (missing acima/abaixo de quantil de X0)
- Self-masking (X0 faz mask em si mesmo)
- Logistic self-dependence (probabilidade de missing depende logisticamente de X0)
- Quantile cut (corte por quartis superior/inferior)

### Meta de dados sinteticos

- 5 variantes x 3 mecanismos x 100 datasets = **1500 datasets sinteticos**
- Com distribuicoes de base variadas: uniforme, normal(0.5, 0.15), exponencial(1), beta(2,5)
- GroupShuffleSplit por variante (nao por arquivo) para testar generalizacao entre variantes

### Resultado da implementacao (2026-04-12)

Criado `gerador_v2.py` com **12 variantes** (3 MCAR + 5 MAR + 4 MNAR):

| Mecanismo | Variante | Descricao |
|-----------|----------|-----------|
| MCAR | uniform | Cada celula prob independente |
| MCAR | fixed | Exatamente N posicoes aleatorias |
| MCAR | block | Blocos contiguos (falha de sensor) |
| MAR | logistic | sigmoid(beta * standardize(X1)) |
| MAR | threshold | Missing quando X1 > percentil |
| MAR | rank | Missing nos top-k ranks de X1 |
| MAR | quantile_group | Prob varia por quartil de X1 |
| MAR | multi_predictor | Depende de X1 + X2 |
| MNAR | self_logistic | sigmoid(beta * standardize(X0)) |
| MNAR | quantile_threshold | Missing quando X0 > quantil |
| MNAR | tail_censoring | Missing nos extremos de X0 |
| MNAR | self_noisy | Self-masking com ruido gaussiano |

**Total gerado: 1200 datasets** (300 MCAR + 500 MAR + 400 MNAR), 4 distribuicoes base.

**Nota sobre MissMecha:** A API do MissMecha nao suporta diretamente missing apenas em X0 para MAR/MNAR (modo global aplica em todas as colunas). As variantes foram implementadas manualmente seguindo a mesma logica dos mecanismos do MissMecha, mas com controle de coluna-alvo. O Little's MCAR test do MissMecha foi usado na validacao de rotulos.

---

## Parte B: Validar Rotulos dos Datasets Reais

### Motivacao

A Fase 3 revelou que oceanbuoys (rotulado MCAR) mostra forte evidencia de MAR. Rotulos errados envenenam o treino.

### Estrategia de validacao em 3 testes

**Teste 1 — Little's MCAR test (via MissMecha):**
- Hipotese nula: dados sao MCAR
- p > 0.05 → nao rejeita MCAR (consistente com rotulo MCAR)
- p < 0.05 → rejeita MCAR (inconsistente se rotulado como MCAR)
- Usar `MCARTest.little_mcar_test()` do MissMecha

**Teste 2 — Correlacao mask-Xi (evidencia MAR):**
- Calcular correlacao ponto-biserial entre mascara de X0 e cada Xi (i=1..4)
- Se algum |corr| > 0.1 com p < 0.05 → evidencia de MAR
- Quanto maior a correlacao, mais forte a evidencia

**Teste 3 — KS observados vs imputados (evidencia MNAR):**
- Imputar X0 com mediana
- Teste KS entre X0_observado e X0_imputado
- Se p < 0.05 → distribuicao dos observados difere do esperado → evidencia MNAR

### O que fazer com cada resultado

| Resultado | Acao |
|-----------|------|
| Little p > 0.05, sem correlacao mask-Xi, sem KS significativo | Rotulo MCAR confirmado |
| Little p < 0.05, correlacao mask-Xi significativa | Reclassificar como MAR |
| Little p < 0.05, KS significativo, sem correlacao mask-Xi | Possivelmente MNAR |
| Testes conflitantes | Marcar como "ambiguo", remover do treino ou usar com cautela |

### Criar script `validar_rotulos.py`

- Recebe diretorio com datasets reais processados
- Roda os 3 testes em cada dataset original (nao nos bootstraps)
- Gera relatorio com: dataset, rotulo_atual, little_p, max_corr_Xi, ks_stat, diagnostico, rotulo_recomendado

### Acao imediata: oceanbuoys

Os 2 datasets oceanbuoys (airtemp, humidity) devem ser removidos ou reclassificados como MAR antes de qualquer experimento futuro. Eles representam 69% das amostras MCAR e introduzem ruido no treino.

### Resultado da validacao (2026-04-12)

Validacao executada em 23 datasets reais. Resultados completos em `Output/v2_improved/step03/real/validacao_rotulos/`.

**Oceanbuoys confirmado como MAR:** Little p=0.000, |corr(mask, X1)|=0.333 para ambos.

**Resumo geral: 8/23 consistentes, 15/23 inconsistentes.**

Datasets com rotulo confirmado:
- MCAR: autompg_horsepower, breastcancer_barenuclei, cylinderbands_esavoltage (3 de 7)
- MAR: mammographic_density, sick_tsh, titanic_age, titanic_age_v2 (4 de 9)
- MNAR: mroz_wages (1 de 7, marcado como "ambiguo")

**Descoberta critica:** O teste KS para MNAR tem poder muito baixo com missing rate ~10%. Quase todos os datasets MNAR da literatura testam como MCAR — isso e uma limitacao teorica conhecida (MNAR e indetectavel apenas com dados observados quando a taxa de missing e baixa).

---

## Parte C: Expandir Datasets Reais

### Meta: 10+ datasets por mecanismo

Para cada dataset candidato, verificar:
1. Tem missing natural (nao imputado)?
2. Ha documentacao/literatura sobre o mecanismo?
3. Tem pelo menos 200 linhas e 5 colunas?
4. A validacao estatistica (Parte B) confirma o rotulo?

### Processo por dataset

1. Baixar dataset (UCI, Kaggle, CDC)
2. Selecionar coluna com missing (X0) e 4 colunas auxiliares (X1-X4)
3. Normalizar para [0,1] (MinMaxScaler)
4. Salvar em formato padrao (tab-separated, 5 colunas)
5. Rodar validacao de rotulo
6. Se confirmado, gerar 50 bootstraps
7. Salvar em `Dataset/real_data/processado/{MCAR|MAR|MNAR}/`

### Datasets prioritarios (da literatura)

Listar 5-7 por mecanismo com fonte, variavel com missing, e evidencia do mecanismo. Priorizar datasets que ja sao usados em papers de missing data (MechDetect usou 101 do OpenML — podemos usar os mesmos).

### Resultado da expansao (2026-04-12)

Script `expandir_dados_reais.py` baixa datasets do OpenML e URLs diretas.

**Inventario final: 23 datasets (7 MCAR + 9 MAR + 7 MNAR)**

| Mecanismo | Dataset | Rows | Missing | Fonte | Validacao |
|-----------|---------|------|---------|-------|-----------|
| MCAR | autompg_horsepower | 398 | 1.5% | OpenML | CONFIRMADO |
| MCAR | breastcancer_barenuclei | 699 | 2.3% | Pre-existente | CONFIRMADO |
| MCAR | cylinderbands_bladepressure | 540 | 10.0% | OpenML | INCONSISTENTE → MAR |
| MCAR | cylinderbands_esavoltage | 540 | 10.0% | OpenML | CONFIRMADO |
| MCAR | hypothyroid_t4u | 3772 | 10.0% | OpenML | INCONSISTENTE → MNAR |
| MCAR | oceanbuoys_airtemp | 736 | 9.9% | Pre-existente | INCONSISTENTE → MAR |
| MCAR | oceanbuoys_humidity | 736 | 9.9% | Pre-existente | INCONSISTENTE → MAR |
| MAR | airquality_ozone | 153 | 9.8% | Pre-existente | INCONSISTENTE → MCAR |
| MAR | colic_resprate | 368 | 9.8% | OpenML | INCONSISTENTE → MCAR |
| MAR | hearth_chol | 294 | 7.8% | OpenML | INCONSISTENTE → MCAR |
| MAR | kidney_hemo | 400 | 10.0% | OpenML | INCONSISTENTE → MCAR |
| MAR | mammographic_density | 886 | 6.3% | Pre-existente | CONFIRMADO |
| MAR | sick_t3 | 3772 | 10.0% | OpenML | INCONSISTENTE → MNAR |
| MAR | sick_tsh | 3772 | 9.8% | OpenML | CONFIRMADO |
| MAR | titanic_age | 891 | 10.0% | Pre-existente | CONFIRMADO |
| MAR | titanic_age_v2 | 891 | 10.0% | URL (GitHub) | AMBIGUO |
| MNAR | adult_capitalgain | 1000 | 10.0% | Pre-existente | INCONSISTENTE → MCAR |
| MNAR | colic_refluxph | 368 | 9.8% | OpenML | INCONSISTENTE → MCAR |
| MNAR | cylinderbands_varnishpct | 540 | 10.0% | OpenML | INCONSISTENTE → MAR |
| MNAR | kidney_pot | 400 | 10.0% | OpenML | INCONSISTENTE → MCAR |
| MNAR | kidney_sod | 400 | 10.0% | OpenML | INCONSISTENTE → MCAR |
| MNAR | mroz_wages | 753 | 10.0% | Pre-existente | AMBIGUO |
| MNAR | pima_insulin | 768 | 9.9% | Pre-existente | INCONSISTENTE → MCAR |

**Total de bootstraps gerados: 1132** (332 MCAR + 450 MAR + 350 MNAR).

**Meta nao atingida para MCAR e MNAR validados.** Com apenas 3 MCAR e 1 MNAR confirmados estatisticamente, o treino com dados reais e fragil. Ver Secao "Aprendizados" abaixo.

---

## Parte D: Classificar MNAR como Focused vs Diffuse

### Motivacao (do review Zhou et al., 2024)

- **Focused MNAR:** f(M | X_m) — missing depende SO do valor faltante. Detectavel.
- **Diffuse MNAR:** f(M | X_m, X_o) — missing depende do valor faltante E de outros valores. Teoricamente indetectavel apenas com dados observados.

### O que fazer

Para cada dataset MNAR, verificar:
- Se remover X1-X4 do modelo MechDetect (tarefa Excluded) muda significativamente o AUC → Diffuse (X_o contribui)
- Se nao muda → Focused (so X_m importa)

Documentar na tese quais datasets sao focused vs diffuse e reportar accuracy separada.

### Resultado (2026-04-12)

Script `classificar_mnar.py` implementa a comparacao AUC_complete vs AUC_excluded.

**Dados reais:** Todos os 7 datasets MNAR → Diffuse (delta > 0.03 para todos). Consistente com a teoria: dados reais raramente apresentam MNAR puro/focused.

**Dados sinteticos:** 64 Focused, 36 Diffuse (de 100 amostras). Variantes self_logistic e quantile_threshold sao Focused por design; tail_censoring e self_noisy sao parcialmente Diffuse.

**Teste 5 (Focused recall > Diffuse recall) nao pode ser verificado** em dados reais porque nao ha datasets Focused. Em sinteticos, a diferenca e pequena mas na direcao esperada.

---

## Testes de Validacao

### Teste 1: Diversidade sintetica
Gerar dados com MissMecha. Verificar que datasets de variantes diferentes do mesmo mecanismo (ex: MCAR tipo 1 vs tipo 3) tem features discriminativas com valores diferentes mas features MechDetect similares (ambos devem ter AUC_complete ~0.5).

**Status: NAO EXECUTADO FORMALMENTE.** Os dados foram gerados mas o teste especifico de comparacao entre variantes nao foi rodado. Pode ser feito analisando o `X_features.csv` agrupado por variante.

### Teste 2: Validacao de rotulos
Rodar `validar_rotulos.py` nos 9 datasets atuais. Oceanbuoys deve ser flaggado como inconsistente (Little p < 0.05 e correlacao mask-Xi significativa).

**Status: PASSOU.** Oceanbuoys flaggado com Little p=0.000 e corr=0.333 (ambos datasets).

### Teste 3: Generalizacao sintetico → sintetico
Treinar com variantes 1-3 de cada mecanismo, testar com variantes 4-5. Se accuracy > 65%, as features generalizam entre variantes.

**Status: NAO EXECUTADO FORMALMENTE.** O treinamento usou GroupShuffleSplit que mistura variantes. Um teste hold-out por variante requer modificacao do pipeline. Resultado do treinamento geral: 76.7% accuracy.

### Teste 4: Volume de dados reais
Apos coleta, verificar:
- >= 10 datasets por mecanismo
- Todos passam validacao de rotulo
- GroupKFold com >= 8 folds por classe

**Status: NAO PASSOU.** 7/9/7 datasets coletados (meta era 10+). Apenas 3/4/1 passam validacao.

### Teste 5: Focused vs Diffuse
Para pelo menos 2 datasets MNAR, classificar como focused ou diffuse e verificar que focused MNAR tem melhor recall que diffuse.

**Status: PARCIAL.** 7 MNAR classificados, todos Diffuse. Sem Focused em dados reais para comparar recall.

---

## Aprendizados e Implicacoes para a Tese

### 1. MNAR e fundamentalmente dificil de validar

A grande maioria dos datasets MNAR da literatura (pima_insulin, adult_capitalgain, kidney_pot/sod) **nao apresenta evidencia estatistica de MNAR** quando analisados com os 3 testes. Isso nao significa que nao sao MNAR — significa que **MNAR e indetectavel com dados observados quando a taxa de missing e baixa** (limitacao teorica conhecida, ver Rubin 1976, Little & Rubin 2019).

Implicacao: a tese deve discutir esta limitacao e considerar que o classificador pode estar aprendendo "rotulos da literatura" ao inves de "mecanismos reais".

### 2. Cap de 10% missing reduz o sinal

O cap de missing rate em 10% (necessario para compatibilidade com dados sinteticos) remove a maior parte do sinal MNAR. Datasets como pima_insulin (48.7% missing original → 9.9% pos-cap) perdem a estrutura de auto-dependencia.

Implicacao: considerar experimentos com missing rate 10-30% nos sinteticos para medir o efeito.

### 3. Rotulos reais vs rotulos sinteticos

Dados sinteticos: rotulos sao **ground truth** (sabemos exatamente qual mecanismo foi usado). Dados reais: rotulos sao **hipoteses da literatura** que podem estar erradas.

Implicacao: a tese deve tratar accuracy em dados reais com cautela, e usar os testes de validacao como evidencia qualitativa.

### 4. LLM contribui positivamente pela primeira vez (rodada step03_final)

Na rodada final com oceanbuoys reclassificado, **LLM (Gemini) melhorou 5/7 modelos** em dados reais:
- SVM: +8.5pp (31.5% → 40.0%)
- KNN: +5.8pp (37.6% → 43.4%)
- MLP: +5.4pp (38.0% → 43.4%)
- Media: +3.1pp

Hipotese: com rotulos ruidosos e features estatisticas insuficientes para separar MCAR/MNAR, o raciocinio qualitativo do LLM oferece sinal complementar. Isso reforça a tese de que **LLMs sao mais uteis quando o sinal estatistico e fraco** — exatamente o cenario MCAR vs MNAR.

Implicacao: o STEP04 (LLM focado em MCAR vs MNAR) tem potencial real de contribuicao.

### 5. Rodada final (step03_final) — Metricas completas

Pipeline completa via `run_all.py --data real --experiment step03_final`:
- Baseline: 41.4% (RF melhor)
- ML+LLM (Gemini): **43.4%** (KNN/MLP melhores)
- 10/23 datasets validados (oceanbuoys agora CONSISTENTE como MAR)
- Features LLM: 14% importancia total (llm_pattern_clarity 2.3%, llm_mcar_vs_mnar 1.7%)
- Feature mais importante geral: X0_censoring_score (15.2%)
- Confusion matrix: MCAR recall 49%, MAR recall 41%, MNAR recall 38%
- Gap sintetico→real: 33pp (76.7% → 43.4%), causado por rotulos inconsistentes

Resultados detalhados em `Output/v2_improved/step03/STEP03_RESULTADOS.md`.

---

## Criterio de Conclusao

- [x] `gerador_v2.py` gera dados com multiplas variantes (12 variantes, 1200 datasets)
- [x] `validar_rotulos.py` funciona e flagga oceanbuoys (Little p=0.000, corr=0.333)
- [x] Oceanbuoys reclassificado como MAR (movido de processado/MCAR/ para processado/MAR/)
- [ ] >= 10 datasets reais por mecanismo coletados e validados — **5/11/7 coletados, 3/6/1 validados**
- [x] Datasets MNAR classificados como focused/diffuse (7 datasets, todos Diffuse)
- [ ] Testes 1-5 passam — **Teste 2 passa, Testes 1/3 nao executados formalmente, Teste 4/5 nao passam**
- [x] `run_all.py` atualizado para chamar `validar_rotulos.py` e `classificar_mnar.py`
- [x] `validate_labels.py` antigo removido
- [x] `requirements.txt` atualizado com `missmecha-py`
- [x] `CLAUDE.md` atualizado com novos scripts e comandos

---

## Proximos Passos Propostos

### STEP 03.1: Quarentena e retreino com dados limpos

1. ~~Mover oceanbuoys para MAR~~ — **FEITO** (step03_final)
2. Considerar mover hypothyroid_t4u para MNAR, cylinderbands_bladepressure para MAR
3. Remover datasets que testam como MCAR mas estao em MAR/MNAR (ou vice-versa)
4. Retreinar com apenas datasets validados (10/23)
5. Comparar accuracy sintetico vs real-limpo

### STEP 03.2: Teste formal de generalizacao entre variantes

1. Treinar com variantes 1-3, testar com 4-5 (hold-out por variante)
2. Mede se as features capturam o mecanismo geral, nao a variante especifica

### STEP 03.3: Features especificas para MNAR

1. Adicionar features que comparem distribuicao de X0_observado por quartis
2. Testar sensitivity analysis: remover subsets de X0 e medir impacto nas features
3. Considerar Mann-Whitney entre X0_observado e X0_imputado por faixas
