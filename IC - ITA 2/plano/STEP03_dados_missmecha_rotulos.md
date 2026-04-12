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

---

## Testes de Validacao

### Teste 1: Diversidade sintetica
Gerar dados com MissMecha. Verificar que datasets de variantes diferentes do mesmo mecanismo (ex: MCAR tipo 1 vs tipo 3) tem features discriminativas com valores diferentes mas features MechDetect similares (ambos devem ter AUC_complete ~0.5).

### Teste 2: Validacao de rotulos
Rodar `validar_rotulos.py` nos 9 datasets atuais. Oceanbuoys deve ser flaggado como inconsistente (Little p < 0.05 e correlacao mask-Xi significativa).

### Teste 3: Generalização sintetico → sintetico
Treinar com variantes 1-3 de cada mecanismo, testar com variantes 4-5. Se accuracy > 65%, as features generalizam entre variantes.

### Teste 4: Volume de dados reais
Apos coleta, verificar:
- >= 10 datasets por mecanismo
- Todos passam validacao de rotulo
- GroupKFold com >= 8 folds por classe

### Teste 5: Focused vs Diffuse
Para pelo menos 2 datasets MNAR, classificar como focused ou diffuse e verificar que focused MNAR tem melhor recall que diffuse.

---

## Criterio de Conclusao

- [ ] `gerador.py` (ou v2) gera dados com multiplas variantes MissMecha
- [ ] `validar_rotulos.py` funciona e flagga oceanbuoys
- [ ] Oceanbuoys removido ou reclassificado
- [ ] >= 10 datasets reais por mecanismo coletados e validados
- [ ] Datasets MNAR classificados como focused/diffuse
- [ ] Testes 1-5 passam
