# Analise: extract_features.py e Modulos de Features

**Arquivos:**
- `Scripts/v2_improved/extract_features.py`
- `Scripts/v2_improved/features/statistical.py`
- `Scripts/v2_improved/features/discriminative.py`

---

## Bugs Criticos

### BUG-EF1: Resume de checkpoint corrompe alinhamento features-labels [CRITICO]

**Arquivo:** `extract_features.py`, linhas 166-175

Ao resumir de um checkpoint, resultados parciais sao carregados por indice posicional. Porem, o CSV parcial foi salvo de um array `results` onde entradas sao colocadas no indice original da task (`results[idx] = ...`). A funcao `save_checkpoint()` filtra entradas `None` (linha 189), **removendo a informacao de indice**. No resume, a linha posicional `i` no CSV nao corresponde mais ao indice de task `i`.

**Exemplo:** Tasks nos indices 0, 2, 5 sao completadas. O CSV parcial tem 3 linhas (posicoes 0, 1, 2). No resume, linha 1 do CSV (originalmente indice 2) e carregada em `results[1]`, que e um arquivo completamente diferente.

**Impacto:** Corrupcao silenciosa de dados -- features de um arquivo sao associadas ao label errado.

### BUG-EF2: Modo `--test` so amostra classe MCAR [CRITICO]

**Arquivo:** `extract_features.py`, linha 145

`tasks = tasks[:50]` pega os primeiros 50 arquivos. Como tasks sao construidas iterando `DATASET_PATHS` dict (MCAR primeiro, depois MAR, depois MNAR), e cada classe tem 1000 arquivos, os primeiros 50 serao TODOS MCAR. Zero amostras MAR e MNAR.

**Impacto:** Modo teste completamente inutil para validar o pipeline.

**Correcao:** Usar amostragem estratificada ou intercalar classes antes de truncar.

### BUG-EF3: Dessincronia entre checkpoint set e results array [CRITICO]

**Arquivo:** `extract_features.py`, linhas 152-178

`processed_files` (set de caminhos do JSON) e `results` (do CSV) podem ficar dessincronizados se o write do CSV sucede mas o do JSON falha (ou vice-versa). O script pode pular arquivos sem resultados armazenados, ou re-processar arquivos cujos resultados ja estao no array mas em posicoes erradas.

---

## Bugs de Severidade Media

### BUG-EF4: Thread-safety de scikit-learn no ThreadPoolExecutor [MEDIO]

**Arquivo:** `extract_features.py`, linhas 200-216

`LogisticRegression.fit()` e `StandardScaler.fit_transform()` sao chamados dentro de `process_file` que roda em `ThreadPoolExecutor` com 100 workers. Scikit-learn nao e desenhado para uso concorrente de multiplas threads. Bibliotecas BLAS/LAPACK subjacentes podem nao ser thread-safe.

### BUG-EF5: MAX_WORKERS = 100 e excessivo [MEDIO]

**Arquivo:** `extract_features.py`, linha 45

100 threads e extremamente agressivo. Para chamadas de API LLM, isso dispara rate limits. Para trabalho CPU-bound do scikit-learn, contencao de threads degrada performance. 10-20 seria mais razoavel.

### BUG-EF6: dotenv carregado do diretorio errado [MEDIO/ALTO]

**Arquivo:** `extract_features.py`, linha 61

`SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` sobe dois niveis de `v2_improved/extract_features.py`, caindo em `Scripts/`. Mas CLAUDE.md diz que `.env` fica em `Scripts/v2_improved/`. Entao `load_dotenv` carrega de `Scripts/.env`, nao de `Scripts/v2_improved/.env`. API keys podem nao ser encontradas.

### BUG-EF7: Nome de modelo invalido ignorado silenciosamente [MEDIO]

**Arquivo:** `extract_features.py`, linha 55

Se `--model foo` e passado, `USE_LLM` vira `False` sem aviso. O usuario pode acreditar que esta rodando com features LLM quando nao esta.

### BUG-EF8: Checkpoint save pode ser pulado [MEDIO]

**Arquivo:** `extract_features.py`, linha 215

`if len(processed_files) % 20 == 0` so dispara quando o tamanho do set e exatamente divisivel por 20. Com ThreadPoolExecutor, multiplas futures podem completar quase simultaneamente, pulando de 19 para 21 e perdendo o save.

### BUG-EF9: Writes de checkpoint nao sao atomicos [MEDIO]

**Arquivo:** `extract_features.py`, linhas 184-194

JSON e CSV sao escritos sequencialmente. Crash entre os dois writes causa inconsistencia.

### BUG-EF10: Substituicao de Inf/NaN mascara problemas reais [MEDIO]

**Arquivo:** `extract_features.py`, linhas 193, 246

`.replace([np.inf, -np.inf], 0).fillna(0)` converte silenciosamente infinitos e NaN para 0. Se `log_pval_X1_mask` produz infinito (p-valor exatamente 0), substituir por 0 faz parecer que nao ha significancia -- o oposto da verdade.

---

## Problemas em features/discriminative.py

### BUG-DF1: AUC calculada nos dados de treino [ALTO]

**Arquivo:** `discriminative.py`, linhas 52-61

Regressao logistica e treinada em `X_scaled` e `roc_auc_score` e computada nos MESMOS dados. AUC in-sample sera inflada. Cross-validated AUC seria mais apropriada.

**Impacto na pesquisa:** Feature de AUC inflada pode contribuir sinal enganoso ao classificador downstream.

### BUG-DF2: "Little's proxy" nao e o teste de Little [MEDIO - naming]

**Arquivo:** `discriminative.py`, linhas 98-111

O "Little's proxy score" e a media de estatisticas KS comparando distribuicoes condicionais na missingness de X0. O verdadeiro teste de Little e um teste qui-quadrado baseado no algoritmo EM. A abordagem KS testa uma hipotese nula diferente. O nome da feature e enganoso.

### BUG-DF3: NaN em X1 nao e verificado [BAIXO]

**Arquivo:** `discriminative.py`, linha 70

`X1 = df["X1"].values` nao verifica NaN em X1. Se X1 tiver valores faltantes, correlacao ponto-biserial, diferenca de medias e Mann-Whitney produzirao NaN ou erros.

---

## Problemas em features/statistical.py

### BUG-SF1: Caso de valor unico observado retorna defaults [BAIXO]

**Arquivo:** `statistical.py`, linha 36

`len(X0_obs) > 1` captura o caso all-NaN mas tambem o caso de valor unico, retornando 0.5 para todas as features. Valor unico poderia ter tratamento diferente.

---

## Edge Cases Nao Tratados

- Nomes de colunas diferentes dos esperados (X0-X4) causam KeyError sem mensagem diagnostica
- DataFrames com menos de 2 linhas degenerao multiplas computacoes
- `StandardScaler` produz NaN se algum preditor e constante, causando falha em `LogisticRegression`
- Sem validacao de consistencia entre conjuntos de features (diferentes arquivos podem produzir keys diferentes, `pd.DataFrame` preenche com NaN -> 0)
