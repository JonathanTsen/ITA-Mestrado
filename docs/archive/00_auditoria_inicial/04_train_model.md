# Analise: train_model.py

**Arquivo:** `Scripts/v2_improved/train_model.py`
**Funcao:** Treinamento e avaliacao de 7 classificadores ML

---

## Bugs Criticos

### BUG-TM1: Crash na geracao de graficos per-class [CRITICO]

**Linhas 234-239:** `class_map = {"MCAR": "0", "MAR": "1", "MNAR": "2"}` acessa `r["report"]["0"]`, `r["report"]["1"]`, `r["report"]["2"]`.

Porem, `classification_report(output_dict=True)` do scikit-learn usa os valores REAIS dos labels como chaves. Como labels sao inteiros (0, 1, 2), as chaves no dict serao inteiros `0`, `1`, `2` -- NAO strings `"0"`, `"1"`, `"2"`.

**Resultado:** `r["report"][class_map[classe]]` levanta `KeyError` em tempo de execucao.

**Correcao:** Mudar para `class_map = {"MCAR": 0, "MAR": 1, "MNAR": 2}`.

---

## Bugs de Severidade Media

### BUG-TM2: Cross-validation no dataset completo vs holdout [MEDIO]

**Linhas 170-172:** Cross-validation e feita no dataset completo `X, y`, enquanto os modelos foram `.fit()` em `X_train`. O relatorio mistura dois paradigmas de avaliacao (holdout e full-data CV) sem deixar claro. Pode confundir o leitor.

**Nota:** Nao e data leakage (scikit-learn clona o estimador no CV), mas e uma inconsistencia metodologica.

### BUG-TM3: Divisao por zero possivel na feature importance [BAIXO]

**Linha 160:** `llm_importance/(llm_importance+stat_importance)` -- se todas as importancias sao zero, divide por zero.

---

## Problemas de Design

### DESIGN-TM1: Sem tratamento de NaN/Inf nos dados de entrada [MEDIO]

**Linhas 63-64:** Dados sao carregados e usados diretamente sem verificar NaN ou infinitos. Se `extract_features.py` produziu NaN (chamadas LLM falhadas, divisao por zero em features estatisticas), o treinamento produz resultados ruins silenciosamente ou crasha.

### DESIGN-TM2: Warnings suprimidas globalmente [BAIXO]

**Linha 31:** `warnings.filterwarnings("ignore")` silencia TODAS as warnings, incluindo convergencia de LogisticRegression e MLP, que podem indicar problemas reais de treinamento.

### DESIGN-TM3: GaussianNB com StandardScaler [BAIXO]

**Linhas 98-101:** GaussianNB e embrulhado em Pipeline com StandardScaler. Escalar features antes do GaussianNB muda as estimativas de variancia. A maioria dos praticantes usa GaussianNB sem scaling.

### DESIGN-TM4: Feature importance apenas do RandomForest [BAIXO]

**Linhas 143-147:** Importancia de features extraida apenas do RandomForest. Limita a analise a um unico modelo.

---

## Problema de Squeeze

### BUG-TM4: `squeeze("columns")` fragil para labels [BAIXO]

**Linha 64:** `pd.read_csv(Y_IN).squeeze("columns")` funciona se o CSV tem uma unica coluna. Se tiver multiplas colunas, retorna DataFrame silenciosamente ao inves de Series, causando erros inesperados downstream.
