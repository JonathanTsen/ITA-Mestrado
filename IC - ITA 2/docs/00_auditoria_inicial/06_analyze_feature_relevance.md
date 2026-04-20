# Analise: analyze_feature_relevance.py

**Arquivo:** `Scripts/v2_improved/analyze_feature_relevance.py`
**Funcao:** Analise de importancia e relevancia de features

---

## Bugs Criticos

### BUG-AFR1: Diretorio hardcoded nao existe no pipeline [CRITICO]

**Linha 36:** `OUTPUT_DIR = os.path.join(BASE_DIR, "Output", "v2_improved", "gemini-3-pro-preview")` e hardcoded para `"gemini-3-pro-preview"`. Mas `run_all.py` (linha 18) usa `"gemini-3-flash-preview"`. Este script falhara com `FileNotFoundError` ao tentar carregar `X_features.csv`.

**Correcao:** Aceitar `--model` como argumento CLI ou auto-detectar diretorios disponiveis.

### BUG-AFR2: Permutation importance nos dados de treino [CRITICO]

**Linhas 106-107, 131:** RandomForest para feature importance e permutation importance sao ambos treinados e avaliados no dataset INTEIRO `X, y`. Permutation importance medida nos mesmos dados usados para treino e inflada para features overfitadas.

**Correcao:** Usar train/test split; computar permutation importance no test set.

---

## Bugs de Severidade Media

### BUG-AFR3: Deteccao de baixa variancia excessivamente agressiva [MEDIO]

**Linhas 88-92:** Features com `nunique <= 3` sao marcadas como baixa variancia. Mas para um problema de classificacao com 3 classes, uma feature com 3 valores unicos pode ser extremamente informativa (imagine um preditor perfeito).

### BUG-AFR4: Coeficiente de variacao incorreto para medias negativas [MEDIO]

**Linha 83:** `"cv": X[col].std() / X[col].mean() if X[col].mean() != 0 else 0`

Deveria usar o valor absoluto da media: `X[col].std() / abs(X[col].mean())`. Com media negativa, o CV sera negativo, o que e sem significado.

### BUG-AFR5: Threshold de ranking RFE e vacuo [MEDIO]

**Linha 414:** `rfe_df[rfe_df["rfe_ranking"] > len(X.columns) // 2]`

Com 18 features e `n_features_to_select=15`, o ranking vai de 1 (selecionada) a 4 (mais eliminada). `> 9` (18//2) nao seleciona nada, tornando esse criterio completamente inutil.

### BUG-AFR6: Ablation study com parametros inconsistentes [BAIXO]

**Linhas 341, 352, 363:** Ablation usa `n_estimators=100` enquanto analise principal usa `n_estimators=200`. Comparacao nao e equitativa.

---

## Problemas de Design

### DESIGN-AFR1: Sem argumento `--model` [MEDIO]

Diferente dos outros scripts, nao aceita `--model` e hardcoda `"gemini-3-pro-preview"`. Inutilizavel para outras configuracoes.

### DESIGN-AFR2: RFE step=5 com n_features_to_select=15 [BAIXO]

**Linha 282:** Com 18 features totais, RFE remove apenas 3 features (18->15) em uma unica iteracao. O step=5 e irrelevante nesse cenario, e a analise RFE adiciona pouca informacao.

### DESIGN-AFR3: Hiperparametros RF inconsistentes entre scripts [MEDIO]

| Script | n_estimators |
|--------|-------------|
| train_model.py | 400 |
| analyze_feature_relevance.py (principal) | 200 |
| analyze_feature_relevance.py (ablation) | 100 |
| analyze_feature_relevance.py (RFE) | 100 |

Importancias de features diferem entre scripts, dificultando referencia cruzada.
