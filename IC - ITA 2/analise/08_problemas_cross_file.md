# Analise: Problemas Cross-File

Problemas que envolvem interacao entre multiplos arquivos do projeto.

---

## Problemas Criticos

### CROSS-1: class_map com tipo errado em train_model.py [CRITICO]

`train_model.py` linha 234 usa `class_map = {"MCAR": "0", "MAR": "1", "MNAR": "2"}` (strings).
`extract_features.py` usa `LABEL_MAP = {"MCAR": 0, "MAR": 1, "MNAR": 2}` (inteiros).

`classification_report(output_dict=True)` do scikit-learn usa os tipos reais dos labels como chaves. O resultado e um `KeyError` quando `train_model.py` tenta acessar `report["0"]` ao inves de `report[0]`.

### CROSS-2: analyze_feature_relevance.py aponta para modelo inexistente [CRITICO]

`analyze_feature_relevance.py` hardcoda `"gemini-3-pro-preview"`.
`run_all.py` usa `["none", "gemini-3-flash-preview"]`.

Os nomes nao coincidem -- o script de analise nao consegue encontrar os dados produzidos pelo pipeline.

---

## Problemas de Severidade Media

### CROSS-3: Hiperparametros RF inconsistentes entre scripts [MEDIO]

| Script | n_estimators | Contexto |
|--------|-------------|----------|
| `train_model.py` | 400 | Treinamento principal |
| `analyze_feature_relevance.py` | 200 | Analise de importancia |
| `analyze_feature_relevance.py` (ablation) | 100 | Ablation study |
| `analyze_feature_relevance.py` (RFE) | 100 | Recursive Feature Elimination |

Feature importances diferem significativamente entre scripts, tornando comparacoes cruzadas invalidas.

### CROSS-4: Localizacao do .env inconsistente [MEDIO]

`CLAUDE.md` diz: "API keys go in a `.env` file in `Scripts/v2_improved/`"
`extract_features.py` carrega de: `Scripts/.env` (sobe dois diretorios)

Se o usuario segue a documentacao e coloca `.env` em `v2_improved/`, as API keys nao serao encontradas.

### CROSS-5: Modo --test gera dados enviesados em todo o pipeline [MEDIO]

O fluxo `run_all.py --test` propagara o flag `--test` para `extract_features.py`, que pega apenas os primeiros 50 arquivos (todos MCAR). O `train_model.py` entao treina em dados de uma unica classe, produzindo resultados sem sentido. O modo teste nao valida nada do pipeline real.

### CROSS-6: y_labels.csv acesso inconsistente [BAIXO]

`train_model.py` usa `pd.read_csv(Y_IN).squeeze("columns")` (assume coluna unica).
`analyze_feature_relevance.py` usa `pd.read_csv(Y_PATH)["label"]` (acesso explicito por nome).

Abordagens diferentes podem falhar em cenarios diferentes.

---

## Problemas Metodologicos

### METHOD-1: Avaliacao mista (holdout + full-data CV) em train_model.py

Modelos sao treinados em `X_train` (75%) e avaliados em `X_test` (25%), mas cross-validation e feita no dataset completo `X, y`. O relatorio mistura os dois paradigmas sem diferenciar claramente.

### METHOD-2: AUC como feature e calculada in-sample

`discriminative.py` treina regressao logistica e calcula AUC nos mesmos dados. Essa feature inflada e entao usada pelo classificador de mecanismo, propagando o overfitting para o pipeline completo.

### METHOD-3: Falta de seed global consistente

Nao ha uma seed global unica que garanta reprodutibilidade completa do pipeline. Cada componente tem suas proprias seeds ou nenhuma.

---

## Diagrama de Dependencia com Problemas

```
gerador.py (BUG-G1,G2,G3,G4, STAT-G1,G2)
    |
    v
extract_features.py (BUG-EF1*,EF2*,EF3*,EF4,EF5,EF6,EF7,EF8,EF9,EF10)
  |-- features/statistical.py (BUG-SF1)
  |-- features/discriminative.py (BUG-DF1*,DF2,DF3)
  |-- llm/extractor_v2.py (BUG-LLM1*,LLM2*,LLM3,LLM4*,LLM5,LLM6)
    |
    v
train_model.py (BUG-TM1*,TM2,TM3)
    |
    v
compare_results.py (BUG-CR1,CR2,CR3)

analyze_feature_relevance.py (BUG-AFR1*,AFR2*,AFR3,AFR4,AFR5)

run_all.py (BUG-RA1,RA2,RA3)

* = Critico ou Alto
```
