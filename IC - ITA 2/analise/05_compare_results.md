# Analise: compare_results.py

**Arquivo:** `Scripts/v2_improved/compare_results.py`
**Funcao:** Comparacao de resultados entre configuracoes de modelo

---

## Bugs

### BUG-CR1: Parser fragil de relatorio [MEDIO]

**Linhas 33-47:** O parser procura linhas com `=== NomeModelo ===` e `Acurácia:`. Porem, `train_model.py` tambem gera headers como `=== FEATURE IMPORTANCE (RandomForest) ===` e `=== CROSS-VALIDATION (5-fold) ===`. Estes serao parseados como nomes de modelo. Funciona por acidente (nao ha linha `Acurácia:` apos essas secoes), mas e fragil.

**Risco:** Qualquer mudanca no formato do relatorio que inclua linhas similares a acuracia nessas secoes quebrara o parser.

### BUG-CR2: Modelos faltantes entre configuracoes [MEDIO]

**Linhas 54-57:** `pd.DataFrame(all_results)` assume que todas as configuracoes tem o mesmo conjunto de modelos. Se uma config rodou 7 modelos e outra apenas 5, o DataFrame tera NaN. O grafico de barras plota NaN como altura zero, o que e enganoso.

### BUG-CR3: FileNotFoundError se diretorio de output nao existe [BAIXO]

**Linha 19:** `os.listdir(OUTPUT_BASE)` levanta `FileNotFoundError` se o diretorio de output ainda nao existe.

### BUG-CR4: Legenda incompleta [BAIXO]

**Linha 93:** `ax.axhline(...)` com `label="Random"` e adicionada apos a legenda ser construida, entao a linha de referencia nao aparece na legenda.
