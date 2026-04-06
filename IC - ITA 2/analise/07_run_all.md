# Analise: run_all.py

**Arquivo:** `Scripts/v2_improved/run_all.py`
**Funcao:** Orquestrador do pipeline completo

---

## Problemas

### BUG-RA1: Falha de compare_results.py nao verificada [MEDIO]

**Linha 66:** `subprocess.run(cmd, cwd=SCRIPT_DIR)` para compare_results.py nao verifica `returncode`. Se a comparacao falha, o pipeline ainda reporta sucesso.

### BUG-RA2: analyze_feature_relevance.py nao esta no pipeline [BAIXO]

O pipeline roda extract -> train -> compare, mas nunca roda `analyze_feature_relevance.py`. Provavelmente intencional, mas vale notar.

### BUG-RA3: Output de subprocessos nao capturado [BAIXO]

**Linhas 44, 51, 66:** `subprocess.run()` nao captura stdout/stderr. Se um subprocesso falha, a mensagem de erro e impressa misturada com o output do orquestrador.

### BUG-RA4: Lista MODELS hardcoded [BAIXO]

**Linha 18:** Lista de modelos e hardcoded. Adicionar novos modelos requer editar o script ao inves de passar como argumento CLI.
