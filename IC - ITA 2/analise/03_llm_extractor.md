# Analise: llm/extractor_v2.py

**Arquivo:** `Scripts/v2_improved/llm/extractor_v2.py`
**Funcao:** Extrator de features via LLM com analise de segunda ordem

---

## Bugs e Erros de Logica

### BUG-LLM1: Thread-safety do cache [ALTO]

**Linha 82:** `self._cache` e um `dict` acessado concorrentemente de um `ThreadPoolExecutor` com ate 100 workers. Dicts Python nao sao thread-safe para leitura/escrita concorrente. Pode causar perda de entradas de cache ou, em casos raros, `RuntimeError`.

**Correcao:** Usar `threading.Lock` ao redor do acesso ao cache.

### BUG-LLM2: 100 chamadas LLM simultaneas causam rate limiting [ALTO]

No `extract_features.py`, `MAX_WORKERS=100` threads todas chamando a API LLM simultaneamente. A maioria das APIs tem limites muito abaixo de 100 requisicoes concorrentes. Isso causa rate limiting massivo, e como nao ha backoff (BUG-LLM3), a maioria das chamadas falha e usa defaults.

**Impacto:** Degrada significativamente a qualidade do modelo.

### BUG-LLM3: Sem backoff entre retries [MEDIO]

**Linha 315:** `_call_llm_with_retry` faz 3 tentativas sem delay entre elas. Para erros 429 (rate limit), as retries sao consumidas instantaneamente. Nao ha backoff exponencial, nem distincao entre erros transientes e permanentes.

### BUG-LLM4: Fallback silencioso mascara falha total da API [ALTO]

**Linhas 350-352:** Na falha final, `LLMAnalysisV2().to_feature_dict()` retorna defaults neutros com apenas um print de aviso. Se a API key esta errada, TODOS os arquivos recebem features default silenciosamente. O caller nao tem como saber que todas as features LLM sao defaults.

**Correcao:** Manter contador de fallbacks e reportar no final. Considerar falha apos threshold de fallbacks.

### BUG-LLM5: `corr_X1_mask` e `X1_mean_diff` condicionalmente ausentes do JSON [MEDIO]

**Linhas 158-167:** Se `np.std(X1) == 0` ou `np.std(mask) == 0`, essas chaves nao sao adicionadas ao `stats_dict`. O LLM recebe um JSON incompleto sem indicacao de que campos estao ausentes, enquanto o prompt faz referencia a esses campos.

### BUG-LLM6: Logica de consistencia MNAR pode estar invertida [MEDIO]

**Linhas 201-203:** O comentario diz "MNAR com valores altos missing: mean_dev > 0, skew > 0". Porem, para dados truncados a esquerda (valores altos removidos), o skew da distribuicao restante e tipicamente negativo, nao positivo. A metrica de consistencia pode estar avaliando incorretamente.

---

## Problemas de API

### API-1: Sem validacao de API key na inicializacao [BAIXO]

**Linhas 88-99:** `os.getenv()` pode retornar `None`. O cliente LangChain e inicializado com `api_key=None`, falhando apenas na invocacao. Validacao antecipada economizaria tempo de debug.

### API-2: Sem distincao entre erros de parse e erros de API [MEDIO]

O `except Exception` na linha 349 conflata erros de rede, auth, rate limit, parse JSON e validacao Pydantic. Tipos diferentes de erro justificam estrategias diferentes de retry.

---

## Problemas de Cache

### CACHE-1: Cache e apenas em memoria, nao persistido [MEDIO]

O `_cache` dict vive apenas na memoria do processo. Cada execucao re-consulta o LLM para todos os arquivos. Com 3000 arquivos a ~$0.001-0.01 por chamada, isso despertica dinheiro significativo. Se o processo crasha no arquivo 2999, o cache e perdido.

### CACHE-2: Cache key nao inclui nome do modelo [BAIXO]

Se o mesmo `LLMFeatureExtractorV2` fosse reutilizado com modelos diferentes, o cache retornaria resultados errados. Risco baixo no uso atual, mas design fragil.

---

## Problemas de Schema Pydantic

### PYDANTIC-1: Confidences nao somam 1.0 [MEDIO]

**Linhas 44-46:** `mcar_confidence`, `mar_confidence`, `mnar_confidence` sao individualmente restritos a [0,1] mas nao ha `model_validator` garantindo que somam ~1.0. O prompt diz "devem somar ~1.0" mas o LLM frequentemente ignora isso.

**Correcao:** Adicionar `@model_validator(mode='after')` que normaliza as tres confidences.

### PYDANTIC-2: Defaults criam bias no fallback [BAIXO]

Defaults: `mcar_confidence=0.33, mar_confidence=0.33, mnar_confidence=0.34`. `mnar_confidence` e ligeiramente maior. A cada falha de LLM, o fallback envisa levemente para MNAR.

---

## Problemas de Prompt

### PROMPT-1: Exemplo JSON mostra valores neutros/default [MEDIO]

**Linhas 293-302:** O exemplo mostra todos valores neutros (0.5, 0.33, etc.). LLMs sao ancorados por exemplos. Isso envisa o LLM a retornar valores proximos dos defaults. Melhor mostrar dois exemplos contrastantes.

### PROMPT-2: LLM essencialmente executa if-else hardcoded [MEDIO]

O prompt inclui `mar_combined_evidence`, `mnar_combined_evidence`, `evidence_conflict_score` que ja sao computacoes deterministicas. As "INSTRUCOES DE RACIOCINIO" (linhas 283-286) dizem ao LLM exatamente quais regras seguir, tornando a chamada LLM essencialmente um if-else muito caro.

### PROMPT-3: Thresholds no prompt sao arbitrarios e nao calibrados [MEDIO]

**Linhas 283-286:** Thresholds como `mar_combined_evidence > 0.3`, `mnar_combined_evidence > 0.2` sao hardcoded sem justificativa teorica ou empirica.

### PROMPT-4: Thresholds de deteccao MNAR sao muito pequenos [BAIXO]

**Linha 265:** `X0_mean_deviation > 0.005, X0_obs_skew > 0.02` -- thresholds tao pequenos que disparam em ruido amostral qualquer, podendo causar over-classificacao como MNAR.

---

## Problemas de Parsing JSON

### PARSE-1: Regex guloso para JSON nao-fenced [BAIXO]

**Linha 342:** `re.search(r'\{[\s\S]*\}', raw)` e guloso e captura do primeiro `{` ao ultimo `}` na resposta inteira. Se o LLM produz texto com multiplas estruturas JSON-like, pode capturar lixo.

### PARSE-2: `response.content` pode ser None [BAIXO]

**Linha 323:** Resposta vazia do LLM faz `str(None)` = `"None"`, que falha no parse JSON. Despertica um retry.

---

## Edge Cases

- DataFrame com todos X0 faltantes: tratado mas produz `stats_dict` muito esparso
- DataFrame sem valores faltantes em X0: LLM recebe quadro confuso
- Datasets muito pequenos (< 10 linhas): estatisticas instáveis
- Colunas nao-numericas: nenhuma verificacao de tipo
