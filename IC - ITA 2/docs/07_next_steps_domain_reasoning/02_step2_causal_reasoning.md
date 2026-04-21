# Step 2: Causal Reasoning Prompt (Raciocinio Causal Estruturado)

**Prioridade:** MEDIA (requer reescrita do prompt core, complementa Step 1)
**Estimativa de impacto:** +5-8pp accuracy sobre Step 1
**Custo API:** ~2x (duas chamadas em vez de uma: DAG + classificacao)
**Arquivos a modificar:** `llm/context_aware.py` (prompt template + nova funcao)

## Problema que Resolve

O prompt atual pede uma classificacao **direta**: "qual mecanismo e mais provavel?" Isso permite que a LLM "salte" para a resposta sem raciocinar sobre a **cadeia causal** que gera o missing.

Pesquisa recente (IEEE TKDE 2025, arxiv 2504.10397) mostra que LLMs podem construir grafos causais (DAGs) com precisao comparavel a especialistas humanos quando guiadas com prompts estruturados.

## Solucao Proposta

### Substituir o prompt de classificacao direta por um prompt de 2 etapas

**Etapa 1: Elicitacao do DAG Causal (novo)**

```
## ETAPA 1: CONSTRUA O MODELO CAUSAL DA MISSINGNESS

Antes de classificar o mecanismo, construa um mini-diagrama causal.
Liste TODAS as possiveis causas para os dados de {x0_variable} estarem faltando:

Para cada causa potencial, classifique:
- **Tipo A (MCAR)**: Causa nao relacionada ao valor de X0 nem a X1-X4
  (ex: falha de equipamento, erro de digitacao, dado nao transferido)
- **Tipo B (MAR)**: Causa que depende de uma variavel OBSERVADA (X1-X4)
  (ex: medico pede exame com base em outro resultado)
- **Tipo C (MNAR)**: Causa que depende do PROPRIO VALOR de X0
  (ex: teste nao pedido porque valor esperado e normal/baixo)

Retorne APENAS um JSON valido:

```json
{
  "possible_causes": [
    {
      "cause": "descricao da causa",
      "type": "A|B|C",
      "plausibility": 0.0-1.0,
      "depends_on": "nenhuma|X1|X2|X3|X4|X0"
    }
  ],
  "most_plausible_cause": "descricao",
  "most_plausible_type": "A|B|C"
}
```
```

**Etapa 2: Classificacao informada pelo DAG (revisada)**

```
## ETAPA 2: CLASSIFIQUE O MECANISMO

Com base na sua analise causal anterior:

Causas identificadas:
{causas_formatadas_da_etapa_1}

Causa mais plausivel: {most_plausible_cause} (Tipo {most_plausible_type})

Agora, considerando TAMBEM as evidencias estatisticas:
{statistics_section}

As estatisticas sao CONSISTENTES com a causa mais plausivel?
- Se a causa e Tipo A (MCAR): taxa de missing deve ser uniforme entre quartis
- Se a causa e Tipo B (MAR): deve haver correlacao mask-Xi significativa
- Se a causa e Tipo C (MNAR): taxa de missing deve variar entre quartis de X0

Se as estatisticas CONTRADIZEM a causa mais plausivel, reconsidere.

Retorne APENAS um JSON valido:

```json
{
  "domain_mechanism_prior": "MCAR|MAR|MNAR",
  "domain_confidence": 0.5,
  "stats_consistent_with_domain": 0.5,
  "surprise_factor": 0.0,
  "causal_reasoning": "explicacao de como o DAG levou a esta classificacao",
  "stats_agreement": "concordam|contradizem|inconclusivos"
}
```
```

### Por que 2 etapas em vez de 1?

1. **Decomposicao do problema**: pesquisa mostra que LLMs raciocinam melhor quando o problema e decomposto (Chain-of-Thought, arxiv 2201.11903)
2. **Verificacao cruzada**: a Etapa 2 verifica se as estatisticas concordam com o DAG causal, criando um loop de validacao
3. **Explicabilidade**: o DAG causal gera uma justificativa auditavel para cada classificacao
4. **Reducao do MAR bias**: a LLM precisa NOMEAR especificamente qual variavel Xi causa o missing (Tipo B), nao pode simplesmente dizer "MAR" sem justificar

## Implementacao Tecnica

### Novo metodo em `context_aware.py`

```python
def _extract_causal_dag(self, metadata: dict, stats: dict) -> dict:
    """Etapa 1: Elicita o DAG causal da missingness."""
    prompt = self._build_causal_prompt(metadata, stats)
    response = self._call_llm(prompt)
    return self._parse_causal_response(response)

def _classify_with_dag(self, metadata: dict, stats: dict, dag: dict) -> dict:
    """Etapa 2: Classifica o mecanismo usando o DAG como contexto."""
    prompt = self._build_classification_prompt(metadata, stats, dag)
    response = self._call_llm(prompt)
    return self._parse_classification_response(response)

def extract_features(self, df: pd.DataFrame, filename: str, ...) -> dict:
    """Pipeline completo: DAG -> Classificacao -> Features."""
    stats = self._compute_stats(df)
    metadata = self._load_metadata(filename)
    
    # Etapa 1: Elicitar DAG
    dag = self._extract_causal_dag(metadata, stats)
    
    # Etapa 2: Classificar com DAG
    classification = self._classify_with_dag(metadata, stats, dag)
    
    # Combinar features
    features = self._combine_features(classification)
    
    # NOVA feature: tipo da causa mais plausivel
    cause_type_map = {"A": 0.0, "B": 0.5, "C": 1.0}
    features["llm_ctx_cause_type"] = cause_type_map.get(
        dag.get("most_plausible_type", "B"), 0.5
    )
    
    # NOVA feature: numero de causas identificadas (complexidade causal)
    features["llm_ctx_n_causes"] = min(len(dag.get("possible_causes", [])), 5) / 5.0
    
    # NOVA feature: concordancia stats-DAG
    agreement_map = {"concordam": 1.0, "inconclusivos": 0.5, "contradizem": 0.0}
    features["llm_ctx_stats_agreement"] = agreement_map.get(
        classification.get("stats_agreement", "inconclusivos"), 0.5
    )
    
    return features
```

### Novas Features Geradas

| Feature | Descricao | Range |
|---------|-----------|-------|
| `llm_ctx_cause_type` | Tipo da causa mais plausivel (A=0, B=0.5, C=1.0) | [0, 1] |
| `llm_ctx_n_causes` | Complexidade causal normalizada (n_causas / 5) | [0, 1] |
| `llm_ctx_stats_agreement` | Stats concordam com DAG causal | {0, 0.5, 1} |

Total de features LLM: 6 (existentes) + 3 (novas) = **9 features LLM**.

## Como Executar

```bash
cd "ITA-Mestrado/IC - ITA 2/Scripts/v2_improved"

# 1. Extrair features com novo prompt causal (requer Step 1 ja implementado)
uv run python extract_features.py --model gemini-3-flash-preview --data real \
    --llm-approach context_aware --experiment step2_causal

# 2. Treinar modelos
uv run python train_model.py --model gemini-3-flash-preview --data real \
    --experiment step2_causal

# 3. Analise forense
uv run python forensic_analysis.py --experiment step2_causal --data real

# 4. Comparar com Step 1
uv run python compare_results.py --data real
```

## Como Validar

### Verificacoes Qualitativas

1. **Inspecionar DAGs gerados**: Para cada dataset, verificar se as causas listadas fazem sentido
2. **MNAR_pima_insulin**: O DAG deve incluir causa Tipo C ("teste nao pedido quando glicose normal")
3. **MCAR_autompg_horsepower**: O DAG deve incluir causa Tipo A ("erro no registro")
4. **MAR_oceanbuoys_airtemp**: O DAG deve incluir causa Tipo B ("sensor falha com umidade alta")

### Verificacoes Quantitativas

| Metrica | Step 1 (expected) | Target Step 2 |
|---------|-------------------|---------------|
| Accuracy GKF-5 | ~60% | 65%+ |
| F1-macro | ~55% | 60%+ |
| MCAR acc LODO | ~45% | 50%+ |
| MNAR acc LODO | ~45% | 50%+ |
| stats_agreement = "concordam" | — | >60% dos datasets |

## Riscos

| Risco | Probabilidade | Mitigacao |
|-------|---------------|-----------|
| 2 chamadas = 2x custo API | Certo | Usar gemini-3-flash (barato) |
| LLM gera DAG inconsistente | Media | Parsing robusto + fallback para prompt original |
| Etapa 2 ignora o DAG | Baixa | Incluir causas formatadas explicitamente no prompt |
| Prompt muito longo | Baixa | DAG e ~100-200 tokens |

## Fundamentacao Cientifica

### Dual-Expert Verification (arxiv 2504.10397)
Metodologia onde um modelo gera dependencias e outro verifica inconsistencias. Nosso pipeline adapta: Etapa 1 gera o DAG, Etapa 2 verifica consistencia com as estatisticas.

### CARE Framework (arxiv 2511.16016)
Demonstra que LLMs podem ser transformados em "especialistas em raciocinio causal" via prompt engineering sem fine-tuning. Nosso prompt de DAG causal se inspira nesta abordagem.

### Chain-of-Thought Decomposition (arxiv 2201.11903)
Meta-analise mostra que decompor problemas em sub-etapas melhora significativamente a qualidade do raciocinio em LLMs. Nosso pipeline de 2 etapas implementa esta ideia.
