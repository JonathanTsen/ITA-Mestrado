# Step 3: Self-Consistency — Implementacao

**Data:** 2026-04-20
**Status:** Implementado e testado (sintetico MCAR, 50 arquivos)

## Arquivos Criados

### `llm/self_consistency.py`
Nova classe `SelfConsistencyExtractor` com 5 perspectivas especializadas e agregacao CISC.

**Perspectivas:**
1. **Statistical** — analisa apenas padroes numericos, ignora dominio
2. **Domain** — especialista no dominio, foco em como X0 e medido/coletado
3. **Process** — engenheiro de dados, foco no processo de registro
4. **Adversarial** — anti-MAR, argumenta contra a hipotese default
5. **Censoring** — especialista em censura/truncamento

**Features geradas (8):**

| Feature | Descricao | Range |
|---------|-----------|-------|
| `llm_sc_domain_prior` | Mecanismo vencedor (MCAR=0, MAR=0.5, MNAR=1) | {0, 0.5, 1} |
| `llm_sc_confidence` | Confianca agregada CISC | [0, 1] |
| `llm_sc_agreement` | Fracao de perspectivas que concordam | [0.2, 1.0] |
| `llm_sc_vote_mcar` | Proporcao de votos MCAR | [0, 1] |
| `llm_sc_vote_mar` | Proporcao de votos MAR | [0, 1] |
| `llm_sc_vote_mnar` | Proporcao de votos MNAR | [0, 1] |
| `llm_sc_stats_consistency` | Media da stats_consistency das perspectivas | [0, 1] |
| `llm_sc_surprise` | Media do surprise_factor | [0, 1] |

**Agregacao CISC (Confidence-Informed Self-Consistency):**
- Cada perspectiva vota em um mecanismo com peso = sua confianca
- Mecanismo vencedor = maior soma ponderada
- Confianca agregada = proporcao do vencedor no total de votos
- Chamadas paralelas via ThreadPoolExecutor (5 perspectivas simultaneas)

## Arquivos Modificados

### `utils/args.py`
- Adicionado `"self_consistency"` a `LLM_APPROACHES`

### `extract_features.py`
- Import de `SelfConsistencyExtractor`
- Flag `USE_SC = USE_LLM_API and LLM_APPROACH == "self_consistency"`
- Inicializacao do extractor quando `USE_SC`
- Chamada em `extract_all_features()` (slot 8)
- `MAX_WORKERS = 10` para self_consistency (cada arquivo faz 5 chamadas internas)
- `USE_CAAFE` habilitado junto com self_consistency

### `llm/__init__.py`
- Import de `SelfConsistencyExtractor` e `get_sc_fallback_features`

## Correcao: Artefato de Q-rates por Mediana

### Problema Encontrado

O `_compute_stats` original usava `df["X0"].fillna(df["X0"].median())` para calcular quartis.
Isso concentrava TODOS os valores ausentes no quartil da mediana (Q2/Q3), gerando Q-rates
extremamente desiguais (ex: Q3=33%, resto=0%) mesmo para dados MCAR.

**Impacto:** O LLM via "Q3=33%, resto=0%" e classificava como MNAR em 100% dos casos MCAR.

### Solucao Implementada (3 camadas)

1. **Regressao linear**: `_estimate_x0()` usa X0 ~ X1+X2+X3+X4 nos dados observados para
   predizer X0 nas linhas ausentes. Eficaz quando X0 e Xi sao correlacionados (dados reais).

2. **Threshold R² com fallback uniforme**: Quando R² < 0.1 (variáveis independentes), a
   regressao produz predicoes coladas na media — mesmo artefato da mediana. Nesses casos,
   os Q-rates de X0 sao setados para a taxa de missing global (= hipotese nula MCAR,
   distribuicao uniforme). Nos prompts, marcados como "UNRELIABLE".

3. **Q-rates por X1 quartil**: Novo indicador sempre confiavel (X1 nunca tem missing).
   - Uniforme → nao ha evidencia de MAR
   - Variavel → evidencia de MAR
   - Usado como PRIMARY evidence nos prompts

### Validacao com Dados Reais

| Dataset | R² | X0 Q-rates | Comportamento |
|---------|-----|------------|---------------|
| MCAR_autompg (horsepower) | 0.89 | 2.3/1.8/2.0/0.0 | R² alto, Q-rates reais usados |
| MAR_airquality (ozone) | 0.39 | 2.6/8.1/20.5/7.7 | R² ok, sinal de MNAR/MAR preservado |
| MAR_colic (resprate) | 0.20 | 1.2/11.6/14.0/10.9 | R² ok, Q-rates informativos |
| MNAR_adult (capitalgain) | 0.006 | uniforme | R²<0.1, fallback uniforme (correto) |

Dados sinteticos: R² sempre < 0.01 (variaveis independentes por construcao) → Q-rates
sempre uniformes → LLM depende apenas de X1 Q-rates, correlacoes e skewness.

### Arquivos Corrigidos

- `llm/self_consistency.py` — `_compute_stats()`, `_estimate_x0()` (novo), todos os prompts
- `llm/context_aware.py` — mesma correcao em `_compute_stats()`, `_estimate_x0()` (novo), todos os prompts

### Resultado da Correcao

Teste com 50 arquivos MCAR sinteticos:

| Metrica | Antes (artefato) | Depois (corrigido) |
|---------|------|--------|
| MCAR classificado como MCAR | 0/50 (0%) | **41/50 (82%)** |
| MCAR classificado como MAR | 0/50 | 4/50 (8%) |
| MCAR classificado como MNAR | 50/50 (100%) | 5/50 (10%) |
| vote_mcar medio | 0.003 | **0.642** |
| agreement medio | 0.996 | 0.676 |

## Como Executar

```bash
cd "ITA-Mestrado/IC - ITA 2/Scripts/v2_improved"

# Extrair features com self-consistency (dados sinteticos)
uv run python extract_features.py --model gemini-3-flash-preview --data sintetico \
    --llm-approach self_consistency --experiment step3_sc

# Extrair features com self-consistency (dados reais)
uv run python extract_features.py --model gemini-3-flash-preview --data real \
    --llm-approach self_consistency --experiment step3_sc

# Treinar modelos
uv run python train_model.py --model gemini-3-flash-preview --data sintetico \
    --experiment step3_sc

# Modo teste (50 arquivos)
uv run python extract_features.py --model gemini-3-flash-preview --data sintetico \
    --llm-approach self_consistency --experiment test_sc --test
```

## Limitacoes Conhecidas

1. **Dados sinteticos com variaveis independentes**: Quando X0 e X1-X4 sao independentes
   (como em MCAR sintetico), a regressao tem R² ~ 0, e os Q-rates de X0 sao pouco
   informativos. O LLM depende de X1 Q-rates e correlacoes nesses casos.

2. **Custo de API**: 5 chamadas por dataset (vs 3 do context_aware). Com gemini-3-flash,
   custo total ~$11.50 para 23 datasets x 50 bootstraps.

3. **Tempo**: ~4-7 min para 50 arquivos sinteticos (chamadas paralelas). Pipeline completo
   estimado em ~1-2h para dados sinteticos (1200 arquivos).

## Referencias

- Wang et al. (2022) "Self-Consistency Improves CoT Reasoning" (NeurIPS)
- "Confidence Improves Self-Consistency in LLMs" (ACL Findings 2025)
- Du et al. (2023) "Improving Factuality through Multiagent Debate" (ICML 2024)
