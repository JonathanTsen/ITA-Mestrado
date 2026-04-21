# Step 1: Few-Shot com Exemplos Canonicos + Tipologia MNAR

**Prioridade:** ALTA (menor esforco, maior impacto esperado)
**Estimativa de impacto:** +4-6pp accuracy, reducao significativa do MAR bias
**Custo API:** Nenhum adicional (mesmo numero de chamadas)
**Arquivos a modificar:** `llm/context_aware.py` (prompt template)

## Problema que Resolve

O prompt atual e **zero-shot**: a LLM nao recebe nenhum exemplo de como classificar mecanismos. Resultado:
- A LLM usa seus priors de treinamento, que favorecem MAR em contextos clinicos
- Nao tem calibracao: nao sabe o que "parece MCAR" vs "parece MNAR" neste contexto especifico
- Nao conhece a tipologia de subtipos (MNAR focused vs diffuse, censoring vs selection)

## Solucao Proposta

### 1.1 Adicionar 3 Exemplos Canonicos ao Prompt

Incluir no prompt 3 exemplos reais (1 por classe) que a LLM pode usar como referencia:

```
## EXEMPLOS DE REFERENCIA

### Exemplo MCAR (Missing Completely At Random)
Dominio: Engenharia automotiva
Variavel: Potencia do motor (hp)
Contexto: Valores ausentes por falha aleatoria no registro do banco de dados.
  Nao ha razao tecnica para que carros de alta ou baixa potencia tenham mais dados faltantes.
Evidencias tipicas: taxa de missing uniforme entre quartis de X0, sem correlacao mask-Xi.

### Exemplo MAR (Missing At Random)
Dominio: Oceanografia
Variavel: Temperatura do ar (celsius)
Contexto: Sensores de temperatura falham mais em condicoes de alta umidade (X1).
  A probabilidade de missing depende de outra variavel observada, nao da temperatura em si.
Evidencias tipicas: correlacao significativa mask-X1, taxa de missing NAO depende dos quartis de X0.

### Exemplo MNAR (Missing Not At Random)
Dominio: Economia do trabalho
Variavel: Salario (USD/hora)
Contexto: Mulheres que nao trabalham nao reportam salario. O valor esta ausente PORQUE
  seria zero ou muito baixo — o missing depende do proprio valor nao-observado.
Evidencias tipicas: taxa de missing concentrada em quartis especificos de X0,
  distribuicao observada truncada ou assimetrica.
```

### 1.2 Adicionar Tipologia de MNAR

O MNAR tem subtipos que a LLM precisa conhecer para classificar melhor:

```
## TIPOLOGIA DE MNAR

MNAR nao e monolitico. Existem subtipos distintos:

1. **MNAR por Censura (Censoring)**: valores abaixo/acima de um limite nao sao registrados.
   Exemplo: concentracoes quimicas abaixo do limite de deteccao do equipamento.
   Evidencia: missing concentrado em uma cauda da distribuicao.

2. **MNAR por Selecao (Selection)**: a decisao de coletar o dado depende do valor esperado.
   Exemplo: teste de insulina nao e pedido quando glicose parece normal.
   Evidencia: missing correlacionado com o proprio valor latente de X0.

3. **MNAR Difuso**: o missing depende de X0 mas tambem de outras variaveis.
   Exemplo: pacientes com doenca grave (X0 alto E X1 alto) nao completam o estudo.
   Evidencia: interacao entre X0 e Xi na mascara de missing.
```

### 1.3 Adicionar Anti-Bias Instruction

Instrucao explicita para combater o MAR bias:

```
## INSTRUCAO IMPORTANTE

ATENCAO: Nao assuma que MAR e o mecanismo mais provavel por padrao.
Em datasets reais:
- ~30% sao MCAR (dados faltam por falhas tecnicas, sem padrao)
- ~40% sao MAR (missing depende de outra variavel observada)
- ~30% sao MNAR (missing depende do proprio valor nao-observado)

Antes de classificar como MAR, pergunte-se:
1. Ha uma variavel observada ESPECIFICA (X1-X4) que CAUSA o missing? Qual?
2. Se nao consegue identificar uma variavel especifica, considere MCAR.
3. Se o missing parece depender do PROPRIO valor de X0, considere MNAR.
```

## Implementacao Tecnica

### Arquivo: `llm/context_aware.py`

**Funcao a modificar:** `_build_real_prompt()` (linhas 244-311)

**Mudanca:** Inserir os 3 blocos acima (exemplos, tipologia, anti-bias) ANTES da secao "## TASK" no prompt.

```python
def _build_real_prompt(self, metadata: dict, stats: dict, filename: str) -> str:
    # ... codigo existente para construir secoes DATASET, MISSING CONTEXT, STATISTICS ...
    
    # NOVO: Adicionar secao de exemplos canonicos
    few_shot_section = """
## EXEMPLOS DE REFERENCIA
[... conteudo dos 3 exemplos acima ...]
"""
    
    # NOVO: Adicionar tipologia MNAR
    tipologia_section = """
## TIPOLOGIA DE MNAR
[... conteudo da tipologia acima ...]
"""
    
    # NOVO: Anti-bias instruction
    anti_bias_section = """
## INSTRUCAO IMPORTANTE
[... conteudo do anti-bias acima ...]
"""
    
    # Montar prompt completo
    prompt = f"""{expert_header}
{dataset_section}
{missing_context_section}
{statistics_section}
{few_shot_section}
{tipologia_section}
{anti_bias_section}
{task_section}
"""
    return prompt
```

### Cache Invalidation

O cache usa MD5 de `(stats, filename, data_type)`. Como estamos mudando o prompt (nao os stats), o cache antigo NAO sera invalidado automaticamente.

**Acao necessaria:** Limpar o cache antes de rodar. O cache e in-memory (`self._cache`), entao basta reiniciar o processo.

## Como Executar

```bash
cd "ITA-Mestrado/IC - ITA 2/Scripts/v2_improved"

# 1. Extrair features com o novo prompt (dados reais)
uv run python extract_features.py --model gemini-3-flash-preview --data real \
    --llm-approach context_aware --experiment step1_fewshot

# 2. Treinar modelos
uv run python train_model.py --model gemini-3-flash-preview --data real \
    --experiment step1_fewshot

# 3. Rodar analise forense para comparar com forensic_neutral_v2
uv run python forensic_analysis.py --experiment step1_fewshot --data real

# 4. Comparar resultados
uv run python compare_results.py --data real
```

## Como Validar

### Teste A/B: Step 1 vs forensic_neutral_v2

| Metrica | forensic_neutral_v2 | Target Step 1 |
|---------|---------------------|---------------|
| Accuracy GKF-5 | 56.2% | 60%+ |
| F1-macro | 50.1% | 55%+ |
| MCAR acc (LODO) | ~30% | 45%+ |
| MNAR acc (LODO) | ~34% | 45%+ |
| MAR acc (LODO) | 96.5% | >90% (aceitavel cair um pouco) |

### Verificacao de Sanidade

1. O distribution de domain_prior deve ser mais equilibrada (menos concentrada em 0.5/MAR)
2. Datasets MCAR canonicos (autompg, breastcancer) devem ter domain_prior mais proximo de 0.0
3. Datasets MNAR classicos (pima_insulin, mroz_wages) devem ter domain_prior = 1.0

## Riscos

| Risco | Probabilidade | Mitigacao |
|-------|---------------|-----------|
| Few-shot overfitting (LLM memoriza exemplos) | Media | Usar exemplos de dominios diferentes dos test sets |
| Anti-bias overcorrection (MCAR demais) | Baixa | Monitorar distribuicao de predicoes |
| Prompt muito longo (> token limit) | Baixa | Exemplos sao ~200 tokens extras, bem dentro do limite |
