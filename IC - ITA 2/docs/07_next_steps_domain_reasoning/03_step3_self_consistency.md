# Step 3: Self-Consistency com Multiplas Perspectivas

**Prioridade:** MEDIA-ALTA (maior robustez, maior custo de API)
**Estimativa de impacto:** +3-5pp accuracy sobre Step 2
**Custo API:** ~5x (5 chamadas por dataset por bootstrap)
**Arquivos a modificar:** `llm/context_aware.py` (nova classe ou metodo)

## Problema que Resolve

Mesmo com Few-Shot (Step 1) e Causal Reasoning (Step 2), uma unica chamada a LLM pode:
1. **Ser instavel**: mesma LLM, mesmos dados, respostas diferentes
2. **Ter blind spots**: uma perspectiva pode nao perceber evidencia de MNAR
3. **Ser overconfident**: retornar confidence=0.9 quando deveria ser 0.5

Pesquisa recente (ACL Findings 2025, NAACL 2025) mostra que **self-consistency sampling** com **votacao ponderada por confianca** reduz erros em ~40% e exige ~70% menos amostras que votacao simples.

## Solucao Proposta

### 5 Perspectivas Especializadas

Em vez de uma chamada generica, fazer **5 chamadas com perspectivas complementares**:

#### Perspectiva 1: Estatistica Pura
```
Voce e um estatistico. Analise SOMENTE os padroes numericos.
Ignore completamente o dominio e o nome da variavel.
Baseie sua classificacao exclusivamente nas estatisticas observadas:
- Taxa de missing por quartil de X0
- Correlacao mask-Xi
- Assimetria e curtose de X0 observado
```

#### Perspectiva 2: Dominio e Coleta
```
Voce e um especialista em {domain}.
Foque em COMO {x0_variable} e medido/coletado na pratica:
- Que equipamento ou processo gera este dado?
- Em que circunstancias o dado NAO seria registrado?
- A decisao de coletar depende do valor esperado?
```

#### Perspectiva 3: Processo de Registro
```
Voce e um engenheiro de dados.
Foque no PROCESSO de registro e armazenamento:
- O dado pode ser perdido por falha tecnica? (→ MCAR)
- O registro depende de outra variavel observada? (→ MAR)
- O sistema filtra ou censura valores extremos? (→ MNAR)
```

#### Perspectiva 4: Adversarial (Anti-MAR)
```
Voce e um revisor critico. O analista anterior classificou este
dataset como MAR. Sua tarefa e ARGUMENTAR CONTRA esta classificacao.

Considere:
- Se fosse MAR, QUAL variavel especifica (X1-X4) causaria o missing?
- A correlacao mask-Xi e realmente significativa ou pode ser acaso?
- Ha evidencia de que o missing depende do PROPRIO X0? (→ MNAR)
- Ha evidencia de que o missing e completamente aleatorio? (→ MCAR)
```

#### Perspectiva 5: Censoring/Truncation Expert
```
Voce e um especialista em dados censurados e truncados.
Analise especificamente:
- A distribuicao de X0 observado parece "cortada" em alguma cauda?
- A taxa de missing e muito maior em um extremo de X0?
- O dominio {domain} tipicamente envolve limites de deteccao?
- Ha evidencia de MNAR por censura ou selecao?
```

### Agregacao: Votacao Ponderada por Confianca (CISC-style)

```python
def aggregate_perspectives(responses: list[dict]) -> dict:
    """
    Confidence-Informed Self-Consistency (CISC).
    Referencia: ACL Findings 2025.
    """
    votes = {"MCAR": 0.0, "MAR": 0.0, "MNAR": 0.0}
    total_confidence = 0.0
    
    for resp in responses:
        mechanism = resp["domain_mechanism_prior"]
        confidence = resp["domain_confidence"]
        
        # Peso = confianca da perspectiva
        votes[mechanism] += confidence
        total_confidence += confidence
    
    # Normalizar
    if total_confidence > 0:
        for k in votes:
            votes[k] /= total_confidence
    
    # Mecanismo vencedor
    winner = max(votes, key=votes.get)
    
    # Confianca agregada: proporcao do vencedor
    aggregated_confidence = votes[winner]
    
    # Concordancia: quantas perspectivas concordam com o vencedor
    n_agree = sum(1 for r in responses 
                  if r["domain_mechanism_prior"] == winner)
    agreement_ratio = n_agree / len(responses)
    
    return {
        "domain_mechanism_prior": winner,
        "domain_confidence": aggregated_confidence,
        "vote_distribution": votes,
        "agreement_ratio": agreement_ratio,
        "n_perspectives": len(responses)
    }
```

### Novas Features Geradas

| Feature | Descricao | Range |
|---------|-----------|-------|
| `llm_ctx_domain_prior` | Mecanismo vencedor (0=MCAR, 0.5=MAR, 1=MNAR) | {0, 0.5, 1} |
| `llm_ctx_domain_confidence` | Confianca agregada (CISC) | [0, 1] |
| `llm_ctx_vote_agreement` | Fracao de perspectivas que concordam | [0.2, 1.0] |
| `llm_ctx_vote_mcar` | Proporcao de votos MCAR | [0, 1] |
| `llm_ctx_vote_mar` | Proporcao de votos MAR | [0, 1] |
| `llm_ctx_vote_mnar` | Proporcao de votos MNAR | [0, 1] |
| `llm_ctx_stats_consistency` | Media da stats_consistency das perspectivas | [0, 1] |
| `llm_ctx_surprise` | Media do surprise_factor | [0, 1] |

Total: 8 features (substituem as 6 atuais, adicionando 2 novas: vote_agreement e distribuicao de votos).

## Implementacao Tecnica

### Nova classe: `SelfConsistencyExtractor`

```python
class SelfConsistencyExtractor:
    """
    Self-Consistency com multiplas perspectivas para classificacao
    de mecanismos de missing data.
    
    Referencia: 
    - Self-Consistency (Wang et al., 2022)
    - Confidence-Informed SC (ACL Findings 2025)
    - Multi-Agent Debate (Du et al., ICML 2024)
    """
    
    PERSPECTIVES = [
        ("statistical", _build_statistical_prompt),
        ("domain", _build_domain_prompt),
        ("process", _build_process_prompt),
        ("adversarial", _build_adversarial_prompt),
        ("censoring", _build_censoring_prompt),
    ]
    
    def __init__(self, provider, model, n_perspectives=5, temperature=0.3):
        self.provider = provider
        self.model = model
        self.n_perspectives = n_perspectives
        self.temperature = temperature  # Maior que 0.1 para diversidade
    
    def extract_features(self, df, filename, metadata) -> dict:
        stats = self._compute_stats(df)
        
        # Coletar respostas de todas as perspectivas
        responses = []
        for name, prompt_builder in self.PERSPECTIVES[:self.n_perspectives]:
            prompt = prompt_builder(self, metadata, stats)
            response = self._call_llm(prompt)
            responses.append(response)
        
        # Agregar via CISC
        aggregated = self.aggregate_perspectives(responses)
        
        # Gerar features
        return self._to_features(aggregated, responses)
```

### Otimizacao: Chamadas Paralelas

As 5 perspectivas sao **independentes** e podem ser chamadas em paralelo:

```python
import asyncio

async def extract_features_async(self, df, filename, metadata):
    stats = self._compute_stats(df)
    
    tasks = [
        self._call_llm_async(prompt_builder(self, metadata, stats))
        for name, prompt_builder in self.PERSPECTIVES
    ]
    
    responses = await asyncio.gather(*tasks)
    return self.aggregate_perspectives(responses)
```

### Integracao com Steps 1 e 2

O Step 3 pode incorporar os Steps anteriores:
- **Step 1**: Cada perspectiva inclui os exemplos few-shot e tipologia
- **Step 2**: A perspectiva de "dominio" pode usar o prompt de DAG causal
- A perspectiva "adversarial" ganha mais poder com a tipologia MNAR

## Como Executar

```bash
cd "ITA-Mestrado/IC - ITA 2/Scripts/v2_improved"

# 1. Extrair features com self-consistency (5 perspectivas)
uv run python extract_features.py --model gemini-3-flash-preview --data real \
    --llm-approach context_aware_sc --experiment step3_selfconsistency

# 2. Treinar modelos
uv run python train_model.py --model gemini-3-flash-preview --data real \
    --experiment step3_selfconsistency

# 3. Analise forense
uv run python forensic_analysis.py --experiment step3_selfconsistency --data real

# 4. Comparar com todos os steps
uv run python compare_results.py --data real
```

## Como Validar

### Verificacoes Quantitativas

| Metrica | Step 2 (expected) | Target Step 3 |
|---------|-------------------|---------------|
| Accuracy GKF-5 | ~65% | 68%+ |
| F1-macro | ~60% | 63%+ |
| MCAR acc LODO | ~50% | 55%+ |
| MNAR acc LODO | ~50% | 55%+ |
| vote_agreement medio | — | >0.6 |

### Verificacoes Qualitativas

1. **Perspectiva adversarial deve divergir**: Se todas as 5 concordam sempre, a perspectiva adversarial nao esta funcionando
2. **vote_agreement deve correlacionar com accuracy**: Datasets com alto agreement devem ter maior accuracy
3. **Distribuicao de votos deve ser informativa**: vote_mcar, vote_mar, vote_mnar devem ter poder preditivo como features

### Analise de Custo-Beneficio

| Metrica | Single-call | Self-Consistency (5x) |
|---------|-------------|----------------------|
| Chamadas API por dataset | 2 (Step 2) | 10 (5 persp x 2 etapas) |
| Custo aprox. por dataset | ~$0.002 | ~$0.01 |
| Custo total (23 datasets x 50 boots) | ~$2.30 | ~$11.50 |
| Tempo (paralelo) | ~2s | ~3s (paralelo) |

**Conclusao:** Custo extra e trivial (~$10 adicional) para potencial de +5pp accuracy.

## Variantes a Considerar

### Variante A: 3 perspectivas (economia)
Usar apenas: statistical + domain + adversarial. Reduz custo para 3x.

### Variante B: Temperature sampling (alternativa)
Em vez de 5 perspectivas diferentes, usar o MESMO prompt 5x com temperature=0.5.
Mais simples mas menos diverso.

### Variante C: Debate (2 rodadas)
Apos a primeira rodada de votos, compartilhar os resultados e pedir uma segunda rodada
onde cada perspectiva pode mudar seu voto. Inspirado em Multi-Agent Debate (ICML 2024).

**Recomendacao:** Comecar com as 5 perspectivas (proposta principal). Se o custo for
preocupante, testar Variante A (3 perspectivas) como fallback.

## Riscos

| Risco | Probabilidade | Mitigacao |
|-------|---------------|-----------|
| 5x custo API | Certo | Usar gemini-3-flash (~$0.01/dataset) |
| Perspectivas muito similares | Media | Prompts bem diferenciados + adversarial |
| Votacao empata (2-2-1) | Possivel | Desempate pela confianca mais alta |
| Latencia 5x | Baixa | Chamadas paralelas (asyncio) |

## Fundamentacao Cientifica

### Self-Consistency (Wang et al., 2022; NeurIPS)
Demonstrou que amostrar multiplos caminhos de raciocinio e agregar por votacao majoritaria
melhora significativamente a performance em tarefas de raciocinio.

### Confidence-Informed Self-Consistency (ACL Findings 2025)
Mostrou que ponderar votos pela confianca do modelo reduz em 40% o numero de amostras
necessarias para atingir a mesma accuracy.

### Multi-Agent Debate (Du et al., ICML 2024)
Framework onde multiplos agentes LLM debatem e refinam respostas. Melhora factualidade
e raciocinio. Nossa abordagem adapta: em vez de debate iterativo, usamos perspectivas
complementares com votacao.

### Adaptive Heterogeneous Multi-Agent Debate (JKSU 2025)
Extende MAD com agentes especializados de diferentes dominios. Nossa implementacao
atribui perspectivas especializadas (estatistica, dominio, censoring) — mesmo principio.
