# Discussão e Limitações

**Data:** 2026-04-25

---

## 1. Discussão dos achados

### 1.1 O Step 1 entregou ganho marginal mas não estrutural

A configuração Pro + Step 1 + neutral entregou **+1.9pp CV** e **+4.05pp holdout** sobre `step10_flash_ca_neutral` (mesmo benchmark de 29 datasets). Esse ganho confirma que:

1. **Pro > Flash** mesmo com prompt instrumentado — capacidade de raciocínio do modelo Pro extrai mais sinal das instruções
2. **Few-shot + tipologia + anti-bias** atua nos casos certos: reduziu MAR-bias agregado de 96.5% (em `forensic_neutral_v2`) para 67.6% recall MAR
3. **NaiveBayes domina** o regime — confirmando padrão V3+ de que calibração de incerteza > capacidade do modelo em datasets ruidosos

Porém, **o ganho não é estrutural** — apenas marginal. Para atravessar o platô de 50% CV são necessárias mudanças mais profundas no raciocínio LLM (Steps 2/3), não refinamento de prompting.

### 1.2 O MAR-bias residual é o gargalo

Apesar da instrução explícita anti-MAR-bias, **9 datasets ainda têm recall ≤ 20%** quando o LLM atua sozinho. Em 8 desses 9, o erro é defaulting para MAR. Isso significa que:

- **A instrução textual é insuficiente** para superar priors fortes do modelo (especialmente em domínios clínicos)
- **Decomposição estruturada** (forçar nomear variável causadora, listar causas Tipo A/B/C) é provavelmente necessária
- **Prompt engineering simples atingiu seu teto** neste benchmark expandido

### 1.3 A regressão de −7pp vs `forensic_neutral_v2` é predominantemente metodológica

A análise no documento `04_ANALISE_REGRESSAO.md` atribui −10 a −15pp (de domain_prior solo) à expansão do benchmark e −2 a −4pp (de CV) à mudança LODO → 5-Fold. Apenas −2 a −3pp são atribuíveis ao trade-off do Step 1 (reduzir MAR-bias mas perder casos onde MAR era correto).

**Implicação para a tese:** apresentar números de `forensic_neutral_v2` (23 datasets) e `step1_v2_neutral` (29 datasets) como **dois pontos de avaliação distintos**, não como degradação. O benchmark expandido é uma contribuição metodológica em si.

### 1.4 NaiveBayes vence apesar de assumir independência condicional

A vitória de NB sobre RF/GBT/MLP em CV (49.3% vs 33-39%) é contraintuitiva — NB é "ingênuo" porque assume features condicionalmente independentes dado o rótulo, o que claramente não é verdade aqui (ex: features CAAFE são todas derivadas de momentos da mesma X0).

A explicação é a mesma da fase V3+: em **regime de rótulos ruidosos** (estimado em ~57% inconsistências segundo `forensic_neutral_v2` Cleanlab), NB é mais robusto porque:
1. Não memoriza ruído (alta capacidade de RF é um problema, não vantagem)
2. Estima probabilidades de forma calibrada (essencial para o teste)
3. Tem regularização implícita via independência condicional

Esse achado **deve ser destacado na tese** como insight metodológico geral: para classificação de mecanismos de missing data com rótulos parcialmente ruidosos, **modelos simples calibrados** vencem modelos complexos otimizados.

### 1.5 LLM contribui marginalmente em importância de feature mas significativamente em ganho global

Apesar de LLM responder por apenas **12.6%** da feature importance no RF, o ganho global de adicionar features LLM (vs baseline ML+CAAFE puro) foi historicamente:
- `step01_caafe_real` (sem LLM): ~47.6% CV
- `step1_v2_neutral` (com LLM Step 1): ~49.3% CV
- Δ = ~+1.7pp

Isso é menor que o reportado em `forensic_neutral_v2` (+5.7pp ao adicionar `domain_prior`) — sugerindo que o ganho do LLM **diminui com a expansão do benchmark** (porque os 6 datasets novos são onde o LLM mais falha).

**Implicação:** o valor agregado do LLM é **fortemente dependente da composição do benchmark**. Generalizações como "LLM adiciona +5pp" são frágeis — o ganho real depende de se o benchmark contém ou não datasets clinicamente difíceis.

## 2. Limitações do experimento

### 2.1 Variância CV alta limita confiança nos números

| Modelo | CV avg | CV std | std/avg |
|--------|:------:|:------:|:-------:|
| NB | 49.3% | 14.2% | 0.29 |
| LogReg | 41.5% | 23.5% | 0.57 |
| RF | 39.0% | 26.6% | 0.68 |

Coeficiente de variação > 0.25 em todos os modelos. Isso significa que:
- O ranking entre modelos pode variar com seed/folds diferentes
- Diferenças de < 5pp entre experimentos podem não ser estatisticamente significantes
- Re-execuções com seeds diferentes seriam necessárias para confiança formal — **mas custam ~$30/run em Pro**

### 2.2 Único seed (=42), única execução

Todos os números reportados vêm de **uma única execução com seed=42**. Não há intervalo de confiança bootstrap nem múltiplas execuções para estimar incerteza de procedimento.

**Mitigação parcial:** o CV 5-Fold já fornece 5 estimativas por modelo (a std reportada). Mas estimativa de incerteza **entre runs** (mudando ordem dos bootstraps, init de SMOTE, etc.) não está coberta.

**Custo de mitigação completa:** ~3 runs × $30 = $90 USD para um intervalo de confiança razoável. Decisão custo/valor cabe ao usuário.

### 2.3 Comparação com `forensic_neutral_v2` é não-controlada

Mudaram simultaneamente entre os dois experimentos:
- Benchmark: 23 → 29 datasets
- CV: LODO → Group 5-Fold
- Prompt: original → Step 1 instrumentado

O efeito de cada mudança isolada é **estimado**, não medido. Para isolamento real seria necessário:
1. Re-rodar `forensic_neutral_v2` com 29 datasets (mantém prompt original)
2. Re-rodar `step1_v2_neutral` com 23 datasets (mantém Step 1)
3. Re-rodar `step1_v2_neutral` com LODO

Custo: ~$90 USD de Pro.

### 2.4 O `step1_fewshot` antigo permanece não-comparável

`Output/v2_improved/step1_fewshot/` (executado antes de 2026-04-12) tem **dataset incompleto e variante de metadata desconhecida**. Não pode ser usado como baseline. Isso significa:
- Não temos um experimento "Step 1 com Pro mas com prompt original" para comparação direta
- A magnitude do ganho do Step 1 prompt isolado é **desconhecida** sem reexecução

### 2.5 Validade externa: 29 datasets ainda é pequeno

Embora seja um aumento sobre os 23 anteriores, 29 datasets é um benchmark relativamente pequeno. Em particular:
- 9 dos 29 datasets vêm de apenas 4 fontes (UCI hepatitis, kidney, pima, cylinderbands) — **dependência amostral**
- 23 dos 29 são clínicos/médicos — **viés de domínio**
- Apenas 2 são genuinamente industriais (`cylinderbands` MCAR/MNAR)

Generalização para missing data em **outros domínios** (financeiro, sensores IoT, dados sociais) requer datasets adicionais que não foram coletados.

### 2.6 Não testado contra concorrentes recentes

Não foi feita comparação direta com:
- **PKLM** (já avaliado em fase anterior, mas não com Pro)
- **MechDetect otimizado** (já avaliado)
- **Métodos de testes estatísticos clássicos** (Little 1988, Jamshidian 2010)

Embora a fase V3+ tenha feito essas comparações, é prudente refazê-las com o benchmark expandido para garantir que o Step 1 V2 mantém superioridade.

## 3. Validade interna dos números

### 3.1 O que está bem estabelecido

✅ **Ausência de leakage:** auditado nos 6 canais (A-F), todos fechados. Confirmado por log do treino ("0 grupos compartilhados").

✅ **Reprodutibilidade:** seed determinística, listas versionadas, comandos exatos documentados, todos os artefatos gerados.

✅ **Cobertura completa:** 100% dos 1.421 bootstraps processados, sem timeouts ou falhas, distribuição de classes balanceada.

✅ **Comparabilidade temporal interna:** mesma arquitetura de features, mesmos 7 classificadores, mesma estratégia CV — comparação `step1_v2_neutral` vs `step10_flash_ca_neutral` é justa (mesmo benchmark).

### 3.2 O que carece de validação adicional

❌ **Confiança estatística da diferença Pro vs Flash** (+1.9pp CV): seria necessário teste de significância via bootstrap ou múltiplas seeds.

❌ **Robustez do ranking de modelos:** NB > LogReg > RF é estável neste run, mas pode mudar com seeds.

❌ **Estabilidade da feature importance:** importance de RF é conhecida por ser instável; teste com permutation importance ou SHAP daria visão mais robusta.

❌ **Validação cruzada por domínio (LODO):** não rodada para `step1_v2_neutral`. A comparação direta com `forensic_neutral_v2` LODO é, portanto, parcialmente não-controlada.

## 4. O que este experimento ensina

### 4.1 Sobre o pipeline atual

1. **A arquitetura V3+ (NB + features estatísticas + CAAFE + LLM context) é robusta** — números consistentes entre flash e Pro, entre 23 e 29 datasets, com diferenças explicáveis.

2. **Prompt engineering tem retorno decrescente** — Step 1 prompt elaborado deu apenas +1.9pp sobre Flash sem prompt elaborado. O ganho marginal de mais elaboração de prompt é pequeno.

3. **O gargalo está no raciocínio causal do LLM, não nas features estatísticas** — porque mesmo com features estatísticas exaustivas, o ML estabiliza em ~50% CV. Mais features estatísticas não vão resolver — precisamos que o LLM **raciocine melhor**.

### 4.2 Sobre direção de pesquisa

1. **Step 2 (Causal DAG) é a próxima alavanca natural** — ataca diretamente o gargalo identificado (raciocínio causal LLM).

2. **Self-Consistency com Pro (Step 3) é menos prioritário** — porque Self-Consistency com Flash falhou (38.4%) e Pro tem prior dominial similar; votação consolidaria erros.

3. **Investigação dos 6 datasets problemáticos** é mais impactful que ensaios genéricos — focar prompt engineering nos casos onde está falhando, não em "melhorar tudo" genericamente.

### 4.3 Sobre escrita da tese/paper

1. **Reportar `forensic_neutral_v2` (23) e `step1_v2_neutral` (29) como dois pontos** — não como progressão linear.

2. **Apresentar `step1_v2_neutral` como contribuição metodológica** — benchmark expandido + Step 1 prompt + auditoria de leakage formal.

3. **Contextualizar o platô de ~50% CV** como limite teórico aproximado para abordagens baseadas em features estatísticas + LLM com prompting simples. Steps 2/3 viram "futuras direções" propostas mas não-validadas.

4. **Destacar achado negativo** sobre 9 datasets críticos como observação científica importante — não esconder.

5. **Mencionar custo computacional** ($30+/run em Pro) como barreira para iteração rápida — argumento para futuro fine-tuning de modelos menores.
