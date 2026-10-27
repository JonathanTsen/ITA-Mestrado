# Pré-registro das análises confirmatórias (passo 38) — versão 2

> **Situação: RASCUNHO v2 (27/10/2026) para ratificação do orientador (D11).**
> - A tag `prereg-v1` só é criada depois da resposta do Prof. Vitor Curtis. Até lá, nenhuma instância das famílias confirmatórias é gerada.
> - A versão 1 (21–22/10) foi revista criticamente antes de qualquer dado confirmatório. O que mudou e por quê está na seção 0.3.
> - Toda mudança depois da tag vai para `desvios.md` (seção 9).

- **Configuração congelada:** tag `freeze-v1`, com `inject-v1.yaml`, os descritores 34b e as famílias. O adendo `data_rb/freeze_v1_addendum.json` registra o N efetivo.
- **Corpus sintético:** tag `synth-v2`.
- **Código da análise:** `src/missdetect/rb/stats.py` (decisão), `analyses.py` (AUC MAR × MNAR, testes por gerador) e `pipeline.py` (etapas `cross-generator` e `stats`). Entram no commit da tag.
- **Decisões:** são do autor, preparadas com apoio e ratificadas pelo orientador; a evidência de cada uma está em `plano-revisao-banca/execucao/08-decisoes-tomadas.md`, no repositório da tese.
- **Versões:**
  - Python: scikit-learn 1.8.0, SciPy 1.17.1, NumPy 2.4.4, pandas 2.3.3, baycomp 1.0.3, mdatagen 0.2.0, tab-err 0.2.1;
  - R 4.6.1 com PKLMtest 1.0.1;
  - `uv.lock` com hash no manifesto.

## 0. Leitura rápida

### 0.1 O problema, a unidade e as perguntas

**Unidade classificada.** Uma coluna numérica Y com valores ausentes numa tabela, com colunas auxiliares Z completas e o indicador de ausência M. O método recebe só o que é observável (Y nas linhas observadas, Z e M) e devolve P(MCAR), P(MAR) e P(MNAR).

**Perguntas de pesquisa** (Cap1):
- **RQ1:** com os mesmos descritores e o mesmo protocolo, decompor o problema em detecção de MCAR (L1) seguida de MAR × MNAR (L2) melhora a classificação em relação à classificação direta em três classes?
- **RQ2:** quais grupos de descritores observáveis contribuem para separar os três mecanismos, em particular MAR de MNAR?
- **RQ3:** como a corrupção controlada dos rótulos de treino afeta o desempenho, e quanto uma mitigação por Confident Learning reduz esse efeito?
- **RQ4:** com controles explícitos de informação e de vazamento, como um LLM usado como classificador direto se compara ao classificador por descritores, e descritores extraídos por um LLM acrescentam desempenho a ele?

**O que esta validação prova e o que não prova.**
- Os rótulos valem **por construção**: as ausências são injetadas por código em tabelas reais completas. O rótulo é provado **em relação ao gerador e às auxiliares Z** usadas na injeção.
- Por isso, os resultados valem para a classe de mecanismos injetada (seção 1, "Escopo"), não para dados com ausência natural. Esses aparecem só num estudo de caso, sem métrica de acerto.

### 0.2 Glossário

| Termo | Significado |
|---|---|
| Família de dados | Grupo de tabelas que são versões ou derivações da mesma base; é a unidade estatística (uma observação por família) |
| Contexto | Uma escolha de tabela, alvo Y, auxiliares Z, n, k, taxa, β e direção |
| Trio | As três instâncias (MCAR, MAR, MNAR) de um contexto, com a mesma contagem de ausências |
| Gerador simétrico | Gerador principal: sorteia as linhas ausentes com peso exp(β·posto) do driver (MAR) ou do próprio Y (MNAR); β = 0 é o MCAR, β = ∞ é determinístico |
| mdatagen, tab_err | Geradores externos: a biblioteca mdatagen 0.2.0 e o tab_err 0.2.1 (o gerador usado no artigo do MechDetect) |
| Descritores 34b | As 28 medidas observáveis que o classificador usa, em 5 grupos: G0 taxa de ausência; G1 forma de Y observado; G2 máscara × auxiliares; G3 Y dado Z; G4 análogo observável do 1º teste do MechDetect |
| Configuração primária | Classificador direto (3 classes), HGB com hiperparâmetros fixos, calibração escolhida na CV interna |
| Logística simples | Regressão logística com 4 estatísticas (taxa de ausência, AUC de M a partir de Z, desvio do AUC de M a partir de Ŷ(Z), assimetria de Y observado) |
| MechDetect-O | Adaptação nossa, pré-especificada, do MechDetect que usa só dados observados |
| Painel oráculo | Métodos que veem o Y completo (o MechDetect como publicado e o nosso classificador com descritores-oráculo); nunca ranqueados com os demais |
| Só-nuisance | Classificador que só vê fatores do desenho (taxa, n, k, gerador); auditoria de atalhos |
| Macro-recall | Média dos recalls das três classes; com classes balanceadas, igual à acurácia |
| HL | Estimador de Hodges–Lehmann da diferença entre métodos |
| Holm | Correção para múltiplas hipóteses na mesma família |
| TOST | Teste de equivalência por dois testes unilaterais |
| δ | Margem de equivalência e de relevância prática: 0,02 de macro-recall (seção 5) |
| Desenvolvimento / confirmatória | 8 famílias usadas no piloto e nos ensaios; 50 famílias efetivas guardadas para a prova |
| LLM-A1, A2, A3 | LLM como classificador direto (A1), como extrator de descritores (A2) e com metadados semânticos (A3, exploratório) |

### 0.3 O que foi visto antes deste documento e o que mudou por causa disso

**Nas famílias confirmatórias, só se viu:**
- a elegibilidade calculada nas tabelas completas, sem nenhuma máscara;
- as descrições e os nomes do OpenML;
- o agrupamento em famílias.

Nenhuma instância confirmatória foi gerada, nenhum modelo foi treinado com elas e nenhum escore foi calculado.

**Nas 8 famílias de desenvolvimento (e no corpus sintético), viram-se:**
- os pilotos r1 (invalidado por um defeito de desempate) e r2;
- a checagem da configuração congelada;
- o ensaio geral de 22/10, com a cadeia completa, a RQ3 e o teste de fumaça do estudo de caso.

**Mudanças na versão 1 (21–22/10), motivadas pelo piloto:**
- grade de β {1, 2, 4, 8, ∞};
- HGB como aprendiz, pela curva de aprendizado no corpus sintético;
- roteamento fullprob;
- descritores 34b;
- MBOV excluído;
- N efetivo de 50;
- regra dos sentinelas.

**Mudanças na versão 2 (27/10), depois de uma revisão crítica feita com os números do ensaio geral (só desenvolvimento, N = 8, sem valor confirmatório):**

| Mudança | Motivo e número que a motivou |
|---|---|
| A H3 deixa de comparar com o classificador uniforme e passa a comparar com a **logística simples** | Contra o uniforme, o resultado era conhecido (0,656 contra 0,333). Contra a logística simples, o desfecho está em aberto (Δ ≈ 0,001 no ensaio), e é a pergunta que mede o valor dos descritores |
| H5 nova: **geradores externos** | Treinado e testado no mesmo gerador, o desempenho pode refletir a assinatura do gerador. No ensaio, treinando só no simétrico e testando na mdatagen e no tab_err: 0,684 contra 0,641 do MechDetect-O |
| H6 nova: **MAR × MNAR** | É a questão central da área e não tinha teste confirmatório. No ensaio, a AUC foi 0,83 |
| H2 com **equivalência por TOST**; a ROPE bayesiana fica só descritiva | Com a regra da v1, um Δ ≈ 0 terminaria em "evidência insuficiente" e não poderia ser afirmado |
| A H4 vira **família própria** | A H4 depende de chaves de API e de um adendo; não deve mexer no Holm das demais |
| Secundárias novas: painel oráculo por gerador, mdatagen com a própria CV, estatísticas do MechDetect como descritores | A comparação justa com o MechDetect como publicado não tinha teste. No ensaio, o MechDetect publicado empata com o nosso classificador oráculo no tab_err |
| Tabela de interpretação (5.2), proteções (9) e registro externo (10) | Evitar "mover a trave" e dar anterioridade verificável por terceiros |

Essas mudanças foram feitas vendo só resultados de desenvolvimento. Elas tornam a família confirmatória **mais difícil**: duas das hipóteses novas têm desfecho em aberto no ensaio.

## 1. Dados

**Fontes.** As 101 tabelas OpenML avaliadas pelo MechDetect, a partir dos CSV distribuídos com o código dos autores (`calgo-lab/MechDetect@43c0f00`).
- O SHA-256 de cada arquivo está em `data_rb/sources_local.csv` e é conferido a cada leitura.
- Os metadados do OpenML (nome, versão, licença, URL de origem) estão em `data_rb/openml_metadata_20261022.csv`, coletados em 22/10/2026.
- O número de linhas de cada CSV confere com o OpenML nas 101 tabelas, e os nomes do OpenML confirmaram o agrupamento em famílias sem mudança.
- A tese cita os IDs do OpenML e não redistribui os dados.

**Filtros de tabela.** Ficam as tabelas:
- sem valores ausentes;
- com ≥ 500 linhas;
- com ≥ 3 colunas numéricas não constantes, depois de remover a coluna-alvo do benchmark original e as colunas de identificador (nome com "id" como palavra, ou inteiros únicos por linha).

Resultado: 98 de 101 tabelas.

**Famílias.** Duas tabelas são candidatas a família quando a sobreposição de nomes de colunas é ≥ 0,6, com ≥ 3 colunas em comum. Fundem-se quando a contenção média de valores nas colunas comuns é ≥ 0,9, além das decisões manuais em `data_rb/family_overrides.csv`. O agrupamento é transitivo, e cada família tem um representante:
- critério: o maior nº de colunas numéricas não constantes; desempate pelo maior n e depois pelo menor ID;
- **69 famílias**, das quais 8 de desenvolvimento, sorteadas com `child_rng(20261021, "development_families")`;
- as famílias de desenvolvimento são 37, 151, 251, 1049 (representante 1050), 44089, 44124, 44145 e 44980. Foram usadas no piloto e nos ensaios e **nunca** entram nas análises confirmatórias.

**Colunas sentinela.** Uma coluna sai da tabela, como alvo e como auxiliar, quando um valor codifica "não medido", seja por documentação da fonte, seja por ser fisicamente impossível pela definição da variável.
- A lista está em `data_rb/sentinel_exclusions.csv`. Hoje tem só a Pima, que é de desenvolvimento: `plas`, `pres`, `skin`, `insu` e `mass` iguais a 0.
- Não há regra automática por massa num valor de fronteira.
- As descrições do OpenML das 69 tabelas representantes foram conferidas em 22/10/2026 (`scripts/revisao_banca/passo31/openml_sentinelas.py`). Nenhuma documenta um código de "não medido" em coluna numérica, e a própria descrição da Pima diz "Missing Attribute Values: None". Por isso os zeros da Pima entram pelo critério de impossibilidade física, e a lista não muda.

**Elegibilidade do par (Y, Z).** Calculada na subamostra de até 2.000 linhas da base (a maior subamostra), com todas as auxiliares candidatas, antes de qualquer máscara:
- Y e o driver precisam ter ≥ 20 valores distintos e moda ≤ 50% das linhas;
- o R² de Y predito das auxiliares (HistGradientBoosting, CV de 5 folds) precisa ser ≤ 0,95;
- o máximo de |Spearman(Y, Z_j)| precisa ser ≤ 0,95.

Bases inelegíveis são sorteadas de novo, até 20 tentativas. **Uma família sem nenhuma base elegível sai por regra** e fica registrada em `family_status.parquet`.

**N efetivo.** Das 61 famílias confirmatórias listadas, **50 geram instâncias**. A conta foi feita com as regras congeladas, só pela elegibilidade nas tabelas completas, sem máscara (`data_rb/eligibility_freeze_v1.csv`). Excluídas por regra:
- **6, 23, 4534, 44055, 44061, 44958, 44960, 44966, 44967 e 44994:** no máximo 1 coluna elegível;
- **44969:** R² ou |Spearman| > 0,95 em todos os alvos.

As 50 famílias efetivas, no formato família:representante:

> 31, 44, 54, 310, 725, 823, 1046, 1461, 1464, 1480, 1494, 1510, 4135, 23517, 40701, 40994, 42493, 44026:45012, 44054, 44056, 44059, 44062, 44063, 44064, 44066:44989, 44091:44136, 44122:44133, 44123:44139, 44125, 44127, 44130:44157, 44131, 44132, 44134:44137, 44147:44983, 44148:44964, 44158, 44160, 44162, 44956, 44957, 44959, 44962, 44963, 44970, 44974, 44976, 44981, 44993, 45402

**Desenho por família.**

| Fator | Níveis e regra |
|---|---|
| Bases | Uma por taxa r ∈ {0,1; 0,2; 0,3; 0,4; 0,5}. Alvo, driver e ordem das auxiliares sorteados; regra do driver (maxcorr ou random) balanceada, em 3 : 2 por família |
| n | {500, 1000, 2000}, subamostras aninhadas, limitado ao tamanho da tabela. Níveis que colidem são descartados e contados |
| k | {2, 4, 8}, auxiliares aninhadas, com o driver sempre incluído e a ordem das colunas observadas sorteada. Os três k compartilham a máscara |
| β e direção | β ∈ {1, 2, 4, 8, ∞} e direção ∈ {lower, upper, both}, balanceados **nas marginais** das células (base, n). O cruzamento β × direção não é controlado |

**Geradores.**
- **Principal: simétrico.** Gumbel-top-k com pesos exp(β·s) sobre o posto, contagem exata de ausências, linhas na ordem da permutação aleatória; β = 0 é o MCAR.
- **Por que o simétrico, e não a mdatagen, é o principal** (decisão D4):
  - ele controla β e o determinismo **do mesmo modo** no MAR e no MNAR;
  - na mdatagen 0.2.0, as variantes estocásticas existem só no MAR. Usá-la como gerador principal confundiria o mecanismo com o determinismo e abriria um atalho: "determinístico ⇒ MNAR";
  - o MBOV tem déficit de taxa.
  - O plano de resposta à banca previa a mdatagen como principal. Esta mudança vai declarada no documento de resposta, e a mdatagen entra em duas análises testadas: na H5 (confirmatória) e numa secundária com a própria CV (seção 6).
- **Externos**, nos contextos com n = min(1000, nº de linhas):
  - **mdatagen 0.2.0** em pares determinísticos: lower → `lowest`/`run_0`; upper → `highest`/`run_1`; both → `mix`/`run_0.5`; o MCAR usa `random`;
  - **tab_err 0.2.1:** ECAR/EAR/ENAR semeados, só na coluna-alvo.
- **Excluído:** o MBOV da mdatagen, por déficit de taxa (0,15–0,23 p.p. com r = 0,1; 1,73–2,22 com 0,3; 4,42–6,41 com 0,5).
- **Trios desiguais:** um trio (MCAR, MAR, MNAR) de um contexto e gerador é recusado se as contagens de ausências diferirem. Isso acontece na mdatagen `mix`/`run_0.5` com N ímpar; os trios recusados são reportados.

**Escopo (o que a validação cobre).**
- Uma coluna numérica com ausência por vez, com as auxiliares completas.
- Taxas de 10% a 50%; n entre 500 e 2.000; k entre 2 e 8; classes balanceadas.
- MAR pelo posto de um único driver observado.
- MNAR por auto-mascaramento: a ausência depende do próprio Y (inferior, superior ou nas duas caudas).
- **Fora do escopo:** o MNAR por variável latente (a ausência depende de uma coluna retirada da tabela). O rótulo dessa variante dependeria de um limiar estimado e deixaria de valer por construção; ela também não foi congelada em `freeze-v1`. Também ficam fora várias colunas incompletas ao mesmo tempo, Z com ausências e MCAR raro (este último aparece só na análise exploratória de prevalência, seção 6).

**Separação observado × oráculo.** Os métodos recebem só `y` (com NaN) e `z`. IDs, caminhos e o oráculo nunca entram em descritores nem em prompts.

## 2. Pipeline primário (configuração primária)

- **Descritores 34b:** 28 descritores, grupos G0–G4 (`configs/rb/feature_schema_34b.json`, SHA-256 `d513af8f06f8352c1f9d437bce96447d2ef328c31ff252b997d18ac9a6ec2b95`).
- **Aprendiz primário (D7):** `HistGradientBoostingClassifier` com os hiperparâmetros padrão do scikit-learn 1.8.0, `early_stopping=False`, `random_state` fixo e **sem grade**.
- **Calibração:** nenhuma, Platt ou isotônica (um-contra-o-resto, normalizada). A escolha é feita pelo menor log-loss em predições fora-da-dobra da CV interna (4 folds por família), com cross-fitting do calibrador.
- **Configuração primária:** classificador **direto** (3 classes).
- **Hierárquico (RQ1):**
  - L1 (MCAR × não-MCAR) treinado em todas as instâncias de treino;
  - L2 (MAR × MNAR) treinado nas instâncias de treino verdadeiramente não-MCAR;
  - mesmo aprendiz, mesmos descritores e mesma regra de calibração;
  - roteamento **fullprob**: (P(MCAR), P(MAR), P(MNAR)) = (p₁, (1 − p₁)·p₂, (1 − p₁)·(1 − p₂)), com p₁ = P(MCAR) do L1 e p₂ = P(MAR | não-MCAR) do L2.
- **Logística simples (H3):** `LogisticRegression` multinomial do scikit-learn 1.8.0 (`max_iter=2000`, demais parâmetros padrão), com imputação pela mediana e padronização ajustadas só no treino. Usa as 4 colunas `g0_missing_rate`, `g2_cv_auc_boosting`, `g3_auc_mask_from_yhat_dev` e `g1_skewness`, a mesma regra de calibração e as mesmas partições.
- **Validação cruzada:** CV externa de 5 folds por família, com 3 repetições em atribuições diferentes. A unidade de teste é a família inteira.
- **Escopo do treino:** a avaliação primária treina e testa **só nas instâncias do gerador principal** (`evaluate --generators symmetric`). A H5 usa os mesmos modelos de cada fold, aplicados aos geradores externos (etapa `cross-generator`).
- **Falhas:** uma falha de extração, de baseline ou de chamada conta como erro (probabilidades uniformes, classe −1). Nunca é descartada.

## 3. Estimandos e métricas

**Escore por família.** Macro-recall das instâncias da família em cada repetição, depois a média das 3 repetições.

**Estimandos:**
- **de nível:** a média, entre as 50 famílias efetivas, do escore por família da configuração primária no gerador principal, com IC95% por bootstrap de famílias. A referência de acaso (1/3) é descritiva. É reportado **sempre com a quebra por β** e por taxa, porque depende da grade do desenho. O mesmo vale para a configuração primária nos geradores externos (H5);
- **comparativo** (hipóteses de diferença): o **Hodges–Lehmann** das diferenças por família, isto é, a mediana das médias de Walsh. A diferença média ± dp é só descritiva;
- **H6:** a AUC MAR × MNAR por família. É a AUC de P(MNAR) / (P(MAR) + P(MNAR)) nas instâncias verdadeiramente não-MCAR (positivo = MNAR), média das 3 repetições. Falha vale 0,5.

**Métricas secundárias:** F1-macro, recall por classe, log-loss, Brier multiclasse, ECE por classe (10 bins de igual massa) e AUC um-contra-o-resto. No desenho balanceado, a acurácia coincide com o macro-recall e não conta como evidência independente.

## 4. Hipóteses confirmatórias

### Família principal (Holm, m = 6, α = 0,05; todos os testes bilaterais)

- **H1 — aplicabilidade do MechDetect sem o Y removido** (resultado esperado por construção): a configuração primária e o **MechDetect-O** diferem.
  - **MechDetect-O:** teste 1 sobre [Z, Ŷ(Z)] × máscara permutada; teste 2 sobre [Z, Ŷ] × Z; Ŷ por cross-fit só nas linhas observadas; HGB(200), 10 folds; AUC sobre `predict()`, como no código dos autores; Mann-Whitney unilateral com α/2.
  - **Expectativa declarada:** sem os valores removidos, o teste 2 compara funções de Z com Z e quase nunca rejeita. Por isso o MechDetect-O tende a decidir MAR entre os não-MCAR (recall de MNAR 0,3% no ensaio).
  - **Leitura:** um resultado favorável mede se o MechDetect se aplica sem Y. Ele **não** mostra que o nosso método supera o MechDetect publicado, que usa informação não observável na prática. Essa comparação está no painel oráculo (seção 6).
- **H2 — RQ1, diferença:** o hierárquico (fullprob) e o direto, com o mesmo aprendiz e os mesmos descritores, diferem.
- **H2e — RQ1, equivalência:** o hierárquico e o direto são equivalentes dentro de ±δ (TOST).
  - As nulas de H2 e H2e são disjuntas, e as duas entram na mesma família de Holm.
- **H3 — RQ2, valor dos descritores:** a configuração primária e a **logística simples** diferem.
  - Com δ = 0,02 e o dp observado no ensaio (0,058), a equivalência com N = 50 não teria poder. Por isso a H3 só tem teste de diferença.
  - O classificador uniforme fica como referência descritiva do estimando de nível.
- **H5 — generalização para geradores externos:** nas instâncias da mdatagen e do tab_err das famílias de teste, a configuração primária e o MechDetect-O diferem.
  - A configuração primária é a mesma, treinada **só** no gerador principal, em cada fold da avaliação primária, e aplicada sem ajuste.
  - O escore por família junta as instâncias dos dois geradores externos.
  - Descritivos, sem teste: o nível com IC, a quebra por gerador, a logística simples treinada no principal e a **perda fora do gerador**. A perda é o HL de (configuração primária treinada nos próprios geradores externos − treinada no principal), nas mesmas instâncias.
- **H6 — MAR × MNAR (substituto testável do "teto"):** a AUC MAR × MNAR por família difere de 0,5 (Wilcoxon de uma amostra sobre AUC − 0,5).

### Família da H4 (m = 1, α = 0,05)

- **H4 (RQ4):** o **LLM-A1** do provedor confirmatório e a configuração primária diferem, no painel de LLM.
  - **Provedor confirmatório:** Google (Gemini). O OpenAI (GPT) roda como secundário descritivo.
  - **Painel:**
    - 30 das 50 famílias efetivas, sorteadas com `child_rng(20261101, "llm_panel")`;
    - em cada família, os 6 contextos do gerador principal com r ∈ {0,1; 0,3; 0,5} × k ∈ {2, maior k disponível} e n = min(1000, nº de linhas);
    - total: 540 instâncias.
  - **Escores:** o LLM tem o macro-recall por família nas 18 instâncias do painel. A configuração primária tem as predições da CV principal restritas às mesmas 18 instâncias (escore por repetição, média das 3).
  - **Informação desigual, declarada:** o LLM vê no máximo 200 linhas sorteadas; a configuração primária vê a instância inteira. Na sensibilidade pré-registrada, a configuração primária é aplicada à mesma subamostra de ≤ 200 linhas que o LLM recebe, com os descritores recalculados.
  - **Memorização:** uma família em que o LLM reconhece a base (sondas do `tabmemcheck` ou acerto na pergunta "qual é o nome desta base?") sai da análise principal da H4 e entra numa sensibilidade.
  - **Adendo LLM, com tag `prereg-v1-llm`, antes de qualquer chamada confirmatória.** Congela:
    - IDs e datas dos modelos (o mais capaz, de uso geral, não-preview de cada provedor na data do adendo);
    - prompt e renderizador por hash (tabela cega: colunas `c0…ck`, transformação afim positiva por coluna, 3 algarismos significativos, ≤ 200 linhas sorteadas, linhas e colunas permutadas);
    - esquema JSON;
    - temperatura 0 ou o menor esforço de raciocínio;
    - até 2 retentativas; saída inválida = falha = erro.
    - Prompt e renderizador são desenvolvidos **só com famílias de desenvolvimento**; nenhum resultado das famílias confirmatórias informa o adendo.
  - **Poder:** com 18 instâncias por família, só o ruído binomial dá dp de ≈ 0,11 no escore de um método. O MDE fica entre ≈ 6 e 8 p.p. com N = 30; ele é recalculado por simulação no adendo. (A estimativa de 5,4 p.p. da v1 usava o dp do classificador e subestimava o MDE.)
  - **Se a H4 não rodar** (sem chaves ou corte de orçamento), a RQ4 fica sem resposta confirmatória, e isso é declarado. A família principal não muda.

## 5. Testes, estimativas e decisão

- **Teste de diferença:** Wilcoxon de postos sinalizados sobre as diferenças por família (`scipy.stats.wilcoxon`, `zero_method="pratt"`, `method="asymptotic"`, `correction=False`, bilateral).
- **Teste de equivalência (H2e):** TOST de Wilcoxon, com as mesmas opções. São dois testes unilaterais, sobre d + δ (H0: mediana ≤ −δ) e d − δ (H0: mediana ≥ δ); o p é o maior dos dois.
- **Teste de uma amostra (H6):** o mesmo Wilcoxon sobre AUC − 0,5.
- **Holm** sobre os 6 testes da família principal. Ele controla o erro do tipo I sob dependência arbitrária, inclusive com N diferentes entre hipóteses.
- **Efeito:**
  - Hodges–Lehmann com **IC95% percentil por bootstrap de famílias** (10.000 réplicas, semente 20261101). O IC **não** é ajustado por multiplicidade e é só descritivo: a decisão vem do p ajustado. Se o IC excluir zero com p ajustado ≥ 0,05, vale o p;
  - correlação rank-biserial pareada (Kerby), sem as diferenças nulas. O Pratt do teste inclui os zeros no ranqueamento; a diferença é declarada.
- **Teste bayesiano de postos sinalizados:** `baycomp.SignedRankTest.probs(comparador, método, rope=0.02, prior=0.5, nsamples=50000, random_state=20261101)`, com a região direita = método melhor. **Só descritivo.**
- **Tabela de decisão** (única; implementada em `stats.decide` e `stats.decide_equivalence`):
  - **teste de diferença:** "superior" ou "inferior" se o p de Holm < 0,05, pelo sinal do HL. Se |HL| < δ, o rótulo é "superior (efeito menor que δ)": significativo, mas abaixo da relevância prática. Nos demais casos, "evidência insuficiente";
  - **TOST:** "equivalentes dentro de ±δ" se o p de Holm < 0,05; senão, "equivalência não demonstrada".
- **Por que δ = 0,02:**
  - no ensaio, só a repartição dos folds já muda o escore de uma família em ≈ 0,02: a mediana do dp entre as 3 repetições foi 0,024 na configuração primária e 0,022 no hierárquico (0,011 na logística simples, que tem menos parâmetros);
  - numa família típica (135 instâncias, 45 por classe), cada instância vale 1/45 no recall da sua classe, ou 0,0074 no macro-recall. Então 0,02 equivale a ≈ 3 instâncias a mais acertadas.
  - Diferenças menores que isso ficam no ruído da partição e não mudariam a escolha de um usuário.
- **Dependência entre famílias:** o escore de cada família vem de modelos treinados nas outras famílias (Nadeau e Bengio, 2003). O pareamento atenua o efeito, mas os testes podem ser um pouco anticonservadores, e isso é declarado.

### 5.1 Poder

Os dp das diferenças por família vêm do ensaio geral com a grade final (8 famílias de desenvolvimento). O MDE é calculado com Holm m = 6, poder 0,8 e N = 50 (aproximação normal, fator 3,48). O "pior caso" usa o limite superior de 95% do dp com 7 graus de liberdade (× 1,80).

| Hipótese | dp no ensaio | MDE | MDE no pior caso |
|---|---|---|---|
| H1 | 0,094 | 4,6 p.p. | 8,3 p.p. |
| H2 | 0,031 | 1,5 p.p. | 2,8 p.p. |
| H3 | 0,058 | 2,9 p.p. | 5,1 p.p. |
| H5 | 0,086 | 4,2 p.p. | 7,6 p.p. |
| H6 (AUC − 0,5) | 0,098 | 0,048 de AUC | 0,087 de AUC |

- **H2e:** com o dp do ensaio (0,031), o TOST tem poder de 0,8 para declarar equivalência se a diferença verdadeira for menor que ≈ 0,006 a 0,009, conforme a posição da H2e no Holm. Com o dp do pior caso, o poder fica marginal.
- **H4:** ver a seção 4.

### 5.2 Tabela de interpretação (desfecho → frase na tese → efeito na contribuição)

| Hipótese | Desfecho | Frase na tese | Efeito na contribuição |
|---|---|---|---|
| H1 | superior | "Sem acesso aos valores removidos, o MechDetect quase não detecta MNAR; o nosso método supera essa adaptação" | Delimita a aplicabilidade do estado da arte; não é vitória sobre o MechDetect publicado |
| H1 | evidência insuficiente ou inferior | "Mesmo a adaptação observed-only do MechDetect não foi superada" | Contribuição frágil: a tese reporta isso no resumo |
| H2 / H2e | equivalentes | "Decompor em L1 e L2 não muda o desempenho (±2 p.p.)" | Responde a RQ1 com um nulo informativo; a configuração direta fica como recomendação |
| H2 | superior ou inferior | "O hierárquico foi superior (ou inferior) em d [IC]" | Responde a RQ1 diretamente |
| H2 / H2e | nenhum dos dois | "Os dados não permitem afirmar diferença nem equivalência" | RQ1 sem resposta conclusiva |
| H3 | superior | "Os 28 descritores acrescentam d [IC] a um modelo de 4 estatísticas" | Sustenta o conjunto de descritores |
| H3 | evidência insuficiente | "Não há evidência de que o conjunto completo supere a logística simples" | A contribuição passa a ser o protocolo e a formulação; recomenda-se o modelo simples; a RQ2 é discutida à luz disso |
| H3 | inferior | "A logística simples foi superior" | Idem, com mais força |
| H5 | superior | "Treinado só no gerador principal, o método mantém vantagem em geradores externos", sempre com a perda fora do gerador ao lado | Responde à crítica de circularidade |
| H5 | evidência insuficiente ou inferior | "Fora do gerador de treino, a vantagem não se sustenta" | Limitação central, declarada no resumo |
| H6 | superior | "Sob a classe de mecanismos injetada, MAR e MNAR são distinguíveis acima do acaso (AUC [IC])" | É o substituto testável do "teto"; não implica identificabilidade em geral |
| H6 | evidência insuficiente | "MAR e MNAR não foram distinguíveis acima do acaso neste protocolo" | Limitação central |
| H4 | qualquer desfecho | Frase no nível correspondente da seção 8 | Responde a RQ4; se não rodar, a RQ4 fica sem resposta confirmatória |

**"Praticamente relevante"** só quando |HL| ≥ δ. O estimando de nível nunca é chamado de "bom" ou "ruim" sem a comparação correspondente.

## 6. Análises secundárias (cada uma com o seu próprio Holm) e exploratórias

**Secundária, L1** (m = 3 fixo; uma comparação que não puder rodar entra no Holm com p = 1):
- **Classificador L1:** o L1 do hierárquico (`hier_hgb_fullprob`), que decide não-MCAR quando P(MCAR) < 0,5.
- **Comparadores:** teste de Little (α = 0,05), teste 1 do MechDetect-O (α/2) e PKLM (α = 0,05; 100 projeções e 200 repetições; `set.seed(20261101)` antes de cada instância).
- **Subconjunto do PKLM:** 8 contextos inteiros do gerador principal por família, 1.200 instâncias, escolhidos com `child_rng(20261101, "pklm_subset")` sobre os contextos ordenados por ID.
- **Métricas:** sensibilidade, especificidade e acurácia balanceada (AB) por família. O teste usa a AB, com Wilcoxon-Pratt e Holm.
- A comparação com o PKLM usa só o seu subconjunto.
- A especificidade dos testes é reportada **como medida**: no piloto, o PKLM rejeitou 12,5% sob MCAR.
- O teste 1 do MechDetect-O usa a AUC sobre `predict()`, por fidelidade ao código dos autores (passo 36).

**Secundária, RQ2** (m = 4): para k = 1…4, configuração primária com todos os descritores × sem o grupo G_k. Escore por família, Wilcoxon-Pratt e Holm. O SHAP é só descritivo.

**Secundária, estatísticas do MechDetect como descritores** (m = 1): configuração primária × o mesmo aprendiz só com o grupo G4 (`g4only_hgb`).

**Secundária, painel oráculo por gerador** (m = 3, um teste por gerador, cada um com a sua CV): o nosso classificador com descritores-oráculo × o MechDetect como publicado (clean, com o Y completo), no simétrico, na mdatagen e no tab_err. É a comparação justa com o estado da arte, com a mesma informação dos dois lados; o tab_err é o gerador do artigo do MechDetect.

**Secundária, mdatagen com a própria CV** (m = 2): configuração primária × logística simples e configuração primária × MechDetect-O, treinando e testando na mdatagen.

**Secundária, RQ3** (m = 4; D2):
- **Contrastes:** ponderação × nenhuma e poda × nenhuma, para ruído simétrico e MAR ↔ MNAR, com taxa de 0,20.
- **Escore:** macro-recall no teste limpo, por família.
- As curvas com as taxas {0; 0,1; 0,2; 0,4} são descritivas.

**Secundária, RQ4-A2** (m = 1): pipeline + descritores do LLM × pipeline, no painel.

**Exploratórias:**
- por mecanismo: recall por classe, com Wilcoxon-Holm dentro dessa família;
- geradores externos: a perda fora do gerador e a quebra por gerador (descritivas, da etapa `cross-generator`);
- leave-one-generator-out;
- ensaios por fator (passo 40), com forest plots e IC por bootstrap, sem p por célula. Não há modelo misto confirmatório. Inclui:
  - a AUC MAR × MNAR em função de R²(Y | Z) e de β, que é o substituto descritivo do "teto";
  - **prevalência realista** (prior-shift): a acurácia esperada sob prevalências diferentes das balanceadas e o L1 quando o MCAR é raro (a implementar, a partir das predições já gravadas);
  - **ausência também nas auxiliares:** uma execução limitada com 5–10% de NaN MCAR em Z (a implementar; se não for feita, a impossibilidade é justificada);
- roteamentos hard, threshold e soft:
  - threshold: θ ∈ {0,20; 0,25; …; 0,80}, escolhido pelo macro-recall no OOF interno;
  - soft: zonas (a, b) ∈ {0,2; 0,3; 0,4} × {0,6; 0,7; 0,8}, pelo mesmo critério;
- calibração;
- classificadores descritivos com Friedman + Nemenyi (diagrama CD): HGB, logreg, RF, SVM, KNN, MLP e NB;
- transferência sintético → real (`synth-v2`);
- LLM-A3 e o provedor secundário;
- sensibilidade sem os contextos cujo Y completo tem massa ≥ 10% em 0 ou no seu mínimo: diferença pareada nas famílias que sobram nos dois conjuntos (IC por bootstrap), com a lista das famílias que saem por inteiro;
- controle sintético: CV agrupada por tabela-base no `synth-v2`;
- avaliação do L2: diagnóstica (macro-recall MAR × MNAR nas instâncias verdadeiramente não-MCAR) e operacional (nas verdadeiramente não-MCAR que o L1 roteou ao L2), com a fração de MCAR roteadas à parte; falha conta como erro;
- regime "dirty" do MechDetect, reportado como degenerado.

**Auditoria de atalhos nas famílias confirmatórias:** se o classificador só-nuisance passar de 0,40 de macro-recall, isso é reportado como alerta, ao lado das hipóteses, sem mudar a análise confirmatória.

**Estudo de caso (D3):** dados com ausência natural, **sem** acurácia nem concordância como métrica de acerto.

## 7. Situação da implementação

| Componente | Situação |
|---|---|
| Injeção, descritores 34b, avaliador, baselines (MechDetect-O, oráculo, dirty, Little, PKLM, logística simples), relatório | Implementados e testados |
| `stats.py` (família principal com Holm m = 6, TOST, teste de uma amostra, Hodges–Lehmann com bootstrap, rank-biserial, ROPE descritiva, decisão; H4 em família própria; estimando de nível com IC) | Implementado e testado (versão 2); congelado com a tag `prereg-v1` |
| H5 (`cross-generator`) e H6 (`analyses.mar_mnar_auc`) | Implementados e testados; ensaio nas famílias de desenvolvimento |
| L1 secundário, recall por mecanismo, L2, ablação da RQ2, painel oráculo por gerador, mdatagen, estatísticas do MechDetect como descritores, LOGO, transferência, CD, sensibilidade da massa na fronteira | Implementados e testados (`analyses.py`; `pipeline.run_secondary`, `run_ablation`, `run_logo`, `run_transfer`) |
| RQ3 (passo 44) e estudo de caso (passo 45) | Implementados e testados (`noise.py`, `case_study.py`) |
| Prevalência realista e NaN em Z (passo 40) | **A implementar** |
| LLM (passos 42/43) | **A implementar.** O adendo LLM, com tag, vem antes de qualquer chamada confirmatória |

Implementar um componente da tabela acima **não** é desvio. Mudar a especificação é, e vai para `desvios.md`.

## 8. Regras de redação

Toda comparação usa exatamente um destes níveis:
- **só estimativa pontual:** "the point estimate of X is higher than that of Y in this experiment";
- **teste significativo após Holm:** "X was statistically superior to Y on the N evaluated dataset families under protocol P (Hodges–Lehmann difference d, 95% CI [a, b], adjusted p)";
- **não significativo:** "there is insufficient evidence of a difference between X and Y";
- **equivalência** (só pelo TOST pré-registrado): "X and Y are practically equivalent within ±δ (TOST, adjusted p)".

**Também:**
- As conclusões valem só para as famílias, os geradores e as condições avaliadas (seção 1, "Escopo").
- Nenhum resultado secundário ou exploratório aparece no resumo, nas conclusões ou no título sem o rótulo correspondente.
- O estimando de nível aparece sempre com a quebra por β.
- O alerta da auditoria de atalhos, se houver, aparece junto das hipóteses.

**A tese pode afirmar:**
- os níveis e as comparações acima, no protocolo descrito;
- a distinguibilidade de MAR e MNAR sob a classe injetada, se a H6 passar;
- a perda fora do gerador;
- que o MechDetect sem acesso a Y não decide MNAR, por construção.

**A tese não pode afirmar:**
- que o método identifica o mecanismo em dados com ausência natural;
- uma acurácia "na prática";
- que "supera o MechDetect" sem qualificar o regime de informação;
- que generaliza para MNAR por variável latente, MAR com vários drivers, várias colunas incompletas ou Z com ausências;
- qualquer "teto".

## 9. Desvios e proteções

- **Desvios:** qualquer mudança de especificação depois da tag `prereg-v1` entra em `docs/revisao-banca/desvios.md` (data, o quê, por quê, impacto esperado) e gera um **novo** run, sem apagar o anterior. Resultados de runs com desvio são reportados como tal.
- **Manchete:** o run pré-registrado é sempre o resultado principal. Um run com desvio só o acompanha.
- **Invalidação:** um run só pode ser invalidado por erro de implementação contra o texto deste documento. Mesmo assim, os números dele são reportados, junto com o erro e a correção, como aconteceu com o piloto r1.
- **Ordem:** a ordem das etapas da produção está em `docs/revisao-banca/30-infra.md`. A etapa `cross-generator` vem antes do `stats`.

## 10. Registro externo e anterioridade

A tag no git sozinha não basta, porque uma tag pode ser apagada e recriada. Por isso, no dia da tag:
1. **Tag anotada `prereg-v1`** no commit que contém este documento e o código da análise, com push para o repositório público `JonathanTsen/ITA-Mestrado`.
2. **Registro de terceiros:** um release do GitHub dessa tag arquivado no Zenodo, que gera um DOI com data, ou um registro no OSF com este documento.
3. **E-mail ao orientador:** com o SHA do commit e o SHA-256 de `preregistro.md` e de `stats.py`, para que a data de recebimento sirva de anterioridade independente.

Os hashes e o link do registro entram em `plano-revisao-banca/execucao/38-preregistro.md`, no repositório da tese.
