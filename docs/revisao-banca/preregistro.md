# Pré-registro das análises confirmatórias (passo 38)

> **Situação: RASCUNHO para ratificação do orientador (D11).** A tag `prereg-v1` só é criada depois da resposta do Prof. Vitor Curtis. Até lá, nenhuma instância das famílias confirmatórias é gerada. Toda mudança depois da tag vai para `desvios.md` (item 9).

- **Rascunho:** 21/10/2026, revisado depois de uma revisão independente.
- **Configuração congelada:** tag `freeze-v1`, com `inject-v1.yaml`, os descritores 34b e as famílias. O adendo `data_rb/freeze_v1_addendum.json` registra o N efetivo.
- **Corpus sintético:** tag `synth-v2`.
- **Código da análise:** `src/missdetect/rb/stats.py`, entra no commit da tag.
- **Decisões:** `plano-revisao-banca/execucao/08-decisoes-tomadas.md`, no repositório da tese.
- **Versões:**
  - Python: scikit-learn 1.8.0, SciPy 1.17.1, NumPy 2.4.4, pandas 2.3.3, baycomp 1.0.3, mdatagen 0.2.0, tab-err 0.2.1;
  - R 4.6.1 com PKLMtest 1.0.1;
  - `uv.lock` com hash no manifesto.

## 1. Dados

**Fontes.** As 101 tabelas OpenML avaliadas pelo MechDetect, a partir dos CSV distribuídos com o código dos autores (`calgo-lab/MechDetect@43c0f00`). O SHA-256 de cada arquivo está em `data_rb/sources_local.csv` e é conferido a cada leitura.

**Filtros de tabela.** Ficam as tabelas:
- sem valores ausentes;
- com ≥ 500 linhas;
- com ≥ 3 colunas numéricas não constantes, depois de remover a coluna-alvo do benchmark original e as colunas de identificador (nome com "id" como palavra, ou inteiros únicos por linha).

Resultado: 98 de 101 tabelas.

**Famílias.** Duas tabelas são candidatas a família quando a sobreposição de nomes de colunas é ≥ 0,6, com ≥ 3 colunas em comum. Fundem-se quando a contenção média de valores nas colunas comuns é ≥ 0,9, além das decisões manuais em `data_rb/family_overrides.csv`. O agrupamento é transitivo, e cada família tem um representante:
- critério: o maior nº de colunas numéricas não constantes; desempate pelo maior n e depois pelo menor ID;
- **69 famílias**, das quais 8 de desenvolvimento, sorteadas com `child_rng(20261021, "development_families")`;
- as famílias de desenvolvimento são 37, 151, 251, 1049 (representante 1050), 44089, 44124, 44145 e 44980. Foram usadas no piloto e **nunca** entram nas análises confirmatórias.

**Colunas sentinela.** Uma coluna sai da tabela quando um valor codifica "não medido", seja por documentação da fonte, seja por ser fisicamente impossível pela definição da variável.
- A lista está em `data_rb/sentinel_exclusions.csv`. Hoje tem só a Pima, que é de desenvolvimento: `plas`, `pres`, `skin`, `insu` e `mass` iguais a 0.
- Não há regra automática por massa num valor de fronteira.
- Se os metadados do OpenML (pendentes) revelarem outra coluna desse tipo numa tabela confirmatória, ela entra na lista **antes** da produção, com registro em `desvios.md`.

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
- **Secundários**, nos contextos com n = min(1000, nº de linhas):
  - **mdatagen 0.2.0** em pares determinísticos: lower → `lowest`/`run_0`; upper → `highest`/`run_1`; both → `mix`/`run_0.5`; o MCAR usa `random`. A mistura de variantes fica assim pré-especificada;
  - **tab_err 0.2.1:** ECAR/EAR/ENAR semeados, só na coluna-alvo.
- **Excluído:** o MBOV da mdatagen, por déficit de taxa (0,15–0,23 p.p. com r = 0,1; 1,73–2,22 com 0,3; 4,42–6,41 com 0,5).
- **Trios desiguais:** um trio (MCAR, MAR, MNAR) de um contexto e gerador é recusado se as contagens de ausências diferirem. Isso acontece na mdatagen `mix`/`run_0.5` com N ímpar e pode desbalancear um pouco as direções do gerador secundário; os trios recusados são reportados.

**Separação observado × oráculo.** Os métodos recebem só `y` (com NaN) e `z`. IDs, caminhos e o oráculo nunca entram em descritores nem em prompts.

## 2. Pipeline primário (configuração primária)

- **Descritores 34b:** 28 descritores, grupos G0–G4 (`configs/rb/feature_schema_34b.json`, SHA-256 `d513af8f06f8352c1f9d437bce96447d2ef328c31ff252b997d18ac9a6ec2b95`).
- **Aprendiz primário (D7):** `HistGradientBoostingClassifier` com os hiperparâmetros padrão do scikit-learn 1.8.0, `early_stopping=False`, `random_state` fixo e **sem grade**.
- **Calibração:** nenhuma, Platt ou isotônica (um-contra-o-resto, normalizada). A escolha é feita pelo menor log-loss em predições fora-da-dobra da CV interna (4 folds por família), com cross-fitting do calibrador.
- **Configuração primária:** classificador **direto** (3 classes).
- **Hierárquico (RQ1/H2):**
  - L1 (MCAR × não-MCAR) treinado em todas as instâncias de treino;
  - L2 (MAR × MNAR) treinado nas instâncias de treino verdadeiramente não-MCAR;
  - mesmo aprendiz, mesmos descritores e mesma regra de calibração;
  - roteamento **fullprob**: (P(MCAR), P(MAR), P(MNAR)) = (p₁, (1 − p₁)·p₂, (1 − p₁)·(1 − p₂)), com p₁ = P(MCAR) do L1 e p₂ = P(MAR | não-MCAR) do L2.
- **Validação cruzada:** CV externa de 5 folds por família, com 3 repetições em atribuições diferentes. A unidade de teste é a família inteira.
- **Escopo do treino:** a análise primária treina e testa **só nas instâncias do gerador principal** (`evaluate --generators symmetric`).
- **Falhas:** uma falha de extração, de baseline ou de chamada conta como erro (probabilidades uniformes, classe −1). Nunca é descartada.

## 3. Estimando primário e métricas

**Escore por família.** Macro-recall das instâncias da família em cada repetição, depois a média das 3 repetições.

**Estimandos:**
- **de nível:** a média, entre as 50 famílias efetivas, do escore por família da configuração primária, com o gerador principal;
- **comparativo** (H1–H4): o **Hodges–Lehmann** das diferenças por família, ou seja, a mediana das médias de Walsh. A diferença média ± dp é reportada só como descrição.

**Métricas secundárias:** F1-macro, recall por classe, log-loss, Brier multiclasse, ECE por classe (10 bins de igual massa) e AUC um-contra-o-resto. No desenho balanceado, a acurácia coincide com o macro-recall e não conta como evidência independente.

## 4. Hipóteses confirmatórias (família de Holm, m = 4, α = 0,05; todas bilaterais)

- **H1** (objetivo específico 3): a configuração primária e o **MechDetect-O** diferem.
  - O MechDetect-O é uma adaptação observed-only, pré-especificada, do MechDetect: teste 1 sobre [Z, Ŷ(Z)] × máscara permutada; teste 2 sobre [Z, Ŷ] × Z; Ŷ por cross-fit só nas linhas observadas; HGB(200), 10 folds, AUC sobre `predict()` como no código dos autores; Mann-Whitney unilateral com α/2.
  - **Expectativa declarada:** sem os valores removidos, o teste 2 compara funções de Z com Z e quase nunca rejeita, então o MechDetect-O tende a decidir MAR entre os não-MCAR. Espera-se que a configuração primária o supere, sobretudo nas instâncias MNAR. Um resultado favorável **não** indica erro do MechDetect publicado, que usa informação não observável na prática.
  - O **painel oráculo** (MechDetect como publicado, com o Y completo) é reportado à parte e nunca ranqueado com os métodos observed-only.
- **H2** (RQ1): o hierárquico (fullprob) e o direto, com o mesmo aprendiz e os mesmos descritores, diferem. A equivalência prática é avaliada pela ROPE (item 5).
- **H3** (objetivo específico 3): a configuração primária e o classificador uniforme diferem. O uniforme tem macro-recall de exatamente 1/3 em toda família, então este é, na prática, um teste de uma amostra contra 1/3.
- **H4** (RQ4): o **LLM-A1** do provedor confirmatório e a configuração primária diferem, no painel de LLM.
  - **Provedor confirmatório:** Google (Gemini). O OpenAI (GPT) roda como secundário descritivo.
  - **Painel:**
    - 30 das 50 famílias efetivas, sorteadas com `child_rng(20261101, "llm_panel")`;
    - em cada família, os 6 contextos do gerador principal com r ∈ {0,1; 0,3; 0,5} × k ∈ {2, maior k disponível} e n = min(1000, nº de linhas);
    - total: 540 instâncias.
  - **Escores:**
    - LLM: macro-recall por família nas 18 instâncias do painel;
    - configuração primária: as predições da CV principal **restritas às mesmas 18 instâncias** (escore por repetição, média das 3).
  - **Adendo LLM, com tag `prereg-v1-llm`, antes de qualquer chamada confirmatória.** Congela:
    - IDs e datas dos modelos (o mais capaz, de uso geral, não-preview de cada provedor);
    - prompt e renderizador por hash (tabela cega: colunas `c0…ck`, transformação afim positiva por coluna, 3 algarismos significativos, ≤ 200 linhas sorteadas, linhas e colunas permutadas);
    - esquema JSON;
    - temperatura 0 ou o menor esforço de raciocínio;
    - até 2 retentativas; saída inválida = falha = erro;
    - critério de memorização: as sondas do `tabmemcheck` e a pergunta "qual é o nome desta base?".
  - **Se a H4 não rodar** (sem chaves ou corte de orçamento): ela entra no Holm com p₄ = 1, e m continua 4.
  - MDE da H4, pelo piloto: ≈ 5,4 p.p. com N = 30.

## 5. Testes, estimativas e decisão

- **Teste:** Wilcoxon de postos sinalizados sobre as diferenças por família (`scipy.stats.wilcoxon`, `zero_method="pratt"`, `method="asymptotic"`, `correction=False`, bilateral), com **Holm** sobre H1–H4.
  - O Holm controla o erro do tipo I sob dependência arbitrária, inclusive com N diferentes: 50 em H1–H3 e 30 na H4.
- **Efeito:**
  - Hodges–Lehmann com **IC95% percentil por bootstrap de famílias** (10.000 réplicas, semente 20261101). O IC **não** é ajustado por multiplicidade; o p é ajustado;
  - correlação rank-biserial pareada (Kerby): (soma dos postos de |d| > 0 − soma dos postos de |d| < 0) / soma total, sem as diferenças nulas.
- **Teste bayesiano de postos sinalizados:** `baycomp.SignedRankTest.probs(comparador, método, rope=0.02, prior=0.5, nsamples=50000, random_state=20261101)`.
  - Como o pacote usa a diferença `método − comparador`, a região **direita = método melhor**.
  - As probabilidades são as frações da posterior em que cada região (esquerda, ROPE, direita) é a mais provável.
- **Tabela de decisão** (única; implementada em `stats.decide`):
  - **superior / inferior:** p de Holm < 0,05, com o sinal do Hodges–Lehmann;
  - **praticamente equivalentes:** P(ROPE) ≥ 0,95, mesmo sem significância;
  - **superior/inferior, dentro da ROPE:** as duas condições anteriores ao mesmo tempo;
  - **evidência insuficiente:** nos demais casos.
- **Desfecho esperado da H2:** "evidência insuficiente" é um desfecho plausível. A ROPE do baycomp avalia a dispersão das diferenças entre famílias, e numa simulação com N = 50 (200 réplicas, diferenças normais) a equivalência foi declarada em 58% dos casos com média 0 e dp 0,04, em 9% com o perfil do piloto (média −0,015, dp 0,04) e em 0% com dp 0,06.
- **Poder (piloto r2):** sd_Δ de 0,04 a 0,09, o que dá MDE ≤ 4,2 p.p. com N = 50 (aproximação normal, Holm m = 4, poder 0,8).

## 6. Análises secundárias (cada uma com o seu próprio Holm) e exploratórias

**Secundária, L1** (m = 3):
- **Classificador L1:** o L1 do hierárquico (`hier_hgb_fullprob`), que decide não-MCAR quando P(MCAR) < 0,5.
- **Comparadores:** teste de Little (α = 0,05), teste 1 do MechDetect-O (α/2) e PKLM (α = 0,05; 100 projeções e 200 repetições; `set.seed(20261101)` antes de cada instância).
- **Subconjunto do PKLM:** 8 contextos inteiros do gerador principal por família, 1.200 instâncias, escolhidos com `child_rng(20261101, "pklm_subset")` sobre os contextos ordenados por ID.
- **Métricas:** sensibilidade, especificidade e acurácia balanceada (AB) por família. O teste usa a AB, com Wilcoxon-Pratt e Holm.
- A comparação com o PKLM usa só o seu subconjunto.
- A especificidade dos testes é reportada **como medida**: no piloto, o PKLM rejeitou 12,5% sob MCAR.

**Secundária, RQ2** (m = 4): para k = 1…4, configuração primária com todos os descritores × sem o grupo G_k. Escore por família, Wilcoxon-Pratt e Holm. O SHAP é só descritivo.

**Secundária, RQ3** (m = 4; D2):
- **Contrastes:** ponderação × nenhuma e poda × nenhuma, para ruído simétrico e MAR ↔ MNAR, com taxa de 0,20.
- **Escore:** macro-recall no teste limpo, por família.
- As curvas com as taxas {0; 0,1; 0,2; 0,4} são descritivas.

**Secundária, RQ4-A2** (m = 1): pipeline + descritores do LLM × pipeline, no painel.

**Exploratórias:**
- por mecanismo: recall por classe, com Wilcoxon-Holm dentro dessa família;
- geradores secundários, cada um com a sua própria CV;
- leave-one-generator-out;
- ensaios por fator, com forest plots e IC por bootstrap, sem p por célula (passo 40). Não há modelo misto confirmatório;
- roteamentos hard, threshold e soft:
  - threshold: θ ∈ {0,20; 0,25; …; 0,80}, escolhido pelo macro-recall no OOF interno;
  - soft: zonas (a, b) ∈ {0,2; 0,3; 0,4} × {0,6; 0,7; 0,8}, pelo mesmo critério;
- calibração;
- classificadores descritivos com Friedman + Nemenyi (diagrama CD): HGB, logreg, RF, SVM, KNN, MLP e NB;
- transferência sintético → real (`synth-v2`);
- LLM-A3 e o provedor secundário;
- sensibilidade sem os contextos cujo Y tem massa ≥ 10% num valor de fronteira;
- controle sintético: CV agrupada por tabela-base no `synth-v2`;
- avaliação do L2 diagnóstica (nas instâncias verdadeiramente não-MCAR) e operacional (nas roteadas);
- painel oráculo: o MechDetect como publicado e o nosso classificador com **descritores-oráculo** (34b + AUC de M predito do Y completo + [AUC(M | Y, Z) − AUC(M | Z)], por CV com HGB), mais o regime "dirty" do MechDetect, reportado como degenerado.

**Auditoria de atalhos nas famílias confirmatórias:** se o classificador só com nuisance passar de 0,40 de macro-recall, isso é reportado como alerta, sem mudar a análise confirmatória.

**Fora do escopo:** a variante "MNAR latente" do plano (passo 32) não será executada.

**Estudo de caso (D3):** dados com ausência natural, **sem** acurácia nem concordância como métrica de acerto.

## 7. Situação da implementação

| Componente | Situação |
|---|---|
| Injeção, descritores 34b, avaliador, baselines (MechDetect-O, oráculo, Little, PKLM), relatório | Implementados e testados |
| `stats.py` (H1–H4, Holm, Hodges–Lehmann, bootstrap, rank-biserial, ROPE, decisão) | Implementado e testado; congelado com a tag `prereg-v1` |
| L1 secundário (teste pareado), ablação da RQ2, descritores-oráculo, regime dirty, leave-one-generator-out, transferência, diagrama CD, sensibilidade da massa na fronteira | **A implementar antes da produção**, exatamente como especificado acima |
| LLM (passos 42/43) e ruído (passo 44) | **A implementar.** O adendo LLM, com tag, vem antes de qualquer chamada confirmatória |

Implementar um componente da tabela acima **não** é desvio. Mudar a especificação é, e vai para `desvios.md`.

## 8. Regras de redação (VIT-06; R3 do passo 25)

Toda comparação usa exatamente um destes níveis:
- **só estimativa pontual:** "the point estimate of X is higher than that of Y in this experiment";
- **teste significativo após Holm:** "X was statistically superior to Y on the N evaluated dataset families under protocol P (Hodges–Lehmann difference d, 95% CI [a, b], adjusted p)";
- **não significativo:** "there is insufficient evidence of a difference between X and Y";
- **equivalência** (só pela ROPE): "X and Y are practically equivalent within ±δ (posterior probability …)".

As conclusões valem só para as famílias, os geradores e as condições avaliadas.

## 9. Desvios

Qualquer mudança de especificação depois da tag `prereg-v1` entra em `docs/revisao-banca/desvios.md` (data, o quê, por quê, impacto esperado) e gera um **novo** run, sem apagar o anterior. Resultados de runs com desvio são reportados como tal.
