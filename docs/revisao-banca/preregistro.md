# Pré-registro das análises confirmatórias (passo 38)

> **Situação: RASCUNHO para ratificação do orientador (D11).** A tag `prereg-v1` só é criada depois da resposta do Prof. Vitor Curtis. Até lá, nenhuma instância das famílias confirmatórias é gerada. Toda mudança depois da tag vai para `desvios.md` (item 8).

- **Data do rascunho:** 21/10/2026.
- **Configuração congelada:** tag `freeze-v1`.
  - Benchmark: `configs/rb/inject-v1.yaml`.
  - Descritores: `configs/rb/feature_schema_34b.json`.
  - Corpus sintético: tag `synth-v2`.
- **Decisões que este documento implementa:** `plano-revisao-banca/execucao/08-decisoes-tomadas.md` (repositório da tese).

## 1. Dados

- **Fontes:** as 101 tabelas OpenML avaliadas pelo MechDetect (CSV distribuídos com o código dos autores, `calgo-lab/MechDetect@43c0f00`, com SHA-256 de cada arquivo em `data_rb/sources_local.csv`).
- **Filtros de tabela:**
  - sem valores ausentes;
  - ≥ 500 linhas;
  - ≥ 3 colunas numéricas não constantes, depois de remover a coluna-alvo do benchmark original e as colunas de identificador.
  
  Resultado: 98 de 101 tabelas incluídas.
- **Famílias:** versões e derivações da mesma base formam uma família (sobreposição de nomes de colunas ≥ 0,6 e contenção de valores ≥ 0,9, mais as decisões manuais de `data_rb/family_overrides.csv`). Um representante por família. São 69 famílias:
  - **desenvolvimento (8):** 37, 151, 251, 1049 (representante 1050), 44089, 44124, 44145, 44980. Usadas no piloto; **nunca** entram nas análises confirmatórias;
  - **confirmatórias (61)**, no formato família:representante: 6, 23, 31, 44, 54, 310, 725, 823, 1046, 1461, 1464, 1480, 1494, 1510, 4135, 4534, 23517, 40701, 40994, 42493, 44026:45012, 44054, 44055, 44056, 44059, 44061, 44062, 44063, 44064, 44066:44989, 44091:44136, 44122:44133, 44123:44139, 44125, 44127, 44130:44157, 44131, 44132, 44134:44137, 44147:44983, 44148:44964, 44158, 44160, 44162, 44956, 44957, 44958, 44959, 44960, 44962, 44963, 44966, 44967, 44969, 44970, 44974, 44976, 44981, 44993, 44994, 45402.
- **Colunas sentinela:** uma coluna sai da tabela se um valor codifica "não medido", seja por documentação da fonte, seja por ser fisicamente impossível pela definição da variável. A lista está em `data_rb/sentinel_exclusions.csv`; hoje inclui só a Pima, que é de desenvolvimento. Se os metadados do OpenML (pendentes) revelarem outra coluna assim numa tabela confirmatória, ela entra na lista **antes** da produção, com registro em `desvios.md`. Não há regra automática por massa num valor de fronteira.
- **Elegibilidade do par (Y, Z)**, calculada na tabela completa antes de qualquer máscara:
  - Y e o driver com ≥ 20 valores distintos e moda ≤ 50% das linhas;
  - R² de Y predito de todas as auxiliares candidatas (HistGradientBoosting, CV de 5 folds) ≤ 0,95;
  - máximo |Spearman(Y, Z_j)| ≤ 0,95.
  
  Bases inelegíveis são sorteadas de novo (até 20 tentativas), e as rejeições são reportadas por regra.
- **Desenho por família:**
  - **bases:** uma base por taxa r ∈ {0,1; 0,2; 0,3; 0,4; 0,5}, com alvo, driver e ordem das auxiliares sorteados; regra do driver (maxcorr ou random) balanceada;
  - **subamostras aninhadas:** n ∈ {500, 1000, 2000}, limitado ao tamanho da tabela; níveis que colidem são descartados e contados;
  - **auxiliares aninhadas:** k ∈ {2, 4, 8}, com o driver sempre incluído e a ordem das colunas observadas sorteada;
  - **força e direção:** β ∈ {1, 2, 4, 8, ∞} e direção ∈ {lower, upper, both}, balanceados nas células (base, n);
  - a mesma máscara serve aos três níveis de k.
- **Geradores:**
  - **principal = simétrico** (Gumbel-top-k com pesos exp(β·s), contagem exata; β = 0 é o MCAR);
  - **secundários:** mdatagen 0.2.0 em pares determinísticos (lower: `lowest`/`run_0`; upper: `highest`/`run_1`; both: `mix`/`run_0.5`; MCAR `random`) e tab_err 0.2.1 (ECAR/EAR/ENAR semeados, só na coluna-alvo), nos contextos com n = 1000;
  - o MBOV da mdatagen está excluído (déficit de taxa de 1,7 a 6,4 p.p.);
  - um trio (MCAR, MAR, MNAR) de um contexto e gerador é recusado se as contagens de ausências diferirem.
- **Separação observado × oráculo:** os métodos recebem só `y` (com NaN) e `z`. IDs, caminhos e o oráculo nunca entram em descritores nem em prompts.

## 2. Pipeline primário (configuração primária)

- **Descritores 34b:** 28 descritores, grupos G0–G4 (`configs/rb/feature_schema_34b.json`, SHA-256 registrado na tag `freeze-v1`).
- **Aprendiz primário (D7):** `HistGradientBoostingClassifier` com os hiperparâmetros padrão do scikit-learn, `early_stopping=False` e semente fixa. **Sem grade de hiperparâmetros.**
- **Calibração:** nenhuma, Platt ou isotônica (um-contra-o-resto). A escolha é feita pelo log-loss em predições fora-da-dobra da CV interna (4 folds por família, cross-fitting).
- **Configuração primária:** classificador **direto** (3 classes) com o aprendiz, os descritores e a calibração acima.
- **Hierárquico (RQ1/H2):**
  - L1 (MCAR × não-MCAR) treinado em todas as instâncias de treino;
  - L2 (MAR × MNAR) treinado nas instâncias de treino verdadeiramente não-MCAR;
  - mesmo aprendiz e mesmos descritores;
  - roteamento **fullprob**: p = (1 − q, q(1 − r), q·r).
- **CV externa:** 5 folds por família × 3 repetições com atribuições diferentes. A unidade de teste é a família inteira.
- **Treino e teste:** a análise primária treina e testa **só nas instâncias do gerador principal**. Cada gerador secundário tem a sua própria CV, e o leave-one-generator-out é ensaio (passo 40).
- **Falhas:** uma falha de extração, de baseline ou de chamada conta como erro (probabilidades uniformes, classe −1). Nunca é descartada.

## 3. Estimando primário

**Média, entre as 61 famílias confirmatórias, do macro-recall por família** (média das 3 repetições), na configuração observed-only e com o gerador principal. Métricas secundárias:
- acurácia;
- F1-macro;
- recall por classe;
- log-loss;
- Brier multiclasse;
- ECE por classe (10 bins de igual massa);
- AUC um-contra-o-resto.

## 4. Hipóteses confirmatórias (família de Holm, m = 4, α = 0,05; todas bilaterais)

- **H1** (estado da arte; objetivo específico 3): a configuração primária e o **MechDetect-O** diferem.
  - **Expectativa declarada:** o MechDetect-O tende a decidir MAR por construção. Sem os valores removidos, o seu segundo teste compara funções de Z com Z. Espera-se que a configuração primária o supere, sobretudo nas instâncias MNAR. A H1 mede o ganho de usar descritores observáveis, não uma falha de implementação do MechDetect.
  - O **painel oráculo** (MechDetect como publicado, com o Y completo) é reportado à parte e nunca ranqueado com os métodos observed-only.
- **H2** (RQ1): o hierárquico (fullprob) e o direto, com o mesmo aprendiz e os mesmos descritores, diferem. Reporta-se também a equivalência prática (item 5).
- **H3** (comparação com o acaso; objetivo específico 3): a configuração primária e o classificador uniforme (1/3) diferem.
- **H4** (RQ4): o **LLM-A1** (classificador direto, cego) e a configuração primária diferem, no **painel de LLM**:
  - **famílias:** 30 famílias confirmatórias sorteadas com `child_rng(20261101, "llm_panel")`;
  - **contextos:** em cada família, os 6 contextos do gerador principal com r ∈ {0,1; 0,3; 0,5} × k ∈ {2, maior k disponível (8, ou menos se a tabela tiver menos auxiliares)} e n = 1000 (limitado ao tamanho da tabela);
  - **total:** 30 × 6 × 3 = 540 instâncias.
  - Os IDs e as datas dos modelos (D5) são registrados num adendo **antes** de rodar o painel.

## 5. Testes e estimativas

- **Teste:** Wilcoxon de postos sinalizados sobre os escores por família (`scipy.stats.wilcoxon`, `zero_method="pratt"`, bilateral), com Holm sobre H1–H4.
- **Efeito:**
  - estimador de Hodges–Lehmann da diferença (mediana das médias de Walsh), com IC95% por bootstrap de famílias (10.000 réplicas, semente 20261101);
  - correlação rank-biserial pareada.
- **Teste bayesiano de postos sinalizados** (`baycomp`) com ROPE de ±0,02 em macro-recall. Regra de decisão:
  - "praticamente equivalentes" se P(ROPE) ≥ 0,95;
  - "superior" se P(diferença > 0,02) ≥ 0,95;
  - indeterminado nos demais casos.
- **Poder (piloto r2):** sd_Δ de 0,04 a 0,09, o que dá MDE ≤ 3,8 p.p. com N = 61 (aproximação normal; ver `execucao/37-piloto.md`).

## 6. Análises secundárias e exploratórias (rotuladas assim no texto)

- **Secundária, L1** (família de Holm própria, 3 comparações): o L1 do hierárquico (argmax de MCAR × não-MCAR) contra:
  - o **teste de Little** (α = 0,05);
  - o **teste 1 do MechDetect-O** (α/2);
  - o **PKLM** (α = 0,05; 100 projeções, 200 repetições; subconjunto pré-especificado de 6 contextos inteiros por família, 1.098 instâncias, escolhidos com `child_rng(20261101, "pklm_subset")`).
  
  Métricas: sensibilidade, especificidade e acurácia balanceada por família; o teste usa a acurácia balanceada. A especificidade dos testes é reportada **como medida**, sem supor o nível nominal: no piloto, o PKLM rejeitou 12,5% sob MCAR.
- **Secundária, RQ2:** ablação pareada por grupo de descritores (G1, G2, G3, G4; todos menos um), com Holm próprio. O d de Cohen e o SHAP são só descritivos.
- **Secundária, RQ3 (D2):** ruído controlado nos rótulos de treino, com família de testes própria (passo 44).
- **Secundária, RQ4 A2:** pipeline + descritores do LLM × pipeline, no painel de LLM.
- **Exploratórias:**
  - por mecanismo (Wilcoxon-Holm dentro dessa família);
  - ensaios por fator (forest plots com IC por bootstrap, sem p por célula; passo 40);
  - Friedman + Nemenyi (diagrama CD) só **descritivo** entre os classificadores logreg, RF, SVM, KNN, MLP e NB;
  - os roteamentos hard, threshold e soft;
  - calibração;
  - transferência sintético → real (corpus `synth-v2`);
  - LLM-A3;
  - sensibilidade sem os contextos cujo Y tem massa ≥ 10% num valor de fronteira;
  - controle sintético (CV agrupada por tabela-base no `synth-v2`).
- **Estudo de caso (D3):** dados com ausência natural, **sem** acurácia nem concordância como métrica de acerto.

## 7. Regras de redação (VIT-06; R3 do passo 25)

Toda comparação usa exatamente um destes níveis:
- **só estimativa pontual:** "the point estimate of X is higher than that of Y in this experiment";
- **teste significativo após Holm:** "X was statistically superior to Y on the N evaluated dataset families under protocol P (median difference d, 95% CI [a, b], adjusted p)";
- **não significativo:** "there is insufficient evidence of a difference between X and Y";
- **equivalência** (só pelo ROPE): "X and Y are practically equivalent within ±δ (posterior probability …)".

As conclusões valem só para as famílias, os geradores e as condições avaliadas.

## 8. Desvios

Qualquer mudança depois da tag `prereg-v1` entra em `docs/revisao-banca/desvios.md` (data, o quê, por quê, impacto esperado) e gera um **novo** run, sem apagar o anterior. Resultados de runs com desvio são reportados como tal.
