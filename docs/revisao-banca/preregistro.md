# Pré-registro das análises confirmatórias (passo 38) — versão 2

> **Situação: versão 2, aprovada pelo orientador (Prof. Vitor Curtis) em 29/10/2026 na versão `444d1d8`, com as emendas de 29/10/2026 listadas na seção 0.4, que voltam a ele antes da tag.**
> - A tag `prereg-v1` só é criada depois da aprovação das emendas. Até lá, nenhuma instância das famílias confirmatórias é gerada.
> - A versão 1 (21–22/10) foi revista criticamente antes de qualquer dado confirmatório. O que mudou e por quê está na seção 0.3.
> - Toda mudança depois da tag vai para `desvios.md` (seção 9).

- **Configuração congelada:** tag `freeze-v1`, com `inject-v1.yaml`, os descritores 34b e as famílias. O adendo `data_rb/freeze_v1_addendum.json` registra o N efetivo.
- **Corpus sintético:** tag `synth-v2`.
- **Código da análise:** `src/missdetect/rb/stats.py` (decisão), `analyses.py` (AUC MAR × MNAR, testes por gerador) e `pipeline.py` (etapas `cross-generator`, `secondary` e `stats`). Entram no commit da tag.
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
| Só-nuisance | Classificador que só vê fatores do desenho e da instância que não informam o mecanismo (taxa de ausência, n, k, nº de valores distintos de Y observado e o gerador); auditoria de atalhos |
| Macro-recall | Média dos recalls das três classes; com classes balanceadas, igual à acurácia |
| HL | Estimador de Hodges–Lehmann da diferença entre métodos |
| Holm | Correção para múltiplas hipóteses na mesma família |
| TOST | Teste de equivalência por dois testes unilaterais |
| δ | Margem de equivalência e de relevância prática: 0,02 de macro-recall (seção 5) |
| Desenvolvimento / confirmatória | 8 famílias usadas no piloto e nos ensaios; 50 famílias efetivas guardadas para a prova |
| LLM-A1, A2, A3 | LLM como classificador direto (A1), como extrator de descritores (A2) e com metadados semânticos (A3, exploratório) |

### 0.3 O que foi visto antes deste documento e o que mudou por causa disso

**Nas famílias confirmatórias, só se viu:**
- a elegibilidade calculada nas tabelas completas, sem nenhuma máscara (inclui o R² de um HGB e a correlação de Spearman entre o alvo e as auxiliares, que definem os pares elegíveis);
- a distribuição de valores das colunas elegíveis dos 69 representantes (massa em zero ou no mínimo), na varredura de zeros sentinela de 21/10 (`scripts/revisao_banca/decisoes/d11_zeros_sentinela.py`). Ela levou a não adotar uma regra automática; a única exclusão (Pima) é de desenvolvimento;
- as descrições, os nomes e as contagens do OpenML (instâncias, atributos e ausências), usadas para conferir os CSV-fonte;
- o agrupamento em famílias;
- a reexecução do código oficial do MechDetect (passo 36, 21/10/2026) em 12 tabelas pequenas. Sete delas são representantes de famílias confirmatórias efetivas: 1464, 1480, 40994, 44957, 44959, 44962 e 44970. A reexecução incluiu a injeção do tab_err pelo código dos autores e as AUCs e decisões do MechDetect, vistas só em agregado;
- o cubo de resultados publicado pelos autores do MechDetect, com as 101 tabelas (E1).

Nenhum desses cálculos usou o gerador, os descritores ou os modelos deste documento. O segundo e o terceiro itens foram acrescentados na auditoria de 28/10, que encontrou a varredura de sentinelas fora desta lista. Os dois últimos foram acrescentados na verificação de defensabilidade de 29/10 (emenda P5).

Nenhuma instância confirmatória deste protocolo foi gerada, nenhum modelo deste documento foi treinado com elas e nenhum escore dos métodos deste documento foi calculado nelas.

**Nas 8 famílias de desenvolvimento (e no corpus sintético), viram-se:**
- os pilotos r1 (invalidado por um defeito de desempate) e r2;
- a checagem da configuração congelada;
- o ensaio geral de 22/10, com a cadeia completa, a RQ3 e o teste de fumaça do estudo de caso;
- na verificação de defensabilidade de 29/10, recálculos sobre as predições do ensaio, que motivaram as emendas da seção 0.4:
  - **MNAR por censura:** com β = ∞, o recall de MNAR da configuração primária foi 0,24 (0,11 na direção inferior), e 60% dessas instâncias foram classificadas como MAR (72% na direção inferior). Com β = 2, o recall de MNAR foi 0,68 (0,83 na direção inferior); com β = ∞, o recall de MAR foi 0,97;
  - **H5:** os pares da mdatagen coincidem com o simétrico com β = ∞ (15 de 27 máscaras MAR e 15 de 27 MNAR idênticas, e as demais diferem só em empates no limiar). Cerca de 83% da diferença da H5 vem do recall de MCAR;
  - **H6:** a vantagem foi de +0,064 no gerador principal (7 de 8 famílias) e de +0,010 fora dele (2 de 8); na concordância dentro do trio, foi de +0,032;
  - **tamanho do PKLM:** 20 de 200 tabelas gaussianas com MCAR puro (E19).

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
| A H3 deixa de comparar com o classificador uniforme e passa a comparar com o **modelo logístico mínimo** (a logística simples) | Contra o uniforme, o resultado era conhecido (0,656 contra 0,333). Contra a logística simples, o desfecho está em aberto: HL = +0,007 no ensaio (diferença média +0,001) |
| H5 nova: **geradores externos**, contra a logística simples treinada no mesmo gerador | Treinado e testado no mesmo gerador, o desempenho pode refletir a assinatura do gerador. No ensaio, treinando só no simétrico e testando na mdatagen e no tab_err: configuração primária 0,684, logística simples 0,721 (HL −0,046), MechDetect-O 0,641; perda fora do gerador 0,164 [0,054; 0,243]. O MechDetect-O é fraco por construção, então ele vira comparador secundário |
| H6 nova: **MAR × MNAR**, contra a logística simples | É a questão central da área e não tinha teste confirmatório. Contra 0,5 o teste seria trivial (no ensaio, as 8 famílias ficaram acima de 0,5, mínimo 0,679; AUC média 0,83). Contra a logística simples: HL +0,069 [0,024; 0,095] |
| H2 com **equivalência por TOST**; a ROPE bayesiana fica só descritiva | Com a regra da v1, um Δ ≈ 0 terminaria em "evidência insuficiente" e não poderia ser afirmado |
| A H4 vira **família própria** | A H4 depende de chaves de API e de um adendo; não deve mexer no Holm das demais |
| Secundárias novas: painel oráculo por gerador, mdatagen com a própria CV, estatísticas do MechDetect como descritores, geradores externos × MechDetect-O | A comparação com o MechDetect como publicado não tinha teste. No ensaio, não houve diferença detectável entre ele e o nosso classificador oráculo no tab_err |
| Tabela de interpretação (5.2), proteções (9) e registro externo (10) | Evitar "mover a trave" e dar anterioridade verificável por terceiros |

Uma segunda revisão independente (27/10) trocou os comparadores da H5 e da H6 pelos descritos acima (antes eram o MechDetect-O e 0,5), passou a contar falhas como erro na H6 e completou a tabela de interpretação.

Essas mudanças foram feitas vendo só resultados de desenvolvimento. Elas tornam a família confirmatória **mais difícil**: H3 e H5 têm desfecho em aberto no ensaio (a H5 é desfavorável ao método), e a H6 deixa de ser um teste contra o acaso. Só a H1 continua esperada por construção, e isso está declarado.

### 0.4 Emendas depois da aprovação (29/10/2026)

A versão `444d1d8` foi aprovada pelo orientador em 29/10/2026, com o pedido de verificar se o desenho é defensável. A verificação foi feita no mesmo dia: três revisões adversariais, uma no papel de cada membro da banca, com cada achado conferido no arquivo. Ela está registrada em `plano-revisao-banca/execucao/38b-defensabilidade.md`, no repositório da tese.

As emendas abaixo vêm dessa verificação e das decisões do autor. **Nenhuma muda as hipóteses H1–H6, os comparadores ou os testes da família principal.** Mudam frases de leitura, descritivos, secundárias, as regras do LLM e as proteções. Todas foram feitas vendo só resultados de desenvolvimento e voltam ao orientador antes da tag.

| ID | O quê | Onde |
|---|---|---|
| P1 | D5: só o Google. O confirmatório segue a regra do adendo, o `gemini-3.8-flash` é o secundário descritivo, e o OpenAI saiu | §4 |
| P2 | Situação do documento e esta lista | Cabeçalho; §0.4 |
| P3 | Definição do run pré-registrado e trava no código | §9 |
| P4 | Segundo depósito com os hashes dos resultados e o arquivo da aprovação | §10 |
| P5 | O §0.3 completo (reexecução do MechDetect e cubo dos autores) | §0.3 |
| P6 | Recall por classe e AUC da H6 por β e direção; linha da censura determinística | §3; §5.2; §8 |
| P7 | H6 restrita ao gerador principal, com a H6 fora do gerador e a concordância dentro do trio | §4; §5.2 |
| P8 | Relação dos geradores externos com o simétrico; decomposições da H5 | §1; §4; §5.2; §8 |
| P9 | Linha da combinação H3/H5/H6 e regra do resumo com os seis desfechos | §5.2; §8 |
| P10 | δ como convenção, com o poder para \|d\| = δ; regras de leitura coerentes com o §8 | §5; §5.2 |
| P11 | Sensibilidade à partição, por repetição | §5 |
| P12 | Secundária dos erros que mudam a decisão do usuário. O experimento a jusante (passo 47) foi dispensado | §6; §8 |
| P13 | L1 sem limiar (AUC) e linha do L1 × Little | §6; §5.2 |
| P14 | Tamanho do PKLM pelo E19 | §6 |
| P15 | Sensibilidade sem as bases com Y nominal | §6 |
| P16 | As famílias são tabelas de benchmark, algumas de simulação | §8 |
| P17 | Escopo do MAR e do MNAR; exploratória opcional com vários drivers e MNAR misto | §6; §8 |
| P18 | Redação exata sobre as variantes da mdatagen | §1 |
| P19 | Estudo de caso só com a NHANES | §6 |
| P20 | Regras de geração do LLM (temperatura ignorada, raciocínio `medium`, uma resposta por instância) | §4 |
| P21 | Adendo LLM antes do `evaluate`; memorização pela renderização cega | §4 |
| P22 | Regra de custo do LLM | §4 |
| P23 | Linhas da H4 na tabela de interpretação | §5.2 |
| P24 | Secundária com um gerador externo estocástico: o ampute (Schouten, Lugtig e Vink, 2018) | §1; §6 |

**Proposta adicional de 30/10, encaminhada a pedido do autor e pendente do orientador:** H7, efeito do conjunto de informações (valores originais e contexto com faixas típicas aproximadas) em 30 bases inventadas novas, comparado à tabela cega. O desenho está no [adendo LLM, §10](adendo-llm.md#10-proposta-h7-efeito-do-contexto-em-bases-inventadas-30102026-encaminhamento-autorizado-pelo-autor-pendente-do-vitor-não-vale-como-registro). É uma família própria com um contraste, sem alteração de H1–H6. Os 16 domínios exploratórios já vistos não serão usados como confirmatórios. A proposta ainda não está lacrada, e não autoriza execução confirmatória.

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
  - na mdatagen 0.2.0, o MAR tem variantes estocásticas (`rank`, `median`), mas o MNAR univariado é determinístico na prática: `uMNAR.run(deterministic=False)` também ordena pela própria coluna, e o aleatório só desempata. A única variante MNAR estocástica que se aplica só à coluna-alvo, `MBOV_randomness`, entrega taxa abaixo da pedida (déficit de 0,15 a 6,41 p.p.) e quebraria a regra dos trios com a mesma contagem. As demais variantes MNAR da mdatagen não servem aqui: o `MBOUV` distribui as ausências entre as colunas a partir de uma variável oculta; o `MBIR` torna não observada a auxiliar que comanda a ausência (MNAR latente, fora do escopo); e o `MBOV_median` é determinístico e não tem par MAR;
  - usar a mdatagen como principal deixaria MAR estocástico contra MNAR determinístico, e o determinismo viraria atalho ("determinístico ⇒ MNAR").
  - O plano de resposta à banca previa a mdatagen como principal. Esta mudança vai declarada no documento de resposta, e a mdatagen entra em duas análises testadas: na H5 (confirmatória) e numa secundária com a própria CV (seção 6).
- **Externos**, nos contextos com n = min(1000, nº de linhas):
  - **mdatagen 0.2.0** em pares determinísticos: lower → `lowest`/`run_0`; upper → `highest`/`run_1`; both → `mix`/`run_0.5`; o MCAR usa `random`;
  - **tab_err 0.2.1:** ECAR/EAR/ENAR semeados, só na coluna-alvo.
- **Relação dos externos com o principal** (verificada no ensaio de 22/10; emenda P8):
  - os pares determinísticos da mdatagen selecionam as mesmas linhas que o gerador simétrico com β = ∞ na direção correspondente, a menos do desempate entre valores empatados no limiar;
  - o MCAR externo tem a mesma lei do MCAR simétrico;
  - só as janelas contíguas do tab_err em posição interior ficam fora do suporte do gerador principal.

  Por isso, a H5 mede a generalização para mecanismos determinísticos sob o prior de β do treino, e não o efeito de um gerador independente.
- **Externo estocástico, só para teste (secundária; emenda P24):** o procedimento de amputação de Schouten, Lugtig e Vink (2018), na implementação `pyampute` 0.0.3, nos mesmos contextos dos externos (seção 6).
- **Excluído:** o MBOV da mdatagen, por déficit de taxa (0,15–0,23 p.p. com r = 0,1; 1,73–2,22 com 0,3; 4,42–6,41 com 0,5).
- **Trios desiguais:** um trio (MCAR, MAR, MNAR) de um contexto e gerador é recusado se as contagens de ausências diferirem. Isso acontece na mdatagen `mix`/`run_0.5` com N ímpar; os trios recusados são reportados.

**Escopo (o que a validação cobre).**
- Uma coluna numérica com ausência por vez, com as auxiliares completas.
- Taxas de 10% a 50%; n entre 500 e 2.000; k entre 2 e 8; classes balanceadas.
- MAR pelo posto de um único driver observado.
- MNAR por auto-mascaramento: a ausência depende do próprio Y (inferior, superior ou nas duas caudas).
- **Geradores externos:** só as variantes **determinísticas** (os pares da mdatagen e o tab_err). A H5 mede a generalização para elas, não para mecanismos externos estocásticos. Um gerador externo estocástico, o ampute, entra só numa secundária (seção 6).
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
- **Escopo do treino:** a avaliação primária treina e testa **só nas instâncias do gerador principal** (`evaluate --generators symmetric`). Na H5, a configuração primária e a logística simples são treinadas no gerador principal das famílias de treino de cada fold, nas mesmas partições, e aplicadas sem ajuste aos geradores externos das famílias de teste (etapa `cross-generator`). O modelo da configuração primária é idêntico ao do fold correspondente da avaliação primária.
- **Falhas:** uma falha de extração, de baseline ou de chamada conta como erro (probabilidades uniformes, classe −1). Nunca é descartada.

## 3. Estimandos e métricas

**Escore por família.** Macro-recall das instâncias da família em cada repetição, depois a média das 3 repetições.

**Estimandos:**
- **de nível:** a média, entre as 50 famílias efetivas, do escore por família da configuração primária no gerador principal, com IC95% por bootstrap de famílias. A referência de acaso (1/3) é descritiva. É reportado **sempre com a quebra por β** e por taxa, porque depende da grade do desenho. O mesmo vale para a configuração primária nos geradores externos (H5);
- **por regime** (descritivo; emenda P6): o recall de cada classe e a AUC da H6 são reportados por β e, com β = ∞, por direção, com IC95% por bootstrap de famílias. É o que mostra se a censura determinística de Y (β = ∞) é reconhecida como MNAR;
- **comparativo** (hipóteses de diferença): o **Hodges–Lehmann** das diferenças por família, isto é, a mediana das médias de Walsh. A diferença média ± dp é só descritiva;
- **H6:** a AUC MAR × MNAR por família, da configuração primária e da logística simples. É a AUC de P(MNAR) / (P(MAR) + P(MNAR)) nas instâncias verdadeiramente não-MCAR (positivo = MNAR), média das 3 repetições. Uma falha conta como erro: recebe o pior escore para a sua classe verdadeira (MNAR → 0, MAR → 1). A AUC da configuração primária, com IC, entra no estimando de nível.

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
- **H3 — RQ2, configuração completa × modelo mínimo:** a configuração primária e o **modelo logístico mínimo** (a logística simples) diferem.
  - A comparação mistura aprendiz (HGB × logística) e conjunto de descritores (28 × 4). Por isso a frase na tese fala da configuração, não só dos descritores, e a decomposição é reportada como descritiva (HL com IC): aprendiz = primária × logística com os 28 descritores; descritores = logística com 28 × logística com 4. No ensaio: −0,010 e +0,006.
  - Com δ = 0,02 e o dp observado no ensaio (0,058), a equivalência com N = 50 não teria poder. Por isso a H3 só tem teste de diferença.
  - O classificador uniforme fica como referência descritiva do estimando de nível.
- **H5 — generalização para geradores externos:** nas instâncias da mdatagen e do tab_err das famílias de teste, a configuração primária e a logística simples, **ambas treinadas só no gerador principal**, diferem.
  - Pergunta: o conjunto completo é mais (ou menos) robusto à troca de gerador do que o modelo mínimo?
  - Os modelos são os de cada fold da avaliação primária, aplicados sem ajuste. O escore por família junta as instâncias dos dois geradores externos.
  - Descritivos, sem teste: o nível da configuração primária com IC, a quebra por gerador e a **perda fora do gerador**. A perda é o HL de (configuração primária treinada nos próprios geradores externos − treinada no principal), nas mesmas instâncias.
  - **Leitura (emenda P8):** os geradores externos são determinísticos, e os pares da mdatagen coincidem com o simétrico com β = ∞ a menos de empates (seção 1). Por isso, a H5 compara os modelos em mecanismos determinísticos sob o prior de β do treino e não isola o efeito de um gerador independente.
  - **Descritivos novos (emenda P8),** por gerador externo e no conjunto:
    - a **decomposição da diferença por classe** (recall de MCAR, MAR e MNAR; HL com IC), ao lado da mesma decomposição no gerador principal;
    - o **efeito do gerador:** primária no gerador externo − primária no recorte β = ∞ do simétrico, nas mesmas famílias de teste;
    - a **especialização:** primária treinada no próprio gerador externo − treinada no simétrico.
  - O MechDetect-O nas mesmas instâncias é uma secundária (seção 6).
- **H6 — MAR × MNAR (substituto testável do "teto"; RQ2, "em particular MAR de MNAR"):** a AUC MAR × MNAR por família da configuração primária e a da logística simples diferem (pareado).
  - Na escala de AUC, δ = 0,02 também é a margem de relevância prática.
  - A AUC da configuração primária contra 0,5 é descritiva (estimando de nível).
  - **O teste usa só o gerador principal** (`eval-symmetric`). **Descritivos novos (emenda P7),** sempre reportados com a H6:
    - a **H6 fora do gerador:** a mesma diferença de AUC, com os modelos da H5, nas instâncias dos geradores externos;
    - a **concordância dentro do trio:** a fração dos contextos em que a instância MNAR recebe P(MNAR) / (P(MAR) + P(MNAR)) maior que a instância MAR do mesmo contexto, com empate valendo 1/2. É calculada por família, nas duas configurações, com HL e IC. Ela separa a informação sobre o mecanismo da comparabilidade dos escores entre contextos, que a AUC agrupada mistura.

### Família da H4 (m = 1, α = 0,05)

- **H4 (RQ4):** o **LLM-A1** do provedor confirmatório e a configuração primária diferem, no painel de LLM.
  - **Provedor:** só o Google (Gemini). O modelo confirmatório segue a regra do adendo (item abaixo), que em 29/10/2026 apontava para o `gemini-4-argon`. O `gemini-3.8-flash` roda como secundário descritivo, a pedido do autor. O OpenAI (GPT), secundário na versão `444d1d8`, saiu em 29/10/2026 por decisão do autor, que não tem chave de API. Nenhuma hipótese muda.
  - **Painel:**
    - 30 das 50 famílias efetivas, sorteadas com `child_rng(20261101, "llm_panel")`;
    - em cada família, os 6 contextos do gerador principal com r ∈ {0,1; 0,3; 0,5} × k ∈ {2, maior k disponível} e n = min(1000, nº de linhas);
    - total: 540 instâncias.
  - **Escores:** o LLM tem o macro-recall por família nas 18 instâncias do painel. A configuração primária tem as predições da CV principal restritas às mesmas 18 instâncias (escore por repetição, média das 3).
  - **Informação desigual, declarada:** o LLM vê no máximo 200 linhas sorteadas; a configuração primária vê a instância inteira. Na sensibilidade pré-registrada, a configuração primária é **retreinada** com subamostras de ≤ 200 linhas das instâncias de treino, nas mesmas partições, e aplicada à mesma subamostra que o LLM recebe (aplicar o modelo treinado com n ≥ 500 a 200 linhas seria extrapolação contra a primária).
  - **Memorização (emenda P21):**
    - uma família sai da análise principal da H4 se as sondas aplicadas à **renderização cega** do A1 reconhecerem a base. É a renderização que o LLM de fato recebe;
    - o reconhecimento da tabela **crua**, com os nomes reais, é covariável. "LLM superior" só é afirmado se o sinal se mantiver nas famílias não reconhecidas na tabela crua;
    - famílias excluídas não são repostas, e o N efetivo e o MDE simulado para esse N acompanham a H4;
    - as sondas exatas (testes do `tabmemcheck` e a pergunta "qual é o nome desta base?"), o número de perguntas e os limiares são congelados no adendo LLM.
  - **Adendo LLM, com tag `prereg-v1-llm`, antes do `evaluate` da produção (emenda P21).**
    - Assim ele vem antes de qualquer escore confirmatório do classificador e de qualquer chamada confirmatória.
    - As sondas rodam, e a lista das famílias excluídas é publicada com SHA-256, também antes do `evaluate`.
    - Até lá, só rodam as etapas que não calculam predições: `inject`, `features` e `oracle-features`.
    - O adendo congela:
      - IDs e datas dos modelos: o confirmatório é o mais capaz, de uso geral e não-preview do provedor na data do adendo; o secundário descritivo é o `gemini-3.8-flash`;
      - prompt e renderizador por hash. A tabela é cega: colunas `c0…ck`; uma transformação afim positiva por coluna que leva os valores observados a inteiros de 0 a 999 (no máximo 3 algarismos significativos; emenda P20); ≤ 200 linhas sorteadas; linhas e colunas permutadas;
      - esquema JSON só com as três probabilidades;
      - **geração (emenda P20):**
        - os modelos Gemini 3.6+ e 4 ignoram temperatura, top_p e top_k e não aceitam semente. Nenhum desses parâmetros é enviado, e a saída é tratada como estocástica;
        - o nível de raciocínio é `medium`, o padrão do provedor, igual em todas as chamadas, sem ferramentas: execução de código e busca ficam desligadas;
        - vale a primeira resposta válida de cada instância, gravada crua num cache que nunca é apagado. O SHA-256 do cache entra no registro externo;
        - cada chamada grava o alias, a data e hora, o `modelVersion` e o `responseId`;
        - uma subamostra de 10% do painel (`child_rng(20261101, "llm_repeat")`) é consultada uma segunda vez, e a concordância (κ de Cohen e diferença de macro-recall) é descritiva;
        - o A1 confirmatório roda numa janela contínua de até 72 h. Se a versão do modelo mudar, nada é refeito, e a mudança é reportada;
        - o modo de lote do provedor pode ser usado, porque o modelo e a pergunta são os mesmos; isso é registrado;
      - até 2 retentativas, só para saída inválida; saída inválida = falha = erro;
      - o critério de memorização (item anterior).
    - Prompt e renderizador são desenvolvidos **só com famílias de desenvolvimento**; nenhum resultado das famílias confirmatórias informa o adendo.
  - **Poder:** com 18 instâncias por família, só o ruído binomial dá dp de ≈ 0,11 no escore de um método. O MDE fica entre ≈ 6 e 8 p.p. com N = 30; ele é recalculado por simulação no adendo. (A estimativa de 5,4 p.p. da v1 usava o dp do classificador e subestimava o MDE.)
  - **Custo (emenda P22):**
    - é projetado no piloto de desenvolvimento, no nível de raciocínio congelado, com o preço da data prevista;
    - se 1,5 × a projeção passar de US$ 300, cortam-se, nesta ordem, o A3, o modelo secundário e o A2;
    - o A1 roda primeiro, com as famílias em ordem sorteada (`child_rng(20261101, "llm_order")`). Se o orçamento acabar, a H4 usa as famílias completas, com o N e o MDE reportados.
  - **Se a H4 não rodar** (sem chaves ou corte de orçamento), a RQ4 fica sem resposta confirmatória, e isso é declarado. A família principal não muda.

## 5. Testes, estimativas e decisão

- **Teste de diferença:** Wilcoxon de postos sinalizados sobre as diferenças por família (`scipy.stats.wilcoxon`, `zero_method="pratt"`, `method="asymptotic"`, `correction=False`, bilateral).
- **Teste de equivalência (H2e):** TOST de Wilcoxon, com as mesmas opções. São dois testes unilaterais, sobre d + δ (H0: mediana ≤ −δ) e d − δ (H0: mediana ≥ δ); o p é o maior dos dois.
- **Holm** sobre os 6 testes da família principal. Ele controla o erro do tipo I sob dependência arbitrária, inclusive com N diferentes entre hipóteses. Uma hipótese que não puder rodar entra no Holm com p = 1, e m continua 6.
- **Efeito:**
  - Hodges–Lehmann com **IC95% percentil por bootstrap de famílias** (10.000 réplicas, semente 20261101). O IC **não** é ajustado por multiplicidade e é só descritivo: a decisão vem do p ajustado. Se o IC excluir zero com p ajustado ≥ 0,05, vale o p;
  - correlação rank-biserial pareada (Kerby), sem as diferenças nulas. O Pratt do teste inclui os zeros no ranqueamento; a diferença é declarada.
- **Teste bayesiano de postos sinalizados:** `baycomp.SignedRankTest.probs(comparador, método, rope=0.02, prior=0.5, nsamples=50000, random_state=20261101)`, com a região direita = método melhor. **Só descritivo.**
- **Tabela de decisão** (única; implementada em `stats.decide` e `stats.decide_equivalence`):
  - **teste de diferença:** "superior" ou "inferior" se o p de Holm < 0,05, pelo sinal do HL. Se |HL| < δ, o rótulo ganha "(efeito menor que δ)", isto é, "superior (efeito menor que δ)" ou "inferior (efeito menor que δ)": significativo, mas abaixo da relevância prática. Nos demais casos, "evidência insuficiente";
  - **TOST:** "equivalentes dentro de ±δ" se o p de Holm < 0,05; senão, "equivalência não demonstrada".
- **Por que δ = 0,02:**
  - numa família típica (135 instâncias, 45 por classe), cada instância vale 1/45 no recall da sua classe, ou 0,0074 no macro-recall. Então 0,02 equivale a ≈ 3 instâncias a mais acertadas;
  - é também a ordem do ruído de partição: no ensaio, a mediana do dp entre repetições foi 0,024 por repetição na configuração primária, o que dá ≈ 0,014 na média das 3 repetições que forma o escore.
  - **Natureza do δ (emenda P10):**
    - δ é uma convenção de resolução e de ruído do escore de uma família, não uma consequência a jusante demonstrada;
    - com N = 50 e o dp do ensaio, o poder para |d| = δ é ≈ 0,4–0,7 na H3, ≈ 0,2–0,4 na H5 e ≈ 0,1–0,3 na H1, conforme a posição no Holm;
    - por isso, "evidência insuficiente" nessas hipóteses não exclui diferenças da ordem de δ, e o IC95% do HL é sempre reportado ao lado.
- **Dependência entre famílias:** o escore de cada família vem de modelos treinados nas outras famílias (Nadeau e Bengio, 2003). O pareamento atenua o efeito, mas os testes podem ser um pouco anticonservadores, e isso é declarado.
- **Sensibilidade à partição (descritiva; emenda P11):** o HL e o p de Wilcoxon-Pratt de cada hipótese são calculados também em cada uma das 3 repetições, separadamente. A tese informa se o rótulo de decisão mudaria em alguma delas.

### 5.1 Poder

Os dp das diferenças por família vêm do ensaio geral com a grade final (8 famílias de desenvolvimento). O MDE é calculado com Holm m = 6, poder 0,8 e N = 50 (aproximação normal, fator 3,48). O "pior caso" usa o limite superior de 95% do dp com 7 graus de liberdade (× 1,80).

| Hipótese | dp no ensaio | MDE | MDE no pior caso |
|---|---|---|---|
| H1 | 0,094 | 4,6 p.p. | 8,3 p.p. |
| H2 | 0,031 | 1,5 p.p. | 2,8 p.p. |
| H3 | 0,058 | 2,9 p.p. | 5,1 p.p. |
| H5 | 0,086 | 4,2 p.p. | 7,6 p.p. |
| H6 (diferença de AUC) | 0,051 | 0,025 de AUC | 0,045 de AUC |

- **H2e:** com o dp do ensaio (0,031), o TOST tem poder de 0,8 para declarar equivalência se a diferença verdadeira for menor que ≈ 0,006 a 0,009, conforme a posição da H2e no Holm. Com o dp do pior caso, o poder é baixo: ≈ 0,1 a 0,6 mesmo com diferença verdadeira nula.
- **H4:** ver a seção 4.

### 5.2 Tabela de interpretação (desfecho → frase na tese → efeito na contribuição)

| Hipótese | Desfecho | Frase na tese | Efeito na contribuição |
|---|---|---|---|
| H1 | superior | "Sem acesso aos valores removidos, a adaptação observed-only do MechDetect foi inferior à configuração primária" | Delimita a aplicabilidade do estado da arte; não é vitória sobre o MechDetect publicado |
| H1 | evidência insuficiente | "Mesmo a adaptação observed-only do MechDetect não foi superada" | Contribuição frágil, declarada no resumo |
| H1 | inferior | "A adaptação observed-only do MechDetect foi superior à configuração primária" | Idem, com mais força |
| H1, H2, H3, H5, H6 | superior ou inferior "(efeito menor que δ)" | "A diferença foi significativa, com estimativa menor que 2 p.p. (ou 0,02 de AUC) (IC [a, b])". "Sem relevância prática" só se o IC inteiro ficar dentro de ±δ; senão, "relevância prática indeterminada" | Não sustenta afirmação de vantagem prática |
| H2 / H2e | equivalentes (H2 não significativa) | "Decompor em L1 e L2 não muda o desempenho (±2 p.p.)" | Responde a RQ1 com um nulo informativo; a configuração direta fica como recomendação |
| H2 / H2e | H2 significativa **e** equivalentes | "Há diferença, mas menor que 2 p.p.: na prática, equivalentes" | Igual à linha anterior |
| H2 | superior ou inferior, sem equivalência | "O hierárquico foi superior (ou inferior) em d [IC]" | Responde a RQ1 diretamente |
| H2 / H2e | nenhum dos dois | "Os dados não permitem afirmar diferença nem equivalência" | RQ1 sem resposta conclusiva |
| H3 | superior | "A configuração completa supera o modelo logístico mínimo em d [IC]", com a decomposição aprendiz × descritores ao lado | Sustenta a configuração; a parte dos descritores vem da decomposição e da ablação (RQ2) |
| H3 | evidência insuficiente | "Não há evidência de que a configuração completa supere o modelo mínimo", com o limite superior do IC95% do HL como o maior ganho da configuração completa compatível com os dados | A contribuição passa a ser o protocolo e a formulação. Por parcimônia, o modelo mínimo é preferível enquanto não houver evidência de ganho, o que não demonstra equivalência. A RQ2 é discutida à luz disso |
| H3 | inferior | "O modelo mínimo foi superior" | Idem, com mais força |
| H5 | superior | "Nas instâncias dos geradores externos, todas determinísticas, a configuração completa treinada no simétrico foi superior ao modelo mínimo", sempre com a perda fora do gerador, a decomposição por classe e o efeito do gerador ao lado | Responde em parte à crítica de circularidade; a resposta com um gerador estocástico independente vem da secundária do ampute |
| H5 | evidência insuficiente | "Nas instâncias dos geradores externos, todas determinísticas, não há evidência de diferença entre a configuração completa e o modelo mínimo" | A perda fora do gerador, com a decomposição efeito do gerador × especialização, vira a limitação central, declarada no resumo |
| H5 | inferior | "Nas instâncias dos geradores externos, todas determinísticas, o modelo mínimo treinado no simétrico foi superior" | Idem; recomenda-se o modelo mínimo para mecanismos determinísticos fora do gerador de treino |
| H5 (leitura) | qualquer desfecho com diferença | Só se diz que "a configuração completa é mais (ou menos) sensível à assinatura do gerador" se o IC do efeito do gerador excluir 0 **e** a diferença vier do recall de MAR ou de MNAR. Se vier do recall de MCAR, a frase é: "a diferença fora do gerador repete a diferença no reconhecimento do MCAR já observada no gerador principal" | Evita atribuir ao gerador o que é do MCAR (emenda P8) |
| H6 | superior | "No gerador principal (simétrico), sob MAR por uma única auxiliar e MNAR por auto-mascaramento, a configuração completa ordena MAR e MNAR melhor que o modelo mínimo (diferença de AUC [IC])", sempre com a H6 fora do gerador e a concordância dentro do trio ao lado. Se o IC da concordância incluir 0, acrescenta-se que a vantagem vem da comparabilidade dos escores entre contextos | Resposta principal da RQ2 sobre MAR × MNAR. Não implica identificabilidade em geral nem vale para outros geradores sem a H6 fora do gerador |
| H6 | evidência insuficiente | "Não há evidência de que os descritores completos separem MAR de MNAR melhor que o modelo mínimo" | A separação MAR × MNAR é reportada só pelo nível (AUC com IC) |
| H6 | inferior | "O modelo mínimo separou MAR de MNAR melhor" | Os descritores não ajudam onde mais importa; declarado no resumo |
| H3, H5, H6 combinadas | H3 sem superioridade, H5 inferior ou insuficiente, H6 superior | "No gerador principal, o conjunto completo ordena MAR e MNAR melhor que o modelo mínimo. Não há evidência de que classifique melhor, e, nos geradores externos, o modelo mínimo foi superior (ou: não há evidência de diferença)" | A contribuição é o protocolo e a delimitação; a recomendação de uso segue a H5, porque nenhum dado real vem do gerador simétrico (emenda P9) |
| Recall de MNAR por regime (descritivo) | com β = ∞, abaixo de 1/3 | "Quando a ausência é uma censura determinística de Y, o método a classifica como MAR em x% das instâncias" | Limitação declarada no resumo (emenda P6) |
| L1 × Little (secundária) | inferior | "No ponto de operação pré-registrado, o teste de Little detectou o afastamento do MCAR melhor que o L1 aprendido", com a AUC sem limiar ao lado | A contribuição do classificador fica nas três classes e no L2 (emenda P13) |
| H4 | primária superior | "O classificador supervisionado, treinado no mesmo gerador, foi superior ao &lt;modelo, alias, data, nível de raciocínio&gt; sem treino, com tabelas cegas de até 200 linhas", sempre com a sensibilidade de 200 linhas ao lado | Esperado em boa parte por construção, como a H1 (emenda P23) |
| H4 | LLM superior | "O &lt;modelo&gt; foi superior à configuração primária", só se as sensibilidades de 200 linhas e de memorização tiverem o mesmo sinal; senão, "resultado sensível à informação disponível ou à memorização" | Resposta da RQ4 |
| H4 | evidência insuficiente | "Não há evidência de diferença entre o &lt;modelo&gt; e a configuração primária", com o MDE do N efetivo | RQ4 sem resposta conclusiva |
| H4 | não rodou | "A RQ4 fica sem resposta confirmatória" | Declarado |

**"Praticamente relevante"** só quando |HL| ≥ δ. O estimando de nível nunca é chamado de "bom" ou "ruim" sem a comparação correspondente.

## 6. Análises secundárias (cada uma com o seu próprio Holm) e exploratórias

**Secundária, L1** (m = 3 fixo; uma comparação que não puder rodar entra no Holm com p = 1):
- **Classificador L1:** o L1 do hierárquico (`hier_hgb_fullprob`), que decide não-MCAR quando P(MCAR) < 0,5.
- **Comparadores:** teste de Little (α = 0,05), teste 1 do MechDetect-O (α/2) e PKLM (α = 0,05; 100 projeções e 200 repetições; `set.seed(20261101)` antes de cada instância).
- **Subconjunto do PKLM:** 8 contextos inteiros do gerador principal por família, 1.200 instâncias, escolhidos com `child_rng(20261101, "pklm_subset")` sobre os contextos ordenados por ID.
- **Métricas:** sensibilidade, especificidade e acurácia balanceada (AB) por família. O teste usa a AB, com Wilcoxon-Pratt e Holm.
- A comparação com o PKLM usa só o seu subconjunto.
- A especificidade dos testes é reportada **como medida** (emenda P14):
  - no piloto, o PKLM rejeitou 12,5% sob MCAR;
  - no E19 (29/10/2026), na configuração congelada (100 projeções, 200 repetições), ele rejeitou 20 de 200 tabelas gaussianas com MCAR puro: 10,0%, com IC95% (Clopper–Pearson) de 6,2–15,0% (`data_rb/decisoes/e19_pklm_tamanho_200_20261029.txt`);
  - as 48 primeiras tabelas repetem o refeito de 28/10 (6/48), e os 4,2% registrados em 21/10 não se reproduziram. Por isso, o excesso de rejeições não é atribuído aos dados reais.
  - A comparação L1 × PKLM vale para essa configuração, de tamanho empírico ≈ 10%: uma rejeição do PKLM não é lida como evidência ao nível de 5%.
- O teste 1 do MechDetect-O usa a AUC sobre `predict()`, por fidelidade ao código dos autores (passo 36).

**Secundária, RQ2** (m = 4): para k = 1…4, configuração primária com todos os descritores × sem o grupo G_k. Escore por família, Wilcoxon-Pratt e Holm. O SHAP é só descritivo.

**Secundária, estatísticas do MechDetect como descritores** (m = 1): configuração primária × o mesmo aprendiz só com o grupo G4 (`g4only_hgb`).

**Secundária, painel oráculo por gerador** (m = 3, um teste por gerador, cada um com a sua CV): o nosso classificador com descritores-oráculo × o MechDetect como publicado (clean, com o Y completo), no simétrico, na mdatagen e no tab_err. É a comparação com o estado da arte sob a mesma informação privilegiada (o Y completo). Os dois lados não são idênticos: o nosso oráculo é supervisionado dentro do gerador, e o MechDetect é uma regra fixa. O tab_err é o gerador do artigo do MechDetect.

**Secundária, geradores externos × MechDetect-O** (m = 1): nas instâncias da H5, a configuração primária treinada no principal × o MechDetect-O.

**Secundária, mdatagen com a própria CV** (m = 2): configuração primária × logística simples e configuração primária × MechDetect-O, treinando e testando na mdatagen.

**Secundária, RQ3** (m = 4; D2):
- **Contrastes:** ponderação × nenhuma e poda × nenhuma, para ruído simétrico e MAR ↔ MNAR, com taxa de 0,20.
- **Escore:** macro-recall no teste limpo, por família.
- As curvas com as taxas {0; 0,1; 0,2; 0,4} são descritivas.

**Secundária, RQ4-A2** (m = 1): pipeline + descritores do LLM × pipeline, no painel. O A2 é avaliado por CV de famílias dentro do painel, nos dois braços (emenda P23).

**Secundária, erros que mudam a decisão do usuário** (m = 2, Holm próprio; emenda P12):
- por família, no gerador principal, compara a configuração primária com a logística simples em duas taxas:
  - (i) a taxa de **falso MCAR**: instâncias MAR ou MNAR classificadas como MCAR;
  - (ii) a taxa **MNAR → MAR**: instâncias MNAR classificadas como MAR;
- usa Wilcoxon-Pratt e HL, como na seção 5;
- as mesmas taxas são descritas, sem teste, para o L1, o teste de Little, o PKLM e o teste 1 do MechDetect-O;
- os dois erros são reportados separados, sem pesos de custo. São os erros que mudam o tratamento: com o falso MCAR, quem usa o método apagaria linhas; com MNAR → MAR, imputaria achando que o viés foi resolvido;
- o experimento a jusante (passo 47 do plano) foi dispensado.

**Secundária, gerador externo estocástico: o ampute** (m = 1; emenda P24):
- **Procedimento:** o de amputação de Schouten, Lugtig e Vink (2018), *Generating missing values for simulation purposes: a multivariate amputation procedure* (J. Stat. Comput. Simul., DOI 10.1080/00949655.2018.1491577), na implementação `pyampute` 0.0.3.
- **Configuração:**
  - só na coluna-alvo, nos contextos com n = min(1000, nº de linhas);
  - direção lower → `sigmoid-left`, upper → `sigmoid-right`, both → `sigmoid-tail`;
  - `prop` = taxa;
  - MAR com peso só no driver, MNAR com peso só em Y, e o MCAR com o mecanismo `MCAR`;
  - semente derivada do contexto.
- **Teste:** a configuração primária e a logística simples, treinadas só no gerador principal (os modelos de cada fold da H5), são aplicadas sem ajuste às instâncias do ampute das famílias de teste. Wilcoxon-Pratt sobre o macro-recall por família.
- **Descritivo:** a **mudança** de desempenho da configuração primária entre o simétrico e o ampute, nas mesmas famílias (HL com IC). Ela responde se o desempenho muda muito com um gerador estocástico e independente.
- **Exceção declarada à regra dos trios:** o ampute sorteia linha a linha, então as contagens de ausências variam entre os mecanismos de um contexto. Este gerador só é usado no teste, a diferença de contagem é aleatória e não sistemática, e as taxas realizadas são reportadas.

**Sensibilidade, Y nominal** (descritiva; emenda P15):
- H1–H6 são recalculadas sem as bases cujo Y é um código guardado como número. A lista é fechada:
  - a família 4135 inteira (Y = `ROLE_*`);
  - as bases da 42493 com Y = `Flight`;
  - a base da 40701 com Y = `phone_number`;
- reporta-se a diferença pareada nas famílias que sobram, com IC;
- a análise principal não muda.

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
- LLM-A3 e o modelo secundário (`gemini-3.8-flash`);
- sensibilidade sem os contextos cujo Y completo tem massa ≥ 10% em 0 ou no seu mínimo: diferença pareada nas famílias que sobram nos dois conjuntos (IC por bootstrap), com a lista das famílias que saem por inteiro;
- controle sintético: CV agrupada por tabela-base no `synth-v2`;
- avaliação do L2: diagnóstica (macro-recall MAR × MNAR nas instâncias verdadeiramente não-MCAR) e operacional (nas verdadeiramente não-MCAR que o L1 roteou ao L2), com a fração de MCAR roteadas à parte; falha conta como erro;
- regime "dirty" do MechDetect, reportado como degenerado;
- **L1 sem limiar** (emenda P13): AUC por família de P(não-MCAR) do L1 e de 1 − p do teste de Little, do PKLM (no seu subconjunto) e do teste 1 do MechDetect-O, nas mesmas instâncias, com HL da diferença e IC, sem p. Compara a discriminação sem depender do ponto de operação nem do tamanho empírico do PKLM;
- **(opcional) MAR com vários drivers e MNAR misto, só no teste** (emenda P17): MAR pelo posto de uma combinação de 2 ou 3 auxiliares e MNAR pelo posto de Y + Z_d, com os mesmos β, direção e contagem. Aplica-se o modelo de cada fold sem retreino. Se não for feita, a omissão é declarada.

**Auditoria de atalhos nas famílias confirmatórias:** se o classificador só-nuisance passar de 0,40 de macro-recall, isso é reportado como alerta, ao lado das hipóteses, sem mudar a análise confirmatória.

**Estudo de caso (D3; emenda P19, decidida em 29/10):**
- **O que entra:** só casos cujo mecanismo é documentado pela fonte, isto é, a NHANES (cádmio, mercúrio, cotinina). A ausência ali é a censura no limite de detecção, marcada pela fonte: MNAR determinístico inferior.
- **Como:** em 10 subamostras de n = 2.000, com sementes fixas, dentro da faixa de treino.
- **Sem** acurácia nem concordância como métrica de acerto.
- **Leitura declarada:** no ensaio, esse regime foi classificado como MAR (seção 0.3).
- **O que sai:** as hipóteses atribuídas pelo autor. O Mroz aparece só como exemplo conceitual, sem probabilidades do modelo.

## 7. Situação da implementação

| Componente | Situação |
|---|---|
| Injeção, descritores 34b, avaliador, baselines (MechDetect-O, oráculo, dirty, Little, PKLM, logística simples), relatório | Implementados e testados |
| `stats.py` (família principal com Holm m = 6, TOST, Hodges–Lehmann com bootstrap, rank-biserial, ROPE descritiva, decisão; H4 em família própria; estimando de nível com IC) | Implementado e testado (versão 2); congelado com a tag `prereg-v1` |
| H5 (`cross-generator`, contra a logística simples treinada no principal) e H6 (`analyses.mar_mnar_auc`, AUC pareada contra a logística simples, falha como erro); decomposição descritiva da H3 | Implementados e testados; ensaio nas famílias de desenvolvimento |
| L1 secundário, recall por mecanismo, L2, ablação da RQ2, painel oráculo por gerador, mdatagen, estatísticas do MechDetect como descritores, geradores externos × MechDetect-O, LOGO, transferência, CD, sensibilidade da massa na fronteira | Implementados e testados (`analyses.py`; `pipeline.run_secondary`, `run_ablation`, `run_logo`, `run_transfer`) |
| RQ3 (passo 44) e estudo de caso (passo 45) | Implementados e testados (`noise.py`, `case_study.py`) |
| Prevalência realista e NaN em Z (passo 40) | **A implementar** |
| LLM (passos 42/43) | **A implementar.** O adendo LLM, com tag, vem antes do `evaluate` da produção (P21) |
| Emendas de 29/10: análises novas (P6, P7, P8, P11, P12, P13, P15), ampute (P24), `stats` com p = 1 por hipótese e saídas atômicas, etapa `stats-llm` separada, trava do papel `confirm` (P3), estudo de caso só com a NHANES (P19) | **A implementar antes da tag**, com testes e uma re-execução de conferência do ensaio, que deve repetir os números existentes |

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
- **O resumo e as conclusões trazem os seis desfechos da família principal**, na ordem H1–H6, com o rótulo da tabela de decisão. O nível aparece ao lado do nível da logística simples e do recall por classe (emenda P9).

**A tese pode afirmar:**
- os níveis e as comparações acima, no protocolo descrito;
- o nível de separação de MAR e MNAR no gerador principal (AUC com IC) e, se a H6 passar, que a configuração completa ordena MAR e MNAR melhor que o modelo mínimo no gerador principal, com a H6 fora do gerador e a concordância dentro do trio ao lado (emenda P7);
- a perda fora do gerador, com a decomposição efeito do gerador × especialização (emenda P8), e a mudança simétrico → ampute (emenda P24);
- que o MechDetect sem acesso a Y não decide MNAR, por construção.

**A tese não pode afirmar:**
- que o método identifica o mecanismo em dados com ausência natural;
- uma acurácia "na prática";
- que "supera o MechDetect" sem qualificar o regime de informação;
- que generaliza para MNAR por variável latente, MAR com vários drivers, várias colunas incompletas ou Z com ausências;
- qualquer "teto";
- que separa MAR de MNAR quando a ausência depende de várias auxiliares, de Y e de uma auxiliar ao mesmo tempo (MNAR misto) ou de uma função dos valores que não seja o posto de uma única coluna (emenda P17);
- que detecta MNAR por censura, isto é, por corte determinístico, sem reportar o recall com β = ∞ (emenda P6);
- que generaliza para outros geradores ou que é robusto à troca de gerador. Nos geradores externos, a frase é a da tabela 5.2 (emenda P8);
- que a classificação melhora a análise final, porque o experimento a jusante foi dispensado (emenda P12);
- que os resultados representam tabelas reais em geral. As famílias são tabelas de benchmark do OpenML, completas, algumas de simulação (emenda P16).

## 9. Desvios e proteções

- **Desvios:** qualquer mudança de especificação depois da tag `prereg-v1` entra em `docs/revisao-banca/desvios.md` (data, o quê, por quê, impacto esperado) e gera um **novo** run, sem apagar o anterior. Resultados de runs com desvio são reportados como tal.
- **Manchete:** o run pré-registrado é sempre o resultado principal. Um run com desvio só o acompanha.
- **Run pré-registrado (emenda P3):** é o primeiro run de produção iniciado no commit da tag `prereg-v1`, com a árvore limpa. Outro run com a mesma configuração é reportado e nunca o substitui. As etapas com papel `confirm` recusam rodar fora da tag ou com a árvore suja.
- **Invalidação:** um run só pode ser invalidado por erro de implementação contra o texto deste documento. Mesmo assim, os números dele são reportados, junto com o erro e a correção, como aconteceu com o piloto r1.
- **Ordem:** a ordem das etapas da produção está em `docs/revisao-banca/30-infra.md`. A etapa `cross-generator` vem antes do `stats`.

## 10. Registro externo e anterioridade

A tag no git sozinha não basta, porque uma tag pode ser apagada e recriada. Por isso, no dia da tag:
1. **Tag anotada `prereg-v1`** no commit que contém este documento e o código da análise, com push para o repositório público `JonathanTsen/ITA-Mestrado`.
2. **Registro de terceiros:** um release do GitHub dessa tag arquivado no Zenodo, que gera um DOI com data, ou um registro no OSF com este documento.
3. **E-mail ao orientador:** com o SHA do commit e o SHA-256 de `preregistro.md` e de `stats.py`, para que a data de recebimento sirva de anterioridade independente.
4. **Segundo depósito (emenda P4):** ao fim do `stats`, os SHA-256 de `stats_confirmatory.csv`, `stats_level.csv`, `oracle.parquet` e dos `run*.json` do run pré-registrado são publicados num segundo depósito no Zenodo, ligado ao primeiro. O mesmo vale para o `stats-llm` da H4.

A aprovação do orientador (29/10/2026) e a aprovação das emendas são arquivadas com data em `plano-revisao-banca/execucao/08-reuniao-orientador.md`, no repositório da tese.

Os hashes e o link do registro entram em `plano-revisao-banca/execucao/38-preregistro.md`, no repositório da tese.
