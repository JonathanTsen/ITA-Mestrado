# Adendo LLM ao pré-registro v2 (`prereg-v1-llm`): versão final de 03/11/2026

> **Situação: versão final, lacrada pela tag `prereg-v1-llm` em 03/11/2026, no mesmo commit da `prereg-v1`, com a autorização do autor (lacre e registro no Zenodo). O histórico abaixo fica como registro.**
>
> **O que falta antes da tag:**
> **Quem decide (03/11/2026):** o autor informou que, segundo sua conversa com o orientador, ele próprio toma as decisões científicas, de escopo e de execução, e o Vitor revisa os resultados depois. As decisões abaixo são do autor; não são aprovações individuais do Vitor.
>
> **Decididas pelo autor em 03/11/2026:** (a) a H7 entra, com 30 bases inventadas novas e um único contraste S7 × S0, sem H7b (§10); (b) a regra das sondas: limiares mantidos, nomes claros automáticos, nomes ambíguos e respostas fora da lista para a adjudicação cega do autor, dúvida persistente como inconclusiva, sem exclusão automática (§5). Antes, em 30/10: `medium`, A3 sujeito ao orçamento e o teste Argon de 30 perguntas de desenvolvimento.
>
> 1. ~~A revisão, pelo autor, do pacote final~~: feita em 03/11; o autor aprovou a lista de nomes ambíguos como está e autorizou o lacre.
> 2. Teste Argon concluído em 30/10, autorizado pelo pedido de implementação: 30/30 válidas, US$ 0,212231 (§8). A produção ainda exige autorização própria.
> 3. ~~Reconferir o modelo confirmatório~~: feito em 03/11 (`gemini-4-argon`, §1).
> 4. ~~Que o pré-registro principal tenha a tag `prereg-v1`~~: as duas tags ficam no mesmo commit.
>
> O adendo é lacrado **antes do `evaluate` da produção** (emenda P21).

Este adendo congela o que o §4 do pré-registro (H4, RQ4) deixou para depois do piloto de desenvolvimento. O piloto rodou no `gemini-3.8-flash`, só nas 8 famílias de desenvolvimento, em 30/10/2026; o relatório está em `plano-revisao-banca/execucao/38d-piloto-llm.md` e `results_rb/_audit/20261030-piloto-llm/`. Nenhum resultado das famílias confirmatórias informa este texto.

## 1. Modelos e datas

- **Provedor:** só o Google (Gemini), pela API pública, com o SDK `google-genai` 1.74.0.
- **Confirmatório:** o modelo mais capaz, de uso geral e não-preview do provedor na data do adendo.
  - Em 29/10/2026 era o `gemini-4-argon`.
  - **Reconferido em 03/11/2026 com `models.list`:** continua sendo o `gemini-4-argon` (o mais capaz entre os modelos de uso geral, não-preview, que a chave lista).
- **Secundário descritivo:** o `gemini-3.8-flash`.
- **Registro por chamada:** a API não oferece snapshot datado; o `modelVersion` devolvido é o próprio alias. Cada resposta grava o alias, a data e hora (UTC), o `modelVersion`, o `responseId`, os tokens e o plano da chave (pago ou gratuito).
- **Janela:** o A1 confirmatório roda numa janela contínua de até 72 h. Se a versão mudar, nada é refeito, e a mudança é reportada.

## 2. Perguntas e renderizadores (congelados pelo commit da tag)

| Braço | Instruções (SHA-256 do texto) | Mensagem do usuário |
|---|---|---|
| A1, classificador direto cego | `d6c047c5…cbc08` (`render.A1_SYSTEM`) | tabela CSV cega: colunas `c0…ck`; inteiros de 0 a 999 por coluna (afim positiva sobre os valores observados das linhas mostradas, seguida de arredondamento); ≤ 200 linhas sorteadas; linhas e colunas permutadas; `NA` na ausência (`render.render_a1`, versão 1) |
| A2, extrator de descritores | `6968989f…b032d1` (`render.A2_SYSTEM`) | os 28 descritores 34b em JSON com chaves ordenadas, 4 algarismos significativos e `null` no indefinido (`render.render_a2`) |
| A3, semântico (exploratório) | `render.A3_SYSTEM` (hash no commit da tag) | a **mesma tabela do A1** (mesmas linhas, permutação e escala) com os **nomes reais** das colunas, precedida de uma descrição neutra da base (`render.render_a3`). Mantido pelo plano solicitado em 30/10; primeiro corte se necessário pela P22. |

- **Semente da renderização:** `child_seed(20261101, "llm_render", contexto)`, a mesma para os três mecanismos do contexto.
- **Teste byte a byte** (`tests/rb/test_llm_render.py`): mudar o rótulo, o `sample_id`, os nomes das colunas ou os valores ausentes de Y, ou aplicar uma transformação afim positiva aos dados, não muda o prompt.
- **Descrição neutra do A3** (`render.neutral_description`), a partir do texto cru congelado em `data_rb/openml_descriptions_20261030.csv` (101 tabelas, SHA-256 por descrição):
  - a descrição do OpenML é dividida em seções;
  - as de identificação saem inteiras (autoria, fonte, citação, título, uso anterior, artigos, dono, datas, licença, contagens e valores ausentes);
  - nas demais, saem as frases com termos de ausência, proveniência ou referências e o nome distintivo da base;
  - até 600 caracteres, igual para os três mecanismos;
  - as regras foram fixadas olhando só as 8 tabelas de desenvolvimento.
- **Perguntas idênticas:** quando duas instâncias têm os mesmos dados observados (a máscara MAR igual à MNAR acontece em ≈ 1% dos contextos simétricos do ensaio, sempre em Y com massa na fronteira), as perguntas são idênticas e recebem **a mesma resposta**. A contagem é reportada.

## 3. Esquema e geração (emenda P20)

- **Esquema:** só as três probabilidades (MCAR, MAR, MNAR), números em [0, 1] (SHA-256 do JSON Schema `6763a003…22b9f5`).
  - É enviado como `response_schema` (subconjunto OpenAPI), nos dois modos.
  - Em 30/10/2026, o modo de lote ignorou o `response_json_schema`; o diagnóstico está em `results_rb/_audit/20261030-schema-lote/`.
- **Parâmetros:**
  - nenhum parâmetro de amostragem é enviado: temperatura, top_p, top_k e semente são ignorados pelos modelos 3.6+ e 4;
  - sem ferramentas;
  - raciocínio **`medium`**, mantido pelo plano solicitado pelo autor em 30/10; o piloto Flash mediu ≈ 7,9 mil tokens de raciocínio por chamada no A1;
  - teto de saída de 32.768 tokens (o máximo observado no piloto foi 26.304).
- **Resposta válida:** um objeto JSON com exatamente as chaves MCAR, MAR e MNAR, números finitos em [0, 1] cuja soma difere de 1 em no máximo 0,05, depois normalizados.
  - Até 2 retentativas, só para saída inválida. Erros transitórios (429, 5xx, rede) são repetidos e não contam como tentativa.
  - Sem resposta válida em 3 tentativas: falha = erro (probabilidades uniformes, classe −1).
- **Classe prevista:** a de maior probabilidade. Empates são desfeitos por sorteio com `child_seed(20261101, "llm_tie", instância, repetição)`.
- **Caderno de respostas:** vale a primeira resposta válida de cada pergunta.
  - Cada resposta fica crua em `results_rb/_llm_cache/<config>/`, que nunca é apagado nem sobrescrito.
  - O SHA-256 do caderno inteiro entra no registro externo.
- **Modo de lote:** permitido (−50%), porque o modelo e a pergunta são os mesmos. Fica registrado em cada resposta.

## 4. Painel, repetição e ordem

- **Painel (§4 do pré-registro):**
  - 30 das 50 famílias efetivas, sorteadas com `child_rng(20261101, "llm_panel")`;
  - em cada família, os contextos do gerador principal com r ∈ {0,1; 0,3; 0,5} × k ∈ {2, maior k da base} e n = min(1000, linhas);
  - 18 instâncias por família (9 se a base só tem k = 2), num total de ≈ 540.
- **Repetição:** 10% das instâncias do painel (`child_rng(20261101, "llm_repeat")`) são perguntadas uma segunda vez. A concordância (κ de Cohen e diferença de macro-recall) é descritiva.
- **Ordem:** as famílias rodam na ordem de `child_rng(20261101, "llm_order")`, e a repetição vem por último.

## 5. Memorização (emenda P21)

- **Bateria por tabela e modelo:** o `tabmemcheck` 0.1.6, adaptado ao Gemini pelo caderno, com raciocínio `low`:
  - cabeçalho (cortes nas linhas 2, 4, 6 e 8);
  - linhas (10 perguntas);
  - atributo (10 perguntas, no atributo com mais valores distintos);
  - uma sonda própria de nome: as 5 primeiras linhas e "qual é o nome desta base?".
- **Duas tabelas por família do painel:**
  - **crua:** a tabela-fonte o mais perto da versão pública (sem índice, inteiros sem ".0", alvo com o nome do OpenML). É covariável;
  - **cega:** a renderização do A1 da primeira instância, por `sample_id`, do contexto com o maior k e a menor taxa. É o critério de exclusão.
- **Reconhecida** se, em qualquer teste:
  - cabeçalho ≥ 50 caracteres literais;
  - ou linhas ≥ 2 de 10;
  - ou atributo ≥ 3 de 10 **e** acima do chute do valor mais frequente (binomial, p < 0,05);
  - ou nome correto.

  **Limiares aprovados pelo autor em 03/11/2026** (sem mudança).

  No piloto, a pc3 teve 4 de 10 valores exatos de um atributo cuja moda ocupa 3,1% da tabela (p ≈ 0): é memorização, não acaso.
- **Nome (regra aprovada pelo autor em 03/11/2026; `probes.RULE_VERSION` = 2):**
  - a lista fechada `data_rb/llm_dataset_aliases.csv`, montada **antes** das sondas confirmatórias, tem o nome do OpenML e os apelidos de cada representante, com o motivo e o `tipo`: `claro` ou `ambiguo` (nome genérico ou curto, só a família, versões relacionadas; o nome canônico também pode ser ambíguo, como `houses`);
  - **reconhecida** automaticamente: um critério numérico acima ou um nome `claro` na resposta (normalizados);
  - **revisar:** só um nome `ambiguo`, ou uma resposta de nome que não casa com a lista e não é uma recusa ("unknown", "I do not know"), pois pode identificar a família ou uma versão vizinha;
  - **adjudicação:** o autor decide cada caso em `revisar` (`reconhecida`, `nao_reconhecida` ou `inconclusiva`), com motivo, data e responsável, **depois das sondas e antes do `evaluate`**, quando ainda não existe nenhum resultado da comparação LLM × classificador. A planilha (`adjudicacao_pendente.csv`) só traz a base, a resposta e o nome casado; as respostas fora da lista vêm pré-preenchidas como `nao_reconhecida`, para confirmar;
  - **inconclusiva** não exclui: a família fica na H4 e é listada no log;
  - `llm-probes-finalize` grava a lista final (`exclusions_final.csv`) e os SHA-256 da planilha e da lista. A planilha fica no run, fora do git, para não tirar o HEAD da tag, e o SHA-256 é publicado com a lista de exclusão.

  **Regra e lista aprovadas pelo autor em 03/11/2026; ativadas no código** (commit `6db8541`): 93 nomes cobrindo os 69 representantes, 22 marcados como ambíguos (lista para revisão no pacote do lacre). A proposta e sua evidência estão em [sondas-revisao-20261030.md](sondas-revisao-20261030.md). No piloto, sem a lista, a conferência deixou passar "Breast Cancer Wisconsin" para o `BNG(breast-w)` e "ipums-la-99" para o `kdd_ipums_la_97-small`. Com a lista de rascunho, a tabela crua foi reconhecida em 7 de 8 famílias, e a renderização cega em 0 de 8.
- **Exclusão:**
  - uma família reconhecida na renderização cega (automaticamente ou na adjudicação) sai da análise principal da H4, sem reposição; uma inconclusiva fica;
  - o reconhecimento da tabela crua é covariável: a H4 restrita às famílias não reconhecidas na crua é reportada à parte (`stats_llm_covariate.csv`);
  - "LLM superior" só é afirmado se o sinal se mantiver nelas.
- **Ordem:** as sondas rodam, o autor adjudica os casos em `revisar`, o `llm-probes-finalize` grava a lista final, e a lista de famílias excluídas é publicada com SHA-256, tudo antes do `evaluate` da produção. O `stats-llm` confirmatório só aceita a lista final.

## 6. Análise

- **H4 (família própria, m = 1):** Wilcoxon-Pratt do macro-recall por família, LLM-A1 confirmatório × configuração primária restrita às mesmas instâncias (média das 3 repetições da CV principal).
  - Etapas: `llm`, depois `stats-llm --probes`.
  - Decisão e frases: as do §5 e do §8 do pré-registro.
- **Sensibilidade de 200 linhas** (`llm-sensitivity`), sempre ao lado da H4:
  - a primária é retreinada com as subamostras de ≤ 200 linhas, nas mesmas partições;
  - ela é aplicada às mesmas linhas que o A1 mostra;
  - comparação descritiva.
  - No piloto, a primária caiu de 0,618 para 0,519 com 200 linhas, e a diferença para o LLM-A1 passou de −0,125 para −0,014 (p = 0,57). Por isso a frase da H4 na tese sempre traz a sensibilidade ao lado.
- **Secundária RQ4-A2** (m = 1; P23; `llm-a2-ablation`): a primária com os 28 descritores × a mesma com os 28 mais as 3 probabilidades do A2, numa CV de famílias dentro do painel.
- **Descritivos:**
  - o LLM-A2 sozinho (classe de maior escore);
  - o modelo secundário;
  - a concordância da repetição;
  - a taxa de "rótulo constante no trio";
  - tokens, custo e falhas.

## 7. Poder

- **MDE simulado:** com o dp das diferenças por família do piloto (0,156; 8 famílias), Wilcoxon-Pratt, α = 0,05 e poder de 80% (`results_rb/_audit/20261030-piloto-llm/mde_h4.py`):

  | N | MDE |
  |---|---|
  | 30 | ≈ 8,5 p.p. |
  | 25 | ≈ 9,5 p.p. |
  | 20 | ≈ 11 p.p. |

- **Cenário pessimista:** com o dp no limite superior do IC95% (0,318), o MDE é ≈ 17,5 p.p. com N = 30.
- **Família excluída pelas sondas:** o N efetivo e o MDE para esse N acompanham a H4.

## 8. Orçamento (emenda P22)

### Atualização de 03/11 com a H7 incluída (prevalece sobre as projeções abaixo)

- Conta reproduzível, sem chamadas: `scripts/revisao_banca/llm/projecao_orcamento.py`, saída em `data_rb/decisoes/orcamento_llm_20261103.json`. Gasto acumulado global: US$ 43,36 (sem chamadas novas desde 30/10).
- A H7 usa a entrada e o raciocínio medidos no S0 e no S7 dos dois exploratórios: no Flash, ≈ 7.500 tokens de saída por chamada, o que leva a H7 no Flash a ≈ US$ 17,28 (não US$ 10). No Argon, conservador (com o raciocínio do Flash) ≈ US$ 46,08; com o raciocínio do teste A1 do Argon, ≈ US$ 8,10.
- **Conservador até 30/11/2026:** futuro US$ 154,36; gasto + 1,5 × futuro = **US$ 274,90**, dentro do teto global de US$ 300.
- Com o Argon medido no A1 (informativo): US$ 185,23.
- **Em dezembro o Argon dobra:** conservador US$ 446,03, **não cabe**; com o Argon medido, US$ 266,69. Executar a produção Argon até 30/11 ou aplicar a ordem de cortes (P22) antes de qualquer resultado; não aumentar o teto.
- Preços reconferidos na página oficial em 03/11/2026: iguais aos do código (Argon 2/10 até 30/11 e 4/20 depois; Flash 0,75/3,75 até 31/12; lote com −50%).
- **Execução em duas etapas (P25):** a etapa 1 custa cerca de metade do previsto (conservador ≈ US$ 77 até 30/11). O run completo reaproveita as respostas da etapa 1 pelo caderno; a etapa 2 paga a outra metade.

### Atualização de 30/10 após o teste autorizado (prevalece sobre a projeção histórica abaixo)

- Teste A1 Argon em lote, `medium`, 30 perguntas distintas de desenvolvimento: 30 respostas válidas, 0 retentativas; custo calculado pelos tokens de **US$ 0,212231**, não uma conferência de fatura. Entrada média 4.345,37 tokens, raciocínio médio 498,3 (máximo 2.398), saída final média 47,5. Modelo informado: `gemini-4-argon`. Resultado em `data_rb/decisoes/argon_limit30_20261030.json`; execução no código `f9c230c`, árvore limpa ao início.
- Preços e disponibilidade reconferidos em 30/10: `models.list` e [preços oficiais](https://ai.google.dev/gemini-api/docs/pricing). Argon em lote: US$ 1/5 por milhão de tokens de entrada/saída até 30/11; Flash: US$ 0,375/1,875 até 31/12. A saída inclui raciocínio.
- **Gasto acumulado global:** US$ 43,35953825, incluindo os dois experimentos de contexto e este teste.
- **Projeção conservadora com H7 e A3:** preservar os US$ 91 anteriores e acrescentar US$ 60 da H7 = US$ 151 futuros; com margem de 1,5× e gasto passado, **US$ 269,86**, dentro do teto global de US$ 300.
- Se extrapolado o A1 medido para 594 chamadas, esse braço do Argon custa US$ 4,20, e o total global com margem cai para US$ 237,16. Esse cenário é informativo: o painel pequeno não mede A2, A3, sondas ou H7. Mantém-se a projeção conservadora para planejar.
- **Em dezembro, a projeção conservadora com H7 sobe para US$ 446,86, incluindo gasto anterior e margem: não cabe no teto.** Aplicar P22 antes de resultados; não aumentar o teto automaticamente.
- A trava foi corrigida para somar todas as configurações sob `results_rb/_llm_cache`, e não apenas a configuração corrente; testes de regressão em modo síncrono e lote verificam o bloqueio antes de geração. Executar os lotes pagos sequencialmente: não há reserva de custo de jobs em andamento. Conservar todos os cadernos sob a mesma raiz.
- Revalidar a projeção com o corpus e os prompts finais antes da produção. Cálculos e premissas em `data_rb/decisoes/orcamento_llm_com_h7_20261030.json`.

### Projeção histórica anterior ao teste (não inclui H7 nem gasto passado)

- **Teste do Argon** (`llm --limit 30`, em lote, nas famílias de desenvolvimento; **só com a autorização do autor**): mede o raciocínio real do Argon antes desta projeção ser fechada.
- **Projeção** com os tokens do piloto, em lote; o Argon com o mesmo raciocínio do Flash **[a medir no teste do Argon]**:
  - Argon ≈ US$ 68 (A1 26, A2 3, A3 27, sondas 12);
  - Flash ≈ US$ 23;
  - total ≈ US$ 91, ou ≈ US$ 137 com a margem de 1,5×, contra o teto de US$ 300.
- **Preço do Argon:** dobra em 01/12/2026, o que leva a ≈ US$ 159, ou US$ 239 com a margem.
- **Trava no código:** antes de cada execução, gasto + 1,5 × projeção ≤ US$ 300; senão, cortam-se, nesta ordem, o A3, o modelo secundário e o A2, antes de ver resultados.
- **Retomada:** o A1 confirmatório roda primeiro; se o orçamento acabar, a H4 usa as famílias completas.

## 9. Ordem na produção

**Duas etapas (P25):** a ordem abaixo roda primeiro com as configurações da etapa 1 e depois com as da etapa 2; a análise confirmatória final usa as configurações completas, e as perguntas já respondidas saem do caderno. A adjudicação das sondas de cada etapa acontece antes do `evaluate` daquela etapa.


1. tag `prereg-v1-llm` neste texto;
2. `llm-probes` nas famílias do painel; adjudicação do autor dos casos em `revisar`; `llm-probes-finalize`; publicação da lista de exclusão e da planilha (SHA-256);
3. `evaluate` da produção e as demais etapas do pipeline principal;
4. `llm` A1 (confirmatório, depois secundário), A2 e A3;
5. `llm-sensitivity`, `llm-a2-ablation` e `stats-llm --probes <exclusions_final.csv>`;
6. H7 (§10): o `inject` de `configs/rb/contexto-h7.yaml` roda na worktree da tag `prereg-v1`; `contexto_h7.py perguntar` (S0 e S7, confirmatório e depois Flash) e `analisar`, no commit da `prereg-v1-llm`.

Nenhuma chamada ao Argon roda sem a autorização explícita do autor, pedida antes de cada execução (regra de 29/10/2026; trava no código).

## 10. H7, efeito do contexto em bases inventadas (proposta de 30/10/2026; **incluída por decisão do autor em 03/11/2026**)

**De onde vem:** o experimento exploratório do autor (`scripts/revisao_banca/llm/contexto_sintetico.py`; relatório no plano da tese, `execucao/38d-piloto-llm.md`), feito em 16 bases inventadas de desenvolvimento.
- A IA com contexto detalhado (faixas típicas aproximadas em ±25%) e valores originais (S7) acertou 0,622, contra 0,510 da tabela cega (S0): +11,1 p.p., em 12 de 16 bases, McNemar p < 0,0001.
- Replicou em dois grupos de domínios.

**Desenho:**
- **Hipótese:** em bases inventadas com documentação, o macro-recall por base do LLM com contexto e valores originais (S7) difere do da tabela cega (S0). Família própria, α = 0,05, fora do Holm de H1–H6.
  - Um único contraste confirmatório, S7 × S0; não há H7b. Uma eventual comparação com o classificador permanece descritiva, com a diferença de informação declarada.
- **Dados:** 30 domínios inventados **novos**, sem reutilizar os 16 de desenvolvimento, com 1.000 linhas e 8 colunas cada. Preparados em 03/11/2026, sem nenhuma chamada ao LLM, com o gerador, o molde de contexto e a regra das faixas aproximadas do exploratório (±25%, semente fixa): `scripts/revisao_banca/llm/contexto_h7_dominios.py` (IDs 990201–990230, temas fora dos 16 de desenvolvimento, colunas contínuas com ≥ 30 valores distintos, como lá) e `data_rb/contexto_h7/` (SHA-256 de `sources.csv` `b42203db…96c2b5`, de `contexts.json` `3d2bbb77…b14413`, do gerador `3b7c67ac…98bb4d`; todos os arquivos no `manifest.json`). Usa o apagamento do benchmark no desenho do painel (`configs/rb/contexto-h7.yaml`, papel `confirm`, `root_seed` 20261103): 18 instâncias por base, 6 de cada tipo, 540 por braço. As instâncias só são geradas no commit da tag `prereg-v1` (trava do `inject`); uma checagem de elegibilidade com papel `dev` e outra semente, fora do repositório, confirmou as 30 bases no painel com 18 instâncias cada.
- **Informação:** S0 e S7 veem as mesmas até 200 linhas e as mesmas colunas; S0 anonimiza e escala, S7 recebe os valores originais e documentação com faixas aproximadas. A documentação é idêntica entre os três mecanismos e não menciona ausências. O contraste testa esse conjunto de informações, não o efeito isolado da descrição textual.
- **Congelado neste adendo:**
  - os prompts S0 e S7, pelo código no commit da tag (`contexto_sintetico.prompts`, `RENDERER` = `contexto-sintetico-1`) e pelos SHA-256 das bases e dos contextos acima; as mensagens renderizadas dependem das instâncias, geradas só depois da tag;
  - o molde do contexto e a regra das faixas;
  - o modelo pela regra do §1 (mais o Flash secundário);
  - o raciocínio, o esquema e o lote;
  - o código de análise: Wilcoxon-Pratt por base, HL com IC por bootstrap de bases, McNemar descritivo;
  - as frases permitidas por desfecho.
- **Poder** (dp por base de 0,113 nas 16 bases de desenvolvimento):

  | Bases | Efeito de 7,5 p.p. | Efeito de 10 p.p. |
  |---|---|---|
  | 24 | 86% | 98% |
  | 30 | 92% | ≈ 100% |
- **Custo:** 30 bases × 18 × 2 braços = 1.080 chamadas, em lote: ≈ US$ 46 no Argon (conservador; ≈ US$ 8 com o raciocínio medido no A1) e ≈ US$ 17 no Flash (§8, atualização de 03/11). O Argon só com a autorização do autor.
- **Interpretação:** usar o teste bilateral por base; reportar HL, intervalo, sinal e recalls por classe. Ausência de diferença significativa significa evidência insuficiente, não equivalência. Não generalizar para bases reais nem atribuir causalmente o ganho apenas ao texto. Os p-valores por instância do exploratório não substituem a análise confirmatória por base.
- **Execução:** `scripts/revisao_banca/llm/contexto_h7.py` (`perguntar` com a trava da tag `prereg-v1-llm`, a autorização do autor e a trava do orçamento; `analisar` com `analisar_h7`). A inclusão da H7 não autoriza as chamadas H7 nem a produção Argon: elas entram na autorização de execução. O custo H7 entra na projeção global e no teto P22; não se atribui um novo teto de US$ 300 a cada experimento.
- **Limites a declarar:**
  - as bases inventadas são mais simples que as reais;
  - a imperfeição das faixas é simulada;
  - nada se afirma sobre bases reais conhecidas, onde contexto e memória se confundem.

