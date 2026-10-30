# Adendo LLM ao pré-registro v2 (`prereg-v1-llm`): RASCUNHO de 30/10/2026

> **Situação: rascunho, sem valor de registro.**
>
> **O que falta antes da tag:**
> 1. As decisões do autor marcadas com **[AUTOR]**.
> 2. O teste do Argon (≈ 30 chamadas), que **só roda com a autorização do autor**.
> 3. Reconferir, na data do adendo, qual é o modelo confirmatório pela regra do §4.
> 4. Que o pré-registro principal tenha a tag `prereg-v1`.
>
> O adendo é lacrado **antes do `evaluate` da produção** (emenda P21).

Este adendo congela o que o §4 do pré-registro (H4, RQ4) deixou para depois do piloto de desenvolvimento. O piloto rodou no `gemini-3.8-flash`, só nas 8 famílias de desenvolvimento, em 30/10/2026; o relatório está em `plano-revisao-banca/execucao/38d-piloto-llm.md` e `results_rb/_audit/20261030-piloto-llm/`. Nenhum resultado das famílias confirmatórias informa este texto.

## 1. Modelos e datas

- **Provedor:** só o Google (Gemini), pela API pública, com o SDK `google-genai` 1.74.0.
- **Confirmatório:** o modelo mais capaz, de uso geral e não-preview do provedor na data do adendo.
  - Em 29/10/2026 era o `gemini-4-argon`.
  - **[A reconferir com `models.list` na data do adendo]**
- **Secundário descritivo:** o `gemini-3.8-flash`.
- **Registro por chamada:** a API não oferece snapshot datado; o `modelVersion` devolvido é o próprio alias. Cada resposta grava o alias, a data e hora (UTC), o `modelVersion`, o `responseId`, os tokens e o plano da chave (pago ou gratuito).
- **Janela:** o A1 confirmatório roda numa janela contínua de até 72 h. Se a versão mudar, nada é refeito, e a mudança é reportada.

## 2. Perguntas e renderizadores (congelados pelo commit da tag)

| Braço | Instruções (SHA-256 do texto) | Mensagem do usuário |
|---|---|---|
| A1, classificador direto cego | `d6c047c5…cbc08` (`render.A1_SYSTEM`) | tabela CSV cega: colunas `c0…ck`; inteiros de 0 a 999 por coluna (afim positiva sobre os valores observados das linhas mostradas, seguida de arredondamento); ≤ 200 linhas sorteadas; linhas e colunas permutadas; `NA` na ausência (`render.render_a1`, versão 1) |
| A2, extrator de descritores | `6968989f…b032d1` (`render.A2_SYSTEM`) | os 28 descritores 34b em JSON com chaves ordenadas, 4 algarismos significativos e `null` no indefinido (`render.render_a2`) |
| A3, semântico (exploratório) | **[AUTOR: implementar ou cortar já (P22)]** | — |

- **Semente da renderização:** `child_seed(20261101, "llm_render", contexto)`, a mesma para os três mecanismos do contexto.
- **Teste byte a byte** (`tests/rb/test_llm_render.py`): mudar o rótulo, o `sample_id`, os nomes das colunas ou os valores ausentes de Y, ou aplicar uma transformação afim positiva aos dados, não muda o prompt.
- **Perguntas idênticas:** quando duas instâncias têm os mesmos dados observados (a máscara MAR igual à MNAR acontece em ≈ 1% dos contextos simétricos do ensaio, sempre em Y com massa na fronteira), as perguntas são idênticas e recebem **a mesma resposta**. A contagem é reportada.

## 3. Esquema e geração (emenda P20)

- **Esquema:** só as três probabilidades (MCAR, MAR, MNAR), números em [0, 1] (SHA-256 do JSON Schema `6763a003…22b9f5`).
  - É enviado como `response_schema` (subconjunto OpenAPI), nos dois modos.
  - Em 30/10/2026, o modo de lote ignorou o `response_json_schema`; o diagnóstico está em `results_rb/_audit/20261030-schema-lote/`.
- **Parâmetros:**
  - nenhum parâmetro de amostragem é enviado: temperatura, top_p, top_k e semente são ignorados pelos modelos 3.6+ e 4;
  - sem ferramentas;
  - raciocínio **`medium`** **[AUTOR: confirmar; o piloto mediu ≈ 7,9 mil tokens de raciocínio por chamada no A1]**;
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
  - ou atributo ≥ 3 de 10;
  - ou nome correto.

  **[AUTOR: aprovar os limiares]**

  No piloto, o atributo marcou a pc3 com 4 de 10. Sugestão: exigir também mais acertos do que o chute do valor mais frequente.
- **Nome correto:**
  - a resposta contém o nome do OpenML (normalizado) **ou** um apelido de uma lista fechada, montada **antes** de rodar as sondas confirmatórias;
  - os casos não cobertos pela lista são adjudicados pelo autor, cego aos resultados da H4, e registrados.

  **[AUTOR: aprovar a regra.]** No piloto, a conferência automática deixou passar "Breast Cancer Wisconsin" para o `BNG(breast-w)` e "ipums-la-99" para o `kdd_ipums_la_97-small`.
- **Exclusão:**
  - uma família reconhecida na renderização cega sai da análise principal da H4, sem reposição;
  - o reconhecimento da tabela crua é covariável: a H4 restrita às famílias não reconhecidas na crua é reportada à parte (`stats_llm_covariate.csv`);
  - "LLM superior" só é afirmado se o sinal se mantiver nelas.
- **Ordem:** as sondas rodam, e a lista de famílias excluídas é publicada com SHA-256, antes do `evaluate` da produção.

## 6. Análise

- **H4 (família própria, m = 1):** Wilcoxon-Pratt do macro-recall por família, LLM-A1 confirmatório × configuração primária restrita às mesmas instâncias (média das 3 repetições da CV principal).
  - Etapas: `llm`, depois `stats-llm --probes`.
  - Decisão e frases: as do §5 e do §8 do pré-registro.
- **Sensibilidade de 200 linhas** (`llm-sensitivity`), sempre ao lado da H4:
  - a primária é retreinada com as subamostras de ≤ 200 linhas, nas mesmas partições;
  - ela é aplicada às mesmas linhas que o A1 mostra;
  - comparação descritiva.
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

- **Projeção** com os tokens do piloto, em lote; o Argon com o mesmo raciocínio do Flash **[a medir no teste do Argon]**:
  - Argon ≈ US$ 68 (A1 26, A2 3, A3 27, sondas 12);
  - Flash ≈ US$ 23;
  - total ≈ US$ 91, ou ≈ US$ 137 com a margem de 1,5×, contra o teto de US$ 300.
- **Preço do Argon:** dobra em 01/12/2026, o que leva a ≈ US$ 159, ou US$ 239 com a margem.
- **Trava no código:** antes de cada execução, gasto + 1,5 × projeção ≤ US$ 300; senão, cortam-se, nesta ordem, o A3, o modelo secundário e o A2, antes de ver resultados.
- **Retomada:** o A1 confirmatório roda primeiro; se o orçamento acabar, a H4 usa as famílias completas.

## 9. Ordem na produção

1. tag `prereg-v1-llm` neste texto;
2. `llm-probes` nas famílias do painel e publicação da lista de exclusão (SHA-256);
3. `evaluate` da produção e as demais etapas do pipeline principal;
4. `llm` A1 (confirmatório, depois secundário), A2 e A3;
5. `llm-sensitivity`, `llm-a2-ablation` e `stats-llm --probes`.

Nenhuma chamada ao Argon roda sem a autorização explícita do autor, pedida antes de cada execução (regra de 29/10/2026; trava no código).
