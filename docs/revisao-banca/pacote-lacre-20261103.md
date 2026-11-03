# Pacote para o lacre (C) e a autorização de execução (D) — 03/11/2026

**Para o autor revisar.** Este pacote reúne o que foi preparado depois das decisões de 03/11. Ele **não** é o lacre e **não** autoriza nenhum gasto. Nenhuma tag foi criada, nenhum release foi publicado e nenhuma chamada paga foi feita: o caderno de respostas tinha 6.480 arquivos antes e depois deste trabalho.

## 1. Decisões registradas

| Data | Quem | Decisão |
|---|---|---|
| 03/11/2026 | autor (relato da conversa com o Vitor) | O autor toma as decisões científicas, de escopo e de execução; o Vitor revisa os resultados depois. Não é uma aprovação individual de cada escolha pelo Vitor. |
| 03/11/2026 | autor | **A — H7 incluída:** 30 bases inventadas novas, um único contraste S7 × S0, sem H7b. |
| 03/11/2026 | autor | **B — regra das sondas aprovada:** limiares numéricos mantidos; nomes claros excluem automaticamente; nomes ambíguos e respostas fora da lista vão para a adjudicação cega do autor; dúvida persistente fica inconclusiva e não exclui. |
| 30/10/2026 | autor | `medium`; A3 sujeito ao orçamento; teste Argon de 30 perguntas (feito, US$ 0,21). |
| antes | autor | Só NHANES no estudo de caso; dispensa do passo 47; teto global de US$ 300 incluindo o já gasto. |

## 2. O que mudou no código (ramo `revisao-banca/fase-c`)

| Commit | O quê |
|---|---|
| `6db8541` | Regra das sondas: estados `reconhecida` / `revisar` / `nao_reconhecida`; lista ativa com a coluna `tipo`; `adjudicacao_pendente.csv` sem resultados; comando `llm-probes-finalize` (sob a trava do adendo); `stats-llm` confirmatório só aceita a lista final; `inconclusiva` não exclui. |
| `1b96a7e` | H7: 30 domínios novos, `contexto_h7.py` (gerar, perguntar com as travas, analisar por base), bases e manifesto em `data_rb/contexto_h7/`, `configs/rb/contexto-h7.yaml`. |
| `be1b9d5` | `projecao_orcamento.py` e `data_rb/decisoes/orcamento_llm_20261103.json`. |
| `07f6fa8` | Pré-registro, adendo, `sondas-revisao` e `30-infra` com as decisões. |

## 3. Para revisar: os nomes marcados como ambíguos

Uma resposta que contém **só** um destes nomes não exclui a base automaticamente; ela vai para a sua adjudicação. Você pode mudar a marcação de qualquer linha antes do lacre (`data_rb/llm_dataset_aliases.csv`, coluna `tipo`).

| Base (OpenML) | Nome ambíguo | Por quê | Nomes claros da mesma base |
|---|---|---|---|
| 6 | `letter` | palavra comum | `Letter Recognition` |
| 37 | `diabetes` | palavra comum; existe outra base 'diabetes' (regressão, scikit-learn) | `pima`, `pima indians diabetes` |
| 54 | `vehicle` | palavra comum | `Statlog (Vehicle Silhouettes)` |
| 151 | `electricity` | palavra comum; várias bases de consumo de energia | `elec2` |
| 251 | `breast cancer wisconsin` | família Wisconsin: casa também com a WDBC (1510), base distinta | `BNG(breast-w)` |
| 251 | `breast-w` | base de origem da BNG(breast-w): versão relacionada, não a representante | `BNG(breast-w)` |
| 251 | `wisconsin breast cancer` | família Wisconsin: casa também com a WDBC (1510), base distinta | `BNG(breast-w)` |
| 310 | `mammography` | palavra comum; várias bases de mamografia | — |
| 823 | `houses` | palavra comum; também casa com Brazilian_houses e outras bases de imóveis | — |
| 40701 | `churn` | palavra comum; várias bases de churn | — |
| 42493 | `airlines` | palavra comum; várias bases de companhias aéreas | — |
| 44089 | `credit` | palavra comum; casa com credit-g e outras bases de crédito | `give me some credit` |
| 44124 | `ipums` | nome do censo IPUMS: identifica a família, não o extrato kdd_ipums_la_97-small | `kdd_ipums_la_97-small` |
| 44127 | `phoneme` | palavra comum | — |
| 44133 | `pol` | sigla curta | — |
| 44145 | `sulfur` | palavra comum | `sulfur recovery unit` |
| 44160 | `rl` | sigla curta | — |
| 44162 | `compass` | grafia alternativa e palavra comum | `compas` |
| 44980 | `kin8` | família kin8 (kin8nm, kin8fh, ...): não identifica a versão | `kin8nm` |
| 44993 | `health_insurance` | várias bases de seguro de saúde | — |
| 44994 | `cars` | palavra comum; várias bases de carros | — |
| 45012 | `fifa` | várias bases da FIFA | — |

"—": a base não tem nome claro; ela só sai automaticamente por um critério numérico (cabeçalho, linhas ou atributo) ou pela sua adjudicação.

**Dois detalhes de implementação para você conferir:**
- Também vai para a planilha **toda resposta de nome que não casa com a lista e não é uma recusa** ("unknown", "I do not know"). Foi assim que o piloto deixou passar "ipums-la-99" para a `kdd_ipums_la_97-small`. Essas linhas vêm com `nao_reconhecida` pré-preenchido, para você confirmar ou mudar. No máximo, cerca de 69 bases × 2 tabelas × 2 modelos.
- A planilha preenchida fica no diretório do run (fora do git). Se ela fosse commitada, o HEAD sairia da tag e a trava recusaria as etapas seguintes. O SHA-256 dela vai para o resumo e é publicado com a lista de exclusão.

## 4. H7: o que fica congelado

- **30 bases novas** (IDs 990201–990230), 1.000 linhas e 8 colunas contínuas cada, temas fora dos 16 de desenvolvimento: 990201 Orvalho Dairy Cooperative Milk Recording · 990202 Serra Alta Wind Farm Turbine Records · 990203 Trans-Planalto Freight Trip Ledger · 990204 Casa Laguna Hotel Group Stays · 990205 Colmeia Viva Apiary Sensor Network · 990206 Picos Brancos Ski Resort Daily Operations · 990207 Ponte Nova Used-Vehicle Auction House · 990208 Montes Claros Permanent Forest Plots · 990209 Rio Manso Basin Gauging Network · 990210 Nuvem Sul Data Center Rack Telemetry · 990211 Aeroporto Lagoa Seca Gate Turnarounds · 990212 Torra Fina Specialty Roastery Batch Log · 990213 Costa Brava Norte Offshore Buoy Array · 990214 Movimento Fitness Club Membership Panel · 990215 Seguradora Atlas Motor Claims File · 990216 Banco Horizonte Mortgage Applications · 990217 Palco Aberto Live Music Venues · 990218 Hemocentro Vale Verde Donor Sessions · 990219 Cervejaria Três Pedras Fermentation Tanks · 990220 Mineração Serra Dourada Haul Fleet · 990221 Terminal Porto Azul Vessel Calls · 990222 Lumina Games Player Telemetry · 990223 Linha Verde Commuter Rail Trips · 990224 Recife das Tartarugas Reef Survey · 990225 Estação Anilhadora Ponta do Vento · 990226 Geleira Branca Mass-Balance Stakes · 990227 Elevadores Ascende Maintenance Records · 990228 Museu do Litoral Visitor Counts · 990229 Tinturaria Arco-Íris Dye-House Batches · 990230 Abrigo Quatro Patas Adoption Records.
- **SHA-256:**
  - `data_rb/contexto_h7/sources.csv`: `b42203db8cf2bf98f6aaac0f18bdad993a84a67e5db6cf92059b9f513b96c2b5`;
  - `data_rb/contexto_h7/contexts.json`: `3d2bbb77fb0bde2a695df1c0b52b9185ad0eeba6f0a23d297a75c92ad3b14413`;
  - gerador (`contexto_h7_dominios.py`): `3b7c67acc2708867dadd048d38ac861d6499bab2935b4da264c5d9a97898bb4d`;
  - `configs/rb/contexto-h7.yaml` (papel `confirm`, `root_seed` 20261103): `00bfe1f11036112797507aa79bf6e99ff42c72ee3d7eee906263c3f63d0b6284`.
- **Conferido:** gerar duas vezes dá os mesmos arquivos. As 16 bases exploratórias regeneradas continuam idênticas. Uma checagem com papel `dev` e outra semente, fora do repositório, pôs as 30 bases no painel: 540 instâncias, 18 por base, 6 por mecanismo, ou seja, 1.080 perguntas por modelo. As instâncias confirmatórias **não** foram geradas: o `inject` com papel `confirm` só roda no commit da tag `prereg-v1`.
- **Ajuste feito antes de qualquer chamada:** na primeira versão, várias colunas eram binárias ou contagens com poucos valores. Os 16 domínios de desenvolvimento têm pelo menos 30 valores distintos por coluna. Para seguir o mesmo gerador, essas colunas foram trocadas por medidas contínuas equivalentes.
- **Análise:** Wilcoxon-Pratt bilateral sobre d = macro-recall(S7) − macro-recall(S0) por base, HL com IC por bootstrap de bases, α = 0,05, fora do Holm. McNemar e recalls por classe são só descritivos. A leitura sai pronta: "S7 acima de S0", "S7 abaixo de S0" ou "evidência insuficiente de diferença (não é equivalência)".

## 5. Orçamento (sem chamadas; `orcamento_llm_20261103.json`)

| Cenário | Futuro | Gasto + 1,5 × futuro | Cabe em US$ 300? |
|---|---|---|---|
| Conservador, execução até 30/11 | US$ 154,36 | **US$ 274,90** | sim |
| Argon medido no A1, até 30/11 (informativo) | US$ 94,58 | US$ 185,23 | sim |
| Conservador, em dezembro (Argon dobra) | US$ 268,44 | US$ 446,03 | **não** |
| Argon medido, em dezembro | US$ 148,88 | US$ 266,69 | sim |

- Gasto acumulado: US$ 43,36.
- Na H7, o Flash raciocina cerca de 7.500 tokens por pergunta, então ela custa ≈ US$ 17 no Flash (eram US$ 10 em 30/10). No Argon, ≈ US$ 46 no cenário conservador, ou ≈ US$ 8 se ele raciocinar como no teste A1.
- **Consequência prática:** para ficar no cenário conservador, a produção Argon precisa terminar até 30/11/2026. Em dezembro, a ordem de cortes já registrada (A3, depois o modelo secundário, depois o A2) precisa ser aplicada antes de qualquer resultado.
- **Os preços foram conferidos em 30/10.** Não os reconferi hoje; reconferir no dia da execução, junto com o modelo confirmatório (`models.list`).

## 6. Ordem e comandos do lacre (C) — **não executados**

1. Você revisa este pacote e a lista da seção 3.
2. Tag anotada `prereg-v1` no commit final, com push para `JonathanTsen/ITA-Mestrado`:
   `git tag -a prereg-v1 -m "pré-registro v2 + emendas; decisões do autor de 03/11" && git push origin prereg-v1`
3. Registro de terceiros (pré-registro, §10): release do GitHub dessa tag arquivado no Zenodo (gera DOI). **Publica.** Só com a sua autorização explícita; não usar release como rascunho ou backup.
4. O passo 3 do §10 do pré-registro prevê um **e-mail ao orientador** com os SHA. Pela regra vigente, eu não envio e-mails. Se você quiser manter esse passo, o envio é seu.
5. Reconferir o modelo confirmatório e preencher o §1 do adendo. Depois, tag `prereg-v1-llm` no commit do adendo.

## 7. O que a autorização de execução (D) precisaria nomear

Em lote, em sequência (a trava do orçamento não reserva o custo de jobs pendentes), com backup depois de cada execução paga:

| Lote | Modelo | Estimativa conservadora até 30/11 |
|---|---|---|
| Sondas (P21) | Argon + Flash | US$ 12 (Argon); Flash na base |
| A1 confirmatório | Argon | US$ 26 (US$ 4,20 com o raciocínio medido) |
| A1 secundário | Flash | na base Flash (US$ 23 no total) |
| A2 | Argon (+ Flash) | US$ 3 |
| A3 (exploratório; primeiro corte) | Argon | US$ 27 |
| H7 S0 + S7 | Argon | US$ 46,08 (US$ 8,10 com o raciocínio medido) |
| H7 S0 + S7 | Flash | US$ 17,28 |

Toda chamada ao Argon exige a sua autorização, com texto, antes de cada execução (`--authorization`).

## 8. Verificação desta preparação

- Suíte completa `uv run --frozen pytest tests/rb` (03/11/2026): 295 passaram e 1 falhou. A falha foi `test_sources.py::test_local_benchmark_counts_and_roles`, por falta dos CSV-fonte em `/tmp/rb_mechdetect_run`, que tinha sido apagado; ela não tem relação com estas mudanças. Depois de `checagem_previa.py`, que reextraiu os 101 CSV conferidos por SHA-256, `test_sources.py` passou (5/5). Resultado: 296 de 296.
- Ruff sem erros nos arquivos tocados.
- Caderno de respostas `results_rb/_llm_cache`: 6.480 arquivos antes e depois, ou seja, nenhuma chamada.
