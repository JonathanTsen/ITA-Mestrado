# Desvios do pré-registro (§9)

Toda mudança feita depois da tag `prereg-v1` fica registrada aqui, com data, motivo, o que se sabia no momento e o efeito. As tags nunca são apagadas nem movidas; uma correção ganha uma tag nova.

## D1 — erro de implementação no `inject` da família 44059 (03/11/2026)

- **O que aconteceu:** a etapa 1 (P25) começou às 23:37 UTC no commit da tag `prereg-v1` (`98c01d7`). O `inject` abortou com `ValueError: auxiliar chamada 'y' colide com a coluna observada`.
- **Causa:** o arquivo observado guardava a variável-alvo numa coluna chamada `y`. A família 44059 (`diamonds`, colunas `x`, `y`, `z`, que são as dimensões do diamante) tem uma auxiliar chamada `y`, e uma trava de segurança abortava em vez de dar outro nome. O ensaio usou só as famílias de desenvolvimento, que não têm essa coluna.
- **O que se sabia:** nenhuma predição, nenhuma resposta da IA, nenhuma métrica e nenhum resultado. O run parcial (`20261103-inject-v1-738bdf7b5937`, só com arquivos observados) foi movido, sem ser usado, para `results_rb/_superseded/`.
- **Correção (sem mudar dados, regras, hipóteses, comparadores ou testes):** o alvo passa a ser gravado com o nome reservado `__y__` (`runs.OBSERVED_TARGET`). A leitura aceita os dois formatos, e as auxiliares mantêm os nomes originais.
- **Verificação antes da nova tag:** `inject` com uma cópia de papel `dev` e outra semente nas 50 famílias efetivas, fora do repositório: 50 de 50 famílias com bases e 10.356 instâncias. 135 instâncias da 44059 têm a auxiliar `y`, e os descritores foram calculados nelas sem erro. Suíte completa com `OMP_NUM_THREADS=1`: 300 de 300 (duas verificações liam a coluna antiga `y` direto do arquivo e foram atualizadas).
- **Nova tag:** `prereg-v1.1` e `prereg-v1.1-llm`, no commit da correção. A trava (`runs.PREREG_TAG`, `runs.LLM_ADDENDUM_TAG`) passa a exigir essas tags. Os documentos (`preregistro.md`, `adendo-llm.md`) não mudam de conteúdo além desta referência.
- **Decisão:** do executor, sob a instrução do autor de 03/11 ("se acontecer alguma coisa, eu gostaria que você continuasse até onde parou"); é registrada para revisão do autor.

## D2 — sondas de memorização: duas falhas do `tabmemcheck` em famílias confirmatórias (04/11/2026)

- **O que aconteceu:** na etapa 1, já no commit `prereg-v1.1`, as sondas do Argon abortaram na passada que só registra as perguntas, antes de qualquer chamada à API. Não houve gasto nem resposta. O `budget.jsonl` nem chegou a ser criado.
- **Causas, achadas numa passada seca sem rede** (caderno temporário) em todas as 60 tabelas do painel completo, cruas e cegas:
  1. **Família 310 (mammography), tabelas crua e cega:** a base tem muitas linhas repetidas. O teste de completar linhas do `tabmemcheck` 0.1.6 não acha uma linha cujo final não apareça nos exemplos e levanta `Exception` ("Unable to construct a query...").
  2. **Família 44148 (superconduct, etapa 2), tabela crua:** o `tabmemcheck` adivinha o separador lendo só 5.000 caracteres (`utils.get_delimiter`). Com 82 colunas de números longos, a adivinhação falha (`Could not determine delimiter`).
- **O que se sabia:** nenhuma resposta da IA, predição ou resultado.
- **Correções, sem mudar os critérios, os limiares ou a regra de decisão das sondas:**
  1. Quando o pacote não consegue montar o teste de linhas, ele fica **indisponível** (`row_test = "indisponivel_linhas_repetidas"`, 0 acertos). Nome, cabeçalho e atributo continuam valendo para essa tabela. Só a família 310 é afetada, nas duas tabelas.
  2. O separador é fixado em vírgula (`probes._comma`), que é como as tabelas das sondas são sempre gravadas. Nas tabelas em que a adivinhação já funcionava, o resultado é o mesmo.
- **Verificação antes da nova tag:**
  - passada seca: 60 de 60 tabelas do painel completo e 30 de 30 da etapa 1;
  - fumaça sem rede dos comparadores (MechDetect-O, Little, oráculo, PKLM) em 2 instâncias de cada uma das 50 famílias;
  - montagem das perguntas A1, A2 e A3 da etapa 1 e A1 e A3 do painel completo, com backend falso;
  - testes novos (linhas repetidas, tabela larga) e suíte completa com `OMP_NUM_THREADS=1`: 302 de 302.
- **Nova tag:** `prereg-v1.2` e `prereg-v1.2-llm`. As anteriores ficam como estão.
- **Decisão:** do executor, sob a instrução do autor de 03/11 de continuar se algo acontecer. É registrada para revisão do autor.

## D3 — PKLM: a mesma colisão da coluna `y` da D1, em outro lugar (04/11/2026)

- **O que aconteceu:**
  - A etapa `baselines` da etapa 1, no commit `prereg-v1.2`, terminou o MechDetect e o teste de Little (`baselines.parquet`, 5.214 instâncias, gravado às 13:01 UTC).
  - Em seguida, o PKLM abortou: `pklm_r.run_pklm` ainda gravava o CSV de entrada com a coluna `y`, que colide com a auxiliar `y` da família 44059.
  - A correção da D1 tinha coberto o arquivo observado, mas não esse segundo ponto.
  - A fumaça da D2 não pegou o problema: a instância sorteada da 44059 não tinha a auxiliar `y`.
- **O que se sabia:** nenhuma resposta da IA, predição do classificador ou métrica. O `baselines.parquet` contém só os p-valores e decisões dos comparadores, que ninguém inspecionou.
- **Correção:**
  - `pklm_r.write_instance_csv` grava Y com o nome reservado `OBSERVED_TARGET`.
  - O `pklm.R` lê a matriz por posição, então o p-valor não depende do nome. Um teste novo mostra p idêntico com e sem a auxiliar chamada `y`.
  - Uma busca em `src/missdetect/rb` não achou outro lugar que junte Y e as auxiliares sob o nome `y`.
- **`baselines.parquet` mantido:**
  - Foi produzido no commit `2e30247`, cujo código do MechDetect e do Little é idêntico ao da correção.
  - Verificação: 20 instâncias, 6 delas da 44059 com auxiliar `y`, recalculadas com o código corrigido deram valores idênticos nas 25 colunas.
  - O log `run.baselines.json` não foi gravado, porque o comando abortou no PKLM. O PKLM roda à parte com `baselines --only-pklm`, que grava `run.baselines-pklm.json`.
  - SHA-256 do `baselines.parquet` registrado no HANDOFF.
- **Verificação antes da nova tag:**
  - entrada do PKLM gravada para as 1.200 instâncias do subconjunto (15 com auxiliar `y`);
  - R rodado numa amostra de 8 instâncias, 8 de 8 ok;
  - suíte completa de testes com `OMP_NUM_THREADS=1`: 303 de 303.
- **Nova tag:** `prereg-v1.3` e `prereg-v1.3-llm`. As anteriores ficam como estão.
- **Decisão:** do executor, sob a instrução do autor de continuar se algo acontecer. É registrada para revisão do autor.

## D4 — resíduo de ponto flutuante nas diferenças pareadas (05/11/2026; correção autorizada pelo autor)

- **O que se viu:** ao documentar os resultados parciais, uma base da H7 com acerto idêntico nos dois braços (8/18 − 8/18) apareceu com diferença de ±1e-17 em vez de 0.
  - O Wilcoxon-Pratt tratava esse resíduo como diferença real, e não como zero.
  - As contagens de melhor/pior/empate da H7 ficavam erradas: Argon etapa 2, 13/2/0 em vez de 12/1/2.
  - O mesmo acontecia nas diferenças entre métodos da família principal, que vêm de médias de macro-recall.
- **Correção:** `stats.clean_differences` arredonda as diferenças a 12 casas antes de qualquer teste. Ela é usada em:
  - `paired_differences`, `_describe`, `wilcoxon_p`, `wilcoxon_tost` e `rank_biserial`;
  - `contexto_h7.analisar_h7`.
  - Os escores são médias de acertos em dezenas ou centenas de instâncias, então nenhuma diferença real fica abaixo de 1e-12.
  - Testes novos: o resíduo conta como empate no teste, no TOST, no rank-biserial, nas diferenças pareadas e nas contagens da H7.
- **Efeito medido nos resultados parciais da etapa 1** (recalculados à parte com o código corrigido, sem regravar o run). **Nenhuma decisão muda e os HLs ficam idênticos.** Os p-valores mudam pouco:
  - H2: 0,872 → 0,850;
  - H2e: 0,000246 → 0,000245;
  - H3: 3,22e-5 → 3,21e-5;
  - H5: 0,1388 → 0,1387;
  - H1 e H6: iguais;
  - H4: 0,00072 → 0,00071;
  - H7, Argon etapa 1: 0,0081 → 0,0075.
- **O que se sabia:** os resultados parciais e descritivos da etapa 1 (P25) e a H7 das duas etapas. A correção não depende de nenhum resultado: só faz a conta tratar empate como empate. Os critérios, testes, limiares e frases não mudam.
- **Autorização:** o autor, em 05/11/2026, depois da explicação: "Pode corrigir".
- **Verificação:** suíte completa com `OMP_NUM_THREADS=1`: 305 de 305.
- **Nova tag:** `prereg-v1.4` e `prereg-v1.4-llm`. A etapa 2 continua nela, e o run completo (a análise oficial) usa este código. Os resultados já gravados da etapa 1 ficam como estão, descritivos, e o run completo recalcula tudo.
