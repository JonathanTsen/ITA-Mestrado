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

## D5 — perda do computador da produção e run completo refeito direto (05–06/11/2026)

- **O que aconteceu:** em 05/11/2026, o computador em que a produção rodava parou de funcionar. Junto com ele se perderam:
  - `results_rb/` inteiro: os runs da etapa 1 (`20261104-inject-v1-738bdf7b5937`), da etapa 2 (`…-500f42cb6ef8`) e da H7 (`20261104-contexto-h7-…`);
  - o caderno de respostas do Gemini de 03–05/11;
  - as cópias de segurança de 04/11, que estavam só no mesmo disco.
- **O que sobrou:** o git, as tags e releases `prereg-v1`…`prereg-v1.4`, os depósitos do Zenodo, os registros no plano da tese (`RESULTADOS-2026-11-04.md` e anexo, gerados dos arquivos antes da perda) e as cópias de 30/10 e 02/11 (Drive e repositório privado), anteriores à produção.
- **Decisões do autor (05/11/2026, na conversa):**
  1. ir **direto ao run completo** (`inject-v1.yaml`, 50 famílias; `contexto-h7.yaml`, 30 bases), sem refazer as etapas 1 e 2 como runs separados. Elas já cumpriram o papel da P25 (o autor viu a primeira metade antes de autorizar o resto). As etapas usam o mesmo `name` e a mesma `root_seed` do completo, então as instâncias são as mesmas;
  2. **refazer as chamadas do Gemini** no escopo de 04/11 (Argon: sondas, H7, A1, A2, A3; Flash: sondas e H7). Depois, "total autorização para rodar tudo que for preciso" no Gemini;
  3. manter o código lacrado e **conferir a trava do orçamento à mão**: o gasto perdido (US$ 59,65 = 98,74 − 43,36 do anexo, mais ≈ 4,27 do A1/A2/A3 da etapa 2 citado na PR #71 da tese) é somado ao do caderno antes de cada lote (`~/work/rb-producao/trava_manual.py`).
- **Máquina nova:** MacBook Air, 8 núcleos; R 4.6.2 (antes 4.6.1) e PKLMtest 1.0.1; Python e pacotes pelo mesmo `uv.lock`. Os CSV-fonte foram reextraídos do MechDetect e conferidos pelo SHA-256 (101 de 101).
- **Reprodutibilidade conferida:** o `inject` do completo descarta as mesmas 11 famílias por regra (50 efetivas), e as 25 famílias da etapa 1 geram exatamente as **5.214 instâncias** do `baselines.parquet` perdido. Run: `20261105-inject-v1-52ed6f93a2a0` (10.332 instâncias); H7: `20261105-contexto-h7-c635b9dbd73a`.
- **O que muda nos números da IA:** o Gemini não responde igual a cada vez, então as respostas novas são uma nova amostra. Os números das etapas 1 e 2 registrados no plano da tese ficam como histórico descritivo. A análise oficial é a do run completo.
- **Ordem:** as chamadas do Gemini foram feitas antes das etapas longas de CPU, a pedido do autor. O `llm` grava as respostas no caderno e só monta a comparação com a configuração primária depois do `evaluate`. Por isso a adjudicação das sondas (06/11/2026) foi feita sem nenhum resultado da comparação, como manda a regra.
- **Sondas e adjudicação:** nas tabelas cegas, 0 de 30 famílias foram reconhecidas pelas duas IAs, e nenhuma foi excluída. O autor aprovou as sugestões em 06/11/2026.
  - Argon: 4 reconhecidas, 3 inconclusivas e 35 não reconhecidas, todas na crua. SHA-256 da planilha `b0be01471e78eba09256454138f7fe8917c97ce44d3a045a1870c3cab44dc291`; da lista `506373f2558c687d960b97e11ceacbef5f832b7dedba517a479bc7e84eb5f17a`.
  - Flash: 4, 2 e 37. Planilha `9c7158d29a005c240f14b49030dcc9e0071c500c1c3b5068802c05c112646e30`; lista `045290e208f7d11ca1436adfcd9c5863dabd2fe6ab12f5db898da64d5ff21b4e`.
- **Gasto do refazer:** US$ 59,98. O total estimado é ≈ US$ 163, de um teto de US$ 300. O valor oficial é o do faturamento do Google.
- **Cópia de segurança:** depois do último lote pago, o run, a H7 e o caderno foram enviados ao repositório privado `ITA-dados-privado` (commit `c95183f58`).
- **Nada mudou** em regras, hipóteses, comparadores, testes ou código. Não há nova tag; a produção continua em `prereg-v1.4`.
- **Fim da produção (06/11/2026):** a fase de CPU terminou às 21:26 UTC sem erros, em `prereg-v1.4` e com a árvore limpa. Ela rodou baselines, `evaluate` (×4), PKLM (1.200 de 1.200 ok), `stats`, secundárias, robustez, ablação, LOGO, transferência, `report`, `noise` e `fit-final`. Depois rodaram o estudo de caso (`20261106-case-study-f826114aad4f`) e o controle sintético (`evaluate` no `synth-v2`).
  - A pedido do autor, os processos ficaram congelados (`SIGSTOP`) das 14:53 às 17:25 UTC, no meio do PKLM, e continuaram do mesmo ponto. Nada foi recalculado.
  - O PKLM levou ≈ 240 s por instância com 8 processos, contra ≈ 55 s na máquina antiga.
- **PKLM com R 4.6.2:**
  - rodado nas mesmas 24 primeiras tabelas do R4 de 30/10, com a semente de produção, deu **p-valores idênticos nas 24** (`results_rb/_audit/e19-r462-tabelas-antigas-24.csv`). A troca 4.6.1 → 4.6.2 não muda o PKLM.
  - Uma ressalva encontrada nessa checagem: o script de decisão `d4_pklm_tamanho.py` gera as tabelas com `numpy.random.multivariate_normal` (método SVD), e nesta máquina elas saem diferentes. A causa provável é a biblioteca de álgebra linear. Com essas tabelas novas, os p-valores mudam, mas a rejeição é parecida: 2 de 24 (`e19-r462-24-seed20261101.csv`).
  - A produção não depende disso. O `inject` não usa essa função, e o `synth-v2` (que usa `method="cholesky"`) foi copiado, não regerado.
- **Suíte de testes nesta máquina:** 343 de 343, com `OMP_NUM_THREADS=1`, em `prereg-v1.4`.
- **Cópias de segurança:** a cada marco no `ITA-dados-privado`. O último, depois do estudo de caso e do controle sintético, é o commit `11dc90c89`.

## Nota de reprodutibilidade — desempate nos geradores externos (revisão de 11–12/11/2026)

Não é um desvio: nada mudou em regras, hipóteses, código ou resultados. É uma limitação da reprodução, achada na revisão do código depois da produção.

- **O que se viu:** os adaptadores do `mdatagen` e do `tab_err` (`src/missdetect/rb/inject.py`) passam às bibliotecas uma ordenação cujo desempate entre valores iguais não é estável. As linhas empatadas no limiar, ou na borda da janela, podem ser escolhidas de forma diferente conforme a rotina de ordenação da plataforma.
- **Medida feita na revisão:** regenerando as máscaras em Linux x86-64 com as versões do `uv.lock`, a partir dos arquivos gravados e das sementes, as do gerador simétrico saíram idênticas às da produção (feita em ARM), e as dos geradores externos saíram idênticas em 39% a 45% das instâncias MAR e MNAR conferidas. Todas as diferenças estão em linhas empatadas num valor de fronteira, e a contagem de ausências é sempre a mesma. Esta medida é da revisão e não foi repetida por um segundo conferente.
- **O que não muda:** os rótulos continuam válidos, porque a máscara depende só do driver (MAR) ou só de Y (MNAR), a menos dos empates.
- **Consequência:** a H5 e as avaliações nos geradores externos só se reproduzem bit a bit com as instâncias gravadas (identificadas pelo `mask_sha256`) ou na mesma arquitetura. As instâncias gravadas são a referência.
- **Para execuções futuras:** passar à biblioteca um posto com desempate estável pela ordem da permutação.

## Nota — SHAP calculado depois da produção (12/11/2026)

O pré-registro prevê o SHAP só como complemento descritivo da ablação da RQ2 ("O SHAP é só descritivo"), e o run de 05–06/11 não o calculou. Em 12/11/2026 ele foi calculado a partir do modelo final congelado (`final_model/model.joblib` do run `20261105-inject-v1-52ed6f93a2a0`), nas 6.090 instâncias do gerador principal em que o modelo foi treinado, sobre a saída bruta do gradient boosting (antes da calibração), com `shap` 0.51.0. O script está no repositório da dissertação (`scripts/figuras_rb/gerar_figuras.py`) e os valores em `figuras/rb/shap_importancia.csv`. Nenhuma hipótese, teste ou decisão depende desse cálculo, e a dissertação o apresenta como descritivo e posterior à análise confirmatória.

