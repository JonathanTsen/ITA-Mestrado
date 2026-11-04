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
