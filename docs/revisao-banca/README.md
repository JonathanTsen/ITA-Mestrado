# Revisão pós-banca de qualificação: experimentos (Fase C)

**Situação em 11/11/2026.** O pré-registro foi lacrado em 03/11 (tags `prereg-v1` … `prereg-v1.4`), a produção terminou em 06/11 na tag `prereg-v1.4` (`57746ca`) e está tudo na `main` (PRs #8 e #9). O que mudou depois do lacre está em [`desvios.md`](desvios.md) (D1–D5). Os hashes das saídas do run oficial estão em [`deposito-p4-resultados.md`](deposito-p4-resultados.md) (tag `resultados-v1`, DOI 10.5281/zenodo.24024942; o DOI do conceito é 10.5281/zenodo.23933349). O estado vivo do plano fica em `plano-revisao-banca/execucao/00-ESTADO.md`, no repositório da dissertação. As seções abaixo foram escritas até 30/10 e descrevem o caminho até a produção.

O **plano** da revisão vive no repositório da dissertação, `JonathanTsen/ModeloTesePPGPO`, na pasta `plano-revisao-banca/` da branch `main`. O estado consolidado está em `plano-revisao-banca/execucao/HANDOFF-2026-10-29.md` (tudo de 20/10 a 29/10, inclusive os experimentos, a auditoria de 28/10 e os próximos passos). Este diretório guarda só a documentação do **código** dos experimentos, para não manter duas cópias do plano.

- **Branch de trabalho:** `revisao-banca/fase-c`, criada a partir de `41e00a6` e integrada à `main` pelo PR #8 (06/11/2026).
- **Pacote:** `src/missdetect/rb/`, isolado do código legado; testes em `tests/rb/`.
- **Testes:** **203 passando** (`OMP_NUM_THREADS=1 uv run --frozen pytest -q`).

## Documentos deste diretório

| Arquivo | Conteúdo |
|---|---|
| [`30-infra.md`](30-infra.md) | Módulos, layout de um run, ordem de execução da produção e mapa dos testes |
| [`preregistro.md`](preregistro.md) | Pré-registro das análises confirmatórias, **versão 2 (27/10/2026)**, revista criticamente antes de qualquer dado confirmatório. Lacrado em 03/11/2026 na tag `prereg-v1`; as emendas posteriores estão nas tags `prereg-v1.1` … `prereg-v1.4` |
| [`adendo-llm.md`](adendo-llm.md) | Adendo do pré-registro para a IA: braços A1–A3, H4, H7, sondas e orçamento (tags `prereg-v1-llm` … `prereg-v1.4-llm`) |
| [`desvios.md`](desvios.md) | Desvios do pré-registro, D1 a D5, com data, motivo e efeito |
| [`deposito-p4-resultados.md`](deposito-p4-resultados.md) | Segundo depósito: SHA-256 das saídas do run oficial, da H7 e do estudo de caso |
| [`pacote-lacre-20261103.md`](pacote-lacre-20261103.md), [`sondas-revisao-20261030.md`](sondas-revisao-20261030.md) | Pacote do lacre e revisão dos critérios das sondas |

## Tags (congelamentos)

| Tag | Commit | O que congela |
|---|---|---|
| `synth-v1` | `698cfeb` | Corpus sintético do piloto: 1.800 bases, 5.400 instâncias, β {0,5; 1; 2; 4; ∞} |
| `synth-v2` | `c578982` | Corpus sintético da produção: β {1, 2, 4, 8, ∞}, código corrigido |
| `freeze-v1` | `83c994e` | Benchmark confirmatório (`configs/rb/inject-v1.yaml`), descritores 34b (`configs/rb/feature_schema_34b.json`), famílias (`data_rb/families_freeze_v1.csv`) e regra dos sentinelas. O N efetivo (50 famílias) está no adendo `data_rb/freeze_v1_addendum.json` |
| `prereg-v1` | `98c01d7` | Pré-registro lacrado (03/11/2026); `prereg-v1-llm` no mesmo commit |
| `prereg-v1.1` … `prereg-v1.4` | `c77c29b`, `2e30247`, `8a5cd57`, `57746ca` | Emendas dos desvios D1 a D4, cada uma com a tag `-llm` correspondente. **A produção rodou em `prereg-v1.4`** |
| `resultados-v1` | `64c6f06` | Segundo depósito (hashes das saídas do run oficial) |

## Dados versionados (`data_rb/`)

| Arquivo | Conteúdo |
|---|---|
| `sources_local.csv`, `sources_local.pairs.csv`, `family_overrides.csv` | 101 tabelas, filtros, famílias, representantes e papéis (passo 31) |
| `openml_metadata_20261022.csv` | Nome, versão, licença e URL de origem do OpenML, com o número de linhas conferido |
| `sentinel_exclusions.csv` | Colunas que codificam "não medido" (hoje só a Pima, por impossibilidade física) |
| `families_freeze_v1.csv`, `freeze_v1_manifest.json`, `freeze_v1_addendum.json`, `eligibility_freeze_v1.csv` | Congelamento e N efetivo |
| `synth_manifest.*`, `synth_v2_manifest.*` | Manifestos dos corpora sintéticos, com SHA-256 por instância |
| `case_study_sources.csv` | Registro dos casos do estudo de caso (passo 45): alvo, regra de ausência, Z, hipótese, evidência, citação e motivo de exclusão |
| `case_study_manifest.csv` | 18 fontes do estudo de caso obtidas pela origem: URL, data, SHA-256 dos brutos e da tabela, checagem e licença. Brutos e tabelas ficam fora do git (`case_study_raw/`, `case_study_tables/`) |

## Runs (fora do git: `results_rb/`, `data_rb/synth/`)

| Run | O que é |
|---|---|
| `results_rb/20261021-pilot-dev-cba8168b099b` | Piloto r1. **Inválido:** desempate das máscaras pela ordem do arquivo |
| `results_rb/20261021-pilot-dev-r2-197f689ca15a` | Piloto r2 (válido), nas 8 famílias de desenvolvimento |
| `results_rb/20261022-freeze-check-dev-cc3e7e671881` | Checagem da configuração congelada, mais os testes de fumaça das análises, nas famílias de desenvolvimento (sem baselines) |
| `results_rb/20261022-rehearsal-dev-b4ca243467d6` | **Ensaio geral da produção** (`f3bb944`, árvore limpa): cadeia completa, inclusive baselines com o regime dirty, PKLM, `stats`, `secondary` e `report`, nas famílias de desenvolvimento ([`30-infra.md`](30-infra.md)) |
| `results_rb/20261022-rehearsal-dev-b4ca243467d6/noise/` | RQ3 (passo 44) no ensaio geral (`8b32248`): 19 configurações de ruído, nenhuma falha |
| `results_rb/20261022-rehearsal-dev-b4ca243467d6/final_model/` | Modelo de **desenvolvimento** congelado (`fit-final`), só para o teste de fumaça do estudo de caso |
| `results_rb/_superseded/` | Primeira fumaça do estudo de caso (`28b6a27`/`1fdd523`), anterior às correções da revisão (Z que codificava a máscara em Mroz, Pima e NHANES; formato antigo do modelo) |
| `data_rb/synth/20261021-synth-v1-be5730da3d49` | Corpus do piloto, com os descritores calculados (análise da D7) |
| `data_rb/synth/20261022-synth-v2-5d31a737645b` | Corpus da produção, com os descritores calculados. O primeiro cálculo registrou `dirty: true` (`53a03f9`); foi refeito com a árvore limpa em `1355b73`, com valores idênticos, e o antigo está em `data_rb/synth/_superseded/` |

**Runs oficiais (05–06/11/2026, `prereg-v1.4`):** `results_rb/20261105-inject-v1-52ed6f93a2a0` (run completo, 50 famílias, 10.332 instâncias), `results_rb/20261105-contexto-h7-c635b9dbd73a` (H7) e `results_rb/20261106-case-study-f826114aad4f` (estudo de caso). Até 30/10, data da tabela acima, nenhuma instância das famílias confirmatórias tinha sido gerada.

## Scripts de apoio às decisões

- `scripts/revisao_banca/decisoes/`: curva de aprendizado da D7, tamanho do PKLM, varredura de sentinelas, elegibilidade das famílias no freeze.
- `scripts/revisao_banca/passo31/`: metadados, qualidades e sentinelas do OpenML.
- `scripts/revisao_banca/passo45/`: obtenção das fontes do estudo de caso (`obter_fontes.py`; `exportar_r.R` lê os pacotes do CRAN sem instalar).
