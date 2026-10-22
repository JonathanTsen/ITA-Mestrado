# Infraestrutura dos experimentos da revisão (passo 30)

Pacote `src/missdetect/rb/`, isolado do código legado. Nenhum módulo executa código ao ser importado.

## Módulos

| Módulo | Passo | Papel |
|---|---|---|
| `config.py` | 30 | YAML → dataclasses imutáveis; chave desconhecida é erro; hash da config canônica |
| `ids.py` | 30 | `stable_id` (SHA-256 do JSON canônico) e sementes por `SeedSequence` |
| `runs.py` | 30 | diretório do run (nunca sobrescreve), `run.json`, gravação separada observado × oráculo |
| `cli.py` | 30 | `python -m missdetect.rb.cli <comando> --config configs/rb/<arq>.yaml` |
| `sources.py` | 31 | tabelas-fonte, filtros, famílias, papéis dev/confirm |
| `inject.py`, `bench.py` | 32 | geradores e elegibilidade; desenho por família (bases × n × k; β e direção balanceados) |
| `synth.py` | 33 | corpus sintético (cópula gaussiana, R² latente controlado, mesmos geradores) |
| `features.py` | 34 | descritores observáveis (schema `34b`, congelado em `freeze-v1`) |
| `splits.py`, `models.py`, `evaluate.py` | 35 | CV por família, direto/hierárquico, calibração interna, métricas |
| `baselines/` | 36, 39 | MechDetect oficial (clean e dirty) e observed-only, Little, PKLM (R), logística simples |
| `pipeline.py`, `report.py` | 37 | etapas de um run e relatório (tempos, inelegíveis, detectabilidade, atalhos, resumos separados observed-only × painel oráculo) |
| `stats.py` | 38/46 | H1–H4: Wilcoxon-Pratt + Holm, Hodges–Lehmann com bootstrap, rank-biserial, ROPE (baycomp), decisão |
| `analyses.py` | 39/40/46 | L1 secundária, recall por mecanismo, L2, classificadores descritivos (CD), sensibilidade da massa na fronteira |
| `baselines/oracle_descriptors.py` | 39 | descritores-oráculo (Y completo) do painel de informação privilegiada |
| `noise.py` | 44 | RQ3: corrupção coerente por contexto só no treino, flags e mitigações do Confident Learning (pesos, poda), contrastes (Holm m = 4), curvas, qualidade das flags e sobreposição (ORA-14) |
| `case_study.py` | 45 | estudo de caso com ausência natural: registro, regras de ausência documentadas, auxiliares completas, sem fabricar mecanismo |

## Layout de um run

`results_rb/<AAAAMMDD>-<nome>-<hash da config>/` (fora do git; o corpus sintético fica em `data_rb/synth/`):

| Arquivo | Quem lê |
|---|---|
| `run.json`, `run.<etapa>.json` | todos: commits do código e do plano (com flag de árvore suja), sha256 do `uv.lock`, config, sementes, versões (Python, R, PKLMtest), máquina, data |
| `eligibility.parquet`, `bases.parquet`, `contexts.parquet`, `skipped.parquet`, `family_status.parquet`, `timing_inject.parquet` | desenho: tentativas de elegibilidade, bases, contextos, colisões, sentinelas excluídos, trios recusados e famílias sem base |
| `instances.parquet` | `sample_id`, `context_id`, `family_id`, `observed_path`, `realized_rate` — **sem** mecanismo |
| `observed/<sample_id>.parquet` | descritores e baselines observed-only: coluna `y` (NaN nas ausentes) + auxiliares em ordem sorteada |
| `oracle.parquet` | só a avaliação: mecanismo, gerador, variante, determinismo, hash da máscara |
| `oracle_complete/<context_id>.parquet` | só o painel oráculo e os testes: Y completo |
| `features.parquet` (+ `features_parts/`), `feature_schema.json` | descritores 34b, com checkpoint por `sample_id` |
| `oracle_features.parquet` | descritores-oráculo, só do painel oráculo |
| `baselines.parquet`, `pklm.parquet` | MechDetect-O, MechDetect oráculo (clean), MechDetect dirty, Little; PKLM (subconjunto) |
| `eval-symmetric/`, `eval-all/`, `eval-<gerador>/` | `predictions.parquet`, `per_family.parquet`, `summary.csv`, `choices.json` |
| `stats_confirmatory.csv` | H1–H4 (Holm m = 4) |
| `secondary/` | L1 secundária, recall por mecanismo, L2, CD, sensibilidade da massa na fronteira |
| `eval-ablation/`, `eval-logo/`, `transfer-<corpus>/` | RQ2, leave-one-generator-out e transferência sintético → real |
| `noise/` | RQ3: `predictions.parquet` (teste limpo), `flags.parquet`, `per_family.parquet`, `rq3_tests.csv`, `degradation_curves.csv`, `flag_summary.csv`, `overlap_by_beta.csv`, `overlap_by_r2.csv`, `settings.csv` (taxa realizada), `degradation.png`; hashes dos rótulos em `run.noise.json` |
| `final_model/` | `model.joblib` + `model.json` (SHA-256, colunas, faixa de k, famílias): configuração primária congelada para o estudo de caso |
| `report/` | relatório: tempos, inelegíveis, detectabilidade, atalhos, sd_Δ e os resumos **separados** `summary_primary.csv` (observed-only) e `summary_oracle_panel.csv`. O `summary.csv` de cada `eval-*` é bruto e lista todos os métodos, inclusive os do painel oráculo |

## Ordem de execução da produção (depois da tag `prereg-v1`)

Os CSV-fonte ficam em `/tmp/rb_mechdetect_run/mech-detect-datasets/` (fora do repositório). Se tiverem sumido, reextraia com `tar -xzf ~/work/MechDetect/src/auc_roc_scores/datasets.tar.gz -C /tmp/rb_mechdetect_run/`. O `inject` confere o SHA-256 de cada arquivo antes de ler.

```bash
export OMP_NUM_THREADS=1
C=configs/rb/inject-v1.yaml
uv run --frozen python -m missdetect.rb.cli inject          --config $C                # → results_rb/<run>
uv run --frozen python -m missdetect.rb.cli features        --config $C --run <run>
uv run --frozen python -m missdetect.rb.cli oracle-features --config $C --run <run>    # antes do evaluate
uv run --frozen python -m missdetect.rb.cli baselines       --config $C --run <run>    # PKLM: 8 contextos por família
uv run --frozen python -m missdetect.rb.cli evaluate        --config $C --run <run> --generators symmetric
uv run --frozen python -m missdetect.rb.cli evaluate        --config $C --run <run>
uv run --frozen python -m missdetect.rb.cli evaluate        --config $C --run <run> --generators mdatagen
uv run --frozen python -m missdetect.rb.cli evaluate        --config $C --run <run> --generators tab_err
uv run --frozen python -m missdetect.rb.cli stats           --config $C --run <run>    # H1–H3 (+ H4 com --llm-per-family)
uv run --frozen python -m missdetect.rb.cli secondary       --config $C --run <run>
uv run --frozen python -m missdetect.rb.cli ablation        --config $C --run <run>
uv run --frozen python -m missdetect.rb.cli logo            --config $C --run <run>
uv run --frozen python -m missdetect.rb.cli transfer        --config $C --run <run> --synth-run data_rb/synth/20261022-synth-v2-5d31a737645b
uv run --frozen python -m missdetect.rb.cli report          --config $C --run <run>    # resumos observed-only × painel oráculo
uv run --frozen python -m missdetect.rb.cli noise           --config $C --run <run>    # RQ3 (passo 44)
uv run --frozen python -m missdetect.rb.cli fit-final       --config $C --run <run>    # modelo congelado (passo 45)
uv run --frozen python scripts/revisao_banca/passo45/obter_fontes.py                # se data_rb/case_study_tables/ não existir
uv run --frozen python -m missdetect.rb.cli case-study      --config configs/rb/case-study.yaml --model <run>/final_model --registry data_rb/case_study_sources.csv --tables data_rb/case_study_tables
uv run --frozen python -m missdetect.rb.cli evaluate        --config configs/rb/synth-v2.yaml --run data_rb/synth/20261022-synth-v2-5d31a737645b   # controle sintético
```

O PKLM pode rodar à parte com `baselines --only-pklm` (ou ser pulado com `--skip-pklm`). O piloto usou a mesma sequência com `configs/rb/pilot-dev.yaml`.

**Ensaio geral (22/10/2026):** a sequência completa acima, incluindo `report`, rodou sem erro com `configs/rb/rehearsal-dev.yaml` (inject-v1 nas 8 famílias de desenvolvimento; PKLM reduzido a 2 contextos por família) no commit `f3bb944`, com a árvore limpa. Run `results_rb/20261022-rehearsal-dev-b4ca243467d6`: 1.632 instâncias, nenhuma falha em nenhuma etapa, ≈ 52 min de relógio. Tempos por instância (núcleo): descritores 1,22 s; descritores-oráculo ≈ 0,6 s; MechDetect-O 3,98 s; oráculo 3,53 s; dirty 2,36 s; PKLM 57,1 s.

**RQ3 e estudo de caso (22/10/2026):** `noise` rodou no mesmo run (`8b32248`; 975 instâncias, 19 configurações, 182 s, nenhuma falha; a referência limpa reproduz exatamente o `direct_hgb` da avaliação primária). `fit-final` + `case-study` rodaram como teste de fumaça com o modelo de desenvolvimento. A revisão independente encontrou auxiliares que codificavam a máscara em três casos (Mroz, Pima, NHANES), corrigidas em `5ba99b0` junto com as salvaguardas (colunas e schema no arquivo do modelo, SHA-256 das tabelas contra o manifesto, diagnóstico `mask_auc_from_z`); a fumaça foi refeita depois disso.

## Testes que provam a ausência de vazamento (`tests/rb/`)

| Teste do plano 30 | Onde |
|---|---|
| 1 sementes entre processos | `test_ids.py` |
| 2 disjunção por família; 3 pareamento 1:1 | `test_evaluate.py`, `test_pipeline.py` |
| 4 oráculo lacrado; 5 permutação de linhas | `test_features.py` |
| 6 sem rótulo no observado | `test_runs.py`, `test_pipeline.py` |
| 7 falha vira erro | `test_evaluate.py`, `pipeline.mark_failures` |
| 8 imputação só no treino | `test_evaluate.py` |
| regressão do Ŷ cross-fit (achado do piloto) | `test_baselines.py` |
| RQ3: ruído só no treino e coerente por contexto; teste sempre com o rótulo verdadeiro (hash e `check_test_labels`); referência limpa idêntica à avaliação primária; sem pesos, `DirectClassifier` idêntico ao original; `noise/` nunca fica pela metade | `test_noise.py`, `test_pipeline.py` |
| Estudo de caso: Y observado idêntico à fonte e nenhuma ausência preenchida (IND-22); registro válido e coerente com o manifesto e com os sentinelas da D4; diagnóstico de Z que codifica a máscara; modelo (com colunas e schema) e tabelas protegidos por hash; nenhuma coluna de acerto | `test_case_study.py`, `test_pipeline.py` |
