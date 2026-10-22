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
| `baselines/` | 36, 39 | MechDetect oficial e observed-only, Little, PKLM (R), logística simples |
| `pipeline.py`, `report.py` | 37 | etapas de um run e relatório do piloto |
| `stats.py` | 38/46 | H1–H4: Wilcoxon-Pratt + Holm, Hodges–Lehmann com bootstrap, rank-biserial, ROPE (baycomp), decisão |
| `analyses.py` | 39/40/46 | L1 secundária, recall por mecanismo, L2, classificadores descritivos (CD), sensibilidade da massa na fronteira |
| `baselines/oracle_descriptors.py` | 39 | descritores-oráculo (Y completo) do painel de informação privilegiada |

## Layout de um run

`results_rb/<AAAAMMDD>-<nome>-<hash da config>/` (fora do git; o corpus sintético fica em `data_rb/synth/`):

| Arquivo | Quem lê |
|---|---|
| `run.json`, `run.<etapa>.json` | todos: commits do código e do plano, sha256 do `uv.lock`, config, sementes, versões, máquina, data |
| `eligibility.parquet`, `bases.parquet`, `contexts.parquet`, `skipped.parquet` | desenho |
| `instances.parquet` | `sample_id`, `context_id`, `family_id`, `observed_path`, `realized_rate` — **sem** mecanismo |
| `observed/<sample_id>.parquet` | descritores e baselines observed-only: coluna `y` (NaN nas ausentes) + auxiliares |
| `oracle.parquet` | só a avaliação: mecanismo, gerador, variante, determinismo, hash da máscara |
| `oracle_complete/<context_id>.parquet` | só o painel oráculo e os testes: Y completo |
| `features.parquet` (+ `features_parts/`) | checkpoint por `sample_id` |
| `baselines.parquet`, `pklm.parquet` | MechDetect-O, MechDetect oráculo, Little; PKLM |
| `eval-<geradores>/` | `predictions.parquet`, `per_family.parquet`, `summary.csv`, `choices.json` |
| `report/` | relatório do piloto |

## Ordem de execução

```bash
export OMP_NUM_THREADS=1
uv run --frozen python -m missdetect.rb.cli inject    --config configs/rb/pilot-dev.yaml
uv run --frozen python -m missdetect.rb.cli features  --config configs/rb/pilot-dev.yaml --run <run>
uv run --frozen python -m missdetect.rb.cli baselines --config configs/rb/pilot-dev.yaml --run <run>
uv run --frozen python -m missdetect.rb.cli evaluate  --config configs/rb/pilot-dev.yaml --run <run> --generators symmetric
uv run --frozen python -m missdetect.rb.cli evaluate  --config configs/rb/pilot-dev.yaml --run <run>
uv run --frozen python -m missdetect.rb.cli report    --config configs/rb/pilot-dev.yaml --run <run>
```

O PKLM pode rodar à parte com `baselines --only-pklm` (ou ser pulado com `--skip-pklm`).

Análises do pré-registro (depois do `evaluate`):

```bash
uv run --frozen python -m missdetect.rb.cli stats            --config <cfg> --run <run> [--llm-per-family <painel>]
uv run --frozen python -m missdetect.rb.cli oracle-features  --config <cfg> --run <run>   # antes do evaluate, para o painel oráculo
uv run --frozen python -m missdetect.rb.cli ablation         --config <cfg> --run <run>
uv run --frozen python -m missdetect.rb.cli logo             --config <cfg> --run <run>
uv run --frozen python -m missdetect.rb.cli transfer         --config <cfg> --run <run> --synth-run data_rb/synth/<synth-v2>
uv run --frozen python -m missdetect.rb.cli secondary        --config <cfg> --run <run>
```
Os CSV das tabelas vêm de `MechDetect/src/auc_roc_scores/datasets.tar.gz` (ver `sources.py`).

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
