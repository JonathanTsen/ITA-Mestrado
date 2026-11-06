#!/usr/bin/env zsh
# B4 do plano 38c (29/10/2026): re-execução de conferência do ensaio geral (8 famílias de desenvolvimento) com o código
# atual, incluindo as etapas novas das emendas. Os números das etapas antigas precisam sair idênticos aos do ensaio de
# 22/10 (results_rb/20261022-rehearsal-dev-b4ca243467d6); a comparação é feita depois, à parte.
#
# Uso (no ITA-Mestrado, com a árvore limpa):
#   nohup zsh scripts/revisao_banca/reexecucao_ensaio.sh > results_rb/_logs/reexecucao-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
# Retomada: passe o run já criado como argumento. O marcador de cada etapa é o arquivo que ela grava por último
# (`run.<etapa>.json`; no report, `report/report.json`); a etapa cujo marcador existe é pulada. Uma etapa interrompida no
# meio deixa saídas parciais, e a CLI se recusa a sobrescrevê-las: mova-as para `results_rb/_superseded/` antes de
# retomar (o `features` e o `inject-ampute` retomam sozinhos pelo checkpoint).
#   nohup zsh scripts/revisao_banca/reexecucao_ensaio.sh results_rb/<run> >> <mesmo log> 2>&1 &
set -euo pipefail
export OMP_NUM_THREADS=1
C=configs/rb/rehearsal-dev.yaml
SYNTH=data_rb/synth/20261022-synth-v2-5d31a737645b
cli() { uv run --frozen python -m missdetect.rb.cli "$@" 2>&1 | { /usr/bin/grep -a -v -E "pkg_resources|get_distribution" || true; }; }
step() {  # step <marcador> <nome> <argumentos da CLI...>
  local marker=$1 name=$2; shift 2
  if [[ -e $marker ]]; then echo "== $name: já feito ($marker), pulado"; return; fi
  echo "== $name: início $(date -u +%FT%TZ)"
  cli "$@"
  echo "== $name: fim $(date -u +%FT%TZ)"
}

echo "inicio $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
uv run --frozen python -I scripts/revisao_banca/checagem_previa.py --config $C 2>&1 | { /usr/bin/grep -a -v -E "pkg_resources|get_distribution" || true; }
RUN=${1:-}
if [[ -z $RUN ]]; then
  RUN=$(cli inject --config $C | tail -1)
  echo "== inject: run $RUN"
fi
step $RUN/run.features.json features features --config $C --run $RUN
step $RUN/run.oracle-features.json oracle-features oracle-features --config $C --run $RUN
step $RUN/run.baselines.json baselines baselines --config $C --run $RUN --skip-pklm
step $RUN/run.baselines-pklm.json pklm baselines --config $C --run $RUN --only-pklm
step $RUN/run.eval-symmetric.json evaluate-symmetric evaluate --config $C --run $RUN --generators symmetric
step $RUN/run.eval-all.json evaluate-all evaluate --config $C --run $RUN
step $RUN/run.eval-mdatagen.json evaluate-mdatagen evaluate --config $C --run $RUN --generators mdatagen
step $RUN/run.eval-tab_err.json evaluate-tab_err evaluate --config $C --run $RUN --generators tab_err
step $RUN/run.cross-generator.json cross-generator cross-generator --config $C --run $RUN
step $RUN/run.stats.json stats stats --config $C --run $RUN
step $RUN/run.secondary.json secondary secondary --config $C --run $RUN
step $RUN/run.secondary-extra.json secondary-extra secondary-extra --config $C --run $RUN
step $RUN/run.inject-ampute.json inject-ampute inject-ampute --config $C --run $RUN
step $RUN/run.cross-ampute.json cross-ampute cross-ampute --config $C --run $RUN
step $RUN/run.ablation.json ablation ablation --config $C --run $RUN
step $RUN/run.logo.json logo logo --config $C --run $RUN
step $RUN/run.transfer-${SYNTH:t}.json transfer transfer --config $C --run $RUN --synth-run $SYNTH
step $RUN/report/report.json report report --config $C --run $RUN
step $RUN/run.noise.json noise noise --config $C --run $RUN
step $RUN/run.fit-final.json fit-final fit-final --config $C --run $RUN
echo "fim $(date -u +%FT%TZ) run=$RUN"
