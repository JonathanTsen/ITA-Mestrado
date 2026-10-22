"""Linha de comando dos experimentos da revisão da banca (passo 30 do plano).

    uv run python -m missdetect.rb.cli <comando> --config configs/rb/<arquivo>.yaml [opções]

Comandos:
    sources    tabelas-fonte, filtros e famílias (repassa para `missdetect.rb.sources`)
    inject     gera o benchmark real injetado num run novo (`results_rb/<data>-<nome>-<hash>/`)
    synth      gera o corpus sintético num run novo
    features   descritores observáveis de um run (checkpoint por sample_id)
    baselines  MechDetect-O, painel oráculo, Little e PKLM de um run
    evaluate   avaliador único (CV por família) sobre os descritores e baselines de um run
    report     relatório do piloto (tempos, inelegíveis, detectabilidade, atalhos, sd_Δ, MDE)
    llm        passos 42/43 (depende da D5)
    noise      passo 44
    stats      testes confirmatórios H1–H4 (Wilcoxon-Pratt + Holm, Hodges–Lehmann, ROPE) de um run avaliado

Execute com OMP_NUM_THREADS=1; os workers também limitam threads com `threadpoolctl`.
"""

import argparse
import sys
from pathlib import Path

from missdetect.rb import pipeline, report, sources
from missdetect.rb.config import load_config

COMMANDS = ("sources", "inject", "synth", "features", "baselines", "evaluate", "report", "llm", "noise", "stats")
NOT_YET = {
    "llm": "passos 42/43: dependem da D5 (provedores e orçamento) — ainda não implementado",
    "noise": "passo 44 — ainda não implementado",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="missdetect.rb.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{" + ",".join(COMMANDS) + "}")
    sub.add_parser("sources", help="tabelas-fonte e famílias (argumentos repassados)", add_help=False)
    for name in ("inject", "synth"):
        p = sub.add_parser(name, help=f"{name}: cria um run novo")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--n-jobs", type=int, default=None)
    for name in ("features", "baselines", "evaluate", "report"):
        p = sub.add_parser(name, help=f"{name} de um run existente")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--run", type=Path, required=True, help="diretório do run")
        p.add_argument("--n-jobs", type=int, default=None)
        if name == "baselines":
            p.add_argument("--skip-pklm", action="store_true", help="não roda o PKLM")
            p.add_argument("--only-pklm", action="store_true", help="roda só o PKLM (pklm.parquet)")
        if name == "evaluate":
            p.add_argument("--generators", nargs="*", default=[], help="filtra instâncias por gerador")
    p = sub.add_parser("stats", help="testes confirmatórios H1–H4 de um run avaliado")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="diretório do run (usa eval-symmetric/per_family.parquet)")
    p.add_argument("--llm-per-family", type=Path, default=None, help="per_family.parquet do painel de LLM (H4)")
    for name in NOT_YET:
        p = sub.add_parser(name, help=NOT_YET[name])
        p.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["sources"]:
        sys.argv = ["missdetect.rb.sources", *argv[1:]]
        return sources.main()
    args = build_parser().parse_args(argv)
    if args.command in NOT_YET:
        raise SystemExit(f"{args.command}: {NOT_YET[args.command]}")
    config = load_config(args.config)
    n_jobs = getattr(args, "n_jobs", None) or config.features.n_jobs
    if args.command == "inject":
        print(pipeline.run_inject(config, n_jobs))
    elif args.command == "synth":
        print(pipeline.run_synth(config, n_jobs))
    elif args.command == "features":
        print(pipeline.run_features(args.run, config, n_jobs))
    elif args.command == "baselines":
        n_jobs = args.n_jobs or config.baselines.n_jobs
        print(pipeline.run_baselines(args.run, config, n_jobs, args.skip_pklm, args.only_pklm))
    elif args.command == "evaluate":
        print(pipeline.run_evaluate(args.run, config, n_jobs, tuple(args.generators)))
    elif args.command == "report":
        print(report.run_report(args.run, config))
    elif args.command == "stats":
        print(pipeline.run_stats(args.run, config, args.llm_per_family))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
