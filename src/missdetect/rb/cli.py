"""Linha de comando dos experimentos da revisão da banca (passo 30 do plano).

    uv run python -m missdetect.rb.cli <comando> --config configs/rb/<arquivo>.yaml [opções]

Comandos:
    sources    tabelas-fonte, filtros e famílias (repassa para `missdetect.rb.sources`)
    inject     gera o benchmark real injetado num run novo (`results_rb/<data>-<nome>-<hash>/`)
    synth      gera o corpus sintético num run novo
    features   descritores observáveis de um run (checkpoint por sample_id)
    baselines  MechDetect-O, MechDetect oráculo (clean e dirty), Little e PKLM de um run
    evaluate   avaliador único (CV por família) sobre os descritores e baselines de um run
    report     relatório do piloto (tempos, inelegíveis, detectabilidade, atalhos, sd_Δ, MDE)
    llm        passos 42/43: painel de LLM de um run (A1 ou A2), síncrono ou em lote, com caderno de respostas;
               o gemini-4-argon só roda com --authorization (regra do autor); --budget-only só projeta o custo
    llm-probes sondas de memorização (P21) nas famílias do painel: tabela crua (covariável) e renderização cega do A1
               (critério de exclusão da H4); mesmas opções do llm
    llm-sensitivity  H4 com 200 linhas: primária retreinada nas subamostras que o LLM vê (mesmas partições); descritiva
    noise      RQ3: ruído de rótulo só no treino, mitigação com Confident Learning (pesos, poda) e flags (passo 44)
    fit-final  congela a configuração primária treinada no gerador principal de um run (SHA-256; passo 45)
    case-study aplica o modelo congelado aos casos com ausência natural do registro, sem métrica de acerto (passo 45)
    stats      família confirmatória H1, H2, H2e, H3, H5, H6 (Wilcoxon-Pratt, TOST, Holm m = 6); saídas gravadas juntas
    stats-llm  H4 (LLM-A1 × configuração primária, m = 1), etapa separada, depois do painel de LLM
    oracle-features  descritores-oráculo (Y completo) para o painel oráculo; rodar antes do evaluate
    ablation   RQ2: todos os descritores × sem cada grupo G1–G4 (Holm m = 4)
    logo       leave-one-generator-out (exploratória)
    transfer   treino só no corpus sintético, aplicado ao run real (exploratória; --synth-run)
    secondary  L1 secundária, recall por mecanismo, L2, classificadores descritivos (CD), sensibilidade, painel oráculo
               por gerador, mdatagen e estatísticas do MechDetect como descritores
    cross-generator  H5: primária treinada só no gerador principal, testada nos geradores externos; rodar antes do stats
    secondary-extra  análises das emendas de 29/10 (P6, P7, P8, P11, P12, P13, P15); depois do cross-generator
    inject-ampute    P24: instâncias do ampute (Schouten et al. 2018) só para teste, com descritores; sem predição
    cross-ampute     P24: modelos treinados no simétrico aplicados ao ampute; secundária m = 1 e mudança

Execute com OMP_NUM_THREADS=1; os workers também limitam threads com `threadpoolctl`.
"""

import argparse
import sys
from pathlib import Path

from missdetect.rb import pipeline, report, sources
from missdetect.rb.config import load_config
from missdetect.rb.llm.run import ARMS, MODES, build_panel, run_llm, run_probes
from missdetect.rb.llm.sensitivity import run_llm_sensitivity
from missdetect.rb.runs import confirmatory_guard, repo_root

COMMANDS = (
    "sources", "inject", "synth", "features", "baselines", "evaluate", "report", "stats",
    "oracle-features", "ablation", "logo", "transfer", "secondary", "llm", "noise", "fit-final", "case-study",
    "cross-generator", "stats-llm", "secondary-extra", "inject-ampute", "cross-ampute", "llm-probes",
    "llm-sensitivity",
)  # fmt: skip
NOT_YET: dict[str, str] = {}


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
    p = sub.add_parser("stats", help="testes confirmatórios H1, H2, H2e, H3, H5, H6 de um run avaliado")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="diretório do run (usa eval-symmetric/ e eval-cross/)")
    for name in ("llm", "llm-probes"):
        p = sub.add_parser(name, help="painel de LLM de um run (passos 42/43; H4)" if name == "llm" else
                           "sondas de memorização nas famílias do painel (P21)")  # fmt: skip
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--run", type=Path, required=True, help="diretório do run")
        p.add_argument("--model", required=True, help="alias do modelo (ex.: gemini-3.8-flash)")
        if name == "llm":
            p.add_argument("--arm", choices=sorted(ARMS), default="A1")
        p.add_argument("--mode", choices=MODES, default="batch", help="batch (−50%%, plano pago) ou sync")
        p.add_argument("--authorization", default=None, help="autorização do autor (obrigatória para modelos ≠ Flash)")
        p.add_argument("--env-file", type=Path, default=None, help="arquivo .env com as chaves (padrão: raiz do repo)")
        p.add_argument("--max-workers", type=int, default=4, help="chamadas síncronas em paralelo")
        p.add_argument("--budget-only", action="store_true", help="só conta tokens e projeta o custo, sem gerar")
    p = sub.add_parser("llm-sensitivity", help="H4 com 200 linhas (primária retreinada; descritiva)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="diretório do run")
    p.add_argument("--llm-predictions", type=Path, default=None, help="predictions.parquet do LLM-A1, para comparar")
    p.add_argument("--n-jobs", type=int, default=None)
    p = sub.add_parser("stats-llm", help="H4: LLM-A1 × configuração primária no painel de LLM (etapa separada)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="diretório do run")
    p.add_argument("--llm-per-family", type=Path, required=True, help="per_family.parquet do painel de LLM (H4)")
    for name in (
        "oracle-features",
        "ablation",
        "logo",
        "transfer",
        "secondary",
        "noise",
        "fit-final",
        "cross-generator",
        "secondary-extra",
        "inject-ampute",
        "cross-ampute",
    ):
        p = sub.add_parser(name, help=f"{name} de um run existente")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--run", type=Path, required=True, help="diretório do run")
        p.add_argument("--n-jobs", type=int, default=None)
        if name == "transfer":
            p.add_argument("--synth-run", type=Path, required=True, help="corpus sintético com descritores (synth-v2)")
        if name == "secondary-extra":
            p.add_argument("--nominal-list", type=Path, default=None, help="lista fechada de Y nominal (P15)")
    p = sub.add_parser("case-study", help="estudo de caso com ausência natural (modelo congelado)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="diretório final_model/ de um run (fit-final)")
    p.add_argument("--registry", type=Path, required=True, help="registro dos casos (data_rb/case_study_sources.csv)")
    p.add_argument("--tables", type=Path, required=True, help="diretório das tabelas padronizadas (<fonte>.csv)")
    p.add_argument(
        "--manifest", type=Path, default=Path("data_rb/case_study_manifest.csv"),
        help="manifesto com o SHA-256 de cada tabela (conferido antes de usar)",
    )  # fmt: skip
    p.add_argument("--n-jobs", type=int, default=None)
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
    if "confirm" in config.sources.roles:  # pré-registro §9 (P3) e §4 (P21)
        confirmatory_guard(repo_root(), args.command)
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
    elif args.command == "oracle-features":
        print(pipeline.run_oracle_features(args.run, config, n_jobs))
    elif args.command == "ablation":
        print(pipeline.run_ablation(args.run, config, n_jobs))
    elif args.command == "logo":
        print(pipeline.run_logo(args.run, config, n_jobs))
    elif args.command == "transfer":
        print(pipeline.run_transfer(args.synth_run, args.run, config))
    elif args.command == "secondary":
        print(pipeline.run_secondary(args.run, config))
    elif args.command == "noise":
        print(pipeline.run_noise(args.run, config, n_jobs))
    elif args.command == "cross-generator":
        print(pipeline.run_cross_generator(args.run, config, n_jobs))
    elif args.command == "secondary-extra":
        print(pipeline.run_secondary_extra(args.run, config, args.nominal_list))
    elif args.command == "inject-ampute":
        print(pipeline.run_inject_ampute(args.run, config, n_jobs))
    elif args.command == "cross-ampute":
        print(pipeline.run_cross_ampute(args.run, config, n_jobs))
    elif args.command == "fit-final":
        print(pipeline.run_fit_final(args.run, config))
    elif args.command == "case-study":
        print(pipeline.run_case_study(args.model, args.registry, args.tables, config, n_jobs, args.manifest))
    elif args.command == "stats":
        print(pipeline.run_stats(args.run, config))
    elif args.command == "llm":
        out = run_llm(
            args.run, config, model=args.model, arm=args.arm, mode=args.mode, authorization=args.authorization,
            env_file=args.env_file, max_workers=args.max_workers, budget_only=args.budget_only,
        )  # fmt: skip
        print(out)
    elif args.command == "llm-probes":
        out = run_probes(
            args.run, config, model=args.model, mode=args.mode, authorization=args.authorization,
            env_file=args.env_file, max_workers=args.max_workers, budget_only=args.budget_only,
        )  # fmt: skip
        print(out)
    elif args.command == "llm-sensitivity":
        panel = build_panel(args.run, config)
        panel_ids = panel.loc[panel.replicate == 0, "sample_id"].tolist()
        print(run_llm_sensitivity(args.run, config, n_jobs, panel_ids, args.llm_predictions))
    elif args.command == "stats-llm":
        print(pipeline.run_stats_llm(args.run, config, args.llm_per_family))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
