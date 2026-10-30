"""Testes do passo 30: a CLI lista todos os comandos e os não implementados falham com mensagem explícita."""

import subprocess
import sys

import pytest

from missdetect.rb import cli
from missdetect.rb.cli import COMMANDS, main


def test_help_lists_all_commands():
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "missdetect.rb.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for command in ("sources", "inject", "features", "evaluate", "baselines", "llm", "noise", "stats", "synth"):
        assert command in out.stdout
    assert set(COMMANDS) >= {"sources", "inject", "features", "evaluate", "baselines", "llm", "noise", "stats"}


@pytest.mark.parametrize("command", ["llm"])
def test_pending_commands_fail_explicitly(command):
    with pytest.raises(SystemExit, match=r"não implementado|D5"):
        main([command])


def test_noise_is_implemented_and_requires_a_run(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["noise", "--config", "configs/rb/inject-v1.yaml"])
    assert exc.value.code == 2  # erro do argparse: falta --run
    assert "--run" in capsys.readouterr().err


def test_stats_llm_is_its_own_command(capsys):
    """C1 (29/10): a H4 sai do `stats` e vira a etapa `stats-llm`, que exige o painel de LLM."""
    assert "stats-llm" in COMMANDS
    with pytest.raises(SystemExit) as exc:
        main(["stats", "--config", "configs/rb/inject-v1.yaml", "--run", "x", "--llm-per-family", "y"])
    assert exc.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(["stats-llm", "--config", "configs/rb/inject-v1.yaml", "--run", "x"])
    assert exc.value.code == 2
    assert "--llm-per-family" in capsys.readouterr().err


def test_case_study_commands_are_listed_and_require_their_inputs(capsys):
    assert {"fit-final", "case-study"} <= set(COMMANDS)
    with pytest.raises(SystemExit) as exc:
        main(["case-study", "--config", "configs/rb/inject-v1.yaml"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--model" in err and "--registry" in err and "--tables" in err


def test_confirmatory_configs_go_through_the_guard(monkeypatch):
    """C3: com papel `confirm` (inject-v1), a CLI chama a trava antes de qualquer etapa; com `dev`, não."""
    calls = []

    def guard(repo, command):
        calls.append(command)
        raise RuntimeError("trava")

    monkeypatch.setattr(cli, "confirmatory_guard", guard)
    monkeypatch.setattr(cli.pipeline, "run_inject", lambda config, n_jobs: "rodou")
    with pytest.raises(RuntimeError, match="trava"):
        main(["inject", "--config", "configs/rb/inject-v1.yaml"])
    assert calls == ["inject"]
    assert main(["inject", "--config", "configs/rb/rehearsal-dev.yaml"]) == 0
    assert calls == ["inject"]
