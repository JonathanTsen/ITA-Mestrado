"""Testes do passo 30: a CLI lista todos os comandos e os não implementados falham com mensagem explícita."""

import subprocess
import sys

import pytest

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


@pytest.mark.parametrize("command", ["llm", "noise"])
def test_pending_commands_fail_explicitly(command):
    with pytest.raises(SystemExit, match=r"não implementado|D5"):
        main([command])
