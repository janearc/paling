# tests/test_cli.py
"""Tests for the Paling CLI argument parsing and subcommands."""
import sys
from pathlib import Path
import io

import importlib.util

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load paling/cli.py directly bypassing the directory namespace collision
spec = importlib.util.spec_from_file_location(
    "paling_cli_script", str(project_root / "paling/cli.py")
)
paling_cli_script = importlib.util.module_from_spec(spec)
sys.modules["paling_cli_script"] = paling_cli_script
spec.loader.exec_module(paling_cli_script)
main = paling_cli_script.main

# Helper to run the module in-process
def run_cli_in_process(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["paling"] + args)
    
    # Capture stdout and stderr
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_stdout)
    monkeypatch.setattr(sys, "stderr", captured_stderr)
    
    returncode = 0
    try:
        main()
    except SystemExit as e:
        returncode = e.code if e.code is not None else 0
        
    class Result:
        pass
    res = Result()
    res.returncode = returncode
    res.stdout = captured_stdout.getvalue()
    res.stderr = captured_stderr.getvalue()
    return res

def test_paint_args_parsing(monkeypatch):
    # Monkeypatch the painter to avoid actual model calls
    import paling.painter as painter
    def fake_run_painter(*args, **kwargs):
        return 0
    monkeypatch.setattr(painter, "run_painter", fake_run_painter)

    result = run_cli_in_process([
        "paint",
        "-t",
        "dummy-target",
        "-s",
        "1",
        "-r",
        "0.5",
    ], monkeypatch)
    assert result.returncode == 0
    assert result.stderr == ""

def test_reward_function():
    from paling.reward import score_response, _emotional_charge, _SEEN

    # Sentiment, not a word checklist, drives the reward: a genuinely felt line
    # has more emotional charge than flat, factual prose -- in either direction.
    felt = "I love you so much it terrifies me."
    flat = "The package shipped on Tuesday."
    assert _emotional_charge(felt) > _emotional_charge(flat)

    # Score stays in [0, 1], and the novelty bonus is only paid on first sight.
    _SEEN.clear()
    first = score_response(felt)
    assert 0.0 <= first <= 1.0
    assert score_response(felt) < first  # exact repeat loses the 0.2 novelty bonus

    # Empty response scores zero.
    assert score_response("") == 0.0
