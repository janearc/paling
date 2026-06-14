# tests/test_cli.py
"""Tests for the Paling CLI argument parsing and subcommands."""
import sys
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch
import io

import importlib.util

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load paling/cli.py directly bypassing the directory namespace collision
spec = importlib.util.spec_from_file_location("paling_cli_script", str(project_root / "paling/cli.py"))
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

def test_checkpoint_args_parsing(tmp_path, monkeypatch):
    import sys
    from unittest.mock import MagicMock
    
    # Mock the delightd module since it may not be installed
    mock_delightd = MagicMock()
    mock_backup_module = MagicMock()
    mock_backup_module.backup.CreateCheckpoint.return_value = ("/tmp/fake-archive.tgz", None)
    mock_delightd.pkg.backup = mock_backup_module
    
    sys.modules["delightd"] = mock_delightd
    sys.modules["delightd.pkg"] = mock_delightd.pkg
    sys.modules["delightd.pkg.backup"] = mock_backup_module

    import logging
    logging.getLogger("paling_cli_script").setLevel(logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    result = run_cli_in_process([
        "checkpoint",
        "-p",
        "mybento",
        "-b",
        str(tmp_path),
        "--dry-run",
    ], monkeypatch)
    assert result.returncode == 0
    assert "Checkpoint created" in result.stdout or "Checkpoint created" in result.stderr

def test_reward_function():
    from paling.reward import score_response
    # Novelty gives 0.2, length gives 5/200 * 0.4 = 0.01 -> total 0.21
    assert abs(score_response("short") - 0.21) < 1e-5
    # Check that a repeated string loses novelty
    assert abs(score_response("short") - 0.01) < 1e-5
    
    # Long response with cue word
    long_resp = "a" * 31 + " star"
    # length: 36/200 * 0.4 = 0.072, cue: 1/10 * 0.4 = 0.04, novelty: 0.2 -> 0.312
    assert abs(score_response(long_resp) - 0.312) < 1e-5
