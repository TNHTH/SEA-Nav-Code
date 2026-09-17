# SPDX-License-Identifier: MIT
"""CLI help/describe/validate-only without Isaac import (A7.2/A8.4)."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(PACKAGE_ROOT))


def test_cli_import_has_no_isaac():
    spec = importlib.util.find_spec("isaaclab")
    # Module may not exist; ensure cli import does not pull it in.
    before = "isaaclab" in sys.modules
    import cli  # noqa: F401
    after = "isaaclab" in sys.modules
    assert after == before or spec is None


def test_describe_json():
    import cli

    data = cli.describe()
    assert "tools" in data
    assert data["train_ready"] is False
    assert "env_cfg" in data


def test_validate_only_ok(capsys):
    import cli

    rc = cli.main(["--validate-only", "--profile", "full", "--master-seed", "42"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "validated"


def test_validate_only_bad_profile():
    import cli

    rc = cli.main(["--validate-only", "--profile", "bogus"])
    assert rc == 2


def test_help_exits_zero(capsys):
    import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "SEA DashGo" in capsys.readouterr().out


TOOL_SCRIPTS = [
    "sea_train_dashgo.py",
    "sea_play_dashgo.py",
    "sea_smoke_dashgo.py",
    "sea_evaluate_dashgo.py",
    "sea_export_dashgo.py",
    "sea_summarize_dashgo.py",
    "sea_validate_dashgo.py",
    "sea_run_supervisor.py",
]


@pytest.mark.parametrize("script", TOOL_SCRIPTS)
def test_tool_help_exits_zero(script):
    import subprocess

    tool = TOOLS_ROOT / script
    proc = subprocess.run(
        [sys.executable, str(tool), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()


@pytest.mark.parametrize("script", ["sea_train_dashgo.py", "sea_export_dashgo.py"])
def test_tool_describe_json(script):
    import subprocess

    tool = TOOLS_ROOT / script
    proc = subprocess.run(
        [sys.executable, str(tool), "--describe"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["train_ready"] is False
    assert "tool" in data


def test_train_refuses_execution():
    import subprocess

    tool = TOOLS_ROOT / "sea_train_dashgo.py"
    proc = subprocess.run(
        [sys.executable, str(tool), "--profile", "full", "--preset", "smoke_4060"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3
    assert "TRAIN_READY" in proc.stderr or "refused" in proc.stderr
