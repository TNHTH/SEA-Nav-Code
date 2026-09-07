from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RSL_ROOT = ROOT / "training" / "rsl_rl"
NUMPY_OVERRIDE = Path("/tmp/sea-nav-cpu-20260907.l2Pp6b/site")


def _subprocess_env() -> dict[str, str]:
    python_paths = [str(NUMPY_OVERRIDE), str(RSL_ROOT)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    return dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(python_paths),
    )


def test_runners_import_without_wandb_installed() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import rsl_rl.runners; print('runner-imported')"],
        text=True,
        capture_output=True,
        env=_subprocess_env(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "runner-imported"


def test_missing_enabled_wandb_has_a_precise_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from rsl_rl.runners import on_policy_runner

    def missing(name: str):
        assert name == "wandb"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="wandb logging requested"):
        on_policy_runner._require_wandb()


def test_wandb_flag_is_safe_when_args_are_absent() -> None:
    from rsl_rl.runners import on_policy_runner

    class DisabledArgs:
        wandb = False

    class EnabledArgs:
        wandb = True

    assert on_policy_runner._wandb_enabled(None) is False
    assert on_policy_runner._wandb_enabled(DisabledArgs()) is False
    assert on_policy_runner._wandb_enabled(EnabledArgs()) is True
