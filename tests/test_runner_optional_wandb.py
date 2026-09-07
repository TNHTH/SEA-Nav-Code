from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RSL_ROOT = ROOT / "training" / "rsl_rl"


def _subprocess_env() -> dict[str, str]:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "training/rsl_rl",
    }


def test_runners_import_without_wandb_installed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['wandb'] = None; "
                "import pathlib, rsl_rl, rsl_rl.runners; "
                "print(pathlib.Path(rsl_rl.__file__).resolve())"
            ),
        ],
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == (RSL_ROOT / "rsl_rl" / "__init__.py").resolve()


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
