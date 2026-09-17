# SPDX-License-Identifier: MIT
"""Supervisor PID identity and attempt isolation tests (A7.5)."""

import sys
import tempfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from supervisor import ProcessIdentity, RunSupervisor  # noqa: E402


def test_supervisor_runs_true_and_records_attempt():
    with tempfile.TemporaryDirectory() as tmp:
        sup = RunSupervisor(Path(tmp), "run-1")
        rec = sup.start_attempt("a1", ["/bin/true"])
        rc = sup.wait(timeout=5.0)
        assert rc == 0
        assert rec.returncode == 0
        assert (rec.run_root / "identity.json").is_file()
        assert len(sup.attempts) == 1


def test_second_attempt_while_active_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        sup = RunSupervisor(Path(tmp), "run-2")
        sup.start_attempt(
            "a1",
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        with pytest.raises(RuntimeError, match="active"):
            sup.start_attempt("a2", ["/bin/true"])
        sup.signal_attempt(__import__("signal").SIGTERM)
        sup.wait(timeout=5.0)


def test_refuse_pid_reuse_when_live():
    ident = ProcessIdentity(
        pid=1,  # init usually pid 1
        start_time=0.0,
        executable="systemd",
        parent_pid=0,
        argv=["systemd"],
    )
    sup = RunSupervisor(Path(tempfile.mkdtemp()), "run-3")
    if ident.matches_live():
        with pytest.raises(RuntimeError, match="refusing"):
            sup.refuse_pid_reuse(ident)
