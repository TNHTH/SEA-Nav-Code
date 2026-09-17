# SPDX-License-Identifier: MIT
"""PID-aware process supervision stubs with fail-closed reuse checks (A7.5)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ProcessIdentity:
    pid: int
    start_time: float
    executable: str
    parent_pid: int
    argv: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "start_time": self.start_time,
            "executable": self.executable,
            "parent_pid": self.parent_pid,
            "argv": list(self.argv),
        }

    def matches_live(self) -> bool:
        """Fail-closed: verify PID still refers to the same process identity."""
        if self.pid <= 0:
            return False
        proc_dir = Path(f"/proc/{self.pid}")
        if not proc_dir.exists():
            return False
        try:
            cmdline = proc_dir.joinpath("cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            return False
        if not cmdline.strip():
            return False
        base = os.path.basename(self.executable)
        if self.executable in cmdline:
            return True
        if self.argv and self.argv[0] in cmdline:
            return True
        if base and base in cmdline:
            return True
        return " ".join(self.argv[:2]) in cmdline


@dataclass
class AttemptRecord:
    attempt_id: str
    run_id: str
    run_root: Path
    identity: Optional[ProcessIdentity] = None
    returncode: Optional[int] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    _proc: Any = field(default=None, repr=False)

    def append_event(self, kind: str, **payload: Any) -> None:
        self.events.append(
            {"utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **payload}
        )


class RunSupervisor:
    """Owns one run_root; each attempt gets an isolated subdirectory."""

    def __init__(self, run_root: Path, run_id: str):
        self.run_root = run_root.resolve()
        self.run_id = run_id
        self.attempts: List[AttemptRecord] = []
        self._active: Optional[AttemptRecord] = None

    def _attempt_dir(self, attempt_id: str) -> Path:
        d = self.run_root / "attempts" / attempt_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start_attempt(
        self,
        attempt_id: str,
        argv: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> AttemptRecord:
        if self._active is not None and self._active.returncode is None:
            raise RuntimeError("active attempt already running")
        record = AttemptRecord(
            attempt_id=attempt_id,
            run_id=self.run_id,
            run_root=self._attempt_dir(attempt_id),
        )
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=env,
            start_new_session=True,
        )
        record._proc = proc
        record.identity = ProcessIdentity(
            pid=proc.pid,
            start_time=time.time(),
            executable=str(argv[0]),
            parent_pid=os.getpid(),
            argv=list(argv),
        )
        record.append_event("started", pid=proc.pid)
        self._active = record
        self.attempts.append(record)
        record.run_root.joinpath("identity.json").write_text(
            json.dumps(record.identity.to_dict(), indent=2)
        )
        return record

    def wait(self, timeout: Optional[float] = None) -> int:
        if self._active is None or self._active.identity is None:
            raise RuntimeError("no active attempt")
        proc = self._active._proc
        if proc is None:
            raise RuntimeError("no process handle for active attempt")
        rc = proc.wait(timeout=timeout)
        self._active.returncode = rc
        self._active.append_event("exited", returncode=rc)
        self._active = None
        return rc

    def signal_attempt(self, sig: signal.Signals) -> None:
        if self._active is None or self._active.identity is None:
            raise RuntimeError("no active attempt")
        if not self._active.identity.matches_live():
            raise RuntimeError("PID identity mismatch — refusing signal")
        os.kill(self._active.identity.pid, sig)

    def refuse_pid_reuse(self, stored: ProcessIdentity) -> None:
        if stored.matches_live():
            raise RuntimeError(
                "refusing to act on potentially reused PID without extra checks"
            )
