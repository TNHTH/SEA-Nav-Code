#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Process supervisor wrapper for SEA DashGo runs (A7.5).

Starts an owned process group without shell interpolation and exports
``SEA_RUN_DIR`` / ``SEA_RUN_ID`` to the child environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from _dashgo_common import configure_pythonpath, repo_root, supervisor_env

TOOL = "sea_run_supervisor"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL, description="SEA DashGo run supervisor")
    parser.add_argument("--run-root", required=True, help="Root directory for attempt artifacts")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command argv after '--'",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_pythonpath()
    from supervisor import RunSupervisor

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("missing child command after '--'")
    if args.command[0] == "--":
        command = args.command[1:]
    else:
        command = list(args.command)
    if not command:
        parser.error("empty child command")

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    sup = RunSupervisor(run_root, args.run_id)
    child_env = supervisor_env(run_root, args.run_id)
    attempt_id = f"{args.run_id}-attempt-{len(sup.attempts) + 1}"
    sup.start_attempt(attempt_id, command, cwd=repo_root(), env=child_env)
    return sup.wait()


if __name__ == "__main__":
    raise SystemExit(main())
