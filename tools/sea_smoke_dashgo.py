#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""4060 smoke gate runner for SEA DashGo (A7.2).

Gate receipts are checked in ledger order; missing prior receipts block later gates.
Real smoke execution requires Isaac runtime (NOT RUN on CPU-only hosts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from _dashgo_common import (
    EXIT_BLOCKED,
    EXIT_FAIL,
    EXIT_USAGE,
    add_common_flags,
    base_describe,
    configure_pythonpath,
    refuse_train_entry,
    run_tool_main,
)

TOOL = "sea_smoke_dashgo"
GATE_RANGE = range(1, 14)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL, description="SEA DashGo smoke gate ledger runner")
    add_common_flags(parser)
    parser.add_argument("--gate", type=int, default=None, help="Single gate id (1-13)")
    parser.add_argument("--all-gates", action="store_true", help="Run gates 1..13 in order")
    parser.add_argument("--receipt-root", default=None, help="Directory containing gate receipt JSON files")
    return parser


def _receipt_path(root: Path, gate: int) -> Path:
    return root / f"gate_{gate:02d}_receipt.json"


def check_prior_receipts(receipt_root: Path, gate: int) -> List[str]:
    missing: List[str] = []
    for prior in range(1, gate):
        if not _receipt_path(receipt_root, prior).is_file():
            missing.append(f"gate_{prior:02d}")
    return missing


def describe_args(args: argparse.Namespace) -> Dict[str, Any]:
    return base_describe(
        TOOL,
        requires_isaac=True,
        extra={
            "gate": args.gate,
            "all_gates": args.all_gates,
            "receipt_root": args.receipt_root,
            "runtime_status": "NOT RUN",
        },
    )


def validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.gate is None and not args.all_gates:
        raise ValueError("specify --gate or --all-gates")
    if args.gate is not None and args.gate not in GATE_RANGE:
        raise ValueError("--gate must be in 1..13")
    if args.receipt_root is not None and not Path(args.receipt_root).is_dir():
        raise ValueError("--receipt-root must be an existing directory")
    return {
        "status": "validated",
        "tool": TOOL,
        "gate": args.gate,
        "all_gates": args.all_gates,
        "receipt_root": args.receipt_root,
        "runtime_status": "NOT RUN",
    }


def execute(args: argparse.Namespace) -> int:
    gates = list(GATE_RANGE) if args.all_gates else [int(args.gate)]
    receipt_root = Path(args.receipt_root) if args.receipt_root else None
    if receipt_root is not None:
        for gate in gates:
            missing = check_prior_receipts(receipt_root, gate)
            if missing:
                print(
                    f"blocked: gate {gate} missing prior receipts: {', '.join(missing)}",
                    file=__import__("sys").stderr,
                )
                return EXIT_BLOCKED
            receipt = _receipt_path(receipt_root, gate)
            if receipt.is_file():
                try:
                    payload = json.loads(receipt.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    print(f"invalid receipt JSON: {receipt}", file=__import__("sys").stderr)
                    return EXIT_FAIL
                if payload.get("status") != "passed":
                    print(f"gate {gate} receipt not passed", file=__import__("sys").stderr)
                    return EXIT_FAIL
    refuse_train_entry()
    return EXIT_BLOCKED


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_pythonpath()
    return run_tool_main(
        tool=TOOL,
        parser=build_parser(),
        argv=argv,
        requires_isaac=True,
        describe_fn=describe_args,
        validate_fn=validate_args,
        execute_fn=execute,
    )


if __name__ == "__main__":
    raise SystemExit(main())
