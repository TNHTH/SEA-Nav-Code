#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Interactive play entry for SEA DashGo (A8.4).

Execution requires Isaac runtime and TRAIN_READY=True.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from _dashgo_common import (
    add_common_flags,
    base_describe,
    configure_pythonpath,
    refuse_train_entry,
    run_tool_main,
)

TOOL = "sea_play_dashgo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL, description="SEA DashGo play (Isaac runtime required)")
    add_common_flags(parser)
    parser.add_argument("--checkpoint", required=False, default=None)
    parser.add_argument("--model-manifest", required=False, default=None)
    parser.add_argument("--fixture-id", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    return parser


def describe_args(args: argparse.Namespace) -> Dict[str, Any]:
    return base_describe(
        TOOL,
        requires_isaac=True,
        extra={
            "checkpoint": args.checkpoint,
            "model_manifest": args.model_manifest,
            "fixture_id": args.fixture_id,
            "duration_seconds": args.duration_seconds,
            "runtime_status": "NOT RUN",
        },
    )


def validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.duration_seconds <= 0:
        raise ValueError("duration-seconds must be positive")
    if args.fixture_id < 0:
        raise ValueError("fixture-id must be non-negative")
    return {
        "status": "validated",
        "tool": TOOL,
        "checkpoint": args.checkpoint,
        "model_manifest": args.model_manifest,
        "fixture_id": args.fixture_id,
        "duration_seconds": args.duration_seconds,
        "runtime_status": "NOT RUN",
    }


def execute(args: argparse.Namespace) -> int:
    if not args.checkpoint or not args.model_manifest:
        print("play requires --checkpoint and --model-manifest", file=sys.stderr)
        return 2
    checkpoint = Path(args.checkpoint)
    manifest = Path(args.model_manifest)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    if not manifest.is_file():
        print(f"model manifest not found: {manifest}", file=sys.stderr)
        return 1
    refuse_train_entry()
    return 3


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
