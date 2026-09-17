#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PPO training entry for SEA DashGo (A7.2).

Real training requires Isaac SimulationApp and TRAIN_READY=True.
``--help``, ``--describe``, and ``--validate-only`` are CPU-safe.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Sequence

from _dashgo_common import (
    EXIT_BLOCKED,
    add_common_flags,
    base_describe,
    configure_pythonpath,
    refuse_train_entry,
    run_tool_main,
    validate_num_envs,
    validate_preset,
    validate_profile,
    validate_seed,
)

TOOL = "sea_train_dashgo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="SEA DashGo PPO training (Isaac runtime required for execution)",
    )
    add_common_flags(parser)
    parser.add_argument("--profile", default="full", choices=["full", "without_acsi", "without_shield", "without_lreg"])
    parser.add_argument("--preset", default="smoke_4060", choices=["smoke_4060", "formal_train"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=None, choices=[32, 16, 8, 4])
    parser.add_argument("--resume", default=None, help="Checkpoint path for resume attempt")
    return parser


def describe_args(args: argparse.Namespace) -> Dict[str, Any]:
    return base_describe(
        TOOL,
        requires_isaac=True,
        extra={
            "profile": args.profile,
            "preset": args.preset,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "resume": args.resume,
            "runtime_status": "NOT RUN",
        },
    )


def validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    validate_profile(args.profile)
    validate_preset(args.preset)
    validate_seed(args.seed)
    validate_num_envs(args.num_envs, preset=args.preset)
    return {
        "status": "validated",
        "tool": TOOL,
        "profile": args.profile,
        "preset": args.preset,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "runtime_status": "NOT RUN",
    }


def execute(args: argparse.Namespace) -> int:
    del args
    refuse_train_entry()
    return EXIT_BLOCKED  # unreachable


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
