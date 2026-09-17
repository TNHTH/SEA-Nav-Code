# SPDX-License-Identifier: MIT
"""Shared CPU-safe helpers for SEA DashGo CLI tools (A7.2/A8.4).

Importing this module must not load Isaac, ROS, or SimulationApp.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

PROFILES = ("full", "without_acsi", "without_shield", "without_lreg")
PRESETS = ("smoke_4060", "formal_train")
SMOKE_NUM_ENVS = (32, 16, 8, 4)
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_pythonpath() -> Path:
    root = repo_root()
    for rel in (
        "packages/sea_nav_core/src",
        "training/sea_nav_diffdrive_isaaclab",
        "training/rsl_rl",
    ):
        path = str(root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Emit resolved JSON inventory for this tool",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate argv/config without Isaac side effects",
    )


def validate_profile(profile: str) -> None:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")


def validate_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")


def validate_preset(preset: str) -> None:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset}")


def validate_num_envs(num_envs: Optional[int], *, preset: str) -> None:
    if num_envs is None:
        return
    if preset == "formal_train":
        raise ValueError("formal_train rejects --num-envs budget override")
    if num_envs not in SMOKE_NUM_ENVS:
        raise ValueError(f"--num-envs must be one of {SMOKE_NUM_ENVS} for smoke")


def base_describe(tool: str, *, requires_isaac: bool, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import env_cfg

    payload: Dict[str, Any] = {
        "tool": tool,
        "requires_isaac": requires_isaac,
        "train_ready": env_cfg.TRAIN_READY,
        "train_block_reason": env_cfg.TRAIN_BLOCK_REASON,
        "repo_root": str(repo_root()),
        "profiles": list(PROFILES),
        "presets": list(PRESETS),
    }
    if extra:
        payload.update(extra)
    return payload


def emit_json(data: Dict[str, Any]) -> None:
    json.dump(data, sys.stdout, indent=2, sort_keys=True)
    print()


def refuse_train_entry() -> None:
    import env_cfg

    print(env_cfg.TRAIN_BLOCK_REASON, file=sys.stderr)
    raise SystemExit(EXIT_BLOCKED)


def run_tool_main(
    *,
    tool: str,
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]],
    requires_isaac: bool,
    describe_fn: Callable[[argparse.Namespace], Dict[str, Any]],
    validate_fn: Callable[[argparse.Namespace], Dict[str, Any]],
    execute_fn: Callable[[argparse.Namespace], int],
) -> int:
    args = parser.parse_args(argv)
    if args.describe:
        emit_json(describe_fn(args))
        return EXIT_OK
    if args.validate_only:
        try:
            emit_json(validate_fn(args))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_USAGE
        return EXIT_OK
    if requires_isaac:
        import env_cfg

        if not env_cfg.TRAIN_READY:
            refuse_train_entry()
    return execute_fn(args)


def supervisor_env(run_root: Path, run_id: str, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    if base:
        env.update(base)
    env["SEA_RUN_DIR"] = str(run_root.resolve())
    env["SEA_RUN_ID"] = run_id
    return env
