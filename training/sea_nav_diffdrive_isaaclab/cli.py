# SPDX-License-Identifier: MIT
"""CPU-safe CLI surface for SEA DashGo tools (A7.2/A8.4).

Importing this module must not load Isaac, ROS, or SimulationApp.

``TRAIN_READY`` remains hard-false until A4–A5 integration completes; Isaac-bound
tools must refuse real execution even when A6–A9 CPU modules are present.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import env_cfg

CLI_VERSION = "sea_nav_dashgo_cli_v1"

TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "train": {
        "module": "tools.sea_train_dashgo",
        "description": "PPO training entry (requires Isaac runtime)",
        "requires_isaac": True,
    },
    "play": {
        "module": "tools.sea_play_dashgo",
        "description": "Interactive play (requires Isaac runtime)",
        "requires_isaac": True,
    },
    "smoke": {
        "module": "tools.sea_smoke_dashgo",
        "description": "4060 smoke training (requires Isaac runtime)",
        "requires_isaac": True,
    },
    "evaluate": {
        "module": "tools.sea_evaluate_dashgo",
        "description": "Formal 300-fixture evaluation",
        "requires_isaac": True,
    },
    "export": {
        "module": "tools.sea_export_dashgo",
        "description": "TorchScript/ONNX export",
        "requires_isaac": False,
    },
    "summarize": {
        "module": "tools.sea_summarize_dashgo",
        "description": "Aggregate evaluation statistics",
        "requires_isaac": False,
    },
    "validate": {
        "module": "tools.sea_validate_dashgo",
        "description": "CPU/static validation harness",
        "requires_isaac": False,
    },
}


def build_parser(prog: str = "sea_dashgo") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="SEA DashGo adapter CLI")
    parser.add_argument("--describe", action="store_true", help="Emit JSON tool inventory")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config identity without Isaac side effects",
    )
    parser.add_argument("--profile", default="full", help="Ablation profile name")
    parser.add_argument("--master-seed", type=int, default=42)
    return parser


def describe() -> Dict[str, Any]:
    return {
        "cli_version": CLI_VERSION,
        "train_ready": env_cfg.TRAIN_READY,
        "train_block_reason": env_cfg.TRAIN_BLOCK_REASON,
        "env_cfg": env_cfg.env_cfg_source(),
        "tools": TOOL_SPECS,
    }


def validate_only(profile: str, master_seed: int) -> Dict[str, Any]:
    source = env_cfg.env_cfg_source()
    env_cfg.validate_env_cfg_source(source)
    allowed_profiles = {"full", "without_acsi", "without_shield", "without_lreg"}
    if profile not in allowed_profiles:
        raise ValueError(f"unknown profile: {profile}")
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    return {
        "status": "validated",
        "profile": profile,
        "master_seed": master_seed,
        "train_ready": env_cfg.TRAIN_READY,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe:
        json.dump(describe(), sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    if args.validate_only:
        try:
            result = validate_only(args.profile, args.master_seed)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
