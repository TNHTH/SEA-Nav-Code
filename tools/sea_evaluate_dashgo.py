#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Formal 300-fixture evaluation entry for SEA DashGo (A8.1/A8.4).

Formal evaluation requires Isaac runtime. CPU ``--describe`` / ``--validate-only``
validate manifest paths and fixture counts without launching SimulationApp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from _dashgo_common import (
    add_common_flags,
    base_describe,
    configure_pythonpath,
    refuse_train_entry,
    run_tool_main,
)

TOOL = "sea_evaluate_dashgo"
FORMAL_FIXTURE_COUNT = 300


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="SEA DashGo formal evaluation (300 fixtures, Isaac runtime required)",
    )
    add_common_flags(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-manifest", default=None)
    parser.add_argument("--fixture-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def _load_fixture_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def describe_args(args: argparse.Namespace) -> Dict[str, Any]:
    return base_describe(
        TOOL,
        requires_isaac=True,
        extra={
            "checkpoint": args.checkpoint,
            "model_manifest": args.model_manifest,
            "fixture_manifest": args.fixture_manifest,
            "output_dir": args.output_dir,
            "formal_fixture_count": FORMAL_FIXTURE_COUNT,
            "runtime_status": "NOT RUN",
        },
    )


def validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.fixture_manifest:
        manifest_path = Path(args.fixture_manifest)
        if not manifest_path.is_file():
            raise ValueError(f"fixture manifest not found: {manifest_path}")
        manifest = _load_fixture_manifest(manifest_path)
        fixtures = manifest.get("fixtures") or manifest.get("cases")
        if fixtures is None:
            raise ValueError("fixture manifest missing fixtures/cases list")
        if len(fixtures) != FORMAL_FIXTURE_COUNT:
            raise ValueError(
                f"formal evaluation requires exactly {FORMAL_FIXTURE_COUNT} fixtures, got {len(fixtures)}"
            )
    return {
        "status": "validated",
        "tool": TOOL,
        "checkpoint": args.checkpoint,
        "model_manifest": args.model_manifest,
        "fixture_manifest": args.fixture_manifest,
        "output_dir": args.output_dir,
        "formal_fixture_count": FORMAL_FIXTURE_COUNT,
        "runtime_status": "NOT RUN",
    }


def execute(args: argparse.Namespace) -> int:
    required = ("checkpoint", "model_manifest", "fixture_manifest", "output_dir")
    missing = [name for name in required if getattr(args, name.replace("-", "_"), None) is None]
    if missing:
        print(f"evaluate requires: {', '.join('--' + m.replace('_', '-') for m in missing)}", file=__import__("sys").stderr)
        return 2
    for name in required:
        value = getattr(args, name)
        if name == "output_dir":
            continue
        if not Path(value).is_file():
            print(f"{name} not found: {value}", file=__import__("sys").stderr)
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
