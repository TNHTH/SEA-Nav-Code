#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evaluation corpus summarizer for SEA DashGo (A8.2/A8.4).

Pure CPU: validates episode schema/hash identity then aggregates statistics.
Incomplete corpora emit diagnostic summaries only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from _dashgo_common import (
    EXIT_FAIL,
    EXIT_OK,
    EXIT_USAGE,
    add_common_flags,
    base_describe,
    configure_pythonpath,
    run_tool_main,
)

TOOL = "sea_summarize_dashgo"
FORMAL_CASE_COUNT = 3600


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL, description="SEA DashGo evaluation summarizer (CPU-safe)")
    add_common_flags(parser)
    parser.add_argument("--evaluation-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def describe_args(args: argparse.Namespace) -> Dict[str, Any]:
    return base_describe(
        TOOL,
        requires_isaac=False,
        extra={
            "evaluation_root": args.evaluation_root,
            "output_dir": args.output_dir,
            "formal_case_count": FORMAL_CASE_COUNT,
        },
    )


def validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.evaluation_root and not Path(args.evaluation_root).is_dir():
        raise ValueError("--evaluation-root must be an existing directory")
    if args.output_dir:
        out = Path(args.output_dir)
        if out.exists() and not out.is_dir():
            raise ValueError("--output-dir must be a directory")
    return {
        "status": "validated",
        "tool": TOOL,
        "evaluation_root": args.evaluation_root,
        "output_dir": args.output_dir,
    }


def execute(args: argparse.Namespace) -> int:
    from episode_schema import validate_corpus
    from statistics import aggregate_profile_seeds

    if not args.evaluation_root or not args.output_dir:
        print("--evaluation-root and --output-dir are required", file=sys.stderr)
        return EXIT_USAGE

    eval_root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(eval_root.glob("**/*.jsonl"))
    if not jsonl_files:
        print(f"no JSONL episodes under {eval_root}", file=sys.stderr)
        return EXIT_FAIL

    records: List[Dict[str, Any]] = []
    for path in jsonl_files:
        records.extend(_load_jsonl(path))

    try:
        validate_corpus(records)
    except ValueError as exc:
        summary = {
            "status": "diagnostic_incomplete",
            "tool": TOOL,
            "reason": str(exc),
            "record_count": len(records),
            "formal_case_count": FORMAL_CASE_COUNT,
        }
        out_path = output_dir / "summary_diagnostic.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()
        return EXIT_OK

    by_seed: Dict[int, List[str]] = {}
    for rec in records:
        by_seed.setdefault(int(rec["seed"]), []).append(str(rec["outcome"]))

    aggregate = aggregate_profile_seeds(by_seed)
    complete = len(records) == FORMAL_CASE_COUNT
    summary = {
        "status": "formal" if complete else "diagnostic_partial",
        "tool": TOOL,
        "record_count": len(records),
        "formal_case_count": FORMAL_CASE_COUNT,
        "aggregate": aggregate,
    }
    out_path = output_dir / ("summary_formal.json" if complete else "summary_diagnostic.json")
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    print()
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_pythonpath()
    return run_tool_main(
        tool=TOOL,
        parser=build_parser(),
        argv=argv,
        requires_isaac=False,
        describe_fn=describe_args,
        validate_fn=validate_args,
        execute_fn=execute,
    )


if __name__ == "__main__":
    raise SystemExit(main())
