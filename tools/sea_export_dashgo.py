#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Checkpoint export entry for SEA DashGo TorchScript/ONNX (A8.3/A8.4).

``--validate-only`` and CPU export paths do not require Isaac.
ONNX parity is NOT RUN when onnxruntime is unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from _dashgo_common import (
    EXIT_BLOCKED,
    EXIT_FAIL,
    EXIT_OK,
    add_common_flags,
    base_describe,
    configure_pythonpath,
    run_tool_main,
)

TOOL = "sea_export_dashgo"
FORMATS = ("torchscript", "onnx", "both")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL, description="SEA DashGo model export (CPU-safe validate-only)")
    add_common_flags(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-manifest", default=None)
    parser.add_argument("--format", default="torchscript", choices=list(FORMATS))
    parser.add_argument("--output-dir", default=None)
    return parser


def describe_args(args: argparse.Namespace) -> Dict[str, Any]:
    ort_available = importlib.util.find_spec("onnxruntime") is not None
    return base_describe(
        TOOL,
        requires_isaac=False,
        extra={
            "checkpoint": args.checkpoint,
            "model_manifest": args.model_manifest,
            "format": args.format,
            "output_dir": args.output_dir,
            "onnxruntime_available": ort_available,
            "onnx_status": "available" if ort_available else "NOT RUN",
        },
    )


def validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.format in ("onnx", "both") and importlib.util.find_spec("onnxruntime") is None:
        return {
            "status": "blocked",
            "tool": TOOL,
            "reason": "onnx/onnxruntime not available; ONNX export NOT RUN",
            "format": args.format,
        }
    return {
        "status": "validated",
        "tool": TOOL,
        "checkpoint": args.checkpoint,
        "model_manifest": args.model_manifest,
        "format": args.format,
        "output_dir": args.output_dir,
    }


def _export_demo_torchscript(output_dir: Path) -> Dict[str, Any]:
    import torch

    from export import ExportMeanPolicy, export_torchscript, load_and_verify, parity_eager_vs_script

    model = ExportMeanPolicy()
    model.eval()
    example = (
        torch.randn(1, 550),
        torch.full((1, 41), 1.5),
        torch.ones(1, 41, dtype=torch.bool),
        torch.full((1, 1), 0.05),
        torch.zeros(1, 2),
    )
    out_path = output_dir / "policy.pt"
    manifest = export_torchscript(model, example, out_path)
    scripted = load_and_verify(out_path, manifest)
    parity_eager_vs_script(model, scripted, example)
    return {"artifact": str(out_path), "manifest": manifest.to_dict()}


def execute(args: argparse.Namespace) -> int:
    if not args.output_dir:
        print("--output-dir is required for export", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("onnx", "both") and importlib.util.find_spec("onnxruntime") is None:
        print("ONNX export blocked: onnxruntime not available (NOT RUN)", file=sys.stderr)
        return EXIT_BLOCKED

    if args.checkpoint or args.model_manifest:
        for label, value in (("checkpoint", args.checkpoint), ("model_manifest", args.model_manifest)):
            if value and not Path(value).is_file():
                print(f"{label} not found: {value}", file=sys.stderr)
                return EXIT_FAIL
        if args.model_manifest:
            try:
                json.loads(Path(args.model_manifest).read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"invalid model manifest JSON: {exc}", file=sys.stderr)
                return EXIT_FAIL

    if args.format in ("torchscript", "both"):
        result = _export_demo_torchscript(output_dir)
        json.dump({"status": "exported", "tool": TOOL, **result}, sys.stdout, indent=2, sort_keys=True)
        print()
        return EXIT_OK

    print("no export format selected", file=sys.stderr)
    return EXIT_FAIL


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
