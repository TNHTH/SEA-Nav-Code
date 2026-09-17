#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Unified CPU/static validator for SEA DashGo milestone A (A9.1).

Runs real subprocess checks with preserved argv/returncode. Does not treat
AST string presence as API completeness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from _dashgo_common import (
    EXIT_BLOCKED,
    EXIT_FAIL,
    EXIT_OK,
    EXIT_USAGE,
    configure_pythonpath,
    repo_root,
)

TOOL = "sea_validate_dashgo"
CPU_PY_DEFAULT = os.environ.get(
    "CPU_PY",
    sys.executable,
)

A6_A9_CPU_TESTS = [
    "training/sea_nav_diffdrive_isaaclab/tests/test_acsi.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_snapshot.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_rng.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_checkpoint.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_runner.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_cli.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_evaluation.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_episode_schema.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_statistics.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_export.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_supervisor.py",
    "training/sea_nav_diffdrive_isaaclab/tests/test_static_api.py",
    "tests/test_sea_runtime_preflight.py",
    "packages/sea_nav_core/tests",
]


@dataclass
class CheckResult:
    name: str
    argv: List[str]
    returncode: int
    status: str
    log_path: Optional[str] = None
    detail: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=TOOL, description="SEA DashGo CPU/static validator")
    parser.add_argument("--level", choices=["cpu", "static-api"], default=None)
    parser.add_argument("--output-dir", default=None, help="Directory for JSON/JUnit/raw logs")
    parser.add_argument(
        "--isaaclab-source",
        default=None,
        help="Required for --level static-api: path to Isaac Lab source tree",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="List symbolic checks without executing",
    )
    parser.add_argument(
        "--python",
        default=CPU_PY_DEFAULT,
        help="Python executable for subprocess pytest (default: CPU_PY or sys.executable)",
    )
    return parser


def pythonpath_env(root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    paths = [
        root / "packages/sea_nav_core/src",
        root / "training/sea_nav_diffdrive_isaaclab",
        root / "training/rsl_rl",
    ]
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    return env


def list_checks() -> List[str]:
    configure_pythonpath()
    from static_api import list_checks as static_checks

    checks = list(static_checks())
    checks.extend(
        [
            "tool:sea_train_dashgo",
            "tool:sea_play_dashgo",
            "tool:sea_smoke_dashgo",
            "tool:sea_evaluate_dashgo",
            "tool:sea_export_dashgo",
            "tool:sea_summarize_dashgo",
            "tool:sea_run_supervisor",
            "tool:sea_runtime_preflight",
            "pytest:a6_a9_cpu_suite",
        ]
    )
    return sorted(set(checks))


def _write_junit(path: Path, results: Sequence[CheckResult]) -> None:
    suite = ET.Element("testsuite", name=TOOL, tests=str(len(results)))
    for result in results:
        case = ET.SubElement(suite, "testcase", classname=TOOL, name=result.name)
        if result.status != "passed":
            ET.SubElement(case, "failure", message=result.detail or result.status)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def run_cpu_suite(root: Path, python_exe: str, output_dir: Optional[Path]) -> CheckResult:
    argv = [
        python_exe,
        "-m",
        "pytest",
        *A6_A9_CPU_TESTS,
        "-q",
        "--tb=short",
    ]
    log_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(output_dir / "pytest_a6_a9_cpu.log")
    proc = subprocess.run(
        argv,
        cwd=str(root),
        env=pythonpath_env(root),
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    if log_path:
        Path(log_path).write_text(combined, encoding="utf-8")
    status = "passed" if proc.returncode == 0 else "failed"
    return CheckResult(
        name="pytest:a6_a9_cpu_suite",
        argv=argv,
        returncode=proc.returncode,
        status=status,
        log_path=log_path,
        detail="" if status == "passed" else "pytest failures; see log",
    )


def run_tool_help(root: Path, python_exe: str, script: str) -> CheckResult:
    path = root / "tools" / script
    argv = [python_exe, str(path), "--help"]
    proc = subprocess.run(argv, cwd=str(root), capture_output=True, text=True)
    status = "passed" if proc.returncode == 0 else "failed"
    return CheckResult(
        name=f"tool:{script.replace('.py', '')}",
        argv=argv,
        returncode=proc.returncode,
        status=status,
        detail="" if status == "passed" else proc.stderr.strip(),
    )


def run_static_api_check(isaaclab_source: Path) -> CheckResult:
    configure_pythonpath()
    from static_api import inventory

    argv = ["static_api.inventory", str(isaaclab_source)]
    if not isaaclab_source.is_dir():
        return CheckResult(
            name="static_api.inventory",
            argv=argv,
            returncode=EXIT_BLOCKED,
            status="blocked",
            detail=f"isaaclab source not found: {isaaclab_source}",
        )
    inv = inventory()
    if not inv.get("function_consumer_test_map"):
        return CheckResult(
            name="static_api.inventory",
            argv=argv,
            returncode=EXIT_FAIL,
            status="failed",
            detail="empty function_consumer_test_map",
        )
    return CheckResult(
        name="static_api.inventory",
        argv=argv,
        returncode=EXIT_OK,
        status="passed",
        detail=f"symbols={len(inv['function_consumer_test_map'])}",
    )


def assemble_report(level: str, results: Sequence[CheckResult]) -> Dict[str, Any]:
    blocked = [r for r in results if r.status == "blocked"]
    failed = [r for r in results if r.status == "failed"]
    if blocked:
        overall = "blocked"
    elif failed:
        overall = "failed"
    else:
        overall = "passed"
    return {
        "schema_version": 1,
        "tool": TOOL,
        "level": level,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "runtime_verified": False,
        "runtime_status": "NOT RUN",
        "checks": [
            {
                "name": r.name,
                "argv": r.argv,
                "returncode": r.returncode,
                "status": r.status,
                "log_path": r.log_path,
                "detail": r.detail,
            }
            for r in results
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()

    if args.list_checks:
        json.dump({"checks": list_checks()}, sys.stdout, indent=2, sort_keys=True)
        print()
        return EXIT_OK

    if args.level is None:
        parser.error("--level is required unless --list-checks")

    output_dir = Path(args.output_dir) if args.output_dir else None
    results: List[CheckResult] = []

    if args.level == "cpu":
        tool_scripts = [
            "sea_train_dashgo.py",
            "sea_play_dashgo.py",
            "sea_smoke_dashgo.py",
            "sea_evaluate_dashgo.py",
            "sea_export_dashgo.py",
            "sea_summarize_dashgo.py",
            "sea_run_supervisor.py",
            "sea_runtime_preflight.py",
        ]
        for script in tool_scripts:
            results.append(run_tool_help(root, args.python, script))
        results.append(run_cpu_suite(root, args.python, output_dir))
    elif args.level == "static-api":
        if not args.isaaclab_source:
            print("--isaaclab-source is required for --level static-api", file=sys.stderr)
            return EXIT_USAGE
        results.append(run_static_api_check(Path(args.isaaclab_source)))
    else:
        return EXIT_USAGE

    report = assemble_report(args.level, results)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validate_report.json").write_text(text + "\n", encoding="utf-8")
        _write_junit(output_dir / "validate_report.xml", results)

    if report["overall"] == "passed":
        return EXIT_OK
    if report["overall"] == "blocked":
        return EXIT_BLOCKED
    return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
