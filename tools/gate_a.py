#!/usr/bin/env python3
"""Portable CPU/static foundation gate for the complete SEA-Nav checkout."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import io
import json
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


_STATUSES = frozenset(("passed", "blocked", "failed"))

_LEGGED_GYM_BASELINE_INVENTORY = Path(__file__).with_name("gate_a_legged_gym_baseline_v1.txt")

_EXPECTED_ISAAC_BOUNDARIES = frozenset(
    (
        "training/legged_gym/legged_gym/envs/base/base_task.py",
        "training/legged_gym/legged_gym/envs/base/legged_robot.py",
        "training/legged_gym/legged_gym/envs/base/legged_robot_pos.py",
        "training/legged_gym/legged_gym/utils/helpers.py",
        "training/legged_gym/legged_gym/utils/terrain.py",
        "training/legged_gym/legged_gym/utils/torch_math.py",
        "training/legged_gym/legged_gym/scripts/play.py",
    )
)


@dataclass(frozen=True)
class GateCase:
    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError("unsupported Gate A status: {}".format(self.status))


def discover_repo_root(start: Optional[Path] = None) -> Path:
    candidate = Path(start) if start is not None else Path(__file__)
    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for path in (candidate, *candidate.parents):
        if (path / ".git").exists() and (path / "training" / "rsl_rl").is_dir():
            return path
    raise FileNotFoundError("could not discover SEA-Nav repository root from {}".format(candidate))


def _tracked_python_files(repo_root: Path) -> List[Path]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", "*.py"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip() or "git ls-files failed")
    return [repo_root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _compile_python(repo_root: Path) -> GateCase:
    try:
        paths = _tracked_python_files(repo_root)
        if not paths:
            raise RuntimeError("no tracked Python files found")
        for path in paths:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec", dont_inherit=True)
    except Exception as exc:
        return GateCase("python_syntax", "failed", "{}: {}".format(type(exc).__name__, exc))
    return GateCase("python_syntax", "passed", "compiled {} tracked Python files without output".format(len(paths)))


def _load_legged_gym_baseline_inventory() -> List[str]:
    lines = _LEGGED_GYM_BASELINE_INVENTORY.read_text(encoding="utf-8").splitlines()
    paths = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not paths:
        raise RuntimeError("legged_gym baseline inventory is empty")
    if paths != sorted(paths):
        raise RuntimeError("legged_gym baseline inventory must be sorted")
    if len(paths) != len(set(paths)):
        raise RuntimeError("legged_gym baseline inventory contains duplicate paths")
    for relative in paths:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative.startswith("training/legged_gym/"):
            raise RuntimeError("invalid legged_gym baseline path: {}".format(relative))
    return paths


def _check_complete_legged_gym_tree(repo_root: Path) -> GateCase:
    try:
        required_paths = _load_legged_gym_baseline_inventory()
        missing = []
        empty = []
        non_regular = []
        symlink_substitutions = set()
        for relative in required_paths:
            current = repo_root
            entry_stat = None
            for component in Path(relative).parts:
                current = current / component
                try:
                    entry_stat = current.lstat()
                except FileNotFoundError:
                    entry_stat = None
                    break
                if stat.S_ISLNK(entry_stat.st_mode):
                    symlink_substitutions.add(current.relative_to(repo_root).as_posix())
                    entry_stat = None
                    break
            if entry_stat is None:
                if not any(relative.startswith(link + "/") or relative == link for link in symlink_substitutions):
                    missing.append(relative)
            elif not stat.S_ISREG(entry_stat.st_mode):
                non_regular.append(relative)
            elif entry_stat.st_size == 0:
                empty.append(relative)
    except Exception as exc:
        return GateCase("legged_gym_complete_tree", "failed", "{}: {}".format(type(exc).__name__, exc))
    if missing or empty or non_regular or symlink_substitutions:
        details = []
        if missing:
            details.append("missing: {}".format(", ".join(missing)))
        if empty:
            details.append("empty: {}".format(", ".join(empty)))
        if non_regular:
            details.append("non-regular: {}".format(", ".join(non_regular)))
        if symlink_substitutions:
            details.append("symlink substitutions: {}".format(", ".join(sorted(symlink_substitutions))))
        return GateCase("legged_gym_complete_tree", "failed", "; ".join(details))
    return GateCase(
        "legged_gym_complete_tree",
        "passed",
        "{} versioned baseline package, entry-point, controller, license, and Go2 asset files are present and non-empty".format(
            len(required_paths)
        ),
    )


def _imports_isaacgym(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "isaacgym" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "isaacgym" or node.module.startswith("isaacgym.")
        ):
            return True
    return False


def _check_legged_gym_static_boundary(repo_root: Path) -> GateCase:
    package_root = repo_root / "training" / "legged_gym" / "legged_gym"
    try:
        python_paths = sorted(package_root.rglob("*.py"))
        if not python_paths:
            raise RuntimeError("legged_gym package has no Python files")
        isaac_bound = set()
        for path in python_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if _imports_isaacgym(tree):
                isaac_bound.add(path.relative_to(repo_root).as_posix())
        missing_boundaries = sorted(_EXPECTED_ISAAC_BOUNDARIES - isaac_bound)
        if missing_boundaries:
            raise RuntimeError("expected simulator import boundaries absent: {}".format(", ".join(missing_boundaries)))
        root_tree = ast.parse((package_root / "__init__.py").read_text(encoding="utf-8"))
        if _imports_isaacgym(root_tree):
            raise RuntimeError("legged_gym root package became simulator-bound")
    except Exception as exc:
        return GateCase("legged_gym_static_boundary", "failed", "{}: {}".format(type(exc).__name__, exc))
    return GateCase(
        "legged_gym_static_boundary",
        "passed",
        "parsed {} package files; {} simulator-bound files remain static-only".format(
            len(python_paths), len(isaac_bound)
        ),
    )


def _import_cpu_packages(repo_root: Path) -> GateCase:
    rsl_root = (repo_root / "training" / "rsl_rl").resolve()
    module_names = (
        "rsl_rl",
        "rsl_rl.algorithms",
        "rsl_rl.env",
        "rsl_rl.modules",
        "rsl_rl.runners",
        "rsl_rl.storage",
        "rsl_rl.utils",
    )
    try:
        imported = [importlib.import_module(name) for name in module_names]
        for module in imported:
            module_path = Path(module.__file__).resolve()
            if rsl_root not in module_path.parents:
                raise RuntimeError("{} imported from outside checkout: {}".format(module.__name__, module_path))
    except Exception as exc:
        return GateCase("rsl_rl_cpu_imports", "failed", "{}: {}".format(type(exc).__name__, exc))
    return GateCase("rsl_rl_cpu_imports", "passed", "imported {} CPU-safe packages".format(len(module_names)))


def _run_rsl_cpu_smoke() -> GateCase:
    try:
        import torch
        from rsl_rl.algorithms import PPO
        from rsl_rl.modules import ActorCritic
        from rsl_rl.storage import RolloutStorage

        with contextlib.redirect_stdout(io.StringIO()):
            actor = ActorCritic(
                num_actions=3,
                actor_hidden_dims=[8],
                critic_hidden_dims=[8],
                encoder_hidden_dims=[8],
                num_props=12,
                num_rays=31,
                his_len=2,
            )
        observations = torch.zeros(2, 90)
        actions = actor.act_inference(observations)
        values = actor.evaluate(observations)
        storage = RolloutStorage(2, 3, [90], [3], device="cpu")
        algorithm = PPO(actor, device="cpu")
        algorithm.init_storage(2, 3, [90], [3])
        if tuple(actions.shape) != (2, 3) or tuple(values.shape) != (2, 1):
            raise RuntimeError("unexpected actor output shapes: actions={}, values={}".format(actions.shape, values.shape))
        if tuple(storage.actions.shape) != (3, 2, 3):
            raise RuntimeError("unexpected rollout shape: {}".format(storage.actions.shape))
    except Exception as exc:
        return GateCase("rsl_rl_cpu_smoke", "failed", "{}: {}".format(type(exc).__name__, exc))
    return GateCase("rsl_rl_cpu_smoke", "passed", "actor, value, PPO, and rollout-storage CPU smoke completed")


def probe_isaac_gym() -> GateCase:
    try:
        importlib.import_module("isaacgym")
    except ModuleNotFoundError:
        return GateCase(
            "isaac_gym_runtime",
            "blocked",
            "Isaac Gym Preview 4 is not installed; simulator execution remains blocked",
        )
    except Exception as exc:
        return GateCase("isaac_gym_runtime", "failed", "{}: {}".format(type(exc).__name__, exc))
    return GateCase("isaac_gym_runtime", "passed", "real Isaac Gym package imported")


def run_cpu_gate(repo_root: Path) -> List[GateCase]:
    root = Path(repo_root).expanduser().resolve()
    previous_sys_path = list(sys.path)
    rsl_root = str(root / "training" / "rsl_rl")
    try:
        sys.path.insert(0, rsl_root)
        importlib.invalidate_caches()
        return [
            _compile_python(root),
            _check_complete_legged_gym_tree(root),
            _check_legged_gym_static_boundary(root),
            _import_cpu_packages(root),
            _run_rsl_cpu_smoke(),
            probe_isaac_gym(),
        ]
    finally:
        sys.path[:] = previous_sys_path


def write_report(path: Path, cases: Iterable[GateCase]) -> None:
    report_path = Path(path).expanduser()
    case_rows = [asdict(case) for case in cases]
    statuses = {row["status"] for row in case_rows}
    status = "failed" if "failed" in statuses else "passed_with_blockers" if "blocked" in statuses else "passed"
    payload = {"cases": case_rows, "schema_version": 1, "status": status}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, help="SEA-Nav checkout root (auto-discovered when omitted)")
    parser.add_argument("--report", required=True, type=Path, help="caller-selected JSON report path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = discover_repo_root(args.repo_root)
    cases = run_cpu_gate(repo_root)
    write_report(args.report, cases)
    statuses = {case.status for case in cases}
    status = "failed" if "failed" in statuses else "passed_with_blockers" if "blocked" in statuses else "passed"
    print(json.dumps({"report": str(args.report), "status": status}))
    return 1 if any(case.status == "failed" for case in cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
