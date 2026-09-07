from __future__ import annotations

import dataclasses
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sea_nav_current_isaaclab_full_method.adapters.manifest import default_manifest
from tools.gate_a import GateCase, discover_repo_root, probe_isaac_gym, run_cpu_gate, write_report


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SHELL = ROOT / "sea_nav_current_isaaclab_full_method" / "run_full_method_runtime_smoke.sh"


def test_gate_writes_only_to_the_requested_path(tmp_path: Path) -> None:
    checkout_reports_before = set(ROOT.rglob("gate-a*.json"))
    original_sys_path = list(__import__("sys").path)

    cases = run_cpu_gate(ROOT)

    assert list(__import__("sys").path) == original_sys_path
    assert all(case.status != "failed" for case in cases)
    assert {case.name for case in cases} == {
        "python_syntax",
        "legged_gym_complete_tree",
        "legged_gym_static_boundary",
        "rsl_rl_cpu_imports",
        "rsl_rl_cpu_smoke",
        "isaac_gym_runtime",
        "isaac_gym_dependency",
        "isaaclab_dependency",
        "isaaclab_runtime",
    }
    report = tmp_path / "requested" / "gate-a.json"
    write_report(report, cases)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] in {"passed", "passed_with_blockers"}
    assert payload["cases"]
    assert {row["status"] for row in payload["cases"]} <= {"passed", "blocked", "failed"}
    assert set(ROOT.rglob("gate-a*.json")) == checkout_reports_before


def test_sparse_tree_is_rejected_as_incomplete(tmp_path: Path) -> None:
    (tmp_path / "training" / "legged_gym" / "legged_gym").mkdir(parents=True)

    cases = {case.name: case for case in run_cpu_gate(tmp_path)}

    assert cases["legged_gym_complete_tree"].status == "failed"
    assert "go2_description" in cases["legged_gym_complete_tree"].detail


def test_complete_tree_rejects_missing_baseline_xacro(tmp_path: Path) -> None:
    source = ROOT / "training" / "legged_gym"
    target = tmp_path / "training" / "legged_gym"
    shutil.copytree(source, target)
    missing = target / "resources" / "go2_description" / "xacro" / "robot.xacro"
    missing.unlink()

    cases = {case.name: case for case in run_cpu_gate(tmp_path)}

    assert cases["legged_gym_complete_tree"].status == "failed"
    assert "training/legged_gym/resources/go2_description/xacro/robot.xacro" in cases[
        "legged_gym_complete_tree"
    ].detail


@pytest.mark.parametrize("substitution", ["leaf", "directory"])
def test_complete_tree_rejects_external_symlink_substitution(tmp_path: Path, substitution: str) -> None:
    source = ROOT / "training" / "legged_gym"
    target = tmp_path / "candidate" / "training" / "legged_gym"
    shutil.copytree(source, target)
    xacro_dir = target / "resources" / "go2_description" / "xacro"

    if substitution == "leaf":
        external = tmp_path / "external-robot.xacro"
        external.write_text("<robot name='external'/>\n", encoding="utf-8")
        substituted = xacro_dir / "robot.xacro"
        substituted.unlink()
        substituted.symlink_to(external)
        expected_path = "training/legged_gym/resources/go2_description/xacro/robot.xacro"
    else:
        external = tmp_path / "external-xacro"
        shutil.copytree(xacro_dir, external)
        shutil.rmtree(xacro_dir)
        xacro_dir.symlink_to(external, target_is_directory=True)
        expected_path = "training/legged_gym/resources/go2_description/xacro"

    cases = {case.name: case for case in run_cpu_gate(tmp_path / "candidate")}

    assert cases["legged_gym_complete_tree"].status == "failed"
    assert "symlink" in cases["legged_gym_complete_tree"].detail
    assert expected_path in cases["legged_gym_complete_tree"].detail


def test_missing_isaac_gym_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str):
        raise ModuleNotFoundError(name)

    case = probe_isaac_gym(import_probe=missing)
    assert (case.name, case.status) == ("isaac_gym_dependency", "blocked")
    assert "isaacgym" in case.detail

@pytest.mark.parametrize('package,prefix',[('isaacgym','isaac_gym'),('isaaclab','isaaclab')])
@pytest.mark.parametrize('outcome',['available','missing','broken'])
def test_dependency_result_never_implies_runtime(package,prefix,outcome):
    from tools.gate_a import probe_dependency, runtime_not_executed
    def boundary(name):
        assert name==package
        if outcome=='missing': raise ModuleNotFoundError(name)
        if outcome=='broken': raise ImportError('binary dependency failed')
    dependency=probe_dependency(package,prefix,import_probe=boundary)
    assert dependency.status=={'available':'passed','missing':'blocked','broken':'failed'}[outcome]
    runtime=runtime_not_executed(prefix)
    assert runtime.name==prefix+'_runtime' and runtime.status=='blocked'


def test_gate_case_is_immutable() -> None:
    case = GateCase("example", "passed", "ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        case.status = "failed"  # type: ignore[misc]


def test_discover_repo_root_accepts_nested_and_explicit_paths() -> None:
    nested = ROOT / "training" / "rsl_rl" / "rsl_rl" / "runners"
    assert discover_repo_root(nested) == ROOT
    assert discover_repo_root(ROOT) == ROOT


def test_default_manifest_uses_repository_relative_paths() -> None:
    rsl_rl_root = ROOT / "training" / "rsl_rl"
    sys.path.insert(0, str(rsl_rl_root))
    try:
        from rsl_rl.experiment_config import resolve_run_config

        resolved = resolve_run_config(
            registry_path=ROOT / "configs" / "parity_registry.yaml",
            algorithm_profile="upstream_fbce672c",
            runtime_stack="isaaclab_adapter",
            implementation_delta=("ppo_state_identity_repair",),
        )
    finally:
        sys.path.remove(str(rsl_rl_root))
    manifest = default_manifest(
        repo_root=ROOT,
        adapter_root=ROOT / "sea_nav_current_isaaclab_full_method",
        resolved_config=resolved,
        validation_rung="unverified",
    )
    payload = manifest.to_dict()

    assert payload["source_repo"] == "."
    assert payload["adapter_root"] == "sea_nav_current_isaaclab_full_method"
    for value in [payload["source_repo"], payload["adapter_root"], *payload["upstream_reference_paths"]]:
        assert not Path(value).is_absolute(), value
        assert "/home/" not in value


def test_default_manifest_rejects_legacy_identity_free_call() -> None:
    with pytest.raises(TypeError, match="explicit.*resolved_config.*Task 6"):
        default_manifest(str(ROOT / "sea_nav_current_isaaclab_full_method"))


def test_manifest_module_imports_without_rsl_rl_on_path() -> None:
    adapter_dir = ROOT / "sea_nav_current_isaaclab_full_method" / "adapters"
    env = {"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    code = (
        "import sys; "
        "sys.modules['rsl_rl'] = None; "
        f"sys.path.insert(0, {str(adapter_dir)!r}); "
        "import manifest; "
        "print(manifest.AdapterManifest.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "AdapterManifest"


def test_committed_manifest_does_not_claim_missing_current_evidence() -> None:
    payload = json.loads(
        (ROOT / "sea_nav_current_isaaclab_full_method" / "adapter_manifest.json").read_text(encoding="utf-8")
    )

    assert payload["gate_a"]["status"] == "unverified"
    for section_name in ("runtime_smoke", "checkpoint_init_and_load_smoke"):
        section = payload[section_name]
        assert section["status"] == "blocked"
        assert "reason" in section
        assert "completed_at" not in section
        assert not any(key.endswith("exit_code") for key in section)


def test_runtime_wrapper_rejects_missing_launcher_before_writing(tmp_path: Path) -> None:
    requested_run_dir = tmp_path / "must-not-exist"
    env = dict(os.environ, SEA_NAV_FULL_METHOD_RUN_DIR=str(requested_run_dir))
    env.pop("SEA_NAV_FULL_METHOD_LAUNCHER", None)

    result = subprocess.run(
        ["bash", str(RUNTIME_SHELL)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "launcher" in result.stderr.lower()
    assert not requested_run_dir.exists()


def test_runtime_wrapper_rejects_missing_run_directory() -> None:
    env = dict(os.environ, SEA_NAV_FULL_METHOD_LAUNCHER="/bin/true")
    env.pop("SEA_NAV_FULL_METHOD_RUN_DIR", None)

    result = subprocess.run(
        ["bash", str(RUNTIME_SHELL)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "--run-root" in result.stderr and "--launcher" in result.stderr
