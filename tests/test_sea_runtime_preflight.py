# SPDX-License-Identifier: MIT
"""CPU/static tests for the evidence-driven SEA runtime preflight (G3).

No Isaac, GPU, or Torch pass is ever mocked here.  The genuine CLI run in
these tests executes on this CPU host and honestly reports its blocked and
failed checks.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "sea_runtime_preflight.py"

spec = importlib.util.spec_from_file_location("sea_runtime_preflight", TOOL)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


# ---------------------------------------------------------------------------
# Version matching.
# ---------------------------------------------------------------------------

def test_version_matches_exact_and_major_minor_pattern():
    assert preflight.version_matches("2.5.1", "2.5.1")
    assert preflight.version_matches("2.5.1+cu121", "2.5.1")
    assert not preflight.version_matches("2.6.0", "2.5.1")
    assert preflight.version_matches("3.10.12", "3.10.x")
    assert preflight.version_matches("3.10.0", "3.10.x")
    assert not preflight.version_matches("3.11.0", "3.10.x")
    assert not preflight.version_matches(None, "2.5.1")


# ---------------------------------------------------------------------------
# Report assembly semantics.
# ---------------------------------------------------------------------------

def _result(name, status):
    return preflight.ProbeResult(name=name, status=status, observed=None,
                                 expected=None, detail="")


def test_all_passed_means_launch_ready_but_never_runtime_verified():
    report = preflight.assemble_report([_result("a", "passed"), _result("b", "passed")])
    assert report["launch_ready"] is True
    assert report["overall"] == "launch_ready"
    assert report["runtime_verified"] is False
    assert "construct/reset/step/close" in report["runtime_verified_note"]
    assert report["blocked_reasons"] == []


def test_blocked_is_not_a_pass_and_lists_reasons():
    report = preflight.assemble_report(
        [_result("isaac_sim", "blocked"), _result("cuda", "blocked")]
    )
    assert report["launch_ready"] is False
    assert report["overall"] == "blocked"
    assert len(report["blocked_reasons"]) == 2
    assert report["runtime_verified"] is False


def test_identity_failure_dominates_blocked():
    report = preflight.assemble_report(
        [_result("torch", "failed"), _result("isaac_sim", "blocked")]
    )
    assert report["overall"] == "failed"
    assert report["launch_ready"] is False
    assert report["failed_checks"] == ["torch"]


def test_go2_artifacts_are_never_readiness_requirements():
    assert "go2.usd" in preflight.FORBIDDEN_READINESS_REQUIREMENTS
    report = preflight.assemble_report([_result("a", "passed")])
    assert "go2.usd" in report["forbidden_readiness_requirements"]


# ---------------------------------------------------------------------------
# RSL-RL identity (injected module metadata, no real import needed).
# ---------------------------------------------------------------------------

class _FakeModule:
    def __init__(self, file_path):
        self.__file__ = file_path


def _patch_rsl_rl(monkeypatch, file_path, version):
    real_find_spec = importlib.util.find_spec
    real_import_module = preflight.importlib.import_module

    def fake_find_spec(name):
        if name == "rsl_rl":
            return object()
        return real_find_spec(name)

    def fake_import_module(name):
        if name == "rsl_rl":
            return _FakeModule(str(file_path))
        return real_import_module(name)

    monkeypatch.setattr(preflight.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(preflight.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(preflight.importlib.metadata, "version",
                        lambda name: version)


def test_rsl_rl_identity_accepts_checkout_local(monkeypatch, tmp_path):
    local_root = tmp_path / "training" / "rsl_rl"
    local = local_root / "rsl_rl" / "__init__.py"
    local.parent.mkdir(parents=True)
    local.write_text("", encoding="utf-8")
    (local_root / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='rsl_rl', version='1.0.2')\n",
        encoding="utf-8",
    )
    _patch_rsl_rl(monkeypatch, local, "9.9.9")  # global metadata must be ignored
    result = preflight.probe_rsl_rl_identity(str(tmp_path))
    assert result.status == "passed"
    assert result.observed["version"] == "1.0.2"


def test_rsl_rl_identity_rejects_global_install(monkeypatch, tmp_path):
    elsewhere = tmp_path / "site-packages" / "rsl_rl" / "__init__.py"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("", encoding="utf-8")
    _patch_rsl_rl(monkeypatch, elsewhere, "3.0.1")
    result = preflight.probe_rsl_rl_identity(str(tmp_path))
    assert result.status == "failed"
    assert "repository-local" in result.detail


def test_rsl_rl_identity_rejects_local_but_wrong_version(monkeypatch, tmp_path):
    local = tmp_path / "training" / "rsl_rl" / "rsl_rl" / "__init__.py"
    local.parent.mkdir(parents=True)
    local.write_text("", encoding="utf-8")
    _patch_rsl_rl(monkeypatch, local, "3.0.1")
    result = preflight.probe_rsl_rl_identity(str(tmp_path))
    assert result.status == "failed"
    assert "1.0.2" in result.detail


def test_rsl_rl_missing_module_is_blocked_not_failed(monkeypatch, tmp_path):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(preflight.importlib.util, "find_spec",
                        lambda name: None if name == "rsl_rl" else real_find_spec(name))
    result = preflight.probe_rsl_rl_identity(str(tmp_path))
    assert result.status == "blocked"


# ---------------------------------------------------------------------------
# File hash and asset manifest probes.
# ---------------------------------------------------------------------------

def test_file_hash_probe_passes_blocks_and_fails_correctly(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("key: value\n", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    passed = preflight.probe_file_hash("resolved_config", str(target), digest)
    assert passed.status == "passed"

    mismatch = preflight.probe_file_hash("resolved_config", str(target), "0" * 64)
    assert mismatch.status == "failed"

    absent = preflight.probe_file_hash("resolved_config",
                                       str(tmp_path / "absent.yaml"), "0" * 64)
    assert absent.status == "blocked"
    assert "does not exist" in absent.detail

    unsupplied = preflight.probe_file_hash("fixture_manifest", None, None)
    assert unsupplied.status == "blocked"


def test_asset_manifest_validates_schema_and_required_files(tmp_path):
    asset = tmp_path / "room_mesh.obj"
    asset.write_text("v 0 0 0\n", encoding="utf-8")
    manifest = tmp_path / "asset_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "assets": {"room_mesh": str(asset)},
    }), encoding="utf-8")
    ok = preflight.probe_asset_manifest(str(manifest), ["room_mesh"])
    assert ok.status == "passed"

    missing = preflight.probe_asset_manifest(str(manifest), ["robot_usd"])
    assert missing.status == "failed"
    assert "robot_usd" in missing.detail

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    invalid = preflight.probe_asset_manifest(str(bad), [])
    assert invalid.status == "failed"

    none = preflight.probe_asset_manifest(None, [])
    assert none.status == "blocked"
    assert "G4" in none.detail


def test_asset_manifest_pins_its_own_hash(tmp_path):
    manifest = tmp_path / "asset_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "assets": {},
    }), encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    ok = preflight.probe_asset_manifest(str(manifest), [],
                                        manifest_sha256=digest)
    assert ok.status == "passed"
    tampered = preflight.probe_asset_manifest(str(manifest), [],
                                              manifest_sha256="0" * 64)
    assert tampered.status == "failed"
    assert "hash mismatch" in tampered.detail


def test_asset_manifest_relative_entries_resolve_against_manifest_dir(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "room.obj").write_text("v 0 0 0\n", encoding="utf-8")
    manifest = tmp_path / "asset_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "assets": {"room_mesh": "assets/room.obj"},
    }), encoding="utf-8")
    import subprocess as sp
    # Run from an unrelated CWD; relative entries must still resolve.
    code = (
        "import importlib.util, json;"
        "spec = importlib.util.spec_from_file_location('p', r'" + str(TOOL) + "');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "r = m.probe_asset_manifest(r'" + str(manifest) + "', ['room_mesh']);"
        "print(r.status)"
    )
    result = sp.run([sys.executable, "-c", code], capture_output=True, text=True,
                    cwd="/")
    assert result.stdout.strip() == "passed", result.stderr


def test_asset_manifest_verifies_per_asset_hashes(tmp_path):
    asset = tmp_path / "room_mesh.obj"
    asset.write_text("v 0 0 0\n", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    manifest = tmp_path / "asset_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "assets": {"room_mesh": {"path": str(asset), "sha256": digest}},
    }), encoding="utf-8")
    ok = preflight.probe_asset_manifest(str(manifest), ["room_mesh"])
    assert ok.status == "passed"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "assets": {"room_mesh": {"path": str(asset), "sha256": "0" * 64}},
    }), encoding="utf-8")
    mismatch = preflight.probe_asset_manifest(str(manifest), ["room_mesh"])
    assert mismatch.status == "failed"
    assert "asset hash mismatch" in mismatch.detail


# ---------------------------------------------------------------------------
# Real-import rsl_rl identity (subprocess, PYTHONPATH-controlled; no fakes).
# ---------------------------------------------------------------------------

_RSL_PROBE = (
    "import importlib.util, json;"
    "spec = importlib.util.spec_from_file_location('p', r'" + str(TOOL) + "');"
    "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
    "r = m.probe_rsl_rl_identity(r'{root}');"
    "print(r.status); print(r.detail);"
    "print(json.dumps(r.observed, default=str))"
)


def _run_rsl_probe(root, env_pythonpath):
    import os as _os
    env = dict(_os.environ, PYTHONPATH=env_pythonpath,
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-c", _RSL_PROBE.format(root=root)],
        capture_output=True, text=True, env=env, cwd=str(root),
    )


def test_real_spoofed_sibling_rsl_rl_package_fails(tmp_path):
    # A same-named package NEXT TO training/rsl_rl must not pass containment.
    sibling_root = tmp_path / "training" / "rsl_rl2"
    (sibling_root / "rsl_rl").mkdir(parents=True)
    (sibling_root / "rsl_rl" / "__init__.py").write_text("", encoding="utf-8")
    (sibling_root / "setup.py").write_text(
        'from setuptools import setup\nsetup(name="rsl_rl", version="1.0.2")\n',
        encoding="utf-8",
    )
    result = _run_rsl_probe(str(tmp_path), str(sibling_root))
    assert "passed" not in result.stdout.split(), result.stdout + result.stderr
    assert "failed" in result.stdout.split()


def test_real_checkout_local_rsl_rl_package_with_setup_version_passes(tmp_path):
    local_root = tmp_path / "training" / "rsl_rl"
    (local_root / "rsl_rl").mkdir(parents=True)
    (local_root / "rsl_rl" / "__init__.py").write_text("", encoding="utf-8")
    (local_root / "setup.py").write_text(
        'from setuptools import setup\nsetup(name="rsl_rl", version="1.0.2")\n',
        encoding="utf-8",
    )
    result = _run_rsl_probe(str(tmp_path), str(local_root))
    lines = result.stdout.splitlines()
    assert lines and lines[0].strip() == "passed", result.stdout + result.stderr


# ---------------------------------------------------------------------------
# CLI exit codes and json-out validation.
# ---------------------------------------------------------------------------

def test_cli_exit_codes_for_ready_blocked_and_usage(monkeypatch, tmp_path):
    ready_report = {"overall": "launch_ready", "launch_ready": True,
                    "runtime_verified": False, "blocked_reasons": [],
                    "failed_checks": [], "unpinned_checks": [],
                    "runtime_verified_note": "x",
                    "forbidden_readiness_requirements": [],
                    "checks": []}
    blocked_report = dict(ready_report, overall="blocked", launch_ready=False)
    failed_report = dict(ready_report, overall="failed", launch_ready=False,
                         failed_checks=["torch"])

    def fake_report(*args, **kwargs):
        return fake_report.next
    monkeypatch.setattr(preflight, "run_preflight", fake_report)

    fake_report.next = ready_report
    assert preflight.main(["--checkout-root", str(tmp_path)]) == 0
    fake_report.next = blocked_report
    assert preflight.main(["--checkout-root", str(tmp_path)]) == 3
    fake_report.next = failed_report
    assert preflight.main(["--checkout-root", str(tmp_path)]) == 1


def test_cli_usage_error_for_missing_json_out_parent(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TOOL), "--checkout-root", str(REPO_ROOT),
         "--json-out", str(tmp_path / "no_such_dir" / "report.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_stray_ancestor_version_outside_checkout_cannot_spoof(tmp_path):
    # Local package inside training/rsl_rl with NO local packaging marker:
    # version must stay unknown and fail closed, even when an ancestor
    # directory of tmp_path carries version = "1.0.2".
    local_root = tmp_path / "training" / "rsl_rl"
    (local_root / "rsl_rl").mkdir(parents=True)
    (local_root / "rsl_rl" / "__init__.py").write_text("", encoding="utf-8")
    stray = tmp_path / "pyproject.toml"
    stray.write_text('[project]\nname = "unrelated"\nversion = "1.0.2"\n',
                     encoding="utf-8")
    result = _run_rsl_probe(str(tmp_path), str(local_root))
    lines = result.stdout.splitlines()
    assert lines and lines[0].strip() == "failed", result.stdout + result.stderr
    observed = json.loads(lines[2]) if len(lines) > 2 else {}
    assert observed.get("version", "missing") is None, observed


def test_unreadable_file_hash_is_a_failed_probe_not_a_crash(tmp_path):
    import os as _os
    target = tmp_path / "secret.yaml"
    target.write_text("key: value\n", encoding="utf-8")
    target.chmod(0o000)
    try:
        result = preflight.probe_file_hash("resolved_config", str(target), None)
        assert result.status in ("failed",)
        assert "not readable" in result.detail
    finally:
        target.chmod(0o644)


def test_record_mode_commit_and_manifest_appear_in_unpinned_checks(tmp_path):
    commit = preflight.probe_git_commit(str(REPO_ROOT), None)
    assert commit.detail.startswith("unpinned")
    manifest = tmp_path / "asset_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "assets": {}}),
                        encoding="utf-8")
    probed = preflight.probe_asset_manifest(str(manifest), [])
    assert probed.detail.startswith("unpinned")
    report = preflight.assemble_report([commit, probed])
    assert set(report["unpinned_checks"]) == {"sea_commit", "asset_manifest"}


def test_report_lists_unpinned_hash_checks():
    report = preflight.assemble_report([
        preflight.ProbeResult("resolved_config", "passed", "abc", "recorded",
                              "unpinned: hash recorded but not yet verified"),
        _result("b", "passed"),
    ])
    assert report["unpinned_checks"] == ["resolved_config"]
    assert report["launch_ready"] is True


# ---------------------------------------------------------------------------
# Import discipline: the module itself loads no heavy dependencies.
# ---------------------------------------------------------------------------

def test_importing_the_module_does_not_load_torch_isaac_ros_or_rsl_rl():
    forbidden = {"torch", "isaacsim", "isaaclab", "isaacgym", "rclpy", "rsl_rl"}
    code = (
        "import sys, importlib.util; "
        "spec = importlib.util.spec_from_file_location('p', r'" + str(TOOL) + "'); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "loaded = {n.split('.')[0] for n in sys.modules}; "
        "assert not (loaded & " + repr(forbidden) + "), sorted(loaded & " + repr(forbidden) + ")"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Genuine CLI run on this CPU host: honest blocked/failed, correct exit code.
# ---------------------------------------------------------------------------

def test_cli_on_this_cpu_host_reports_blocked_honestly(tmp_path):
    out = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--checkout-root", str(REPO_ROOT),
         "--json-out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode in (1, 3), result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["launch_ready"] is False
    assert report["runtime_verified"] is False
    names = {check["name"]: check for check in report["checks"]}
    # This host genuinely lacks Isaac packages and the GPU runtime.
    assert names["isaac_sim"]["status"] == "blocked"
    assert names["isaac_lab"]["status"] == "blocked"
    assert names["cuda"]["status"] in ("blocked", "failed")
    # Host Torch is 2.6.0+cpu, an honest identity failure against 2.5.1.
    assert names["torch"]["status"] == "failed"
    assert names["sea_commit"]["status"] == "passed"
    if result.returncode == 1:
        assert "torch" in report["failed_checks"]


def test_cli_exit_code_zero_never_occurs_on_this_host(tmp_path):
    # A launch_ready result would require the full target stack; this test
    # pins that the tool does not fabricate one on a CPU-only host.
    result = subprocess.run(
        [sys.executable, str(TOOL), "--checkout-root", str(REPO_ROOT)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
