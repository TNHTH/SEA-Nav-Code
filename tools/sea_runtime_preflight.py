#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evidence-driven runtime preflight for the SEA DashGo IsaacLab target stack.

The tool records only what it actually observes.  It never fabricates a
version, never turns a missing GPU/Isaac into a pass, and never claims
``runtime_verified``: launch readiness is static identity and hash readiness
only.  A real ``construct -> reset -> step -> close`` receipt on the target
stack is the sole permitted source of runtime verification (G12).

Import discipline: importing this module loads neither Torch, Isaac, ROS, nor
the bundled RSL-RL; heavy imports happen lazily inside the probes that need
them so CPU/static consumers stay dependency-free.

Exit codes: 0 = launch_ready, 1 = identity/hash failure, 3 = blocked
(missing external runtime), 2 = usage error.
"""

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Optional, Sequence

TARGET = {
    "python": "3.10.x",
    "torch": "2.5.1",
    "isaac_sim": "4.5.0",
    "isaac_lab": "2.0.2",
    "rsl_rl": "1.0.2",
}

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"

RUNTIME_VERIFIED_NOTE = (
    "launch_ready is static identity and hash readiness only; runtime_verified"
    " requires a real target-stack construct/reset/step/close receipt (G12)"
)

# Go2 locomotion artifacts are explicitly NOT readiness prerequisites.
FORBIDDEN_READINESS_REQUIREMENTS = (
    "locomotion_JIT_actor",
    "locomotion_JIT_encoder",
    "locomotion_JIT_state_estimator",
    "go2.usd",
)


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    name: str
    status: str
    observed: object
    expected: object
    detail: str = ""


def version_matches(observed: Optional[str], expected: str) -> bool:
    """Exact match, or major.minor prefix match when expected ends with .x."""
    if observed is None:
        return False
    observed = str(observed).split("+")[0].strip()
    expected = str(expected).strip()
    if expected.endswith(".x"):
        prefix = expected[:-2]
        return observed.startswith(prefix + ".")
    return observed == expected


def probe_python(expected: str = TARGET["python"]) -> ProbeResult:
    observed = "{}.{}.{}".format(
        sys.version_info[0], sys.version_info[1], sys.version_info[2]
    )
    implementation = platform.python_implementation()
    ok = version_matches(observed, expected) and implementation == "CPython"
    return ProbeResult(
        name="python",
        status=STATUS_PASSED if ok else STATUS_FAILED,
        observed=observed + " (" + implementation + ") at " + sys.executable,
        expected=expected + " (CPython)",
    )


def probe_torch(expected: str = TARGET["torch"]) -> ProbeResult:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return ProbeResult("torch", STATUS_BLOCKED, None, expected,
                           "torch is not importable in this environment")
    try:
        torch = importlib.import_module("torch")  # lazy: only when installed
        observed = torch.__version__
    except Exception as exc:
        return ProbeResult("torch", STATUS_FAILED, None, expected,
                           "torch import failed: " + repr(exc))
    ok = version_matches(observed, expected)
    return ProbeResult(
        "torch", STATUS_PASSED if ok else STATUS_FAILED, observed, expected
    )


def probe_cuda() -> ProbeResult:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return ProbeResult("cuda", STATUS_BLOCKED, None, "available",
                           "torch is not importable in this environment")
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return ProbeResult(
                "cuda", STATUS_BLOCKED, False, True,
                "nvidia_gpu_runtime unavailable; GPU gates stay blocked"
            )
        device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        return ProbeResult("cuda", STATUS_FAILED, None, True,
                           "cuda probe failed: " + repr(exc))
    return ProbeResult("cuda", STATUS_PASSED, True, True, device_name)


def probe_package_version(module_name: str, expected: str,
                          display_name: str) -> ProbeResult:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return ProbeResult(
            display_name, STATUS_BLOCKED, None, expected,
            module_name + " is not installed in this environment"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return ProbeResult(
            display_name, STATUS_FAILED, None, expected,
            module_name + " import failed: " + repr(exc),
        )
    observed = getattr(module, "__version__", None)
    if observed is None:
        try:
            observed = importlib.metadata.version(module_name)
        except Exception:
            observed = None
    ok = version_matches(observed, expected)
    return ProbeResult(
        display_name,
        STATUS_PASSED if ok else STATUS_FAILED,
        observed, expected,
    )


def probe_rsl_rl_identity(checkout_root: Optional[str]) -> ProbeResult:
    """The SEA-bundled modified RSL-RL must live inside this checkout."""
    expected = TARGET["rsl_rl"] + " at <checkout>/training/rsl_rl"
    spec = importlib.util.find_spec("rsl_rl")
    if spec is None:
        return ProbeResult("rsl_rl", STATUS_BLOCKED, None, expected,
                           "rsl_rl is not importable in this environment")
    try:
        rsl_rl = importlib.import_module("rsl_rl")
    except Exception as exc:
        return ProbeResult("rsl_rl", STATUS_FAILED, None, expected,
                           "rsl_rl import failed: " + repr(exc))
    module_file = getattr(rsl_rl, "__file__", None)
    if not module_file:
        return ProbeResult("rsl_rl", STATUS_FAILED, None, expected,
                           "rsl_rl resolved to a namespace package without __file__")
    resolved = str(Path(module_file).resolve())
    inside = False
    if checkout_root:
        allowed = Path(checkout_root).resolve() / "training" / "rsl_rl"
        # Strict prefix containment only; substring matches are spoofable.
        inside = resolved.startswith(str(allowed) + os.sep)
    version = getattr(rsl_rl, "__version__", None)
    if version is None:
        # Version provenance is strictly the fixed packaging root inside this
        # checkout; neither unrelated ancestors outside it nor a same-named
        # global distribution may ever supply or rewrite the version.  An
        # unknown version fails closed below.
        packaging_root = None
        if checkout_root:
            fixed = Path(checkout_root).resolve() / "training" / "rsl_rl"
            if fixed.is_dir() and any(
                (fixed / name).is_file()
                for name in ("PKG-INFO", "setup.py", "pyproject.toml")
            ):
                packaging_root = fixed
        if packaging_root is not None:
            for candidate_name in ("PKG-INFO", "setup.py", "pyproject.toml"):
                candidate = packaging_root / candidate_name
                if not candidate.is_file():
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if candidate.name == "PKG-INFO":
                    match = re.search(r"(?m)^Version:\s*(\S+)", text)
                    if match:
                        version = match.group(1)
                else:
                    match = re.search(
                        r"""version\s*=\s*["\']([^"\']+)["\']""", text)
                    if match:
                        version = match.group(1)
                if version is not None:
                    break
    if not inside:
        return ProbeResult(
            "rsl_rl", STATUS_FAILED,
            {"file": resolved, "version": version}, expected,
            "resolved rsl_rl is not the repository-local training/rsl_rl package",
        )
    if not version_matches(version, TARGET["rsl_rl"]):
        return ProbeResult(
            "rsl_rl", STATUS_FAILED,
            {"file": resolved, "version": version}, expected,
            "rsl_rl version is not the SEA-modified 1.0.2",
        )
    return ProbeResult(
        "rsl_rl", STATUS_PASSED,
        {"file": resolved, "version": version}, expected,
    )


def probe_git_commit(checkout_root: str,
                     expected: Optional[str] = None) -> ProbeResult:
    try:
        observed = subprocess.run(
            ["git", "-C", checkout_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception as exc:
        return ProbeResult("sea_commit", STATUS_FAILED, None, expected,
                           "git rev-parse failed: " + str(exc))
    if expected is not None and observed != expected:
        return ProbeResult("sea_commit", STATUS_FAILED, observed, expected)
    if expected is None:
        return ProbeResult("sea_commit", STATUS_PASSED, observed, "recorded",
                           "unpinned: commit recorded but not pinned")
    return ProbeResult("sea_commit", STATUS_PASSED, observed, expected)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_file_hash(label: str, path: Optional[str],
                    expected_sha256: Optional[str]) -> ProbeResult:
    if path is None:
        return ProbeResult(label, STATUS_BLOCKED, None,
                           expected_sha256 or "required",
                           "no " + label + " path supplied for this run mode")
    if not os.path.isfile(path):
        return ProbeResult(label, STATUS_BLOCKED, None,
                           expected_sha256 or "required",
                           path + " does not exist yet (owned by a later batch)")
    try:
        observed = file_sha256(path)
    except OSError as exc:
        return ProbeResult(label, STATUS_FAILED, None,
                           expected_sha256 or "required",
                           path + " is not readable: " + str(exc))
    if expected_sha256 is None:
        return ProbeResult(label, STATUS_PASSED, observed, "recorded",
                           "unpinned: hash recorded but not yet verified")
    if observed != expected_sha256:
        return ProbeResult(label, STATUS_FAILED, observed, expected_sha256)
    return ProbeResult(label, STATUS_PASSED, observed, expected_sha256)


def probe_asset_manifest(manifest_path: Optional[str],
                         required_files: Sequence[str],
                         manifest_sha256: Optional[str] = None) -> ProbeResult:
    if manifest_path is None:
        return ProbeResult("asset_manifest", STATUS_BLOCKED, None, "required",
                           "no asset manifest supplied (owned by G4)")
    if not os.path.isfile(manifest_path):
        return ProbeResult("asset_manifest", STATUS_BLOCKED, None, "required",
                           manifest_path + " does not exist yet (owned by G4)")
    try:
        observed_hash = file_sha256(manifest_path)
    except OSError as exc:
        return ProbeResult("asset_manifest", STATUS_FAILED, None, "required",
                           manifest_path + " is not readable: " + str(exc))
    if manifest_sha256 is not None and observed_hash != manifest_sha256:
        return ProbeResult("asset_manifest", STATUS_FAILED, observed_hash,
                           manifest_sha256, "asset manifest hash mismatch")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        return ProbeResult("asset_manifest", STATUS_FAILED, None, "valid JSON",
                           "manifest is not valid JSON: " + str(exc))
    if not isinstance(manifest, dict) or "schema_version" not in manifest:
        return ProbeResult("asset_manifest", STATUS_FAILED,
                           sorted(manifest) if isinstance(manifest, dict) else None,
                           "schema_version field present",
                           "manifest schema is invalid")
    base = Path(manifest_path).resolve().parent
    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        return ProbeResult(
            "asset_manifest", STATUS_FAILED,
            {"schema_version": manifest.get("schema_version")},
            {"required_files": list(required_files)},
            "manifest has no assets mapping",
        )
    missing = []
    hash_mismatches = []
    for name in required_files:
        entry = assets.get(name)
        if isinstance(entry, dict):
            path = str(entry.get("path", ""))
            expected = entry.get("sha256")
        else:
            path = str(entry)
            expected = None
        if not path:
            missing.append(name)
            continue
        # Relative entries resolve against the manifest's directory, never
        # the caller's working directory.
        resolved_asset = path if os.path.isabs(path) else str(base / path)
        if not os.path.isfile(resolved_asset):
            missing.append(name)
        elif expected is not None:
            try:
                if file_sha256(resolved_asset) != expected:
                    hash_mismatches.append(name)
            except OSError as exc:
                return ProbeResult(
                    "asset_manifest", STATUS_FAILED, None, "required",
                    resolved_asset + " is not readable: " + str(exc),
                )
    if missing:
        return ProbeResult(
            "asset_manifest", STATUS_FAILED,
            {"schema_version": manifest.get("schema_version"),
             "manifest_sha256": observed_hash},
            {"required_files": list(required_files)},
            "missing required asset files: " + ", ".join(missing),
        )
    if hash_mismatches:
        return ProbeResult(
            "asset_manifest", STATUS_FAILED,
            {"schema_version": manifest.get("schema_version"),
             "manifest_sha256": observed_hash},
            {"required_files": list(required_files)},
            "asset hash mismatch: " + ", ".join(hash_mismatches),
        )
    detail = ""
    if manifest_sha256 is None:
        detail = "unpinned: manifest hash recorded but not yet verified"
    return ProbeResult(
        "asset_manifest", STATUS_PASSED,
        {"schema_version": manifest.get("schema_version"),
         "manifest_sha256": observed_hash},
        {"required_files": list(required_files)},
        detail,
    )


def assemble_report(probes: Sequence[ProbeResult]) -> dict:
    failed = [p for p in probes if p.status == STATUS_FAILED]
    blocked = [p for p in probes if p.status == STATUS_BLOCKED]
    if failed:
        overall = "failed"
    elif blocked:
        overall = "blocked"
    else:
        overall = "launch_ready"
    return {
        "schema_version": 1,
        "overall": overall,
        "launch_ready": overall == "launch_ready",
        "runtime_verified": False,
        "runtime_verified_note": RUNTIME_VERIFIED_NOTE,
        "blocked_reasons": [p.name + ": " + (p.detail or "missing") for p in blocked],
        "failed_checks": [p.name for p in failed],
        "unpinned_checks": [p.name for p in probes
                            if p.status == STATUS_PASSED
                            and str(p.detail).startswith("unpinned")],
        "forbidden_readiness_requirements": list(FORBIDDEN_READINESS_REQUIREMENTS),
        "checks": [dataclasses.asdict(p) for p in probes],
    }


def run_preflight(checkout_root: str, *,
                  expected_commit: Optional[str] = None,
                  config_path: Optional[str] = None,
                  config_sha256: Optional[str] = None,
                  asset_manifest_path: Optional[str] = None,
                  asset_required_files: Sequence[str] = (),
                  asset_manifest_sha256: Optional[str] = None,
                  fixture_manifest_path: Optional[str] = None,
                  fixture_sha256: Optional[str] = None) -> dict:
    probes = [
        probe_python(),
        probe_torch(),
        probe_cuda(),
        probe_package_version("isaacsim", TARGET["isaac_sim"], "isaac_sim"),
        probe_package_version("isaaclab", TARGET["isaac_lab"], "isaac_lab"),
        probe_rsl_rl_identity(checkout_root),
        probe_git_commit(checkout_root, expected_commit),
        probe_file_hash("resolved_config", config_path, config_sha256),
        probe_asset_manifest(asset_manifest_path, asset_required_files,
                             manifest_sha256=asset_manifest_sha256),
        probe_file_hash("fixture_manifest", fixture_manifest_path,
                        fixture_sha256),
    ]
    return assemble_report(probes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evidence-driven SEA IsaacLab runtime preflight"
    )
    parser.add_argument("--checkout-root", required=True)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--config-sha256", default=None)
    parser.add_argument("--asset-manifest", default=None)
    parser.add_argument("--asset-manifest-sha256", default=None)
    parser.add_argument("--asset-file", action="append", default=[],
                        help="required asset file name; repeatable")
    parser.add_argument("--fixture-manifest", default=None)
    parser.add_argument("--fixture-sha256", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    if args.json_out:
        parent = Path(args.json_out).resolve().parent
        if not parent.is_dir():
            parser.error("--json-out parent directory does not exist: "
                         + str(parent))

    report = run_preflight(
        args.checkout_root,
        expected_commit=args.expected_commit,
        config_path=args.config_path,
        config_sha256=args.config_sha256,
        asset_manifest_path=args.asset_manifest,
        asset_required_files=args.asset_file,
        asset_manifest_sha256=args.asset_manifest_sha256,
        fixture_manifest_path=args.fixture_manifest,
        fixture_sha256=args.fixture_sha256,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    if report["overall"] == "launch_ready":
        return 0
    if report["overall"] == "blocked":
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
