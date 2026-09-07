#!/usr/bin/env python3
"""Operator-only legacy recovery in a required Linux namespace sandbox.

The ordinary runtime never imports this tool. The approved input hash is an
identity decision, not a safety certificate. Failed output is untrusted and
retained for the operator; only conversion-receipt.json signifies independently
validated v2 output. This tool requires an installed CPU Python/Torch runtime.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys

BWRAP = "/usr/bin/bwrap"
TIMEOUT = "/usr/bin/timeout"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _checkpoint_module(path):
    spec = importlib.util.spec_from_file_location("_operator_safe_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _limits(settings):
    names = {"cpu_seconds": resource.RLIMIT_CPU, "memory_bytes": resource.RLIMIT_AS,
             "file_bytes": resource.RLIMIT_FSIZE, "open_files": resource.RLIMIT_NOFILE,
             "processes": resource.RLIMIT_NPROC}
    for key, number in names.items():
        value = settings[key]
        resource.setrlimit(number, (value, value))
        if resource.getrlimit(number) != (value, value):
            raise ValueError("blocked: resource limit readback differs")


def _check_sandbox(settings, validation):
    import socket
    if os.geteuid() == 0 or os.path.exists("/home") or os.path.exists("/root"):
        raise ValueError("blocked: unprivileged credential-isolated sandbox required")
    for name, previous in settings["host_namespaces"].items():
        if os.readlink("/proc/self/ns/" + name) == previous:
            raise ValueError("blocked: namespace isolation missing: " + name)
    if any(name != "lo" for _, name in socket.if_nameindex()):
        raise ValueError("blocked: network interfaces were not isolated")
    allowed_env = {"LC_CTYPE", "PWD", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
    if set(os.environ) - allowed_env:
        raise ValueError("blocked: unexpected inherited environment")
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("CapEff:") and int(line.split()[1], 16) != 0:
            raise ValueError("blocked: capabilities were not dropped")
    if not validation and not (os.statvfs("/input/pinned").f_flag & os.ST_RDONLY):
        raise ValueError("blocked: input mount is not read-only")
    if validation and not (os.statvfs("/output").f_flag & os.ST_RDONLY):
        raise ValueError("blocked: validator output mount is not read-only")
    _limits(settings)


def _sandbox_stage(settings, validation=False):
    """Internal entry: the same mandatory isolation checks precede either load."""
    _check_sandbox(settings, validation)
    cp = _checkpoint_module("/safe_checkpoint.py")
    import torch
    torch.set_num_threads(1)
    cp.require_weights_only()
    manifest_path = Path("/output/checkpoint.manifest.json")
    if not validation:
        with open("/input/pinned", "rb") as handle:
            if cp._hash_file(handle) != settings["approved_sha256"]:
                raise ValueError("pinned input SHA-256 changed")
            # The sole unsafe load in this checkout. OS confinement was checked
            # above; no public runtime/training API can select this path.
            legacy = torch.load(handle, map_location="cpu", weights_only=False)
        if not isinstance(legacy, dict) or "model_state_dict" not in legacy:
            raise ValueError("legacy payload requires explicit model_state_dict")
        allowed = {"model_state_dict", "optimizer_state_dict", "iter", "iteration", "infos"}
        if set(legacy) - allowed or ("iter" in legacy and "iteration" in legacy):
            raise ValueError("unsupported legacy payload fields")
        iteration = legacy.get("iteration", legacy.get("iter", 0))
        cp.save_checkpoint_v2(manifest_path, model_state_dict=legacy["model_state_dict"],
            optimizer_state_dict=legacy.get("optimizer_state_dict"), iteration=iteration,
            producer_commit=settings["producer_commit"], resolved_config_sha256=settings["resolved_config_sha256"])
        with open("/output/conversion-input.json", "x") as handle:
            json.dump({"input_sha256": settings["approved_sha256"]}, handle)
        return None
    # A separate process never executes the unsafe branch or trusts worker stdout.
    loaded = cp.load_checkpoint_v2(manifest_path, artifact_root=Path("/output"),
        map_location="cpu", expected_resolved_config_sha256=settings["resolved_config_sha256"])
    with cp._directory(Path("/output")) as fd:
        with cp._open_file(fd, "conversion-input.json") as handle:
            if os.fstat(handle.fileno()).st_size > 256:
                raise ValueError("invalid conversion input receipt")
            conversion_input = json.loads(handle.read(257), object_pairs_hook=cp._unique_json)
        expected = {"checkpoint.manifest.json", loaded.manifest.artifact_path, "conversion-input.json"}
        if set(os.listdir(fd)) != expected:
            raise ValueError("unexpected sandbox output files")
    if conversion_input != {"input_sha256": settings["approved_sha256"]}:
        raise ValueError("conversion input receipt hash mismatch")
    if loaded.manifest.producer_commit != settings["producer_commit"]:
        raise ValueError("conversion producer commit mismatch")
    return {"schema": "sea_nav_legacy_conversion_receipt_v1", "input_sha256": settings["approved_sha256"],
            "output_manifest_sha256": loaded.manifest_sha256, "output_payload_sha256": loaded.manifest.sha256,
            "iteration": loaded.iteration, "producer_commit": settings["producer_commit"],
            "resolved_config_sha256": settings["resolved_config_sha256"],
            "isolation": "bubblewrap-unshare-all", "independent_safe_validation": True,
            "validation_torch": torch.__version__,
            "resource_limits": {key: settings[key] for key in
                ("cpu_seconds", "memory_bytes", "file_bytes", "open_files", "processes", "wall_seconds")}}


def _sandbox_command(settings, descriptors, *, validation):
    command = [TIMEOUT, "--signal=TERM", "--kill-after=2s", str(settings["wall_seconds"]) + "s",
        BWRAP, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--cap-drop", "ALL",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/runtime",
        "--ro-bind-fd", str(descriptors["runtime"]), "/runtime/python",
        "--ro-bind-fd", str(descriptors["script"]), "/converter.py",
        "--ro-bind-fd", str(descriptors["checkpoint"]), "/safe_checkpoint.py",
        "--ro-bind-fd" if validation else "--bind-fd", str(descriptors["output"]), "/output"]
    if not validation:
        # bwrap 0.6 cannot resolve memfd's anonymous source pathname for
        # --ro-bind-fd. --ro-bind-data copies the already sealed approved FD
        # into a private read-only bind mount, never reopening the host path.
        command += ["--dir", "/input", "--ro-bind-data", str(descriptors["input"]), "/input/pinned"]
    for key, value in (("PYTHONDONTWRITEBYTECODE", "1"), ("OMP_NUM_THREADS", "1"),
                       ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
        command += ["--setenv", key, value]
    command += ["--chdir", "/tmp", "/runtime/python/bin/python", "-I", "-B", "/converter.py",
                "--_validate" if validation else "--_sandbox-worker", json.dumps(settings, sort_keys=True)]
    return command


def convert_legacy_checkpoint_once(input_path, *, output_root, approved_sha256, producer_commit,
                                   resolved_config_sha256, memory_bytes=4 * 1024**3,
                                   cpu_seconds=20, wall_seconds=30, file_bytes=64 * 1024**2):
    """Operator API only; requires a fresh empty directory and mandatory sandbox."""
    cp = _checkpoint_module(REPO_ROOT / "training/rsl_rl/rsl_rl/utils/checkpoint.py")
    cp.require_weights_only()
    cp._identity(producer_commit, resolved_config_sha256)
    cp._digest(approved_sha256, "approved input SHA-256")
    for value in (memory_bytes, cpu_seconds, wall_seconds, file_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("resource limits must be positive integers")
    if os.geteuid() == 0 or not all(Path(tool).is_file() and os.access(tool, os.X_OK) for tool in (BWRAP, TIMEOUT)):
        raise ValueError("blocked: nonroot Linux bubblewrap/timeout capability required")
    runtime = Path(sys.prefix)
    if runtime == Path(sys.base_prefix) or not (runtime / "bin/python").exists():
        raise ValueError("blocked: dedicated CPU Python runtime required")
    settings = dict(approved_sha256=approved_sha256, producer_commit=producer_commit,
        resolved_config_sha256=resolved_config_sha256, memory_bytes=memory_bytes, cpu_seconds=cpu_seconds,
        wall_seconds=wall_seconds, file_bytes=file_bytes, open_files=64, processes=64,
        host_namespaces={name: os.readlink("/proc/self/ns/" + name) for name in ("user", "mnt", "net", "pid", "ipc", "uts")})
    try:
        with ExitStack() as stack:
            input_path = cp._absolute(input_path)
            parent_fd = stack.enter_context(cp._directory(input_path.parent))
            input_file = stack.enter_context(cp._open_file(parent_fd, input_path.name))
            size = os.fstat(input_file.fileno()).st_size
            if size > file_bytes:
                raise ValueError("approved input exceeds converter file-size budget")
            pinned_input = stack.enter_context(cp._sealed_snapshot(input_file, size))
            if cp._hash_file(pinned_input) != approved_sha256:
                raise ValueError("approved input SHA-256 mismatch")
            output_fd = stack.enter_context(cp._directory(output_root))
            if os.listdir(output_fd):
                raise ValueError("converter output_root must be empty")
            runtime_fd = stack.enter_context(cp._directory(runtime))
            source_fd = stack.enter_context(cp._directory(Path(__file__).absolute().parent))
            script = stack.enter_context(cp._open_file(source_fd, Path(__file__).name))
            safe_parent = stack.enter_context(cp._directory(REPO_ROOT / "training/rsl_rl/rsl_rl/utils"))
            safe_module = stack.enter_context(cp._open_file(safe_parent, "checkpoint.py"))
            descriptors = dict(input=pinned_input.fileno(), output=output_fd, runtime=runtime_fd,
                               script=script.fileno(), checkpoint=safe_module.fileno())
            validated = None
            for validation in (False, True):
                command = _sandbox_command(settings, descriptors, validation=validation)
                inherited = tuple(fd for name, fd in descriptors.items() if name != "input" or not validation)
                result = subprocess.run(command, pass_fds=inherited,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE if validation else subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, text=True, timeout=wall_seconds + 5,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
                if result.returncode != 0:
                    # Untrusted subprocess output is not replayed into logs.
                    raise ValueError("blocked or failed: isolated %s exited %d; output remains untrusted" %
                                     ("safe validator" if validation else "converter", result.returncode))
                if validation:
                    validated = json.loads(result.stdout)
            if not isinstance(validated, dict) or validated.get("input_sha256") != approved_sha256 or validated.get("independent_safe_validation") is not True:
                raise ValueError("independent validation receipt rejected")
            fd = os.open("conversion-receipt.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=output_fd)
            with os.fdopen(fd, "w") as handle:
                json.dump(validated, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(output_fd)
            return validated
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ValueError("blocked or failed: sandbox conversion did not validate") from exc


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] in ("--_sandbox-worker", "--_validate"):
        result = _sandbox_stage(json.loads(argv[1]), validation=argv[0] == "--_validate")
        if result is not None:
            print(json.dumps(result, sort_keys=True))
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--approved-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(convert_legacy_checkpoint_once(args.input, output_root=args.output_root,
        approved_sha256=args.approved_sha256, producer_commit=args.producer_commit,
        resolved_config_sha256=args.resolved_config_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
