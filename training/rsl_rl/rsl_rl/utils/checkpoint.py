"""Schema-v2 model/Adam persistence. No legacy pickle loading lives here.

The manifest is the publication commit point. Payload generations are immutable
and retained, so readers of an older manifest keep a usable generation. Loads
pin a no-follow input, then hash and deserialize the same kernel-sealed private
snapshot descriptor, including protection from concurrent in-place writes.
Model/optimizer continuation
does not restore RNG, environment state or physical trajectories.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Optional, Tuple

import torch


class CheckpointError(ValueError):
    """Invalid artifact or unavailable safe checkpoint capability."""


@dataclass(frozen=True)
class CheckpointManifest:
    schema: str
    artifact_path: str
    byte_size: int
    sha256: str
    iteration: int
    producer_commit: str
    resolved_config_sha256: str
    allowed_sections: Tuple[str, ...]


@dataclass(frozen=True)
class LoadedCheckpoint:
    manifest: CheckpointManifest
    model_state_dict: dict
    optimizer_state_dict: Optional[dict]
    iteration: int
    manifest_sha256: str


SCHEMA = "sea_nav_checkpoint_v2"
_FIELDS = set(CheckpointManifest.__dataclass_fields__)
_SECTIONS = {"model_state_dict", "optimizer_state_dict", "iteration"}


def require_weights_only(torch_module=torch):
    try:
        supported = "weights_only" in inspect.signature(torch_module.load).parameters
    except (ValueError, TypeError):
        supported = False
    if not supported:
        raise CheckpointError("safe checkpoint load blocked: torch.load lacks explicit weights_only")


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise CheckpointError(label + " must be a nonnegative integer")


def _digest(value, label, length=64):
    if type(value) is not str or re.fullmatch("[0-9a-f]{%d}" % length, value) is None:
        raise CheckpointError(label + " must be a lowercase hexadecimal digest")


def _identity(producer_commit, config_hash):
    _digest(producer_commit, "producer_commit", 40)
    _digest(config_hash, "resolved_config_sha256")


def _tree(value, label, depth=0):
    if depth > 100:
        raise CheckpointError(label + " exceeds container depth limit")
    if isinstance(value, torch.Tensor) or value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise CheckpointError(label + " dictionaries require string keys")
            _tree(item, label + "." + key, depth + 1)
        return
    if type(value) in (list, tuple):
        for item in value:
            _tree(item, label, depth + 1)
        return
    raise CheckpointError(label + " contains unsupported value")


def _adam(value):
    if not isinstance(value, dict) or set(value) != {"state", "param_groups"}:
        raise CheckpointError("optimizer requires exactly state and param_groups")
    if not isinstance(value["state"], dict) or type(value["param_groups"]) is not list:
        raise CheckpointError("invalid optimizer containers")
    ids = set()
    for group in value["param_groups"]:
        if not isinstance(group, dict) or "params" not in group or type(group["params"]) is not list:
            raise CheckpointError("optimizer param_groups require params lists")
        _tree(group, "optimizer.param_groups")
        lr = group.get("lr")
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr < 0:
            raise CheckpointError("optimizer learning rate must be finite and nonnegative")
        for parameter_id in group["params"]:
            _integer(parameter_id, "optimizer parameter ID")
            if parameter_id in ids:
                raise CheckpointError("duplicate optimizer parameter ID")
            ids.add(parameter_id)
    if not value["param_groups"]:
        raise CheckpointError("optimizer requires parameter groups")
    for key, state_value in value["state"].items():
        _integer(key, "optimizer state key")
        if key not in ids:
            raise CheckpointError("optimizer state key absent from parameter groups")
        if not isinstance(state_value, dict):
            raise CheckpointError("optimizer state must be an Adam dictionary")
        required = {"step", "exp_avg", "exp_avg_sq"}
        if not required <= set(state_value) or set(state_value) - required - {"max_exp_avg_sq"}:
            raise CheckpointError("optimizer state is not supported Adam state")
        _tree(state_value, "optimizer.state")
        avg, square = state_value["exp_avg"], state_value["exp_avg_sq"]
        if not isinstance(avg, torch.Tensor) or not isinstance(square, torch.Tensor) or avg.shape != square.shape:
            raise CheckpointError("Adam moments require equal-shaped tensors")
        maximum = state_value.get("max_exp_avg_sq", square)
        if not isinstance(maximum, torch.Tensor) or maximum.shape != avg.shape:
            raise CheckpointError("invalid Adam maximum moment")
        step = state_value["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise CheckpointError("invalid Adam step")
            step = step.item()
        if type(step) not in (int, float) or not math.isfinite(step) or step < 0:
            raise CheckpointError("invalid Adam step")


def _payload(value):
    if not isinstance(value, dict) or not {"model_state_dict", "iteration"} <= set(value) or set(value) - _SECTIONS:
        raise CheckpointError("payload requires model_state_dict, iteration and optional optimizer_state_dict only")
    _integer(value["iteration"], "iteration")
    if not isinstance(value["model_state_dict"], dict):
        raise CheckpointError("model_state_dict must be a dictionary")
    _tree(value["model_state_dict"], "model_state_dict")
    if "optimizer_state_dict" in value:
        _adam(value["optimizer_state_dict"])


def _basename(value):
    if type(value) is not str or not value or value in (".", "..") or "/" in value or "\\" in value or "\x00" in value:
        raise CheckpointError("artifact_path must be a canonical adjacent basename")


def _manifest(data):
    if type(data) is not dict or set(data) != _FIELDS or data.get("schema") != SCHEMA:
        raise CheckpointError("invalid checkpoint manifest schema/fields; legacy checkpoints are rejected")
    _basename(data["artifact_path"])
    _integer(data["byte_size"], "byte_size")
    _integer(data["iteration"], "iteration")
    _digest(data["sha256"], "sha256")
    _identity(data["producer_commit"], data["resolved_config_sha256"])
    sections = data["allowed_sections"]
    if type(sections) not in (list, tuple) or any(type(x) is not str for x in sections):
        raise CheckpointError("invalid allowed_sections")
    if len(set(sections)) != len(sections) or set(sections) not in ({"model_state_dict", "iteration"}, _SECTIONS):
        raise CheckpointError("invalid allowed_sections")
    return CheckpointManifest(**dict(data, allowed_sections=tuple(sections)))


def _platform():
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")) or os.open not in os.supports_dir_fd or os.rename not in os.supports_dir_fd:
        raise CheckpointError("checkpoint blocked: no-follow directory descriptors unavailable")


def _absolute(path):
    path = Path(path)
    if ".." in path.parts:
        raise CheckpointError("checkpoint path must not contain parent traversal")
    return Path(os.path.abspath(str(path)))


@contextmanager
def _directory(path):
    """Pin every component, starting at filesystem root; never follow links."""
    _platform()
    path = _absolute(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _manifest_directory(path, artifact_root):
    path, root = _absolute(path), _absolute(artifact_root)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CheckpointError("manifest escapes artifact_root") from exc
    if not relative.parts:
        raise CheckpointError("manifest must be a regular file")
    with _directory(root) as root_fd:
        fd = os.dup(root_fd)
        try:
            for component in relative.parts[:-1]:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd, relative.name
        finally:
            os.close(fd)


def _open_file(fd, name):
    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=fd)
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise CheckpointError("checkpoint requires regular files")
        return os.fdopen(file_fd, "rb")
    except BaseException:
        os.close(file_fd)
        raise


@contextmanager
def _sealed_snapshot(source, byte_size):
    """Stable bytes even when another writer changes the pinned source inode."""
    import fcntl
    if not all(hasattr(os, name) for name in ("memfd_create", "MFD_ALLOW_SEALING", "MFD_CLOEXEC")) or not all(
            hasattr(fcntl, name) for name in ("F_ADD_SEALS", "F_GET_SEALS", "F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")):
        raise CheckpointError("safe checkpoint load blocked: kernel-sealed snapshots unavailable")
    snapshot_fd = os.memfd_create("sea-nav-checkpoint", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        snapshot = os.fdopen(snapshot_fd, "w+b")
    except BaseException:
        os.close(snapshot_fd)
        raise
    with snapshot:
        remaining = byte_size
        while remaining:
            block = source.read(min(remaining, 1024 * 1024))
            if not block:
                raise CheckpointError("checkpoint changed size while snapshotting")
            snapshot.write(block)
            remaining -= len(block)
        if source.read(1):
            raise CheckpointError("checkpoint changed size while snapshotting")
        snapshot.flush()
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        fcntl.fcntl(snapshot.fileno(), fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(snapshot.fileno(), fcntl.F_GET_SEALS) & seals != seals:
            raise CheckpointError("safe checkpoint snapshot seal verification failed")
        snapshot.seek(0)
        yield snapshot


def _hash_file(file):
    file.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(block)
    file.seek(0)
    return digest.hexdigest()


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError("duplicate manifest key")
        result[key] = value
    return result


def load_checkpoint_v2(manifest_path, *, artifact_root, map_location="cpu", expected_resolved_config_sha256=None):
    require_weights_only()
    try:
        with _manifest_directory(manifest_path, artifact_root) as (fd, name):
            with _open_file(fd, name) as file:
                if os.fstat(file.fileno()).st_size > 65536:
                    raise CheckpointError("manifest exceeds size limit")
                manifest_bytes = file.read(65537)
                manifest = _manifest(json.loads(manifest_bytes, object_pairs_hook=_unique_json))
            if expected_resolved_config_sha256 is not None and manifest.resolved_config_sha256 != expected_resolved_config_sha256:
                raise CheckpointError("checkpoint resolved config hash mismatch")
            with _open_file(fd, manifest.artifact_path) as file:
                if os.fstat(file.fileno()).st_size != manifest.byte_size:
                    raise CheckpointError("checkpoint byte size mismatch")
                with _sealed_snapshot(file, manifest.byte_size) as snapshot:
                    if os.fstat(snapshot.fileno()).st_size != manifest.byte_size:
                        raise CheckpointError("checkpoint snapshot byte size mismatch")
                    if _hash_file(snapshot) != manifest.sha256:
                        raise CheckpointError("checkpoint SHA-256 mismatch")
                    value = torch.load(snapshot, map_location=map_location, weights_only=True)
            _payload(value)
            if value["iteration"] != manifest.iteration or set(value) != set(manifest.allowed_sections):
                raise CheckpointError("manifest/payload iteration or allowed_sections mismatch")
            return LoadedCheckpoint(manifest, value["model_state_dict"], value.get("optimizer_state_dict"), value["iteration"],
                                    hashlib.sha256(manifest_bytes).hexdigest())
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError("safe checkpoint load failed: " + str(exc)) from exc


def validate_checkpoint_mode(loaded, mode):
    if mode == "warm_start":
        if loaded.optimizer_state_dict is not None or loaded.iteration != 0:
            raise CheckpointError("warm start requires model-only checkpoint at iteration zero")
    elif mode == "resume":
        if loaded.optimizer_state_dict is None:
            raise CheckpointError("resume requires optimizer state")
    elif mode != "inference":
        raise CheckpointError("unknown checkpoint mode: " + str(mode))
    return loaded


def _publication_boundary(name):
    """Deterministic fault-injection seam; production has no callback or hooks."""


def _temp_file(fd):
    name = ".checkpoint-" + secrets.token_hex(16) + ".tmp"
    file_fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
    return name, os.fdopen(file_fd, "w+b")


def save_checkpoint_v2(manifest_path, *, model_state_dict, iteration, producer_commit,
                       resolved_config_sha256, optimizer_state_dict=None):
    _identity(producer_commit, resolved_config_sha256)
    value = {"model_state_dict": model_state_dict, "iteration": iteration}
    if optimizer_state_dict is not None:
        value["optimizer_state_dict"] = optimizer_state_dict
    _payload(value)
    path = _absolute(manifest_path)
    _basename(path.name)
    try:
        with _directory(path.parent) as fd:
            pending = []
            try:
                name, file = _temp_file(fd)
                pending.append(name)
                with file:
                    torch.save(value, file)
                    file.flush()
                    os.fsync(file.fileno())
                    size = os.fstat(file.fileno()).st_size
                    digest = _hash_file(file)
                _publication_boundary("payload_synced")
                artifact_name = path.stem + "." + digest + ".pt"
                # Hard-link publication is no-clobber, unlike rename/replace.
                try:
                    os.link(name, artifact_name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
                except FileExistsError:
                    with _open_file(fd, artifact_name) as existing:
                        if os.fstat(existing.fileno()).st_size != size or _hash_file(existing) != digest:
                            raise CheckpointError("existing immutable payload generation differs")
                os.fsync(fd)
                _publication_boundary("payload_published")
                manifest = _manifest(dict(schema=SCHEMA, artifact_path=artifact_name, byte_size=size,
                                          sha256=digest, iteration=iteration, producer_commit=producer_commit,
                                          resolved_config_sha256=resolved_config_sha256,
                                          allowed_sections=sorted(value)))
                manifest_name, file = _temp_file(fd)
                pending.append(manifest_name)
                with file:
                    file.write((json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":")) + "\n").encode())
                    file.flush()
                    os.fsync(file.fileno())
                _publication_boundary("manifest_synced")
                os.replace(manifest_name, path.name, src_dir_fd=fd, dst_dir_fd=fd)
                pending.remove(manifest_name)
                os.fsync(fd)
                _publication_boundary("manifest_published")
                return manifest
            finally:
                for name in pending:
                    os.unlink(name, dir_fd=fd)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError("checkpoint publication failed: " + str(exc)) from exc
