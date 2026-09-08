"""Schema-v2 model/Adam persistence. No legacy pickle loading lives here.

The manifest is the publication commit point. Payload generations are immutable
and retained, so readers of an older manifest keep a usable generation. Loads
pin a no-follow input, then hash and deserialize the same kernel-sealed private
snapshot descriptor, including protection from concurrent in-place writes.
Model/optimizer continuation does not restore producer RNG, environment state
or physical trajectories. Applying a checkpoint preserves the caller's RNG.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
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
from torch.optim import optimizer as torch_optimizer


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


def _model_for_target(model, state):
    """Reject before copy_: strict=True alone can partially apply bad state."""
    target = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(target):
        raise CheckpointError("checkpoint model keys must exactly match target")
    buffers = dict(model.named_buffers())
    prepared = {}
    for name, reference in target.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or not isinstance(reference, torch.Tensor):
            raise CheckpointError("checkpoint target supports tensor model state only: " + name)
        if (value.shape != reference.shape or value.dtype != reference.dtype
                or value.layout != reference.layout or value.layout != torch.strided
                or value.is_quantized or reference.is_quantized):
            raise CheckpointError("checkpoint model shape/dtype/layout mismatch: " + name)
        value = value.detach().to(device=reference.device).clone()
        if name in buffers and name.rsplit(".", 1)[-1] == "ray_unit_vectors":
            if not torch.equal(value, reference):
                raise CheckpointError("checkpoint fixed CBF geometry differs from resolved target: " + name)
        prepared[name] = value
    return prepared


def _adam_for_target(optimizer, state):
    """Validate native Adam structure against actual parameters, without a step.

    No optimizer construction, load, gradient computation, or hooks belong in
    this check. In particular eps=0 and beta=0 are legal: finite behavior of a
    future update depends on its real gradients, not invented zero gradients.
    """
    if type(optimizer) is not torch.optim.Adam:
        raise CheckpointError("checkpoint resume supports an actual Adam target only")
    _adam(state)
    if len(state["param_groups"]) != len(optimizer.param_groups):
        raise CheckpointError("Adam target parameter group count mismatch")
    for source, target in zip(state["param_groups"], optimizer.param_groups):
        required_group = {"params", "lr", "betas", "eps", "weight_decay", "amsgrad"}
        if not required_group <= set(source) or set(source) != set(target):
            raise CheckpointError("Adam target requires matching parameter group fields")
        if len(source["params"]) != len(target["params"]):
            raise CheckpointError("Adam target parameter count mismatch")
        for field in ("eps", "weight_decay"):
            value = source.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise CheckpointError("invalid Adam " + field)
        betas = source.get("betas")
        if type(betas) not in (tuple, list) or len(betas) != 2 or any(
                type(beta) not in (int, float) or not math.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
            raise CheckpointError("invalid Adam betas")
        for field in ("amsgrad", "maximize", "capturable", "differentiable"):
            if field in source and type(source[field]) is not bool:
                raise CheckpointError("invalid Adam " + field)
        for field in ("foreach", "fused"):
            if field in source and source[field] is not None and type(source[field]) is not bool:
                raise CheckpointError("invalid Adam " + field)
        fused, foreach = source.get("fused"), source.get("foreach")
        differentiable, capturable = source.get("differentiable"), source.get("capturable")
        if fused and (foreach or differentiable):
            raise CheckpointError("Adam fused is incompatible with foreach/differentiable")
        if foreach and differentiable:
            raise CheckpointError("Adam foreach does not support differentiable")
        for key, parameter in zip(source["params"], target["params"]):
            if (parameter.layout != torch.strided or parameter.is_quantized
                    or not parameter.is_floating_point()):
                raise CheckpointError("Adam requires a dense target parameter")
            if fused:
                # Native metadata-only check, not an optimizer or kernel probe.
                check_fused = getattr(torch_optimizer, "_device_dtype_check_for_fused", None)
                if check_fused is None:
                    raise CheckpointError("Adam fused target validation unavailable")
                try:
                    check_fused(parameter)
                except (RuntimeError, ValueError) as exc:
                    raise CheckpointError("Adam fused target device/dtype unsupported") from exc
            elif capturable:
                devices = getattr(torch_optimizer, "_get_capturable_supported_devices", None)
                supported = devices(supports_xla=not foreach) if devices else ("cuda",)
                if parameter.device.type not in supported:
                    raise CheckpointError("Adam capturable target device unsupported")
            if differentiable and parameter.is_leaf and parameter.requires_grad:
                raise CheckpointError("Adam differentiable cannot update a leaf target parameter")
            moments = state["state"].get(key)
            if moments is not None:
                required = {"step", "exp_avg", "exp_avg_sq"}
                if source["amsgrad"]:
                    required.add("max_exp_avg_sq")
                if set(moments) != required:
                    raise CheckpointError("Adam state fields do not match amsgrad mode")
                step = moments["step"]
                if isinstance(step, torch.Tensor) and (
                        step.layout != torch.strided or step.is_quantized
                        or step.dtype not in (torch.float32, torch.float64) or step.requires_grad
                        or step.device.type == "meta"):
                    raise CheckpointError("Adam step requires a plain floating scalar tensor")
                for field in required - {"step"}:
                    moment = moments[field]
                    if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape
                            or moment.dtype != parameter.dtype or moment.device != parameter.device
                            or moment.layout != parameter.layout or moment.is_quantized):
                        raise CheckpointError("Adam moment shape/dtype/device/layout differs from target")
                    if not torch.isfinite(moment).all() or (field != "exp_avg" and (moment < 0).any()):
                        raise CheckpointError("Adam moments must be finite with nonnegative squared moments")
    return copy.deepcopy(state)


def apply_checkpoint_state(model, model_state_dict, *, optimizer=None, optimizer_state_dict=None):
    """Bounded failure-atomic application, preserving caller RNG and gradients.

    Roll back existing parameters/buffers, gradient presence/values, and supplied
    optimizer state/groups/defaults without re-invoking load hooks. An optimizer
    without optimizer_state_dict is snapshot-only (inference/warm start).
    Arbitrary external hook side effects and replaced module structure are not
    rollback-guaranteed. This does not restore producer RNG/physical trajectories.
    """
    cpu_rng = torch.get_rng_state().clone()
    # Do not initialize a CUDA context merely to load a CPU/model-only artifact.
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        _apply_checkpoint_transaction(model, model_state_dict, optimizer, optimizer_state_dict)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _apply_checkpoint_transaction(model, model_state_dict, optimizer, optimizer_state_dict):
    if optimizer is None and optimizer_state_dict is not None:
        raise CheckpointError("optimizer_state_dict requires a target optimizer")
    live_tensors = list(model.parameters()) + list(model.buffers())
    original_model = [(tensor, tensor.detach().clone()) for tensor in live_tensors]
    parameters = dict.fromkeys(model.parameters())
    if optimizer is not None:
        for group in optimizer.param_groups:
            parameters.update(dict.fromkeys(group["params"]))
        original_state = copy.copy(optimizer.state)
        for parameter, value in original_state.items():
            original_state[parameter] = copy.deepcopy(value)
        original_groups = [{key: list(value) if key == "params" else copy.deepcopy(value)
                            for key, value in group.items()} for group in optimizer.param_groups]
        original_defaults = copy.deepcopy(optimizer.defaults)
    original_grads = [(parameter, parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
                      for parameter in parameters]
    try:
        prepared_model = _model_for_target(model, model_state_dict)
        prepared_optimizer = (None if optimizer_state_dict is None
                              else _adam_for_target(optimizer, optimizer_state_dict))
        model.load_state_dict(prepared_model, strict=True)
        if prepared_optimizer is not None:
            optimizer.load_state_dict(prepared_optimizer)
    except BaseException:
        with torch.no_grad():
            for tensor, original in original_model:
                tensor.copy_(original)
        if optimizer is not None:
            optimizer.state = original_state
            optimizer.param_groups = original_groups
            optimizer.defaults = original_defaults
        raise
    finally:
        # Checkpoints carry no gradients, including on a successful load.
        with torch.no_grad():
            for parameter, grad, original in original_grads:
                parameter.grad = grad
                if grad is not None:
                    grad.copy_(original)


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
