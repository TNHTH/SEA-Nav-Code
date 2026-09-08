import hashlib
import json
import os

import pytest
import torch

from rsl_rl.utils import checkpoint
from test_checkpoint_v2 import save


def test_reduce_rejected_without_side_effect(tmp_path):
    marker = tmp_path / "unsafe-executed"
    class Exploit:
        def __reduce__(self):
            return os.system, ("touch " + str(marker),)
    path = tmp_path / "manifest.json"
    save(path, torch.nn.Linear(1, 1))
    record = json.loads(path.read_text())
    artifact = tmp_path / record["artifact_path"]
    torch.save({"model_state_dict": {"x": Exploit()}, "iteration": 0}, artifact)
    record.update(byte_size=artifact.stat().st_size, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    path.write_text(json.dumps(record))
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)
    assert not marker.exists()


def test_no_unsafe_fallback_without_explicit_weights_only(tmp_path, monkeypatch):
    calls = []
    def old_load(f, **kwargs):
        calls.append(kwargs)
    monkeypatch.setattr(torch, "load", old_load)
    with pytest.raises(checkpoint.CheckpointError, match="weights_only"):
        checkpoint.load_checkpoint_v2(tmp_path / "manifest.json", artifact_root=tmp_path)
    assert calls == []


@pytest.mark.parametrize("target", ["root", "ancestor", "manifest", "artifact"])
def test_symlinks_rejected(tmp_path, target):
    real = tmp_path / "real"
    real.mkdir()
    path = real / "model.json"
    manifest = save(path, torch.nn.Linear(1, 1))
    root = real
    if target in ("root", "ancestor"):
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        if target == "root":
            root = link
            path = link / path.name
        else:
            sub = real / "sub"
            sub.mkdir()
            save(sub / "model.json", torch.nn.Linear(1, 1))
            root = link / "sub"
            path = root / "model.json"
    else:
        victim = path if target == "manifest" else real / manifest.artifact_path
        moved = real / "moved"
        victim.rename(moved)
        victim.symlink_to(moved)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load_checkpoint_v2(path, artifact_root=root)


@pytest.mark.parametrize("name", ["../out.pt", "/out.pt", "sub/out.pt", ".", ".."])
def test_adjacent_artifact_path_required(tmp_path, name):
    path = tmp_path / "model.json"
    save(path, torch.nn.Linear(1, 1))
    data = json.loads(path.read_text())
    data["artifact_path"] = name
    path.write_text(json.dumps(data))
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)


def test_path_swap_after_hash_uses_same_pinned_file(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    model = torch.nn.Linear(1, 1)
    manifest = save(path, model)
    original = checkpoint._hash_file
    replaced = []
    def swap(file):
        result = original(file)
        artifact = tmp_path / manifest.artifact_path
        artifact.rename(tmp_path / "old-generation")
        artifact.write_bytes(b"replaced pathname is never deserialized")
        replaced.append(True)
        return result
    monkeypatch.setattr(checkpoint, "_hash_file", swap)
    loaded = checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)
    assert replaced and torch.equal(loaded.model_state_dict["weight"], model.weight)


@pytest.mark.parametrize("boundary", ["payload_synced", "payload_published", "manifest_synced", "manifest_published"])
def test_manifest_commit_point_retains_readable_generations(tmp_path, monkeypatch, boundary):
    path = tmp_path / "model.json"
    model = torch.nn.Linear(1, 1)
    old = save(path, model)
    old_bytes = path.read_bytes()
    def fault(name):
        if name == boundary:
            raise OSError("injected publication failure")
    monkeypatch.setattr(checkpoint, "_publication_boundary", fault)
    with pytest.raises(checkpoint.CheckpointError):
        save(path, model, iteration=1)
    loaded = checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)
    assert loaded.iteration == (1 if boundary == "manifest_published" else 0)
    old_copy = tmp_path / "old.json"
    old_copy.write_bytes(old_bytes)
    assert checkpoint.load_checkpoint_v2(old_copy, artifact_root=tmp_path).iteration == 0
    assert (tmp_path / old.artifact_path).exists()
    assert not list(tmp_path.glob(".checkpoint-*.tmp"))


def test_raw_legacy_and_duplicate_manifest_keys_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"iter": 3}, path)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)


def test_manifest_outside_root_rejected(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    path = tmp_path / "outside.json"
    save(path, torch.nn.Linear(1, 1))
    with pytest.raises(checkpoint.CheckpointError, match="escapes"):
        checkpoint.load_checkpoint_v2(path, artifact_root=root)


def test_directory_swap_after_manifest_open_stays_in_pinned_directory(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "model.json"
    model = torch.nn.Linear(1, 1)
    save(path, model)
    original = checkpoint._manifest
    def swap(data):
        manifest = original(data)
        root.rename(tmp_path / "old-root")
        root.mkdir()
        (root / manifest.artifact_path).write_bytes(b"outside pinned directory")
        return manifest
    monkeypatch.setattr(checkpoint, "_manifest", swap)
    loaded = checkpoint.load_checkpoint_v2(path, artifact_root=root)
    assert torch.equal(loaded.model_state_dict["weight"], model.weight)


def test_same_inode_equal_size_write_after_hash_cannot_change_loaded_weights(tmp_path, monkeypatch):
    import io
    path = tmp_path / "model.json"
    model = torch.nn.Linear(1, 1)
    manifest = save(path, model)
    artifact = tmp_path / manifest.artifact_path
    value = torch.load(artifact, weights_only=True)
    value["model_state_dict"]["weight"].fill_(777)
    replacement = io.BytesIO()
    torch.save(value, replacement)
    assert len(replacement.getvalue()) == artifact.stat().st_size
    inode = artifact.stat().st_ino
    original = checkpoint._hash_file
    def mutate(file):
        result = original(file)
        with artifact.open("r+b") as target:
            target.write(replacement.getvalue())
        assert artifact.stat().st_ino == inode
        return result
    monkeypatch.setattr(checkpoint, "_hash_file", mutate)
    loaded = checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)
    assert torch.equal(loaded.model_state_dict["weight"], model.weight)


def test_size_hash_and_load_share_the_sealed_snapshot_descriptor(tmp_path, monkeypatch):
    import fcntl
    path = tmp_path / "model.json"
    save(path, torch.nn.Linear(1, 1))
    seen = []
    hash_file, torch_load = checkpoint._hash_file, torch.load
    def hashed(file):
        seen.append(file.fileno())
        assert fcntl.fcntl(file.fileno(), fcntl.F_GET_SEALS) & fcntl.F_SEAL_WRITE
        return hash_file(file)
    def loaded(file, *, map_location, weights_only):
        assert file.fileno() == seen[-1] and weights_only is True
        with pytest.raises(OSError):
            os.write(file.fileno(), b"cannot modify sealed snapshot")
        return torch_load(file, map_location=map_location, weights_only=weights_only)
    monkeypatch.setattr(checkpoint, "_hash_file", hashed)
    monkeypatch.setattr(torch, "load", loaded)
    checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)


def test_unavailable_snapshot_seals_fail_closed_before_load(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    save(path, torch.nn.Linear(1, 1))
    monkeypatch.delattr(os, "memfd_create")
    with pytest.raises(checkpoint.CheckpointError, match="sealed snapshots unavailable"):
        checkpoint.load_checkpoint_v2(path, artifact_root=tmp_path)
