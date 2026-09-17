# SPDX-License-Identifier: MIT
"""Checkpoint save/load and resume identity tests (A7.4)."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from checkpoint import (  # noqa: E402
    CheckpointManifest,
    RunIdentity,
    load_checkpoint,
    new_run_id,
    resume_from_checkpoint,
    save_checkpoint,
)


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        manifest = CheckpointManifest(
            identity=RunIdentity(run_id=new_run_id()),
            iteration=7,
            profile="full",
            master_seed=42,
            physical_params_hash="abc",
        )
        tensors = {"w": torch.tensor([1.0, 2.0])}
        save_checkpoint(path, manifest, tensors)
        loaded, loaded_tensors = load_checkpoint(path)
        assert loaded.iteration == 7
        assert loaded.profile == "full"
        torch.testing.assert_close(loaded_tensors["w"], tensors["w"])


def test_tampered_digest_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        manifest = CheckpointManifest(identity=RunIdentity(run_id=new_run_id()))
        save_checkpoint(path, manifest)
        payload = torch.load(path, weights_only=False)
        payload["manifest"]["body_digest"] = "tampered"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="tampered"):
            load_checkpoint(path)


def test_resume_gets_new_run_id():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        old_id = new_run_id()
        manifest = CheckpointManifest(identity=RunIdentity(run_id=old_id))
        save_checkpoint(path, manifest)
        resumed, _ = resume_from_checkpoint(path)
        assert resumed.identity.run_id != old_id
        assert resumed.identity.parent_run_id == old_id
        assert old_id in resumed.identity.training_lineage
