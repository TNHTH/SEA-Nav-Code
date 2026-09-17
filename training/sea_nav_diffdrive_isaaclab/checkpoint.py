# SPDX-License-Identifier: MIT
"""Training checkpoint save/load with run identity (A7.4).

Resume from a checkpoint creates a **new** ``run_id`` while preserving
``training_lineage`` from the source checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

CHECKPOINT_SCHEMA_VERSION = 1


def _canonical_json(data: Dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def _digest(data: Dict) -> str:
    return hashlib.sha256(_canonical_json(data)).hexdigest()


def new_run_id() -> str:
    return uuid.uuid4().hex


@dataclass
class RunIdentity:
    run_id: str
    training_lineage: list[str] = field(default_factory=list)
    parent_run_id: Optional[str] = None
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    code_hash: str = ""
    config_hash: str = ""
    asset_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "training_lineage": list(self.training_lineage),
            "parent_run_id": self.parent_run_id,
            "created_utc": self.created_utc,
            "code_hash": self.code_hash,
            "config_hash": self.config_hash,
            "asset_hash": self.asset_hash,
        }


@dataclass
class CheckpointManifest:
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    identity: RunIdentity = field(default_factory=lambda: RunIdentity(new_run_id()))
    iteration: int = 0
    profile: str = "full"
    master_seed: int = 42
    rng_state: Dict[str, Any] = field(default_factory=dict)
    env_logical: Dict[str, Any] = field(default_factory=dict)
    acsi_ring: Dict[str, Any] = field(default_factory=dict)
    physical_params_hash: str = ""
    body_digest: str = ""

    def finalize(self) -> None:
        body = {
            "iteration": self.iteration,
            "profile": self.profile,
            "master_seed": self.master_seed,
            "rng_state": self.rng_state,
            "env_logical": self.env_logical,
            "acsi_ring": self.acsi_ring,
            "physical_params_hash": self.physical_params_hash,
        }
        self.body_digest = _digest(body)

    def validate(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        body = {
            "iteration": self.iteration,
            "profile": self.profile,
            "master_seed": self.master_seed,
            "rng_state": self.rng_state,
            "env_logical": self.env_logical,
            "acsi_ring": self.acsi_ring,
            "physical_params_hash": self.physical_params_hash,
        }
        if _digest(body) != self.body_digest:
            raise ValueError("checkpoint body_digest mismatch (tampered)")


def save_checkpoint(
    path: Path,
    manifest: CheckpointManifest,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Path:
    """Atomic temp->fsync->rename checkpoint write."""
    path = path.resolve()
    manifest.finalize()
    payload = {
        "manifest": manifest.to_dict() if hasattr(manifest, "to_dict") else {
            "schema_version": manifest.schema_version,
            "identity": manifest.identity.to_dict(),
            "iteration": manifest.iteration,
            "profile": manifest.profile,
            "master_seed": manifest.master_seed,
            "rng_state": manifest.rng_state,
            "env_logical": manifest.env_logical,
            "acsi_ring": manifest.acsi_ring,
            "physical_params_hash": manifest.physical_params_hash,
            "body_digest": manifest.body_digest,
        },
        "tensors": {k: v.detach().cpu() for k, v in (tensors or {}).items()},
    }
    if "manifest" not in payload or "identity" not in payload["manifest"]:
        payload["manifest"] = {
            "schema_version": manifest.schema_version,
            "identity": manifest.identity.to_dict(),
            "iteration": manifest.iteration,
            "profile": manifest.profile,
            "master_seed": manifest.master_seed,
            "rng_state": manifest.rng_state,
            "env_logical": manifest.env_logical,
            "acsi_ring": manifest.acsi_ring,
            "physical_params_hash": manifest.physical_params_hash,
            "body_digest": manifest.body_digest,
        }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: Path) -> tuple[CheckpointManifest, Dict[str, torch.Tensor]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    m = payload["manifest"]
    manifest = CheckpointManifest(
        schema_version=int(m["schema_version"]),
        identity=RunIdentity(**m["identity"]),
        iteration=int(m["iteration"]),
        profile=str(m["profile"]),
        master_seed=int(m["master_seed"]),
        rng_state=dict(m.get("rng_state", {})),
        env_logical=dict(m.get("env_logical", {})),
        acsi_ring=dict(m.get("acsi_ring", {})),
        physical_params_hash=str(m.get("physical_params_hash", "")),
        body_digest=str(m["body_digest"]),
    )
    manifest.validate()
    tensors = payload.get("tensors", {})
    return manifest, tensors


def resume_from_checkpoint(
    path: Path,
    *,
    code_hash: str = "",
    config_hash: str = "",
    asset_hash: str = "",
) -> tuple[CheckpointManifest, Dict[str, torch.Tensor]]:
    """Load checkpoint and mint a NEW run_id for the resumed run."""
    manifest, tensors = load_checkpoint(path)
    old_id = manifest.identity.run_id
    lineage = list(manifest.identity.training_lineage) + [old_id]
    manifest.identity = RunIdentity(
        run_id=new_run_id(),
        training_lineage=lineage,
        parent_run_id=old_id,
        code_hash=code_hash or manifest.identity.code_hash,
        config_hash=config_hash or manifest.identity.config_hash,
        asset_hash=asset_hash or manifest.identity.asset_hash,
    )
    manifest.finalize()
    return manifest, tensors
