# SPDX-License-Identifier: MIT
"""TorchScript export parity hooks with hash tamper detection (A8.3)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

PARITY_RTOL = 1e-5
EXPORT_ABI = {
    "inputs": [
        {"name": "policy_obs", "shape": [-1, 550], "dtype": "float32"},
        {"name": "safety_ranges_m", "shape": [-1, 41], "dtype": "float32"},
        {"name": "safety_valid", "shape": [-1, 41], "dtype": "bool"},
        {"name": "safety_age_s", "shape": [-1, 1], "dtype": "float32"},
        {"name": "previous_command", "shape": [-1, 2], "dtype": "float32"},
    ],
    "outputs": [
        {"name": "command", "shape": [-1, 2], "dtype": "float32"},
        {"name": "status", "shape": [-1], "dtype": "int64"},
    ],
}


class ExportMeanPolicy(nn.Module):
    """Minimal deterministic mean policy for CPU TorchScript parity tests."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(550, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        policy_obs: torch.Tensor,
        safety_ranges_m: torch.Tensor,
        safety_valid: torch.Tensor,
        safety_age_s: torch.Tensor,
        previous_command: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del safety_ranges_m, safety_valid, safety_age_s, previous_command
        cmd = self.net(policy_obs)
        status = torch.zeros(policy_obs.size(0), dtype=torch.int64, device=policy_obs.device)
        bad = ~torch.isfinite(policy_obs).all(dim=1)
        status[bad] = 128
        cmd[bad] = 0.0
        return cmd, status


def artifact_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ExportManifest:
    model_hash: str
    abi_version: str = "sea_export_abi_v1"
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "model_hash": self.model_hash,
            "artifacts": dict(self.artifacts),
            "abi": EXPORT_ABI,
        }

    def verify_artifact(self, name: str, path: Path) -> None:
        expected = self.artifacts.get(name)
        if expected is None:
            raise ValueError(f"artifact not registered: {name}")
        actual = artifact_digest(path)
        if actual != expected:
            raise ValueError(f"artifact hash tamper detected for {name}")


def export_torchscript(
    model: nn.Module,
    example: Tuple[torch.Tensor, ...],
    path: Path,
) -> ExportManifest:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(path))
    digest = artifact_digest(path)
    manifest = ExportManifest(model_hash=digest, artifacts={"torchscript": digest})
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return manifest


def parity_eager_vs_script(
    eager: nn.Module,
    scripted: torch.jit.ScriptModule,
    example: Tuple[torch.Tensor, ...],
    *,
    rtol: float = PARITY_RTOL,
) -> None:
    eager.eval()
    with torch.no_grad():
        e_out = eager(*example)
        s_out = scripted(*example)
    for e, s in zip(e_out, s_out):
        torch.testing.assert_close(e, s, rtol=rtol, atol=rtol)


def load_and_verify(path: Path, manifest: ExportManifest) -> torch.jit.ScriptModule:
    manifest.verify_artifact("torchscript", path)
    return torch.jit.load(str(path), map_location="cpu")
