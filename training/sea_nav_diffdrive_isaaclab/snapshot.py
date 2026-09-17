# SPDX-License-Identifier: MIT
"""Versioned per-row snapshot reserve/restore/ack (A6.2/A6.3).

Supports atomic replay transactions: select -> reserve -> restore -> validate
-> ack. Fail-closed restore rolls back reservation without mutating other rows.
Physical parameters are hashed; row isolation is enforced throughout.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor

SNAPSHOT_SCHEMA_VERSION = 1


def physical_parameters_hash(params: Dict[str, Any]) -> str:
    """Deterministic SHA-256 over canonical physical-parameter dict."""
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def estimate_snapshot_bytes(num_envs: int = 2048) -> int:
    """Conservative CPU memory estimate for full-batch snapshot (A6.2)."""
    per_env = (
        7 * 4  # root pose
        + 12 * 4  # joint positions/velocities (4 joints)
        + 550 * 4  # observation history
        + 41 * 4 * 2  # raw rays + valid
        + 64  # misc timers/RNG counters
    )
    return num_envs * per_env + 4096


@dataclass
class SnapshotPayload:
    """Versioned logical + physical fields for one env row."""

    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    env_id: int = 0
    tick: int = 0
    root_pose: Optional[List[float]] = None
    joint_state: Optional[Dict[str, List[float]]] = None
    physical_params: Dict[str, Any] = field(default_factory=dict)
    physical_params_hash: str = ""
    observation_history: Optional[List[float]] = None
    perception_queue: Optional[List[Any]] = None
    rng_counters: Dict[str, int] = field(default_factory=dict)
    l_goal: int = 0
    collision_prev_tick: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def finalize_hash(self) -> None:
        self.physical_params_hash = physical_parameters_hash(self.physical_params)

    def validate(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported snapshot schema")
        if not self.physical_params_hash:
            raise ValueError("physical_params_hash required")
        expected = physical_parameters_hash(self.physical_params)
        if self.physical_params_hash != expected:
            raise ValueError("physical_params_hash mismatch")


class ReservationState(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    RESTORING = "restoring"
    ACKED = "acked"


@dataclass
class RowReservation:
    state: ReservationState = ReservationState.FREE
    snapshot_id: Optional[str] = None
    payload: Optional[SnapshotPayload] = None
    backup: Optional[SnapshotPayload] = None


class SnapshotStore:
    """Per-row snapshot bank with reserve/restore/ack transactions."""

    def __init__(self, num_envs: int):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self._rows: List[RowReservation] = [RowReservation() for _ in range(num_envs)]
        self._live: List[Optional[SnapshotPayload]] = [None for _ in range(num_envs)]

    def save_point(self, env_id: int, payload: SnapshotPayload) -> str:
        if not 0 <= env_id < self.num_envs:
            raise ValueError("env_id out of range")
        payload = copy.deepcopy(payload)
        payload.env_id = env_id
        payload.finalize_hash()
        payload.validate()
        snap_id = hashlib.sha256(
            json.dumps(
                {"env_id": env_id, "tick": payload.tick, "hash": payload.physical_params_hash},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        self._live[env_id] = payload
        return snap_id

    def reserve(self, env_id: int, snapshot_id: str) -> SnapshotPayload:
        if not 0 <= env_id < self.num_envs:
            raise ValueError("env_id out of range")
        row = self._rows[env_id]
        if row.state != ReservationState.FREE:
            raise RuntimeError(f"env {env_id}: reservation already active")
        live = self._live[env_id]
        if live is None:
            raise RuntimeError(f"env {env_id}: no snapshot to restore")
        expected = hashlib.sha256(
            json.dumps(
                {
                    "env_id": env_id,
                    "tick": live.tick,
                    "hash": live.physical_params_hash,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        if snapshot_id != expected:
            raise ValueError("snapshot_id mismatch")
        row.state = ReservationState.RESERVED
        row.snapshot_id = snapshot_id
        row.payload = copy.deepcopy(live)
        row.backup = copy.deepcopy(live)
        return row.payload

    def restore(self, env_id: int, payload: SnapshotPayload) -> None:
        if not 0 <= env_id < self.num_envs:
            raise ValueError("env_id out of range")
        row = self._rows[env_id]
        if row.state not in (ReservationState.RESERVED, ReservationState.RESTORING):
            raise RuntimeError(f"env {env_id}: restore without reservation")
        row.state = ReservationState.RESTORING
        try:
            payload.validate()
            if payload.env_id != env_id:
                raise ValueError("row isolation violation: env_id mismatch")
            self._live[env_id] = copy.deepcopy(payload)
        except Exception:
            if row.backup is not None:
                self._live[env_id] = copy.deepcopy(row.backup)
            row.state = ReservationState.FREE
            row.snapshot_id = None
            row.payload = None
            row.backup = None
            raise

    def ack(self, env_id: int) -> None:
        if not 0 <= env_id < self.num_envs:
            raise ValueError("env_id out of range")
        row = self._rows[env_id]
        if row.state != ReservationState.RESTORING:
            raise RuntimeError(f"env {env_id}: ack without successful restore")
        row.state = ReservationState.ACKED
        row.snapshot_id = None
        row.payload = None
        row.backup = None

    def cancel(self, env_id: int) -> None:
        if not 0 <= env_id < self.num_envs:
            raise ValueError("env_id out of range")
        row = self._rows[env_id]
        if row.backup is not None:
            self._live[env_id] = copy.deepcopy(row.backup)
        row.state = ReservationState.FREE
        row.snapshot_id = None
        row.payload = None
        row.backup = None

    def get_live(self, env_id: int) -> SnapshotPayload:
        live = self._live[env_id]
        if live is None:
            raise RuntimeError(f"env {env_id}: no live snapshot")
        return copy.deepcopy(live)

    def other_rows_unchanged(self, env_id: int, baseline: List[Optional[SnapshotPayload]]) -> bool:
        for i, snap in enumerate(baseline):
            if i == env_id:
                continue
            cur = self._live[i]
            if cur is None and snap is None:
                continue
            if cur is None or snap is None:
                return False
            if cur.physical_params_hash != snap.physical_params_hash:
                return False
        return True


def tensor_snapshot_slice(t: Tensor, row: int) -> Tensor:
    """Return a detached copy of one env row from a batch tensor."""
    return t[row].detach().clone()
