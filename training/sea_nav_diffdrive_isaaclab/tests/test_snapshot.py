# SPDX-License-Identifier: MIT
"""Snapshot reserve/restore/ack and fail-closed tests (A6.2/A6.3)."""

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from snapshot import (  # noqa: E402
    SnapshotPayload,
    SnapshotStore,
    estimate_snapshot_bytes,
    physical_parameters_hash,
)


def _payload(env_id=0, mass=13.7):
    p = SnapshotPayload(
        env_id=env_id,
        tick=10,
        root_pose=[0.0, 0.0, 0.0632, 0.0, 0.0, 0.0, 1.0],
        physical_params={"mass_kg": mass, "friction": 0.8},
        observation_history=[0.0] * 550,
    )
    p.finalize_hash()
    return p


def test_physical_params_hash_stable():
    p = _payload()
    assert physical_parameters_hash(p.physical_params) == p.physical_params_hash


def test_save_reserve_restore_ack():
    store = SnapshotStore(2)
    snap_id = store.save_point(0, _payload(0))
    baseline = [store.get_live(i) if i == 0 else None for i in range(2)]
    store.save_point(1, _payload(1, mass=12.0))
    baseline[1] = store.get_live(1)
    reserved = store.reserve(0, snap_id)
    reserved.tick = 99
    store.restore(0, reserved)
    store.ack(0)
    assert store.get_live(0).tick == 99
    assert store.other_rows_unchanged(0, baseline)


def test_restore_hash_tamper_fail_closed():
    store = SnapshotStore(1)
    snap_id = store.save_point(0, _payload())
    original_tick = store.get_live(0).tick
    payload = store.reserve(0, snap_id)
    payload.physical_params_hash = "deadbeef"
    with pytest.raises(ValueError, match="hash"):
        store.restore(0, payload)
    assert store.get_live(0).tick == original_tick


def test_restore_row_isolation_rejected():
    store = SnapshotStore(2)
    snap_id = store.save_point(0, _payload(0))
    payload = store.reserve(0, snap_id)
    payload.env_id = 1
    with pytest.raises(ValueError, match="isolation"):
        store.restore(0, payload)


def test_reserve_twice_rejected():
    store = SnapshotStore(1)
    snap_id = store.save_point(0, _payload())
    store.reserve(0, snap_id)
    with pytest.raises(RuntimeError, match="reservation"):
        store.reserve(0, snap_id)


def test_cancel_restores_backup():
    store = SnapshotStore(1)
    snap_id = store.save_point(0, _payload())
    original_tick = store.get_live(0).tick
    payload = store.reserve(0, snap_id)
    payload.tick = 500
    store.restore(0, payload)
    store.cancel(0)
    assert store.get_live(0).tick == original_tick


def test_memory_estimate_positive():
    assert estimate_snapshot_bytes(2048) > 0
