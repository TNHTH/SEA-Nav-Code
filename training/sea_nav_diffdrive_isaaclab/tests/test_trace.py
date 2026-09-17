# SPDX-License-Identifier: MIT
"""Per-row trace validity, generation, and isolation tests (A3.3)."""

import sys
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from trace import ACTION_STAGES, ActionStageTrace  # noqa: E402


def _stages(num_envs=2, fill=0.0):
    return {
        name: torch.full((num_envs, 2), fill + i * 0.01)
        for i, name in enumerate(ACTION_STAGES)
    }


def test_record_then_snapshot_returns_copies():
    trace = ActionStageTrace(2, 4, torch.device("cpu"))
    stages = _stages(fill=1.0)
    trace.record(0, stages)
    snap = trace.snapshot(0)
    snap["executed_command"][0, 0] = 99.0
    assert trace.stage("executed_command", 0)[0, 0].item() != 99.0


def test_unwritten_slot_raises():
    trace = ActionStageTrace(2, 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="not recorded"):
        trace.stage("policy_action", 0)
    trace.record(1, _stages(fill=2.0))
    with pytest.raises(ValueError, match="not recorded"):
        trace.stage("policy_action", 0)


def test_partial_record_does_not_mutate_on_validation_error():
    trace = ActionStageTrace(2, 4, torch.device("cpu"))
    good = _stages(fill=3.0)
    trace.record(0, good)
    bad = dict(good)
    bad["policy_action"] = torch.zeros(2, 3)  # wrong shape
    with pytest.raises(ValueError):
        trace.record(1, bad)
    # slot 1 must remain unwritten
    assert not bool(trace._valid[1].any())
    torch.testing.assert_close(
        trace.stage("policy_action", 0), good["policy_action"]
    )


def test_clear_rows_isolates_and_invalidates_generation():
    trace = ActionStageTrace(3, 4, torch.device("cpu"))
    trace.record(0, _stages(num_envs=3, fill=1.0))
    before = trace.stage("executed_command", 0).clone()
    trace.clear_rows([1])
    # Row 0 and 2 buffers were zero-filled for cleared rows only in storage,
    # but generation bump makes row1 unreadable while others stay valid.
    assert bool(trace._valid[0, 0].item())
    assert not bool(trace._valid[0, 1].item())
    assert bool(trace._valid[0, 2].item())
    with pytest.raises(ValueError, match="unwritten|stale"):
        trace._ensure_readable(0, 1)
    # Re-record row1 under new generation.
    stages = _stages(num_envs=3, fill=5.0)
    trace.record(0, stages, rows=torch.tensor([1]))
    assert bool(trace._valid[0, 1].item())
    assert before[0].tolist() == trace.stage("executed_command", 0)[0].tolist()


def test_missing_stage_rejected_before_write():
    trace = ActionStageTrace(1, 2, torch.device("cpu"))
    stages = _stages(num_envs=1)
    del stages["clipped_policy_action"]
    with pytest.raises(ValueError, match="missing stages"):
        trace.record(0, stages)
