# SPDX-License-Identifier: MIT
"""Tests for the 550-D observation pipeline adapter (A4.1)."""

import math
from pathlib import Path
import sys

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/sea_nav_core/src"))

from sea_nav_core import (  # noqa: E402
    FRAME_COUNT,
    FRAME_DIM,
    POLICY_OBS_DIM,
    build_policy_frame,
)
from observation_pipeline import ObservationPipeline  # noqa: E402


def _frame(batch=2, fill=1.0):
    command = torch.zeros(batch, 3)
    command[:, 0] = fill
    command[:, 2] = fill + 0.1
    return build_policy_frame(
        torch.full((batch, 3), fill),
        command,
        torch.full((batch, 3), fill + 0.2),
        torch.full((batch, 3), fill + 0.3),
        torch.full((batch, 41), 1.5),
        torch.full((batch, 2), fill + 0.4),
    )


def test_policy_obs_shape_and_field_order():
    pipe = ObservationPipeline(2, torch.float32, torch.device("cpu"))
    pipe.normal_reset(_frame(batch=2, fill=1.0))
    obs = pipe.policy_obs()
    assert obs.shape == (2, POLICY_OBS_DIM)
    assert obs[0, 0].item() == pytest.approx(1.0)
    assert obs[0, FRAME_DIM * 9 + 3].item() == pytest.approx(1.0)
    assert obs[0, 4].item() == 0.0
    assert obs[0, 12].item() == pytest.approx(math.log2(1.5))


def test_push_once_per_tick_and_history_moves():
    pipe = ObservationPipeline(1, torch.float32, torch.device("cpu"))
    pipe.normal_reset(_frame(batch=1, fill=1.0))
    pipe.begin_policy_tick()
    pipe.push_frame_once(_frame(batch=1, fill=2.0))
    with pytest.raises(RuntimeError, match="already called"):
        pipe.push_frame_once(_frame(batch=1, fill=3.0))
    slots = [pipe.policy_obs()[0, i * FRAME_DIM].item() for i in range(FRAME_COUNT)]
    assert slots == [1.0] * 9 + [2.0]


def test_repeated_query_does_not_mutate():
    pipe = ObservationPipeline(1, torch.float32, torch.device("cpu"))
    pipe.normal_reset(_frame(batch=1, fill=1.0))
    first = pipe.policy_obs()
    second = pipe.policy_obs()
    assert torch.equal(first, second)
    pipe.begin_policy_tick()
    pipe.push_frame_once(_frame(batch=1, fill=2.0))
    assert torch.equal(first, second)


def test_normal_reset_fills_all_ten_slots():
    pipe = ObservationPipeline(2, torch.float32, torch.device("cpu"))
    pipe.normal_reset(_frame(batch=2, fill=7.0))
    frames = pipe.frames()
    for row in range(2):
        for slot in range(FRAME_COUNT):
            assert frames[row, slot, 0].item() == pytest.approx(7.0)


def test_restore_snapshot_without_refill():
    pipe = ObservationPipeline(1, torch.float32, torch.device("cpu"))
    stored = torch.arange(POLICY_OBS_DIM, dtype=torch.float32).view(1, POLICY_OBS_DIM)
    pipe.restore_snapshot(stored)
    assert torch.equal(pipe.policy_obs(), stored)
    pipe.begin_policy_tick()
    pipe.push_frame_once(_frame(batch=1, fill=9.0))
    assert pipe.policy_obs()[0, FRAME_DIM * 9].item() == pytest.approx(9.0)
    assert pipe.policy_obs()[0, 0].item() == pytest.approx(stored[0, FRAME_DIM].item())


def test_lateral_velocity_is_preserved_in_frame():
    frame = build_policy_frame(
        torch.zeros(1, 3),
        torch.tensor([[0.1, 0.0, 0.2]]),
        torch.tensor([[0.3, 0.4, 0.5]]),
        torch.tensor([[0.01, 0.02, 0.03]]),
        torch.full((1, 41), 1.0),
        torch.zeros(1, 2),
    )
    assert frame[0, 6].item() == pytest.approx(0.3)
    assert frame[0, 7].item() == pytest.approx(0.4)
    assert frame[0, 10].item() == pytest.approx(0.02)
    assert frame[0, 11].item() == pytest.approx(0.03)
