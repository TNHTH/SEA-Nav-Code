# SPDX-License-Identifier: MIT
"""Tests for the 10 Hz perception delay queue (A4.2)."""

from pathlib import Path
import sys

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/sea_nav_core/src"))

from perception_delay import (  # noqa: E402
    ACQUISITION_PERIOD_TICKS,
    LATENCY_MAX_S,
    LATENCY_MIN_S,
    POLICY_DT_S,
    PerceptionDelayQueue,
)


def _sensor(batch=2, marker=1.0):
    return (
        torch.full((batch, 41), marker),
        torch.full((batch, 2), marker),
        torch.ones(batch, 41, dtype=torch.bool),
    )


def test_independent_env_phases_do_not_share_global_tick_mod():
    q = PerceptionDelayQueue(2, device=torch.device("cpu"))
    ids = torch.arange(2)
    ranges, goal, valid = _sensor(2, 0.0)
    q.normal_reset(ids[:1], 0.0, ranges[:1], goal[:1], valid[:1],
                   next_acquisition_tick=torch.tensor([1]))
    q.normal_reset(ids[1:], 0.0, ranges[1:], goal[1:], valid[1:],
                   next_acquisition_tick=torch.tensor([3]))
    acquired = [[], []]
    gen = torch.Generator().manual_seed(3)
    for tick in range(1, 10):
        now = tick * POLICY_DT_S
        before = q.next_acquisition_tick.clone()
        q.push_acquisitions(now, * _sensor(2, float(tick)), generator=gen)
        for env in range(2):
            if q.next_acquisition_tick[env].item() != before[env].item():
                acquired[env].append(tick)
    assert acquired[0] == [1, 5]
    assert acquired[1] == [3, 5]
    assert acquired[0] != acquired[1][:1]


def test_delay_window_and_age_from_acquisition():
    q = PerceptionDelayQueue(1, device=torch.device("cpu"))
    ids = torch.arange(1)
    ranges, goal, valid = _sensor(1, 0.0)
    q.normal_reset(ids, 0.0, ranges, goal, valid, next_acquisition_tick=torch.tensor([5]))
    gen = torch.Generator().manual_seed(11)
    seen_latencies = set()
    seen_ages = set()
    for tick in range(1, 41):
        now = tick * POLICY_DT_S
        q.begin_policy_tick()
        q.push_acquisitions(now, * _sensor(1, float(tick)), generator=gen)
        held = q.observe(now)
        assert torch.all(held.acquisition_time <= now + 1e-8)
        assert torch.allclose(
            held.age_s.reshape(-1).to(torch.float64),
            torch.as_tensor(now, dtype=torch.float64) - held.acquisition_time,
            atol=1e-6,
        )
        if tick >= 5 and tick % 5 == 0 and held.sampled_latency.item() > 0:
            lat = held.sampled_latency.item()
            seen_latencies.add(round(lat, 4))
            seen_ages.add(round(held.age_s.item(), 2))
            assert LATENCY_MIN_S <= lat < LATENCY_MAX_S
    assert any(age > LATENCY_MAX_S for age in seen_ages)


def test_replay_jump_preserves_phase_at_251_not_255():
    q = PerceptionDelayQueue(1, device=torch.device("cpu"))
    ids = torch.arange(1)
    ranges, goal, valid = _sensor(1, 0.0)
    q.normal_reset(ids, 0.0, ranges, goal, valid)
    gen = torch.Generator().manual_seed(21)
    for tick in range(1, 105):
        now = tick * POLICY_DT_S
        q.push_acquisitions(now, * _sensor(1, float(tick)), generator=gen)
        q.observe(now)
    snap = q.snapshot()
    assert snap.next_acquisition_tick.item() == 105
    q.restore_snapshot(snap, current_tick=250, source_tick=104)
    assert q.next_acquisition_tick.item() == 251
    assert q.next_acquisition_tick.item() != 255


def test_hold_newest_arrived_packet():
    q = PerceptionDelayQueue(1, device=torch.device("cpu"), policy_dt_s=POLICY_DT_S)
    ids = torch.arange(1)
    q.normal_reset(ids, 0.0, *_sensor(1, 0.0), next_acquisition_tick=torch.tensor([0]))
    gen = torch.Generator().manual_seed(5)
    # Force two acquisitions before any delivery by using tiny capacity and fast phase.
    q.capacity = 2
    q.push_acquisitions(0.0, *_sensor(1, 1.0), generator=gen)
    q.push_acquisitions(0.1, *_sensor(1, 2.0), generator=gen)
    held = q.observe(0.18)
    assert held.ranges_m[0, 0].item() == pytest.approx(2.0)


def test_queue_overflow_sets_diagnostic():
    q = PerceptionDelayQueue(1, device=torch.device("cpu"))
    q.capacity = 2
    q.acquisition_period_ticks = 1
    ids = torch.arange(1)
    q.normal_reset(ids, 0.0, *_sensor(1, 0.0), next_acquisition_tick=torch.tensor([0]))
    gen = torch.Generator().manual_seed(1)
    for step in range(5):
        now = step * POLICY_DT_S
        q.begin_policy_tick()
        q.push_acquisitions(now, *_sensor(1, float(step)), generator=gen)
        held = q.observe(now)
    assert q.queue_overflow_total.item() >= 1
    assert bool(held.queue_overflow.any()) or q.queue_overflow_total.item() >= 1


def test_reset_masks_only_requested_env():
    q = PerceptionDelayQueue(2, device=torch.device("cpu"))
    ids = torch.arange(2)
    q.normal_reset(ids, 0.0, *_sensor(2, 1.0))
    q.push_acquisitions(0.02, *_sensor(2, 2.0), generator=torch.Generator().manual_seed(0))
    before = q.held_ranges_m[1, 0].item()
    q.normal_reset(torch.tensor([0]), 0.03, *_sensor(1, 9.0))
    assert q.held_ranges_m[0, 0].item() == pytest.approx(9.0)
    assert q.held_ranges_m[1, 0].item() == pytest.approx(before)


def test_nonfinite_time_is_rejected():
    q = PerceptionDelayQueue(1, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="finite"):
        q.push_acquisitions(float("nan"), *_sensor(1, 1.0))
