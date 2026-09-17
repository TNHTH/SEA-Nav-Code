# SPDX-License-Identifier: MIT
"""ACSI probability, curriculum, and replay tests (A6.1)."""

import sys
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from acsi import (  # noqa: E402
    ACSI_FALLBACK_END,
    ACSI_FALLBACK_START,
    AcsiManager,
    acsi_probability,
    rising_edge_collision,
    update_l_goal_from_terminal_distance,
)


def test_acsi_probability_endpoints_and_mid():
    assert acsi_probability(1) == 0.5
    assert acsi_probability(2) == 0.5
    assert acsi_probability(-1) == 0.1
    assert acsi_probability(-2) == 0.1
    assert acsi_probability(0) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "d,expected",
    [(0.49, 1), (0.5, 0), (2.0, 0), (2.01, -1)],
)
def test_l_goal_curriculum_boundaries(d, expected):
    new, err = update_l_goal_from_terminal_distance(0, d)
    assert err is None
    assert new == expected


def test_replay_does_not_update_l_goal():
    new, err = update_l_goal_from_terminal_distance(0, 0.1, is_replay=True)
    assert err is None
    assert new == 0


def test_non_finite_distance_is_integrity_fault():
    new, err = update_l_goal_from_terminal_distance(0, float("nan"))
    assert new == 0
    assert err is not None


def test_rising_edge_collision():
    assert rising_edge_collision(True, False) is True
    assert rising_edge_collision(True, True) is False
    assert rising_edge_collision(False, False) is False


def test_manager_curriculum_and_replay():
    mgr = AcsiManager(2)
    mgr.apply_terminal_curriculum([0], torch.tensor([0.49]))
    assert mgr.states[0].l_goal == 1
    mgr.apply_terminal_curriculum([0], torch.tensor([0.1]), is_replay=True)
    assert mgr.states[0].l_goal == 1


def test_single_bernoulli_draw_uses_pre_update_l_goal():
    mgr = AcsiManager(1)
    mgr.states[0].l_goal = 1
    gen = torch.Generator()
    gen.manual_seed(12345)
    triggered, probs = mgr.draw([0], gen)
    assert probs[0].item() == 0.5
    assert triggered.dtype == torch.bool


def test_acsi_off_does_not_consume_stream():
    mgr = AcsiManager(1, acsi_enabled=False)
    gen = torch.Generator()
    gen.manual_seed(0)
    triggered, probs = mgr.draw([0], gen)
    assert not triggered.any()
    assert (probs == 0).all()


def test_fallback_ring_range():
    st = AcsiManager(1).states[0]
    gen = torch.Generator()
    gen.manual_seed(7)
    for t in range(ACSI_FALLBACK_START, ACSI_FALLBACK_END + 1):
        st.push_fallback(t)
    tick = st.sample_fallback(gen)
    assert ACSI_FALLBACK_START <= tick <= ACSI_FALLBACK_END
