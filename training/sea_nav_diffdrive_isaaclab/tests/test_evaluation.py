# SPDX-License-Identifier: MIT
"""Evaluation dwell and outcome priority tests (A8.1)."""

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from evaluation import (  # noqa: E402
    EVAL_MAX_TICKS,
    SUCCESS_DWELL_TICKS,
    EpisodeOutcome,
    EvalEpisodeState,
    resolve_outcome_priority,
)


def test_success_after_150_tick_dwell():
    st = EvalEpisodeState()
    outcome = None
    for _ in range(SUCCESS_DWELL_TICKS - 1):
        outcome = st.step(0.4, False)
        assert outcome is None
    outcome = st.step(0.4, False)
    assert outcome == EpisodeOutcome.SUCCESS
    assert st.completion_tick == SUCCESS_DWELL_TICKS


def test_dwell_resets_when_leaving_goal():
    st = EvalEpisodeState()
    for _ in range(10):
        st.step(0.4, False)
    st.step(0.6, False)
    assert st.dwell == 0


def test_collision_beats_success():
    assert resolve_outcome_priority(collision=True, success=True, timeout=False) == EpisodeOutcome.COLLISION


def test_success_beats_timeout():
    assert resolve_outcome_priority(collision=False, success=True, timeout=True) == EpisodeOutcome.SUCCESS


def test_timeout_at_eval_horizon():
    st = EvalEpisodeState()
    for _ in range(EVAL_MAX_TICKS):
        st.step(2.0, False)
    assert st.finalize() == EpisodeOutcome.TIMEOUT


def test_30s_is_1500_ticks():
    assert EVAL_MAX_TICKS == 1500
