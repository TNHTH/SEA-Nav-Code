# SPDX-License-Identifier: MIT
"""Runner train-ready gate tests (A7.2)."""

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import env_cfg  # noqa: E402
from runner import SeaNavRunner, create_runner  # noqa: E402


class _FakeEnv:
    def reset(self):
        return {"obs": 0}

    def step(self, action):
        return {"obs": 1}, 0.0, False, {}

    def close(self):
        pass


class _FakePolicy:
    def act(self, obs):
        return [0.0, 0.0]


def test_runner_refuses_when_not_train_ready():
    assert env_cfg.TRAIN_READY is False
    with pytest.raises(RuntimeError, match="refused"):
        create_runner(_FakeEnv(), _FakePolicy())


def test_runner_works_when_train_ready_monkeypatched(monkeypatch):
    monkeypatch.setattr(env_cfg, "TRAIN_READY", True)
    runner = create_runner(_FakeEnv(), _FakePolicy())
    n = runner.run_iterations(3)
    assert n == 3
    runner.close()
