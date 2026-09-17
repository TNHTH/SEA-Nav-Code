# SPDX-License-Identifier: MIT
"""Ablation profile and resolved preset tests (A5.5 / A7.1 stubs)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/sea_nav_core/src"))

from configs.ablation_profiles import (  # noqa: E402
    ABLATION_PROFILE_NAMES,
    resolve_ablation_profile,
)
from configs.formal_train import (  # noqa: E402
    FORMAL_TRAIN_BUDGET,
    apply_formal_overrides,
    refuse_formal_budget_override,
)
from configs.smoke_4060 import SMOKE_4060_BUDGET, apply_smoke_overrides  # noqa: E402


@pytest.mark.parametrize(
    "name,expected",
    [
        ("full", (True, True, True)),
        ("without_acsi", (False, True, True)),
        ("without_shield", (True, False, True)),
        ("without_lreg", (True, True, False)),
    ],
)
def test_only_one_ablation_axis_changes(name, expected):
    profile = resolve_ablation_profile(name)
    flags = (profile.acsi_enabled, profile.shield_enabled, profile.lreg_enabled)
    assert flags == expected
    assert profile.alpha_min == 0.1


@pytest.mark.parametrize("name", ["baseline", "without_everything", "", None])
def test_unknown_ablation_profile_rejected(name):
    with pytest.raises(ValueError):
        resolve_ablation_profile(name)


def test_adapter_reexport_matches_core_registry():
    assert set(ABLATION_PROFILE_NAMES) == {
        "full", "without_acsi", "without_shield", "without_lreg",
    }


def test_smoke_preset_matches_runbook():
    cfg = apply_smoke_overrides("without_shield", num_envs=16)
    assert cfg.preset == "smoke_4060"
    assert cfg.num_envs == 16
    assert cfg.num_steps_per_env == 24
    assert cfg.max_iterations == 4
    assert cfg.ppo["num_learning_epochs"] == 5
    assert cfg.ppo["num_mini_batches"] == 4


def test_smoke_rejects_invalid_num_envs():
    with pytest.raises(ValueError, match="num_envs"):
        apply_smoke_overrides(num_envs=64)


def test_formal_preset_is_locked():
    cfg = apply_formal_overrides("full", master_seed=43)
    assert cfg.num_envs == FORMAL_TRAIN_BUDGET.num_envs == 2048
    assert cfg.num_steps_per_env == 48
    assert cfg.max_iterations == 2000
    assert cfg.metadata["budget_locked"] is True


def test_formal_refuses_budget_override():
    with pytest.raises(ValueError, match="locked"):
        refuse_formal_budget_override({"num_envs": 512})
    with pytest.raises(ValueError, match="locked"):
        refuse_formal_budget_override({"ppo": {"max_iterations": 100}})
    with pytest.raises(ValueError, match="locked"):
        apply_formal_overrides("full", master_seed=42, num_envs=32)


def test_formal_rejects_unknown_seed():
    with pytest.raises(ValueError, match="master_seed"):
        apply_formal_overrides(master_seed=99)


def test_smoke_and_formal_share_ppo_hparams_except_budget():
    smoke = SMOKE_4060_BUDGET
    formal = FORMAL_TRAIN_BUDGET
    shared_keys = {
        "num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam",
        "learning_rate", "entropy_coef", "max_grad_norm",
    }
    for key in shared_keys:
        assert smoke.ppo[key] == formal.ppo[key]
