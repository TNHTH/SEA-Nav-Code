# SPDX-License-Identifier: MIT
"""CPU tests for DashGo 2D actor/critic and CBF mean adapter (A5.1–A5.2)."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys

import pytest
import torch
from torch.distributions import Normal
from torch.nn import functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/sea_nav_core/src"))
sys.path.insert(0, str(REPO_ROOT / "training/rsl_rl"))

from sea_nav_core import SeaNavActorInput  # noqa: E402
from sea_nav_core.observation import POLICY_OBS_DIM  # noqa: E402

from actor_critic import SeaNavDiffDriveActorCritic, try_register_rsl_policy_class  # noqa: E402
from cbf_adapter import CBFMeanAdapter  # noqa: E402


def _make_obs(batch: int = 2, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, POLICY_OBS_DIM)


def _safe_context(policy_obs: torch.Tensor, fill: float = 1.5) -> SeaNavActorInput:
    batch = policy_obs.shape[0]
    return SeaNavActorInput(
        policy_obs=policy_obs,
        safety_ranges_m=torch.full((batch, 41), fill),
        safety_valid=torch.ones(batch, 41, dtype=torch.bool),
        safety_age_s=torch.zeros(batch, 1),
    )


def _actor(**kwargs) -> SeaNavDiffDriveActorCritic:
    defaults = dict(
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        encoder_hidden_dims=[32, 16],
        init_noise_std=1.5,
    )
    defaults.update(kwargs)
    return SeaNavDiffDriveActorCritic(**defaults)


def test_rejects_wrong_observation_and_action_dims():
    actor = _actor()
    with pytest.raises(ValueError, match="550"):
        actor.action_mean_for(torch.zeros(2, 100))
    with pytest.raises(ValueError, match="2D navigation"):
        SeaNavDiffDriveActorCritic(num_actions=3)


def test_network_outputs_two_d_mean_positive_alpha_and_std_about_one_point_five():
    actor = _actor()
    obs = _make_obs()
    context = _safe_context(obs)
    mean = actor.action_mean_for(obs, actor_context=context)
    assert mean.shape == (2, 2)
    assert torch.isfinite(mean).all()
    actor.update_distribution(obs, actor_context=context)
    assert actor.alpha.shape == (2, 1)
    assert (actor.alpha > 0).all()
    std = actor.log_std.exp()
    assert std.shape == (2,)
    assert std.mean().item() == pytest.approx(1.5, rel=0, abs=1e-6)


def test_actor_latent_detach_critic_does_not_detach():
    actor = _actor()
    obs = _make_obs(batch=1)
    context = _safe_context(obs)

    actor.action_mean_for(obs, actor_context=context).square().sum().backward()
    encoder_grads = [p.grad for p in actor.encoder.parameters() if p.grad is not None]
    assert not encoder_grads, "encoder must not receive gradients through detached actor latent"
    assert any(p.grad is not None for p in actor.nav_head.parameters())

    actor.zero_grad(set_to_none=True)
    actor.evaluate(obs).square().sum().backward()
    assert any(p.grad is not None for p in actor.encoder.parameters())
    assert any(p.grad is not None for p in actor.critic.parameters())


def test_pure_query_preserves_distribution_rng_and_aux_state():
    actor = _actor()
    obs = _make_obs()
    context = _safe_context(obs)
    actor.act(obs, actor_context=context)
    saved_dist = actor.distribution
    saved_alpha = actor.alpha.detach().clone()
    saved_u_bar = actor.u_bar.detach().clone()
    saved_u_s = actor.u_s.detach().clone()
    saved_std = actor.log_std.detach().clone()

    torch.manual_seed(99)
    query = actor.action_mean_for(_make_obs(seed=7), actor_context=context)
    assert query.requires_grad
    assert actor.distribution is saved_dist
    torch.testing.assert_close(actor.alpha, saved_alpha)
    torch.testing.assert_close(actor.u_bar, saved_u_bar)
    torch.testing.assert_close(actor.u_s, saved_u_s)
    torch.testing.assert_close(actor.log_std, saved_std)


def test_sample_is_draw_from_cbf_mean_not_resampled_through_shield():
    actor = _actor()
    obs = _make_obs(batch=1)
    context = _safe_context(obs, fill=0.35)
    torch.manual_seed(23)
    sample = actor.act(obs, actor_context=context)
    torch.manual_seed(23)
    expected = Normal(actor.action_mean.detach(), actor.action_std.detach()).sample()
    torch.testing.assert_close(sample, expected, rtol=0, atol=0)
    assert not torch.allclose(actor.u_bar, actor.u_s)


def test_go2_none_context_skips_cbf_without_extra_allocation():
    actor = _actor()
    obs = _make_obs()
    mean = actor.action_mean_for(obs, actor_context=None)
    assert mean.shape == (2, 2)
    actor.act(obs, actor_context=None)
    torch.testing.assert_close(actor.u_bar, actor.u_s)


def test_action_mean_for_aux_uses_geometric_range_interpolation():
    actor = _actor()
    obs = _make_obs(batch=1)
    current = _safe_context(obs, fill=1.0)
    nxt = _safe_context(obs, fill=2.0)
    beta = torch.tensor([[0.5]])
    u_bar, alpha = actor._nominal_mean_and_alpha(obs)
    mid_direct = actor._cbf.action_mean_for_aux(
        u_bar, alpha, current, nxt, beta,
    )
    interp = actor._cbf.interpolate_aux_context(current, nxt, beta)
    expected_ranges = torch.full((1, 41), math.sqrt(2.0))
    assert torch.allclose(interp.safety_ranges_m, expected_ranges, atol=1e-5)
    wrapped = actor.action_mean_for_aux(obs, current, nxt, beta)
    torch.testing.assert_close(wrapped, mid_direct)


def test_real_raw_outside_domain_rejected_for_filter_mean():
    adapter = CBFMeanAdapter()
    obs = _make_obs(batch=1)
    bad = SeaNavActorInput(
        policy_obs=obs,
        safety_ranges_m=torch.full((1, 41), 5.0),
        safety_valid=torch.ones(1, 41, dtype=torch.bool),
        safety_age_s=torch.zeros(1, 1),
    )
    with pytest.raises(ValueError, match="raw safety"):
        adapter.filter_mean(torch.zeros(1, 2), torch.ones(1, 1), bad)


def test_left_obstacle_pushes_mean_right_of_nominal():
    adapter = CBFMeanAdapter()
    nominal = torch.tensor([[0.25, 0.0]])
    alpha = torch.ones(1, 1)
    ranges = torch.full((1, 41), 3.0)
    # Left-forward ray (~-6 deg) with a close return triggers CBF braking.
    ranges[0, 19] = 0.25
    valid = torch.ones(1, 41, dtype=torch.bool)
    context = SeaNavActorInput(
        policy_obs=_make_obs(1),
        safety_ranges_m=ranges,
        safety_valid=valid,
        safety_age_s=torch.zeros(1, 1),
    )
    stages = adapter.filter_mean(nominal, alpha, context)
    assert stages.distribution_mean[0, 0] < nominal[0, 0]


def test_declares_actor_context_for_ppo_compatibility():
    for name in ("act", "action_mean_for", "evaluate"):
        method = getattr(SeaNavDiffDriveActorCritic, name)
        assert "actor_context" in inspect.signature(method).parameters, name


def test_registration_stub_adds_class_without_eval():
    try:
        from rsl_rl.runners.on_policy_runner import POLICY_REGISTRY, resolve_policy_class
    except ImportError:
        pytest.skip("rsl_rl not importable")
    assert try_register_rsl_policy_class()
    assert resolve_policy_class("SeaNavDiffDriveActorCritic") is SeaNavDiffDriveActorCritic
    assert "SeaNavDiffDriveActorCritic" in POLICY_REGISTRY


def test_policy_factory_registers_dashgo_actor_without_eval():
    try:
        from rsl_rl.policy_factory import build_dashgo_actor_critic, register_dashgo_policy_class
        from rsl_rl.runners.on_policy_runner import resolve_policy_class
    except ImportError:
        pytest.skip("rsl_rl not importable")
    assert register_dashgo_policy_class()
    actor = build_dashgo_actor_critic(
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        encoder_hidden_dims=[32, 16],
    )
    assert isinstance(actor, SeaNavDiffDriveActorCritic)
    assert resolve_policy_class("SeaNavDiffDriveActorCritic") is SeaNavDiffDriveActorCritic
