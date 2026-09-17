# SPDX-License-Identifier: MIT
"""Tests for the RSL-facing observation wrapper (A4.3)."""

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
    EffectiveCommandEnvelopeSpec,
    INVALID_RAY_POLICY_PLACEHOLDER_M,
    POLICY_OBS_DIM,
    RAW_SAFETY_REASON_ALL_INVALID,
    RAW_SAFETY_REASON_NONFINITE,
    RAW_SAFETY_REASON_OK,
    SEA_RAY_COUNT,
    VALID_NO_HIT_RANGE_M,
    UnicycleLookaheadLSECBFLayer,
    sea_nav_dashgo_candidate_platform_spec,
    sea_nav_dashgo_raw_safety_spec,
)
from wrapper import (  # noqa: E402
    ProprioceptionSample,
    SeaNavObservationWrapper,
    SensorSample,
    assemble_rsl_step_bundle,
    build_timeout_bootstrap_infos,
    encode_sensor_boundary,
    isolate_actor_context_for_forward,
)


def _proprio(batch=2, fill=0.0):
    command = torch.zeros(batch, 3)
    command[:, 0] = fill
    command[:, 2] = fill + 0.05
    return ProprioceptionSample(
        projected_gravity=torch.full((batch, 3), fill + 0.01),
        previous_executed_command=command,
        base_linear_velocity=torch.full((batch, 3), fill + 0.02),
        base_angular_velocity=torch.full((batch, 3), fill + 0.03),
    )


def _sensor(batch=2, fill=1.5):
    return SensorSample(
        ranges_m=torch.full((batch, 41), fill),
        goal=torch.full((batch, 2), fill + 0.1),
        valid=torch.ones(batch, 41, dtype=torch.bool),
    )


def test_wrapper_produces_rsl_fields():
    wrap = SeaNavObservationWrapper(2, device=torch.device("cpu"))
    result = wrap.normal_reset(
        torch.arange(2), 0.0, _proprio(2), _sensor(2),
    )
    assert result.policy_obs.shape == (2, POLICY_OBS_DIM)
    assert set(result.actor_context) == {"safety_ranges_m", "safety_valid", "safety_age_s"}
    assert result.bad_masks.shape == (2,)
    assert result.bad_masks.dtype == torch.uint8


def test_no_hit_only_at_sensor_boundary():
    ranges = torch.tensor([[3.0, float("nan"), 0.5]])
    valid = torch.tensor([[True, True, True]])
    out_ranges, out_valid = encode_sensor_boundary(ranges, valid)
    assert out_valid[0, 0].item() is True
    assert out_valid[0, 1].item() is False
    assert out_valid[0, 2].item() is True
    assert out_ranges[0, 0].item() == pytest.approx(VALID_NO_HIT_RANGE_M)


def test_raw_safety_independent_of_policy_obs():
    wrap = SeaNavObservationWrapper(1, device=torch.device("cpu"))
    wrap.normal_reset(torch.tensor([0]), 0.0, _proprio(1), _sensor(1, fill=1.0))
    obs_before = wrap.policy_tick(0.02, _proprio(1), _sensor(1, fill=1.0)).policy_obs.clone()
    sensor = _sensor(1, fill=2.0)
    sensor.ranges_m[:] = 999.0  # would break policy encoding if used directly
    result = wrap.policy_tick(0.04, _proprio(1), sensor,
                              generator=torch.Generator().manual_seed(0))
    assert result.actor_context["safety_ranges_m"][0, 0].item() != 999.0
    assert torch.isfinite(result.policy_obs).all()
    assert result.policy_obs.shape == obs_before.shape


def test_all_invalid_row_is_ineligible():
    wrap = SeaNavObservationWrapper(1, device=torch.device("cpu"))
    sensor = _sensor(1)
    sensor.valid.zero_()
    wrap.normal_reset(torch.tensor([0]), 0.0, _proprio(1), sensor)
    result = wrap.policy_tick(
        0.02, _proprio(1), sensor, generator=torch.Generator().manual_seed(0),
    )
    assert result.bad_masks.item() == 1
    assert result.safety_reason_code.item() == RAW_SAFETY_REASON_ALL_INVALID


def test_isolate_bad_rows_before_forward():
    ctx = {
        "safety_ranges_m": torch.full((1, SEA_RAY_COUNT), float("nan")),
        "safety_valid": torch.ones(1, SEA_RAY_COUNT, dtype=torch.bool),
        "safety_age_s": torch.tensor([[0.05]]),
    }
    bad = torch.tensor([1], dtype=torch.uint8)
    safe = isolate_actor_context_for_forward(ctx, bad)
    assert torch.isfinite(safe["safety_ranges_m"]).all()
    assert not bool(safe["safety_valid"].any())

    mixed_ctx = {
        "safety_ranges_m": torch.cat(
            (ctx["safety_ranges_m"], torch.full((1, SEA_RAY_COUNT), 1.5)), dim=0
        ),
        "safety_valid": torch.cat(
            (ctx["safety_valid"], torch.ones(1, SEA_RAY_COUNT, dtype=torch.bool)), dim=0
        ),
        "safety_age_s": torch.tensor([[0.05], [0.02]]),
    }
    mixed_bad = torch.tensor([1, 0], dtype=torch.uint8)
    mixed_safe = isolate_actor_context_for_forward(mixed_ctx, mixed_bad)
    layer = UnicycleLookaheadLSECBFLayer(
        sea_nav_dashgo_candidate_platform_spec(),
        sea_nav_dashgo_raw_safety_spec(),
        EffectiveCommandEnvelopeSpec(
            profile_id="test",
            min_linear_velocity_m_s=-0.15,
            max_linear_velocity_m_s=0.30,
            max_abs_yaw_rate_rad_s=1.0,
            envelope_provenance="test",
        ),
    )
    angles = torch.tensor(
        sea_nav_dashgo_raw_safety_spec().ray_angles_rad,
        dtype=torch.float32,
    ).unsqueeze(0).expand(2, -1)
    good = 1
    projected, _ = layer.project_cbf_command_with_residual(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        0.02,
        mixed_safe["safety_ranges_m"][good:good + 1],
        angles[good:good + 1],
        mixed_safe["safety_valid"][good:good + 1],
        mixed_safe["safety_age_s"][good:good + 1],
        torch.ones(1, 1),
        sea_nav_dashgo_raw_safety_spec().manifest_sha256,
    )
    assert torch.isfinite(projected).all()
    assert not bool(mixed_safe["safety_valid"][0].any())


def test_changing_raw_with_fixed_policy_obs_affects_safety_status():
    wrap = SeaNavObservationWrapper(1, device=torch.device("cpu"))
    good_sensor = _sensor(1)
    wrap.normal_reset(torch.tensor([0]), 0.0, _proprio(1), good_sensor)
    good = wrap.policy_tick(0.02, _proprio(1), good_sensor,
                            generator=torch.Generator().manual_seed(0))
    assert good.safety_reason_code.item() == RAW_SAFETY_REASON_OK
    bad_sensor = _sensor(1)
    bad_sensor.ranges_m[:] = float("nan")
    wrap.normal_reset(torch.tensor([0]), 0.04, _proprio(1), bad_sensor)
    bad = wrap.policy_tick(0.04, _proprio(1), bad_sensor,
                           generator=torch.Generator().manual_seed(0))
    assert bad.safety_reason_code.item() == RAW_SAFETY_REASON_NONFINITE
    assert bad.policy_obs.shape == good.policy_obs.shape


def test_terminal_payload_captures_rows():
    wrap = SeaNavObservationWrapper(2, device=torch.device("cpu"))
    wrap.normal_reset(torch.arange(2), 0.0, _proprio(2), _sensor(2))
    result = wrap.policy_tick(
        0.02, _proprio(2), _sensor(2),
        generator=torch.Generator().manual_seed(0),
        terminal_rows=torch.tensor([1]),
    )
    assert "terminal_rows" in result.terminal_payload
    assert result.terminal_payload["terminal_rows"].tolist() == [1]
    assert result.terminal_payload["terminal_policy_obs"].shape[0] == 1
    assert result.terminal_payload["terminal_critic_obs"].shape == (
        1, result.policy_obs.shape[1],
    )
    torch.testing.assert_close(
        result.terminal_payload["terminal_critic_obs"],
        result.terminal_payload["terminal_policy_obs"],
    )


def test_timeout_bootstrap_uses_terminal_not_reset_value():
    wrap = SeaNavObservationWrapper(1, device=torch.device("cpu"))
    wrap.normal_reset(torch.tensor([0]), 0.0, _proprio(1), _sensor(1))
    terminal = wrap.policy_tick(
        0.02, _proprio(1, fill=1.0), _sensor(1),
        generator=torch.Generator().manual_seed(0),
        terminal_rows=torch.tensor([0]),
    )
    reset = wrap.normal_reset(
        torch.tensor([0]), 0.04, _proprio(1, fill=9.0), _sensor(1, fill=9.0),
    )
    assert not torch.allclose(
        terminal.terminal_payload["terminal_policy_obs"],
        reset.policy_obs,
    )
    infos = build_timeout_bootstrap_infos(
        batch_size=1,
        time_outs=torch.tensor([1.0]),
        terminal_bootstrap_values=torch.tensor([7.0]),
    )
    bundle = assemble_rsl_step_bundle(
        tick=terminal,
        next_tick=reset,
        time_outs=infos["time_outs"],
        terminal_bootstrap_values=infos["timeout_bootstrap_values"],
    )
    assert bundle.infos["timeout_bootstrap_values"].item() == pytest.approx(7.0)
    assert bundle.next_policy_obs is reset.policy_obs


def test_ppo_timeout_bootstrap_prefers_terminal_value_over_reset():
    try:
        from rsl_rl.algorithms.ppo import PPO
    except ImportError:
        pytest.skip("rsl_rl not importable")
    from actor_critic import SeaNavDiffDriveActorCritic  # noqa: E402

    actor = SeaNavDiffDriveActorCritic(
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[16, 8],
        encoder_hidden_dims=[16, 8],
    )
    ppo = PPO(actor, num_learning_epochs=1, num_mini_batches=1, device="cpu")
    obs = torch.randn(1, 550)
    ppo.init_storage(1, 1, (550,), (2,))
    ppo.act(obs, obs)
    ppo.transition.values[:] = 99.0
    next_obs = torch.randn(1, 550)
    reward = torch.zeros(1)
    ppo.process_env_step(
        next_obs,
        reward,
        torch.zeros(1),
        {
            "time_outs": torch.tensor([1.0]),
            "timeout_bootstrap_values": torch.tensor([[7.0]]),
        },
    )
    expected = ppo.gamma * 7.0
    assert ppo.storage.rewards[0, 0, 0].item() == pytest.approx(expected, rel=0, abs=1e-6)


def test_partial_invalid_ray_keeps_row_eligible():
    wrap = SeaNavObservationWrapper(1, device=torch.device("cpu"))
    sensor = _sensor(1)
    sensor.valid[0, 3] = False
    wrap.normal_reset(torch.tensor([0]), 0.0, _proprio(1), sensor)
    result = wrap.policy_tick(
        0.02, _proprio(1), sensor, generator=torch.Generator().manual_seed(0),
    )
    assert result.bad_masks.item() == 0
    assert result.safety_reason_code.item() == RAW_SAFETY_REASON_OK
    assert result.policy_obs[0, 12 + 3].item() == pytest.approx(math.log2(INVALID_RAY_POLICY_PLACEHOLDER_M))
