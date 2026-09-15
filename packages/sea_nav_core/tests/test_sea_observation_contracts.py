# SPDX-License-Identifier: MIT
"""Golden tests for the SEA 550-D / Normal / 41-ray core contracts (G2).

These pin the frozen paper adaptation contract: frame field order, history
movement, both reset bootstraps, raw-safety isolation and fail-closed rules,
legacy-ABI rejection, manifest tampering, and the four-profile switch set.
"""

import json
import math

import pytest
import torch

from sea_nav_core import (
    FRAME_COUNT,
    FRAME_DIM,
    INVALID_RAY_POLICY_PLACEHOLDER_M,
    POLICY_OBS_DIM,
    RAW_SAFETY_REASON_ALL_INVALID,
    RAW_SAFETY_REASON_DOMAIN,
    RAW_SAFETY_REASON_NEGATIVE_AGE,
    RAW_SAFETY_REASON_NONFINITE,
    RAW_SAFETY_REASON_OK,
    RAW_SAFETY_REASON_STALE_AGE,
    ActionSpec,
    ObservationSpec,
    PolicyObservationHistory,
    RawSafetyObservationSpec,
    SeaNavActionSpec,
    SeaNavDashgoContractSet,
    SeaNavObservationSpec,
    build_policy_frame,
    policy_range_channels,
    policy_range_hold,
    raw_safety_row_status,
    resolve_ablation_profile,
    sea_nav_dashgo_candidate_platform_spec,
    sea_nav_dashgo_raw_safety_spec,
    sea_ray_angles_deg,
    sea_ray_angles_rad,
)


# ---------------------------------------------------------------------------
# Explicit 41-ray pattern.
# ---------------------------------------------------------------------------

def test_ray_pattern_is_exactly_41_rays_120_to_120_inclusive_at_6_degrees():
    degrees = sea_ray_angles_deg()
    assert len(degrees) == 41
    assert degrees[0] == -120.0
    assert degrees[-1] == 120.0
    assert degrees[20] == 0.0
    steps = {
        round(degrees[i + 1] - degrees[i], 12) for i in range(len(degrees) - 1)
    }
    assert steps == {6.0}


def test_ray_radians_are_the_degree_pattern():
    radians = sea_ray_angles_rad()
    degrees = sea_ray_angles_deg()
    assert len(radians) == 41
    for rad, deg in zip(radians, degrees):
        assert rad == pytest.approx(math.radians(deg), rel=0.0, abs=0.0)


def test_frozen_safety_spec_matches_the_41_ray_geometry_and_domain():
    spec = sea_nav_dashgo_raw_safety_spec()
    assert spec.num_rays == 41
    assert spec.range_min_m == pytest.approx(0.1)
    assert spec.range_max_m == pytest.approx(3.0)
    assert spec.max_sensor_age_s == pytest.approx(0.18)
    assert list(spec.ray_angles_rad) == list(sea_ray_angles_rad())
    assert spec.normalized is False


# ---------------------------------------------------------------------------
# 55-value frame and 550-D history.
# ---------------------------------------------------------------------------

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


def test_frame_field_order_is_frozen_with_distinctive_values():
    frame = _frame(batch=1, fill=0.0)
    assert frame.shape == (1, FRAME_DIM)
    assert frame[0, 0:3].eq(0.0).all()
    assert frame[0, 3].item() == pytest.approx(0.0)
    assert frame[0, 4].item() == 0.0, "middle channel of [v, 0, omega] is a hard zero"
    assert frame[0, 5].eq(0.1).all()
    assert frame[0, 6:9].eq(0.2).all()
    assert frame[0, 9:12].eq(0.3).all()
    assert frame[0, 53:55].eq(0.4).all()
    assert frame[0, 12:53].eq(math.log2(1.5)).all()


def test_frame_rejects_nonzero_lateral_command_channel():
    bad = torch.zeros(1, 3)
    bad[0, 0] = 0.2
    bad[0, 1] = 0.05  # lateral velocity must not enter the frame
    bad[0, 2] = 0.3
    with pytest.raises(ValueError, match="v, 0, omega"):
        build_policy_frame(
            torch.zeros(1, 3), bad, torch.zeros(1, 3), torch.zeros(1, 3),
            torch.full((1, 41), 1.0), torch.zeros(1, 2),
        )


def test_policy_range_channels_are_log2_of_clamped_metric_ranges():
    ranges = torch.full((1, 41), 1.0)
    for column, value in ((0, 0.05), (1, 1.0), (2, 3.0), (3, 5.0)):
        ranges[0, column] = value
    channels = policy_range_channels(ranges)
    assert channels.shape == (1, 41)
    expected = {0: math.log2(0.1), 1: math.log2(1.0), 2: math.log2(3.0),
                3: math.log2(3.0)}
    for column, want in expected.items():
        assert channels[0, column].item() == pytest.approx(want)
    assert channels[0, 4].item() == pytest.approx(math.log2(1.0))


def test_history_flatten_is_frame_major_oldest_to_newest_and_moves():
    history = PolicyObservationHistory(1, torch.float32, torch.device("cpu"))
    history.bootstrap_normal_reset(_frame(batch=1, fill=1.0))
    for value in (2.0, 3.0):
        history.push(_frame(batch=1, fill=value))
    flat = history.flatten()
    assert flat.shape == (1, POLICY_OBS_DIM)
    # gravity slot of each frame carries its fill value
    slots = [flat[0, index * FRAME_DIM].item() for index in range(FRAME_COUNT)]
    assert slots == [1.0] * 8 + [2.0, 3.0]


def test_normal_reset_bootstrap_fills_all_ten_slots_atomically():
    history = PolicyObservationHistory(2, torch.float32, torch.device("cpu"))
    history.bootstrap_normal_reset(_frame(batch=2, fill=7.0))
    frames = history.frames()
    assert frames.shape == (2, FRAME_COUNT, FRAME_DIM)
    for row in range(2):
        for slot in range(FRAME_COUNT):
            assert frames[row, slot, 0].item() == pytest.approx(7.0)
            assert frames[row, slot, 3].item() == pytest.approx(7.0)
            assert frames[row, slot, 4].item() == 0.0
            assert frames[row, slot, 12].item() == pytest.approx(math.log2(1.5))


def test_replay_bootstrap_restores_exact_history_without_refill():
    history = PolicyObservationHistory(1, torch.float32, torch.device("cpu"))
    stored = torch.arange(POLICY_OBS_DIM, dtype=torch.float32).view(1, POLICY_OBS_DIM)
    history.bootstrap_replay_restore(stored)
    assert torch.equal(history.flatten(), stored)
    frames = history.frames()
    for slot in range(FRAME_COUNT):
        expected = stored[0, slot * FRAME_DIM:(slot + 1) * FRAME_DIM]
        assert torch.equal(frames[0, slot], expected)
    # replay bootstrap is not a refill: pushing continues from restored state
    history.push(_frame(batch=1, fill=9.0))
    assert history.flatten()[0, FRAME_DIM * 8].item() == pytest.approx(
        stored[0, FRAME_DIM * 9].item()
    )
    assert history.flatten()[0, FRAME_DIM * 9].item() == pytest.approx(9.0)


def test_flatten_and_frames_return_copies_not_live_views():
    history = PolicyObservationHistory(1, torch.float32, torch.device("cpu"))
    history.bootstrap_normal_reset(_frame(batch=1, fill=1.0))
    flat = history.flatten()
    frames = history.frames()
    flat_before = flat.clone()
    frames_before = frames.clone()
    history.push(_frame(batch=1, fill=2.0))
    assert torch.equal(flat, flat_before), "retained observation must not mutate"
    assert torch.equal(frames, frames_before)


def test_invalid_ray_garbage_payload_is_retained_as_held_value():
    arrived = torch.full((1, 41), 1.5)
    arrived[0, 5] = float("inf")
    valid = torch.ones(1, 41, dtype=torch.bool)
    valid[0, 5] = False
    held = torch.full((1, 41), 0.7)
    result = policy_range_hold(arrived, valid, held)
    assert result[0, 5].item() == pytest.approx(0.7)
    valid_ray_inf = arrived.clone()
    arrived2 = torch.full((1, 41), float("inf"))
    with pytest.raises(ValueError, match="valid must be finite"):
        policy_range_hold(arrived2, torch.ones(1, 41, dtype=torch.bool), held)
    _ = valid_ray_inf


def test_exported_constants_stay_pinned_to_the_frozen_safety_spec():
    from sea_nav_core import (
        MAX_SENSOR_AGE_S as exported_max_age,
        VALID_NO_HIT_RANGE_M as exported_no_hit,
    )
    assert exported_max_age == pytest.approx(_SPEC.max_sensor_age_s)
    assert exported_no_hit == pytest.approx(_SPEC.range_max_m)
    assert exported_no_hit == pytest.approx(3.0)


def test_bound_contract_sha256_is_deterministic_and_fail_closed():
    spec = sea_nav_dashgo_raw_safety_spec()
    platform = sea_nav_dashgo_candidate_platform_spec()
    first = SeaNavDashgoContractSet.bound_contract_sha256(
        SeaNavObservationSpec(), SeaNavActionSpec(), spec, platform
    )
    second = SeaNavDashgoContractSet.bound_contract_sha256(
        SeaNavObservationSpec(), SeaNavActionSpec(), spec, platform
    )
    assert first == second and len(first) == 64
    import dataclasses
    with pytest.raises(ValueError):
        SeaNavDashgoContractSet.bound_contract_sha256(
            SeaNavObservationSpec(), SeaNavActionSpec(), spec,
            dataclasses.replace(platform, wheel_radius_m=0.0625),
        )


def test_unbootstrapped_history_fails_closed_on_push_and_flatten():
    history = PolicyObservationHistory(1, torch.float32, torch.device("cpu"))
    with pytest.raises(ValueError, match="bootstrapped"):
        history.flatten()
    with pytest.raises(ValueError, match="bootstrapped"):
        history.push(_frame(batch=1))


def test_history_rejects_nonfinite_and_mismatched_frames():
    history = PolicyObservationHistory(1, torch.float32, torch.device("cpu"))
    bad = _frame(batch=1)
    bad = bad.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        history.bootstrap_normal_reset(bad)
    with pytest.raises(ValueError, match="shape"):
        history.bootstrap_replay_restore(torch.zeros(1, 549))


# ---------------------------------------------------------------------------
# Raw-safety isolation and fail-closed rules.
# ---------------------------------------------------------------------------

_SPEC = sea_nav_dashgo_raw_safety_spec()


def test_valid_no_hit_and_partial_invalid_rows_are_ok():
    ranges = torch.tensor([[3.0] * 41, [0.5] * 41])
    valid = torch.ones(2, 41, dtype=torch.bool)
    valid[1, 7] = False
    age = torch.zeros(2, 1)
    status = raw_safety_row_status(_SPEC, ranges, valid, age)
    assert status.ok.tolist() == [True, True]
    assert status.reason_code.tolist() == [RAW_SAFETY_REASON_OK, RAW_SAFETY_REASON_OK]
    assert status.any_invalid_ray.tolist() == [False, True]
    assert status.all_invalid.tolist() == [False, False]


@pytest.mark.parametrize("bad_ranges,bad_valid,bad_age,reason", [
    (float("nan"), True, 0.0, RAW_SAFETY_REASON_NONFINITE),
    (float("inf"), True, 0.0, RAW_SAFETY_REASON_NONFINITE),
    (1.0, True, float("nan"), RAW_SAFETY_REASON_NONFINITE),
    (0.05, True, 0.0, RAW_SAFETY_REASON_DOMAIN),
    (3.5, True, 0.0, RAW_SAFETY_REASON_DOMAIN),
    (1.0, True, -0.01, RAW_SAFETY_REASON_NEGATIVE_AGE),
    (1.0, True, 0.19, RAW_SAFETY_REASON_STALE_AGE),
])
def test_fail_closed_conditions(bad_ranges, bad_valid, bad_age, reason):
    ranges = torch.full((1, 41), 1.0)
    valid = torch.ones(1, 41, dtype=torch.bool)
    age = torch.zeros(1, 1)
    ranges[0, 3] = bad_ranges
    valid[0, 3] = bad_valid
    age[0, 0] = bad_age
    status = raw_safety_row_status(_SPEC, ranges, valid, age)
    assert status.ok.tolist() == [False]
    assert status.reason_code.tolist() == [reason]


def test_all_invalid_row_fails_closed_not_free_space():
    ranges = torch.full((1, 41), 1.0)
    valid = torch.zeros(1, 41, dtype=torch.bool)
    status = raw_safety_row_status(_SPEC, ranges, valid, torch.zeros(1, 1))
    assert status.ok.tolist() == [False]
    assert status.reason_code.tolist() == [RAW_SAFETY_REASON_ALL_INVALID]


def test_invalid_rays_do_not_affect_domain_checks_of_valid_rays():
    ranges = torch.full((1, 41), 1.0)
    ranges[0, 2] = 99.0  # invalid ray carries garbage
    valid = torch.ones(1, 41, dtype=torch.bool)
    valid[0, 2] = False
    status = raw_safety_row_status(_SPEC, ranges, valid, torch.zeros(1, 1))
    assert status.ok.tolist() == [True]
    assert status.any_invalid_ray.tolist() == [True]


def test_policy_range_hold_replaces_valid_and_retains_invalid():
    arrived = torch.full((1, 41), 2.0)
    arrived[0, 0] = 1.0
    arrived[0, 2] = 2.0
    valid = torch.ones(1, 41, dtype=torch.bool)
    valid[0, 1] = False
    held = torch.full((1, 41), 0.6)
    held[0, 0] = 0.5
    held[0, 2] = 0.7
    result = policy_range_hold(arrived, valid, held)
    assert result[0, 0].item() == pytest.approx(1.0)
    assert result[0, 1].item() == pytest.approx(0.6)
    assert result[0, 2].item() == pytest.approx(2.0)
    assert result[0, 3].item() == pytest.approx(2.0)


def test_reset_placeholder_is_conservative_and_never_a_valid_no_hit():
    assert INVALID_RAY_POLICY_PLACEHOLDER_M == pytest.approx(0.1)
    # A reset-time invalid channel without history uses the 0.1 m placeholder
    # in the *policy* channels; the safety path keeps validity false.
    arrived = torch.full((1, 41), 3.0)
    valid = torch.zeros(1, 41, dtype=torch.bool)
    held = torch.full((1, 41), INVALID_RAY_POLICY_PLACEHOLDER_M)
    policy_channels = policy_range_channels(policy_range_hold(arrived, valid, held))
    assert policy_channels[0].eq(math.log2(0.1)).all()


def test_policy_observation_cannot_substitute_raw_safety_inputs():
    frame = _frame(batch=1)
    channels = frame[0, 12:53]
    # The channels are log2-clamped held values; after an invalid-ray hold the
    # policy channel can diverge from the raw metric packet, so the safety
    # layer must consume its own structured inputs.
    arrived = torch.full((1, 41), 2.5)
    valid = torch.zeros(1, 41, dtype=torch.bool)
    held = torch.full((1, 41), 0.8)
    held_channels = policy_range_channels(policy_range_hold(arrived, valid, held))
    assert not held_channels.eq(policy_range_channels(arrived)).all()
    assert channels.shape[-1] == 41  # frame carries only transformed channels


def test_actor_input_validation_enforces_shapes_and_dtypes():
    from sea_nav_core import SeaNavActorInput

    good = SeaNavActorInput(
        policy_obs=torch.zeros(2, POLICY_OBS_DIM),
        safety_ranges_m=torch.full((2, 41), 1.0),
        safety_valid=torch.ones(2, 41, dtype=torch.bool),
        safety_age_s=torch.zeros(2, 1),
    )
    good.validate()
    with pytest.raises(ValueError):
        SeaNavActorInput(
            policy_obs=torch.zeros(2, 549),
            safety_ranges_m=torch.full((2, 41), 1.0),
            safety_valid=torch.ones(2, 41, dtype=torch.bool),
            safety_age_s=torch.zeros(2, 1),
        ).validate()
    with pytest.raises(ValueError):
        SeaNavActorInput(
            policy_obs=torch.zeros(2, POLICY_OBS_DIM),
            safety_ranges_m=torch.full((2, 40), 1.0),
            safety_valid=torch.ones(2, 40, dtype=torch.bool),
            safety_age_s=torch.zeros(2, 1),
        ).validate()


# ---------------------------------------------------------------------------
# Legacy ABI rejection and manifest fail-closed behavior.
# ---------------------------------------------------------------------------

def test_contract_set_binds_the_sea_contracts():
    spec = sea_nav_dashgo_raw_safety_spec()
    platform = sea_nav_dashgo_candidate_platform_spec()
    binding = SeaNavDashgoContractSet.bind(
        SeaNavObservationSpec(), SeaNavActionSpec(), spec, platform
    )
    payload = json.loads(json.dumps(binding.to_manifest()))
    assert payload["kind"] == "sea_nav_dashgo_contract_set_v1"
    parameters = payload["parameters"]
    assert parameters["result_classification"] == "cross_platform_method_adaptation"
    assert parameters["validation_identity"] == "simulation_surrogate_candidate"
    assert parameters["algorithm_profile"] == "sea_nav_paper_method_operational_v1"
    assert parameters["platform_profile"] == "dashgo_d1_primitive_candidate_v1"


def test_contract_set_rejects_legacy_246d_observation_abi():
    legacy_obs = ObservationSpec(
        last_action_stage="policy_action", normalizer_id="legacy_front_180"
    )
    assert legacy_obs.obs_dim == 246
    with pytest.raises(ValueError, match="246-D"):
        SeaNavDashgoContractSet.bind(
            legacy_obs, SeaNavActionSpec(),
            sea_nav_dashgo_raw_safety_spec(),
            sea_nav_dashgo_candidate_platform_spec(),
        )


def test_contract_set_rejects_legacy_bounded_tanh_action_abi():
    legacy_action = ActionSpec()
    assert legacy_action.distribution == "bounded_tanh_gaussian"
    with pytest.raises(ValueError, match="bounded-tanh"):
        SeaNavDashgoContractSet.bind(
            SeaNavObservationSpec(), legacy_action,
            sea_nav_dashgo_raw_safety_spec(),
            sea_nav_dashgo_candidate_platform_spec(),
        )


def test_contract_set_rejects_non_41_ray_or_wrong_domain_safety_specs():
    wrong_rays = RawSafetyObservationSpec(
        sensor_frame="dashgo_sim_lidar",
        ray_angles_rad=tuple(0.1 * i for i in range(41)),
        max_sensor_age_s=0.18,
        range_min_m=0.1,
        range_max_m=3.0,
        range_definition="ros_laserscan_radial_range",
        range_parameter_provenance="tampered",
    )
    with pytest.raises(ValueError, match="41-ray"):
        SeaNavDashgoContractSet.bind(
            SeaNavObservationSpec(), SeaNavActionSpec(), wrong_rays,
            sea_nav_dashgo_candidate_platform_spec(),
        )
    long_range = RawSafetyObservationSpec(
        sensor_frame="dashgo_sim_lidar",
        ray_angles_rad=sea_ray_angles_rad(),
        max_sensor_age_s=0.18,
        range_min_m=0.1,
        range_max_m=12.0,
        range_definition="ros_laserscan_radial_range",
        range_parameter_provenance="legacy 12 m",
    )
    with pytest.raises(ValueError, match="41-ray"):
        SeaNavDashgoContractSet.bind(
            SeaNavObservationSpec(), SeaNavActionSpec(), long_range,
            sea_nav_dashgo_candidate_platform_spec(),
        )


def test_contract_set_rejects_modified_dashgo_candidate_geometry():
    platform = sea_nav_dashgo_candidate_platform_spec()
    import dataclasses
    tampered = dataclasses.replace(platform, wheel_radius_m=0.0625)
    with pytest.raises(ValueError, match="candidate geometry"):
        SeaNavDashgoContractSet.bind(
            SeaNavObservationSpec(), SeaNavActionSpec(),
            sea_nav_dashgo_raw_safety_spec(), tampered,
        )


def test_sea_observation_spec_rejects_abi_changes():
    with pytest.raises(ValueError):
        SeaNavObservationSpec(history_frames=3)
    with pytest.raises(ValueError):
        SeaNavObservationSpec(ray_count=72)
    with pytest.raises(ValueError):
        SeaNavObservationSpec(raw_safety_reconstruction_forbidden=False)


def test_sea_action_spec_freezes_normal_contract_and_dashgo_bounds():
    spec = SeaNavActionSpec()
    assert spec.distribution == "diagonal_normal"
    assert spec.initial_std == pytest.approx(1.5)
    assert spec.log_prob_stage == "policy_action"
    assert spec.cbf_applies_to == "distribution_mean"
    assert spec.range_loss_lower == (-0.15, -1.0)
    assert spec.range_loss_upper == (0.3, 1.0)
    with pytest.raises(ValueError):
        SeaNavActionSpec(tanh_forbidden=False)
    with pytest.raises(ValueError):
        SeaNavActionSpec(distribution="bounded_tanh_gaussian")
    with pytest.raises(ValueError):
        SeaNavActionSpec(initial_std=0.5)


def test_manifest_tampering_fails_closed_on_roundtrip():
    spec = SeaNavObservationSpec()
    manifest = spec.to_manifest()
    assert SeaNavObservationSpec.from_manifest(manifest) == spec
    tampered = json.loads(json.dumps(manifest))
    tampered["parameters"]["history_frames"] = 3
    with pytest.raises(ValueError):
        SeaNavObservationSpec.from_manifest(tampered)
    tampered_action = json.loads(json.dumps(SeaNavActionSpec().to_manifest()))
    tampered_action["parameters"]["initial_std"] = 0.9
    with pytest.raises(ValueError):
        SeaNavActionSpec.from_manifest(tampered_action)


def test_four_profiles_differ_only_by_the_named_switch():
    flags = {}
    for name in ("full", "without_acsi", "without_shield", "without_lreg"):
        profile = resolve_ablation_profile(name)
        flags[name] = (profile.acsi_enabled, profile.shield_enabled,
                       profile.lreg_enabled)
    assert flags["full"] == (True, True, True)
    assert flags["without_acsi"] == (False, True, True)
    assert flags["without_shield"] == (True, False, True)
    assert flags["without_lreg"] == (True, True, False)
    # each non-full profile differs from full in exactly one component
    for name in ("without_acsi", "without_shield", "without_lreg"):
        differences = sum(
            1 for a, b in zip(flags["full"], flags[name]) if a != b
        )
        assert differences == 1
    with pytest.raises(ValueError):
        resolve_ablation_profile("without_everything")
