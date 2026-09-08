from dataclasses import FrozenInstanceError, replace
import json

import pytest
import sea_nav_core

from sea_nav_core import (
    ADAPTATION_ID, DASHGO_PLATFORM_PROVENANCE,
    DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE,
    DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE,
    DASHGO_SIM_CAMERA_RANGE_PROVENANCE,
    ISAACLAB_CAMERA_DISTANCE_TO_CAMERA, ISAACLAB_CAMERA_DISTANCE_TO_IMAGE_PLANE,
    ROS_LASERSCAN_RADIAL_RANGE,
    RESULT_CLASSIFICATION, SAFETY_SEMANTICS, ActionSpec, DifferentialDrivePlatformSpec,
    EffectiveCommandEnvelopeSpec, ObservationSpec, RawSafetyObservationSpec,
    resolve_ablation_profile,
)


def platform():
    return DifferentialDrivePlatformSpec(footprint_radius_m=0.203, lookahead_distance_m=0.2)


def raw_safety():
    return RawSafetyObservationSpec(
        sensor_frame="front_lidar",
        ray_angles_rad=(-1.0, 0.0, 1.0),
        max_sensor_age_s=0.1,
        range_min_m=0.15,
        range_definition=ROS_LASERSCAN_RADIAL_RANGE,
        range_parameter_provenance=DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE,
    )


def forward_envelope():
    return EffectiveCommandEnvelopeSpec(
        profile_id="dashgo_forward_sensor_experiment_v1",
        min_linear_velocity_m_s=0.0,
        max_linear_velocity_m_s=0.3,
        max_abs_yaw_rate_rad_s=1.0,
        envelope_provenance=DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE,
    )


@pytest.mark.parametrize("spec", [
    platform(), raw_safety(), forward_envelope(), ActionSpec(),
    ObservationSpec("policy_action", "normalizer-sha256"),
])
def test_exact_manifest_roundtrip_identity_and_hash(spec):
    payload = json.loads(json.dumps(spec.to_manifest(), allow_nan=False))
    restored = type(spec).from_manifest(payload)
    assert restored == spec
    assert restored.manifest_sha256 == spec.manifest_sha256
    assert len(spec.manifest_sha256) == 64
    assert payload["adaptation_id"] == ADAPTATION_ID == "dashgo_diffdrive_transfer_v1"
    assert payload["result_classification"] == RESULT_CLASSIFICATION == "cross_platform_method_adaptation"
    assert payload["safety_semantics"] == SAFETY_SEMANTICS == "differentiable_safety_bias"
    with pytest.raises(FrozenInstanceError):
        spec.kind = "changed"


@pytest.mark.parametrize("spec", [
    platform(), raw_safety(), forward_envelope(), ActionSpec(),
    ObservationSpec("executed_command", "n1"),
])
@pytest.mark.parametrize("mutation", ["extra", "missing", "parameter_extra", "parameter_missing", "version", "identity", "kind"])
def test_manifests_reject_ambiguous_or_forged_fields(spec, mutation):
    payload = spec.to_manifest()
    if mutation == "extra":
        payload["unknown"] = 1
    elif mutation == "missing":
        del payload["kind"]
    elif mutation == "parameter_extra":
        payload["parameters"]["unknown"] = 1
    elif mutation == "parameter_missing":
        payload["parameters"].pop(next(iter(payload["parameters"])))
    elif mutation == "version":
        payload["schema_version"] = True
    elif mutation == "identity":
        payload["adaptation_id"] = "paper_v1"
    else:
        payload["kind"] = "hard_safety"
    with pytest.raises(ValueError):
        type(spec).from_manifest(payload)


@pytest.mark.parametrize("field", [
    "footprint_radius_m", "lookahead_distance_m", "max_forward_m_s",
    "max_reverse_m_s", "max_yaw_rad_s", "policy_dt_s", "wheel_radius_m",
    "track_width_m", "max_wheel_velocity_rad_s", "max_linear_acceleration_mps2",
    "max_angular_acceleration_radps2",
])
@pytest.mark.parametrize("value", [0, -1.0, True, float("nan"), float("inf"), "1"])
def test_positive_platform_numbers(field, value):
    with pytest.raises(ValueError):
        replace(platform(), **{field: value})


@pytest.mark.parametrize("field,value", [
    ("safety_margin_m", -1), ("sensor_x_m", float("nan")), ("sensor_y_m", True),
    ("sensor_yaw_rad", float("inf")), ("base_frame", ""), ("base_frame", " base_link"),
    ("plant_parameter_provenance", ""), ("plant_parameter_provenance", " source"),
])
def test_other_platform_validation(field, value):
    with pytest.raises(ValueError):
        replace(platform(), **{field: value})


def test_platform_defaults_provenance_and_every_plant_limit_change_identity():
    spec = platform()
    assert (spec.wheel_radius_m, spec.track_width_m) == (0.0632, 0.342)
    assert spec.max_wheel_velocity_rad_s == 5.0
    assert (spec.max_linear_acceleration_mps2,
            spec.max_angular_acceleration_radps2) == (1.0, 0.6)
    assert spec.plant_parameter_provenance == DASHGO_PLATFORM_PROVENANCE
    assert replace(spec, sensor_x_m=0.1).manifest_sha256 != spec.manifest_sha256
    for field in (
        "wheel_radius_m", "track_width_m", "max_wheel_velocity_rad_s",
        "max_linear_acceleration_mps2",
        "max_angular_acceleration_radps2", "plant_parameter_provenance",
    ):
        value = "another-pinned-source" if field == "plant_parameter_provenance" else getattr(spec, field) * 1.01
        assert replace(spec, **{field: value}).manifest_sha256 != spec.manifest_sha256
    assert replace(spec, safety_margin_m=0).safety_margin_m == 0
    assert replace(spec, safety_margin_m=0).manifest_sha256 == replace(spec, safety_margin_m=0.0).manifest_sha256
    assert ObservationSpec("policy_action", "n1", fov_deg=180).manifest_sha256 == ObservationSpec("policy_action", "n1", fov_deg=180.0).manifest_sha256


def test_effective_command_envelope_separates_forward_experiment_from_reverse_capability():
    envelope_type = getattr(sea_nav_core, "EffectiveCommandEnvelopeSpec")
    forward = forward_envelope()
    reverse_enabled = envelope_type(
        profile_id="dashgo_reverse_capability_runtime_v1",
        min_linear_velocity_m_s=-0.15,
        max_linear_velocity_m_s=0.3,
        max_abs_yaw_rate_rad_s=1.0,
        envelope_provenance="declared reverse-capability runtime profile",
    )
    assert platform().max_reverse_m_s == 0.15
    assert forward.min_linear_velocity_m_s == 0.0
    assert reverse_enabled.min_linear_velocity_m_s == -0.15
    assert forward.manifest_sha256 != reverse_enabled.manifest_sha256
    assert type(forward).from_manifest(forward.to_manifest()) == forward
    with pytest.raises(ValueError):
        replace(forward, min_linear_velocity_m_s=0.31)


@pytest.mark.parametrize("field,value", [
    ("profile_id", ""), ("profile_id", " profile"),
    ("min_linear_velocity_m_s", True),
    ("min_linear_velocity_m_s", float("nan")),
    ("max_linear_velocity_m_s", 0),
    ("max_abs_yaw_rate_rad_s", float("inf")),
    ("envelope_provenance", ""),
])
def test_effective_command_envelope_rejects_ambiguous_values(field, value):
    with pytest.raises(ValueError):
        replace(forward_envelope(), **{field: value})


@pytest.mark.parametrize("field,value", [
    ("sensor_frame", ""), ("sensor_frame", " lidar"),
    ("ray_angles_rad", ()), ("ray_angles_rad", (0.0, float("nan"))),
    ("ray_angles_rad", (True,)), ("max_sensor_age_s", 0),
    ("max_sensor_age_s", float("inf")), ("range_min_m", 0), ("range_max_m", -1),
    ("range_definition", "anonymous_depth"), ("range_parameter_provenance", ""),
    ("contract_id", "policy_observation_v1"), ("range_units", "normalized"),
    ("angle_units", "deg"), ("age_units", "ms"),
    ("ranges_shape", "[B,246]"), ("angles_shape", "implicit_fov"),
    ("validity_shape", "none"), ("sensor_age_shape", "scalar"),
    ("ray_order", "unknown"), ("angle_semantics", "base_frame"),
    ("validity_semantics", "zero_is_valid"), ("age_semantics", "timestamp"),
    ("normalized", True),
])
def test_raw_safety_contract_rejects_wrong_units_shapes_version_and_semantics(field, value):
    with pytest.raises(ValueError):
        replace(raw_safety(), **{field: value})


def test_raw_safety_shape_and_actual_angle_geometry_are_in_identity():
    spec = raw_safety()
    assert spec.num_rays == 3 and spec.flattened_dim == 10
    assert replace(spec, sensor_frame="rear_lidar").manifest_sha256 != spec.manifest_sha256
    assert replace(spec, ray_angles_rad=(-1.0, 0.1, 1.0)).manifest_sha256 != spec.manifest_sha256
    restored = RawSafetyObservationSpec.from_manifest(spec.to_manifest())
    assert isinstance(restored.ray_angles_rad, tuple) and restored == spec


def test_raw_safety_identity_binds_minimum_range_definition_and_provenance():
    common = {
        "sensor_frame": "front_lidar",
        "ray_angles_rad": (-1.0, 0.0, 1.0),
        "max_sensor_age_s": 0.1,
        "range_max_m": 12.0,
    }
    real_scan = RawSafetyObservationSpec(
        **common,
        range_min_m=0.15,
        range_definition=ROS_LASERSCAN_RADIAL_RANGE,
        range_parameter_provenance=DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE,
    )
    with pytest.raises(ValueError, match="range_definition"):
        RawSafetyObservationSpec(
            **common,
            range_min_m=0.1,
            range_definition=ISAACLAB_CAMERA_DISTANCE_TO_IMAGE_PLANE,
            range_parameter_provenance=DASHGO_SIM_CAMERA_RANGE_PROVENANCE,
        )
    simulated_camera = RawSafetyObservationSpec(
        **common,
        range_min_m=0.1,
        range_definition=ISAACLAB_CAMERA_DISTANCE_TO_CAMERA,
        range_parameter_provenance=(
            DASHGO_SIM_CAMERA_RANGE_PROVENANCE
            + ";runtime_override=data_type=distance_to_camera"
        ),
    )
    assert real_scan.manifest_sha256 != simulated_camera.manifest_sha256
    for field, value in (
        ("range_min_m", 0.1),
        ("range_definition", ISAACLAB_CAMERA_DISTANCE_TO_CAMERA),
        ("range_parameter_provenance", "different-pinned-source"),
    ):
        assert replace(real_scan, **{field: value}).manifest_sha256 != real_scan.manifest_sha256
    for field, value in (
        ("range_min_m", 0.0),
        ("range_min_m", 12.0),
        ("range_definition", "anonymous_depth"),
        ("range_parameter_provenance", ""),
    ):
        with pytest.raises(ValueError):
            replace(real_scan, **{field: value})


@pytest.mark.parametrize("field,value", [
    ("num_rays", 41), ("num_rays", True), ("history_frames", 10), ("fov_deg", 240),
    ("range_max_m", 3), ("history_layout", "frame_major"), ("term_order", "go2"),
    ("last_action_stage", "unknown"), ("normalizer_id", ""), ("contract_id", "go2"),
])
def test_observation_abi_is_not_dimension_only(field, value):
    with pytest.raises(ValueError):
        replace(ObservationSpec("policy_action", "n1"), **{field: value})


def test_observation_action_dimensions():
    assert ObservationSpec("policy_action", "n1").obs_dim == 246
    assert ActionSpec().action_dim == 2
    for field in ActionSpec.__dataclass_fields__:
        if field != "kind":
            with pytest.raises(ValueError):
                replace(ActionSpec(), **{field: "unsupported"})


@pytest.mark.parametrize("name,expected", [
    ("full", (True, True, True)),
    ("without_acsi", (False, True, True)),
    ("without_shield", (True, False, True)),
    ("without_lreg", (True, True, False)),
])
def test_only_one_config_axis_changes_each_ablation(name, expected):
    profile = resolve_ablation_profile(name)
    assert (profile.acsi_enabled, profile.shield_enabled, profile.lreg_enabled) == expected
    assert profile.alpha_min == 0.1
    assert profile.shield_loss_weight_owner == "sea_nav_core.paper_v1_shield_loss"
    assert profile.lreg_loss_weight_owner == "sea_nav_core.paper_v1_lreg_loss"
    assert not hasattr(profile, "lambda_shield")
    assert type(profile).from_manifest(profile.to_manifest()) == profile


@pytest.mark.parametrize("name", ["baseline", "cbf_only", "FULL", "", None, True])
def test_no_unregistered_profiles(name):
    with pytest.raises(ValueError):
        resolve_ablation_profile(name)
