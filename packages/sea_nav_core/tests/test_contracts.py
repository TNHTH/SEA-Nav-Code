from dataclasses import FrozenInstanceError, replace
import json

import pytest

from sea_nav_core import (
    ADAPTATION_ID, RESULT_CLASSIFICATION, SAFETY_SEMANTICS, ActionSpec,
    DifferentialDrivePlatformSpec, ObservationSpec, resolve_ablation_profile,
)


def platform():
    return DifferentialDrivePlatformSpec(footprint_radius_m=0.203, lookahead_distance_m=0.2)


@pytest.mark.parametrize("spec", [platform(), ActionSpec(), ObservationSpec("policy_action", "normalizer-sha256")])
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


@pytest.mark.parametrize("spec", [platform(), ActionSpec(), ObservationSpec("executed_command", "n1")])
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


@pytest.mark.parametrize("field", ["footprint_radius_m", "lookahead_distance_m", "max_forward_m_s", "max_reverse_m_s", "max_yaw_rad_s", "policy_dt_s"])
@pytest.mark.parametrize("value", [0, -1.0, True, float("nan"), float("inf"), "1"])
def test_positive_platform_numbers(field, value):
    with pytest.raises(ValueError):
        replace(platform(), **{field: value})


@pytest.mark.parametrize("field,value", [
    ("safety_margin_m", -1), ("sensor_x_m", float("nan")), ("sensor_y_m", True),
    ("sensor_yaw_rad", float("inf")), ("base_frame", ""), ("base_frame", " base_link"),
])
def test_other_platform_validation(field, value):
    with pytest.raises(ValueError):
        replace(platform(), **{field: value})


def test_geometry_changes_manifest_hash_and_limits_do_not_claim_projection():
    spec = platform()
    assert replace(spec, sensor_x_m=0.1).manifest_sha256 != spec.manifest_sha256
    assert replace(spec, safety_margin_m=0).safety_margin_m == 0
    assert replace(spec, safety_margin_m=0).manifest_sha256 == replace(spec, safety_margin_m=0.0).manifest_sha256
    assert ObservationSpec("policy_action", "n1", fov_deg=180).manifest_sha256 == ObservationSpec("policy_action", "n1", fov_deg=180.0).manifest_sha256


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
    assert profile.lambda_shield == (0.1 if expected[1] else 0)
    assert profile.lambda_reg == (1.0 if expected[2] else 0)
    assert (profile.alpha_min, profile.lambda_pi, profile.lambda_v) == (0.1, 0.05, 0.005)
    assert type(profile).from_manifest(profile.to_manifest()) == profile


@pytest.mark.parametrize("name", ["baseline", "cbf_only", "FULL", "", None, True])
def test_no_unregistered_profiles(name):
    with pytest.raises(ValueError):
        resolve_ablation_profile(name)
