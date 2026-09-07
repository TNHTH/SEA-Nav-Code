"""Typed, simulator-free resolution of SEA-Nav scientific run identity.

The registry is authoritative.  Profile YAML files are redundant validation
inputs so a profile cannot silently override a registry selection.  Runtime
consumers are deliberately represented as inactive projections until the
owning implementation tasks wire them into PPO, Gym, and IsaacLab.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


ALGORITHM_PROFILES = ("paper_v1", "upstream_fbce672c")
RUNTIME_STACKS = (
    "go2_l1_stock_mpc",
    "go2_rplidar_agile_controller",
    "isaac_gym_preview4",
    "isaaclab_adapter",
)
RESOLUTION_STATUSES = (
    "blocked",
    "implementation_delta",
    "profile_fork",
    "record_only_runtime",
    "resolved",
    "resolved_effective",
    "resolved_upstream_fallback",
)
DECLARED_IMPLEMENTATION_DELTAS = (
    "ppo_state_identity_repair",
    "replay_reset_reconstruction_v1",
)

_REGISTRY_KEYS = {"schema_version", "profile_files", "rows"}
_ROW_KEYS = {
    "contract",
    "paper_value",
    "upstream_effective_value",
    "selected_values",
    "evidence",
    "resolution_status",
    "rationale",
}
_PROFILE_KEYS = {"schema_version", "algorithm_profile", "selected_contracts"}
_EVIDENCE_KEYS = {"paper", "upstream"}

_COMMON_SELECTED_SCHEMAS = {
    "lidar_history": {
        "num_rays": "positive_int",
        "observation_fov_deg": "fov_degrees",
        "history_frames": "positive_int",
    },
    "damped_cbf": {
        "cbf_mode": ("enum", ("paper_damped",)),
        "epsilon_d": "positive_number",
        "result_classification": ("enum", ("differentiable_safety_bias",)),
    },
    "cbf_geometry": {"cbf_fov_deg": "fov_degrees"},
    "cbf_kappa_safe_radius_margin": {
        "kappa": "positive_number",
        "safe_radius_m": "positive_number",
        "safety_margin_m": "nonnegative_number",
    },
    "cbf_footprint_preprocessing": {
        "footprint_radius_m": "nonnegative_number",
        "ray_preprocess_mode": ("enum", ("positive_raw_rays",)),
        "min_effective_clearance_m": "optional_positive_number",
        "nonzero_footprint_policy": ("enum", ("requires_named_ablation",)),
    },
    "shield_objective": {
        "intervention_coefficient": "nonnegative_number",
        "alpha_penalty_coefficient": "nonnegative_number",
        "alpha_min": "nonnegative_number",
    },
    "reward_weights": {"raw_weights": "reward_weights"},
    "reward_formula": {
        "formula_mode": ("enum", ("paper_table_iii", "upstream_fbce672c")),
        "angular_velocity": ("enum", ("l2_norm_xy", "squared_l2_norm_xy")),
        "velocity_term": (
            "enum",
            ("cosine_heading_times_forward_plus_goal_bias", "clipped_forward_progress_plus_goal_bias"),
        ),
        "clearance_boundary_m": "positive_number",
        "stuck_history_anchor": ("enum", ("first_history_point", "current_position")),
    },
    "reward_time_integration": {
        "integrate_over_policy_dt": "bool",
        "reward_scale_unit": ("enum", ("raw_weight_times_policy_dt",)),
        "policy_dt_s": "positive_number",
    },
    "reward_operational_fallbacks": {"operational_fallbacks": "reward_fallbacks"},
    "smoothness_weights": {
        "actor_smoothness_coefficient": "nonnegative_number",
        "critic_smoothness_coefficient": "nonnegative_number",
    },
    "ppo_state_identity_repair": {
        "ppo_auxiliary_state_mode": (
            "enum",
            ("pure_action_mean_for_preserves_current_minibatch_state",),
        )
    },
    "paper_table_action_bounds": {
        "action_range_low": ("number_vector", 3),
        "action_range_high": ("number_vector", 3),
    },
    "execution_bounds": {
        "executed_command_low": ("number_vector", 3),
        "executed_command_high": ("number_vector", 3),
    },
    "perception_timing_semantics": {
        "perception_scheduler": (
            "enum",
            ("timestamped_acquisition_transport_hold", "policy_tick_history_refresh"),
        ),
        "arrival_quantization": ("enum", ("blocked_paper_unspecified", "policy_tick")),
        "output_mode": ("enum", ("sample_and_hold",)),
        "delay_goal_with_rays": ("enum_or_bool", ("blocked_paper_unspecified",)),
        "startup_policy": (
            "enum",
            ("blocked_paper_unspecified", "current_sample_until_history_available"),
        ),
    },
    "acsi_decision_stage": {
        "decision_stages": (
            "sequence_enum",
            (
                ("collision_replay_decision_carried_to_reset",),
                ("collision_onset_termination_draw", "terminal_replay_draw"),
            ),
        ),
        "terminal_replay_probability": "probability",
        "decision_order": (
            "enum",
            ("probability_then_reservation_then_reset_commit", "collision_onset_then_reset_selection"),
        ),
    },
    "acsi_level_storage_and_update_event": {
        "level_update_policy": ("enum", ("strict_distance_events",)),
        "initial_level": "nonnegative_number",
        "stored_level_bounds": ("sequence_exact", (0.0, "max_terrain_level")),
        "level_update_event": ("enum", ("normal_episode_reset_after_replay_selection",)),
    },
    "goal_completion": {
        "goal_distance_threshold_m": "positive_number",
        "stay_ticks": "positive_int",
        "stay_counter_mode": ("enum", ("accumulated_in_goal_ticks",)),
    },
    "replay_reset_policy": {
        "reconstruction_policy": ("enum", ("new_replay_episode_v1",)),
        "activation_delta": ("enum", ("replay_reset_reconstruction_v1",)),
        "curriculum_state_rewound": "bool",
    },
}

_PROFILE_SELECTED_SCHEMAS = {
    "ray_delay": {
        "paper_v1": {
            "delay_mode": ("enum", ("timestamped_continuous_latency",)),
            "acquisition_period_s": "positive_number",
            "latency_min_s": "nonnegative_number",
            "latency_max_s": "positive_number",
            "history_sampling_policy": ("enum", ("latest_nonfuture_for_sampled_latency",)),
        },
        "upstream_fbce672c": {
            "delay_mode": ("enum", ("discrete_history_sample_and_hold",)),
            "acquisition_period_s": "positive_number",
            "output_refresh_period_s": "positive_number",
            "history_indices": ("sequence_exact", (-3, -4)),
            "refresh_sample_age_s": ("number_vector", 2),
            "maximum_held_age_s": "positive_number",
        },
    },
    "acsi_curriculum": {
        "paper_v1": {
            "curriculum_mode": ("enum", ("paper_eq_1",)),
            "p_min": "probability",
            "p_max": "probability",
            "probability_level_clip": ("number_vector", 2),
            "d_up_m": "positive_number",
            "d_down_m": "positive_number",
        },
        "upstream_fbce672c": {
            "curriculum_mode": ("enum", ("upstream_two_stage",)),
            "p_min": "probability",
            "p_max": "probability",
            "probability_level_divisor": "positive_number",
            "d_up_m": "positive_number",
            "d_down_m": "positive_number",
        },
    },
    "time_horizons": {
        "paper_v1": {
            "training_episode_s": "positive_number",
            "evaluation_timeout_s": "positive_number",
        },
        "upstream_fbce672c": {
            "training_episode_s": "positive_number",
            "evaluation_timeout_s": "none",
            "evaluation_timeout_status": (
                "enum",
                ("unsupported_no_complete_upstream_metric_runner",),
            ),
        },
    },
}

_REWARD_NAMES = {"termination", "reach", "velocity", "clearance", "stuck", "collision", "angular"}
_CONTACT_GROUPS = {"generic", "head_base", "leg"}


@dataclass(frozen=True)
class ParityRow:
    contract: str
    paper_value: Any
    upstream_effective_value: Any
    selected_values: Mapping[str, Any]
    evidence: Mapping[str, Tuple[str, ...]]
    resolution_status: str
    rationale: str


@dataclass(frozen=True)
class RunIdentity:
    algorithm_profile: str
    runtime_stack: str
    implementation_delta: Tuple[str, ...]

    @property
    def is_exact_upstream(self) -> bool:
        return (
            self.algorithm_profile == "upstream_fbce672c"
            and self.runtime_stack == "isaac_gym_preview4"
            and not self.implementation_delta
        )


@dataclass(frozen=True)
class ResolvedAlgorithmConfig:
    observation: Mapping[str, Any]
    cbf: Mapping[str, Any]
    loss: Mapping[str, Any]
    reward: Mapping[str, Any]
    perception: Mapping[str, Any]
    acsi: Mapping[str, Any]
    horizons: Mapping[str, Any]
    execution: Mapping[str, Any]
    replay: Mapping[str, Any]


@dataclass(frozen=True)
class ConfigProjection:
    consumer: str
    application_status: str
    activation_task: int
    consumed_contracts: Tuple[str, ...]
    values: Mapping[str, Any]
    required_activation_deltas: Tuple[str, ...] = ()

    def materialize_values(self) -> Dict[str, Any]:
        """Return a detached mutable copy suitable for constructor kwargs."""
        return _thaw(self.values)


@dataclass(frozen=True)
class ResolvedRunConfig:
    identity: RunIdentity
    registry_sha256: str
    selected_contracts: Mapping[str, Any]
    algorithm: ResolvedAlgorithmConfig
    application_projections: Tuple[ConfigProjection, ...]
    resolved_sha256: str


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(label))
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unknown = sorted(actual_set - expected_set)
        raise ValueError("{} keys invalid; missing={}, unknown={}".format(label, missing, unknown))


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_number(value: Any, label: str, lower: float, inclusive: bool) -> None:
    if not _is_finite_number(value):
        raise ValueError("{} must be a finite number".format(label))
    if (inclusive and value < lower) or (not inclusive and value <= lower):
        comparison = ">=" if inclusive else ">"
        raise ValueError("{} must be {} {}".format(label, comparison, lower))


def _validate_reward_weights(value: Any, label: str) -> None:
    weights = _require_mapping(value, label)
    _require_exact_keys(weights, _REWARD_NAMES, label)
    for name, weight in weights.items():
        if not _is_finite_number(weight):
            raise ValueError("{}.{} must be a finite number".format(label, name))


def _validate_reward_fallbacks(value: Any, label: str) -> None:
    fallbacks = _require_mapping(value, label)
    _require_exact_keys(
        fallbacks,
        {"contact_group_coefficients", "opening_tie_semantics", "initial_contact_penalty_suppressed"},
        label,
    )
    groups = _require_mapping(fallbacks["contact_group_coefficients"], "{}.contact_group_coefficients".format(label))
    _require_exact_keys(groups, _CONTACT_GROUPS, "{}.contact_group_coefficients".format(label))
    for group, coefficient in groups.items():
        _validate_number(coefficient, "{}.contact_group_coefficients.{}".format(label, group), 0.0, True)
    if fallbacks["opening_tie_semantics"] != "smoothed_150_degree_cone_center_bias":
        raise ValueError("{}.opening_tie_semantics has an unsupported value".format(label))
    if not isinstance(fallbacks["initial_contact_penalty_suppressed"], bool):
        raise ValueError("{}.initial_contact_penalty_suppressed must be bool".format(label))


def _validate_field(value: Any, spec: Any, label: str) -> None:
    if spec == "positive_int":
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("{} must be a positive integer".format(label))
        return
    if spec == "positive_number":
        _validate_number(value, label, 0.0, False)
        return
    if spec == "fov_degrees":
        _validate_number(value, label, 0.0, False)
        if value > 360.0:
            raise ValueError("{} must be <= 360 degrees".format(label))
        return
    if spec == "nonnegative_number":
        _validate_number(value, label, 0.0, True)
        return
    if spec == "optional_positive_number":
        if value is not None:
            _validate_number(value, label, 0.0, False)
        return
    if spec == "probability":
        _validate_number(value, label, 0.0, True)
        if value > 1.0:
            raise ValueError("{} must be <= 1.0".format(label))
        return
    if spec == "bool":
        if not isinstance(value, bool):
            raise ValueError("{} must be bool".format(label))
        return
    if spec == "none":
        if value is not None:
            raise ValueError("{} must be null".format(label))
        return
    if spec == "reward_weights":
        _validate_reward_weights(value, label)
        return
    if spec == "reward_fallbacks":
        _validate_reward_fallbacks(value, label)
        return

    kind, expected = spec
    if kind == "enum":
        if value not in expected:
            raise ValueError("{} must be one of {}".format(label, expected))
    elif kind == "enum_or_bool":
        if not isinstance(value, bool) and value not in expected:
            raise ValueError("{} must be bool or one of {}".format(label, expected))
    elif kind == "number_vector":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != expected:
            raise ValueError("{} must contain {} numbers".format(label, expected))
        for index, item in enumerate(value):
            if not _is_finite_number(item):
                raise ValueError("{}[{}] must be a finite number".format(label, index))
    elif kind == "sequence_exact":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or tuple(value) != expected:
            raise ValueError("{} must equal {}".format(label, expected))
    elif kind == "sequence_enum":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or tuple(value) not in expected:
            raise ValueError("{} must be one of {}".format(label, expected))
    else:
        raise ValueError("{} has an unknown schema validator".format(label))


def _validate_selected_contract(contract: str, profile: str, value: Any) -> None:
    selected = _require_mapping(value, "{} selected value for {}".format(contract, profile))
    if contract in _COMMON_SELECTED_SCHEMAS:
        schema = _COMMON_SELECTED_SCHEMAS[contract]
    elif contract in _PROFILE_SELECTED_SCHEMAS:
        schema = _PROFILE_SELECTED_SCHEMAS[contract][profile]
    else:
        raise ValueError("unknown parity contract {!r}".format(contract))
    _require_exact_keys(selected, schema, "{} selected fields for {}".format(contract, profile))
    for field, spec in schema.items():
        _validate_field(selected[field], spec, "{}.{} for {}".format(contract, field, profile))

    if contract in ("paper_table_action_bounds", "execution_bounds"):
        prefix = "action_range" if contract == "paper_table_action_bounds" else "executed_command"
        if any(low > high for low, high in zip(selected[prefix + "_low"], selected[prefix + "_high"])):
            raise ValueError("{} lower bounds must not exceed upper bounds".format(contract))
    elif contract == "cbf_footprint_preprocessing" and selected["footprint_radius_m"] != 0.0:
        raise ValueError("{}.footprint_radius_m requires a named ablation profile".format(contract))
    elif contract == "ray_delay":
        if profile == "paper_v1":
            if selected["latency_min_s"] > selected["latency_max_s"]:
                raise ValueError("ray_delay latency_min_s must not exceed latency_max_s")
        else:
            acquisition = selected["acquisition_period_s"]
            refresh = selected["output_refresh_period_s"]
            refresh_steps = refresh / acquisition
            if not math.isclose(refresh_steps, round(refresh_steps), rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("ray_delay output refresh must be an integer number of acquisition periods")
            expected_ages = tuple((abs(index) - 1) * acquisition for index in selected["history_indices"])
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
                for actual, expected in zip(selected["refresh_sample_age_s"], expected_ages)
            ):
                raise ValueError("ray_delay refresh_sample_age_s is inconsistent with history indices")
            expected_maximum = max(expected_ages) + (round(refresh_steps) - 1) * acquisition
            if not math.isclose(
                selected["maximum_held_age_s"], expected_maximum, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError("ray_delay maximum_held_age_s is inconsistent with sample-and-hold cadence")
    elif contract == "acsi_curriculum":
        if selected["p_min"] > selected["p_max"]:
            raise ValueError("acsi_curriculum p_min must not exceed p_max")
        if selected["d_up_m"] >= selected["d_down_m"]:
            raise ValueError("acsi_curriculum d_up_m must be less than d_down_m")
        if profile == "paper_v1" and tuple(selected["probability_level_clip"]) != (0.0, 1.0):
            raise ValueError("acsi_curriculum probability_level_clip must equal [0, 1]")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _normalize_row(raw: Any, index: int) -> ParityRow:
    if isinstance(raw, ParityRow):
        raw = {
            "contract": raw.contract,
            "paper_value": raw.paper_value,
            "upstream_effective_value": raw.upstream_effective_value,
            "selected_values": raw.selected_values,
            "evidence": raw.evidence,
            "resolution_status": raw.resolution_status,
            "rationale": raw.rationale,
        }
    mapping = _require_mapping(raw, "row {}".format(index))
    _require_exact_keys(mapping, _ROW_KEYS, "row keys")
    contract = mapping["contract"]
    if not _is_nonempty_string(contract):
        raise ValueError("row contract must be a non-empty string")
    status = mapping["resolution_status"]
    if status not in RESOLUTION_STATUSES:
        raise ValueError("unknown resolution_status {!r} for {}".format(status, contract))
    rationale = mapping["rationale"]
    if not _is_nonempty_string(rationale):
        raise ValueError("rationale must be non-empty for {}".format(contract))

    selections = _require_mapping(mapping["selected_values"], "selected_values for {}".format(contract))
    _require_exact_keys(selections, ALGORITHM_PROFILES, "selected_values")
    for profile in ALGORITHM_PROFILES:
        _validate_selected_contract(contract, profile, selections[profile])
    if status == "resolved" and selections["paper_v1"] != selections["upstream_fbce672c"]:
        raise ValueError("{} status resolved requires identical profile selections".format(contract))
    if status == "profile_fork" and selections["paper_v1"] == selections["upstream_fbce672c"]:
        raise ValueError("{} status profile_fork requires distinct profile selections".format(contract))
    evidence = _require_mapping(mapping["evidence"], "evidence for {}".format(contract))
    _require_exact_keys(evidence, _EVIDENCE_KEYS, "evidence")
    normalized_evidence: Dict[str, Tuple[str, ...]] = {}
    for source in sorted(_EVIDENCE_KEYS):
        locations = evidence[source]
        if (
            not isinstance(locations, Sequence)
            or isinstance(locations, (str, bytes))
            or not locations
            or any(not _is_nonempty_string(location) for location in locations)
        ):
            raise ValueError("evidence {} for {} must contain non-empty locations".format(source, contract))
        normalized_evidence[source] = tuple(locations)
    return ParityRow(
        contract=contract.strip(),
        paper_value=_freeze(mapping["paper_value"]),
        upstream_effective_value=_freeze(mapping["upstream_effective_value"]),
        selected_values=_freeze({profile: selections[profile] for profile in ALGORITHM_PROFILES}),
        evidence=_freeze(normalized_evidence),
        resolution_status=status,
        rationale=rationale.strip(),
    )


def validate_registry(rows: Iterable[Any]) -> Tuple[ParityRow, ...]:
    if isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("rows must be an iterable of row mappings")
    normalized: List[ParityRow] = []
    seen = set()
    for index, raw in enumerate(rows):
        row = _normalize_row(raw, index)
        if row.contract in seen:
            raise ValueError("duplicate parity contract: {}".format(row.contract))
        seen.add(row.contract)
        normalized.append(row)
    if not normalized:
        raise ValueError("rows must contain at least one parity contract")
    return tuple(normalized)


def _load_registry_document(path: Path) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("unable to load parity registry {}: {}".format(path, error)) from error
    root = _require_mapping(document, "registry root")
    _require_exact_keys(root, _REGISTRY_KEYS, "registry root keys")
    if root["schema_version"] != 1:
        raise ValueError("unsupported registry schema_version: {!r}".format(root["schema_version"]))
    profile_files = _require_mapping(root["profile_files"], "profile_files")
    _require_exact_keys(profile_files, ALGORITHM_PROFILES, "profile_files")
    for profile, relative_path in profile_files.items():
        if not _is_nonempty_string(relative_path) or Path(relative_path).is_absolute():
            raise ValueError("profile_files entry for {} must be a non-empty relative path".format(profile))
        resolved = (path.parent / relative_path).resolve()
        try:
            resolved.relative_to(path.parent.resolve())
        except ValueError as error:
            raise ValueError("profile_files entry for {} escapes the registry directory".format(profile)) from error
    if not isinstance(root["rows"], list):
        raise ValueError("registry rows must be a list")
    return root


def load_parity_registry(path: Any) -> Tuple[ParityRow, ...]:
    registry_path = Path(path)
    root = _load_registry_document(registry_path)
    return validate_registry(root["rows"])


def _load_profile(path: Path, profile: str, selected: Mapping[str, Any]) -> None:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("unable to load profile validation input {}: {}".format(path, error)) from error
    root = _require_mapping(document, "profile root")
    _require_exact_keys(root, _PROFILE_KEYS, "profile root")
    if root["schema_version"] != 1:
        raise ValueError("unsupported profile schema_version: {!r}".format(root["schema_version"]))
    if root["algorithm_profile"] != profile:
        raise ValueError("profile identity does not match registry key {}".format(profile))
    profile_selected = _require_mapping(root["selected_contracts"], "selected_contracts")
    if set(profile_selected) != set(selected):
        raise ValueError(
            "profile selected_contracts keys do not match registry; missing={}, unknown={}".format(
                sorted(set(selected) - set(profile_selected)), sorted(set(profile_selected) - set(selected))
            )
        )
    if _freeze(profile_selected) != _freeze(selected):
        raise ValueError("profile selected_contracts does not match registry selections")


def _section(selected: Mapping[str, Any], contracts: Sequence[str]) -> Mapping[str, Any]:
    result: Dict[str, Any] = {}
    for contract in contracts:
        value = selected[contract]
        if not isinstance(value, Mapping):
            raise ValueError("selected contract {} must be a mapping for application".format(contract))
        for key, item in value.items():
            if key in result and result[key] != item:
                raise ValueError("selected application key {} conflicts while applying {}".format(key, contract))
            result[key] = item
    return _freeze(result)


def _build_algorithm(selected: Mapping[str, Any]) -> ResolvedAlgorithmConfig:
    return ResolvedAlgorithmConfig(
        observation=_section(selected, ("lidar_history", "cbf_geometry")),
        cbf=_section(
            selected,
            ("damped_cbf", "cbf_geometry", "cbf_kappa_safe_radius_margin", "cbf_footprint_preprocessing"),
        ),
        loss=_section(
            selected,
            ("shield_objective", "smoothness_weights", "ppo_state_identity_repair", "paper_table_action_bounds"),
        ),
        reward=_section(
            selected,
            ("reward_weights", "reward_formula", "reward_time_integration", "reward_operational_fallbacks"),
        ),
        perception=_section(selected, ("ray_delay", "perception_timing_semantics")),
        acsi=_section(
            selected,
            ("acsi_curriculum", "acsi_decision_stage", "acsi_level_storage_and_update_event"),
        ),
        horizons=_section(selected, ("goal_completion", "time_horizons")),
        execution=_section(selected, ("execution_bounds",)),
        replay=_section(selected, ("replay_reset_policy",)),
    )


def _build_projections(algorithm: ResolvedAlgorithmConfig) -> Tuple[ConfigProjection, ...]:
    policy = ConfigProjection(
        consumer="policy_factory",
        application_status="future_task_4_and_6",
        activation_task=4,
        consumed_contracts=(
            "lidar_history",
            "damped_cbf",
            "cbf_geometry",
            "cbf_kappa_safe_radius_margin",
            "cbf_footprint_preprocessing",
        ),
        values=_freeze(dict(algorithm.observation, **algorithm.cbf)),
    )
    ppo = ConfigProjection(
        consumer="ppo_constructor",
        application_status="future_task_3_and_4",
        activation_task=3,
        consumed_contracts=(
            "shield_objective",
            "smoothness_weights",
            "ppo_state_identity_repair",
            "paper_table_action_bounds",
        ),
        values=_freeze(
            dict(
                algorithm.loss,
                action_stages=(
                    "distribution_mean",
                    "policy_action",
                    "clipped_policy_action",
                    "executed_command",
                ),
            )
        ),
        required_activation_deltas=("ppo_state_identity_repair",),
    )
    env_values: Dict[str, Any] = {}
    for section in (algorithm.execution, algorithm.reward, algorithm.perception, algorithm.acsi, algorithm.horizons):
        for key, value in section.items():
            if key in env_values and env_values[key] != value:
                raise ValueError("environment application key {} has conflicting values".format(key))
            env_values[key] = value
    env = ConfigProjection(
        consumer="environment_profile",
        application_status="future_task_5_and_6",
        activation_task=5,
        consumed_contracts=(
            "execution_bounds",
            "reward_weights",
            "reward_formula",
            "reward_time_integration",
            "reward_operational_fallbacks",
            "ray_delay",
            "perception_timing_semantics",
            "acsi_curriculum",
            "acsi_decision_stage",
            "acsi_level_storage_and_update_event",
            "goal_completion",
            "time_horizons",
        ),
        values=_freeze(env_values),
    )
    replay = ConfigProjection(
        consumer="replay_reset",
        application_status="future_task_5",
        activation_task=5,
        consumed_contracts=("replay_reset_policy",),
        values=_freeze(dict(algorithm.replay)),
        required_activation_deltas=("replay_reset_reconstruction_v1",),
    )
    return (policy, ppo, env, replay)


def _identity_dict(identity: RunIdentity) -> Dict[str, Any]:
    return {
        "algorithm_profile": identity.algorithm_profile,
        "runtime_stack": identity.runtime_stack,
        "implementation_delta": list(identity.implementation_delta),
    }


def _algorithm_dict(algorithm: ResolvedAlgorithmConfig) -> Dict[str, Any]:
    return {
        "observation": _thaw(algorithm.observation),
        "cbf": _thaw(algorithm.cbf),
        "loss": _thaw(algorithm.loss),
        "reward": _thaw(algorithm.reward),
        "perception": _thaw(algorithm.perception),
        "acsi": _thaw(algorithm.acsi),
        "horizons": _thaw(algorithm.horizons),
        "execution": _thaw(algorithm.execution),
        "replay": _thaw(algorithm.replay),
    }


def _projection_dict(projection: ConfigProjection) -> Dict[str, Any]:
    return {
        "consumer": projection.consumer,
        "application_status": projection.application_status,
        "activation_task": projection.activation_task,
        "consumed_contracts": list(projection.consumed_contracts),
        "values": _thaw(projection.values),
        "required_activation_deltas": list(projection.required_activation_deltas),
    }


def _payload_dict(
    identity: RunIdentity,
    registry_sha256: str,
    selected_contracts: Mapping[str, Any],
    algorithm: ResolvedAlgorithmConfig,
    projections: Tuple[ConfigProjection, ...],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": _identity_dict(identity),
        "registry_sha256": registry_sha256,
        "selected_contracts": _thaw(selected_contracts),
        "algorithm": _algorithm_dict(algorithm),
        "application_projections": [_projection_dict(item) for item in projections],
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def resolve_run_config(
    *,
    registry_path: Any,
    algorithm_profile: str,
    runtime_stack: str,
    implementation_delta: Sequence[str],
) -> ResolvedRunConfig:
    if algorithm_profile not in ALGORITHM_PROFILES:
        raise ValueError("unsupported algorithm_profile {!r}; expected one of {}".format(algorithm_profile, ALGORITHM_PROFILES))
    if runtime_stack not in RUNTIME_STACKS:
        raise ValueError("unsupported runtime_stack {!r}; expected one of {}".format(runtime_stack, RUNTIME_STACKS))
    if isinstance(implementation_delta, (str, bytes)):
        raise ValueError("implementation_delta must be a sequence of declared names")
    deltas = tuple(implementation_delta)
    if any(not _is_nonempty_string(delta) for delta in deltas):
        raise ValueError("implementation_delta entries must be non-empty strings")
    if len(set(deltas)) != len(deltas):
        raise ValueError("duplicate implementation delta is not allowed")
    unknown_deltas = sorted(set(deltas) - set(DECLARED_IMPLEMENTATION_DELTAS))
    if unknown_deltas:
        raise ValueError("unknown declared implementation delta(s): {}".format(unknown_deltas))
    if "ppo_state_identity_repair" not in deltas:
        raise ValueError(
            "repaired runs must explicitly declare ppo_state_identity_repair; the resolver never inserts it"
        )

    path = Path(registry_path)
    root = _load_registry_document(path)
    rows = validate_registry(root["rows"])
    selected = _freeze({row.contract: row.selected_values[algorithm_profile] for row in rows})
    relative_profile_path = root["profile_files"][algorithm_profile]
    _load_profile(path.parent / relative_profile_path, algorithm_profile, selected)

    if algorithm_profile == "paper_v1":
        blocked = [row for row in rows if row.resolution_status == "blocked"]
        if blocked:
            details = "; ".join(
                "{}: {} [paper evidence: {}]".format(
                    row.contract, row.rationale, ", ".join(row.evidence["paper"])
                )
                for row in blocked
            )
            raise ValueError("paper_v1 is blocked and cannot identify an accepted run: {}".format(details))

    identity = RunIdentity(algorithm_profile, runtime_stack, deltas)
    algorithm = _build_algorithm(selected)
    projections = _build_projections(algorithm)
    consumed = [name for projection in projections for name in projection.consumed_contracts]
    if len(consumed) != len(set(consumed)) or set(consumed) != set(selected):
        raise ValueError("application projections must consume every selected contract exactly once")
    registry_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = _payload_dict(identity, registry_sha256, selected, algorithm, projections)
    resolved_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return ResolvedRunConfig(
        identity=identity,
        registry_sha256=registry_sha256,
        selected_contracts=selected,
        algorithm=algorithm,
        application_projections=projections,
        resolved_sha256=resolved_sha256,
    )


def build_application_projections(config: ResolvedRunConfig) -> Tuple[ConfigProjection, ...]:
    _validate_resolved_integrity(config)
    return config.application_projections


def _projection(config: ResolvedRunConfig, consumer: str) -> ConfigProjection:
    _validate_resolved_integrity(config)
    matches = [item for item in config.application_projections if item.consumer == consumer]
    if len(matches) != 1:
        raise ValueError("resolved config must contain exactly one {} projection".format(consumer))
    return matches[0]


def build_policy_kwargs(config: ResolvedRunConfig) -> ConfigProjection:
    return _projection(config, "policy_factory")


def build_ppo_kwargs(config: ResolvedRunConfig) -> ConfigProjection:
    return _projection(config, "ppo_constructor")


def build_env_profile_values(config: ResolvedRunConfig) -> ConfigProjection:
    return _projection(config, "environment_profile")


def build_replay_kwargs(config: ResolvedRunConfig) -> ConfigProjection:
    return _projection(config, "replay_reset")


def resolved_config_to_dict(config: ResolvedRunConfig) -> Dict[str, Any]:
    payload = _payload_dict(
        config.identity,
        config.registry_sha256,
        config.selected_contracts,
        config.algorithm,
        config.application_projections,
    )
    actual_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if actual_sha256 != config.resolved_sha256:
        raise ValueError(
            "resolved configuration integrity check failed: stored sha256 {} != payload sha256 {}".format(
                config.resolved_sha256, actual_sha256
            )
        )
    payload["resolved_sha256"] = config.resolved_sha256
    return payload


def _validate_resolved_integrity(config: ResolvedRunConfig) -> None:
    resolved_config_to_dict(config)


def write_resolved_config(path: Any, config: ResolvedRunConfig) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json_bytes(resolved_config_to_dict(config)) + b"\n")


__all__ = [
    "ALGORITHM_PROFILES",
    "DECLARED_IMPLEMENTATION_DELTAS",
    "RESOLUTION_STATUSES",
    "RUNTIME_STACKS",
    "ConfigProjection",
    "ParityRow",
    "ResolvedAlgorithmConfig",
    "ResolvedRunConfig",
    "RunIdentity",
    "build_application_projections",
    "build_env_profile_values",
    "build_policy_kwargs",
    "build_ppo_kwargs",
    "build_replay_kwargs",
    "load_parity_registry",
    "resolve_run_config",
    "resolved_config_to_dict",
    "validate_registry",
    "write_resolved_config",
]
