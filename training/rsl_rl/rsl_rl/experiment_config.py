"""Typed, simulator-free resolution of SEA-Nav scientific run identity.

The registry is authoritative.  Profile YAML files are redundant validation
inputs so a profile cannot silently override a registry selection.  Runtime
consumers are deliberately represented as inactive projections until the
owning implementation tasks wire them into PPO, Gym, and IsaacLab.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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
        paper_value=mapping["paper_value"],
        upstream_effective_value=mapping["upstream_effective_value"],
        selected_values={profile: selections[profile] for profile in ALGORITHM_PROFILES},
        evidence=normalized_evidence,
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
    if dict(profile_selected) != dict(selected):
        raise ValueError("profile selected_contracts does not match registry selections")


def _section(selected: Mapping[str, Any], contracts: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for contract in contracts:
        value = selected[contract]
        if not isinstance(value, Mapping):
            raise ValueError("selected contract {} must be a mapping for application".format(contract))
        for key, item in value.items():
            if key in result and result[key] != item:
                raise ValueError("selected application key {} conflicts while applying {}".format(key, contract))
            result[key] = item
    return result


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
        values=dict(algorithm.observation, **algorithm.cbf),
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
        values=dict(
            algorithm.loss,
            action_stages=(
                "distribution_mean",
                "policy_action",
                "clipped_policy_action",
                "executed_command",
            ),
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
        values=env_values,
    )
    replay = ConfigProjection(
        consumer="replay_reset",
        application_status="future_task_5",
        activation_task=5,
        consumed_contracts=("replay_reset_policy",),
        values=dict(algorithm.replay),
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
        "observation": dict(algorithm.observation),
        "cbf": dict(algorithm.cbf),
        "loss": dict(algorithm.loss),
        "reward": dict(algorithm.reward),
        "perception": dict(algorithm.perception),
        "acsi": dict(algorithm.acsi),
        "horizons": dict(algorithm.horizons),
        "execution": dict(algorithm.execution),
        "replay": dict(algorithm.replay),
    }


def _projection_dict(projection: ConfigProjection) -> Dict[str, Any]:
    return {
        "consumer": projection.consumer,
        "application_status": projection.application_status,
        "activation_task": projection.activation_task,
        "consumed_contracts": list(projection.consumed_contracts),
        "values": dict(projection.values),
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
        "selected_contracts": dict(selected_contracts),
        "algorithm": _algorithm_dict(algorithm),
        "application_projections": [_projection_dict(item) for item in projections],
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


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
    selected = {row.contract: row.selected_values[algorithm_profile] for row in rows}
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
    return config.application_projections


def _projection(config: ResolvedRunConfig, consumer: str) -> ConfigProjection:
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
    payload["resolved_sha256"] = config.resolved_sha256
    return payload


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
