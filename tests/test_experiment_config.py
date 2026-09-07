from __future__ import annotations

import dataclasses
import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RSL_RL_ROOT = ROOT / "training" / "rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

REGISTRY = ROOT / "configs" / "parity_registry.yaml"
PROFILES = ROOT / "configs" / "profiles"

from rsl_rl.experiment_config import (  # noqa: E402
    ALGORITHM_PROFILES,
    RESOLUTION_STATUSES,
    RUNTIME_STACKS,
    ConfigProjection,
    ParityRow,
    ResolvedAlgorithmConfig,
    ResolvedRunConfig,
    RunIdentity,
    build_application_projections,
    build_env_profile_values,
    build_policy_kwargs,
    build_ppo_kwargs,
    build_replay_kwargs,
    load_parity_registry,
    resolve_run_config,
    resolved_config_to_dict,
    validate_registry,
    write_resolved_config,
)


def _evidence():
    return {
        "paper": ["paper:page-7:Table-V"],
        "upstream": ["git:fbce672c:training/rsl_rl/rsl_rl/algorithms/ppo.py:230"],
    }


def _row(contract="damped_cbf", status="resolved", paper_value=None, upstream_value=None):
    selections = {
        "damped_cbf": {
            "paper_v1": {
                "cbf_mode": "paper_damped",
                "epsilon_d": 1.0,
                "result_classification": "differentiable_safety_bias",
            },
            "upstream_fbce672c": {
                "cbf_mode": "paper_damped",
                "epsilon_d": 1.0,
                "result_classification": "differentiable_safety_bias",
            },
        },
        "ppo_state_identity_repair": {
            "paper_v1": {"ppo_auxiliary_state_mode": "pure_action_mean_for_preserves_current_minibatch_state"},
            "upstream_fbce672c": {
                "ppo_auxiliary_state_mode": "pure_action_mean_for_preserves_current_minibatch_state"
            },
        },
        "cbf_geometry": {
            "paper_v1": {"cbf_fov_deg": 240.0},
            "upstream_fbce672c": {"cbf_fov_deg": 180.0},
        },
    }[contract]
    paper_value = selections["paper_v1"] if paper_value is None else paper_value
    upstream_value = selections["upstream_fbce672c"] if upstream_value is None else upstream_value
    return {
        "contract": contract,
        "paper_value": paper_value,
        "upstream_effective_value": upstream_value,
        "selected_values": {
            "paper_v1": paper_value,
            "upstream_fbce672c": upstream_value,
        },
        "evidence": _evidence(),
        "resolution_status": status,
        "rationale": "A complete non-empty rationale.",
    }


def _write_registry(tmp_path, rows, profile_overrides=None):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    profile_files = {
        "paper_v1": "profiles/paper_v1.yaml",
        "upstream_fbce672c": "profiles/upstream_fbce672c.yaml",
    }
    for profile in ALGORITHM_PROFILES:
        selections = {row["contract"]: row["selected_values"][profile] for row in rows}
        if profile_overrides and profile in profile_overrides:
            selections.update(profile_overrides[profile])
        (profiles_dir / f"{profile}.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "algorithm_profile": profile,
                    "selected_contracts": selections,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile_files": profile_files,
                "rows": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return registry


def _resolve_upstream(registry_path=REGISTRY, implementation_delta=("ppo_state_identity_repair",)):
    return resolve_run_config(
        registry_path=registry_path,
        algorithm_profile="upstream_fbce672c",
        runtime_stack="isaac_gym_preview4",
        implementation_delta=implementation_delta,
    )


def _matching_invalid_selection(tmp_path, contract, updates):
    registry_payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    row = next(row for row in registry_payload["rows"] if row["contract"] == contract)
    selected = row["selected_values"]["upstream_fbce672c"]
    selected.update(copy.deepcopy(updates))

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for profile in ALGORITHM_PROFILES:
        source = yaml.safe_load((PROFILES / f"{profile}.yaml").read_text(encoding="utf-8"))
        if profile == "upstream_fbce672c":
            source["selected_contracts"][contract].update(copy.deepcopy(updates))
        (profiles_dir / f"{profile}.yaml").write_text(
            yaml.safe_dump(source, sort_keys=True), encoding="utf-8"
        )
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry_payload, sort_keys=True), encoding="utf-8")
    return path


def test_public_value_objects_are_frozen_and_adapter_import_is_compatible():
    assert dataclasses.is_dataclass(ParityRow)
    assert dataclasses.is_dataclass(RunIdentity)
    assert dataclasses.is_dataclass(ResolvedAlgorithmConfig)
    assert dataclasses.is_dataclass(ResolvedRunConfig)
    assert dataclasses.is_dataclass(ConfigProjection)
    assert ParityRow.__dataclass_params__.frozen is True
    assert RunIdentity.__dataclass_params__.frozen is True
    assert ResolvedAlgorithmConfig.__dataclass_params__.frozen is True
    assert ResolvedRunConfig.__dataclass_params__.frozen is True
    assert ConfigProjection.__dataclass_params__.frozen is True

    from sea_nav_current_isaaclab_full_method.adapters import experiment_config as adapter_config

    assert adapter_config.ParityRow is ParityRow
    assert adapter_config.resolve_run_config is resolve_run_config


def test_resolved_data_is_deeply_immutable():
    config = _resolve_upstream()
    with pytest.raises(TypeError):
        config.selected_contracts["reward_weights"]["raw_weights"]["velocity"] = 99.0
    with pytest.raises(TypeError):
        build_policy_kwargs(config).values["epsilon_d"] = 7.0
    with pytest.raises(TypeError):
        build_env_profile_values(config).values["history_indices"][0] = -99


def test_materialized_consumer_values_and_serialization_are_detached():
    config = _resolve_upstream()
    policy_values = build_policy_kwargs(config).materialize_values()
    policy_values["epsilon_d"] = 7.0
    assert build_policy_kwargs(config).values["epsilon_d"] == 1.0

    serialized = resolved_config_to_dict(config)
    serialized["selected_contracts"]["reward_weights"]["raw_weights"]["velocity"] = 99.0
    serialized["application_projections"][0]["values"]["epsilon_d"] = 7.0
    fresh = resolved_config_to_dict(config)
    assert fresh["selected_contracts"]["reward_weights"]["raw_weights"]["velocity"] == 4.0
    assert fresh["application_projections"][0]["values"]["epsilon_d"] == 1.0


def test_serialization_and_manifest_reject_broken_resolved_hash():
    from sea_nav_current_isaaclab_full_method.adapters.manifest import default_manifest

    config = dataclasses.replace(_resolve_upstream(), resolved_sha256="0" * 64)
    with pytest.raises(ValueError, match="resolved configuration integrity"):
        resolved_config_to_dict(config)
    with pytest.raises(ValueError, match="resolved configuration integrity"):
        default_manifest(
            repo_root=ROOT,
            adapter_root=ROOT / "sea_nav_current_isaaclab_full_method",
            resolved_config=config,
            validation_rung="unverified",
        )


def test_registry_carries_all_evidence_corrected_scientific_contracts():
    rows = load_parity_registry(REGISTRY)
    by_name = {row.contract: row for row in rows}
    required = {
        "lidar_history",
        "damped_cbf",
        "cbf_geometry",
        "cbf_kappa_safe_radius_margin",
        "cbf_footprint_preprocessing",
        "shield_objective",
        "reward_weights",
        "reward_formula",
        "reward_time_integration",
        "reward_operational_fallbacks",
        "smoothness_weights",
        "ppo_state_identity_repair",
        "paper_table_action_bounds",
        "execution_bounds",
        "ray_delay",
        "perception_timing_semantics",
        "acsi_curriculum",
        "acsi_decision_stage",
        "acsi_level_storage_and_update_event",
        "goal_completion",
        "time_horizons",
        "replay_reset_policy",
    }
    assert set(by_name) == required
    assert all(row.resolution_status in RESOLUTION_STATUSES for row in rows)

    bounds = by_name["paper_table_action_bounds"]
    assert bounds.resolution_status == "blocked"
    assert bounds.paper_value == {"low": (-0.5, 0.8, 1.0), "high": (1.7, 0.8, 1.0)}
    assert bounds.upstream_effective_value == {"low": (-0.5, -0.8, -1.0), "high": (1.7, 0.8, 1.0)}

    reward = by_name["reward_formula"]
    assert reward.paper_value["angular_velocity"] == "l2_norm_xy"
    assert reward.upstream_effective_value["angular_velocity"] == "squared_l2_norm_xy"
    assert by_name["reward_time_integration"].selected_values["paper_v1"]["integrate_over_policy_dt"] is True

    acsi = by_name["acsi_decision_stage"]
    assert acsi.paper_value["decision_stages"] == ("collision_replay_decision_carried_to_reset",)
    assert acsi.upstream_effective_value["decision_stages"] == (
        "collision_onset_termination_draw",
        "terminal_replay_draw",
    )
    perception = by_name["perception_timing_semantics"]
    assert perception.resolution_status == "blocked"
    assert perception.paper_value["acquisition_period_s"] == 0.1
    assert perception.paper_value["transport_latency_s"] == {"distribution": "uniform", "low": 0.04, "high": 0.08}
    assert perception.paper_value["output_mode"] == "sample_and_hold"
    assert by_name["cbf_footprint_preprocessing"].selected_values["upstream_fbce672c"]["footprint_radius_m"] == 0.0


def test_upstream_ray_acquisition_and_output_refresh_cadences_are_distinct():
    rows = {row.contract: row for row in load_parity_registry(REGISTRY)}
    upstream = rows["ray_delay"].selected_values["upstream_fbce672c"]
    paper = rows["ray_delay"].selected_values["paper_v1"]
    assert paper["acquisition_period_s"] == 0.1
    assert upstream["acquisition_period_s"] == 0.02
    assert upstream["output_refresh_period_s"] == 0.1
    expected_ages = tuple((abs(index) - 1) * upstream["acquisition_period_s"] for index in upstream["history_indices"])
    assert upstream["refresh_sample_age_s"] == pytest.approx(expected_ages)
    refresh_steps = round(upstream["output_refresh_period_s"] / upstream["acquisition_period_s"])
    expected_maximum_age = max(expected_ages) + (refresh_steps - 1) * upstream["acquisition_period_s"]
    assert upstream["maximum_held_age_s"] == pytest.approx(expected_maximum_age)


def test_differing_evaluation_horizons_are_a_profile_fork():
    row = {row.contract: row for row in load_parity_registry(REGISTRY)}["time_horizons"]
    assert row.resolution_status == "profile_fork"
    assert row.selected_values["paper_v1"]["evaluation_timeout_s"] == 30.0
    assert row.selected_values["upstream_fbce672c"]["evaluation_timeout_s"] is None
    assert row.selected_values["paper_v1"] != row.selected_values["upstream_fbce672c"]


@pytest.mark.parametrize(
    "contract, updates, message",
    [
        ("damped_cbf", {"invented_parameter": True}, "damped_cbf.*fields"),
        ("damped_cbf", {"epsilon_d": "not-a-number"}, "damped_cbf.*epsilon_d"),
        ("damped_cbf", {"epsilon_d": float("nan")}, "damped_cbf.*epsilon_d"),
        ("cbf_geometry", {"cbf_fov_deg": 0.0}, "cbf_geometry.*cbf_fov_deg"),
        ("cbf_geometry", {"cbf_fov_deg": 361.0}, "cbf_geometry.*cbf_fov_deg"),
        ("ray_delay", {"delay_mode": "invented_scheduler"}, "ray_delay.*delay_mode"),
    ],
)
def test_matching_registry_and_profile_invalid_selections_are_rejected(
    tmp_path, contract, updates, message
):
    registry = _matching_invalid_selection(tmp_path, contract, updates)
    with pytest.raises(ValueError, match=message):
        _resolve_upstream(registry)


@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "mapping"),
        ({"schema_version": 1, "profile_files": {}, "rows": [], "extra": True}, "root keys"),
        ({"schema_version": 2, "profile_files": {}, "rows": []}, "schema_version"),
        (
            {
                "schema_version": 1,
                "profile_files": {
                    "paper_v1": "profiles/paper_v1.yaml",
                    "upstream_fbce672c": "profiles/upstream_fbce672c.yaml",
                },
                "rows": "bad",
            },
            "rows",
        ),
    ],
)
def test_load_rejects_invalid_registry_roots(tmp_path, payload, message):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_parity_registry(path)


def test_load_rejects_missing_and_unknown_row_keys(tmp_path):
    missing = _row()
    missing.pop("rationale")
    path = _write_registry(tmp_path / "missing", [missing])
    with pytest.raises(ValueError, match="row keys"):
        load_parity_registry(path)

    unknown = _row()
    unknown["invented"] = True
    path = _write_registry(tmp_path / "unknown", [unknown])
    with pytest.raises(ValueError, match="row keys"):
        load_parity_registry(path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"contract": ""}, "contract"),
        ({"resolution_status": "guessed"}, "resolution_status"),
        ({"rationale": ""}, "rationale"),
        ({"evidence": {}}, "evidence"),
        ({"evidence": {"paper": [], "upstream": ["source"]}}, "evidence"),
        ({"evidence": {"paper": ["source"], "upstream": [""]}}, "evidence"),
        ({"selected_values": {"paper_v1": 1}}, "selected_values"),
    ],
)
def test_validate_registry_rejects_malformed_rows(mutation, message):
    raw = _row()
    raw.update(mutation)
    with pytest.raises(ValueError, match=message):
        validate_registry([raw])


def test_validate_registry_rejects_duplicate_contracts():
    with pytest.raises(ValueError, match="duplicate.*damped_cbf"):
        validate_registry([_row(), _row()])


def test_resolver_rejects_unknown_profile_runtime_and_delta(tmp_path):
    registry = _write_registry(tmp_path, [_row("ppo_state_identity_repair", "implementation_delta")])
    with pytest.raises(ValueError, match="algorithm_profile"):
        resolve_run_config(
            registry_path=registry,
            algorithm_profile="paper_v2",
            runtime_stack="isaac_gym_preview4",
            implementation_delta=("ppo_state_identity_repair",),
        )
    with pytest.raises(ValueError, match="runtime_stack"):
        resolve_run_config(
            registry_path=registry,
            algorithm_profile="upstream_fbce672c",
            runtime_stack="mujoco",
            implementation_delta=("ppo_state_identity_repair",),
        )
    with pytest.raises(ValueError, match="declared implementation delta"):
        _resolve_upstream(registry, ("ppo_state_identity_repair", "silent_science_fix"))


def test_repair_delta_is_explicit_mandatory_unique_and_not_an_exact_upstream_claim():
    with pytest.raises(ValueError, match="ppo_state_identity_repair"):
        _resolve_upstream(implementation_delta=())
    with pytest.raises(ValueError, match="duplicate implementation delta"):
        _resolve_upstream(
            implementation_delta=("ppo_state_identity_repair", "ppo_state_identity_repair")
        )
    repaired = _resolve_upstream()
    assert repaired.identity.implementation_delta == ("ppo_state_identity_repair",)
    assert repaired.identity.is_exact_upstream is False


def test_additional_declared_replay_repair_is_preserved_not_inserted():
    base = _resolve_upstream()
    assert "replay_reset_reconstruction_v1" not in base.identity.implementation_delta
    assert build_replay_kwargs(base).required_activation_deltas == ("replay_reset_reconstruction_v1",)

    activated = _resolve_upstream(
        implementation_delta=("ppo_state_identity_repair", "replay_reset_reconstruction_v1")
    )
    assert activated.identity.implementation_delta == (
        "ppo_state_identity_repair",
        "replay_reset_reconstruction_v1",
    )


def test_profile_validation_input_cannot_override_registry(tmp_path):
    rows = [_row("ppo_state_identity_repair", "implementation_delta")]
    registry = _write_registry(
        tmp_path,
        rows,
        profile_overrides={"upstream_fbce672c": {"ppo_state_identity_repair": {"value": "overridden"}}},
    )
    with pytest.raises(ValueError, match="does not match registry"):
        _resolve_upstream(registry)


def test_missing_profile_selection_is_rejected(tmp_path):
    rows = [
        _row("ppo_state_identity_repair", "implementation_delta"),
        _row("cbf_geometry", "profile_fork"),
    ]
    registry = _write_registry(tmp_path, rows)
    profile_path = tmp_path / "profiles" / "upstream_fbce672c.yaml"
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    payload["selected_contracts"].pop("cbf_geometry")
    profile_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selected_contracts"):
        _resolve_upstream(registry)


def test_accepted_paper_identity_is_blocked_with_actionable_literal_evidence():
    with pytest.raises(ValueError) as error:
        resolve_run_config(
            registry_path=REGISTRY,
            algorithm_profile="paper_v1",
            runtime_stack="isaaclab_adapter",
            implementation_delta=("ppo_state_identity_repair",),
        )
    message = str(error.value)
    assert "paper_v1" in message
    assert "paper_table_action_bounds" in message
    assert "Table V" in message
    assert "perception_timing_semantics" in message


def test_upstream_resolution_has_distinct_geometry_and_staged_application_contracts():
    config = _resolve_upstream()
    assert config.identity.algorithm_profile == "upstream_fbce672c"
    assert config.identity.runtime_stack == "isaac_gym_preview4"
    assert config.selected_contracts["lidar_history"] == {
        "history_frames": 10,
        "num_rays": 41,
        "observation_fov_deg": 240.0,
    }
    assert config.algorithm.observation["cbf_fov_deg"] == 180.0
    assert config.algorithm.cbf["footprint_radius_m"] == 0.0
    assert config.algorithm.cbf["nonzero_footprint_policy"] == "requires_named_ablation"
    assert config.algorithm.reward["integrate_over_policy_dt"] is True
    assert config.algorithm.reward["raw_weights"]["velocity"] == 4.0
    assert config.algorithm.perception["history_indices"] == (-3, -4)
    assert config.algorithm.acsi["terminal_replay_probability"] == 0.8
    assert config.algorithm.horizons["evaluation_timeout_s"] is None
    assert config.algorithm.horizons["evaluation_timeout_status"] == "unsupported_no_complete_upstream_metric_runner"
    assert config.algorithm.horizons["training_episode_s"] == 60.0
    assert config.algorithm.horizons["stay_ticks"] == 150

    policy = build_policy_kwargs(config)
    ppo = build_ppo_kwargs(config)
    env = build_env_profile_values(config)
    replay = build_replay_kwargs(config)
    assert policy.application_status == "future_task_4_and_6"
    assert ppo.application_status == "future_task_3_and_4"
    assert env.application_status == "future_task_5_and_6"
    assert replay.application_status == "future_task_5"
    assert policy.values["cbf_fov_deg"] == 180.0
    assert ppo.values["action_range_low"] == (-0.5, -0.8, -1.0)
    assert ppo.values["action_stages"] == (
        "distribution_mean",
        "policy_action",
        "clipped_policy_action",
        "executed_command",
    )
    assert env.values["reward_scale_unit"] == "raw_weight_times_policy_dt"
    assert replay.values["reconstruction_policy"] == "new_replay_episode_v1"

    projections = build_application_projections(config)
    assert projections == (policy, ppo, env, replay)
    consumed = [name for projection in projections for name in projection.consumed_contracts]
    assert len(consumed) == len(set(consumed))
    assert set(consumed) == set(config.selected_contracts)


def test_supported_identity_axes_are_exact():
    assert ALGORITHM_PROFILES == ("paper_v1", "upstream_fbce672c")
    assert RUNTIME_STACKS == (
        "go2_l1_stock_mpc",
        "go2_rplidar_agile_controller",
        "isaac_gym_preview4",
        "isaaclab_adapter",
    )


def test_resolved_hash_is_stable_and_output_is_canonical(tmp_path):
    config = _resolve_upstream()
    assert config.registry_sha256 == hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    assert config.resolved_sha256 == "6d90936678c80a531ef425b9b994fc4a0f1b610dc976946cc3919d364806953e"

    out = tmp_path / "resolved.json"
    write_resolved_config(out, config)
    raw = out.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert raw[:-1] == json.dumps(
        json.loads(raw), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    round_trip = json.loads(raw)
    assert round_trip["resolved_sha256"] == config.resolved_sha256
    assert round_trip["identity"] == {
        "algorithm_profile": "upstream_fbce672c",
        "implementation_delta": ["ppo_state_identity_repair"],
        "runtime_stack": "isaac_gym_preview4",
    }


def test_manifest_binds_resolved_identity_and_rejects_outside_adapter_root(tmp_path):
    from sea_nav_current_isaaclab_full_method.adapters.manifest import default_manifest, write_manifest

    resolved = _resolve_upstream()
    adapter_root = ROOT / "sea_nav_current_isaaclab_full_method"
    manifest = default_manifest(
        repo_root=ROOT,
        adapter_root=adapter_root,
        resolved_config=resolved,
        validation_rung="rung_2_cpu_static",
    )
    assert manifest.route_id == "sea_nav_profiled_adapter"
    assert manifest.run_identity == resolved.identity
    assert manifest.resolved_config_sha256 == resolved.resolved_sha256
    assert manifest.validation_rung == "rung_2_cpu_static"
    assert manifest.notes == {
        "result_class": "isaaclab_adapter_evidence",
        "simulator_status": "blocked_until_runtime_gate",
    }
    output = tmp_path / "manifest.json"
    write_manifest(output, manifest)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_identity"]["implementation_delta"] == ["ppo_state_identity_repair"]
    assert payload["resolved_config_sha256"] == resolved.resolved_sha256

    with pytest.raises(ValueError, match="inside repo_root"):
        default_manifest(
            repo_root=ROOT,
            adapter_root=tmp_path,
            resolved_config=resolved,
            validation_rung="rung_2_cpu_static",
        )


def test_committed_adapter_examples_are_identity_bound_and_not_original_claims():
    manifest = json.loads(
        (ROOT / "sea_nav_current_isaaclab_full_method" / "adapter_manifest.json").read_text(encoding="utf-8")
    )
    adapter_config = yaml.safe_load(
        (ROOT / "sea_nav_current_isaaclab_full_method" / "configs" / "sea_nav_full_current.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["route_id"] == "sea_nav_profiled_adapter"
    assert manifest["run_identity"]["runtime_stack"] == "isaaclab_adapter"
    assert manifest["run_identity"]["implementation_delta"] == ["ppo_state_identity_repair"]
    assert manifest["validation_rung"] == "unverified"
    assert len(manifest["resolved_config_sha256"]) == 64
    resolved = resolve_run_config(
        registry_path=REGISTRY,
        algorithm_profile=manifest["run_identity"]["algorithm_profile"],
        runtime_stack=manifest["run_identity"]["runtime_stack"],
        implementation_delta=tuple(manifest["run_identity"]["implementation_delta"]),
    )
    assert manifest["resolved_config_sha256"] == resolved.resolved_sha256
    assert manifest["notes"]["result_class"] == "isaaclab_adapter_evidence"
    assert "original_reproduction" not in json.dumps(manifest, sort_keys=True)

    assert adapter_config["schema_version"] == 1
    assert adapter_config["algorithm_profile"] == "upstream_fbce672c"
    assert adapter_config["implementation_delta"] == ["ppo_state_identity_repair", "replay_reset_reconstruction_v1"]
    executable = resolve_run_config(registry_path=REGISTRY,
        algorithm_profile=adapter_config["algorithm_profile"],runtime_stack="isaaclab_adapter",
        implementation_delta=adapter_config["implementation_delta"])
    assert executable.identity.implementation_delta == tuple(adapter_config["implementation_delta"])
    assert adapter_config["runtime"]["enable_collision_replay"] is False
    assert "original_reproduction" not in json.dumps(adapter_config, sort_keys=True)


def test_cpu_requirements_and_lock_match_the_verified_environment():
    direct = (ROOT / "requirements-cpu.txt").read_text(encoding="utf-8").splitlines()
    assert direct == [
        "numpy==1.26.4",
        "pytest==8.4.2",
        "PyYAML==6.0.2",
        "torch==2.6.0+cpu",
    ]
    installed = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in importlib.metadata.distributions()
    }
    locked = (ROOT / "requirements-cpu.lock").read_text(encoding="utf-8").splitlines()
    assert locked == [
        "exceptiongroup==1.3.1",
        "filelock==3.32.3",
        "fsspec==2026.7.0",
        "iniconfig==2.3.0",
        "Jinja2==3.1.6",
        "MarkupSafe==3.0.3",
        "mpmath==1.3.0",
        "networkx==3.4.2",
        "numpy==1.26.4",
        "packaging==26.3",
        "pluggy==1.6.0",
        "Pygments==2.21.0",
        "pytest==8.4.2",
        "PyYAML==6.0.2",
        "sympy==1.13.1",
        "tomli==2.4.1",
        "torch==2.6.0+cpu",
        "typing_extensions==4.16.0",
    ]
    for requirement in locked:
        name, version = requirement.split("==", 1)
        assert installed[name.lower()] == version
