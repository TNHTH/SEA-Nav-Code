from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from rsl_rl.experiment_config import ResolvedRunConfig, RunIdentity


@dataclass(frozen=True)
class FormalEvalContract:
    hard_room: bool = True
    seeds: int = 100
    forbid_astar_policy_input: bool = True
    forbid_centerline_policy_input: bool = True
    forbid_privileged_full_map: bool = True
    collision_replay_enabled: bool = False
    actor_lse_cbf_enabled: bool = True
    target_success_rate_pct_min: float = 72.0
    target_collision_rate_pct_max: float = 6.25
    target_termination_rate_pct_max: float = 6.25


@dataclass(frozen=True)
class RuntimeContract:
    action_chain_mode: str
    command_filter_mode: str
    cbf_fov_deg: float
    cbf_footprint_radius_m: float
    cbf_min_effective_clearance_m: Optional[float]
    timeout_seconds: float
    source_perception_delay_enabled: bool
    source_reward_done_parity_enabled: bool
    source_stand_still_time_steps: int
    source_contact_termination_enabled: bool
    source_play_eval_terminal_semantics_enabled: bool
    application_status: str = "projection_only_not_runtime_evidence"


@dataclass(frozen=True)
class AdapterManifest:
    route_id: str
    source_repo: str
    source_commit: str
    adapter_root: str
    owned_paths: List[str]
    upstream_reference_paths: List[str]
    formal_eval: FormalEvalContract
    runtime_contract: RuntimeContract
    run_identity: RunIdentity
    resolved_config_sha256: str
    validation_rung: str
    notes: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_manifest(
    adapter_root: Optional[str | Path] = None,
    *,
    repo_root: Optional[Path] = None,
    resolved_config: Optional[ResolvedRunConfig] = None,
    validation_rung: Optional[str] = None,
) -> AdapterManifest:
    """Build an identity-bound manifest.

    Identity-free callers must migrate at the Task 6 runtime wiring boundary;
    this layer never invents an implementation delta for them.
    """
    if repo_root is None and resolved_config is None and validation_rung is None and adapter_root is not None:
        raise TypeError(
            "legacy identity-free default_manifest call rejected; pass explicit repo_root, "
            "adapter_root, resolved_config, and validation_rung when migrating the runtime in Task 6"
        )
    elif adapter_root is None or repo_root is None or resolved_config is None or validation_rung is None:
        raise TypeError(
            "default_manifest requires repo_root, adapter_root, resolved_config, and validation_rung"
        )
    adapter_root = Path(adapter_root)
    repo_root = Path(repo_root)
    try:
        relative_root = adapter_root.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("adapter_root must be inside repo_root") from error
    if not relative_root or relative_root == ".":
        raise ValueError("adapter_root must name a child directory inside repo_root")
    if not isinstance(validation_rung, str) or not validation_rung.strip():
        raise ValueError("validation_rung must be a non-empty string")
    from rsl_rl.experiment_config import resolved_config_to_dict

    resolved_config_to_dict(resolved_config)
    return AdapterManifest(
        route_id="sea_nav_profiled_adapter",
        source_repo=".",
        source_commit="fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96",
        adapter_root=relative_root,
        owned_paths=[f"{relative_root}/**"],
        upstream_reference_paths=[
            "training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py",
            "training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py",
            "training/rsl_rl/rsl_rl/algorithms/ppo.py",
            "training/legged_gym/legged_gym/envs/base/legged_robot_pos.py",
            "training/legged_gym/legged_gym/envs/go2/go2_pos_config.py",
        ],
        formal_eval=FormalEvalContract(),
        runtime_contract=RuntimeContract(
            action_chain_mode="current_pre_delay_cbf",command_filter_mode="source_alpha_only",
            cbf_fov_deg=resolved_config.algorithm.cbf["cbf_fov_deg"],
            cbf_footprint_radius_m=resolved_config.algorithm.cbf["footprint_radius_m"],
            cbf_min_effective_clearance_m=resolved_config.algorithm.cbf["min_effective_clearance_m"],
            timeout_seconds=resolved_config.algorithm.horizons["training_episode_s"],
            source_perception_delay_enabled=True,source_reward_done_parity_enabled=True,
            source_stand_still_time_steps=150,source_contact_termination_enabled=True,
            source_play_eval_terminal_semantics_enabled=False),
        run_identity=resolved_config.identity,
        resolved_config_sha256=resolved_config.resolved_sha256,
        validation_rung=validation_rung.strip(),
        notes={
            "result_class": "isaaclab_adapter_evidence",
            "simulator_status": "blocked_until_runtime_gate",
        },
    )


def write_manifest(path: str | Path, manifest: AdapterManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bind_runtime_result(*args, **kwargs):
    from rsl_rl.runtime_preflight import bind_runtime_result as bind
    return bind(*args, **kwargs)


def close_runtime_resources(*args, **kwargs):
    from rsl_rl.runtime_preflight import close_runtime_resources as close
    return close(*args, **kwargs)


def publish_runtime_result(*args, **kwargs):
    from rsl_rl.runtime_preflight import publish_runtime_result as publish
    return publish(*args, **kwargs)
