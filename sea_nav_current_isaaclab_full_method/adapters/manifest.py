from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rsl_rl.experiment_config import ResolvedRunConfig, RunIdentity, resolve_run_config


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
    action_chain_mode: str = "current_pre_delay_cbf"
    command_filter_mode: str = "source_alpha_only"
    cbf_fov_deg: float = 240.0
    cbf_footprint_radius_m: float = 0.55
    cbf_min_effective_clearance_m: float = 0.01
    timeout_seconds: float = 40.0
    source_perception_delay_enabled: bool = True
    source_reward_done_parity_enabled: bool = True
    source_stand_still_time_steps: int = 150
    source_contact_termination_enabled: bool = True
    source_play_eval_terminal_semantics_enabled: bool = True


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

    The positional adapter-only form is retained for Task 1/static callers.
    It produces an explicitly unverified upstream-profile/IsaacLab identity;
    new runtime callers must pass every keyword argument.
    """
    if repo_root is None and resolved_config is None and validation_rung is None and adapter_root is not None:
        adapter_root = Path(adapter_root)
        repo_root = adapter_root.resolve().parent
        resolved_config = resolve_run_config(
            registry_path=repo_root / "configs" / "parity_registry.yaml",
            algorithm_profile="upstream_fbce672c",
            runtime_stack="isaaclab_adapter",
            implementation_delta=("ppo_state_identity_repair",),
        )
        validation_rung = "unverified"
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
        runtime_contract=RuntimeContract(),
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
