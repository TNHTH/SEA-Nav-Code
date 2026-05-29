from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


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
    notes: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_manifest(adapter_root: str) -> AdapterManifest:
    return AdapterManifest(
        route_id="sea_nav_source_graph_parity_footprint_preserved_learned_policy_contract",
        source_repo="/home/gwh/SEA-Nav-Code",
        source_commit="fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96",
        adapter_root=adapter_root,
        owned_paths=[f"{adapter_root}/**"],
        upstream_reference_paths=[
            "/home/gwh/SEA-Nav-Code/training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py",
            "/home/gwh/SEA-Nav-Code/training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py",
            "/home/gwh/SEA-Nav-Code/training/rsl_rl/rsl_rl/algorithms/ppo.py",
            "/home/gwh/SEA-Nav-Code/training/legged_gym/legged_gym/envs/base/legged_robot_pos.py",
            "/home/gwh/SEA-Nav-Code/training/legged_gym/legged_gym/envs/go2/go2_pos_config.py",
        ],
        formal_eval=FormalEvalContract(),
        runtime_contract=RuntimeContract(),
        notes={
            "result_class": "original_reproduction",
            "gate_a": "static contract tests only; no long training",
            "eval_replay": "collision replay disabled during formal eval",
            "cbf_alpha": "strict path uses upstream F.softplus(alpha_raw) without additive gamma_min floor",
            "command_filter": "source alpha filter only; no action queue delay",
            "terminal_defaults": "source stay_time=150 and contact termination enabled by default",
        },
    )


def write_manifest(path: str | Path, manifest: AdapterManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
