# SPDX-License-Identifier: MIT
"""CPU-visible DirectRLEnv configuration constants for SEA DashGo.

Isaac Lab ``DirectRLEnvCfg`` field names are mirrored here so CPU/static tests
can assert the contract without importing Isaac. Target-stack ``env.py`` binds
these values into a real ``DirectRLEnvCfg`` subclass.
"""

from __future__ import annotations

from typing import Dict

# Scientific / task-card fixed timing and spaces.
SIM_DT_S = 0.005
DECIMATION = 4
RENDER_INTERVAL = 4
POLICY_DT_S = SIM_DT_S * DECIMATION  # 0.02
EPISODE_LENGTH_S = 60.0
ACTION_SPACE_DIM = 2
OBSERVATION_SPACE_DIM = 550
PLATFORM_PROFILE = "dashgo_d1_primitive_candidate_v1"
RESULT_CLASSIFICATION = "cross_platform_method_adaptation"
VALIDATION_IDENTITY = "simulation_surrogate_candidate"

# Training is refused until env/policy integration and Isaac runtime wiring complete.
# A6–A9 CPU/static modules may be source-complete while this flag stays False.
TRAIN_READY = False
TRAIN_BLOCK_REASON = (
    "train entry refused: TRAIN_READY=False until A4–A5 env/policy integration "
    "and Isaac SimulationApp wiring complete; A6–A9 CPU modules are not sufficient "
    "to launch real training"
)


def env_cfg_source() -> Dict:
    """Inspectable cfg source matching DirectRLEnvCfg field names."""
    return {
        "class_type": "SeaNavDiffDriveEnvCfg",
        "bases": ["DirectRLEnvCfg"],
        "sim": {
            "dt": SIM_DT_S,
            "render_interval": RENDER_INTERVAL,
        },
        "decimation": DECIMATION,
        "episode_length_s": EPISODE_LENGTH_S,
        "action_space": ACTION_SPACE_DIM,
        "observation_space": OBSERVATION_SPACE_DIM,
        "platform_profile": PLATFORM_PROFILE,
        "result_classification": RESULT_CLASSIFICATION,
        "validation_identity": VALIDATION_IDENTITY,
        "policy_dt_s": POLICY_DT_S,
        "train_ready": TRAIN_READY,
        "train_block_reason": TRAIN_BLOCK_REASON,
    }


def validate_env_cfg_source(source: Dict) -> None:
    if source.get("bases") != ["DirectRLEnvCfg"]:
        raise ValueError("cfg must inherit DirectRLEnvCfg")
    if float(source["sim"]["dt"]) != SIM_DT_S:
        raise ValueError("sim.dt must be 0.005")
    if int(source["decimation"]) != DECIMATION:
        raise ValueError("decimation must be 4")
    if int(source["sim"]["render_interval"]) != RENDER_INTERVAL:
        raise ValueError("render_interval must be 4")
    if float(source["episode_length_s"]) != EPISODE_LENGTH_S:
        raise ValueError("episode_length_s must be 60")
    if int(source["action_space"]) != ACTION_SPACE_DIM:
        raise ValueError("action_space must be 2")
    if int(source["observation_space"]) != OBSERVATION_SPACE_DIM:
        raise ValueError("observation_space must be 550")
    if abs(float(source["policy_dt_s"]) - POLICY_DT_S) > 1e-12:
        raise ValueError("policy_dt_s must equal sim.dt * decimation")


def assert_train_entry_allowed() -> None:
    if not TRAIN_READY:
        raise RuntimeError(TRAIN_BLOCK_REASON)
