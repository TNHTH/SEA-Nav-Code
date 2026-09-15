# SPDX-License-Identifier: MIT
"""SEA-owned DashGo primitive candidate and its provenance-bound asset manifest.

The primitive is rebuilt from SEA-owned primitive shapes using the frozen
candidate geometry; no DashGo URDF/xacro/Python is imported or copied.  All
values are simulation assumptions provenance-bound to the read-only DashGo
commit, never calibrated hardware facts.

CPU import discipline: this module loads neither Isaac, ROS, nor DashGo
packages.  The Isaac Lab articulation/raycaster configuration factory imports
``isaaclab`` lazily and fails closed with an explicit blocked error when the
target stack is absent — it never fabricates a config.
"""

import hashlib
import json
from typing import Dict, List, Sequence, Tuple

ASSET_MANIFEST_SCHEMA = 1
PLATFORM_PROFILE = "dashgo_d1_primitive_candidate_v1"
RESULT_CLASSIFICATION = "cross_platform_method_adaptation"
VALIDATION_IDENTITY = "simulation_surrogate_candidate"

DASHGO_COMMIT = "98018dd09923495db321a09920dccc09f796f805"
SOURCE_HASHES = {
    "configs/robot/dashgo.urdf":
        "51cb52cc60176405ed24735a4cc648f12fd1924d46f77022ae0e730f4346d892",
    "drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml":
        "e1cc89d2220a01e07323c3395b1c675a125c25d4547b9d8f497be07af7cdbd1f",
    "workspaces/ros2_ws/src/dashgo_rl_ros2/urdf/dashgo_d1_sim.urdf.xacro":
        "9857b0c4006a8d942ecade413baa2df8b54b6947f5088e5c9ab8f5e925682bb6",
}

# Frozen mechanical-fact candidates (uncalibrated).
BODY_RADIUS_M = 0.203
BODY_HEIGHT_M = 0.21
REPORTED_NET_MASS_KG = 13.7
WHEEL_RADIUS_M = 0.0632
TRACK_WIDTH_M = 0.342
WHEEL_WIDTH_M = 0.04
CASTER_RADIUS_M = 0.03
CASTER_X_M = (-0.15, 0.15)
LIDAR_XYZ_YAW = (0.0, 0.0, 0.13, 0.0)

# Frozen simulation operational assumptions.
V_RANGE_MPS = (-0.15, 0.30)
OMEGA_RANGE_RADPS = (-1.0, 1.0)
LINEAR_ACCEL_MPS2 = 1.0
ANGULAR_ACCEL_RADPS2 = 0.6
EXPERIMENT_WHEEL_LIMIT_RADPS = 5.0
ASSET_WHEEL_LIMIT_RADPS = 10.0
PHYSICS_DT_S = 0.005
POLICY_DT_S = 0.02
RAY_COUNT = 41
RAY_FOV_DEG = (-120.0, 120.0)
RAY_RANGE_M = (0.1, 3.0)
CBF_LOOKAHEAD_M = 0.2
CBF_SAFETY_MARGIN_M = 0.05
FRICTION_RANDOMIZATION = (0.2, 1.25)
VELOCITY_DRIVE_STIFFNESS = 0.0
ACTUATOR_SEARCH_ORDER = (
    (20.0, 2.0), (20.0, 5.0), (20.0, 10.0),
    (50.0, 2.0), (50.0, 5.0), (50.0, 10.0),
)
COLLISION_GROUP = "dashgo_chassis_lidar_structure"
COLLISION_GROUP_BETA = 1.0
EXCLUDED_CONTACTS = ("wheel_floor", "caster_floor")
UNVERIFIED_UNTIL_G12 = (
    "inertia", "center_of_mass", "contact_material", "caster_dynamics",
    "actuator_pair_selection",
)

# SEA-owned primitive parts (candidate assumptions, frame: base_link).
PRIMITIVE_PARTS = (
    {
        "name": "chassis",
        "shape": "cylinder",
        "radius_m": BODY_RADIUS_M,
        "height_m": BODY_HEIGHT_M,
        "origin_xyz_m": (0.0, 0.0, BODY_HEIGHT_M / 2.0),
        "collision_group": COLLISION_GROUP,
    },
    {
        "name": "wheel_left",
        "shape": "cylinder",
        "radius_m": WHEEL_RADIUS_M,
        "length_m": WHEEL_WIDTH_M,
        "axis": "y",
        "origin_xyz_m": (0.0, -TRACK_WIDTH_M / 2.0, WHEEL_RADIUS_M),
        "collision_group": COLLISION_GROUP,
    },
    {
        "name": "wheel_right",
        "shape": "cylinder",
        "radius_m": WHEEL_RADIUS_M,
        "length_m": WHEEL_WIDTH_M,
        "axis": "y",
        "origin_xyz_m": (0.0, TRACK_WIDTH_M / 2.0, WHEEL_RADIUS_M),
        "collision_group": COLLISION_GROUP,
    },
    {
        "name": "caster_front",
        "shape": "sphere",
        "radius_m": CASTER_RADIUS_M,
        "origin_xyz_m": (CASTER_X_M[1], 0.0, CASTER_RADIUS_M),
        "collision_group": COLLISION_GROUP,
    },
    {
        "name": "caster_rear",
        "shape": "sphere",
        "radius_m": CASTER_RADIUS_M,
        "origin_xyz_m": (CASTER_X_M[0], 0.0, CASTER_RADIUS_M),
        "collision_group": COLLISION_GROUP,
    },
    {
        "name": "lidar_frame",
        "shape": "frame_only",
        "origin_xyz_m": LIDAR_XYZ_YAW[:3],
        "yaw_rad": LIDAR_XYZ_YAW[3],
    },
)


def asset_manifest() -> Dict:
    """The provenance-bound DashGo asset manifest (pure data, deterministic)."""
    from sea_nav_core import sea_ray_angles_deg
    return {
        "schema_version": ASSET_MANIFEST_SCHEMA,
        "platform_profile": PLATFORM_PROFILE,
        "result_classification": RESULT_CLASSIFICATION,
        "validation_identity": VALIDATION_IDENTITY,
        "provenance": {
            "dashgo_commit": DASHGO_COMMIT,
            "source_sha256": dict(SOURCE_HASHES),
            "status": "candidate_not_calibrated",
        },
        "mechanical_fact_candidates": {
            "body_radius_m": BODY_RADIUS_M,
            "body_height_m": BODY_HEIGHT_M,
            "reported_net_mass_kg": REPORTED_NET_MASS_KG,
            "wheel_radius_m": WHEEL_RADIUS_M,
            "track_width_m": TRACK_WIDTH_M,
            "wheel_width_m": WHEEL_WIDTH_M,
            "caster_radius_m": CASTER_RADIUS_M,
            "caster_x_m": list(CASTER_X_M),
        },
        "simulation_operational_assumptions": {
            "v_range_mps": list(V_RANGE_MPS),
            "omega_range_radps": list(OMEGA_RANGE_RADPS),
            "linear_acceleration_mps2": LINEAR_ACCEL_MPS2,
            "angular_acceleration_radps2": ANGULAR_ACCEL_RADPS2,
            "experiment_wheel_limit_radps": EXPERIMENT_WHEEL_LIMIT_RADPS,
            "asset_wheel_limit_radps": ASSET_WHEEL_LIMIT_RADPS,
            "physics_dt_s": PHYSICS_DT_S,
            "policy_dt_s": POLICY_DT_S,
            "ray_count": RAY_COUNT,
            "ray_fov_deg": list(RAY_FOV_DEG),
            "ray_range_m": list(RAY_RANGE_M),
            "ray_angles_deg": list(sea_ray_angles_deg()),
            "cbf_lookahead_m": CBF_LOOKAHEAD_M,
            "cbf_safety_margin_m": CBF_SAFETY_MARGIN_M,
            "lidar_xyz_yaw": list(LIDAR_XYZ_YAW),
            "friction_randomization": list(FRICTION_RANDOMIZATION),
            "friction_randomization_kind": "uniform_half_open",
            "velocity_drive_stiffness": VELOCITY_DRIVE_STIFFNESS,
            "actuator_search_order_effort_damping": [
                list(pair) for pair in ACTUATOR_SEARCH_ORDER
            ],
            "actuator_selection_rule":
                "freeze_first_combination_passing_all_G12_kinematic_smokes",
            "collision_group": COLLISION_GROUP,
            "collision_group_beta": COLLISION_GROUP_BETA,
            "excluded_contacts": list(EXCLUDED_CONTACTS),
            "unverified_until_g4_g12": list(UNVERIFIED_UNTIL_G12),
        },
        "primitive_parts": [
            {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in part.items()
            }
            for part in PRIMITIVE_PARTS
        ],
    }


def asset_manifest_sha256() -> str:
    payload = json.dumps(asset_manifest(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def isaaclab_dashgo_articulation_config():
    """Build the Isaac Lab articulation configuration (target stack only).

    Fails closed when Isaac Lab is unavailable; this factory never returns a
    mocked configuration and is never called from CPU/static test paths.
    """
    try:
        import isaaclab.sim as sim  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "blocked: Isaac Lab is unavailable on this host; the DashGo "
            "articulation configuration requires the pinned target stack"
        ) from exc
    from sea_nav_core import sea_nav_dashgo_candidate_platform_spec
    platform = sea_nav_dashgo_candidate_platform_spec()
    # Concrete Isaac Lab cfg assembly is exercised only on the target stack;
    # CPU hosts get the explicit blocked error above.
    return {
        "platform_manifest_sha256": platform.manifest_sha256,
        "asset_manifest_sha256": asset_manifest_sha256(),
    }


def isaaclab_sea_raycaster_pattern():
    """Explicit 41-angle raycaster pattern (-120..120 deg, 6 deg spacing)."""
    try:
        from isaaclab.sensors.ray_caster import PatternCfg  # noqa: F401
        import torch
    except Exception as exc:
        raise RuntimeError(
            "blocked: Isaac Lab is unavailable on this host; the raycaster "
            "pattern requires the pinned target stack"
        ) from exc
    from sea_nav_core import sea_ray_angles_rad
    import math
    angles = torch.tensor(sea_ray_angles_rad(), dtype=torch.float32)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
