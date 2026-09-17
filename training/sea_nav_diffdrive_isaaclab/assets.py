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

Frame (scientific-contract §5): +x forward / +y left / +z up; base origin at
the driven-wheel axle midpoint. left wheel y=+track/2, right y=-track/2.
"""

import hashlib
import json
import math
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
CHASSIS_MASS_KG = 12.7
WHEEL_MASS_KG = 0.4
CASTER_MASS_KG = 0.1
WHEEL_RADIUS_M = 0.0632
TRACK_WIDTH_M = 0.342
WHEEL_WIDTH_M = 0.04
CASTER_RADIUS_M = 0.03
CASTER_X_M = (-0.15, 0.15)
# Chassis bottom world z after normal reset at wheel radius; local z measured
# from axle midpoint (base origin).
CHASSIS_BOTTOM_WORLD_Z_M = 0.04
BASE_RESET_WORLD_Z_M = WHEEL_RADIUS_M
CHASSIS_LOCAL_Z_M = (
    CHASSIS_BOTTOM_WORLD_Z_M + BODY_HEIGHT_M / 2.0 - BASE_RESET_WORLD_Z_M
)
WHEEL_LOCAL_Z_M = 0.0
CASTER_LOCAL_Z_M = CASTER_RADIUS_M - BASE_RESET_WORLD_Z_M
LEFT_WHEEL_Y_M = TRACK_WIDTH_M / 2.0
RIGHT_WHEEL_Y_M = -TRACK_WIDTH_M / 2.0
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
WHEEL_CASTER_COLLISION_GROUP = "dashgo_wheel_caster"
COLLISION_GROUP_BETA = 1.0
EXCLUDED_CONTACTS = ("wheel_floor", "caster_floor")
UNVERIFIED_UNTIL_G12 = (
    "contact_material", "caster_dynamics", "actuator_pair_selection",
)
DRIVE_JOINT_POSITIVE_ROLL_FORWARD = True


def _cylinder_inertia_diag(mass: float, radius: float, length: float,
                           axis: str) -> Tuple[float, float, float]:
    """Uniform solid cylinder inertia about geometric center (kg·m²)."""
    if mass <= 0 or radius <= 0 or length <= 0:
        raise ValueError("cylinder mass/radius/length must be positive")
    i_axial = 0.5 * mass * radius * radius
    i_radial = (1.0 / 12.0) * mass * (3.0 * radius * radius + length * length)
    if axis == "z":
        return (i_radial, i_radial, i_axial)
    if axis == "y":
        return (i_radial, i_axial, i_radial)
    if axis == "x":
        return (i_axial, i_radial, i_radial)
    raise ValueError("cylinder axis must be x, y, or z")


def _sphere_inertia_diag(mass: float, radius: float) -> Tuple[float, float, float]:
    if mass <= 0 or radius <= 0:
        raise ValueError("sphere mass/radius must be positive")
    i = 0.4 * mass * radius * radius
    return (i, i, i)


def _part(name: str, shape: str, origin: Tuple[float, float, float],
          mass_kg: float, inertia_diag: Tuple[float, float, float],
          collision_group: str, **geometry) -> Dict:
    part = {
        "name": name,
        "shape": shape,
        "origin_xyz_m": origin,
        "mass_kg": mass_kg,
        "inertia_diag_kg_m2": inertia_diag,
        "collision_group": collision_group,
    }
    part.update(geometry)
    return part


def primitive_parts() -> Tuple[Dict, ...]:
    """Deterministic primitive part table in the axle-midpoint base frame."""
    chassis_inertia = _cylinder_inertia_diag(
        CHASSIS_MASS_KG, BODY_RADIUS_M, BODY_HEIGHT_M, "z"
    )
    wheel_inertia = _cylinder_inertia_diag(
        WHEEL_MASS_KG, WHEEL_RADIUS_M, WHEEL_WIDTH_M, "y"
    )
    caster_inertia = _sphere_inertia_diag(CASTER_MASS_KG, CASTER_RADIUS_M)
    return (
        _part(
            "chassis", "cylinder",
            (0.0, 0.0, CHASSIS_LOCAL_Z_M),
            CHASSIS_MASS_KG, chassis_inertia, COLLISION_GROUP,
            radius_m=BODY_RADIUS_M, height_m=BODY_HEIGHT_M, axis="z",
        ),
        _part(
            "wheel_left", "cylinder",
            (0.0, LEFT_WHEEL_Y_M, WHEEL_LOCAL_Z_M),
            WHEEL_MASS_KG, wheel_inertia, WHEEL_CASTER_COLLISION_GROUP,
            radius_m=WHEEL_RADIUS_M, length_m=WHEEL_WIDTH_M, axis="y",
            drive_joint_positive_rolls_forward=DRIVE_JOINT_POSITIVE_ROLL_FORWARD,
        ),
        _part(
            "wheel_right", "cylinder",
            (0.0, RIGHT_WHEEL_Y_M, WHEEL_LOCAL_Z_M),
            WHEEL_MASS_KG, wheel_inertia, WHEEL_CASTER_COLLISION_GROUP,
            radius_m=WHEEL_RADIUS_M, length_m=WHEEL_WIDTH_M, axis="y",
            drive_joint_positive_rolls_forward=DRIVE_JOINT_POSITIVE_ROLL_FORWARD,
        ),
        _part(
            "caster_front", "sphere",
            (CASTER_X_M[1], 0.0, CASTER_LOCAL_Z_M),
            CASTER_MASS_KG, caster_inertia, WHEEL_CASTER_COLLISION_GROUP,
            radius_m=CASTER_RADIUS_M, actuated=False,
        ),
        _part(
            "caster_rear", "sphere",
            (CASTER_X_M[0], 0.0, CASTER_LOCAL_Z_M),
            CASTER_MASS_KG, caster_inertia, WHEEL_CASTER_COLLISION_GROUP,
            radius_m=CASTER_RADIUS_M, actuated=False,
        ),
        {
            "name": "lidar_frame",
            "shape": "frame_only",
            "origin_xyz_m": LIDAR_XYZ_YAW[:3],
            "yaw_rad": LIDAR_XYZ_YAW[3],
            "mass_kg": 0.0,
            "inertia_diag_kg_m2": (0.0, 0.0, 0.0),
            "notes": "LiDAR geometry belongs to chassis rigid body; no extra mass",
        },
    )


PRIMITIVE_PARTS = primitive_parts()


def composed_center_of_mass_xyz_m() -> Tuple[float, float, float]:
    """Mass-weighted COM of massive primitive parts (excludes frame_only)."""
    total = 0.0
    mx = my = mz = 0.0
    for part in PRIMITIVE_PARTS:
        mass = float(part.get("mass_kg", 0.0))
        if mass <= 0.0:
            continue
        x, y, z = part["origin_xyz_m"]
        total += mass
        mx += mass * float(x)
        my += mass * float(y)
        mz += mass * float(z)
    if not math.isclose(total, REPORTED_NET_MASS_KG, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("primitive masses must sum to reported net mass")
    return (mx / total, my / total, mz / total)


def wheel_identity_or_raise(parts: Sequence[Dict] = None) -> Dict[str, Dict]:
    """Reject missing/duplicate wheels and inverted left/right y signs."""
    table = {part["name"]: part for part in (parts or PRIMITIVE_PARTS)}
    if "wheel_left" not in table or "wheel_right" not in table:
        raise ValueError("missing driven wheel identity")
    left = table["wheel_left"]
    right = table["wheel_right"]
    if left["origin_xyz_m"][1] <= 0:
        raise ValueError("wheel_left must have positive y (+left)")
    if right["origin_xyz_m"][1] >= 0:
        raise ValueError("wheel_right must have negative y (-right)")
    names = [part["name"] for part in (parts or PRIMITIVE_PARTS)]
    if names.count("wheel_left") != 1 or names.count("wheel_right") != 1:
        raise ValueError("duplicate driven wheel identity")
    return table


def asset_manifest() -> Dict:
    """The provenance-bound DashGo asset manifest (pure data, deterministic)."""
    from sea_nav_core import sea_ray_angles_deg
    # Lazy import keeps assets import free of USD write side effects until called.
    from asset_builder import artifact_provenance, write_dashgo_usd

    parts = primitive_parts()
    wheel_identity_or_raise(parts)
    com = composed_center_of_mass_xyz_m()
    write_dashgo_usd()
    manifest = {
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
            "mass_breakdown_kg": {
                "chassis": CHASSIS_MASS_KG,
                "wheel_left": WHEEL_MASS_KG,
                "wheel_right": WHEEL_MASS_KG,
                "caster_front": CASTER_MASS_KG,
                "caster_rear": CASTER_MASS_KG,
                "lidar_frame": 0.0,
            },
            "wheel_radius_m": WHEEL_RADIUS_M,
            "track_width_m": TRACK_WIDTH_M,
            "wheel_width_m": WHEEL_WIDTH_M,
            "caster_radius_m": CASTER_RADIUS_M,
            "caster_x_m": list(CASTER_X_M),
            "base_frame": {
                "origin": "driven_wheel_axle_midpoint",
                "axes": "+x_forward_+y_left_+z_up",
                "normal_reset_world_z_m": BASE_RESET_WORLD_Z_M,
                "chassis_bottom_world_z_m": CHASSIS_BOTTOM_WORLD_Z_M,
                "composed_com_xyz_m": list(com),
            },
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
            "wheel_caster_collision_group": WHEEL_CASTER_COLLISION_GROUP,
            "collision_group_beta": COLLISION_GROUP_BETA,
            "excluded_contacts": list(EXCLUDED_CONTACTS),
            "unverified_until_g4_g12": list(UNVERIFIED_UNTIL_G12),
            "drive_joint_positive_rolls_forward": DRIVE_JOINT_POSITIVE_ROLL_FORWARD,
        },
        "primitive_parts": [
            {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in part.items()
            }
            for part in parts
        ],
    }
    manifest["artifact_provenance"] = artifact_provenance(spec_manifest=manifest)
    return manifest


def asset_manifest_sha256() -> str:
    payload = json.dumps(asset_manifest(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def isaaclab_dashgo_articulation_config():
    """Build the Isaac Lab articulation configuration (target stack only).

    Fails closed when Isaac Lab is unavailable. On the target stack this returns
    a real ArticulationCfg assembled from ``asset_builder.articulation_cfg_source``;
    it never returns a hash-only placeholder.
    """
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
    except Exception as exc:
        raise RuntimeError(
            "blocked: Isaac Lab is unavailable on this host; the DashGo "
            "articulation configuration requires the pinned target stack"
        ) from exc
    from asset_builder import articulation_cfg_source, validate_articulation_cfg_source

    source = articulation_cfg_source()
    validate_articulation_cfg_source(source)
    wheel = source["actuators"]["wheel_drive"]
    return ArticulationCfg(
        prim_path=source["prim_path"],
        spawn=sim_utils.UsdFileCfg(
            usd_path=source["spawn"]["usd_path"],
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(
                mass=float(source["spawn"]["mass_props"]["mass"]),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(source["init_state"]["pos"]),
            rot=tuple(source["init_state"]["rot"]),
            lin_vel=tuple(source["init_state"]["lin_vel"]),
            ang_vel=tuple(source["init_state"]["ang_vel"]),
            joint_pos=dict(source["init_state"]["joint_pos"]),
            joint_vel=dict(source["init_state"]["joint_vel"]),
        ),
        actuators={
            "wheel_drive": ImplicitActuatorCfg(
                joint_names_expr=list(wheel["joint_names_expr"]),
                stiffness=float(wheel["stiffness"]),
                damping=float(wheel["damping"]),
                effort_limit_sim=float(wheel["effort_limit_sim"]),
                velocity_limit_sim=float(wheel["velocity_limit_sim"]),
            ),
        },
        soft_joint_pos_limit_factor=float(source["soft_joint_pos_limit_factor"]),
    )


def isaaclab_sea_raycaster_pattern():
    """Explicit 41-angle raycaster pattern (-120..120 deg, 6 deg spacing)."""
    from ray_pattern import isaaclab_sea_41_ray_pattern_cfg, pattern_func_cpu
    try:
        import isaaclab  # noqa: F401
        import torch
    except Exception as exc:
        raise RuntimeError(
            "blocked: Isaac Lab is unavailable on this host; the raycaster "
            "pattern requires the pinned target stack"
        ) from exc
    # Target stack: return live cfg; also expose torch directions for callers.
    cfg = isaaclab_sea_41_ray_pattern_cfg()
    starts, directions = cfg.func(cfg, device="cpu")
    del starts
    return directions
