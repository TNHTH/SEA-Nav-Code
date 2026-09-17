# SPDX-License-Identifier: MIT
"""SEA 41-ray horizontal pattern for Isaac Lab RayCaster (CPU-static source).

On CPU this module exposes the pattern math and a cfg *source* dict matching
Isaac Lab 2.0.2 ``PatternBaseCfg`` field names. Instantiating the live cfg
requires Isaac and stays fail-closed without it.

Import discipline: no Torch/Isaac at module load. Angle table matches
``sea_nav_core.observation.sea_ray_angles_*`` bit-for-bit.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

RAY_COUNT = 41
RAY_FOV_DEG = (-120.0, 120.0)
RAY_STEP_DEG = 6.0
MAX_DISTANCE_M = 3.0
ATTACH_YAW_ONLY = False
NO_HIT_SENTINEL = "range_equals_max_distance"


def sea_ray_angles_deg() -> Tuple[float, ...]:
    start, _end = RAY_FOV_DEG
    return tuple(start + RAY_STEP_DEG * index for index in range(RAY_COUNT))


def sea_ray_angles_rad() -> Tuple[float, ...]:
    return tuple(math.radians(angle) for angle in sea_ray_angles_deg())


def sea_41_ray_directions_xyz(
        angles_rad: Sequence[float] | None = None,
) -> List[Tuple[float, float, float]]:
    angles = list(angles_rad) if angles_rad is not None else list(sea_ray_angles_rad())
    if len(angles) != RAY_COUNT:
        raise ValueError("SEA ray pattern must be exactly 41 angles")
    return [
        (math.cos(angle), math.sin(angle), 0.0)
        for angle in angles
    ]


def sea_41_ray_starts_xyz() -> List[Tuple[float, float, float]]:
    return [(0.0, 0.0, 0.0)] * RAY_COUNT


def pattern_func_cpu(cfg=None, device=None):
    """CPU stand-in for PatternBaseCfg.func(cfg, device) → starts, directions."""
    del cfg, device
    return sea_41_ray_starts_xyz(), sea_41_ray_directions_xyz()


def ray_pattern_cfg_source() -> Dict:
    """Inspectable PatternBaseCfg / RayCasterCfg source structure."""
    return {
        "pattern": {
            "class_type": "Sea41RayPatternCfg",
            "bases": ["PatternBaseCfg"],
            "func": "sea_nav_diffdrive_isaaclab.ray_pattern.pattern_func_cpu",
            "ray_count": RAY_COUNT,
            "angles_rad": list(sea_ray_angles_rad()),
        },
        "ray_caster": {
            "attach_yaw_only": ATTACH_YAW_ONLY,
            "max_distance": MAX_DISTANCE_M,
            "update_period": 0.0,
            "mesh_identity": "shared_with_collision_global_static_mesh",
            "no_hit_policy": NO_HIT_SENTINEL,
            "freshness": "force_read_latest_at_policy_tick",
        },
    }


def validate_ray_pattern_cfg_source(source: Dict) -> None:
    pattern = source.get("pattern") or {}
    caster = source.get("ray_caster") or {}
    if pattern.get("bases") != ["PatternBaseCfg"]:
        raise ValueError("must inherit PatternBaseCfg, not legacy PatternCfg")
    if pattern.get("ray_count") != RAY_COUNT:
        raise ValueError("ray pattern must declare 41 rays")
    if caster.get("attach_yaw_only") is not False:
        raise ValueError("attach_yaw_only must be False for horizontal rays")
    if float(caster.get("max_distance", -1)) != MAX_DISTANCE_M:
        raise ValueError("max_distance must be 3.0 m")
    starts, directions = pattern_func_cpu()
    if len(starts) != RAY_COUNT or len(directions) != RAY_COUNT:
        raise ValueError("func must return [41,3] starts and directions")
    if any(len(item) != 3 for item in starts + directions):
        raise ValueError("starts/directions must be XYZ triples")
    if not math.isclose(directions[RAY_COUNT // 2][0], 1.0, abs_tol=1e-12):
        raise ValueError("center ray must point +x at yaw 0")
    left = directions[35]
    right = directions[5]
    if not (math.isclose(left[0], 0.0, abs_tol=1e-12)
            and math.isclose(left[1], 1.0, abs_tol=1e-12)):
        raise ValueError("+90 deg ray must point +y")
    if not (math.isclose(right[0], 0.0, abs_tol=1e-12)
            and math.isclose(right[1], -1.0, abs_tol=1e-12)):
        raise ValueError("-90 deg ray must point -y")


def isaaclab_sea_41_ray_pattern_cfg():
    """Instantiate Sea41RayPatternCfg on the target Isaac Lab stack only."""
    try:
        from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
        import torch
    except Exception as exc:
        raise RuntimeError(
            "blocked: Isaac Lab is unavailable on this host; Sea41RayPatternCfg "
            "requires the pinned target stack"
        ) from exc

    class Sea41RayPatternCfg(PatternBaseCfg):
        """Horizontal 41-ray pattern: directions = [cos θ, sin θ, 0]."""

        def func(self, cfg, device):
            starts = torch.zeros((RAY_COUNT, 3), device=device, dtype=torch.float32)
            angles = torch.tensor(sea_ray_angles_rad(), device=device, dtype=torch.float32)
            directions = torch.stack(
                (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)),
                dim=-1,
            )
            return starts, directions

    source = ray_pattern_cfg_source()
    validate_ray_pattern_cfg_source(source)
    return Sea41RayPatternCfg()
