# SPDX-License-Identifier: MIT
"""Deterministic DashGo primitive USD + ArticulationCfg source structure.

CPU-static: no Isaac/pxr import at module load. Generates ASCII USDA from
``assets.primitive_parts()`` and an inspectable ArticulationCfg source tree
aligned with Isaac Lab 2.0.2 field names. Target-stack instantiation stays in
``assets.isaaclab_dashgo_articulation_config``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import assets as _assets

PACKAGE_ROOT = Path(__file__).resolve().parent
GENERATED_USD_RELATIVE = Path("assets/generated/dashgo_primitive_candidate_v1.usda")
GENERATOR_ID = "sea_nav_dashgo_primitive_usd_v1"


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generator_source_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def spec_sha256(manifest: Mapping | None = None) -> str:
    if manifest is None:
        raise ValueError("spec_sha256 requires an explicit manifest without artifact side effects")
    stripped = dict(manifest)
    stripped.pop("artifact_provenance", None)
    return _sha256_bytes(_canonical_json(stripped))


def generated_usd_path(*, package_root: Path | None = None) -> Path:
    root = Path(package_root) if package_root is not None else PACKAGE_ROOT
    return (root / GENERATED_USD_RELATIVE).resolve()


def _fmt_triple(values: Sequence[float]) -> str:
    return "({:.12g}, {:.12g}, {:.12g})".format(*values)


def render_dashgo_usda(parts: Sequence[Mapping] | None = None) -> str:
    """ASCII USDA for the primitive candidate (deterministic byte order)."""
    table = _assets.wheel_identity_or_raise(parts or _assets.primitive_parts())
    ordered = list(parts or _assets.primitive_parts())
    lines = [
        "#usda 1.0",
        '(',
        '    defaultPrim = "dashgo_primitive"',
        '    metersPerUnit = 1',
        '    upAxis = "Z"',
        '    doc = "SEA DashGo primitive candidate; not calibrated hardware"',
        ')',
        '',
        'def Xform "dashgo_primitive" (',
        '    kind = "component"',
        ')',
        '{',
        '    custom string sea_platform_profile = "{}"'.format(_assets.PLATFORM_PROFILE),
        '    custom bool sea_enabled_self_collisions = 0',
        '',
    ]
    for part in ordered:
        name = part["name"]
        origin = tuple(float(v) for v in part["origin_xyz_m"])
        mass = float(part.get("mass_kg", 0.0))
        inertia = tuple(float(v) for v in part.get("inertia_diag_kg_m2", (0.0, 0.0, 0.0)))
        shape = part["shape"]
        lines.append('    def Xform "{}"'.format(name))
        lines.append("    {")
        lines.append(
            "        double3 xformOp:translate = {}".format(_fmt_triple(origin))
        )
        lines.append('        uniform token[] xformOpOrder = ["xformOp:translate"]')
        lines.append("        custom double sea_mass_kg = {:.12g}".format(mass))
        lines.append(
            "        custom double3 sea_inertia_diag_kg_m2 = {}".format(_fmt_triple(inertia))
        )
        if shape == "cylinder":
            radius = float(part["radius_m"])
            height = float(part.get("height_m", part.get("length_m")))
            axis = part.get("axis", "z")
            lines.append('        def Cylinder "geom"')
            lines.append("        {")
            lines.append("            double radius = {:.12g}".format(radius))
            lines.append("            double height = {:.12g}".format(height))
            lines.append('            uniform token axis = "{}"'.format(axis))
            lines.append("        }")
        elif shape == "sphere":
            radius = float(part["radius_m"])
            lines.append('        def Sphere "geom"')
            lines.append("        {")
            lines.append("            double radius = {:.12g}".format(radius))
            lines.append("        }")
            lines.append("        custom bool sea_actuated = 0")
        elif shape == "frame_only":
            lines.append('        custom string sea_role = "lidar_frame_only"')
            lines.append("        custom double sea_yaw_rad = {:.12g}".format(
                float(part.get("yaw_rad", 0.0))
            ))
        else:
            raise ValueError("unsupported primitive shape: " + shape)
        group = part.get("collision_group")
        if group:
            lines.append(
                '        custom string sea_collision_group = "{}"'.format(group)
            )
        lines.append("    }")
        lines.append("")
    # Driven joints: both wheels roll +x on positive joint velocity.
    for wheel_name in ("wheel_left", "wheel_right"):
        wheel = table[wheel_name]
        lines.append('    def PhysicsRevoluteJoint "{}_joint"'.format(wheel_name))
        lines.append("    {")
        lines.append('        rel physics:body1 = </dashgo_primitive/chassis>')
        lines.append(
            '        rel physics:body0 = </dashgo_primitive/{}>'.format(wheel_name)
        )
        lines.append('        uniform token physics:axis = "Y"')
        lines.append(
            "        custom bool sea_positive_rolls_forward = {}".format(
                1 if wheel.get("drive_joint_positive_rolls_forward", True) else 0
            )
        )
        lines.append("    }")
        lines.append("")
    for caster_name in ("caster_front", "caster_rear"):
        lines.append('    def PhysicsSphericalJoint "{}_joint"'.format(caster_name))
        lines.append("    {")
        lines.append('        rel physics:body1 = </dashgo_primitive/chassis>')
        lines.append(
            '        rel physics:body0 = </dashgo_primitive/{}>'.format(caster_name)
        )
        lines.append("        custom bool sea_actuated = 0")
        lines.append("    }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_dashgo_usd(*, package_root: Path | None = None,
                     parts: Sequence[Mapping] | None = None) -> Path:
    """Write deterministic USDA and return its absolute path."""
    path = generated_usd_path(package_root=package_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_dashgo_usda(parts).encode("utf-8")
    path.write_bytes(payload)
    return path


def usd_file_sha256(*, package_root: Path | None = None,
                    parts: Sequence[Mapping] | None = None) -> str:
    path = write_dashgo_usd(package_root=package_root, parts=parts)
    return _sha256_bytes(path.read_bytes())


def articulation_cfg_source(
        *,
        package_root: Path | None = None,
        usd_path: Path | None = None,
        effort_damping: Tuple[float, float] = (20.0, 2.0),
) -> Dict:
    """Nested ArticulationCfg-shaped dict (Isaac Lab 2.0.2 field names).

    This is CPU-inspectable source structure, not a live Isaac object.
    """
    root = Path(package_root) if package_root is not None else PACKAGE_ROOT
    path = Path(usd_path) if usd_path is not None else write_dashgo_usd(package_root=root)
    if not path.is_file():
        raise ValueError("usd_path must exist")
    # Paths must not depend on process cwd.
    resolved = path.resolve()
    effort, damping = effort_damping
    return {
        "class_type": "Articulation",
        "prim_path": "{ENV_REGEX_NS}/Robot",
        "spawn": {
            "func": "isaaclab.sim.spawners.from_files.spawn_from_usd",
            "usd_path": str(resolved),
            "articulation_props": {
                "enabled_self_collisions": False,
            },
            "activate_contact_sensors": True,
            "mass_props": {
                "mass": _assets.REPORTED_NET_MASS_KG,
            },
        },
        "init_state": {
            "pos": (0.0, 0.0, _assets.BASE_RESET_WORLD_Z_M),
            "rot": (1.0, 0.0, 0.0, 0.0),
            "lin_vel": (0.0, 0.0, 0.0),
            "ang_vel": (0.0, 0.0, 0.0),
            "joint_pos": {
                "wheel_left_joint": 0.0,
                "wheel_right_joint": 0.0,
            },
            "joint_vel": {
                ".*": 0.0,
            },
        },
        "actuators": {
            "wheel_drive": {
                "class_type": "ImplicitActuator",
                "joint_names_expr": ["wheel_left_joint", "wheel_right_joint"],
                "stiffness": _assets.VELOCITY_DRIVE_STIFFNESS,
                "damping": float(damping),
                "effort_limit_sim": float(effort),
                "velocity_limit_sim": _assets.ASSET_WHEEL_LIMIT_RADPS,
            }
            # Casters intentionally omitted: passive spherical joints.
        },
        "soft_joint_pos_limit_factor": 1.0,
    }


def artifact_provenance(*, package_root: Path | None = None,
                        spec_manifest: Mapping | None = None) -> Dict:
    root = Path(package_root) if package_root is not None else PACKAGE_ROOT
    if spec_manifest is None:
        raise ValueError("artifact_provenance requires spec_manifest")
    usd_sha = usd_file_sha256(package_root=root)
    return {
        "generator_id": GENERATOR_ID,
        "generator_sha256": generator_source_sha256(),
        "spec_sha256": spec_sha256(spec_manifest),
        "usd_relative_path": str(GENERATED_USD_RELATIVE).replace("\\", "/"),
        "usd_sha256": usd_sha,
        "articulation_cfg_source_sha256": _sha256_bytes(
            _canonical_json(articulation_cfg_source(package_root=root))
        ),
    }


def validate_articulation_cfg_source(source: Mapping) -> None:
    """Fail-closed structural checks without importing Isaac."""
    spawn = source.get("spawn")
    if not isinstance(spawn, Mapping):
        raise ValueError("ArticulationCfg source missing spawn")
    props = spawn.get("articulation_props") or {}
    if props.get("enabled_self_collisions") is not False:
        raise ValueError("self-collisions must be explicitly disabled")
    usd_path = spawn.get("usd_path")
    if not usd_path or not Path(str(usd_path)).is_absolute():
        raise ValueError("usd_path must be an absolute filesystem path")
    if not Path(str(usd_path)).is_file():
        raise ValueError("usd_path must exist on disk")
    actuators = source.get("actuators")
    if not isinstance(actuators, Mapping) or "wheel_drive" not in actuators:
        raise ValueError("driven wheel ImplicitActuator required")
    for forbidden in ("caster_front", "caster_rear", "caster"):
        for key in actuators:
            if forbidden in str(key):
                raise ValueError("casters must not have actuators")
    joint_names = actuators["wheel_drive"].get("joint_names_expr") or []
    if "wheel_left_joint" not in joint_names or "wheel_right_joint" not in joint_names:
        raise ValueError("both driven wheel joints must be actuated")
