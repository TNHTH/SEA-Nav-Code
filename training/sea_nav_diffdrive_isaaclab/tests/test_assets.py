# SPDX-License-Identifier: MIT
"""Identity tests for the SEA-owned DashGo primitive asset manifest."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/sea_nav_core/src"))

import assets as A  # noqa: E402
from sea_nav_core import sea_ray_angles_deg  # noqa: E402


def test_manifest_freezes_the_candidate_geometry_and_provenance():
    manifest = A.asset_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["platform_profile"] == "dashgo_d1_primitive_candidate_v1"
    assert manifest["provenance"]["dashgo_commit"] == A.DASHGO_COMMIT
    assert set(manifest["provenance"]["source_sha256"]) == set(A.SOURCE_HASHES)
    assumptions = manifest["simulation_operational_assumptions"]
    assert assumptions["v_range_mps"] == [-0.15, 0.30]
    assert assumptions["omega_range_radps"] == [-1.0, 1.0]
    assert assumptions["ray_count"] == 41
    assert assumptions["ray_angles_deg"] == list(sea_ray_angles_deg())
    assert assumptions["ray_range_m"] == [0.1, 3.0]
    assert assumptions["actuator_search_order_effort_damping"] == [
        [20.0, 2.0], [20.0, 5.0], [20.0, 10.0],
        [50.0, 2.0], [50.0, 5.0], [50.0, 10.0],
    ]
    assert assumptions["friction_randomization_kind"] == "uniform_half_open"
    assert assumptions["friction_randomization"] == [0.2, 1.25]
    assert assumptions["collision_group"] == "dashgo_chassis_lidar_structure"
    assert assumptions["excluded_contacts"] == ["wheel_floor", "caster_floor"]
    parts = {part["name"]: part for part in manifest["primitive_parts"]}
    assert parts["chassis"]["radius_m"] == pytest.approx(0.203)
    # Scientific §5: +y left / -y right (reject the historical inverted signs).
    assert parts["wheel_left"]["origin_xyz_m"][1] == pytest.approx(+0.171)
    assert parts["wheel_right"]["origin_xyz_m"][1] == pytest.approx(-0.171)
    assert parts["wheel_left"]["origin_xyz_m"][2] == pytest.approx(0.0)
    assert parts["wheel_right"]["origin_xyz_m"][2] == pytest.approx(0.0)
    assert parts["caster_front"]["origin_xyz_m"][2] == pytest.approx(0.03 - 0.0632)
    assert parts["chassis"]["origin_xyz_m"][2] == pytest.approx(
        0.04 + 0.21 / 2.0 - 0.0632
    )
    assert parts["lidar_frame"]["origin_xyz_m"] == [0.0, 0.0, 0.13]
    assert parts["lidar_frame"]["mass_kg"] == 0.0


def test_mass_breakdown_inertia_and_com_are_analytical():
    manifest = A.asset_manifest()
    breakdown = manifest["mechanical_fact_candidates"]["mass_breakdown_kg"]
    assert breakdown == {
        "chassis": 12.7,
        "wheel_left": 0.4,
        "wheel_right": 0.4,
        "caster_front": 0.1,
        "caster_rear": 0.1,
        "lidar_frame": 0.0,
    }
    assert sum(breakdown.values()) == pytest.approx(13.7)
    parts = {part["name"]: part for part in manifest["primitive_parts"]}
    chassis_i = A._cylinder_inertia_diag(12.7, 0.203, 0.21, "z")
    assert parts["chassis"]["inertia_diag_kg_m2"] == list(chassis_i)
    wheel_i = A._cylinder_inertia_diag(0.4, 0.0632, 0.04, "y")
    assert parts["wheel_left"]["inertia_diag_kg_m2"] == list(wheel_i)
    caster_i = A._sphere_inertia_diag(0.1, 0.03)
    assert parts["caster_front"]["inertia_diag_kg_m2"] == list(caster_i)
    com = manifest["mechanical_fact_candidates"]["base_frame"]["composed_com_xyz_m"]
    assert com == list(A.composed_center_of_mass_xyz_m())
    # Symmetric left/right → COM y = 0.
    assert com[1] == pytest.approx(0.0)
    # Chassis bottom at world z=0.04 when base reset z=wheel radius.
    assert manifest["mechanical_fact_candidates"]["base_frame"][
        "chassis_bottom_world_z_m"
    ] == pytest.approx(0.04)


def test_wheel_identity_rejects_inverted_or_missing_wheels():
    good = A.wheel_identity_or_raise()
    assert good["wheel_left"]["origin_xyz_m"][1] > 0
    assert good["wheel_right"]["origin_xyz_m"][1] < 0
    inverted = copy.deepcopy(list(A.PRIMITIVE_PARTS))
    for part in inverted:
        if part["name"] == "wheel_left":
            part["origin_xyz_m"] = (0.0, -0.171, 0.0)
        if part["name"] == "wheel_right":
            part["origin_xyz_m"] = (0.0, 0.171, 0.0)
    with pytest.raises(ValueError, match="positive y"):
        A.wheel_identity_or_raise(inverted)
    missing = [part for part in A.PRIMITIVE_PARTS if part["name"] != "wheel_left"]
    with pytest.raises(ValueError, match="missing driven wheel"):
        A.wheel_identity_or_raise(missing)
    duplicate = list(A.PRIMITIVE_PARTS) + [copy.deepcopy(good["wheel_left"])]
    with pytest.raises(ValueError, match="duplicate"):
        A.wheel_identity_or_raise(duplicate)


def test_manifest_is_deterministic_and_hash_stable():
    first = A.asset_manifest_sha256()
    second = A.asset_manifest_sha256()
    assert first == second and len(first) == 64


def test_committed_manifest_file_matches_the_module():
    path = PACKAGE_ROOT / "assets/asset_manifest.json"
    assert path.is_file(), "asset manifest must be committed"
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == A.asset_manifest()


def test_isaaclab_factories_fail_closed_without_the_target_stack():
    with pytest.raises(RuntimeError, match="blocked"):
        A.isaaclab_dashgo_articulation_config()
    with pytest.raises(RuntimeError, match="blocked"):
        A.isaaclab_sea_raycaster_pattern()


def test_package_import_does_not_load_isaac_ros_or_dashgo():
    import subprocess
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(PACKAGE_ROOT) + "');"
        "sys.path.insert(0, r'" + str(REPO_ROOT / "packages/sea_nav_core/src") + "');"
        "import assets, room_mesh, fixtures, asset_builder, scene_builder, ray_pattern;"
        "loaded = {n.split('.')[0] for n in sys.modules};"
        "forbidden = {'isaacsim','isaaclab','isaacgym','rclpy','torch'};"
        "assert not (loaded & forbidden), sorted(loaded & forbidden)"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
