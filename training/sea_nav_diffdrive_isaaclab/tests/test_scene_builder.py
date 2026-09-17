# SPDX-License-Identifier: MIT
"""Tests for training/eval scene banks and global static mesh merge."""

import json
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import fixtures as F  # noqa: E402
import room_mesh as RM  # noqa: E402
import scene_builder as SB  # noqa: E402


def test_training_bank_has_100_tiles_on_12m_lattice():
    bank = SB.build_training_bank()
    assert bank["tile_count"] == 100
    assert bank["tile_center_spacing_m"] == 12.0
    assert len(bank["rooms"]) == 100
    origins = [tuple(room["origin_xy_m"]) for room in bank["rooms"]]
    assert len(set(origins)) == 100
    # Adjacent tile centers along row 0 are 12 m apart.
    assert origins[1][0] - origins[0][0] == pytest.approx(12.0)
    assert origins[0][1] == origins[1][1]


def test_training_maps_are_deterministic_and_metric():
    first = SB.generate_training_coarse_occupancy(7)
    second = SB.generate_training_coarse_occupancy(7)
    assert first == second
    SB.assert_metric_box_size_invariant()
    # Coarse cell is 0.5 m: a single interior obstacle becomes a 0.5 m prism.
    grid = [[0] * 20 for _ in range(20)]
    for c in range(20):
        grid[0][c] = grid[19][c] = 1
        grid[c][0] = grid[c][19] = 1
    grid[5][5] = 1
    boxes = RM.obstacle_boxes(grid)
    assert len(boxes) == 1
    (x0, x1), (y0, y1), _ = boxes[0]
    assert x1 - x0 == pytest.approx(0.5)
    assert y1 - y0 == pytest.approx(0.5)


def test_eval_bank_uses_frozen_fixtures_without_regeneration():
    bank = SB.build_eval_bank()
    assert bank["room_count"] == 300
    assert bank["regenerates_fixtures"] is False
    # Aggregate hash of the formal fixture payload stays frozen.
    assert SB.frozen_fixture_aggregate_sha256().startswith("10202bfce")
    fixtures = SB.load_frozen_eval_fixtures()
    regenerated = F.generate_fixture("easy", 0)
    # Bank occupancy identity matches committed fixture, not a live regen path
    # that could silently drift (regen may match, but bank must read disk).
    assert fixtures[0]["coarse_occupancy"] == json.loads(
        (PACKAGE_ROOT / "assets/formal_fixtures/dashgo_sea_formal_fixture_v1.json")
        .read_text(encoding="utf-8")
    )[0]["coarse_occupancy"]
    assert regenerated["fixture_version"] == fixtures[0]["fixture_version"]


def test_global_mesh_merge_has_face_categories_and_shared_identity():
    bank = SB.build_training_bank()
    # Merge a tiny subset by shrinking rooms for speed.
    tiny = dict(bank)
    tiny["rooms"] = bank["rooms"][:2]
    tiny["tile_count"] = 2
    mesh = SB.merge_global_static_mesh(tiny)
    assert mesh["raycaster_and_collision_share_mesh_identity"] is True
    assert mesh["robot_robot_collision_filter"] == "all_pairs_disabled"
    assert mesh["runtime_vertex_mutation"] is False
    assert mesh["triangle_count"] == len(mesh["face_categories"])
    assert set(mesh["face_categories"]) <= {"floor", "wall", "obstacle"}
    assert mesh["face_categories"].count("floor") == 2 * 12
    assert mesh["mesh_sha256"] and len(mesh["mesh_sha256"]) == 64
    again = SB.merge_global_static_mesh(tiny)
    assert again["mesh_sha256"] == mesh["mesh_sha256"]


def test_scene_builder_import_stays_cpu_safe():
    import subprocess
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(PACKAGE_ROOT) + "');"
        "import scene_builder;"
        "loaded = {n.split('.')[0] for n in sys.modules};"
        "forbidden = {'isaacsim','isaaclab','isaacgym','torch','rclpy'};"
        "assert not (loaded & forbidden), sorted(loaded & forbidden)"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
