# SPDX-License-Identifier: MIT
"""Determinism and geometry tests for the shared static room mesh."""

import hashlib
from pathlib import Path
import sys

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import room_mesh as RM  # noqa: E402


def _empty_room():
    return [
        [1 if (r in (0, 19) or c in (0, 19)) else 0 for c in range(20)]
        for r in range(20)
    ]


def test_mesh_is_deterministic():
    first = RM.canonical_mesh_text(RM.build_room_triangles(_empty_room()))
    second = RM.canonical_mesh_text(RM.build_room_triangles(_empty_room()))
    assert first == second
    assert RM.room_mesh_sha256(_empty_room()) == hashlib.sha256(
        first.encode("ascii")).hexdigest()


def test_empty_room_has_floor_and_four_walls_only():
    boxes = RM.obstacle_boxes(_empty_room())
    assert boxes == []
    triangles = RM.build_room_triangles(_empty_room())
    # floor + 4 walls = 5 boxes = 60 triangles
    assert len(triangles) == 5 * 12


def test_wall_ring_boxes_are_disjoint_and_inside_room():
    boxes = RM._wall_boxes()
    assert len(boxes) == 4
    for (x0, x1), (y0, y1), (z0, z1) in boxes:
        assert -5.0 <= x0 < x1 <= 5.0
        assert -5.0 <= y0 < y1 <= 5.0
        assert (z0, z1) == (0.0, 1.0)
    for i in range(4):
        for j in range(i + 1, 4):
            (ax0, ax1), (ay0, ay1), _ = boxes[i]
            (bx0, bx1), (by0, by1), _ = boxes[j]
            overlap_x = min(ax1, bx1) - max(ax0, bx0)
            overlap_y = min(ay1, by1) - max(ay0, by0)
            assert overlap_x <= 0 or overlap_y <= 0, "wall boxes must not overlap"


def test_adjacent_obstacle_cells_merge_into_runs():
    grid = _empty_room()
    grid[10][5] = 1
    grid[10][6] = 1
    grid[10][7] = 1
    grid[12][5] = 1
    boxes = RM.obstacle_boxes(grid)
    assert len(boxes) == 2
    run = boxes[0]
    assert run[0] == (-5.0 + 5 * 0.1 * 5, -5.0 + 8 * 0.1 * 5)  # 0.5 m-wide run
    assert run[1] == (-5.0 + 10 * 0.5, -5.0 + 11 * 0.5)
    assert run[2] == (0.0, 1.0)
    triangles = RM.build_room_triangles(grid)
    assert len(triangles) == (5 + 2) * 12


def test_mesh_triangles_are_outward_wound_positive_volume():
    def signed_volume(triangles):
        total = 0.0
        for (ax, ay, az), (bx, by, bz), (cx, cy, cz) in triangles:
            total += (ax * (by * cz - bz * cy)
                      - ay * (bx * cz - bz * cx)
                      + az * (bx * cy - by * cx)) / 6.0
        return total
    triangles = RM.build_room_triangles(_empty_room())
    volume = signed_volume(triangles)
    # floor slab 10x10x0.1 = 10 m^3 plus wall ring 19 m^2 x 1 m = 19 m^3
    assert volume == pytest.approx(29.0, rel=1e-9)


def test_bad_grid_shape_fails_closed():
    with pytest.raises(ValueError):
        RM.build_room_triangles([[0] * 10] * 10)


def test_obj_writer_roundtrip(tmp_path):
    grid = _empty_room()
    grid[8][8] = 1
    info = RM.write_obj(grid, str(tmp_path / "room.obj"))
    text = (tmp_path / "room.obj").read_text(encoding="ascii")
    assert info["triangles"] == 6 * 12
    assert info["sha256"] == RM.room_mesh_sha256(grid)
    assert text.startswith("# sea_room_static_mesh_v1")
