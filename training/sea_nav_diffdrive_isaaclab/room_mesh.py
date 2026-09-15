# SPDX-License-Identifier: MIT
"""One shared static triangle mesh for the 10 m x 10 m SEA evaluation room.

Floor slab, four boundary walls, and every occupied non-wall fine cell become
a single deterministic mesh used by BOTH PhysX collision and the RayCaster
(``raycaster_and_collision_share_mesh_identity``).  The wall ring is emitted
as four disjoint boxes (an exact union decomposition: overlapping wall corners
are not double counted) and interior obstacles are merged into maximal
row-wise runs of coarse cells, so no two emitted boxes overlap in volume.

Pure Python; loads neither Isaac, ROS, Torch, nor DashGo packages.
"""

import hashlib
from typing import Dict, List, Sequence, Tuple

ROOM_HALF_M = 5.0
WALL_THICKNESS_M = 0.5
MESH_HEIGHT_M = 1.0
FLOOR_BOTTOM_M = -0.1
FINE_SHAPE = (100, 100)
FINE_CELL_M = 0.1
# Fine cells whose closed prisms lie inside the wall band are wall cells.
WALL_BAND_CELLS = 5

_BOX_FACES = (
    (0, 1, 3, 2), (4, 6, 7, 5),  # -z, +z
    (0, 4, 5, 1), (2, 3, 7, 6),  # -y, +y
    (0, 2, 6, 4), (1, 5, 7, 3),  # -x, +x
)


def _box_to_triangles(x_range, y_range, z_range) -> List[Tuple[int, int, int]]:
    """Twelve outward-wound triangles of an axis-aligned closed box."""
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    corners = (
        (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1),
    )
    triangles: List[Tuple[int, int, int]] = []
    for a, b, c, d in _BOX_FACES:
        # Reversed quad order: normals point outward (positive box volume).
        triangles.append((corners[a], corners[c], corners[b]))
        triangles.append((corners[a], corners[d], corners[c]))
    return triangles


def _wall_boxes() -> List[Tuple[Tuple[float, float], Tuple[float, float],
                                Tuple[float, float]]]:
    """Four disjoint wall boxes whose union is the closed wall ring."""
    outer = (-ROOM_HALF_M, ROOM_HALF_M)
    inner = (-ROOM_HALF_M + WALL_THICKNESS_M, ROOM_HALF_M - WALL_THICKNESS_M)
    return [
        ((-ROOM_HALF_M, -ROOM_HALF_M + WALL_THICKNESS_M), outer, (0.0, MESH_HEIGHT_M)),  # west
        ((ROOM_HALF_M - WALL_THICKNESS_M, ROOM_HALF_M), outer, (0.0, MESH_HEIGHT_M)),    # east
        (inner, (-ROOM_HALF_M, -ROOM_HALF_M + WALL_THICKNESS_M), (0.0, MESH_HEIGHT_M)),  # south
        (inner, (ROOM_HALF_M - WALL_THICKNESS_M, ROOM_HALF_M), (0.0, MESH_HEIGHT_M)),    # north
    ]


def obstacle_boxes(coarse_occupancy: Sequence[Sequence[int]]) -> List[Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Maximal row-wise runs of occupied interior coarse cells as boxes.

    Each coarse cell is a 0.5 m square scaled from the fine grid; adjacent
    occupied coarse cells in one row merge into a single box so face-touching
    duplicates collapse.  Cells inside the wall band never become obstacle
    prisms (they are already wall volume).
    """
    boxes = []
    rows = len(coarse_occupancy)
    for r in range(rows):
        row = coarse_occupancy[r]
        cols = len(row)
        run_start = None
        for c in range(cols + 1):
            # The coarse border ring is wall volume, never an obstacle prism.
            occupied = (c < cols and row[c] == 1
                        and r not in (0, rows - 1) and c not in (0, cols - 1))
            if occupied and run_start is None:
                run_start = c
            elif not occupied and run_start is not None:
                x0 = -ROOM_HALF_M + run_start * 5 * FINE_CELL_M
                x1 = -ROOM_HALF_M + c * 5 * FINE_CELL_M
                y0 = -ROOM_HALF_M + r * 5 * FINE_CELL_M
                y1 = y0 + 5 * FINE_CELL_M
                boxes.append(((x0, x1), (y0, y1), (0.0, MESH_HEIGHT_M)))
                run_start = None
    return boxes


def build_room_triangles(coarse_occupancy: Sequence[Sequence[int]]) -> List[
        Tuple[int, int, int]]:
    """Deterministic triangle list: floor slab + walls + obstacle prisms."""
    if (len(coarse_occupancy), len(coarse_occupancy[0])) != (20, 20):
        raise ValueError("coarse_occupancy must be the 20x20 grid")
    triangles: List[Tuple[int, int, int]] = []
    slab = ((-ROOM_HALF_M, ROOM_HALF_M), (-ROOM_HALF_M, ROOM_HALF_M),
            (FLOOR_BOTTOM_M, 0.0))
    triangles.extend(_box_to_triangles(*slab))
    for box in _wall_boxes():
        triangles.extend(_box_to_triangles(*box))
    for box in obstacle_boxes(coarse_occupancy):
        triangles.extend(_box_to_triangles(*box))
    return triangles


def canonical_mesh_text(triangles: Sequence[Tuple[int, int, int]]) -> str:
    """Stable textual form: integer millimetres, fixed triangle order."""
    lines = ["# sea_room_static_mesh_v1"]
    for a, b, c in triangles:
        for vertex in (a, b, c):
            lines.append("v %d %d %d" % tuple(round(value * 1000) for value in vertex))
    for index in range(0, len(triangles) * 3, 3):
        lines.append("f %d %d %d" % (index + 1, index + 2, index + 3))
    return "\n".join(lines) + "\n"


def room_mesh_sha256(coarse_occupancy: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(
        canonical_mesh_text(build_room_triangles(coarse_occupancy)).encode("ascii")
    ).hexdigest()


def write_obj(coarse_occupancy: Sequence[Sequence[int]], path: str) -> Dict[str, int]:
    text = canonical_mesh_text(build_room_triangles(coarse_occupancy))
    with open(path, "w", encoding="ascii") as handle:
        handle.write(text)
    triangle_count = len(build_room_triangles(coarse_occupancy))
    return {
        "triangles": triangle_count,
        "vertices": triangle_count * 3,
        "sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
    }
