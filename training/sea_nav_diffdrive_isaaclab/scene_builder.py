# SPDX-License-Identifier: MIT
"""Global static scene bank: training tiles + frozen eval fixtures.

Builds one merged triangle mesh identity shared by collision and RayCaster.
Training rooms are 100 fixed hard_room tiles on a 10x10 lattice (12 m centers).
Eval rooms reuse the frozen 300-fixture payload and only select origins — they
never regenerate fixture occupancy. Pure Python; no Isaac/Torch/ROS import.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import fixtures as F
import room_mesh as RM

PACKAGE_ROOT = Path(__file__).resolve().parent
TRAINING_STREAM = "sea_nav_dashgo_training_hard_room_v1"
TRAINING_TILE_COUNT = 100
TRAINING_GRID = (10, 10)
TILE_CENTER_SPACING_M = 12.0
EVAL_BANK_SIZE = 300
FACE_FLOOR = "floor"
FACE_WALL = "wall"
FACE_OBSTACLE = "obstacle"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def training_map_seed(index: int) -> bytes:
    if not 0 <= index < TRAINING_TILE_COUNT:
        raise ValueError("training map index out of range")
    payload = "{}|{:04d}".format(TRAINING_STREAM, index).encode("utf-8")
    return hashlib.sha256(payload).digest()


def generate_training_coarse_occupancy(index: int) -> List[List[int]]:
    """Deterministic 20x20 hard_room coarse grid for one training tile."""
    digest = training_map_seed(index)
    # Border walls always occupied.
    grid = [
        [1 if (r in (0, 19) or c in (0, 19)) else 0 for c in range(20)]
        for r in range(20)
    ]
    # Stamp a fixed number of interior obstacle cells from the digest stream.
    cursor = 0

    def next_byte() -> int:
        nonlocal cursor, digest
        if cursor >= len(digest):
            digest = hashlib.sha256(digest).digest()
            cursor = 0
        value = digest[cursor]
        cursor += 1
        return value

    obstacle_budget = 8 + (next_byte() % 9)  # 8..16 interior cells
    placed = 0
    attempts = 0
    while placed < obstacle_budget and attempts < 512:
        attempts += 1
        r = 1 + (next_byte() % 18)
        c = 1 + (next_byte() % 18)
        if grid[r][c] == 0:
            grid[r][c] = 1
            placed += 1
    return grid


def tile_origin_xy_m(index: int) -> Tuple[float, float]:
    """World XY of tile center on the 10x10 lattice with 12 m spacing."""
    if not 0 <= index < TRAINING_TILE_COUNT:
        raise ValueError("training map index out of range")
    rows, cols = TRAINING_GRID
    row = index // cols
    col = index % cols
    # Center the lattice about the world origin.
    x = (col - (cols - 1) / 2.0) * TILE_CENTER_SPACING_M
    y = (row - (rows - 1) / 2.0) * TILE_CENTER_SPACING_M
    return (float(x), float(y))


def translate_triangles(
        triangles: Sequence[Tuple[Tuple[float, float, float],
                                  Tuple[float, float, float],
                                  Tuple[float, float, float]]],
        origin_xy: Tuple[float, float],
) -> List[Tuple[Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float]]]:
    ox, oy = origin_xy
    moved = []
    for a, b, c in triangles:
        moved.append((
            (a[0] + ox, a[1] + oy, a[2]),
            (b[0] + ox, b[1] + oy, b[2]),
            (c[0] + ox, c[1] + oy, c[2]),
        ))
    return moved


def _categorize_room_triangles(
        coarse: Sequence[Sequence[int]],
) -> List[Tuple[str, Tuple]]:
    """Emit (face_category, triangle) in floor→wall→obstacle order."""
    categorized: List[Tuple[str, Tuple]] = []
    slab = ((-RM.ROOM_HALF_M, RM.ROOM_HALF_M), (-RM.ROOM_HALF_M, RM.ROOM_HALF_M),
            (RM.FLOOR_BOTTOM_M, 0.0))
    for tri in RM._box_to_triangles(*slab):
        categorized.append((FACE_FLOOR, tri))
    for box in RM._wall_boxes():
        for tri in RM._box_to_triangles(*box):
            categorized.append((FACE_WALL, tri))
    for box in RM.obstacle_boxes(coarse):
        for tri in RM._box_to_triangles(*box):
            categorized.append((FACE_OBSTACLE, tri))
    return categorized


def build_training_bank() -> Dict:
    rooms = []
    for index in range(TRAINING_TILE_COUNT):
        coarse = generate_training_coarse_occupancy(index)
        origin = tile_origin_xy_m(index)
        local = _categorize_room_triangles(coarse)
        world_tris = []
        face_categories = []
        for category, tri in local:
            moved = translate_triangles([tri], origin)[0]
            world_tris.append(moved)
            face_categories.append(category)
        rooms.append({
            "index": index,
            "origin_xy_m": list(origin),
            "coarse_occupancy": coarse,
            "triangle_count": len(world_tris),
            "mesh_sha256_local": RM.room_mesh_sha256(coarse),
        })
    return {
        "kind": "training_hard_room_bank_v1",
        "tile_count": TRAINING_TILE_COUNT,
        "grid": list(TRAINING_GRID),
        "tile_center_spacing_m": TILE_CENTER_SPACING_M,
        "rooms": rooms,
    }


def load_frozen_eval_fixtures(
        fixture_path: Path | None = None,
) -> List[dict]:
    path = fixture_path or (
        PACKAGE_ROOT / "assets/formal_fixtures/dashgo_sea_formal_fixture_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != EVAL_BANK_SIZE:
        raise ValueError("eval bank must be the frozen 300-fixture payload")
    return payload


def build_eval_bank(*, fixture_path: Path | None = None) -> Dict:
    fixtures = load_frozen_eval_fixtures(fixture_path)
    rooms = []
    for index, fixture in enumerate(fixtures):
        coarse = fixture["coarse_occupancy"]
        # Eval bank selects origins from fixture indices; does not regenerate.
        origin = (float(index) * TILE_CENTER_SPACING_M, 0.0)
        rooms.append({
            "index": index,
            "fixture_version": fixture["fixture_version"],
            "difficulty": fixture["difficulty"],
            "fixture_index": fixture["index"],
            "origin_xy_m": list(origin),
            "coarse_occupancy_sha256": hashlib.sha256(
                F.canonical_json_bytes(coarse)
            ).hexdigest(),
            "mesh_sha256_local": RM.room_mesh_sha256(coarse),
        })
    return {
        "kind": "eval_frozen_fixture_bank_v1",
        "room_count": len(rooms),
        "rooms": rooms,
        "regenerates_fixtures": False,
    }


def merge_global_static_mesh(
        bank: Mapping,
        *,
        include_geometry: bool = True,
) -> Dict:
    """Merge bank rooms into one static mesh with face category mapping."""
    kind = bank["kind"]
    triangles: List[Tuple] = []
    face_categories: List[str] = []
    origin_map: List[Dict] = []
    if kind == "training_hard_room_bank_v1":
        for room in bank["rooms"]:
            coarse = room["coarse_occupancy"]
            origin = tuple(room["origin_xy_m"])
            local = _categorize_room_triangles(coarse)
            for category, tri in local:
                if include_geometry:
                    triangles.append(translate_triangles([tri], origin)[0])
                face_categories.append(category)
            origin_map.append({
                "index": room["index"],
                "origin_xy_m": list(origin),
            })
    elif kind == "eval_frozen_fixture_bank_v1":
        fixtures = load_frozen_eval_fixtures()
        for room, fixture in zip(bank["rooms"], fixtures):
            coarse = fixture["coarse_occupancy"]
            origin = tuple(room["origin_xy_m"])
            local = _categorize_room_triangles(coarse)
            for category, tri in local:
                if include_geometry:
                    triangles.append(translate_triangles([tri], origin)[0])
                face_categories.append(category)
            origin_map.append({
                "index": room["index"],
                "origin_xy_m": list(origin),
                "fixture_index": room["fixture_index"],
                "difficulty": room["difficulty"],
            })
    else:
        raise ValueError("unsupported scene bank kind")
    mesh_text = RM.canonical_mesh_text(triangles) if include_geometry else ""
    return {
        "kind": "global_static_triangle_mesh_v1",
        "source_bank_kind": kind,
        "triangle_count": len(face_categories),
        "face_categories": face_categories if include_geometry else [
            "omitted_for_hash_only"
        ],
        "origin_map": origin_map,
        "mesh_sha256": _sha256_text(mesh_text) if include_geometry else None,
        "raycaster_and_collision_share_mesh_identity": True,
        "robot_robot_collision_filter": "all_pairs_disabled",
        "runtime_vertex_mutation": False,
    }


def assert_metric_box_size_invariant() -> None:
    """A declared 1 m axis-aligned prism must remain 1 m before import."""
    box = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    tris = RM._box_to_triangles(*box)
    xs = [v[0] for t in tris for v in t]
    ys = [v[1] for t in tris for v in t]
    zs = [v[2] for t in tris for v in t]
    if abs(max(xs) - min(xs) - 1.0) > 1e-12:
        raise ValueError("1m box x extent corrupted")
    if abs(max(ys) - min(ys) - 1.0) > 1e-12:
        raise ValueError("1m box y extent corrupted")
    if abs(max(zs) - min(zs) - 1.0) > 1e-12:
        raise ValueError("1m box z extent corrupted")


def frozen_fixture_aggregate_sha256(
        hashes_path: Path | None = None,
) -> str:
    path = hashes_path or (
        PACKAGE_ROOT / "assets/formal_fixtures/formal_fixture_hashes.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest["aggregate_payload_sha256"]
