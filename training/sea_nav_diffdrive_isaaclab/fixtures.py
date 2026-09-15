# SPDX-License-Identifier: MIT
"""Frozen `dashgo_sea_formal_fixture_v1` generator: 300 paired evaluation cases.

Every degree of freedom is derived from the fixed label
``SEA-Nav|DashGo|formal-eval|v1`` through a SHA-256 counter stream, so the
payload is reproducible bit-for-bit on any host.  Canonical serialization is
RFC 8785 restricted to this schema's value domain (ASCII strings, small
integers, booleans, arrays, objects; floats and nulls are rejected), which is
exactly JCS for that domain.

This module is pure Python (stdlib only) and loads neither Isaac, ROS, Torch,
nor DashGo packages.
"""

import hashlib
import json
from typing import List, Optional, Tuple

FIXTURE_VERSION = "dashgo_sea_formal_fixture_v1"
GENERATOR_ID = "sea_room_static_mesh_v1"
MASTER_SEED_LABEL = "SEA-Nav|DashGo|formal-eval|v1"
DOMAIN_SEPARATION = b"SEA_NAV_DASHGO_FIXTURE_V1\x00"

DIFFICULTY_CODES = {"easy": 1, "medium": 2, "hard": 3}
DIFFICULTY_LEVELS = {"easy": 3, "medium": 6, "hard": 9}
FIXTURES_PER_DIFFICULTY = 100

COARSE_SHAPE = (20, 20)
FINE_SHAPE = (100, 100)
COARSE_CELL_M = 0.5
FINE_CELL_M = 0.1
SCALE = 5  # each coarse cell expands to 5x5 fine cells
ROOM_ORIGIN_M = (-5.0, -5.0)

# Coarse placement window for the local 3x3 mask (top-left row/col in 1..16).
MASK_TOP_LEFT_MAX = 16
# Fine start/goal sampling window (row/col integers 1..98).
FINE_SG_MAX = 98
MAX_EXPANSION_DRAWS_PER_CELL = 128
MAX_START_GOAL_ATTEMPTS = 65536
MAX_SCENE_ATTEMPTS = 4096

CLEARANCE_HALF = 5           # clipped 11x11 clearance square around start/goal
MIN_SG_DISTANCE_CELLS = 35   # strictly greater, Euclidean in fine cells
YAW_MIN_URAD = -3141593
YAW_SPAN = 6283186           # unbiased integer uniform [0, 6283186)
TIMEOUT_S = 30
TERMINAL_CLASSES = ["success", "collision", "timeout"]

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right


def master_seed_uint32() -> int:
    digest = hashlib.sha256(MASTER_SEED_LABEL.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _word(difficulty_code: int, index: int, counter: int) -> int:
    payload = (
        DOMAIN_SEPARATION
        + master_seed_uint32().to_bytes(4, "big")
        + difficulty_code.to_bytes(1, "big")
        + index.to_bytes(2, "big")
        + counter.to_bytes(8, "big")
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


class CaseRng:
    """Per-case SHA-256 counter stream with frozen sampling semantics."""

    def __init__(self, difficulty_code: int, index: int):
        self._difficulty_code = difficulty_code
        self._index = index
        self._counter = 0

    def digest_hex(self) -> str:
        payload = (
            DOMAIN_SEPARATION
            + master_seed_uint32().to_bytes(4, "big")
            + self._difficulty_code.to_bytes(1, "big")
            + self._index.to_bytes(2, "big")
            + self._counter.to_bytes(8, "big")
        )
        return hashlib.sha256(payload).hexdigest()

    def _word(self) -> int:
        return int.from_bytes(bytes.fromhex(self.digest_hex())[:8], "big")

    def next_word(self) -> Tuple[int, str]:
        hexdigest = self.digest_hex()
        word = int.from_bytes(bytes.fromhex(hexdigest)[:8], "big")
        self._counter += 1
        return word, hexdigest

    def uniform(self, low: float, high: float) -> float:
        word, _ = self.next_word()
        return low + (high - low) * (word / 2**64)

    def integer(self, n: int) -> int:
        if n <= 0:
            raise ValueError("integer bound must be positive")
        limit = 2**64 - (2**64 % n)
        while True:
            word, _ = self.next_word()
            if word < limit:
                return word % n


def _occupied_border_coarse() -> List[List[int]]:
    rows, cols = COARSE_SHAPE
    return [
        [1 if (r in (0, rows - 1) or c in (0, cols - 1)) else 0 for c in range(cols)]
        for r in range(rows)
    ]


def _grow_cluster_local(rng: CaseRng, target: int) -> List[Tuple[int, int]]:
    """Grow the cluster shape inside its local 3x3 mask coordinates.

    Draw order follows the authoritative upstream generator: growth draws
    (insertion-ordered base cell, then direction) happen first and stay inside
    the local mask; the placement top-left is drawn afterwards by the caller.
    Exhausting the expansion budget fails closed instead of keeping a
    short cluster.
    """
    cells = [(1, 1)]
    occupied = {(1, 1)}
    while len(cells) < target:
        accepted = False
        draws = 0
        while draws < MAX_EXPANSION_DRAWS_PER_CELL and not accepted:
            base_index = rng.integer(len(cells))   # insertion order
            direction = rng.integer(4)             # up, down, left, right
            draws += 1
            base = cells[base_index]
            candidate = (base[0] + DIRECTIONS[direction][0],
                         base[1] + DIRECTIONS[direction][1])
            if (0 <= candidate[0] < 3 and 0 <= candidate[1] < 3
                    and candidate not in occupied):
                occupied.add(candidate)
                cells.append(candidate)
                accepted = True
        if not accepted:
            raise RuntimeError(
                "cluster growth failed closed after max expansion draws"
            )
    return cells


def _stamp_mask(rng: CaseRng, coarse: List[List[int]],
                cells: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Stamp local mask cells at a drawn top-left (binary max merge)."""
    top = 1 + rng.integer(MASK_TOP_LEFT_MAX)      # top-left in 1..16
    left = 1 + rng.integer(MASK_TOP_LEFT_MAX)
    for (r, c) in cells:
        coarse[top + r][left + c] = max(coarse[top + r][left + c], 1)
    return (top, left)


def _scale_to_fine(coarse: List[List[int]]) -> List[List[int]]:
    fine = [[0] * FINE_SHAPE[1] for _ in range(FINE_SHAPE[0])]
    for r in range(COARSE_SHAPE[0]):
        for c in range(COARSE_SHAPE[1]):
            if coarse[r][c]:
                for dr in range(SCALE):
                    for dc in range(SCALE):
                        fine[r * SCALE + dr][c * SCALE + dc] = 1
    return fine


def _clearance_ok(fine: List[List[int]], row: int, col: int) -> bool:
    rows, cols = FINE_SHAPE
    for r in range(row - CLEARANCE_HALF, row + CLEARANCE_HALF + 1):
        for c in range(col - CLEARANCE_HALF, col + CLEARANCE_HALF + 1):
            if not (0 <= r < rows and 0 <= c < cols):
                continue  # clipped at the grid boundary
            if fine[r][c]:
                return False
    return True


def _bresenham_obstructed(fine: List[List[int]], start: Tuple[int, int],
                          goal: Tuple[int, int]) -> bool:
    """Upstream Bresenham with dx>=dy x-major tie rule; goal cell excluded."""
    (x0, y0), (x1, y1) = start, goal
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx - dy
    x, y = x0, y0
    rows, cols = FINE_SHAPE
    while True:
        if (x, y) != (x1, y1) and fine[x][y]:
            return True
        if (x, y) == (x1, y1):
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        if not (0 <= x < rows and 0 <= y < cols):
            return False


def _connected(fine: List[List[int]], start: Tuple[int, int],
               goal: Tuple[int, int]) -> bool:
    rows, cols = FINE_SHAPE
    if fine[start[0]][start[1]] or fine[goal[0]][goal[1]]:
        return False
    seen = [[False] * cols for _ in range(rows)]
    stack = [start]
    seen[start[0]][start[1]] = True
    while stack:
        r, c = stack.pop()
        if (r, c) == goal:
            return True
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not fine[nr][nc] \
                    and not seen[nr][nc]:
                seen[nr][nc] = True
                stack.append((nr, nc))
    return False


def generate_scene(rng: CaseRng, difficulty_level: int) -> dict:
    coarse = _occupied_border_coarse()
    for _ in range(difficulty_level):
        target = 2 + rng.integer(2)  # uniform {2, 3}
        cells = _grow_cluster_local(rng, target)
        _stamp_mask(rng, coarse, cells)
    for _ in range(5 * difficulty_level):
        _stamp_mask(rng, coarse, [(1, 1)])
    fine = _scale_to_fine(coarse)

    start = goal = None
    for _ in range(MAX_START_GOAL_ATTEMPTS):
        row = 1 + rng.integer(FINE_SG_MAX - 1 + 1)  # uniform [1, 98]
        col = 1 + rng.integer(FINE_SG_MAX - 1 + 1)
        grow = 1 + rng.integer(FINE_SG_MAX - 1 + 1)
        gcol = 1 + rng.integer(FINE_SG_MAX - 1 + 1)
        if not _clearance_ok(fine, row, col) or not _clearance_ok(fine, grow, gcol):
            continue
        distance_sq = (row - grow) ** 2 + (col - gcol) ** 2
        if distance_sq <= MIN_SG_DISTANCE_CELLS ** 2:
            continue
        if not _bresenham_obstructed(fine, (row, col), (grow, gcol)):
            continue
        if not _connected(fine, (row, col), (grow, gcol)):
            continue
        start = (row, col)
        goal = (grow, gcol)
        break
    if start is None:
        raise RuntimeError("fixture generation failed closed: no start/goal")

    yaw_urad = YAW_MIN_URAD + rng.integer(YAW_SPAN)
    return {
        "coarse": coarse,
        "start": start,
        "goal": goal,
        "yaw_urad": yaw_urad,
    }


def generate_fixture(difficulty: str, index: int) -> dict:
    if difficulty not in DIFFICULTY_CODES:
        raise ValueError("difficulty must be easy/medium/hard")
    if not 0 <= index < FIXTURES_PER_DIFFICULTY:
        raise ValueError("index must be in [0, 100)")
    difficulty_code = DIFFICULTY_CODES[difficulty]
    rng = CaseRng(difficulty_code, index)
    level = DIFFICULTY_LEVELS[difficulty]
    last_error: Optional[RuntimeError] = None
    for _ in range(MAX_SCENE_ATTEMPTS):
        try:
            scene = generate_scene(rng, level)
        except RuntimeError as exc:
            last_error = exc
            continue
        return {
            "fixture_version": FIXTURE_VERSION,
            "generator_id": GENERATOR_ID,
            "difficulty": difficulty_code,
            "index": index,
            "coarse_occupancy": scene["coarse"],
            "fine_transform": {
                "shape": list(FINE_SHAPE),
                "scale_each_coarse_cell_to": "5x5_fine_cells",
                "cell_um": 100000,  # 0.1 m in integer micrometres
                "border_fine_cells": 5,
            },
            "start": {"row": scene["start"][0], "col": scene["start"][1]},
            "goal": {"row": scene["goal"][0], "col": scene["goal"][1]},
            "yaw_urad": scene["yaw_urad"],
            "timeout_s": TIMEOUT_S,
            "terminal_classes": list(TERMINAL_CLASSES),
        }
    raise RuntimeError(
        "fixture generation failed closed after max scene attempts: "
        + str(last_error)
    )


def canonical_json_bytes(value) -> bytes:
    """RFC 8785 canonical bytes for this schema's integer/string/bool domain."""
    if value is None or isinstance(value, float):
        raise ValueError("floats and null are outside the canonical domain")
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if isinstance(value, (list, tuple)):
        return b"[" + b",".join(canonical_json_bytes(item) for item in value) + b"]"
    if isinstance(value, dict):
        keys = sorted(value.keys())  # ASCII keys: code-unit order == lexicographic
        return (b"{" + b",".join(
            canonical_json_bytes(key) + b":" + canonical_json_bytes(value[key])
            for key in keys) + b"}")
    raise ValueError("unsupported canonical value type: " + type(value).__name__)


def fixture_sha256(fixture: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(fixture)).hexdigest()


def generate_payload() -> Tuple[List[dict], dict]:
    ordered: List[dict] = []
    for difficulty in ("easy", "medium", "hard"):
        for index in range(FIXTURES_PER_DIFFICULTY):
            ordered.append(generate_fixture(difficulty, index))
    individual = [fixture_sha256(fixture) for fixture in ordered]
    aggregate = hashlib.sha256(canonical_json_bytes(ordered)).hexdigest()
    goldens = {}
    for label, position in (
        ("easy_index_0", 0), ("medium_index_0", 100), ("hard_index_99", 299),
    ):
        goldens[label] = individual[position]
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "generator_id": GENERATOR_ID,
        "master_seed_label": MASTER_SEED_LABEL,
        "master_seed_label_sha256": hashlib.sha256(
            MASTER_SEED_LABEL.encode("utf-8")
        ).hexdigest(),
        "master_seed_uint32_be": master_seed_uint32(),
        "record_order": "difficulty_code_1_2_3_then_index_0_through_99",
        "individual_fixture_sha256_count": len(individual),
        "aggregate_payload_sha256": aggregate,
        "golden_stream_vectors": [
            {"difficulty": "easy", "index": 0, "counter": 0,
             "sha256": CaseRng(1, 0).digest_hex(),
             "word": _word(1, 0, 0)},
            {"difficulty": "easy", "index": 0, "counter": 1,
             "sha256": "", "word": 0},
            {"difficulty": "hard", "index": 99, "counter": 0,
             "sha256": CaseRng(3, 99).digest_hex(),
             "word": _word(3, 99, 0)},
        ],
        "generated_fixture_golden_sha256": goldens,
    }
    # counter==1 vector needs an advanced stream: rebuild without mutation.
    probe = CaseRng(1, 0)
    probe.next_word()
    manifest["golden_stream_vectors"][1]["sha256"] = probe.digest_hex()
    manifest["golden_stream_vectors"][1]["word"] = int.from_bytes(
        bytes.fromhex(probe.digest_hex())[:8], "big"
    )
    return ordered, manifest


def write_payload(fixture_path: str, hashes_path: str) -> dict:
    ordered, manifest = generate_payload()
    with open(fixture_path, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=1, sort_keys=True)
        handle.write("\n")
    manifest["individual_fixture_sha256"] = [
        fixture_sha256(fixture) for fixture in ordered
    ]
    with open(hashes_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return manifest
