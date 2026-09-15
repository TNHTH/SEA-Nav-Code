# SPDX-License-Identifier: MIT
"""Golden and property tests for the frozen formal fixture generator."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import fixtures as F  # noqa: E402

LEDGER = json.loads((
    PACKAGE_ROOT.parents[1]
    / ".codex/delivery/epics/paper-reproduction-80pct/resume_state.json"
).read_text(encoding="utf-8"))["fixture_contract"]


def test_master_seed_identity_matches_the_frozen_contract():
    assert F.master_seed_uint32() == LEDGER["master_seed_uint32_be"]
    assert hashlib.sha256(F.MASTER_SEED_LABEL.encode("utf-8")).hexdigest() \
        == LEDGER["master_seed_label_sha256"]


@pytest.mark.parametrize("vector", LEDGER["rng_golden_vectors"])
def test_counter_stream_golden_vectors(vector):
    codes = {"easy": 1, "medium": 2, "hard": 3}
    rng = F.CaseRng(codes[vector["difficulty"]], vector["index"])
    for _ in range(vector["counter"]):
        rng.next_word()
    digest = rng.digest_hex()
    word = int.from_bytes(bytes.fromhex(digest)[:8], "big")
    assert digest == vector["sha256"]
    assert word == vector["word"]


def test_unbiased_integer_uniform_never_exceeds_bound():
    rng = F.CaseRng(1, 7)
    for _ in range(200):
        assert 0 <= rng.integer(4) < 4
    with pytest.raises(ValueError):
        F.CaseRng(1, 7).integer(0)


def test_generated_fixture_satisfies_geometric_properties():
    fixture = F.generate_fixture("easy", 0)
    assert fixture["fixture_version"] == F.FIXTURE_VERSION
    assert fixture["difficulty"] == 1 and fixture["index"] == 0
    assert len(fixture["coarse_occupancy"]) == 20
    assert all(len(row) == 20 for row in fixture["coarse_occupancy"])
    # border ring occupied
    for c in range(20):
        assert fixture["coarse_occupancy"][0][c] == 1
        assert fixture["coarse_occupancy"][19][c] == 1
    fine = F._scale_to_fine(fixture["coarse_occupancy"])
    start = (fixture["start"]["row"], fixture["start"]["col"])
    goal = (fixture["goal"]["row"], fixture["goal"]["col"])
    assert F._clearance_ok(fine, *start)
    assert F._clearance_ok(fine, *goal)
    distance_sq = (start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2
    assert distance_sq > 35 ** 2
    assert F._bresenham_obstructed(fine, start, goal)
    assert F._connected(fine, start, goal)
    assert -3141593 <= fixture["yaw_urad"] < 3141593
    assert fixture["timeout_s"] == 30
    assert fixture["terminal_classes"] == ["success", "collision", "timeout"]


def test_canonical_json_rejects_floats_and_nulls():
    with pytest.raises(ValueError):
        F.canonical_json_bytes(0.5)
    with pytest.raises(ValueError):
        F.canonical_json_bytes(None)
    payload = F.canonical_json_bytes({"b": 1, "a": [True, "x"]})
    assert payload == b'{"a":[true,"x"],"b":1}'


def test_interior_obstacles_stay_within_the_placement_window():
    # Seeds occupy {2..17} inside a 3x3 mask placed with top-left in 1..16,
    # so interior occupancy must lie within [1..18] on each axis.
    for difficulty, index in (("easy", 0), ("medium", 0), ("hard", 99)):
        fixture = F.generate_fixture(difficulty, index)
        for r in range(1, 19):
            for c in range(1, 19):
                if fixture["coarse_occupancy"][r][c]:
                    assert 1 <= r <= 18 and 1 <= c <= 18
        assert fixture["fine_transform"]["cell_um"] == 100000


def test_stamp_mask_draws_top_left_in_1_to_16():
    class RecordingRng:
        def __init__(self):
            self.draws = []
        def integer(self, n):
            self.draws.append(n)
            return 1  # second-to-minimum value for every bound
    coarse = [[0] * 20 for _ in range(20)]
    rng = RecordingRng()
    top, left = F._stamp_mask(rng, coarse, [(1, 1)])
    assert rng.draws == [16, 16]          # two placement draws on [0, 16)
    assert (top, left) == (2, 2)          # 1 + draw lands in 1..16
    assert coarse[3][3] == 1              # occupied local (1,1) in {2..17}


def test_local_growth_is_bounded_to_the_3x3_mask():
    class ScriptedRng:
        """Deterministic stream: always first base, cycling directions."""
        def __init__(self, plan):
            self.plan = list(plan)
        def integer(self, n):
            return self.plan.pop(0)
    # base=0 always; direction 3=right then 1=down -> cells (1,1),(1,2),(2,1)
    rng = ScriptedRng([0, 3, 0, 1])
    cells = F._grow_cluster_local(rng, 3)
    assert sorted(cells) == [(1, 1), (1, 2), (2, 1)]
    # A stream that can only re-propose the accepted cell fails closed.
    class StuckRng:
        def integer(self, n):
            return 0
    with pytest.raises(RuntimeError, match="failed closed"):
        F._grow_cluster_local(StuckRng(), 3)


def test_payload_file_matches_regenerated_payload_and_hashes():
    fixture_path = PACKAGE_ROOT / "assets/formal_fixtures/dashgo_sea_formal_fixture_v1.json"
    hashes_path = PACKAGE_ROOT / "assets/formal_fixtures/formal_fixture_hashes.json"
    assert fixture_path.is_file() and hashes_path.is_file(), "payload must be committed"
    committed = json.loads(fixture_path.read_text(encoding="utf-8"))
    manifest = json.loads(hashes_path.read_text(encoding="utf-8"))
    assert len(committed) == 300
    assert manifest["individual_fixture_sha256_count"] == 300
    # every committed case rehashes to its recorded individual hash
    for fixture, recorded in zip(committed, manifest["individual_fixture_sha256"]):
        assert F.fixture_sha256(fixture) == recorded
    aggregate = hashlib.sha256(F.canonical_json_bytes(committed)).hexdigest()
    assert aggregate == manifest["aggregate_payload_sha256"]
    # regenerating is bit-stable
    regenerated, regen_manifest = F.generate_payload()
    assert F.fixture_sha256(regenerated[0]) == manifest["individual_fixture_sha256"][0]
    assert F.fixture_sha256(regenerated[299]) == manifest["individual_fixture_sha256"][299]
    assert regen_manifest["aggregate_payload_sha256"] == manifest["aggregate_payload_sha256"]
