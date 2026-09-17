# SPDX-License-Identifier: MIT
"""Fixed episode export schema with duplicate/missing rejection (A8.1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

from evaluation import EpisodeOutcome, EvalCaseKey

EPISODE_SCHEMA_VERSION = 1

REQUIRED_FIELDS = (
    "schema_version",
    "case_key",
    "profile",
    "seed",
    "fixture_id",
    "outcome",
    "completion_tick",
    "attempt_id",
    "code_hash",
    "config_hash",
    "model_hash",
)


@dataclass
class EpisodeRecord:
    schema_version: int = EPISODE_SCHEMA_VERSION
    profile: str = ""
    seed: int = 0
    fixture_id: int = 0
    outcome: str = ""
    completion_tick: int | None = None
    attempt_id: str = ""
    code_hash: str = ""
    config_hash: str = ""
    model_hash: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def case_key(self) -> str:
        return EvalCaseKey(self.profile, self.seed, self.fixture_id).canonical()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_key": self.case_key,
            "profile": self.profile,
            "seed": self.seed,
            "fixture_id": self.fixture_id,
            "outcome": self.outcome,
            "completion_tick": self.completion_tick,
            "attempt_id": self.attempt_id,
            "code_hash": self.code_hash,
            "config_hash": self.config_hash,
            "model_hash": self.model_hash,
            "metrics": dict(self.metrics),
        }

    def body_digest(self) -> str:
        body = {k: self.to_dict()[k] for k in REQUIRED_FIELDS if k != "schema_version"}
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def validate_record(data: Dict[str, Any]) -> EpisodeRecord:
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"missing episode fields: {missing}")
    if int(data["schema_version"]) != EPISODE_SCHEMA_VERSION:
        raise ValueError("unsupported episode schema_version")
    outcome = str(data["outcome"])
    allowed = {o.value for o in EpisodeOutcome}
    if outcome not in allowed:
        raise ValueError(f"invalid outcome: {outcome}")
    record = EpisodeRecord(
        schema_version=int(data["schema_version"]),
        profile=str(data["profile"]),
        seed=int(data["seed"]),
        fixture_id=int(data["fixture_id"]),
        outcome=outcome,
        completion_tick=data["completion_tick"],
        attempt_id=str(data["attempt_id"]),
        code_hash=str(data["code_hash"]),
        config_hash=str(data["config_hash"]),
        model_hash=str(data["model_hash"]),
        metrics=dict(data.get("metrics", {})),
    )
    expected_key = record.case_key
    if data["case_key"] != expected_key:
        raise ValueError("case_key does not match profile/seed/fixture_id")
    return record


def validate_corpus(records: Iterable[Dict[str, Any]]) -> List[EpisodeRecord]:
    seen: Set[str] = set()
    parsed: List[EpisodeRecord] = []
    for data in records:
        rec = validate_record(data)
        if rec.case_key in seen:
            raise ValueError(f"duplicate case: {rec.case_key}")
        seen.add(rec.case_key)
        parsed.append(rec)
    return parsed


def assert_complete_fixture_set(
    records: List[EpisodeRecord],
    *,
    profiles: Iterable[str],
    seeds: Iterable[int],
    fixture_count: int = 100,
) -> None:
    expected = {
        EvalCaseKey(p, s, f).canonical()
        for p in profiles
        for s in seeds
        for f in range(fixture_count)
    }
    got = {r.case_key for r in records}
    missing = expected - got
    if missing:
        raise ValueError(f"missing cases: {len(missing)}")
    extra = got - expected
    if extra:
        raise ValueError(f"unexpected cases: {len(extra)}")
