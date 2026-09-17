# SPDX-License-Identifier: MIT
"""Episode schema validation tests (A8.1)."""

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from episode_schema import (  # noqa: E402
    EPISODE_SCHEMA_VERSION,
    validate_corpus,
    validate_record,
)


def _record(**overrides):
    base = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "case_key": "full:seed42:fixture0",
        "profile": "full",
        "seed": 42,
        "fixture_id": 0,
        "outcome": "success",
        "completion_tick": 150,
        "attempt_id": "formal-1",
        "code_hash": "c" * 64,
        "config_hash": "d" * 64,
        "model_hash": "m" * 64,
        "metrics": {},
    }
    base.update(overrides)
    return base


def test_valid_record():
    rec = validate_record(_record())
    assert rec.profile == "full"


def test_missing_field_rejected():
    data = _record()
    del data["model_hash"]
    with pytest.raises(ValueError, match="missing"):
        validate_record(data)


def test_duplicate_case_rejected():
    corpus = [_record(), _record()]
    with pytest.raises(ValueError, match="duplicate"):
        validate_corpus(corpus)


def test_case_key_mismatch_rejected():
    data = _record(case_key="wrong")
    with pytest.raises(ValueError, match="case_key"):
        validate_record(data)
