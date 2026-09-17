# SPDX-License-Identifier: MIT
"""Statistics mean/std and bootstrap stub tests (A8.2)."""

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from statistics import (  # noqa: E402
    BOOTSTRAP_SEED,
    aggregate_profile_seeds,
    hierarchical_paired_bootstrap_stub,
    mean_std,
    safe_ratio,
    success_only_time_mean,
)


def test_mean_std_ddof1():
    mu, sd = mean_std([1.0, 2.0, 3.0], ddof=1)
    assert mu == pytest.approx(2.0)
    assert sd == pytest.approx(1.0)


def test_aggregate_three_seeds():
    by_seed = {
        42: ["success"] * 80 + ["collision"] * 10 + ["timeout"] * 10,
        43: ["success"] * 70 + ["collision"] * 15 + ["timeout"] * 15,
        44: ["success"] * 90 + ["collision"] * 5 + ["timeout"] * 5,
    }
    agg = aggregate_profile_seeds(by_seed)
    assert agg["success_mean"] == pytest.approx(0.8, abs=0.01)
    assert agg["success_std"] is not None
    assert len(agg["seed_rates"]) == 3


def test_bootstrap_stub_fixed_seed():
    idx1 = hierarchical_paired_bootstrap_stub({"full": [0, 1, 2]}).indices
    idx2 = hierarchical_paired_bootstrap_stub({"full": [0, 1, 2]}).indices
    assert idx1.equal(idx2)
    assert hierarchical_paired_bootstrap_stub({}).seed == BOOTSTRAP_SEED


def test_zero_denominator_null():
    assert safe_ratio(1.0, 0.0) is None


def test_success_only_time_excludes_failures():
    mean, n = success_only_time_mean([1.0, None, 2.0, float("nan")])
    assert mean == pytest.approx(1.5)
    assert n == 2
