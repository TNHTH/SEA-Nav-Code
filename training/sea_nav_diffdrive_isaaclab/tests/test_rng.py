# SPDX-License-Identifier: MIT
"""Named RNG stream derivation tests (A7.1)."""

import sys
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from rng import (  # noqa: E402
    NamedRngBank,
    create_banks,
    derive_stream_seed,
    verify_no_python_hash,
)


def test_derive_stream_seed_stable():
    a = derive_stream_seed(42, "map", 0)
    b = derive_stream_seed(42, "map", 0)
    assert a == b
    assert a != derive_stream_seed(42, "goal", 0)


def test_no_python_hash():
    verify_no_python_hash()


def test_unknown_stream_rejected():
    with pytest.raises(ValueError):
        derive_stream_seed(1, "unknown", 0)


def test_banks_have_all_streams():
    bank = NamedRngBank(master_seed=42, env_id=3)
    for name in ("map", "acsi", "perception_delay", "eval_fixture_delay"):
        assert name in bank.streams


def test_state_roundtrip():
    banks = create_banks(42, 2)
    g1 = banks[0].get("acsi").generator()
    expected = torch.rand(3, generator=g1)
    data = banks[0].state_dict()
    restored = NamedRngBank.from_state_dict(data)
    g2 = restored.get("acsi").generator()
    replay = torch.rand(3, generator=g2)
    torch.testing.assert_close(expected, replay)


def test_acsi_stream_independent_from_map():
    bank = NamedRngBank(99, 0)
    ga = bank.get("acsi").generator()
    gm = bank.get("map").generator()
    assert torch.rand(1, generator=ga).item() != torch.rand(1, generator=gm).item()
