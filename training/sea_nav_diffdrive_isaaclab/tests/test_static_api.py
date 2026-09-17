# SPDX-License-Identifier: MIT
"""Static API inventory tests (A9.1)."""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import static_api  # noqa: E402


def test_models_empty_for_delivery():
    assert static_api.MODELS == []


def test_inventory_has_map_entries():
    inv = static_api.inventory()
    assert inv["models"] == []
    assert "cli" in inv
    assert len(inv["function_consumer_test_map"]) >= 10


def test_list_checks_sorted_unique():
    checks = static_api.list_checks()
    assert checks == sorted(set(checks))
