# SPDX-License-Identifier: MIT
"""Tests for the SEA 41-ray PatternBaseCfg source and CPU geometry."""

import math
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/sea_nav_core/src"))

import ray_pattern as RP  # noqa: E402
from sea_nav_core import sea_ray_angles_deg as core_sea_ray_angles_deg  # noqa: E402


def test_cpu_pattern_returns_41x3_starts_and_horizontal_directions():
    starts, directions = RP.pattern_func_cpu()
    assert len(starts) == 41 and len(directions) == 41
    assert all(s == (0.0, 0.0, 0.0) for s in starts)
    assert all(len(d) == 3 and d[2] == 0.0 for d in directions)
    # Center ray at 0 deg → +x.
    assert directions[20][0] == pytest.approx(1.0)
    assert directions[20][1] == pytest.approx(0.0)
    # ±90 deg geometric oracle.
    assert directions[35] == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert directions[5] == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
    degrees = RP.sea_ray_angles_deg()
    assert degrees == core_sea_ray_angles_deg()
    assert degrees[5] == pytest.approx(-90.0)
    assert degrees[35] == pytest.approx(90.0)


def test_cfg_source_requires_pattern_base_and_attach_yaw_only_false():
    source = RP.ray_pattern_cfg_source()
    RP.validate_ray_pattern_cfg_source(source)
    assert source["pattern"]["bases"] == ["PatternBaseCfg"]
    assert source["ray_caster"]["attach_yaw_only"] is False
    assert source["ray_caster"]["max_distance"] == 3.0
    bad = dict(source)
    bad["pattern"] = dict(source["pattern"], bases=["PatternCfg"])
    with pytest.raises(ValueError, match="PatternBaseCfg"):
        RP.validate_ray_pattern_cfg_source(bad)
    bad2 = dict(source)
    bad2["ray_caster"] = dict(source["ray_caster"], attach_yaw_only=True)
    with pytest.raises(ValueError, match="attach_yaw_only"):
        RP.validate_ray_pattern_cfg_source(bad2)


def test_isaac_pattern_factory_fail_closed_without_target_stack():
    with pytest.raises(RuntimeError, match="blocked"):
        RP.isaaclab_sea_41_ray_pattern_cfg()


def test_ray_pattern_import_stays_cpu_safe():
    import subprocess
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(PACKAGE_ROOT) + "');"
        "sys.path.insert(0, r'" + str(REPO_ROOT / "packages/sea_nav_core/src") + "');"
        "import ray_pattern;"
        "loaded = {n.split('.')[0] for n in sys.modules};"
        "forbidden = {'isaacsim','isaaclab','isaacgym','torch'};"
        "assert not (loaded & forbidden), sorted(loaded & forbidden)"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
