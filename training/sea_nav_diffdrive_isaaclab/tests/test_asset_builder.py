# SPDX-License-Identifier: MIT
"""CPU-static tests for DashGo USD generation and ArticulationCfg source."""

import copy
import json
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/sea_nav_core/src"))

import asset_builder as B  # noqa: E402
import assets as A  # noqa: E402


def test_usd_generation_is_deterministic_and_absolute_path():
    first = B.write_dashgo_usd()
    second = B.write_dashgo_usd()
    assert first == second
    assert first.is_absolute()
    assert first.is_file()
    assert first.read_bytes() == second.read_bytes()
    assert B.usd_file_sha256() == B._sha256_bytes(first.read_bytes())
    text = first.read_text(encoding="utf-8")
    assert text.startswith("#usda 1.0")
    assert "enabled_self_collisions = 0" in text
    assert 'def Xform "wheel_left"' in text
    assert 'def Xform "wheel_right"' in text
    assert "sea_actuated = 0" in text


def test_articulation_cfg_source_disables_self_collision_and_omits_caster_actuators():
    source = B.articulation_cfg_source()
    B.validate_articulation_cfg_source(source)
    assert source["spawn"]["articulation_props"]["enabled_self_collisions"] is False
    assert Path(source["spawn"]["usd_path"]).is_absolute()
    assert "wheel_drive" in source["actuators"]
    assert all("caster" not in key for key in source["actuators"])
    names = source["actuators"]["wheel_drive"]["joint_names_expr"]
    assert "wheel_left_joint" in names and "wheel_right_joint" in names


def test_articulation_cfg_source_rejects_enabled_self_collision_or_caster_drive():
    source = B.articulation_cfg_source()
    bad = copy.deepcopy(source)
    bad["spawn"]["articulation_props"]["enabled_self_collisions"] = True
    with pytest.raises(ValueError, match="self-collisions"):
        B.validate_articulation_cfg_source(bad)
    bad2 = copy.deepcopy(source)
    bad2["actuators"]["caster_front"] = {"class_type": "ImplicitActuator"}
    with pytest.raises(ValueError, match="casters"):
        B.validate_articulation_cfg_source(bad2)
    bad3 = copy.deepcopy(source)
    bad3["spawn"]["usd_path"] = "relative/not/allowed.usda"
    with pytest.raises(ValueError, match="absolute"):
        B.validate_articulation_cfg_source(bad3)


def test_artifact_provenance_in_manifest_and_import_stays_cpu_safe():
    manifest = A.asset_manifest()
    provenance = manifest["artifact_provenance"]
    assert set(provenance) >= {
        "generator_id", "generator_sha256", "spec_sha256",
        "usd_relative_path", "usd_sha256", "articulation_cfg_source_sha256",
    }
    assert provenance["usd_sha256"] == B.usd_file_sha256()
    # Fixture aggregate hash must remain untouched by A2.2 (formal fixtures file).
    fixture_hashes = json.loads(
        (PACKAGE_ROOT / "assets/formal_fixtures/formal_fixture_hashes.json").read_text()
    )
    assert "10202bfce" in json.dumps(fixture_hashes)


def test_asset_builder_import_does_not_load_isaac():
    import subprocess
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(PACKAGE_ROOT) + "');"
        "sys.path.insert(0, r'" + str(REPO_ROOT / "packages/sea_nav_core/src") + "');"
        "import asset_builder;"
        "loaded = {n.split('.')[0] for n in sys.modules};"
        "forbidden = {'isaacsim','isaaclab','isaacgym','pxr','torch'};"
        "assert not (loaded & forbidden), sorted(loaded & forbidden)"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
