# SPDX-License-Identifier: MIT
"""AST/static coverage for the Isaac-dependent env module (A3).

`env.py` imports Isaac Lab at import time; CPU hosts verify structure,
official hook usage, cfg constants, and import boundaries without executing it.
"""

import ast
from pathlib import Path
import subprocess
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PACKAGE_ROOT / "env.py"
CFG_FILE = PACKAGE_ROOT / "env_cfg.py"
SOURCE = ENV_FILE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

sys.path.insert(0, str(PACKAGE_ROOT))
import env_cfg as CFG  # noqa: E402


def _class(name):
    return next(node for node in TREE.body
                if isinstance(node, ast.ClassDef) and node.name == name)


def _method(cls_name, name):
    cls = _class(cls_name)
    return next((node for node in cls.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == name), None)


def test_env_module_parses_and_compiles():
    compile(SOURCE, str(ENV_FILE), "exec")
    assert "class SeaNavDiffDriveEnv" in SOURCE


def test_env_cfg_constants_match_scientific_contract():
    source = CFG.env_cfg_source()
    CFG.validate_env_cfg_source(source)
    assert source["sim"]["dt"] == 0.005
    assert source["decimation"] == 4
    assert source["episode_length_s"] == 60.0
    assert source["action_space"] == 2
    assert source["observation_space"] == 550
    assert source["train_ready"] is False


def test_train_entry_is_refused_until_later_cards_wire_pipeline():
    try:
        CFG.assert_train_entry_allowed()
        raise AssertionError("train should be refused")
    except RuntimeError as exc:
        assert "refused" in str(exc)


def test_step_delegates_to_parent_direct_rl_env():
    step = _method("SeaNavDiffDriveEnv", "step")
    assert step is not None
    text = ast.unparse(step)
    assert "super().step" in text
    assert "write_joint_velocity_target_to_sim" not in SOURCE


def test_official_hooks_exist_and_apply_action_uses_set_joint_velocity_target():
    for name in (
        "_pre_physics_step", "_apply_action", "_get_observations",
        "_get_rewards", "_get_dones", "_reset_idx", "_setup_scene",
    ):
        assert _method("SeaNavDiffDriveEnv", name) is not None, name
    apply_action = _method("SeaNavDiffDriveEnv", "_apply_action")
    assert "set_joint_velocity_target" in ast.unparse(apply_action)
    dones = _method("SeaNavDiffDriveEnv", "_get_dones")
    assert "_capture_terminal_payload" in ast.unparse(dones)


def test_action_stages_do_not_preclamp_axes():
    stages = _method("SeaNavDiffDriveEnv", "_current_action_stages")
    text = ast.unparse(stages)
    assert "torch.clamp" not in text
    assert "clipped_policy_action" in text


def test_get_dones_captures_before_rewards_in_source_contract():
    # Official DirectRLEnv.step order is dones → rewards → reset → obs.
    # Capture must live inside _get_dones so it runs before _reset_idx.
    dones = ast.unparse(_method("SeaNavDiffDriveEnv", "_get_dones"))
    assert "_capture_terminal_payload" in dones
    rewards = ast.unparse(_method("SeaNavDiffDriveEnv", "_get_rewards"))
    assert "_capture_terminal_payload" not in rewards


def test_close_is_idempotent_and_context_manager_exit_closes():
    close = _method("SeaNavDiffDriveEnv", "close")
    assert "if self._closed" in ast.unparse(close)
    assert "self.close()" in ast.unparse(_method("SeaNavDiffDriveEnv", "__exit__"))


def test_step_rejects_use_after_close():
    assert "if self._closed" in ast.unparse(_method("SeaNavDiffDriveEnv", "step"))


def test_env_does_not_import_the_go2_adapter_or_dashgo_code():
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            assert "sea_nav_current_isaaclab_full_method" not in name
            assert "dashgo_rl" not in name
            assert not name.startswith("adaptation.dashgo")


def test_env_import_fails_closed_on_cpu_without_isaaclab():
    code = (
        "import sys\n"
        "sys.path.insert(0, r'" + str(PACKAGE_ROOT) + "')\n"
        "sys.path.insert(0, r'" + str(PACKAGE_ROOT.parents[1] / "packages/sea_nav_core/src") + "')\n"
        "try:\n"
        "    import env\n"
        "    raise SystemExit('unexpectedly imported')\n"
        "except ImportError:\n"
        "    raise SystemExit(0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_modules_still_do_not_load_isaac():
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(PACKAGE_ROOT) + "');"
        "sys.path.insert(0, r'" + str(PACKAGE_ROOT.parents[1] / "packages/sea_nav_core/src") + "');"
        "import execution, trace, env_cfg;"
        "loaded = {n.split('.')[0] for n in sys.modules};"
        "forbidden = {'isaacsim','isaaclab','isaacgym','rclpy'};"
        "assert not (loaded & forbidden), sorted(loaded & forbidden)"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
