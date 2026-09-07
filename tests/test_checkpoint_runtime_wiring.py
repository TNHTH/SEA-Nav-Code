import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_checkpoint_v2 import COMMIT, save
from test_runtime_cli_contract import ROOT, cli, load_trainer, SCRIPTS


def input_checkpoint(tmp_path, stack="isaaclab_adapter", optimizer=False, iteration=0):
    from rsl_rl.experiment_config import resolve_run_config
    from rsl_rl.utils.checkpoint import save_checkpoint_v2
    resolved = resolve_run_config(registry_path=ROOT / "configs/parity_registry.yaml",
        algorithm_profile="upstream_fbce672c", runtime_stack=stack,
        implementation_delta=["ppo_state_identity_repair", "replay_reset_reconstruction_v1"])
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 1)).sum().backward()
    opt.step()
    path = tmp_path / "assets/input.json"
    save_checkpoint_v2(path, model_state_dict=model.state_dict(), iteration=iteration,
        optimizer_state_dict=opt.state_dict() if optimizer else None,
        producer_commit=COMMIT, resolved_config_sha256=resolved.resolved_sha256)
    return path


@pytest.mark.parametrize("script", SCRIPTS[1:])
@pytest.mark.parametrize("mode,flag,optimizer,iteration", [
    ("warm_start", "--init-checkpoint-manifest", False, 0),
    ("resume", "--resume-checkpoint-manifest", True, 3)])
def test_actual_training_manifest_modes(script, mode, flag, optimizer, iteration, tmp_path):
    argv = cli(tmp_path) + ["--producer-commit", COMMIT]
    path = input_checkpoint(tmp_path, optimizer=optimizer, iteration=iteration)
    request = load_trainer(script).preflight(argv + [flag, str(path)])
    assert request.arguments.checkpoint_mode == mode
    assert request.arguments.producer_commit == COMMIT
    assert request.checkpoint["manifest"]["iteration"] == iteration
    assert request.checkpoint["capability"]["weights_only_roundtrip"] is True
    assert request.environment["asset_prerequisites"]["runtime_ready"] is False
    assert not request.paths.run_root.exists()


@pytest.mark.parametrize("extra", [
    ["--checkpoint-mode", "resume"],
    ["--init-checkpoint-manifest", "INPUT", "--resume-checkpoint-manifest", "INPUT"],
    ["--checkpoint-mode", "inference", "--resume-checkpoint-manifest", "INPUT"],
    ["--checkpoint-manifest", "INPUT"],
])
def test_invalid_training_mode_combinations_reject(extra, tmp_path):
    argv = cli(tmp_path) + ["--producer-commit", COMMIT]
    path = input_checkpoint(tmp_path)
    extra = [str(path) if x == "INPUT" else x for x in extra]
    with pytest.raises((ValueError, SystemExit)):
        load_trainer(SCRIPTS[1]).preflight(argv + extra)
    assert not (tmp_path / "output").exists()


def test_gym_inference_preflight_keeps_parent_torch_free(tmp_path):
    argv = cli(tmp_path) + ["--producer-commit", COMMIT]
    path = input_checkpoint(tmp_path, stack="isaac_gym_preview4", optimizer=True, iteration=4)
    argv += ["--checkpoint-manifest", str(path)]
    code = '''
import importlib.util, json, sys
s=importlib.util.spec_from_file_location('entry',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
assert 'torch' not in sys.modules
r=m.preflight(json.loads(sys.argv[2]))
assert 'torch' not in sys.modules
assert r.arguments.checkpoint_mode=='inference'
assert r.checkpoint['manifest']['iteration']==4
assert r.checkpoint['capability']['weights_only_roundtrip'] is True
assert r.checkpoint['manifest_sha256']
assert not r.paths.run_root.exists()
'''
    run = subprocess.run([sys.executable, "-I", "-B", "-c", code,
        str(ROOT / "training/legged_gym/legged_gym/scripts/play.py"), json.dumps(argv)],
        capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("mismatched_rows", [False, True])
def test_actual_gym_play_preparation_preserves_bound_terrain_receipt(tmp_path, mismatched_rows):
    """Real config/play/registry CPU statements, without simulator construction."""
    import inspect
    from types import SimpleNamespace
    import numpy as np
    from rsl_rl.environment_profile import (
        apply_gym_environment, materialize_config, reconcile_environment_receipt)

    argv = cli(tmp_path)
    manifest = input_checkpoint(tmp_path, stack="isaac_gym_preview4")
    gym = ROOT / "training/legged_gym/legged_gym"
    script = gym / "scripts/play.py"
    spec = importlib.util.spec_from_file_location("play_terrain_entry", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = module.preflight(argv + ["--checkpoint-manifest", str(manifest)])
    assert request.arguments.checkpoint_mode == "inference"
    assert request.arguments.source_max_goal_level == 10.0
    assert request.environment["asset_prerequisites"]["runtime_ready"] is False

    namespace = {"inspect": inspect, "np": np}
    for relative, class_name in (
        ("envs/base/base_config.py", "BaseConfig"),
        ("envs/base/legged_robot_config.py", "LeggedRobotCfg"),
        ("envs/base/legged_robot_pos_config.py", "LeggedRobotPosCfg"),
        ("envs/go2/go2_pos_config.py", "Go2PosRoughCfg"),
    ):
        path = gym / relative
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == class_name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    namespace["env_cfg"] = materialize_config(namespace["Go2PosRoughCfg"]())
    assert namespace["env_cfg"].terrain.num_rows == 10

    play = next(n for n in ast.parse(script.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "play")
    start = next(i for i, n in enumerate(play.body) if isinstance(n, ast.Assign)
                 and any(ast.unparse(t) == "env_cfg.env.num_envs" for t in n.targets))
    stop = next(i for i, n in enumerate(play.body) if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "task_registry.make_env")
    exec(compile(ast.Module(body=play.body[start:stop], type_ignores=[]), str(script), "exec"), namespace)
    assert namespace["env_cfg"].env.num_envs == 1
    assert namespace["env_cfg"].terrain.num_cols == 1

    registry = gym / "utils/task_registry.py"
    make_env = next(n for n in ast.walk(ast.parse(registry.read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == "make_env")
    start = next(i for i, n in enumerate(make_env.body)
                 if isinstance(n, ast.If) and ast.unparse(n.test) == "resolved_config is None")
    stop = next(i for i, n in enumerate(make_env.body) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "set_seed")
    prefix = compile(ast.Module(body=make_env.body[start:stop], type_ignores=[]), str(registry), "exec")
    namespace.update(args=SimpleNamespace(runtime_request=request), resolved_config=request.resolved_config,
        materialize_config=materialize_config, apply_gym_environment=apply_gym_environment,
        reconcile_environment_receipt=reconcile_environment_receipt)
    if mismatched_rows:
        namespace["env_cfg"].terrain.num_rows = 9
        with pytest.raises(ValueError, match="source_max_goal_level must match the actual Gym terrain row bound"):
            exec(prefix, namespace)
        assert "receipt" not in namespace
    else:
        exec(prefix, namespace)
        cfg, receipt = namespace["env_cfg"], namespace["receipt"]
        assert cfg.terrain.num_rows == request.arguments.source_max_goal_level == 10
        assert cfg.terrain.max_init_terrain_level == 3 < cfg.terrain.num_rows
        assert receipt["replay"]["stored_level_bounds"] == [0.0, 10.0]
        assert receipt == {key: value for key, value in request.environment.items()
                           if key not in ("constructor_settings", "asset_prerequisites")}
        assert cfg.environment_receipt == receipt
        assert cfg.env.num_envs == request.arguments.num_envs
        assert cfg.controller_root == request.paths.asset_root / "ctrl_model"
    assert request.environment["asset_prerequisites"]["runtime_ready"] is False
    assert not request.paths.run_root.exists()
    assert not any(name == "isaacgym" or name.startswith("isaacgym.") for name in sys.modules)


def test_changed_checkpoint_after_preflight_rejected_before_model_application(tmp_path):
    from rsl_rl.runtime_preflight import apply_model_checkpoint
    argv = cli(tmp_path) + ["--producer-commit", COMMIT]
    path = input_checkpoint(tmp_path)
    request = load_trainer(SCRIPTS[0]).preflight(argv + ["--checkpoint-manifest", str(path)])
    model = torch.nn.Linear(1, 1)
    before = model.weight.clone()
    input_checkpoint(tmp_path, iteration=1)
    with pytest.raises(ValueError, match="changed"):
        apply_model_checkpoint(request, model, map_location="cpu")
    assert torch.equal(before, model.weight)


def test_real_callers_use_manifest_helpers_and_learn_return():
    for script in SCRIPTS[1:]:
        tree = ast.parse((ROOT / "sea_nav_current_isaaclab_full_method" / script).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert any(ast.unparse(n.func) == "apply_runner_checkpoint" for n in calls)
        assert not any(isinstance(n.func, ast.Attribute) and n.func.attr == "glob" for n in calls)
        assert any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                   and ast.unparse(n.value.func) == "runner.learn" for n in ast.walk(tree))
    gym = ast.parse((ROOT / "training/legged_gym/legged_gym/utils/task_registry.py").read_text())
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) == "eval" for n in ast.walk(gym))
    assert any(isinstance(n, ast.Call) and ast.unparse(n.func) == "apply_runner_checkpoint" for n in ast.walk(gym))


def test_initializer_is_pure_import_and_writes_model_only_v2(tmp_path):
    script = ROOT / "sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py"
    spec = importlib.util.spec_from_file_location("initializer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "init.manifest.json"
    result = module.main(["--config", str(ROOT / "sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml"),
                          "--runtime-stack", "isaaclab_adapter", "--producer-commit", COMMIT,
                          "--output-manifest", str(path), "--seed", "42"])
    from rsl_rl.utils.checkpoint import load_checkpoint_v2
    loaded = load_checkpoint_v2(path, artifact_root=tmp_path)
    assert loaded.iteration == 0 and loaded.optimizer_state_dict is None
    assert result["producer_commit"] == COMMIT
    assert result["resolved_config_sha256"] == loaded.manifest.resolved_config_sha256
    assert result["runtime_verified"] is False


def _documented_examples():
    import re
    examples = []
    for name in ("README.md", "training/legged_gym/README.md", "sea_nav_current_isaaclab_full_method/README.md"):
        for label, command in re.findall(r"<!-- checkpoint-example: ([\w-]+) -->\s*```bash\n(.*?)```",
                                        (ROOT / name).read_text(), re.S):
            examples.append((label, command))
    return examples


@pytest.mark.parametrize("label,command", _documented_examples(), ids=lambda x: x if "\n" not in x else None)
def test_current_readme_examples_use_real_pure_parsers(label, command, tmp_path):
    import shlex
    from string import Template
    base = cli(tmp_path)
    stack = "isaac_gym_preview4" if label.startswith("gym-") else "isaaclab_adapter"
    path = input_checkpoint(tmp_path, stack=stack, optimizer=label.endswith("resume"),
                            iteration=3 if label.endswith("resume") else 0)
    init = tmp_path / "assets/init.manifest.json"
    init.write_bytes(path.read_bytes())
    substituted = Template(command.replace("\\\n", " ")).substitute(
        SEA_NAV_PYTHON=sys.executable, SEA_NAV_COMMIT=COMMIT,
        SEA_NAV_LAUNCHER=base[base.index("--launcher") + 1],
        SEA_NAV_ASSET_ROOT=str(tmp_path / "assets"), SEA_NAV_RUN_ROOT=str(tmp_path / "output"),
        SEA_NAV_RESUME_MANIFEST=str(path))
    tokens = shlex.split(substituted)
    assert tokens[:2] == [sys.executable, "-B"]
    script, argv = ROOT / tokens[2], tokens[3:]
    spec = importlib.util.spec_from_file_location("documented_entry", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if script.name == "init_full_method_checkpoint.py":
        parsed = module.build_parser().parse_args(argv)
        assert parsed.runtime_stack == stack and parsed.producer_commit == COMMIT
    else:
        request = module.preflight(argv)
        assert request.arguments.preflight_only is True
        assert request.arguments.producer_commit == COMMIT
        assert request.resolved_config.identity.runtime_stack == stack
        assert not request.paths.run_root.exists()


def test_actual_gym_runner_registry_rejects_expressions_without_execution(tmp_path):
    from rsl_rl.runners import OnPolicyRunner
    path = ROOT / "training/legged_gym/legged_gym/utils/task_registry.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "resolve_runner_class")
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RUNNER_REGISTRY" for t in n.targets))]
    namespace = {"OnPolicyRunner": OnPolicyRunner}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    resolve = namespace["resolve_runner_class"]
    assert resolve("OnPolicyRunner") is OnPolicyRunner
    marker = tmp_path / "unexpected"
    with pytest.raises(ValueError, match="unknown runner class"):
        resolve("__import__('pathlib').Path(%r).touch()" % str(marker))
    assert not marker.exists()


def test_gym_play_checks_prerequisites_before_proprietary_import(tmp_path):
    argv = cli(tmp_path)
    path = input_checkpoint(tmp_path, stack="isaac_gym_preview4")
    script = ROOT / "training/legged_gym/legged_gym/scripts/play.py"
    run = subprocess.run([sys.executable, "-I", "-B", str(script)] + argv + ["--checkpoint-manifest", str(path)],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 3, run.stdout + run.stderr
    result = json.loads((tmp_path / "output/result.json").read_text())
    assert result["status"] == "blocked" and result["runtime_verified"] is False
    assert result["input_checkpoint"]["mode"] == "inference"
    tree = ast.parse(script.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    gate = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == "require_runtime_prerequisites")
    imported = next(n for n in ast.walk(main) if isinstance(n, ast.Import) and any(x.name == "isaacgym" for x in n.names))
    assert gate.lineno < imported.lineno


@pytest.mark.parametrize("caller", ["gym", SCRIPTS[1], SCRIPTS[2]])
@pytest.mark.parametrize("mode", ["warm_start", "resume"])
def test_actual_caller_load_statement_applies_real_runner_state(caller, mode, tmp_path):
    from rsl_rl.runtime_preflight import preflight, apply_runner_checkpoint
    from rsl_rl.utils.checkpoint import save_checkpoint_v2
    from test_runner_checkpoint import runner
    argv = cli(tmp_path)
    if caller == "gym":
        entry = lambda args: preflight(args, runtime_stack="isaac_gym_preview4", repo_root=ROOT)
        source = ROOT / "training/legged_gym/legged_gym/utils/task_registry.py"
    else:
        entry = load_trainer(caller).preflight
        source = ROOT / "sea_nav_current_isaaclab_full_method" / caller
    fresh = entry(argv)
    original = runner(tmp_path / "assets/original")
    original.resolved_config_sha256 = fresh.resolved_config.resolved_sha256
    if mode == "resume":
        manifest = original.learn(2)
        flag = "--resume-checkpoint-manifest"
    else:
        manifest = tmp_path / "assets/init.json"
        save_checkpoint_v2(manifest, model_state_dict=original.alg.actor_critic.state_dict(), iteration=0,
            producer_commit=COMMIT, resolved_config_sha256=original.resolved_config_sha256)
        flag = "--init-checkpoint-manifest"
    request = entry(argv + [flag, str(manifest)])
    target = runner(tmp_path / "target")
    target.resolved_config_sha256 = request.resolved_config.resolved_sha256
    call = next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "apply_runner_checkpoint")
    expression = ast.Expression(body=call)
    eval(compile(expression, str(source), "eval"), dict(apply_runner_checkpoint=apply_runner_checkpoint,
        request=request, runner=target))
    assert target.current_learning_iteration == (2 if mode == "resume" else 0)
    for key, value in original.alg.actor_critic.state_dict().items():
        assert torch.equal(value, target.alg.actor_critic.state_dict()[key])
    assert bool(target.alg.optimizer.state) is (mode == "resume")
