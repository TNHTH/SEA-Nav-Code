"""Ordinary native-v2 target compatibility and failure-atomic consumer tests."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.runtime_preflight import apply_model_checkpoint
from rsl_rl.utils.checkpoint import CheckpointError, load_checkpoint_v2, save_checkpoint_v2
from test_checkpoint_v2 import COMMIT, CONFIG
from test_runner_checkpoint import runner

ROOT = Path(__file__).resolve().parents[1]


def snapshot(actual):
    return (copy.deepcopy(actual.alg.actor_critic.state_dict()),
            copy.deepcopy(actual.alg.optimizer.state_dict()),
            actual.current_learning_iteration, actual.alg.learning_rate)


def equal_tree(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.layout == expected.layout and actual.device == expected.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            equal_tree(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            equal_tree(value, reference)
    else:
        assert actual == expected


def publish(path, actual, model_state=None, optimizer_state=None, mode="inference"):
    return save_checkpoint_v2(path, model_state_dict=model_state if model_state is not None else actual.alg.actor_critic.state_dict(),
        optimizer_state_dict=optimizer_state, iteration=3 if mode == "resume" else 0,
        producer_commit=COMMIT, resolved_config_sha256=actual.resolved_config_sha256)


def request_for(path, actual):
    loaded = load_checkpoint_v2(path, artifact_root=path.parent)
    return SimpleNamespace(paths=SimpleNamespace(checkpoint_manifest=path, asset_root=path.parent),
        arguments=SimpleNamespace(checkpoint_mode="inference"),
        resolved_config=SimpleNamespace(resolved_sha256=actual.resolved_config_sha256),
        checkpoint={"manifest_sha256": loaded.manifest_sha256})


@pytest.mark.parametrize("mode", ["inference", "warm_start", "resume", "model_only"])
@pytest.mark.parametrize("malformation", ["shape", "missing", "extra", "dtype", "layout"])
def test_model_mismatch_rejects_without_any_live_state_change(tmp_path, mode, malformation):
    actual = runner(tmp_path / "target")
    if mode in ("resume", "inference", "model_only"):
        actual.learn(1)
    before = snapshot(actual)
    state = copy.deepcopy(actual.alg.actor_critic.state_dict())
    keys = list(state)
    state[keys[0]].fill_(7)  # A valid earlier key must never be partially applied.
    key = next(k for k in reversed(keys) if state[k].ndim == 2)
    if malformation == "shape": state[key] = state[key][:1]
    elif malformation == "missing": del state[key]
    elif malformation == "extra": state["unexpected"] = torch.ones(1)
    elif malformation == "dtype": state[key] = state[key].double()
    else: state[key] = state[key].to_sparse()
    path = tmp_path / "bad.json"
    publish(path, actual, state, actual.alg.optimizer.state_dict() if mode == "resume" else None, mode)
    with pytest.raises((CheckpointError, RuntimeError)):
        if mode == "model_only":
            apply_model_checkpoint(request_for(path, actual), actual.alg.actor_critic, map_location="cpu")
        else:
            actual.load(path, artifact_root=tmp_path, mode=mode)
    equal_tree(snapshot(actual), before)


@pytest.mark.parametrize("malformation", ["groups", "parameter_count", "missing_betas", "betas", "eps", "amsgrad", "moment_shape", "moment_dtype", "moment_layout", "moment_device", "capturable"])
def test_adam_target_mismatch_rejects_before_model_optimizer_iteration_or_lr_changes(tmp_path, malformation):
    actual = runner(tmp_path / "target")
    actual.learn(1)
    before = snapshot(actual)
    state = copy.deepcopy(actual.alg.actor_critic.state_dict())
    state[next(iter(state))].fill_(7)
    optimizer = copy.deepcopy(actual.alg.optimizer.state_dict())
    group = optimizer["param_groups"][0]
    group["lr"] = 0.007
    if malformation == "groups": optimizer["param_groups"].append(dict(group, params=[]))
    elif malformation == "parameter_count":
        removed = group["params"].pop()
        optimizer["state"].pop(removed, None)
    elif malformation == "missing_betas": del group["betas"]
    elif malformation == "betas": group["betas"] = (0.9, 1.1)
    elif malformation == "eps": group["eps"] = -1.0
    elif malformation == "amsgrad": group["amsgrad"] = True
    elif malformation == "capturable": group["capturable"] = True  # unsupported actual CPU target
    else:
        moment = next(iter(optimizer["state"].values()))
        for key in ("exp_avg", "exp_avg_sq"):
            if malformation == "moment_shape": moment[key] = torch.ones(1)
            elif malformation == "moment_dtype": moment[key] = moment[key].double()
            elif malformation == "moment_device": moment[key] = moment[key].to("meta")
            else: moment[key] = moment[key].to_sparse()
    path = tmp_path / "bad_adam.json"
    publish(path, actual, state, optimizer, "resume")
    with pytest.raises((CheckpointError, RuntimeError)):
        actual.load(path, artifact_root=tmp_path, mode="resume")
    equal_tree(snapshot(actual), before)


@pytest.mark.parametrize("consumer", ["runner", "model_only"])
def test_same_config_hash_cannot_replace_resolved_cbf_geometry(tmp_path, consumer):
    from rsl_rl.experiment_config import resolve_run_config
    from rsl_rl.policy_factory import build_actor_critic
    from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
    resolved = resolve_run_config(registry_path=ROOT / "configs/parity_registry.yaml",
        algorithm_profile="upstream_fbce672c", runtime_stack="isaaclab_adapter",
        implementation_delta=["ppo_state_identity_repair", "replay_reset_reconstruction_v1"])
    actual = runner(tmp_path / "target")
    actual.resolved_config_sha256 = resolved.resolved_sha256
    actual.alg.actor_critic = build_actor_critic(resolved, num_actions=3,
        actor_hidden_dims=[8], critic_hidden_dims=[8], encoder_hidden_dims=[8])
    actual.alg.optimizer = torch.optim.Adam(actual.alg.actor_critic.parameters())
    before = snapshot(actual)
    state = copy.deepcopy(actual.alg.actor_critic.state_dict())
    rays = actual.alg.actor_critic.num_rays
    state["cbf_layer.ray_unit_vectors"] = ExactLSECBFLayer(num_rays=rays, fov_deg=240).ray_unit_vectors
    assert not torch.equal(state["cbf_layer.ray_unit_vectors"], before[0]["cbf_layer.ray_unit_vectors"])
    path = tmp_path / "wrong_geometry.json"
    publish(path, actual, state)
    with pytest.raises(CheckpointError, match="geometry"):
        if consumer == "runner": actual.load(path, artifact_root=tmp_path, mode="inference")
        else: apply_model_checkpoint(request_for(path, actual), actual.alg.actor_critic, map_location="cpu")
    equal_tree(snapshot(actual), before)


@pytest.mark.parametrize("script", ["train_full_method_ppo.py", "train_full_method_acsi_replay_ppo.py"])
def test_real_trainer_receipt_expression_reports_payload_bytes(script, tmp_path):
    actual = runner(tmp_path / "train")
    path = actual.learn(1)
    loaded = load_checkpoint_v2(path, artifact_root=tmp_path)
    tree = ast.parse((ROOT / "sea_nav_current_isaaclab_full_method" / script).read_text())
    expressions = [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                   and any(isinstance(k, ast.Constant) and k.value == "final_checkpoint_manifest" for k in node.keys)
                   for key, value in zip(node.keys, node.values)
                   if isinstance(key, ast.Constant) and key.value == "checkpoint_bytes"]
    assert len(expressions) == 1
    value = eval(compile(ast.Expression(expressions[0]), script, "eval"),
                 {"runner": actual, "Path": Path, "final_checkpoint": str(path)})
    assert value == loaded.manifest.byte_size == (path.parent / loaded.manifest.artifact_path).stat().st_size
    assert value != path.stat().st_size


@pytest.mark.parametrize("mode", ["inference", "warm_start", "resume", "model_only"])
def test_late_model_load_failure_rolls_back_without_reinvoking_hook(tmp_path, mode):
    actual = runner(tmp_path / "target")
    if mode != "warm_start": actual.learn(1)
    before = snapshot(actual)
    state = copy.deepcopy(actual.alg.actor_critic.state_dict())
    state[next(iter(state))].fill_(7)
    path = tmp_path / "valid.json"
    publish(path, actual, state, actual.alg.optimizer.state_dict() if mode == "resume" else None, mode)
    calls = []
    def fail_after_copy(model, incompatible):
        calls.append(True)
        assert next(iter(model.state_dict().values())).flatten()[0] == 7
        raise RuntimeError("injected late model load failure")
    handle = actual.alg.actor_critic.register_load_state_dict_post_hook(fail_after_copy)
    try:
        with pytest.raises(RuntimeError, match="injected late"):
            if mode == "model_only":
                apply_model_checkpoint(request_for(path, actual), actual.alg.actor_critic, map_location="cpu")
            else:
                actual.load(path, artifact_root=tmp_path, mode=mode)
    finally:
        handle.remove()
    assert calls == [True]
    equal_tree(snapshot(actual), before)


def test_late_optimizer_load_failure_restores_model_state_moments_group_lr_and_iteration(tmp_path):
    actual = runner(tmp_path / "target")
    actual.learn(1)
    before = snapshot(actual)
    state = copy.deepcopy(actual.alg.actor_critic.state_dict())
    state[next(iter(state))].fill_(7)
    optimizer_state = copy.deepcopy(actual.alg.optimizer.state_dict())
    optimizer_state["param_groups"][0]["lr"] = 0.007
    path = tmp_path / "valid.json"
    publish(path, actual, state, optimizer_state, "resume")
    calls = []
    def fail_after_optimizer_copy(optimizer):
        calls.append(True)
        optimizer.param_groups[0]["lr"] = 123.0
        next(iter(optimizer.state.values()))["exp_avg"].add_(100)
        raise RuntimeError("injected optimizer post-load failure")
    handle = actual.alg.optimizer.register_load_state_dict_post_hook(fail_after_optimizer_copy)
    try:
        with pytest.raises(RuntimeError, match="injected optimizer"):
            actual.load(path, artifact_root=tmp_path, mode="resume")
    finally:
        handle.remove()
    assert calls == [True]
    equal_tree(snapshot(actual), before)
    actual.learn(1)
    assert actual.current_learning_iteration == 2


@pytest.mark.parametrize("amsgrad", [False, True])
def test_valid_adam_resume_preserves_loaded_state_and_rng_then_real_update_succeeds(tmp_path, amsgrad):
    source = runner(tmp_path / "source")
    source.alg.optimizer = torch.optim.Adam(source.alg.actor_critic.parameters(), lr=0.0023, amsgrad=amsgrad)
    path = source.learn(2)
    target = runner(tmp_path / "target")
    rng_before = torch.get_rng_state().clone()
    target.load(path, artifact_root=tmp_path, mode="resume")
    assert torch.equal(torch.get_rng_state(), rng_before)
    equal_tree(target.alg.actor_critic.state_dict(), source.alg.actor_critic.state_dict())
    equal_tree(target.alg.optimizer.state_dict(), source.alg.optimizer.state_dict())
    assert target.current_learning_iteration == 2
    assert target.alg.learning_rate == 0.0023
    target.learn(1)
    assert target.current_learning_iteration == 3
