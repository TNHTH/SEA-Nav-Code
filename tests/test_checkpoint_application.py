"""Ordinary native-v2 target compatibility and failure-atomic consumer tests."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.runtime_preflight import apply_model_checkpoint
from rsl_rl.utils.checkpoint import CheckpointError, apply_checkpoint_state, load_checkpoint_v2, save_checkpoint_v2
from test_checkpoint_v2 import COMMIT, CONFIG
from test_runner_checkpoint import runner

ROOT = Path(__file__).resolve().parents[1]


def snapshot(actual):
    return (copy.deepcopy(actual.alg.actor_critic.state_dict()),
            copy.deepcopy(actual.alg.optimizer.state_dict()),
            actual.current_learning_iteration, actual.alg.learning_rate,
            [None if p.grad is None else p.grad.clone() for p in actual.alg.actor_critic.parameters()],
            copy.deepcopy(actual.alg.optimizer.defaults), torch.get_rng_state().clone(),
            [b.clone() for b in actual.alg.actor_critic.buffers()])


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


def consume(path, actual, mode):
    if mode == "model_only":
        return apply_model_checkpoint(request_for(path, actual), actual.alg.actor_critic, map_location="cpu")
    return actual.load(path, artifact_root=path.parent, mode=mode)


def test_zero_betas_and_epsilon_resume_then_real_ppo_update_is_finite(tmp_path):
    torch.manual_seed(71)
    source = runner(tmp_path / "source")
    source.learn(1)
    source.alg.optimizer.param_groups[0].update(betas=(0.0, 0.0), eps=0.0)
    path = tmp_path / "zero.json"
    publish(path, source, optimizer_state=source.alg.optimizer.state_dict(), mode="resume")
    target = runner(tmp_path / "target")
    consume(path, target, "resume")
    equal_tree(target.alg.optimizer.state_dict(), source.alg.optimizer.state_dict())
    target.learn(1)
    assert target.current_learning_iteration == 4
    assert all(torch.isfinite(p).all() for p in target.alg.actor_critic.parameters())
    assert torch.isfinite(target.alg.actor_critic.act_inference(target.env.get_observations())).all()


@pytest.mark.parametrize("forbidden", ["__init__", "step", "zero_grad", "backward"])
def test_checkpoint_validation_never_constructs_or_executes_an_optimizer(tmp_path, monkeypatch, forbidden):
    actual = runner(tmp_path / "target")
    path = tmp_path / "valid.json"
    publish(path, actual, optimizer_state=actual.alg.optimizer.state_dict(), mode="resume")
    def unexpected(*args, **kwargs):
        pytest.fail("load executed forbidden " + forbidden)
    owner = torch.Tensor if forbidden == "backward" else torch.optim.Adam
    monkeypatch.setattr(owner, forbidden, unexpected)
    consume(path, actual, "resume")


def test_load_calls_only_actual_optimizer_load_hooks_and_no_global_or_instance_step_hooks(tmp_path, monkeypatch):
    from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
    actual = runner(tmp_path / "target")
    optimizer = actual.alg.optimizer
    path = tmp_path / "valid.json"
    publish(path, actual, optimizer_state=optimizer.state_dict(), mode="resume")
    step_calls, load_calls = [], []
    handles = [register_optimizer_step_pre_hook(lambda *args: step_calls.append("global_pre")),
               register_optimizer_step_post_hook(lambda *args: step_calls.append("global_post")),
               optimizer.register_step_pre_hook(lambda *args: step_calls.append("instance_pre")),
               optimizer.register_step_post_hook(lambda *args: step_calls.append("instance_post"))]
    native_load = torch.optim.Adam.load_state_dict
    def record_load(instance, state):
        load_calls.append(instance)
        return native_load(instance, state)
    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", record_load)
    try:
        consume(path, actual, "resume")
    finally:
        for handle in handles: handle.remove()
    assert step_calls == []
    assert load_calls == [optimizer]


@pytest.mark.parametrize("mode", ["inference", "warm_start", "resume", "model_only"])
@pytest.mark.parametrize("initially_none", [False, True])
def test_late_model_hook_restores_gradient_presence_and_values(tmp_path, mode, initially_none):
    actual = runner(tmp_path / "target")
    model = actual.alg.actor_critic
    parameter = next(model.parameters())
    parameter.grad = None if initially_none else torch.ones_like(parameter)
    other = list(model.parameters())[1]
    other.grad = torch.full_like(other, 2)
    model.register_buffer("transient", torch.ones(2), persistent=False)
    path = tmp_path / "valid.json"
    publish(path, actual, optimizer_state=actual.alg.optimizer.state_dict() if mode == "resume" else None, mode=mode)
    before = snapshot(actual)
    calls = []
    def fail(model, incompatible):
        calls.append(True)
        if initially_none: parameter.grad = torch.full_like(parameter, 10)
        else: parameter.grad.mul_(10)
        other.grad = None
        model.transient.add_(8)
        raise RuntimeError("late gradient failure")
    handle = model.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="late gradient"):
            consume(path, actual, mode)
    finally:
        handle.remove()
    equal_tree(snapshot(actual), before)
    assert calls == [True]  # External hook-owned lists are deliberately not rolled back.


@pytest.mark.parametrize("initially_none", [False, True])
def test_late_optimizer_hook_restores_gradients_rng_and_runner_metadata(tmp_path, initially_none):
    actual = runner(tmp_path / "target")
    actual.learn(1)
    parameter = next(actual.alg.actor_critic.parameters())
    parameter.grad = None if initially_none else torch.ones_like(parameter)
    path = tmp_path / "valid.json"
    publish(path, actual, optimizer_state=actual.alg.optimizer.state_dict(), mode="resume")
    before = snapshot(actual)
    def fail(optimizer):
        if initially_none: parameter.grad = torch.full_like(parameter, 3)
        else: parameter.grad.mul_(3)
        torch.rand(4)
        actual.current_learning_iteration = 88
        actual.alg.learning_rate = 0.7
        optimizer.param_groups[0]["lr"] = 0.8
        optimizer.defaults["eps"] = 0.9
        raise RuntimeError("late optimizer transaction failure")
    handle = actual.alg.optimizer.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="late optimizer transaction"):
            consume(path, actual, "resume")
    finally:
        handle.remove()
    equal_tree(snapshot(actual), before)
    actual.learn(1)
    assert actual.current_learning_iteration == 2


@pytest.mark.parametrize("mode", ["inference", "warm_start", "resume"])
def test_late_model_hook_restores_runner_optimizer_and_metadata_in_every_mode(tmp_path, mode):
    actual = runner(tmp_path / "target")
    path = tmp_path / "valid.json"
    publish(path, actual, optimizer_state=actual.alg.optimizer.state_dict() if mode == "resume" else None, mode=mode)
    before = snapshot(actual)
    def fail(model, incompatible):
        actual.current_learning_iteration = 88
        actual.alg.learning_rate = 0.7
        actual.alg.optimizer.param_groups[0]["lr"] = 0.8
        actual.alg.optimizer.defaults["eps"] = 0.9
        raise RuntimeError("late runner metadata failure")
    handle = actual.alg.actor_critic.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="late runner metadata"):
            consume(path, actual, mode)
    finally:
        handle.remove()
    equal_tree(snapshot(actual), before)


@pytest.mark.parametrize("mode", ["inference", "warm_start", "resume", "model_only"])
@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("cuda_initialized", [False, True])
def test_load_preserves_cpu_and_already_initialized_cuda_rng(tmp_path, monkeypatch, mode, fails, cuda_initialized):
    actual = runner(tmp_path / "target")
    path = tmp_path / "valid.json"
    publish(path, actual, optimizer_state=actual.alg.optimizer.state_dict() if mode == "resume" else None, mode=mode)
    cpu_before = torch.get_rng_state().clone()
    # Simulate two initialized devices without requiring CUDA or initializing it.
    cuda_states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    cuda_before = copy.deepcopy(cuda_states)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_initialized)
    def get_cuda():
        assert cuda_initialized, "load must not initialize CUDA for RNG bookkeeping"
        return copy.deepcopy(cuda_states)
    def set_cuda(states):
        assert cuda_initialized
        cuda_states[:] = copy.deepcopy(states)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", get_cuda)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_cuda)
    def hook(*args):
        torch.rand(5)
        if cuda_initialized:
            for value in cuda_states: value.add_(5)
        if fails: raise RuntimeError("late RNG failure")
    if mode == "resume":
        handle = actual.alg.optimizer.register_load_state_dict_post_hook(hook)
    else:
        handle = actual.alg.actor_critic.register_load_state_dict_post_hook(hook)
    try:
        if fails:
            with pytest.raises(RuntimeError, match="late RNG"):
                consume(path, actual, mode)
        else:
            consume(path, actual, mode)
    finally:
        handle.remove()
    assert torch.equal(torch.get_rng_state(), cpu_before)
    equal_tree(cuda_states, cuda_before)


@pytest.mark.parametrize("malformation", ["fused_foreach", "fused_differentiable", "foreach_differentiable",
    "differentiable_leaf", "flag", "optional_flag", "weight_decay", "negative_moment", "infinite_moment",
    "step_dtype", "amsgrad_extra"])
def test_pure_adam_validation_rejects_invalid_options_and_state(tmp_path, malformation):
    actual = runner(tmp_path / "target")
    actual.learn(1)
    state = copy.deepcopy(actual.alg.optimizer.state_dict())
    group = state["param_groups"][0]
    moment = next(iter(state["state"].values()))
    if malformation == "fused_foreach": group.update(fused=True, foreach=True)
    elif malformation == "fused_differentiable": group.update(fused=True, differentiable=True)
    elif malformation == "foreach_differentiable": group.update(foreach=True, differentiable=True)
    elif malformation == "differentiable_leaf": group["differentiable"] = True
    elif malformation == "flag": group["maximize"] = 1
    elif malformation == "optional_flag": group["foreach"] = "yes"
    elif malformation == "weight_decay": group["weight_decay"] = -0.1
    elif malformation == "negative_moment": moment["exp_avg_sq"].fill_(-1)
    elif malformation == "infinite_moment": moment["exp_avg"].fill_(float("inf"))
    elif malformation == "step_dtype": moment["step"] = moment["step"].to(torch.int64)
    else: moment["max_exp_avg_sq"] = moment["exp_avg_sq"].clone()
    path = tmp_path / "invalid.json"
    publish(path, actual, optimizer_state=state, mode="resume")
    before = snapshot(actual)
    with pytest.raises(CheckpointError):
        consume(path, actual, "resume")
    equal_tree(snapshot(actual), before)


@pytest.mark.parametrize("options", [{"foreach": True}, {"fused": True}, {"fused": True, "capturable": True},
                                    {"maximize": True, "weight_decay": 0.01}])
def test_supported_adam_options_resume_and_real_update(tmp_path, options):
    source = runner(tmp_path / "source")
    source.alg.optimizer = torch.optim.Adam(source.alg.actor_critic.parameters(), **options)
    path = source.learn(1)
    target = runner(tmp_path / "target")
    consume(path, target, "resume")
    equal_tree(target.alg.optimizer.state_dict(), source.alg.optimizer.state_dict())
    target.learn(1)
    assert all(torch.isfinite(p).all() for p in target.alg.actor_critic.parameters())


def test_rollback_covers_optimizer_parameters_outside_the_model(tmp_path):
    actual = runner(tmp_path / "target")
    outside = torch.nn.Parameter(torch.ones(2))
    outside.grad = torch.ones_like(outside)
    actual.alg.optimizer.add_param_group({"params": [outside]})
    model = actual.alg.actor_critic
    def fail(optimizer):
        outside.grad.mul_(3)
        raise RuntimeError("outside gradient failure")
    handle = actual.alg.optimizer.register_load_state_dict_post_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="outside gradient"):
            apply_checkpoint_state(model, model.state_dict(), optimizer=actual.alg.optimizer,
                                   optimizer_state_dict=actual.alg.optimizer.state_dict())
    finally:
        handle.remove()
    assert torch.equal(outside.grad, torch.ones_like(outside))


@pytest.mark.parametrize("mode", ["inference", "warm_start", "resume", "model_only"])
def test_successful_load_preserves_existing_gradient_presence_and_values(tmp_path, mode):
    actual = runner(tmp_path / "target")
    parameters = list(actual.alg.actor_critic.parameters())
    parameters[0].grad = torch.ones_like(parameters[0])
    parameters[1].grad = None
    path = tmp_path / "valid.json"
    publish(path, actual, optimizer_state=actual.alg.optimizer.state_dict() if mode == "resume" else None, mode=mode)
    before = snapshot(actual)[4]
    def hook(*args):
        parameters[0].grad.mul_(3)
        parameters[1].grad = torch.ones_like(parameters[1])
    if mode == "resume": handle = actual.alg.optimizer.register_load_state_dict_post_hook(hook)
    else: handle = actual.alg.actor_critic.register_load_state_dict_post_hook(hook)
    try:
        consume(path, actual, mode)
    finally:
        handle.remove()
    equal_tree(snapshot(actual)[4], before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA; CPU suite covers a simulated two-device branch")
@pytest.mark.parametrize("fails", [False, True])
def test_actual_initialized_cuda_rng_preserved(fails):
    model = torch.nn.Linear(2, 2).cuda()
    torch.cuda.get_rng_state_all()  # Initialize device generators before transaction.
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all()
    def hook(model, incompatible):
        torch.rand(2)
        for device in range(torch.cuda.device_count()): torch.rand(2, device="cuda:%d" % device)
        if fails: raise RuntimeError("actual CUDA RNG failure")
    handle = model.register_load_state_dict_post_hook(hook)
    try:
        if fails:
            with pytest.raises(RuntimeError, match="actual CUDA RNG"):
                apply_checkpoint_state(model, model.state_dict())
        else:
            apply_checkpoint_state(model, model.state_dict())
    finally:
        handle.remove()
    assert torch.equal(torch.get_rng_state(), cpu_before)
    equal_tree(torch.cuda.get_rng_state_all(), cuda_before)
