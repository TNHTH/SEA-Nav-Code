import json

import pytest
import torch

from rsl_rl.utils.checkpoint import CheckpointError, load_checkpoint_v2, save_checkpoint_v2

COMMIT = "5" * 40
CONFIG = "a" * 64


def adam_fixture():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    return model, optimizer


def save(path, model, optimizer=None, iteration=0, **kwargs):
    return save_checkpoint_v2(path, model_state_dict=model.state_dict(),
                              optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
                              iteration=iteration, producer_commit=COMMIT,
                              resolved_config_sha256=CONFIG, **kwargs)


def test_real_adam_moments_and_none_round_trip(tmp_path):
    model, optimizer = adam_fixture()
    path = tmp_path / "model_1.manifest.json"
    manifest = save(path, model, optimizer, 1)
    loaded = load_checkpoint_v2(path, artifact_root=tmp_path, expected_resolved_config_sha256=CONFIG)
    target, restored = adam_fixture()
    target.load_state_dict(loaded.model_state_dict)
    restored.load_state_dict(loaded.optimizer_state_dict)
    assert manifest == loaded.manifest
    assert loaded.iteration == 1
    assert loaded.optimizer_state_dict["state"]
    assert all(type(k) is int for k in loaded.optimizer_state_dict["state"])
    for old, new in zip(optimizer.state.values(), restored.state.values()):
        assert torch.equal(old["exp_avg"], new["exp_avg"])
        assert torch.equal(old["exp_avg_sq"], new["exp_avg_sq"])
        assert torch.equal(old["step"], new["step"])
    assert restored.param_groups[0]["lr"] == 0.003
    assert restored.param_groups[0]["foreach"] is None
    assert set(manifest.allowed_sections) == {"model_state_dict", "optimizer_state_dict", "iteration"}


def test_model_only_payload_and_explicit_identity(tmp_path):
    model, _ = adam_fixture()
    path = tmp_path / "init.manifest.json"
    save(path, model)
    loaded = load_checkpoint_v2(path, artifact_root=tmp_path)
    assert loaded.optimizer_state_dict is None and loaded.iteration == 0
    with pytest.raises(CheckpointError, match="config"):
        load_checkpoint_v2(path, artifact_root=tmp_path, expected_resolved_config_sha256="b" * 64)
    with pytest.raises(CheckpointError, match="producer_commit"):
        save_checkpoint_v2(path, model_state_dict=model.state_dict(), iteration=0,
                           producer_commit="unknown", resolved_config_sha256=CONFIG)


@pytest.mark.parametrize("iteration", [-1, True, 1.5])
def test_invalid_iteration_rejected_before_publication(tmp_path, iteration):
    with pytest.raises(CheckpointError, match="iteration"):
        save(tmp_path / "bad.json", torch.nn.Linear(1, 1), iteration=iteration)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(infos={}),
    lambda p: p["model_state_dict"].update({1: torch.ones(1)}),
    lambda p: p["model_state_dict"].update(x=float("nan")),
    lambda p: p["optimizer_state_dict"]["state"].update({"0": {}}),
    lambda p: p["optimizer_state_dict"]["state"].update({True: {}}),
    lambda p: p["optimizer_state_dict"]["param_groups"][0].update(params=[-1]),
    lambda p: p["optimizer_state_dict"]["state"][0].update(bad={1: 2}),
])
def test_loaded_payload_grammar_is_not_a_pickle_allowlist(tmp_path, mutation):
    import hashlib
    model, optimizer = adam_fixture()
    path = tmp_path / "model.json"
    save(path, model, optimizer, 1)
    record = json.loads(path.read_text())
    artifact = tmp_path / record["artifact_path"]
    payload = torch.load(artifact, weights_only=True)
    mutation(payload)
    torch.save(payload, artifact)
    record.update(byte_size=artifact.stat().st_size, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    path.write_text(json.dumps(record))
    with pytest.raises(CheckpointError):
        load_checkpoint_v2(path, artifact_root=tmp_path)


@pytest.mark.parametrize("change", [{"unknown": 1}, {"iteration": True}, {"iteration": 5},
                                    {"byte_size": 1}, {"sha256": "b" * 64},
                                    {"allowed_sections": ["model_state_dict"]}])
def test_manifest_exactness_and_payload_binding(tmp_path, change):
    path = tmp_path / "model.json"
    save(path, torch.nn.Linear(1, 1))
    record = json.loads(path.read_text())
    record.update(change)
    path.write_text(json.dumps(record))
    with pytest.raises(CheckpointError):
        load_checkpoint_v2(path, artifact_root=tmp_path)
