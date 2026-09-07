from types import SimpleNamespace

import pytest
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.checkpoint import CheckpointError, load_checkpoint_v2
from test_checkpoint_v2 import COMMIT, CONFIG, save


class CpuFixture:
    """Tensor transition fixture for real PPO; no simulator interface claim."""
    num_envs = 2
    num_props = 12
    num_obs = 38
    num_nav_actions = 3
    max_episode_length = 1000
    cfg = SimpleNamespace(env=SimpleNamespace(his_len=2))

    def __init__(self):
        self.rays = torch.ones(2, 5)
        self.t = 0
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)

    def get_observations(self):
        x = torch.linspace(-0.2, 0.3, 38).repeat(2, 1)
        x[1] += 0.1
        return x + (self.t % 3) * 0.01

    def get_privileged_observations(self): return None
    def get_extras(self): return {}
    def reset(self): return self.get_observations(), None

    def step(self, actions):
        self.t += 1
        return self.get_observations(), None, torch.tensor([0.2, -0.1]), torch.zeros(2, dtype=torch.bool), {}


def runner(path):
    torch.set_num_threads(1)
    cfg = {"runner": {"policy_class_name": "ActorCritic", "algorithm_class_name": "PPO",
                      "num_steps_per_env": 2, "save_interval": 1},
           "policy": {"actor_hidden_dims": [8], "critic_hidden_dims": [8],
                      "encoder_hidden_dims": [8], "init_noise_std": 0.3},
           "algorithm": {"num_learning_epochs": 1, "num_mini_batches": 1, "schedule": "fixed"}}
    return OnPolicyRunner(CpuFixture(), cfg, log_dir=str(path), device="cpu",
                          producer_commit=COMMIT, resolved_config_sha256=CONFIG)


def test_actual_runner_persistence_and_n_plus_m_continuation(tmp_path):
    first = runner(tmp_path / "first")
    first.alg.optimizer.param_groups[0]["lr"] = 0.0023
    final = first.learn(3)
    assert final.name == "model_3.manifest.json"
    checkpoint = load_checkpoint_v2(final, artifact_root=tmp_path)
    assert checkpoint.iteration == 3
    assert checkpoint.optimizer_state_dict["state"]
    second = runner(tmp_path / "second")
    second.load(final, artifact_root=tmp_path, mode="resume")
    assert second.current_learning_iteration == 3
    assert second.alg.learning_rate == 0.0023
    for old, new in zip(first.alg.optimizer.state.values(), second.alg.optimizer.state.values()):
        assert torch.equal(old["exp_avg"], new["exp_avg"])
        assert torch.equal(old["exp_avg_sq"], new["exp_avg_sq"])
    before = {k: v.clone() for k, v in second.alg.actor_critic.state_dict().items()}
    final = second.learn(2)
    assert final.name == "model_5.manifest.json"
    assert load_checkpoint_v2(final, artifact_root=tmp_path).iteration == 5
    assert any(not torch.equal(before[k], v) for k, v in second.alg.actor_critic.state_dict().items())
    for path in (tmp_path / "first").glob("*.manifest.json"):
        loaded = load_checkpoint_v2(path, artifact_root=tmp_path)
        assert path.name == "model_%d.manifest.json" % loaded.iteration


def test_iteration_is_updated_immediately_after_success_and_not_after_failure(tmp_path):
    actual = runner(tmp_path / "run")
    update = actual.alg.update
    count = [0]
    def fail_second():
        count[0] += 1
        if count[0] == 2:
            raise RuntimeError("update failed")
        return update()
    actual.alg.update = fail_second
    with pytest.raises(RuntimeError, match="update failed"):
        actual.learn(3)
    assert actual.current_learning_iteration == 1
    assert load_checkpoint_v2(tmp_path / "run/model_1.manifest.json", artifact_root=tmp_path).iteration == 1


def test_warm_start_resume_and_inference_have_distinct_state_effects(tmp_path):
    trained = runner(tmp_path / "trained")
    final = trained.learn(1)
    target = runner(tmp_path / "target")
    with pytest.raises(CheckpointError, match="model-only.*zero"):
        target.load(final, artifact_root=tmp_path, mode="warm_start")
    init = tmp_path / "init.json"
    save(init, trained.alg.actor_critic)
    with pytest.raises(CheckpointError, match="optimizer"):
        target.load(init, artifact_root=tmp_path, mode="resume")
    target.load(init, artifact_root=tmp_path, mode="warm_start")
    assert target.current_learning_iteration == 0 and not target.alg.optimizer.state
    target.load(final, artifact_root=tmp_path, mode="inference")
    assert target.current_learning_iteration == 0 and not target.alg.optimizer.state
