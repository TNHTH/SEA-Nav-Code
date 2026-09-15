"""Regression tests for the SEA transition-eligibility contract.

These tests intentionally use a tiny actor so that failures exercise the PPO
state machine rather than Isaac/physics dependencies.  'bad_masks' is the
legacy external spelling (zero means eligible); PPO must convert it to the
positive internal 'eligible' mask before doing any model work.
"""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage.rollout_storage import RolloutStorage


class SpyActor(nn.Module):
    """Small actor/critic with observable forward and context calls."""

    is_recurrent = False

    def __init__(self, obs_dim=4, action_dim=2):
        super().__init__()
        self.policy = nn.Linear(obs_dim, action_dim)
        self.value = nn.Linear(obs_dim, 1)
        self.std = nn.Parameter(torch.full((action_dim,), 0.25))
        self.distribution = None
        self.act_calls = []
        self.eval_calls = []
        self.mean_calls = []

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _context_bias(self, actor_context, batch, device, dtype):
        if actor_context is None:
            return torch.zeros(batch, 1, device=device, dtype=dtype)
        if isinstance(actor_context, dict):
            bias = actor_context["bias"]
        else:
            bias = actor_context
        return bias.to(device=device, dtype=dtype).reshape(batch, 1)

    def action_mean_for(self, observations, actor_context=None, **kwargs):
        self.mean_calls.append((observations, actor_context))
        return self.policy(observations) + self._context_bias(
            actor_context, observations.shape[0], observations.device, observations.dtype
        )

    def act(self, observations, actor_context=None, **kwargs):
        self.act_calls.append((observations, actor_context))
        mean = self.action_mean_for(observations, actor_context=actor_context)
        self.distribution = Normal(mean, self.std.expand_as(mean))
        return self.distribution.sample()

    def evaluate(self, observations, actor_context=None, **kwargs):
        self.eval_calls.append((observations, actor_context))
        return self.value(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        return None


def _transition(obs, next_obs, *, bad=None, done=None, context=None, next_context=None):
    t = RolloutStorage.Transition()
    n = obs.shape[0]
    t.observations = obs
    t.next_observations = next_obs
    t.actions = torch.zeros(n, 2)
    t.rewards = torch.zeros(n)
    t.dones = torch.zeros(n, dtype=torch.bool) if done is None else done
    t.values = torch.zeros(n, 1)
    t.actions_log_prob = torch.zeros(n)
    t.action_mean = torch.zeros(n, 2)
    t.action_sigma = torch.ones(n, 2)
    t.bad_masks = bad
    t.actor_context = context
    t.next_actor_context = next_context
    return t


def test_missing_bad_masks_overwrites_previous_slot_with_eligible():
    storage = RolloutStorage(1, 1, (2,), (1,))
    first = _transition(torch.zeros(1, 2), torch.zeros(1, 2), bad=torch.ones(1, dtype=torch.bool))
    first.actions = torch.zeros(1, 1)
    first.action_mean = torch.zeros(1, 1)
    first.action_sigma = torch.ones(1, 1)
    storage.add_transitions(first)
    assert bool(storage.bad_masks[0, 0, 0])

    storage.clear()
    second = _transition(torch.zeros(1, 2), torch.zeros(1, 2), bad=None)
    second.actions = torch.zeros(1, 1)
    second.action_mean = torch.zeros(1, 1)
    second.action_sigma = torch.ones(1, 1)
    storage.add_transitions(second)
    assert not bool(storage.bad_masks[0, 0, 0]), "omitted mask must not leak the prior rollout"


def test_compute_returns_truncates_gae_and_normalizes_only_eligible_rows():
    storage = RolloutStorage(1, 3, (1,), (1,))
    storage.rewards[:, 0, 0] = torch.tensor([1.0, 100.0, 3.0])
    storage.values[:, 0, 0] = torch.tensor([10.0, 20.0, 30.0])
    storage.dones[:, 0, 0] = False
    # Step 1 is ineligible.  It must be an exact value baseline and must not
    # carry its huge reward into either neighboring GAE segment.
    storage.bad_masks[:, 0, 0] = torch.tensor([0, 1, 0], dtype=torch.uint8)
    storage.compute_returns(torch.tensor([[40.0]]), gamma=0.9, lam=0.8)

    assert storage.returns[1, 0, 0].item() == pytest.approx(20.0)
    assert storage.advantages[1, 0, 0].item() == pytest.approx(0.0)
    # Step 0 cannot bootstrap through the ineligible step.
    assert storage.returns[0, 0, 0].item() == pytest.approx(1.0)
    # Step 2 remains an independent valid segment.
    assert storage.returns[2, 0, 0].item() == pytest.approx(3.0 + 0.9 * 40.0)
    eligible = storage.bad_masks == 0
    assert torch.isfinite(storage.advantages).all()
    assert storage.advantages[eligible].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert storage.advantages[eligible].std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)


def _fill_storage(ppo, obs, next_obs, bad, done=None, context=None, next_context=None):
    storage = ppo.storage
    n = obs.shape[0]
    storage.observations[0].copy_(obs)
    storage.next_observations[0].copy_(next_obs)
    storage.actions[0].zero_()
    storage.rewards[0].zero_()
    storage.dones[0].copy_(torch.zeros(n, 1, dtype=torch.bool) if done is None else done.view(-1, 1))
    storage.values[0].zero_()
    storage.returns[0].zero_()
    storage.advantages[0].fill_(1.0)
    storage.actions_log_prob[0].zero_()
    storage.mu[0].zero_()
    storage.sigma[0].fill_(0.25)
    storage.bad_masks[0].copy_(bad.view(-1, 1).to(torch.uint8))
    if context is not None:
        storage.set_actor_context(0, context, next_context)
    storage.step = 1


def test_update_filters_nonfinite_ineligible_rows_before_any_forward():
    torch.manual_seed(4)
    actor = SpyActor()
    ppo = PPO(actor, num_learning_epochs=1, num_mini_batches=1, desired_kl=None)
    ppo.init_storage(2, 1, (4,), (2,))
    obs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [float("nan"), 0.0, 0.0, 0.0]])
    _fill_storage(ppo, obs, obs.nan_to_num() + 0.1, torch.tensor([0, 1]))
    ppo.storage.returns[0, :, 0] = 0.5
    ppo.storage.advantages[0, :, 0] = 1.0

    ppo.update()
    assert actor.act_calls and actor.eval_calls
    assert all(call[0].shape[0] == 1 for call in actor.act_calls)
    assert all(torch.isfinite(call[0]).all() for call in actor.act_calls)
    assert all(call[0].shape[0] == 1 for call in actor.eval_calls)


def test_all_bad_update_is_a_true_noop_and_counts_skip():
    torch.manual_seed(55)
    actor = SpyActor()
    ppo = PPO(actor, num_learning_epochs=2, num_mini_batches=2, desired_kl=0.01, schedule="adaptive")
    ppo.init_storage(2, 1, (4,), (2,))
    obs = torch.ones(2, 4)
    _fill_storage(ppo, obs, obs + 0.1, torch.ones(2, dtype=torch.bool))
    before_params = [p.detach().clone() for p in actor.parameters()]
    before_optim = deepcopy(ppo.optimizer.state_dict())
    before_rng = torch.get_rng_state().clone()
    before_lr = ppo.learning_rate
    result = ppo.update()

    for actual, expected in zip(actor.parameters(), before_params):
        torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)
    assert ppo.optimizer.state_dict() == before_optim
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert ppo.learning_rate == before_lr
    assert not actor.act_calls and not actor.eval_calls
    assert ppo.last_update_stats["skipped_all_bad_minibatches"] >= 1
    assert all(value == 0.0 for value in result)


def test_actor_context_is_detached_cloned_and_row_aligned():
    torch.manual_seed(8)
    actor = SpyActor()
    ppo = PPO(actor, num_learning_epochs=1, num_mini_batches=1, desired_kl=None)
    ppo.init_storage(2, 1, (4,), (2,))
    obs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    context = {"bias": torch.tensor([[1.0], [2.0]], requires_grad=True)}
    next_context = {"bias": torch.tensor([[3.0], [4.0]], requires_grad=True)}
    with torch.no_grad():
        ppo.act(obs, obs, actor_context=context)
        ppo.process_env_step(obs + 1, torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                             {"bad_masks": torch.zeros(2, dtype=torch.bool)},
                             next_actor_context=next_context)
    with torch.no_grad():
        context["bias"].fill_(99.0)
        next_context["bias"].fill_(98.0)
    stored = ppo.storage.actor_context
    stored_next = ppo.storage.next_actor_context
    assert stored is not None and stored_next is not None
    assert not stored["bias"].requires_grad and not stored_next["bias"].requires_grad
    torch.testing.assert_close(stored["bias"][0, :, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(stored_next["bias"][0, :, 0], torch.tensor([3.0, 4.0]))
    assert actor.act_calls[0][1]["bias"].shape == (2, 1)


def test_nonterminal_context_transition_requires_next_context():
    actor = SpyActor()
    ppo = PPO(actor)
    ppo.init_storage(1, 1, (4,), (2,))
    obs = torch.zeros(1, 4)
    context = {"bias": torch.zeros(1, 1)}
    with torch.no_grad():
        ppo.act(obs, obs, actor_context=context)
    with pytest.raises(ValueError, match="next_actor_context"):
        ppo.process_env_step(obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), {},
                             next_actor_context=None)


def test_smoothness_pair_mask_has_no_rng_draw_for_empty_pair_set():
    actor = SpyActor()
    ppo = PPO(actor)
    current = torch.zeros(3, 4)
    following = torch.ones(3, 4)
    before = torch.get_rng_state().clone()
    loss = ppo.compute_smoothness_loss(current, following,
                                       pair_mask=torch.zeros(3, dtype=torch.bool))
    assert loss.item() == 0.0
    assert torch.equal(torch.get_rng_state(), before)


def test_mixed_rollout_skips_bad_minibatches_and_steps_good_ones():
    torch.manual_seed(33)
    actor = SpyActor()
    ppo = PPO(actor, num_learning_epochs=1, num_mini_batches=2, desired_kl=None)
    ppo.init_storage(2, 1, (4,), (2,))
    obs = torch.ones(2, 4)
    _fill_storage(ppo, obs, obs + 0.1, torch.tensor([0, 1], dtype=torch.bool))
    before_optim = deepcopy(ppo.optimizer.state_dict())

    ppo.update()

    stats = ppo.last_update_stats
    assert stats["total_minibatches"] == 2
    assert stats["skipped_all_bad_minibatches"] == 1
    assert stats["optimizer_steps"] == 1
    assert stats["eligible_samples_seen"] == 1
    assert len(actor.act_calls) == 1
    assert actor.act_calls[0][0].shape[0] == 1
    assert ppo.optimizer.state_dict() != before_optim


def test_all_done_rollout_has_no_smoothness_pair_and_no_rng_draw(monkeypatch):
    torch.manual_seed(21)
    actor = SpyActor()
    ppo = PPO(actor, num_learning_epochs=1, num_mini_batches=1, desired_kl=None)
    ppo.init_storage(2, 1, (4,), (2,))
    obs = torch.ones(2, 4)
    _fill_storage(ppo, obs, obs + 0.1, torch.zeros(2, dtype=torch.bool),
                  done=torch.ones(2, dtype=torch.bool))
    rand_calls = []
    original_rand = torch.rand

    def spy_rand(*args, **kwargs):
        rand_calls.append((args, kwargs))
        return original_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", spy_rand)
    result = ppo.update()
    assert ppo.last_update_stats["smooth_pairs_seen"] == 0
    assert rand_calls == []
    assert result[3] == 0.0


def test_eligible_done_rows_excluded_from_smoothness_pairs():
    torch.manual_seed(9)
    actor = SpyActor()
    ppo = PPO(actor, num_learning_epochs=1, num_mini_batches=1, desired_kl=None)
    ppo.init_storage(2, 1, (4,), (2,))
    obs = torch.ones(2, 4)
    _fill_storage(ppo, obs, obs + 0.1, torch.zeros(2, dtype=torch.bool),
                  done=torch.tensor([False, True]))

    ppo.update()

    stats = ppo.last_update_stats
    assert stats["eligible_samples_seen"] == 2
    assert stats["smooth_pairs_seen"] == 1
    assert stats["optimizer_steps"] == 1
    assert stats["skipped_all_bad_minibatches"] == 0


def test_range_bounds_use_legacy_prefix_and_reject_shorter_config():
    actor = SpyActor()
    ppo = PPO(actor)
    low, high = ppo._bounds_for(torch.zeros(3, 2))
    torch.testing.assert_close(low, torch.tensor([-0.5, -0.8]))
    torch.testing.assert_close(high, torch.tensor([1.7, 0.8]))
    with pytest.raises(ValueError):
        ppo._bounds_for(torch.zeros(3, 4))
    with pytest.raises(ValueError, match="three entries"):
        PPO(actor, action_range_low=(-1.0, -1.0), action_range_high=(1.0, 1.0))


class KwargsOnlyActor(SpyActor):
    """Actor whose signatures swallow keywords, like the legacy Go2 modules."""

    def act(self, observations, **kwargs):
        return super().act(observations)

    def evaluate(self, observations, **kwargs):
        return super().evaluate(observations)

    def action_mean_for(self, observations, **kwargs):
        return super().action_mean_for(observations)


def test_kwargs_only_actor_is_rejected_in_context_mode():
    ppo = PPO(KwargsOnlyActor())
    ppo.init_storage(2, 1, (4,), (2,))
    context = {"bias": torch.zeros(2, 1)}
    with pytest.raises(TypeError, match="actor_context"):
        ppo.act(torch.zeros(2, 4), torch.zeros(2, 4), actor_context=context)


def test_context_mode_error_paths():
    actor = SpyActor()
    ppo = PPO(actor)
    ppo.init_storage(2, 1, (4,), (2,))
    obs = torch.zeros(2, 4)
    context = {"bias": torch.zeros(2, 1)}
    with torch.no_grad():
        ppo.act(obs, obs, actor_context=context)
    with pytest.raises(ValueError, match="actor_context is required"):
        ppo.act(obs, obs)

    fresh = PPO(SpyActor())
    fresh.init_storage(2, 1, (4,), (2,))
    with pytest.raises(ValueError, match="without actor_context"):
        fresh.process_env_step(obs, torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                               {}, next_actor_context=context)

    with pytest.raises(ValueError, match="last_actor_context is required"):
        ppo.compute_returns(obs)


def test_storage_requires_context_every_transition_once_enabled():
    storage = RolloutStorage(1, 2, (2,), (2,))
    context = {"bias": torch.zeros(1, 1)}
    first = _transition(torch.zeros(1, 2), torch.zeros(1, 2),
                        context=context, next_context=context)
    storage.add_transitions(first)
    second = _transition(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="every transition"):
        storage.add_transitions(second)


def test_context_tensors_must_be_at_least_two_dimensional():
    ppo = PPO(SpyActor())
    ppo.init_storage(2, 1, (4,), (2,))
    obs = torch.zeros(2, 4)
    context = {"bias": torch.zeros(2)}
    with torch.no_grad():
        ppo.act(obs, obs, actor_context=context)
    with pytest.raises(ValueError, match="two-dimensional"):
        ppo.process_env_step(obs + 1, torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                             {}, next_actor_context=context)


def test_mid_rollout_context_arrival_is_rejected():
    storage = RolloutStorage(1, 2, (2,), (2,))
    plain = _transition(torch.zeros(1, 2), torch.zeros(1, 2))
    storage.add_transitions(plain)
    context = {"bias": torch.zeros(1, 1)}
    late = _transition(torch.zeros(1, 2), torch.zeros(1, 2),
                       context=context, next_context=context)
    with pytest.raises(ValueError, match="mid-rollout"):
        storage.add_transitions(late)
