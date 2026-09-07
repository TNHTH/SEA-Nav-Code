"""CPU contracts for sampled actions and current-minibatch graph ownership."""

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic


ACTORS = (ActorCritic, DifferentiableSafeActorCritic)
AUX_NAMES = ("alpha", "rays_real", "u_bar", "u_s")


def make_actor(actor_type=DifferentiableSafeActorCritic):
    torch.manual_seed(41)
    actor = actor_type(num_actions=3, actor_hidden_dims=[16],
                       critic_hidden_dims=[16], encoder_hidden_dims=[16],
                       num_props=12, num_rays=5, his_len=2, init_noise_std=0.3)
    if isinstance(actor, DifferentiableSafeActorCritic):
        with torch.no_grad():
            actor.nav_head[-1].weight.mul_(0.02)
            actor.nav_head[-1].bias.copy_(torch.tensor([0.5, 0.1, 0.05]))
    return actor


def make_observations(batch_size=2, step=0):
    history = torch.zeros(batch_size, 2, 19)
    for env in range(batch_size):
        for frame in range(2):
            offset = 0.13 * env + 0.07 * step + 0.03 * frame
            history[env, frame, :12] = torch.linspace(-0.2, 0.3, 12) + offset
            distances = torch.tensor([1.0, 1.0, 0.1, 1.0, 1.0])
            history[env, frame, 12:17] = torch.log2(distances + offset * 0.1)
            history[env, frame, -2:] = torch.tensor([0.8, -0.2]) + offset
    return history.flatten(1)


def snapshot(actor):
    names = ("mean",) + (AUX_NAMES if hasattr(actor, "alpha") else ())
    refs = {name: getattr(actor, name) for name in names}
    refs.update(mu=actor.action_mean, sigma=actor.action_std)
    return actor.distribution, refs, {k: v.detach().clone() for k, v in refs.items()}


def assert_snapshot(actor, saved):
    distribution, refs, values = saved
    assert actor.distribution is distribution
    for name, ref in refs.items():
        actual = (actor.action_mean if name == "mu" else
                  actor.action_std if name == "sigma" else getattr(actor, name))
        assert actual is ref, name
        torch.testing.assert_close(actual, values[name], rtol=0, atol=0)


class AdditiveShield(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, actions, rays, alpha):
        self.calls += 1
        return actions + actions.new_tensor([0.2, -0.1, 0.3])


def test_sample_is_exact_draw_from_mean_stage_normal():
    actor = make_actor()
    shield = AdditiveShield()
    actor.cbf_layer = shield
    torch.manual_seed(23)
    actions = actor.act(make_observations())
    torch.manual_seed(23)
    expected = Normal(actor.action_mean.detach(), actor.action_std.detach()).sample()
    torch.testing.assert_close(actions, expected, rtol=0, atol=0)
    assert shield.calls == 1


@pytest.mark.parametrize("actor_type", ACTORS)
def test_pure_query_preserves_active_distribution_and_auxiliary_objects(actor_type):
    actor = make_actor(actor_type)
    obs = make_observations()
    actor.act(obs)
    saved = snapshot(actor)
    query = actor.action_mean_for(make_observations(step=4), masks=None, hidden_states=None)
    assert query.requires_grad
    assert not torch.equal(query, saved[2]["mu"])
    assert_snapshot(actor, saved)


def test_first_ordinary_pure_query_does_not_create_distribution():
    actor = make_actor(ActorCritic)
    assert actor.distribution is None
    assert actor.action_mean_for(make_observations()).requires_grad
    assert actor.distribution is None
    assert not hasattr(actor, "mean")


@pytest.mark.parametrize("actor_type", ACTORS)
@pytest.mark.parametrize("supply_originals", [False, True])
def test_smoothness_preserves_minibatch_state(actor_type, supply_originals):
    actor = make_actor(actor_type)
    ppo = PPO(actor)
    obs, next_obs = make_observations(), make_observations(step=4)
    actor.act(obs)
    saved = snapshot(actor)
    kwargs = dict(orig_mu=actor.action_mean, orig_values=actor.evaluate(obs)) if supply_originals else {}
    for original in kwargs.values():
        original.retain_grad()
    loss = ppo.compute_smoothness_loss(obs, next_obs, **kwargs)
    assert torch.isfinite(loss) and loss.item() > 0
    assert_snapshot(actor, saved)
    loss.backward()
    for original in kwargs.values():
        assert original.grad is not None
        assert torch.isfinite(original.grad).all() and original.grad.abs().sum() > 0


@pytest.mark.parametrize("actor_type", ACTORS)
def test_pure_mean_and_critic_have_separate_live_parameter_graphs(actor_type):
    actor = make_actor(actor_type)
    actor.action_mean_for(make_observations()).square().sum().backward()
    policy_modules = (actor.actor, actor.encoder) if actor_type is ActorCritic else (
        actor.backbone, actor.nav_head, actor.alpha_head)
    for module in policy_modules:
        assert_live_gradients(module)
    if actor_type is DifferentiableSafeActorCritic:
        assert all(p.grad is None for p in actor.encoder.parameters())
    actor.zero_grad(set_to_none=True)
    actor.evaluate(make_observations(step=1)).square().sum().backward()
    assert_live_gradients(actor.encoder)
    assert_live_gradients(actor.critic)


def assert_live_gradients(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


@pytest.mark.parametrize("unsupported", ["missing_query", "recurrent"])
def test_unsupported_actor_contract_fails_clearly(unsupported):
    actor = make_actor()
    if unsupported == "missing_query":
        actor.action_mean_for = None
        message = "action_mean_for"
    else:
        actor.is_recurrent = True
        message = "recurrent"
    with pytest.raises((TypeError, ValueError), match=message):
        PPO(actor)


def collect_rollout(ppo, rollout):
    records = []
    with torch.no_grad():
        for step in range(2):
            obs = make_observations(step=rollout * 3 + step)
            next_obs = make_observations(step=rollout * 3 + step + 1)
            action = ppo.act(obs, obs)
            mean = ppo.actor_critic.action_mean.detach().clone()
            std = ppo.actor_critic.action_std.detach().clone()
            log_prob = Normal(mean, std).log_prob(action).sum(-1)
            records.extend(zip(obs.clone(), action.clone(), log_prob.clone(), mean, std))
            # Environment-side clipping is out of place and cannot replace policy_action.
            clipped_policy_action = action.clamp(-0.1, 0.1)
            assert not torch.equal(clipped_policy_action, action)
            ppo.process_env_step(next_obs, torch.tensor([0.2 + step, -0.1 - step]),
                                 torch.zeros(2, dtype=torch.bool),
                                 {"bad_masks": torch.zeros(2, dtype=torch.bool)})
            for stored, expected in ((ppo.storage.observations[step], obs),
                                     (ppo.storage.next_observations[step], next_obs),
                                     (ppo.storage.actions[step], action),
                                     (ppo.storage.mu[step], mean),
                                     (ppo.storage.sigma[step], std),
                                     (ppo.storage.actions_log_prob[step, :, 0], log_prob)):
                torch.testing.assert_close(stored, expected, rtol=0, atol=0)
        ppo.compute_returns(next_obs)
    for name in ("observations", "next_observations", "actions", "actions_log_prob",
                 "mu", "sigma", "values", "returns", "advantages"):
        target = getattr(ppo.storage, name)
        assert not target.requires_grad and target.grad_fn is None, name
    return records


@pytest.mark.parametrize("actor_type", ACTORS)
def test_real_rollouts_keep_sample_likelihood_and_minibatch_identity(monkeypatch, actor_type):
    actor = make_actor(actor_type)
    ppo = PPO(actor, num_learning_epochs=2, num_mini_batches=2,
              learning_rate=1e-3, schedule="fixed", desired_kl=None)
    ppo.init_storage(2, 2, (38,), (3,))
    state, batches, ratios, regularizations, interventions = {}, [], [], [], []
    std_gradient_norms = []
    real_generator = ppo.storage.mini_batch_generator
    real_act, real_evaluate = actor.act, actor.evaluate
    real_log_prob = actor.get_actions_log_prob
    real_smoothness, real_alpha = ppo.compute_smoothness_loss, ppo.compute_alpha_loss
    real_step = ppo.optimizer.step

    def generator(*args, **kwargs):
        for batch in real_generator(*args, **kwargs):
            state.clear()
            state["batch"] = batch
            batches.append(batch)
            assert len(batch) == 12 and batch[0].shape == (2, 38)
            assert (~batch[11].bool()).sum() == 2
            for row in range(2):
                matches = [record for record in records if torch.equal(record[0], batch[0][row])]
                assert len(matches) == 1
                for actual, expected in zip((batch[0][row], batch[2][row], batch[6][row, 0],
                                             batch[7][row], batch[8][row]), matches[0]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            yield batch
            assert state["stepped"]
        state.clear()

    def act(obs, **kwargs):
        result = real_act(obs, **kwargs)
        if "batch" in state:
            assert obs is state["batch"][0]
            state["snapshot"] = snapshot(actor)
            state["fresh_sample"] = result
        return result

    def evaluate(obs, **kwargs):
        result = real_evaluate(obs, **kwargs)
        if "batch" in state and "value" not in state:
            assert obs is state["batch"][0]
            state["value"] = result
        return result

    def log_prob(actions):
        result = real_log_prob(actions)
        if "batch" in state:
            assert actions is state["batch"][2]
            assert actions is not state["fresh_sample"]
            torch.testing.assert_close(result, Normal(actor.action_mean, actor.action_std).log_prob(actions).sum(-1))
            ratios.append(torch.exp(result.detach() - state["batch"][6].flatten()))
        return result

    def smoothness(current, following, *, orig_mu=None, orig_values=None):
        saved = state["snapshot"]
        assert orig_mu is saved[1]["mu"]
        assert orig_values is state["value"]
        loss = real_smoothness(current, following, orig_mu=orig_mu, orig_values=orig_values)
        assert_snapshot(actor, saved)
        mu = saved[2]["mu"]
        range_loss = (mu - mu.clamp(mu.new_tensor([-0.5, -0.8, -1.0]),
                                   mu.new_tensor([1.7, 0.8, 1.0]))).square().sum(-1).mean()
        regularizations.append(range_loss.item() + 0.05 * loss.item())
        if actor_type is DifferentiableSafeActorCritic:
            changed = actor._compute_safe_action_mean(current + 0.5 * (following - current))
            for name, value in zip(("u_s", "alpha", "rays_real", "u_bar"), changed):
                assert not torch.equal(value, saved[2][name]), name
            intervention = (saved[2]["u_s"] - saved[2]["u_bar"]).square().sum(-1).mean().item()
            assert intervention > 0
            interventions.append(intervention)
        else:
            interventions.append(0.0)
        return loss

    def alpha_loss(alpha, **kwargs):
        assert alpha is state["snapshot"][1]["alpha"]
        return real_alpha(alpha, **kwargs)

    def optimizer_step(*args, **kwargs):
        assert_snapshot(actor, state["snapshot"])
        assert actor.std.grad is not None
        assert torch.isfinite(actor.std.grad).all()
        std_gradient_norms.append(actor.std.grad.abs().sum().item())
        for module in ((actor.actor, actor.encoder, actor.critic) if actor_type is ActorCritic else
                       (actor.nav_head, actor.backbone, actor.alpha_head, actor.encoder, actor.critic)):
            assert_live_gradients(module)
        result = real_step(*args, **kwargs)
        state["stepped"] = True
        return result

    monkeypatch.setattr(ppo.storage, "mini_batch_generator", generator)
    monkeypatch.setattr(actor, "act", act)
    monkeypatch.setattr(actor, "evaluate", evaluate)
    monkeypatch.setattr(actor, "get_actions_log_prob", log_prob)
    monkeypatch.setattr(ppo, "compute_smoothness_loss", smoothness)
    monkeypatch.setattr(ppo, "compute_alpha_loss", alpha_loss)
    monkeypatch.setattr(ppo.optimizer, "step", optimizer_step)
    for rollout in range(2):
        records = collect_rollout(ppo, rollout)
        before = [p.detach().clone() for p in actor.parameters()]
        batches.clear()
        ratios.clear()
        regularizations.clear()
        interventions.clear()
        std_gradient_norms.clear()
        metrics = ppo.update()
        assert len(batches) == len(ratios) == len(regularizations) == len(interventions) == 4
        assert len(metrics) == 5 and all(torch.isfinite(torch.tensor(value)) for value in metrics)
        assert metrics[2] == pytest.approx(sum(regularizations) / 4)
        assert metrics[4] == pytest.approx(sum(interventions) / 4)
        torch.testing.assert_close(ratios[0], torch.ones(2), rtol=1e-5, atol=1e-6)
        assert any(not torch.allclose(ratio, torch.ones(2), rtol=1e-5, atol=1e-6) for ratio in ratios[1:])
        assert any(not torch.equal(p, old) for p, old in zip(actor.parameters(), before))
        assert ppo.storage.step == 0
        # A later clipped surrogate may have zero std gradient in a valid update.
        assert max(std_gradient_norms) > 0
