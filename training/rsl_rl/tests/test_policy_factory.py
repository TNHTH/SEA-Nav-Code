"""Real consumer tests catch ignored projections and coefficients."""
import dataclasses
import importlib
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from rsl_rl.experiment_config import build_policy_kwargs, build_ppo_kwargs, resolve_run_config
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.algorithms.ppo import PPO

ROOT = Path(__file__).resolve().parents[3]
DELTA = ("ppo_state_identity_repair",)
SMALL = dict(num_actions=3, actor_hidden_dims=[8], critic_hidden_dims=[8], encoder_hidden_dims=[8])


def factory():
    # A missing implementation fails the test rather than stopping collection.
    return importlib.import_module("rsl_rl.policy_factory")


def resolved():
    return resolve_run_config(registry_path=ROOT / "configs/parity_registry.yaml",
                              algorithm_profile="upstream_fbce672c", runtime_stack="isaac_gym_preview4",
                              implementation_delta=DELTA)


def test_resolved_factory_constructs_real_actor_and_ppo_without_mutating_config():
    config = resolved()
    actor = factory().build_actor_critic(config, **SMALL)
    ppo = factory().build_ppo(config, actor, learning_rate=0.)
    assert isinstance(actor, DifferentiableSafeActorCritic) and isinstance(ppo, PPO)
    assert actor.num_rays == 41 and actor.his_len == 10
    assert actor.observation_fov_deg == 240 and actor.cbf_layer.fov_deg == 180
    command = actor.cbf_layer(torch.zeros(1, 3), torch.full((1, 41), .1), torch.ones(1, 1))
    torch.testing.assert_close(command, torch.tensor([[-.211213003987952, 0., 0.]]), atol=1e-6, rtol=1e-6)
    assert ppo.alpha_min == 1 and ppo.alpha_penalty_coefficient == 1
    assert ppo.intervention_coefficient == .1
    assert ppo.actor_smoothness_coefficient == .05 and ppo.critic_smoothness_coefficient == .005
    assert tuple(ppo.action_range_low) == (-.5, -.8, -1.)
    assert tuple(ppo.action_range_high) == (1.7, .8, 1.)
    assert build_policy_kwargs(config).values["epsilon_d"] == 1


def test_policy_projection_applies_every_numerical_constructor_value():
    projection = build_policy_kwargs(resolved())
    values = projection.materialize_values()
    values.update(num_rays=5, history_frames=2, cbf_fov_deg=240., epsilon_d=2.,
                  kappa=4., safe_radius_m=.25, safety_margin_m=.1)
    changed = dataclasses.replace(projection, values=values)
    kwargs = factory().policy_constructor_kwargs(changed, DELTA, **SMALL)
    actor = DifferentiableSafeActorCritic(**kwargs)
    assert actor.num_rays == 5 and actor.his_len == 2 and actor.num_obs_hist == 38
    layer = actor.cbf_layer
    assert layer.fov_deg == 240 and layer.damping_factor == 2 and layer.kappa == 4
    assert layer.safe_radius == .25 and layer.safety_margin == .1
    assert layer.d_safe == pytest.approx(.35)
    assert torch.isfinite(actor.action_mean_for(torch.zeros(2, 38))).all()
    assert projection.values["num_rays"] == 41


@pytest.mark.parametrize("consumer", ["policy", "ppo"])
def test_projection_rejects_unknown_missing_conflicting_fields_and_activation_delta(consumer):
    config = resolved()
    projection = build_policy_kwargs(config) if consumer == "policy" else build_ppo_kwargs(config)
    convert = getattr(factory(), consumer + "_constructor_kwargs")
    original = projection.materialize_values()
    for values in (dict(original, invented=1), {}):
        with pytest.raises(ValueError, match="fields"):
            convert(dataclasses.replace(projection, values=values), DELTA)
    with pytest.raises(ValueError, match="consumer"):
        convert(dataclasses.replace(projection, consumer="wrong"), DELTA)
    with pytest.raises(ValueError, match="activation"):
        convert(dataclasses.replace(projection, required_activation_deltas=("undeclared",)), DELTA)
    key = "cbf_fov_deg" if consumer == "policy" else "alpha_min"
    with pytest.raises(ValueError, match="conflict"):
        convert(projection, DELTA, **{key: 999})
    with pytest.raises(ValueError, match="unknown"):
        convert(projection, DELTA, misspelled_setting=1)
    if consumer == "ppo":
        with pytest.raises(ValueError, match="activation"):
            convert(projection, ())


@pytest.mark.parametrize("key,value", [
    ("observation_fov_deg", 180), ("cbf_mode", "qp"), ("result_classification", "hard_safe"),
    ("ray_preprocess_mode", "clamp"), ("footprint_radius_m", .55),
    ("min_effective_clearance_m", .01), ("nonzero_footprint_policy", "allow"),
])
def test_unsupported_policy_semantics_rejected(key, value):
    projection = build_policy_kwargs(resolved())
    with pytest.raises(ValueError, match=key):
        factory().policy_constructor_kwargs(dataclasses.replace(
            projection, values=dict(projection.materialize_values(), **{key: value})), DELTA)


@pytest.mark.parametrize("key,value", [
    ("ppo_auxiliary_state_mode", "mutable"), ("action_stages", ["executed_command"]),
])
def test_unsupported_ppo_semantics_rejected(key, value):
    projection = build_ppo_kwargs(resolved())
    with pytest.raises(ValueError, match=key):
        factory().ppo_constructor_kwargs(dataclasses.replace(
            projection, values=dict(projection.materialize_values(), **{key: value})), DELTA)


def test_accepted_paper_identity_stays_blocked():
    with pytest.raises(ValueError, match="paper_table_action_bounds"):
        resolve_run_config(registry_path=ROOT / "configs/parity_registry.yaml",
                           algorithm_profile="paper_v1", runtime_stack="isaac_gym_preview4",
                           implementation_delta=DELTA)


def test_factory_rejects_legacy_ignored_value_loss_override():
    # Existing PPO's main value loss uses 1.0; this boundary cannot promise another weight.
    with pytest.raises(ValueError, match="value_loss_coef"):
        factory().ppo_constructor_kwargs(build_ppo_kwargs(resolved()), DELTA, value_loss_coef=.4)


@pytest.mark.parametrize("key,value", [
    ("alpha_min", -1), ("intervention_coefficient", -1), ("alpha_penalty_coefficient", float("nan")),
    ("actor_smoothness_coefficient", -1), ("critic_smoothness_coefficient", float("inf")),
    ("action_range_low", [0, 0]), ("action_range_high", [-10, -10, -10]),
])
def test_ppo_constructor_rejects_invalid_loss_configuration(key, value):
    actor = DifferentiableSafeActorCritic(**SMALL)
    with pytest.raises(ValueError):
        PPO(actor, **{key: value})


@pytest.mark.parametrize("term", ["intervention_coefficient", "alpha_penalty_coefficient",
                                "actor_smoothness_coefficient", "critic_smoothness_coefficient"])
def test_real_update_applies_effective_coefficient_once_and_zero_disables(term):
    """Compare actual optimizer gradients to independently weighted auxiliary losses."""
    config = resolved()
    projection = build_ppo_kwargs(config)
    base_values = projection.materialize_values()
    coeffs = ["intervention_coefficient", "alpha_penalty_coefficient",
              "actor_smoothness_coefficient", "critic_smoothness_coefficient"]
    base_values.update({key: 0. for key in coeffs})
    base_values.update(alpha_min=2., action_range_low=[-.01]*3, action_range_high=[.01]*3)
    gradients = []
    expected = None
    for coefficient in (0., .37):
        torch.manual_seed(77)
        actor = factory().build_actor_critic(config, **SMALL)
        obs = torch.linspace(-.2, .2, 550).repeat(2, 1)
        next_obs = obs + .13
        # Positive close rays activate the intervention gradient.
        obs[:, -43:-2] = -3.
        next_obs[:, -43:-2] = -2.7
        values = dict(base_values, **{term: coefficient})
        ppo = PPO(actor, **factory().ppo_constructor_kwargs(
            dataclasses.replace(projection, values=values), DELTA,
            learning_rate=0., max_grad_norm=1e9, desired_kl=None))
        assert ppo.alpha_min == 2
        assert tuple(ppo.action_range_high) == (.01, .01, .01)
        ppo.init_storage(2, 1, (550,), (3,))
        with torch.no_grad():
            ppo.act(obs, obs)
            ppo.process_env_step(next_obs, torch.tensor([.2, -.1]), torch.zeros(2, dtype=torch.bool),
                                 {"bad_masks": torch.zeros(2, dtype=torch.bool)})
            ppo.compute_returns(next_obs)
        real_smooth = ppo.compute_smoothness_loss
        captured = {}

        def smooth(current, following, *, orig_mu=None, orig_values=None):
            # Reuse exactly the production RNG draw without replacing the query or loss.
            state = torch.get_rng_state()
            result = real_smooth(current, following, orig_mu=orig_mu, orig_values=orig_values)
            captured["range"] = (orig_mu - orig_mu.clamp(-.01, .01)).square().sum(-1).mean().item()
            after = torch.get_rng_state()
            torch.set_rng_state(state)
            mix = (torch.rand(current.size(0), 1) - .5) * 2
            interp = current + mix * (following - current)
            torch.set_rng_state(after)
            if term == "intervention_coefficient":
                independent = (actor.u_s - actor.u_bar).square().sum(-1).mean()
            elif term == "alpha_penalty_coefficient":
                independent = F.relu(2. - actor.alpha).square().mean()
            elif term == "actor_smoothness_coefficient":
                independent = F.mse_loss(actor.action_mean_for(interp), orig_mu)
            else:
                independent = F.mse_loss(actor.evaluate(interp), orig_values)
            params = list(actor.parameters())
            grads = torch.autograd.grad(independent, params, retain_graph=True, allow_unused=True)
            captured["expected"] = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                              for p, g in zip(params, grads)])
            return result

        ppo.compute_smoothness_loss = smooth
        torch.manual_seed(321)
        metrics = ppo.update()
        # Changing the constructor bounds must change the executed range loss.
        if term in ("intervention_coefficient", "alpha_penalty_coefficient"):
            assert captured["range"] > 0
            assert metrics[2] == pytest.approx(captured["range"])
        gradients.append(torch.cat([p.grad.flatten() for p in actor.parameters()]))
        expected = captured["expected"]
    assert expected.abs().sum() > 0
    torch.testing.assert_close(gradients[1] - gradients[0], .37 * expected, atol=2e-6, rtol=2e-4)


def test_paper_loss_values_are_diagnostics_with_explicit_floor():
    projection = build_ppo_kwargs(resolved())
    values = projection.materialize_values()
    values.update(alpha_min=.1, alpha_penalty_coefficient=.1, intervention_coefficient=.1)
    ppo = PPO(DifferentiableSafeActorCritic(**SMALL), **factory().ppo_constructor_kwargs(
        dataclasses.replace(projection, values=values), DELTA))
    alpha = torch.tensor([[.02], [.2]], requires_grad=True)
    weighted = ppo.alpha_penalty_coefficient * ppo.compute_alpha_loss(alpha)
    # .1 * mean([(.1-.02)^2, 0]) = .00032.
    torch.testing.assert_close(weighted, torch.tensor(.00032))
    weighted.backward()
    torch.testing.assert_close(alpha.grad, torch.tensor([[-.008], [0.]]))
