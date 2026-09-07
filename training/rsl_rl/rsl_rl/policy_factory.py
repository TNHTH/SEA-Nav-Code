"""Small simulator-free boundary from validated profiles to real constructors.

Resolved entry points verify Task 2 integrity. Projection converters are also
usable for explicitly labelled CPU diagnostics; they do not confer run identity.
Runtime startup must still apply the environment projection (Task 6).
"""
from rsl_rl.experiment_config import build_policy_kwargs, build_ppo_kwargs
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.algorithms.ppo import PPO


_POLICY_MAP = {
    "num_rays": "num_rays", "history_frames": "his_len",
    "observation_fov_deg": "observation_fov_deg", "cbf_fov_deg": "cbf_fov_deg",
    "epsilon_d": "cbf_damping_factor", "kappa": "cbf_kappa",
    "safe_radius_m": "cbf_safe_radius", "safety_margin_m": "cbf_safety_margin",
}
_POLICY_FIXED = {
    "cbf_mode": "paper_damped", "result_classification": "differentiable_safety_bias",
    "footprint_radius_m": 0.0, "ray_preprocess_mode": "positive_raw_rays",
    "min_effective_clearance_m": None, "nonzero_footprint_policy": "requires_named_ablation",
}
_PPO_FIELDS = {
    "intervention_coefficient", "alpha_penalty_coefficient", "alpha_min",
    "actor_smoothness_coefficient", "critic_smoothness_coefficient",
    "action_range_low", "action_range_high",
}
_PPO_FIXED = {
    "ppo_auxiliary_state_mode": "pure_action_mean_for_preserves_current_minibatch_state",
    "action_stages": ["distribution_mean", "policy_action", "clipped_policy_action", "executed_command"],
}
_ACTOR_OPTIONS = {
    "num_actions", "actor_hidden_dims", "critic_hidden_dims", "encoder_hidden_dims",
    "activation", "init_noise_std", "num_props",
}
_PPO_OPTIONS = {
    "num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam",
    "value_loss_coef", "entropy_coef", "learning_rate", "penalty_lr", "max_grad_norm",
    "use_clipped_value_loss", "schedule", "desired_kl", "device",
}


def _values(projection, consumer, fields, fixed, implementation_delta):
    if projection.consumer != consumer:
        raise ValueError("projection consumer must be " + consumer)
    required = set(projection.required_activation_deltas)
    if consumer == "ppo_constructor":
        required.add("ppo_state_identity_repair")
    if not required.issubset(set(implementation_delta)):
        raise ValueError("missing required activation deltas: " + repr(sorted(required - set(implementation_delta))))
    values = projection.materialize_values()
    if set(values) != set(fields) | set(fixed):
        raise ValueError("projection fields must exactly match " + consumer)
    for key, expected in fixed.items():
        if values[key] != expected:
            raise ValueError("unsupported " + key + ": " + repr(values[key]))
    return values


def _merge(kwargs, overrides, allowed):
    for key, value in overrides.items():
        if key in kwargs:
            if value != kwargs[key]:
                raise ValueError("constructor override conflicts with projection: " + key)
        elif key not in allowed:
            raise ValueError("unknown constructor override: " + key)
        else:
            kwargs[key] = value
    return kwargs


def policy_constructor_kwargs(projection, implementation_delta=(), **overrides):
    values = _values(projection, "policy_factory", _POLICY_MAP, _POLICY_FIXED, implementation_delta)
    # Observation geometry is externally supplied. This actor only supports the
    # 240-degree sensor contract; never mistake it for the independent CBF FOV.
    if values["observation_fov_deg"] != 240.0:
        raise ValueError("unsupported observation_fov_deg")
    kwargs = {target: values[source] for source, target in _POLICY_MAP.items()}
    return _merge(kwargs, overrides, _ACTOR_OPTIONS)


def ppo_constructor_kwargs(projection, implementation_delta=(), **overrides):
    values = _values(projection, "ppo_constructor", _PPO_FIELDS, _PPO_FIXED, implementation_delta)
    # The inherited main value loss is fixed at 1.0; do not advertise an ignored
    # override as applied scientific configuration through this new boundary.
    if overrides.get("value_loss_coef", 1.0) != 1.0:
        raise ValueError("unsupported value_loss_coef: inherited PPO applies 1.0")
    return _merge({key: values[key] for key in _PPO_FIELDS}, overrides, _PPO_OPTIONS)


def build_actor_critic(config, **overrides):
    return DifferentiableSafeActorCritic(**policy_constructor_kwargs(
        build_policy_kwargs(config), config.identity.implementation_delta, **overrides))


def build_ppo(config, actor_critic, **overrides):
    return PPO(actor_critic, **ppo_constructor_kwargs(
        build_ppo_kwargs(config), config.identity.implementation_delta, **overrides))
