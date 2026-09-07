"""Pure profile-to-environment application before proprietary construction."""
import copy
import math
from dataclasses import is_dataclass
from types import SimpleNamespace
from rsl_rl.experiment_config import build_env_profile_values, build_policy_kwargs, build_ppo_kwargs
from rsl_rl.policy_factory import policy_constructor_kwargs, ppo_constructor_kwargs
from rsl_rl.replay import replay_configs_from_resolved
from rsl_rl.navigation_reward import GYM_REWARD_NAMES
from rsl_rl.perception_delay import PerceptionDelayConfig, require_perception_capability

def materialize_config(value):
    """Deep-copy nested class attributes (copy.deepcopy(class) alone aliases them)."""
    if isinstance(value, type):
        return SimpleNamespace(**{key: materialize_config(getattr(value,key)) for key in dir(value)
                                  if not key.startswith("_") and
                                  (isinstance(getattr(value,key),type) or not callable(getattr(value,key)))})
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**{k:materialize_config(v) for k,v in vars(value).items()})
    if isinstance(value,dict):
        return {k:materialize_config(v) for k,v in value.items()}
    if isinstance(value,list):
        return [materialize_config(v) for v in value]
    if is_dataclass(value) and value.__dataclass_params__.frozen:
        return value
    if hasattr(value,"__dict__"):
        result=copy.copy(value)
        for key in dir(value):
            if not key.startswith("_"):
                item=getattr(value,key)
                if isinstance(item,type) or not callable(item):
                    setattr(result,key,materialize_config(item))
        return result
    return copy.deepcopy(value)

def environment_settings(resolved, *, policy_dt_s=.02, timeout_seconds=None, replay_enabled=False,
                         capacity=180, undo=(100,150), max_level=10., command_filter_alpha=.5):
    values=build_env_profile_values(resolved).materialize_values()
    require_perception_capability()
    if resolved.identity.algorithm_profile != "upstream_fbce672c":
        raise ValueError("unsupported environment formula identity")
    if policy_dt_s != values["policy_dt_s"]:
        raise ValueError("policy_dt conflicts with resolved profile")
    if not math.isfinite(command_filter_alpha) or command_filter_alpha != .5:
        raise ValueError("unsupported command_filter_alpha; named ablation required")
    horizon=values["training_episode_s"] if timeout_seconds is None else timeout_seconds
    if isinstance(horizon,bool) or not math.isfinite(horizon) or horizon<=0:
        raise ValueError("timeout_seconds must be finite and positive")
    horizon_ticks=round(horizon/policy_dt_s)
    if horizon_ticks<1 or not math.isclose(horizon_ticks*policy_dt_s,horizon,rel_tol=0,abs_tol=1e-8):
        raise ValueError("timeout_seconds must contain a positive integral number of policy ticks")
    if not values["integrate_over_policy_dt"] or values["reward_scale_unit"]!="raw_weight_times_policy_dt":
        raise ValueError("unsupported reward integration")
    if (values["history_indices"]!=[-3,-4] or values["perception_scheduler"]!="policy_tick_history_refresh"
            or values["output_mode"]!="sample_and_hold" or not values["delay_goal_with_rays"]
            or values["startup_policy"]!="current_sample_until_history_available"):
        raise ValueError("unsupported perception projection")
    perception=PerceptionDelayConfig(mode=values["delay_mode"],policy_dt_s=policy_dt_s,
        acquisition_period_s=values["acquisition_period_s"],refresh_period_s=values["output_refresh_period_s"])
    replay,acsi=replay_configs_from_resolved(resolved,max_level=max_level,capacity=capacity,undo=undo,enabled=replay_enabled)
    return dict(resolved_config_sha256=resolved.resolved_sha256, runtime_stack=resolved.identity.runtime_stack,
        projection_values=values, formula_mode=values["formula_mode"], raw_weights=values["raw_weights"],
        effective_weights={k:v*policy_dt_s for k,v in values["raw_weights"].items()},
        policy_dt_s=policy_dt_s, actual_horizon_s=float(horizon),horizon_purpose="training_or_local_diagnostic_not_formal_evaluation",
        perception=vars(perception), actuator_delay_s=0.,command_filter_alpha=command_filter_alpha,
        command_low=values["executed_command_low"],command_high=values["executed_command_high"],
        replay=dict(enabled=replay.enabled_during_training,capacity=capacity,undo=list(undo),
                    reset_policy=replay.reconstruction_policy,stored_level_bounds=[0.,float(acsi.max_level)],
                    bootstrap="noise_free_synthetic_step_0",runtime_capabilities="blocked_until_real_context_validation"),
        runtime_status="not_executed")

def apply_algorithm_profile(train_config,resolved):
    final=copy.deepcopy(train_config)
    delta=resolved.identity.implementation_delta
    # Legacy aliases/factory-owned values are checked, not silently discarded.
    policy=dict(final.get("policy",{}))
    policy.pop("class_name",None)
    final["policy"]=policy_constructor_kwargs(build_policy_kwargs(resolved),delta,**policy)
    final["algorithm"]=ppo_constructor_kwargs(build_ppo_kwargs(resolved),delta,**final.get("algorithm",{}))
    return final

def runtime_constructor_settings(resolved, arguments):
    """Exactly the options later consumed by adapter runner/actor constructors."""
    args=arguments
    policy=dict(actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],
                encoder_hidden_dims=[512,256,128],activation="elu",init_noise_std=args.get("init_std",1.5))
    algorithm=dict(num_learning_epochs=args.get("ppo_num_learning_epochs",1),
        num_mini_batches=args.get("ppo_num_mini_batches",1),clip_param=.2,gamma=.99,lam=.95,
        value_loss_coef=1.,entropy_coef=args.get("ppo_entropy_coef",.003),
        learning_rate=args.get("ppo_learning_rate",1e-4),penalty_lr=1e-3,max_grad_norm=1.,
        use_clipped_value_loss=True,schedule=args.get("ppo_schedule","fixed"),desired_kl=.01)
    return apply_algorithm_profile({"policy":policy,"algorithm":algorithm},resolved)

def reconcile_environment_receipt(actual,preflight):
    expected={k:v for k,v in preflight.items() if k not in ('constructor_settings','asset_prerequisites')}
    if actual != expected:
        differences=sorted(k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k))
        raise ValueError('actual environment differs from preflight: '+repr(differences))
    return actual

def apply_gym_environment(cfg,resolved,timeout_seconds=None):
    cfg=materialize_config(cfg)
    settings=environment_settings(resolved,policy_dt_s=cfg.sim.dt*cfg.control.decimation,
        timeout_seconds=timeout_seconds, replay_enabled=cfg.replay.enable_collision_replay,
        capacity=getattr(cfg.replay,"ring_buffer_steps",180),
        undo=tuple(getattr(cfg.replay,"undo_steps_range",(100,150))),max_level=cfg.terrain.num_rows)
    cfg.resolved_run_config=resolved
    cfg.environment_receipt=settings
    cfg.env.episode_length_s=settings["actual_horizon_s"]
    cfg.env.goal_reached_time=settings["projection_values"]["stay_ticks"]
    cfg.env.stay_time=settings["projection_values"]["stay_ticks"]
    cfg.rewards.only_positive_rewards=False
    for name in dir(cfg.rewards.scales):
        if not name.startswith("_") and name not in GYM_REWARD_NAMES.values() and getattr(cfg.rewards.scales,name)!=0:
            raise ValueError("unsupported active Gym reward term: "+name)
    for name,weight in settings["raw_weights"].items():
        setattr(cfg.rewards.scales,GYM_REWARD_NAMES[name],weight)
    cfg.rewards.formula_mode=settings["formula_mode"]
    cfg.commands.delay_time=settings["perception"]["refresh_period_s"]
    cfg.commands.alpha=settings["command_filter_alpha"]
    if not hasattr(cfg.commands,"ranges"):
        cfg.commands.ranges=SimpleNamespace()
    for i,key in enumerate(("limit_vx","limit_vy","limit_vyaw")):
        setattr(cfg.commands.ranges,key,[settings["command_low"][i],settings["command_high"][i]])
    return cfg,settings
