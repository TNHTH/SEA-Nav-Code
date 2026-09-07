from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from rsl_rl.experiment_config import resolve_run_config
from rsl_rl.environment_profile import environment_settings, materialize_config, apply_gym_environment, apply_algorithm_profile
ROOT=Path(__file__).resolve().parents[1]
def resolved(stack="isaac_gym_preview4",deltas=("ppo_state_identity_repair","replay_reset_reconstruction_v1")):
    return resolve_run_config(registry_path=ROOT/"configs/parity_registry.yaml",algorithm_profile="upstream_fbce672c",runtime_stack=stack,implementation_delta=deltas)

def test_nested_class_configs_are_not_shared():
    class C:
        class env:
            num_envs=2
            values=[1,2]
    a=materialize_config(C); b=materialize_config(C)
    a.env.values.append(3)
    assert b.env.values==C.env.values==[1,2]

def test_actual_gym_reward_and_constructor_application():
    c=resolved()
    cfg=NS(env=NS(episode_length_s=40),sim=NS(dt=.005),control=NS(decimation=4),commands=NS(),
           rewards=NS(scales=NS()),replay=NS(enable_collision_replay=False,num_collision_states=180,undo_steps=[100,150]),
           terrain=NS(num_rows=10))
    cfg,receipt=apply_gym_environment(cfg,c,timeout_seconds=40)
    assert cfg.env.episode_length_s==receipt["actual_horizon_s"]==40
    assert cfg.env.stay_time==cfg.env.goal_reached_time==150
    assert cfg.rewards.scales.velo_dir==4 and receipt["effective_weights"]["velocity"]==.08
    assert cfg.resolved_run_config is c
    train={"policy":{"num_props":12,"num_actions":3},"algorithm":{"value_loss_coef":1}}
    final=apply_algorithm_profile(train,c)
    assert final["policy"]["cbf_fov_deg"]==180
    assert final["algorithm"]["alpha_penalty_coefficient"]==1
    assert "cbf_fov_deg" not in train["policy"]

def test_effective_replay_bridge_contract_applies_even_disabled():
    with pytest.raises(ValueError,match="replay_reset_reconstruction"):
        environment_settings(resolved(deltas=("ppo_state_identity_repair",)),replay_enabled=False)
    s=environment_settings(resolved(),replay_enabled=False)
    assert s["replay"]["enabled"] is False
    assert s["replay"]["stored_level_bounds"]==[0.,10.]
    assert s["perception"]["acquisition_period_s"]==.02
    assert s["actuator_delay_s"]==0

def test_conflicting_dt_and_constructor_rejected():
    with pytest.raises(ValueError,match="policy_dt"):
        environment_settings(resolved(),policy_dt_s=.01)
    with pytest.raises(ValueError,match="conflicts"):
        apply_algorithm_profile({"policy":{"cbf_fov_deg":240},"algorithm":{}},resolved())
    for horizon in (.001,.021):
        with pytest.raises(ValueError,match='policy ticks'):
            environment_settings(resolved(),timeout_seconds=horizon)

def test_actual_runtime_consumers_use_shared_application_and_no_layer_replacement():
    import ast
    for name in ('train_full_method_ppo.py','train_full_method_acsi_replay_ppo.py','full_method_runtime_smoke.py'):
        path=ROOT/'sea_nav_current_isaaclab_full_method'/name
        tree=ast.parse(path.read_text())
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        assert 'compute_navigation_reward_terms' in calls
        assert 'weight_reward_terms' in calls
        assert 'TimestampedPerception' in calls
        assert 'environment_settings' in calls
        assert ('build_actor_critic' if name.startswith('full') else 'apply_algorithm_profile') in calls
        assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='cbf_layer' for t in n.targets) for n in ast.walk(tree))
    gym=ast.parse((ROOT/'training/legged_gym/legged_gym/envs/base/legged_robot_pos.py').read_text())
    calls={n.func.id for n in ast.walk(gym) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert 'compute_navigation_reward_terms' in calls and 'TimestampedPerception' in calls

def test_ordinary_ppo_history_and_curriculum_are_actual_consumers():
    import importlib.util
    import torch
    from rsl_rl.replay import CurriculumConfig
    path=ROOT/'sea_nav_current_isaaclab_full_method/train_full_method_ppo.py'
    spec=importlib.util.spec_from_file_location('ordinary_entry',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    env=module.SeaNavOriginalSemanticsIsaacLabEnv.__new__(module.SeaNavOriginalSemanticsIsaacLabEnv)
    env.torch=torch
    env.pos_hist=torch.zeros(2,10,2)
    env.episode_length_buf=torch.tensor([9,10])
    env._update_position_history(torch.tensor([[1.,2.],[3.,4.]]))
    assert torch.count_nonzero(env.pos_hist[0])==0
    assert env.pos_hist[1,-1].tolist()==[3.,4.]
    assert torch.count_nonzero(env.pos_hist[1,:-1])==0
    env.source_goal_levels=torch.tensor([1.,1.])
    env.acsi_config=CurriculumConfig()
    env._update_curriculum(torch.tensor([.4,2.1]))
    assert env.source_goal_levels.tolist()==[2.,0.]

def test_missing_async_tensor_assert_is_rejected_before_environment(monkeypatch):
    import torch
    monkeypatch.delattr(torch,'_assert_async')
    with pytest.raises(RuntimeError,match='blocked'):
        environment_settings(resolved())

def test_adapter_actual_cpu_constructors_consume_preflight_options():
    import torch
    from rsl_rl.environment_profile import runtime_constructor_settings
    from rsl_rl.policy_factory import build_actor_critic,build_ppo
    config=resolved('isaaclab_adapter')
    settings=runtime_constructor_settings(config,{'ppo_learning_rate':.0003,'ppo_entropy_coef':.007,'init_std':.8})
    actor=build_actor_critic(config,num_actions=3,num_props=12,**settings['policy'])
    algorithm=build_ppo(config,actor,**settings['algorithm'])
    assert actor.cbf_layer.fov_deg==180.
    assert torch.allclose(actor.std,torch.full((3,),.8))
    assert algorithm.learning_rate==.0003 and algorithm.entropy_coef==.007
    assert algorithm.intervention_coefficient==.1 and algorithm.alpha_penalty_coefficient==1.

def test_materialized_instance_keeps_contract_methods_but_detaches_nested_classes():
    class Contract:
        class child: values=[1]
        def check(self): return self.child.values
    source=Contract(); fresh=materialize_config(source)
    fresh.child.values.append(2)
    assert fresh.check()==[1,2] and source.check()==[1]
