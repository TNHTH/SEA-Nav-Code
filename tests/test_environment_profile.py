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

def cpu_runner_env():
    import torch
    env=NS(num_obs=550,rays=torch.ones(2,41),num_nav_actions=3,num_props=12,
           cfg=NS(env=NS(his_len=10)),num_envs=2,device='cpu',reset_calls=0)
    def reset():
        env.reset_calls+=1
        return torch.zeros(2,550),None
    env.reset=reset
    return env

def execute_actual_runner_caller(name,env):
    """Execute real config-to-runner source statements, never application setup."""
    import ast
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.environment_profile import runtime_constructor_settings
    if name=='gym':
        # Real inherited PPO configuration classes; no legged_gym package import.
        base=ROOT/'training/legged_gym/legged_gym/envs/base'
        namespace={'inspect':__import__('inspect')}
        for path,clsname in ((base/'base_config.py','BaseConfig'),
                (base/'legged_robot_config.py','LeggedRobotCfgPPO'),
                (ROOT/'training/legged_gym/legged_gym/envs/go2/go2_pos_config.py','Go2PosRoughCfgPPO')):
            node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==clsname)
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        path=ROOT/'training/legged_gym/legged_gym/utils/task_registry.py'
        method=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='make_alg_runner')
        start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.If)
                   and ast.unparse(n.test)=='resolved_config is None')
        nodes=method.body[start:-1]  # return is outside a function in this CPU slice.
        helpers=base.parents[1]/'utils/helpers.py'
        helper=next(n for n in ast.parse(helpers.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='class_to_dict')
        exec(compile(ast.Module(body=[helper],type_ignores=[]),str(helpers),'exec'),namespace)
        namespace.update(env=env,train_cfg=namespace['Go2PosRoughCfgPPO'](),resolved_config=resolved(),
                         apply_algorithm_profile=apply_algorithm_profile,OnPolicyRunner=OnPolicyRunner,
                         args=NS(rl_device='cpu',wandb=False),log_dir=None)
        from rsl_rl import environment_profile
        if hasattr(environment_profile,'runner_config_for_environment'):
            namespace['runner_config_for_environment']=environment_profile.runner_config_for_environment
    else:
        path=ROOT/'sea_nav_current_isaaclab_full_method'/name
        method=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='train_cfg' for t in n.targets))
        stop=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='runner' for t in n.targets))
        nodes=method.body[start:stop+1]
        config=resolved('isaaclab_adapter')
        args=NS(rollout_steps=4,init_std=.8,cbf_fov_deg=180.,ppo_learning_rate=.0003,
                ppo_entropy_coef=.007,ppo_schedule='fixed',ppo_num_learning_epochs=1,ppo_num_mini_batches=1)
        namespace=dict(adapter_env=env,request=NS(resolved_config=config,
            environment={'constructor_settings':runtime_constructor_settings(config,vars(args))}),
            args=args,SimpleNamespace=NS,log_dir=None,OnPolicyRunner=OnPolicyRunner)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['runner'],namespace.get('runner_cfg',namespace.get('train_cfg_dict'))

@pytest.mark.parametrize('name',['gym','train_full_method_ppo.py','train_full_method_acsi_replay_ppo.py'])
def test_actual_runner_callers_apply_shapes_once(name):
    env=cpu_runner_env()
    runner,cfg=execute_actual_runner_caller(name,env)
    assert runner.env is env and env.reset_calls==1
    assert runner.alg.actor_critic.num_rays==41 and runner.alg.actor_critic.his_len==10
    assert cfg['applied_shapes']==dict(num_actions=3,num_props=12,num_rays=41,his_len=10,num_obs=550)
    assert not set(cfg['policy']) & {'num_actions','num_props','num_rays','his_len'}

@pytest.mark.parametrize('field,value',[('num_obs',549),('num_props',11),('num_nav_actions',4),('history',9),('rays',40)])
def test_runner_shape_mismatch_rejected_before_real_constructor(field,value):
    import torch
    env=cpu_runner_env()
    if field=='history': env.cfg.env.his_len=value
    elif field=='rays': env.rays=torch.ones(2,value)
    else: setattr(env,field,value)
    with pytest.raises(ValueError,match='shape'):
        execute_actual_runner_caller('train_full_method_ppo.py',env)
    assert env.reset_calls==0

@pytest.mark.parametrize('key,value',[('num_actions',4),('num_props',11),('num_rays',40),('his_len',9)])
def test_runner_explicit_policy_shape_conflict_is_not_dropped(key,value):
    import copy
    from rsl_rl.environment_profile import runner_config_for_environment
    config={'policy':{key:value},'algorithm':{}}
    original=copy.deepcopy(config)
    env=cpu_runner_env()
    with pytest.raises(ValueError,match='shape|conflict'):
        runner_config_for_environment(config,resolved('isaaclab_adapter'),env)
    assert config==original and env.reset_calls==0

def test_ordinary_reset_prefix_consumes_completed_events_once_not_initial_or_operator():
    import ast
    import torch
    from rsl_rl.replay import CurriculumConfig,update_goal_level
    path=ROOT/'sea_nav_current_isaaclab_full_method/train_full_method_ppo.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
    cutoff=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='reset')
    prefix=compile(ast.Module(body=method.body[:cutoff],type_ignores=[]),str(path),'exec')
    state=NS(torch=torch,reset_count=0,_pending_curriculum_distance=None,level=torch.tensor([1.]),events=[])
    def update(distance):
        state.events.append(distance.clone())
        state.level,_,_=update_goal_level(state.level,distance,torch.tensor([True]),CurriculumConfig())
    state._update_curriculum=update
    exec(prefix,{'self':state})
    state.reset_count=1
    exec(prefix,{'self':state})  # runner's second reset before any transition
    assert state.level.tolist()==[1.] and not state.events
    state._terminal_distance=torch.tensor([.4])  # unfinished transition/operator reset
    exec(prefix,{'self':state})
    assert not state.events
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    terminal=next(n for n in step.body if isinstance(n,ast.If) and ast.unparse(n.test)=='done.any()')
    assign=next(n for n in terminal.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Attribute) and t.attr=='_pending_curriculum_distance' for t in n.targets))
    event=compile(ast.Module(body=[assign],type_ignores=[]),str(path),'exec')
    for distance,expected in ((.4,2.),(2.1,1.)):
        state._terminal_distance=torch.tensor([distance])
        exec(event,{'self':state})
        exec(prefix,{'self':state})
        exec(prefix,{'self':state})  # repeated explicit reset cannot consume twice
        assert state.level.tolist()==[expected]
        assert state._pending_curriculum_distance is None
    assert len(state.events)==2
