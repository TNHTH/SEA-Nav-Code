"""AST checks certify call-site ordering only, never simulator effects."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT=Path(__file__).resolve().parents[1]
GYM=ROOT/'training/legged_gym/legged_gym/envs/base'
ADAPTER=ROOT/'sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py'


def method(path,name):
    tree=ast.parse(path.read_text())
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    return matches[-1]


def calls(node):
    result=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Call):
            f=n.func
            result.append((n.lineno,f.attr if isinstance(f,ast.Attribute) else f.id if isinstance(f,ast.Name) else ''))
    return [name for _,name in sorted(result)]


def test_gym_terminal_capture_precedes_reset_and_base_epilogue_is_masked():
    names=calls(method(GYM/'legged_robot.py','post_physics_step'))
    assert names.index('compute_reward') < names.index('_capture_replay_boundary') < names.index('reset_idx')
    assert 'compute_observations_for' in names
    assert '_post_reset_epilogue' in calls(method(GYM/'legged_robot_pos.py','reset_idx')) or any(
        isinstance(n,ast.Attribute) and n.attr=='_post_reset_epilogue' for n in ast.walk(method(GYM/'legged_robot_pos.py','reset_idx')))
    assert 'execute_reset_transaction' in calls(method(GYM/'legged_robot_pos.py','reset_idx'))


def test_consumers_use_compact_transaction_and_shared_acsi():
    for path in (GYM/'legged_robot_pos.py',ADAPTER):
        tree=ast.parse(path.read_text()); names=calls(tree)
        assert 'select_terminal_replay' in names and 'update_goal_level' in names
        assert 'execute_reset_transaction' in names
        assert 'build_reset_frames' in names
    assert 'advance_history' in calls(method(ADAPTER,'step'))
    assert '_require_replay_runtime_contract' in calls(method(ADAPTER,'_capture_replay_record'))


def test_reconstruction_does_not_run_mutating_reward_termination_or_control():
    for path,name in ((GYM/'legged_robot_pos.py','_reconstruct_reset_rows'),(ADAPTER,'_reconstruct_reset_rows')):
        names=calls(method(path,name))
        assert not set(names)&{'check_termination','compute_reward','_compute_reward_done','_compute_actions','_post_physics_step_callback','_update_observation'}


def test_public_trainer_filter_reset_clears_both_mirrors_and_next_recurrence():
    # Execute only this real, simulator-independent class definition. No task
    # imports, fake simulator modules, parser or AppLauncher are executed.
    import sys
    sys.path.insert(0,str(ROOT/'sea_nav_current_isaaclab_full_method'))
    from adapters.command_delay import CommandDelayFilter,CommandDelayConfig
    from adapters.cbf_shield import clip_body_command
    tree=ast.parse(ADAPTER.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrainerCommandFilter')
    namespace={}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(ADAPTER),'exec'),namespace)
    queue=CommandDelayFilter(CommandDelayConfig(alpha=.5),num_envs=3)
    wrapper=namespace['TrainerCommandFilter']('source_alpha_only',queue,.5,3,'cpu',clip_body_command)
    wrapper.step(torch.ones(3,3))
    wrapper.reset_state(torch.tensor([1]))
    assert wrapper.filtered.tolist()==queue.filtered.tolist()==[[.5,.5,.5],[0,0,0],[.5,.5,.5]]
    output,_=wrapper.step(torch.ones(3,3))
    assert output.tolist()==[[.75,.75,.75],[.5,.5,.5],[.75,.75,.75]]


def test_gym_bad_masks_snapshot_is_not_mutated_by_new_episode_bootstrap():
    node=method(GYM/'legged_robot_pos.py','check_termination')
    assignments=[n for n in ast.walk(node) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant)
                         and t.slice.value=='bad_masks' for t in n.targets)]
    assert len(assignments)==1
    value=assignments[0].value
    assert isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr=='clone'

def test_public_trainer_filter_saturated_state_mirrors_survive_masked_reset():
    import sys
    sys.path.insert(0,str(ROOT/'sea_nav_current_isaaclab_full_method'))
    from adapters.command_delay import CommandDelayFilter,CommandDelayConfig
    from adapters.cbf_shield import clip_body_command
    tree=ast.parse(ADAPTER.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrainerCommandFilter')
    namespace={}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(ADAPTER),'exec'),namespace)
    queue=CommandDelayFilter(CommandDelayConfig(alpha=.5),num_envs=2)
    wrapper=namespace['TrainerCommandFilter']('source_alpha_only',queue,.5,2,'cpu',clip_body_command)
    for _ in range(2): wrapper.step(torch.full((2,3),3.))
    assert torch.equal(wrapper.filtered,queue.filtered) and wrapper.filtered.tolist()==[[2.25]*3]*2
    wrapper.reset_state(torch.tensor([1]))
    out,debug=wrapper.step(torch.zeros(2,3))
    assert out.tolist()==[[1.125,1.,1.],[0.,0.,0.]]
    assert torch.equal(wrapper.filtered,queue.filtered)
    assert torch.equal(debug['executed_command'],out)


def test_adapter_normal_fallback_hooks_surround_writes_before_reconstruction():
    names = calls(method(ADAPTER, '_normal_reset_rows'))
    assert names.index('prepare_rows') < names.index('reset')
    assert names.index('reset') < names.index('_place_robot_at_start') < names.index('refresh_rows')
    replay_names = calls(method(ADAPTER, '_write_replay_selection'))
    assert replay_names.index('prepare_rows') < replay_names.index('_restore_replay_sample') < replay_names.index('refresh_rows')
    reset = method(ADAPTER, 'reset')
    transaction = next(n for n in ast.walk(reset) if isinstance(n,ast.Call)
                       and isinstance(n.func,ast.Name) and n.func.id=='execute_reset_transaction')
    assert [arg.attr for arg in transaction.args[4:8]] == [
        '_normal_reset_rows', '_write_replay_selection', '_reconstruct_reset_rows', '_post_reset_epilogue']
    assert 'finish_rows' in calls(method(ADAPTER, '_post_reset_epilogue'))


@pytest.mark.parametrize('ignore_carrier', [False, True])
def test_selected_carrier_timeout_excludes_replay_without_changing_play_mode(ignore_carrier):
    from rsl_rl.replay import CurriculumConfig, select_terminal_replay
    # Execute the actual pure terminal-source selection block, stopping before
    # capture/observations/reset. No carrier, task, or simulator is fabricated.
    statements = method(ADAPTER,'step').body
    start = next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='carrier_done' for t in n.targets))
    stop = next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Attribute) and t.attr=='reset_buf' for t in n.targets))
    state=SimpleNamespace(device='cpu',source_play_eval_terminal_semantics_enabled=ignore_carrier,
                          _terminal_timeout=torch.tensor([False,False,True,False]))
    scope={'self':state,'semantic_done':torch.tensor([False,True,True,False]),
           'carrier_terminated':torch.tensor([False,False,False,True]),
           'carrier_truncated':torch.tensor([True,True,False,False])}
    exec(compile(ast.Module(body=statements[start:stop],type_ignores=[]),str(ADAPTER),'exec'),scope)
    assert scope['done'].tolist()==([False,True,True,False] if ignore_carrier else [True,True,True,True])
    assert state._terminal_timeout.tolist()==([False,False,True,False] if ignore_carrier else [True,True,True,False])
    wants=select_terminal_replay(torch.ones(4,dtype=torch.bool),torch.zeros(4,dtype=torch.bool),
        state._terminal_timeout,torch.zeros(4,dtype=torch.bool),torch.full((4,),.1),CurriculumConfig())
    assert wants.tolist()==([True,True,False,True] if ignore_carrier else [False,False,False,True])


@pytest.mark.parametrize('collision_mask,expected', [([False,False,False],0),([True,False,True],2)])
@pytest.mark.parametrize('boundary', [0,1], ids=['smoke_result','training_result'])
def test_result_boundaries_json_encode_device_collision_counter(collision_mask,expected,boundary):
    capture=method(ADAPTER,'_capture_replay_record')
    accumulation=next(n for n in ast.walk(capture) if isinstance(n,ast.AugAssign)
                      and isinstance(n.target,ast.Attribute) and n.target.attr=='replay_collision_record_count')
    class ScalarProbe(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.extractions=[]
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if str(func)=='aten._local_scalar_dense.default': self.extractions.append(str(func))
            return func(*args,**(kwargs or {}))
    state=SimpleNamespace(replay_collision_record_count=0)
    scope={'self':state,'active':torch.tensor(collision_mask)}
    probe=ScalarProbe()
    with probe:
        exec(compile(ast.Module(body=[accumulation],type_ignores=[]),str(ADAPTER),'exec'),scope)
    assert isinstance(state.replay_collision_record_count,torch.Tensor)
    assert probe.extractions==[]
    # Evaluate the actual value expression in each emitted result mapping.
    values=[]
    for node in ast.walk(method(ADAPTER,'main')):
        if isinstance(node,ast.Dict):
            values.extend((node.lineno,value) for key,value in zip(node.keys,node.values)
                          if isinstance(key,ast.Constant) and key.value=='replay_collision_record_count')
    assert len(values)==2
    value=sorted(values,key=lambda pair:pair[0])[boundary][1]
    result={'replay_collision_record_count':eval(compile(ast.Expression(body=value),str(ADAPTER),'eval'),
                                                {'adapter_env':state})}
    encoded=json.dumps(result)
    assert json.loads(encoded)=={'replay_collision_record_count':expected}
    assert type(result['replay_collision_record_count']) is int
