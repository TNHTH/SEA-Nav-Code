"""AST checks certify call-site ordering only, never simulator effects."""
import ast
from pathlib import Path
import torch

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
