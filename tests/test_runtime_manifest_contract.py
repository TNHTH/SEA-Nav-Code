import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"sea_nav_current_isaaclab_full_method"))
from adapters.trace_logger import JsonlTraceLogger, count_physical_jsonl_rows
from adapters.manifest import bind_runtime_result
from test_environment_profile import resolved
from rsl_rl.environment_profile import environment_settings
from rsl_rl.runtime_preflight import RuntimePaths

def row():
    return dict(step=1,env_id=0,distribution_mean=[1,2,3],policy_action=[2,3,4],
                clipped_policy_action=[2,3,3],executed_command=[1,1,1])

def test_trace_enforces_four_stages_and_physical_count_after_close(tmp_path):
    path=tmp_path/"trace.jsonl"
    trace=JsonlTraceLogger(path)
    with pytest.raises(ValueError,match="missing"):
        trace.write({"step":1,"u_safe":[0,0,0]})
    trace.write(row()); assert trace.row_count==1
    with pytest.raises(ValueError,match="closed"):
        trace.verified_row_count()
    trace.close()
    assert trace.verified_row_count()==count_physical_jsonl_rows(path)==1
    path.write_text(path.read_text()+json.dumps(row())+"\n")
    with pytest.raises(ValueError,match="count"):
        trace.verified_row_count()

def test_manifest_binds_hash_effective_receipt_and_rejects_stale_pass(tmp_path):
    c=resolved("isaaclab_adapter"); env=environment_settings(c)
    paths=RuntimePaths(tmp_path/"launcher",tmp_path,tmp_path/"assets",None)
    trace=JsonlTraceLogger(tmp_path/"trace.jsonl"); trace.write(row()); trace.close()
    result=bind_runtime_result({"status":"blocked","trace_rows":1},c,paths,env,trace)
    assert result["resolved_config"]["resolved_sha256"]==c.resolved_sha256
    assert result["effective_environment"]==env
    assert result["trace_rows"]==1 and result["runtime_verified"] is False
    for bad in ({"status":"passed","trace_rows":1},{"status":"blocked","trace_rows":2},
                {"status":"blocked","runtime_stack":"isaac_gym_preview4"}):
        with pytest.raises(ValueError):
            bind_runtime_result(bad,c,paths,env,trace)
    with pytest.raises(ValueError,match="hash"):
        bind_runtime_result({"status":"blocked"},c,paths,dict(env,resolved_config_sha256="stale"),trace)
    corrupt=json.loads(json.dumps(env)); corrupt['raw_weights']['velocity']=999999
    with pytest.raises(ValueError,match='receipt'):
        bind_runtime_result({"status":"blocked"},c,paths,corrupt,trace)

def test_close_failure_closes_all_and_preserves_original_failure():
    from adapters.manifest import close_runtime_resources
    calls=[]
    class Resource:
        def __init__(self,name): self.name=name
        def close(self):
            calls.append(self.name)
            if self.name=='trace': raise RuntimeError('trace flush failed')
    errors=close_runtime_resources((('trace',Resource('trace')),('carrier',Resource('carrier')),('app',Resource('app'))))
    assert calls==['trace','carrier','app']
    assert errors==[{'resource':'trace','error':'trace flush failed'}]

def test_failed_physical_trace_verification_keeps_claim_without_asserting_count(tmp_path):
    class BrokenTrace:
        path=tmp_path/'trace.jsonl'
        def verified_row_count(self): raise OSError('trace flush failed')
    config=resolved('isaaclab_adapter')
    paths=RuntimePaths(tmp_path/'launcher',tmp_path,tmp_path/'assets')
    result=bind_runtime_result({'status':'failed','reason':'original failure','trace_rows':1},
                               config,paths,environment_settings(config),BrokenTrace())
    assert result['status']=='failed' and result['reason']=='original failure'
    assert result['trace_rows'] is None and result['trace_verified'] is False
    assert result['claimed_trace_rows']==1 and result['trace_validation_error']=='trace flush failed'

def test_trace_captures_collection_state_before_reset_without_actor_forward(tmp_path):
    import torch
    from adapters.trace_logger import capture_policy_stages,write_transition
    from types import SimpleNamespace as NS
    actor=NS(action_mean=torch.tensor([[1.,2.,3.]]))
    action=torch.tensor([[2.,3.,4.]])
    stages=capture_policy_stages(actor,action)
    actor.action_mean.zero_(); action.zero_()
    stages["clipped_policy_action"]=torch.tensor([[2.,3.,3.]])
    stages["executed_command"]=torch.tensor([[.5,1.,1.]])
    trace=JsonlTraceLogger(tmp_path/"actual.jsonl")
    perception=NS(held_time=torch.tensor([.04]),held_latency=torch.tensor([.06]),synthetic=torch.tensor([False]))
    write_transition(trace,4,stages,torch.tensor([.38]),torch.tensor([True]),perception,torch.tensor([.12]))
    trace.close()
    item=json.loads(trace.path.read_text())
    assert item["distribution_mean"]==[1,2,3] and item["policy_action"]==[2,3,4]
    assert item["executed_command"]==[.5,1,1] and trace.verified_row_count()==1
    assert item["actual_sample_age"]==pytest.approx(.08)
    assert item["sampled_latency"]==pytest.approx(.06)

def test_actual_ordinary_step_snapshots_post_clip_command_without_simulation():
    import ast
    import torch
    from types import SimpleNamespace as NS
    from adapters.trace_logger import capture_policy_stages
    path=Path(__file__).resolve().parents[1]/'sea_nav_current_isaaclab_full_method/train_full_method_ppo.py'
    tree=ast.parse(path.read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
    stop=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='base_ang_vel' for t in n.targets))
    actor=NS(action_mean=torch.tensor([[8.,-8.,8.]]))
    state=NS(torch=torch,trace_logger=object(),trace_policy=lambda:actor,slr_command=torch.zeros(1,3),
             episode_length_buf=torch.tensor([4]),step_dt=.02,
             nav_filtered_state=torch.zeros(1,3),
             command_low=torch.tensor([-.5,-1.,-1.]),command_high=torch.tensor([2.,1.,1.]))
    scope={'self':state,'actions':actor.action_mean.clone()}
    exec(compile(ast.Module(body=method.body[:stop],type_ignores=[]),str(path),'exec'),scope)
    assert scope['trace_stages']['policy_action'].tolist()==[[8.,-8.,8.]]
    assert scope['trace_stages']['clipped_policy_action'].tolist()==[[3.,-3.,3.]]
    assert scope['trace_stages']['executed_command'].tolist()==[[1.5,-1.,1.]]
    state.slr_command.zero_()
    assert scope['trace_stages']['executed_command'].tolist()==[[1.5,-1.,1.]]
    exec(compile(ast.Module(body=method.body[:stop],type_ignores=[]),str(path),'exec'),scope)
    scope['actions']=torch.zeros(1,3)
    exec(compile(ast.Module(body=method.body[:stop],type_ignores=[]),str(path),'exec'),scope)
    assert scope['trace_stages']['executed_command'].tolist()==[[1.125,-1.,1.]]

def test_smoke_first_evidence_record_has_no_legacy_replay_dependency(tmp_path):
    import ast
    import torch
    from types import SimpleNamespace as NS
    from sea_nav_current_isaaclab_full_method import full_method_runtime_smoke as module
    tree=ast.parse(Path(module.__file__).read_text())
    assert not any(isinstance(n,ast.Name) and n.id in {'CollisionReplayBuffer','replay_buffer'} for n in ast.walk(tree))
    record=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='trace_record' for t in n.targets))
    required={'env_id','step','distribution_mean','policy_action','clipped_policy_action','executed_command',
              'observation_timestamp','sample_timestamp','sampled_latency','actual_sample_age','synthetic_bootstrap'}
    pairs=[(k,v) for k,v in zip(record.keys,record.values) if isinstance(k,ast.Constant) and k.value in required]
    assert {k.value for k,_ in pairs}==required
    evidence=ast.Dict(keys=[k for k,_ in pairs],values=[v for _,v in pairs])
    ast.copy_location(evidence,record)
    scope=dict(step=0,tensor_list=module.tensor_list,pre_delay_u_safe=torch.ones(1,3),
               delay_debug={'clipped_new_command':torch.ones(1,3)},u_applied=torch.ones(1,3),
               decision_time=torch.zeros(1),perception=NS(held_time=torch.zeros(1),
               held_latency=torch.zeros(1),synthetic=torch.ones(1,dtype=torch.bool)))
    scope['trace_record']=eval(compile(ast.Expression(evidence),str(module.__file__),'eval'),scope)
    trace=JsonlTraceLogger(tmp_path/'first.jsonl'); scope['trace_logger']=trace
    write=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
               and ast.unparse(n.value.func)=='trace_logger.write')
    exec(compile(ast.Module(body=[write],type_ignores=[]),str(module.__file__),'exec'),scope)
    trace.close()
    assert trace.verified_row_count()==1

@pytest.mark.parametrize('script',['train_full_method_ppo.py','train_full_method_acsi_replay_ppo.py'])
@pytest.mark.parametrize('num_envs',[2,2048])
@pytest.mark.parametrize('enabled',[False,True])
def test_actual_training_step_trace_branch_has_zero_disabled_materialization(script,num_envs,enabled,tmp_path):
    import ast
    import torch
    from collections import Counter
    from types import SimpleNamespace as NS
    from torch.utils._python_dispatch import TorchDispatchMode
    from adapters.trace_logger import capture_policy_stages,write_transition
    from test_runtime_cli_contract import cli,load_trainer,attach_actual_training_trace
    module=load_trainer(script)
    path=tmp_path/'output/selected/trace.jsonl'
    request=module.preflight(cli(tmp_path)+(['--trace',str(path)] if enabled else []))
    mean=torch.ones(num_envs,3)
    actor=NS(action_mean=mean)
    state,trace=attach_actual_training_trace(module,request.arguments,actor)
    if not enabled:
        def forbidden(): raise AssertionError('disabled trace called the policy accessor')
        state.trace_policy=forbidden
    state.episode_length_buf=torch.full((num_envs,),6,dtype=torch.long)
    state.step_dt=.02
    state.trace_step=4
    state.slr_command=torch.full((num_envs,3),.5)
    state.perception=NS(held_time=torch.full((num_envs,),.04),held_latency=torch.full((num_envs,),.06),
                        synthetic=torch.zeros(num_envs,dtype=torch.bool))
    tree=ast.parse(Path(module.__file__).read_text())
    env_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name.endswith('IsaacLabEnv'))
    method=next(n for n in env_class.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    # Only actual trace statements are instrumented; physics/reward/reset are not emulated.
    statements=[n for n in method.body if (isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in ('trace_stages','trace_time') for t in n.targets))
        or (isinstance(n,ast.If) and ast.unparse(n.test)=='trace_stages is not None')]
    assert len(statements)==4
    scope=dict(self=state,torch=torch,actions=torch.full((num_envs,3),4.),
               nav_action_orig=torch.full((num_envs,3),3.),reward=torch.full((num_envs,),.38),
               done=torch.zeros(num_envs,dtype=torch.bool),capture_policy_stages=capture_policy_stages,
               write_transition=write_transition)
    counts=Counter()
    class CountMaterialization(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            counts[func._schema.name]+=1
            return func(*args,**(kwargs or {}))
    with CountMaterialization():
        exec(compile(ast.Module(body=statements,type_ignores=[]),str(module.__file__),'exec'),scope)
    if not enabled:
        assert trace is None and not request.paths.run_root.exists()
        assert scope['trace_stages'] is scope['trace_time'] is None and state.trace_step==4
        assert counts['aten::clone']==counts['aten::_to_copy']==counts['aten::_local_scalar_dense']==0
    else:
        assert counts['aten::clone']==4 and counts['aten::_to_copy']>0
        assert counts['aten::_local_scalar_dense']==7*num_envs
        assert state.trace_step==5
        trace.close()
        assert trace.verified_row_count()==num_envs
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert [r['env_id'] for r in rows]==list(range(num_envs))
        for item in (rows[0],rows[-1]):
            assert item['step']==4 and item['distribution_mean']==[1.,1.,1.]
            assert item['policy_action']==[4.,4.,4.] and item['clipped_policy_action']==[3.,3.,3.]
            assert item['executed_command']==[.5,.5,.5]

def test_disabled_writer_never_accesses_any_trace_payload():
    from adapters.trace_logger import write_transition
    class Forbidden:
        def __getattribute__(self,name): raise AssertionError('disabled writer accessed '+name)
    x=Forbidden()
    write_transition(None,x,x,x,x,x,x)

@pytest.mark.parametrize('enabled',[False,True])
def test_trace_publication_rejects_unrequested_or_foreign_logger(enabled,tmp_path):
    from test_runtime_cli_contract import cli,load_trainer
    from rsl_rl.runtime_preflight import publish_runtime_result
    module=load_trainer('train_full_method_ppo.py')
    argv=cli(tmp_path)
    requested=tmp_path/'output/trace.jsonl'
    request=module.preflight(argv+(['--trace',str(requested)] if enabled else []))
    trace=JsonlTraceLogger(tmp_path/'foreign.jsonl'); trace.write(row()); trace.close()
    with pytest.raises(ValueError,match='trace'):
        publish_runtime_result({'status':'blocked'},request,trace)
    assert not request.paths.run_root.exists()

def test_trace_publication_binds_requested_closed_physical_rows(tmp_path):
    from test_runtime_cli_contract import cli,load_trainer
    from rsl_rl.runtime_preflight import publish_runtime_result
    module=load_trainer('train_full_method_ppo.py')
    request=module.preflight(cli(tmp_path)+['--trace',str(tmp_path/'output/trace.jsonl')])
    trace=JsonlTraceLogger(request.arguments.trace); trace.write(row()); trace.close()
    result=publish_runtime_result({'status':'blocked'},request,trace)
    assert result['trace_rows']==1 and result['trace_path']==str(trace.path) and result['trace_verified'] is True
    assert result['effective_arguments']['trace_enabled'] is True and result['runtime_verified'] is False
    assert json.loads(Path(request.arguments.result).read_text())==json.loads(Path(request.arguments.manifest_out).read_text())
