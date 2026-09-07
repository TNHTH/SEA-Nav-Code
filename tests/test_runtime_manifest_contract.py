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
