"""Actual entrypoint pure surfaces, fresh processes, no proprietary stand-ins."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=("full_method_runtime_smoke.py","train_full_method_ppo.py","train_full_method_acsi_replay_ppo.py")
def cli(tmp_path):
    launcher=tmp_path/"launcher"; launcher.write_text("#!/bin/sh\n"); launcher.chmod(0o700)
    assets=tmp_path/"assets"; assets.mkdir()
    return ["--config",str(ROOT/"sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml"),
            "--launcher",str(launcher),"--asset-root",str(assets),"--run-root",str(tmp_path/"output"),
            "--producer-commit","5"*40]

@pytest.mark.parametrize("script",SCRIPTS)
def test_entrypoint_has_guarded_real_preflight(script,tmp_path):
    path=ROOT/"sea_nav_current_isaaclab_full_method"/script
    tree=ast.parse(path.read_text())
    assert any(isinstance(n,ast.FunctionDef) and n.name=="preflight" for n in tree.body)
    argv=cli(tmp_path)
    code="""
import importlib.util,sys,json
from pathlib import Path
path=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location("entry",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
request=module.preflight(json.loads(sys.argv[2]))
assert request.resolved_config.identity.algorithm_profile=="upstream_fbce672c"
assert request.environment["effective_weights"]["velocity"]==.08
assert request.environment["perception"]["acquisition_period_s"]==.02
assert not request.paths.run_root.exists()
assert not any(n.startswith(("isaacgym","isaaclab","omni")) for n in sys.modules)
print("pure actual preflight passed")
"""
    run=subprocess.run([sys.executable,"-I","-B","-c",code,str(path),json.dumps(argv)],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr

@pytest.mark.parametrize("extra,match",[
    (["--algorithm-profile","paper_v1"],"blocked"),
    (["--algorithm-profile","unknown"],"unknown"),
    (["--implementation-delta","unknown"],"unknown"),
    (["--implementation-delta","ppo_state_identity_repair"],"replay_reset_reconstruction"),
    (["--cbf-fov-deg","240"],"conflicts"),
    (["--init-checkpoint","legacy.pt"],"blocked"),
    (["--checkpoint-mode","resume"],"blocked"),
    (["--timeout-seconds","nan"],"finite"),
    (["--source-stand-still-time-steps","1"],"conflicts"),
    (["--seed","-1"],"uint32"),
    (["--rollout-steps","1"],"two rollout"),
])
def test_preflight_rejects_before_import_or_output(extra,match,tmp_path):
    path=ROOT/"sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py"
    tree=ast.parse(path.read_text())
    assert any(isinstance(n,ast.FunctionDef) and n.name=="preflight" for n in tree.body)
    code="""
import importlib.util,sys,json
spec=importlib.util.spec_from_file_location("entry",sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
try: m.preflight(json.loads(sys.argv[2]))
except (ValueError,SystemExit) as exc:
 print(str(exc))
 assert not any(n.startswith(("isaacgym","isaaclab","omni")) for n in sys.modules)
 sys.exit(7)
raise AssertionError("accepted invalid config")
"""
    argv=cli(tmp_path)+extra
    run=subprocess.run([sys.executable,"-I","-B","-c",code,str(path),json.dumps(argv)],capture_output=True,text=True,timeout=30)
    assert run.returncode==7,run.stdout+run.stderr
    assert match in (run.stdout+run.stderr).lower()
    assert not (tmp_path/"output").exists()

def test_actual_gym_preflight_keeps_parent_torch_free(tmp_path):
    argv=cli(tmp_path)+['--task','go2_pos_rough','--headless']
    path=ROOT/'training/legged_gym/legged_gym/scripts/train.py'
    code='''
import importlib.util,sys,json
s=importlib.util.spec_from_file_location('train',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
assert 'torch' not in sys.modules
r=m.preflight(json.loads(sys.argv[2]))
assert 'torch' not in sys.modules
assert r.remaining_argv==('--task','go2_pos_rough','--headless')
assert r.environment['effective_weights']['velocity']==.08
assert r.arguments.headless is True and r.arguments.no_wandb is True
assert r.arguments.sim_device==r.arguments.rl_device=='cuda:0'
assert r.arguments.trace is None and r.arguments.trace_enabled is False
assert not any(n.startswith(('isaacgym','isaaclab','omni')) for n in sys.modules)
print('pure Gym parent; actual converters checked in child')
'''
    run=subprocess.run([sys.executable,'-I','-B','-c',code,str(path),json.dumps(argv)],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr

def test_real_adapter_missing_prerequisites_emit_bound_blocked_not_simulator_pass(tmp_path):
    argv=cli(tmp_path)
    path=ROOT/'sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py'
    run=subprocess.run([sys.executable,'-I','-B',str(path)]+argv,capture_output=True,text=True,timeout=30)
    assert run.returncode==3,run.stdout+run.stderr
    result=json.loads((tmp_path/'output/result.json').read_text())
    assert result['status']=='blocked' and result['ok'] is False
    assert result['runtime_verified'] is False and result['trace_rows']==0
    assert result['effective_environment']['asset_prerequisites']['runtime_ready'] is False

@pytest.mark.parametrize("kind",["same","nested","existing","asset"])
def test_output_targets_rejected_before_any_creation(kind,tmp_path):
    from rsl_rl.runtime_preflight import preflight
    from sea_nav_current_isaaclab_full_method.full_method_runtime_smoke import build_parser
    argv=cli(tmp_path)
    target=tmp_path/'output/same'
    if kind=="same": extra=['--result',str(target),'--trace',str(target)]
    elif kind=="nested": extra=['--result',str(target),'--trace',str(target/'child')]
    elif kind=="existing":
        (tmp_path/'output').mkdir(); target.write_text('preserve')
        extra=['--result',str(target)]
    else:
        argv[argv.index('--run-root')+1]=str(tmp_path/'assets')
        extra=[]
    with pytest.raises(ValueError,match='output|disjoint'):
        preflight(argv+extra,runtime_stack='isaaclab_adapter',repo_root=ROOT,build_parser=build_parser,entrypoint='smoke')
    if kind=="existing": assert target.read_text()=='preserve'

def test_nested_manifest_created_by_actual_blocked_cli(tmp_path):
    argv=cli(tmp_path)+['--manifest-out',str(tmp_path/'output/a/b/manifest.json')]
    path=ROOT/'sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py'
    run=subprocess.run([sys.executable,'-I','-B',str(path)]+argv,capture_output=True,text=True,timeout=30)
    assert run.returncode==3,run.stdout+run.stderr
    result=json.loads((tmp_path/'output/result.json').read_text())
    manifest=json.loads((tmp_path/'output/a/b/manifest.json').read_text())
    assert result==manifest and result['status']=='blocked'

def test_gym_real_blocked_result_is_bound_without_adapter_dependency(tmp_path):
    argv=cli(tmp_path)
    path=ROOT/'training/legged_gym/legged_gym/scripts/train.py'
    run=subprocess.run([sys.executable,'-I','-B',str(path)]+argv,capture_output=True,text=True,timeout=30)
    assert run.returncode==3,run.stdout+run.stderr
    result=json.loads((tmp_path/'output/result.json').read_text())
    assert result['runtime_stack']=='isaac_gym_preview4'
    assert result['status']=='blocked' and result['trace_rows']==0 and not result['trace_verified']
    assert result['effective_arguments_sha256'] and result['effective_environment_sha256']
    assert 'sea_nav_current_isaaclab_full_method' not in path.read_text()

@pytest.mark.parametrize('bad',[['--test'],['--horovod'],['--nonesuch'],['--pipeline','garbage'],['--sim_device','invalid']])
def test_gym_unknown_flags_rejected_in_pure_parser(bad,tmp_path):
    from rsl_rl.runtime_preflight import preflight
    with pytest.raises((ValueError,SystemExit)):
        preflight(cli(tmp_path)+bad,runtime_stack='isaac_gym_preview4',repo_root=ROOT)
    assert not (tmp_path/'output').exists()

@pytest.mark.parametrize('observed_seed',[42,None])
def test_smoke_actual_seed_and_horizon_setup_before_carrier_construction(observed_seed):
    from types import SimpleNamespace as NS
    from sea_nav_current_isaaclab_full_method import full_method_runtime_smoke as module
    cfg=NS(seed=999,episode_length_s=8.)
    request=NS(arguments=NS(seed=42,timeout_seconds=60.),environment={'actual_horizon_s':60.})
    evidence=module.apply_carrier_run_settings(cfg,request)
    assert cfg.seed==42 and cfg.episode_length_s==60.
    assert evidence['requested_seed']==evidence['configured_env_seed']==42
    cfg.seed=observed_seed
    observed=module.observe_carrier_run_settings(cfg,evidence,request)
    assert observed['env_seed']==observed_seed
    assert observed['nominal_horizon_s']==60.
    assert observed['semantic_timeout_rule']=='episode_length > round(nominal_horizon_s / policy_dt_s)'
    cfg.seed=999
    with pytest.raises(ValueError,match='seed'):
        module.observe_carrier_run_settings(cfg,evidence,request)
    cfg.seed=observed_seed
    cfg.episode_length_s=8.
    with pytest.raises(ValueError,match='horizon'):
        module.observe_carrier_run_settings(cfg,evidence,request)
    tree=ast.parse(Path(module.__file__).read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    make=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and ast.unparse(n.func)=='gym.make')
    application=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and ast.unparse(n.func)=='apply_carrier_run_settings')
    assert application.lineno<make.lineno

def test_smoke_rejects_replay_activation_in_executable_yaml(tmp_path):
    import yaml
    from sea_nav_current_isaaclab_full_method import full_method_runtime_smoke as module
    argv=cli(tmp_path)
    config=yaml.safe_load(Path(argv[1]).read_text())
    config['runtime']['enable_collision_replay']=True
    path=tmp_path/'replay.yaml'; path.write_text(yaml.safe_dump(config))
    argv[1]=str(path)
    with pytest.raises(ValueError,match='no-replay'):
        module.preflight(argv)
    assert not (tmp_path/'output').exists()

def load_trainer(script):
    spec=importlib.util.spec_from_file_location('trace_entry',ROOT/'sea_nav_current_isaaclab_full_method'/script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def attach_actual_training_trace(module,arguments,actor):
    """Execute the actual main attachment; no carrier or simulator substitute."""
    from types import SimpleNamespace as NS
    from adapters.trace_logger import JsonlTraceLogger
    tree=ast.parse(Path(module.__file__).read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    targets={'active_trace','adapter_env.trace_logger','adapter_env.trace_policy'}
    statements=[n for n in main.body if isinstance(n,ast.Assign)
                and any(ast.unparse(t) in targets for t in n.targets)]
    assert len(statements)==3
    env=NS(trace_logger=None,trace_policy=None)
    scope=dict(args=arguments,adapter_env=env,runner=NS(alg=NS(actor_critic=actor)),
               JsonlTraceLogger=JsonlTraceLogger)
    exec(compile(ast.Module(body=statements,type_ignores=[]),str(module.__file__),'exec'),scope)
    return env,scope['active_trace']

@pytest.mark.parametrize('script',SCRIPTS[1:])
@pytest.mark.parametrize('enabled',[False,True])
def test_training_trace_actual_cli_attachment_is_opt_in(script,enabled,tmp_path):
    module=load_trainer(script)
    path=tmp_path/'output/explicit/trace.jsonl'
    request=module.preflight(cli(tmp_path)+(['--trace',str(path)] if enabled else []))
    assert request.arguments.trace_enabled is enabled
    assert request.arguments.trace==(str(path) if enabled else None)
    assert not request.paths.run_root.exists()
    actor=object()
    env,trace=attach_actual_training_trace(module,request.arguments,actor)
    if enabled:
        assert trace is env.trace_logger and trace.path==path
        assert env.trace_policy() is actor
        trace.close()
        assert trace.verified_row_count()==0
    else:
        assert trace is env.trace_logger is env.trace_policy is None
        assert not request.paths.run_root.exists()

def test_smoke_trace_remains_enabled_by_default(tmp_path):
    module=load_trainer(SCRIPTS[0])
    request=module.preflight(cli(tmp_path))
    assert request.arguments.trace_enabled is True
    assert request.arguments.trace==str(tmp_path/'output/trace.jsonl')
    assert not request.paths.run_root.exists()

@pytest.mark.parametrize('script',SCRIPTS[1:])
@pytest.mark.parametrize('kind',['empty','same','nested','existing','outside'])
def test_explicit_training_trace_keeps_output_guards(script,kind,tmp_path):
    module=load_trainer(script)
    argv=cli(tmp_path)
    trace=tmp_path/'output/trace.jsonl'
    extra=[]
    if kind=='empty': trace=''
    elif kind=='same': extra=['--result',str(trace)]
    elif kind=='nested': extra=['--result',str(trace/'result.json')]
    elif kind=='existing': trace.parent.mkdir(); trace.write_text('preserve')
    elif kind=='outside': trace=tmp_path/'outside.jsonl'
    with pytest.raises(ValueError,match='trace|output'):
        module.preflight(argv+['--trace',str(trace)]+extra)
    if kind=='existing': assert trace.read_text()=='preserve'
    else: assert not (tmp_path/'output').exists()

@pytest.mark.parametrize('script',SCRIPTS[1:])
@pytest.mark.parametrize('enabled',[False,True])
def test_actual_blocked_training_cli_never_claims_planned_trace(script,enabled,tmp_path):
    trace=tmp_path/'output/explicit/trace.jsonl'
    argv=cli(tmp_path)+(['--trace',str(trace)] if enabled else [])
    path=ROOT/'sea_nav_current_isaaclab_full_method'/script
    run=subprocess.run([sys.executable,'-I','-B',str(path)]+argv,capture_output=True,text=True,timeout=30)
    assert run.returncode==3,run.stdout+run.stderr
    result=json.loads((tmp_path/'output/result.json').read_text())
    assert result['status']=='blocked' and result['runtime_verified'] is False
    assert result['effective_arguments']['trace_enabled'] is enabled
    assert result['trace_rows']==0 and result['trace_path'] is None and result['trace_verified'] is False
    assert not trace.parent.exists()
