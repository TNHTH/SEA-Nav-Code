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
            "--launcher",str(launcher),"--asset-root",str(assets),"--run-root",str(tmp_path/"output")]

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
