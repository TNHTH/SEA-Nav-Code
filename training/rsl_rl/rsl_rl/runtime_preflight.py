"""Simulator-free runtime input boundary. No import or directory creation here."""
import argparse
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
from typing import Optional
import yaml
from rsl_rl.experiment_config import resolve_run_config, resolved_config_to_dict

@dataclass(frozen=True)
class RuntimePaths:
    launcher: Path
    run_root: Path
    asset_root: Path
    checkpoint_manifest: Optional[Path] = None

def _canonical(path):
    p=Path(path).absolute()
    if ".." in p.parts or p.resolve()!=p:
        raise ValueError("runtime paths must be canonical, without symlinks or parent traversal")
    return p

def validate_runtime_paths(paths):
    launcher,run_root,asset_root=map(_canonical,(paths.launcher,paths.run_root,paths.asset_root))
    if not launcher.is_file() or not os.access(str(launcher),os.X_OK):
        raise ValueError("launcher must be an executable regular file")
    if not asset_root.is_dir():
        raise ValueError("asset_root must be an existing directory")
    if run_root.exists() and not run_root.is_dir():
        raise ValueError("run_root must be a directory")
    if run_root == asset_root or run_root in asset_root.parents or asset_root in run_root.parents:
        raise ValueError("run_root and asset_root must be disjoint")
    cp=None
    if paths.checkpoint_manifest is not None:
        cp=_canonical(paths.checkpoint_manifest)
        if not cp.is_file():
            raise ValueError("checkpoint manifest must be a regular file")
        try: cp.relative_to(asset_root)
        except ValueError as exc: raise ValueError("checkpoint manifest must be inside asset_root") from exc
    return RuntimePaths(launcher,run_root,asset_root,cp)

def contained_output(root,value):
    p=_canonical(value)
    try: relative=p.relative_to(root)
    except ValueError as exc: raise ValueError("output must be inside run_root") from exc
    if relative==Path("."):
        raise ValueError("output must name a file inside run_root")
    return p

def validate_output_targets(paths, arguments, input_paths=()):
    outputs=[contained_output(paths.run_root,getattr(arguments,key))
             for key in ("result","trace","manifest_out","log_dir")
             if key!="trace" or getattr(arguments,key) is not None]
    for i,path in enumerate(outputs):
        if path.exists():
            raise ValueError("output target already exists: "+str(path))
        for other in outputs[i+1:]:
            if path==other or path in other.parents or other in path.parents:
                raise ValueError("output targets must be distinct with no ancestor conflict")
        for other in input_paths:
            if path==other or path in other.parents or other in path.parents:
                raise ValueError("output conflicts with runtime input")
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError("output parent is not a directory")

def validate_command_update(mapping):
    if not isinstance(mapping,dict) or set(mapping)-{"goal_x","goal_y","reset"}:
        raise ValueError("unknown command fields")
    result={}
    for k,v in mapping.items():
        if k=="reset":
            if not isinstance(v,bool): raise ValueError("reset must be boolean")
            result[k]=v
        else:
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
                raise ValueError("goal must be a finite number")
            result[k]=float(v)
    return result

def blocked_result(stack,reason,remediation,dependency_status="unknown"):
    return dict(status="blocked",stack=stack,runtime_stack=stack,reason=str(reason),remediation=remediation,
                dependency_status=dependency_status,validation_rung=0,runtime_verified=False)

def asset_receipt(paths, stack):
    """Explicit local integrity inventory, never a controller safety certificate."""
    import hashlib
    names=["ctrl_model/encoder_vel.jit","ctrl_model/encoder_latent.jit","ctrl_model/body_latest.jit"]
    if stack=="isaaclab_adapter": names.append("go2.usd")
    rows=[]
    for name in names:
        path=_canonical(paths.asset_root/name)
        row={"path":name,"status":"blocked_missing"}
        if path.is_file():
            row.update(status="present_unverified",size_bytes=path.stat().st_size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
    return {"assets":rows,"controller_interface_status":"blocked_unverified",
            "controller_provenance_status":"unknown","runtime_ready":False}

def require_runtime_prerequisites(request):
    # Task8/real runtime integration must establish the low-level controller
    # interface/provenance. Mere presence or matching bytes cannot authorize it.
    if not request.environment["asset_prerequisites"]["runtime_ready"]:
        raise RuntimeError("blocked: controller interface/provenance and runtime assets require real validation")

@dataclass(frozen=True)
class RuntimeRequest:
    resolved_config: object
    paths: RuntimePaths
    arguments_json: str
    environment_json: str
    remaining_argv: tuple = ()
    checkpoint_json: str = "null"

    @property
    def arguments(self):
        return argparse.Namespace(**json.loads(self.arguments_json))
    @property
    def environment(self):
        return json.loads(self.environment_json)
    @property
    def checkpoint(self):
        return json.loads(self.checkpoint_json)


def checkpoint_preflight(manifest_path, artifact_root, config_hash, mode):
    """Run in the actual runtime interpreter (Gym uses its isolated child)."""
    if manifest_path is None:
        return None
    import tempfile
    import torch
    from rsl_rl.utils.checkpoint import load_checkpoint_v2, save_checkpoint_v2, require_weights_only, validate_checkpoint_mode
    require_weights_only()
    # A signature alone is insufficient: verify a benign real v2 round trip.
    with tempfile.TemporaryDirectory(prefix="sea-nav-safe-checkpoint-") as directory:
        path=Path(directory)/"capability.json"
        save_checkpoint_v2(path,model_state_dict={"probe":torch.tensor([1.])},iteration=0,
                           producer_commit="0"*40,resolved_config_sha256=config_hash)
        probe=load_checkpoint_v2(path,artifact_root=Path(directory),map_location="cpu")
        if not torch.equal(probe.model_state_dict["probe"],torch.tensor([1.])):
            raise ValueError("blocked: safe checkpoint capability roundtrip failed")
    loaded=validate_checkpoint_mode(load_checkpoint_v2(manifest_path,artifact_root=artifact_root,
        map_location="cpu",expected_resolved_config_sha256=config_hash),mode)
    return {"manifest":asdict(loaded.manifest),"manifest_sha256":loaded.manifest_sha256,
            "mode":mode,"capability":{"torch":torch.__version__,"weights_only_roundtrip":True},
            "continuation_scope":"model/optimizer/iteration only; no RNG or physical trajectory restoration"}


def apply_runner_checkpoint(request, runner):
    if request.paths.checkpoint_manifest is None:
        return None
    return runner.load(request.paths.checkpoint_manifest,artifact_root=request.paths.asset_root,
        mode=request.arguments.checkpoint_mode,expected_manifest_sha256=request.checkpoint["manifest_sha256"])


def apply_model_checkpoint(request, model, *, map_location):
    if request.paths.checkpoint_manifest is None:
        return None
    if request.arguments.checkpoint_mode!="inference":
        raise ValueError("model-only consumer requires inference mode")
    from rsl_rl.utils.checkpoint import load_checkpoint_v2, apply_checkpoint_state
    loaded=load_checkpoint_v2(request.paths.checkpoint_manifest,artifact_root=request.paths.asset_root,
        map_location=map_location,expected_resolved_config_sha256=request.resolved_config.resolved_sha256)
    if loaded.manifest_sha256!=request.checkpoint["manifest_sha256"]:
        raise ValueError("checkpoint changed since preflight")
    apply_checkpoint_state(model, loaded.model_state_dict)
    return loaded.manifest

def _gym_arguments(argv, runtime_defaults):
    """Validate the native Gym CLI surface without importing gymutil/Torch."""
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--task',choices=['go2_pos_rough'],default='go2_pos_rough')
    for name in ('experiment_name','run_name'):
        parser.add_argument('--'+name)
    parser.add_argument('--rl_device',default='cuda:0')
    parser.add_argument('--sim_device',default='cuda:0')
    parser.add_argument('--pipeline',choices=('cpu','gpu'),default='gpu')
    for name in ('num_envs','seed','max_iterations','graphics_device_id','num_threads','subscenes','slices'):
        parser.add_argument('--'+name,type=int)
    for name in ('headless','no_wandb'):
        parser.add_argument('--'+name,action='store_true',default=True)
    parser.add_argument('--physx',action='store_true')
    native=parser.parse_args(argv)
    if native.max_iterations is not None and native.max_iterations<1:
        raise ValueError('max_iterations must be positive')
    import re
    for name in ('rl_device','sim_device'):
        if not re.fullmatch(r'cpu|cuda:[0-9]+',getattr(native,name)):
            raise ValueError(name+' must be cpu or cuda:N')
    for name in ('num_threads','subscenes','slices'):
        value=getattr(native,name)
        if value is not None and value<0:
            raise ValueError(name+' must be nonnegative')
    result=argparse.Namespace(**runtime_defaults)
    for key,value in vars(native).items():
        if value is not None:
            setattr(result,key,value)
    return result

def reconcile_gym_arguments(request,actual):
    """Bind the real native parser; never silently replace selected device/flags."""
    desired=vars(request.arguments)
    for key in ('task','headless','no_wandb','rl_device','sim_device'):
        if getattr(actual,key)!=desired[key]:
            raise ValueError('actual Gym argument differs from preflight: '+key)
    for key in ('num_envs','seed'):
        setattr(actual,key,desired[key])
    return actual

def preflight(argv, *, runtime_stack, repo_root, build_parser=None, entrypoint="train"):
    """Bind explicit manifest modes before proprietary startup or output creation."""
    common=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    common.add_argument("--config",type=Path,required=True)
    common.add_argument("--launcher",type=Path,required=True)
    common.add_argument("--run-root",type=Path,required=True)
    common.add_argument("--asset-root",type=Path,required=True)
    manifests=common.add_mutually_exclusive_group()
    manifests.add_argument("--checkpoint-manifest",type=Path)
    manifests.add_argument("--init-checkpoint-manifest",type=Path)
    manifests.add_argument("--resume-checkpoint-manifest",type=Path)
    common.add_argument("--checkpoint-mode",choices=("fresh","resume","warm_start","inference"))
    common.add_argument("--producer-commit",required=True)
    common.add_argument("--algorithm-profile")
    common.add_argument("--implementation-delta",action="append")
    common.add_argument("--preflight-only",action="store_true")
    base,remaining=common.parse_known_args(argv)
    import re
    if not re.fullmatch(r"[0-9a-f]{40}",base.producer_commit):
        raise ValueError("producer_commit must be an explicit full lowercase commit SHA")
    mode="fresh"
    manifest=base.checkpoint_manifest
    if base.init_checkpoint_manifest is not None:
        mode,manifest="warm_start",base.init_checkpoint_manifest
    elif base.resume_checkpoint_manifest is not None:
        mode,manifest="resume",base.resume_checkpoint_manifest
    elif manifest is not None:
        mode="inference"
    if base.checkpoint_mode is not None and base.checkpoint_mode!=mode:
        raise ValueError("blocked: checkpoint mode requires its matching explicit manifest flag")
    if entrypoint in ("train","ppo","acsi") and mode=="inference":
        raise ValueError("training requires init/resume manifest flags")
    if entrypoint in ("play","smoke") and mode not in ("fresh","inference"):
        raise ValueError("inference entrypoint requires --checkpoint-manifest")
    if entrypoint=="play" and mode!="inference":
        raise ValueError("play requires --checkpoint-manifest")
    config_path=_canonical(base.config)
    document=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document,dict) or set(document)!={"schema_version","algorithm_profile","implementation_delta","runtime"} or type(document["schema_version"]) is not int or document["schema_version"]!=1:
        raise ValueError("runtime YAML schema/unknown fields")
    if not isinstance(document["implementation_delta"],list) or not all(isinstance(x,str) for x in document["implementation_delta"]):
        raise ValueError("implementation_delta must be explicit strings")
    resolved=resolve_run_config(registry_path=Path(repo_root)/"configs/parity_registry.yaml",
        algorithm_profile=base.algorithm_profile or document["algorithm_profile"],runtime_stack=runtime_stack,
        implementation_delta=document["implementation_delta"] if base.implementation_delta is None else base.implementation_delta)
    paths=validate_runtime_paths(RuntimePaths(base.launcher,base.run_root,base.asset_root,manifest))
    if any(x.split("=")[0] in ("--checkpoint","--init-checkpoint","--resume","--load_run","--load-run") for x in remaining):
        raise ValueError("blocked: legacy raw/numeric checkpoint flags are unsupported; use explicit manifest flags")
    settings_doc=document["runtime"]
    if not isinstance(settings_doc,dict):
        raise ValueError("runtime YAML must be a mapping")
    allowed={"timeout_seconds","num_envs","seed","command_filter_alpha","replay_ring_buffer_steps",
             "replay_undo_min","replay_undo_max","source_max_goal_level","enable_collision_replay"}
    if set(settings_doc)-allowed:
        raise ValueError("unknown runtime YAML settings")
    for key,value in settings_doc.items():
        if key=="enable_collision_replay":
            if not isinstance(value,bool): raise ValueError("enable_collision_replay must be boolean")
        elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError("runtime setting must be a finite number: "+key)
        if key in ("num_envs","seed","replay_ring_buffer_steps","replay_undo_min","replay_undo_max") and type(value) is not int:
            raise ValueError("runtime setting must be integer: "+key)
    if build_parser is None:
        parsed=_gym_arguments(remaining,settings_doc)
        remaining=tuple(remaining)
    else:
        parser=build_parser()
        parser.allow_abbrev=False
        parser.set_defaults(**settings_doc)
        parser.add_argument("--headless",action="store_true")
        parser.add_argument("--device",default="cuda:0")
        parsed=parser.parse_args(remaining)
        remaining=()
    defaults={"timeout_seconds":60.,"num_envs":1,"seed":42,"command_filter_alpha":.5,
              "replay_ring_buffer_steps":180,"replay_undo_min":100,"replay_undo_max":150,
              "source_max_goal_level":10.,"enable_collision_replay":False}
    for key,value in defaults.items():
        if not hasattr(parsed,key): setattr(parsed,key,value)
    protected={"cbf_fov_deg":180.,"cbf_footprint_radius_m":0.,"cbf_min_effective_clearance_m":1e-4,
               "command_filter_mode":"source_alpha_only","action_chain_mode":"current_pre_delay_cbf",
               "source_early_reset_prob_min":.1,"source_early_reset_prob_max":.5,"source_goal_level":0.,
               "replay_prob":.8,"source_pos_hist_interval_steps":10,"command_delay_s":0.,
               "source_stand_still_time_steps":150}
    for key,value in protected.items():
        if hasattr(parsed,key) and getattr(parsed,key)!=value:
            raise ValueError("runtime override conflicts with profile: "+key)
    if entrypoint=="ppo" and parsed.enable_collision_replay:
        raise ValueError("ordinary PPO environment does not support replay")
    if entrypoint=="smoke" and parsed.enable_collision_replay:
        raise ValueError("blocked: smoke is a no-replay diagnostic; use the resolved ACSI trainer")
    if entrypoint in ("ppo","smoke") and parsed.num_envs != 1:
        raise ValueError("blocked: this diagnostic entrypoint requires one environment; use the ACSI trainer for masked multi-env reset")
    if entrypoint == "smoke":
        parsed.stop_on_first_done = True
    for key in ("num_envs","iterations","rollout_steps","steps","ppo_num_learning_epochs","ppo_num_mini_batches"):
        if hasattr(parsed,key) and (type(getattr(parsed,key)) is not int or getattr(parsed,key)<1):
            raise ValueError(key+" must be a positive integer")
    for key in ("ppo_learning_rate","init_std"):
        if hasattr(parsed,key) and (not math.isfinite(getattr(parsed,key)) or getattr(parsed,key)<=0):
            raise ValueError(key+" must be finite and positive")
    for key,value in vars(parsed).items():
        if isinstance(value,float) and not math.isfinite(value):
            raise ValueError(key+' must be finite')
    if getattr(parsed,'ppo_entropy_coef',0)<0:
        raise ValueError('ppo_entropy_coef must be nonnegative')
    if not 0<=parsed.seed<=2**32-1:
        raise ValueError('seed must be in the NumPy/Torch shared uint32 range')
    if entrypoint in ('ppo','acsi'):
        samples=parsed.rollout_steps*parsed.num_envs
        batches=getattr(parsed,'ppo_num_mini_batches',1)
        if samples<2 or samples % batches or samples//batches<2:
            raise ValueError('PPO requires at least two rollout samples per exact mini-batch')
    for key in ("disable_source_contact_termination","source_play_eval_terminal_semantics","force_replay_smoke","force_source_parity_smoke","force_reward_done_parity_smoke"):
        if getattr(parsed,key,False):
            raise ValueError("blocked: unsupported diagnostic runtime setting "+key)
    # These are selected consumers, not optional flags which may silently disable parity.
    parsed.enable_source_perception_delay=True
    parsed.enable_source_reward_done_parity=True
    parsed.implementation_delta=list(resolved.identity.implementation_delta)
    env_options=dict(timeout_seconds=parsed.timeout_seconds,replay_enabled=parsed.enable_collision_replay,
        capacity=parsed.replay_ring_buffer_steps,undo=(parsed.replay_undo_min,parsed.replay_undo_max),
        max_level=parsed.source_max_goal_level,command_filter_alpha=parsed.command_filter_alpha)
    if runtime_stack == "isaac_gym_preview4":
        # Gym Preview4 must import before Torch in its process. Validate the real
        # Torch-dependent constructor/replay consumers in an isolated child.
        import subprocess
        import sys
        code = ("import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                "from rsl_rl.experiment_config import resolve_run_config; "
                "from rsl_rl.environment_profile import environment_settings,apply_algorithm_profile; "
                "from rsl_rl.runtime_preflight import checkpoint_preflight; "
                "r=json.loads(sys.argv[2]); c=resolve_run_config(**r); "
                "apply_algorithm_profile({'policy':{},'algorithm':{}},c); "
                "assert c.resolved_sha256==sys.argv[4]; "
                "print(json.dumps({'environment':environment_settings(c,**json.loads(sys.argv[3])), "
                "'checkpoint':checkpoint_preflight(**json.loads(sys.argv[5]))}))")
        identity=dict(registry_path=str(Path(repo_root)/"configs/parity_registry.yaml"),
            algorithm_profile=resolved.identity.algorithm_profile,runtime_stack=runtime_stack,
            implementation_delta=list(resolved.identity.implementation_delta))
        child=subprocess.run([sys.executable,"-I","-B","-c",code,str(Path(repo_root)/"training/rsl_rl"),
                              json.dumps(identity),json.dumps(env_options),resolved.resolved_sha256,
                              json.dumps(dict(manifest_path=str(paths.checkpoint_manifest) if manifest else None,
                                  artifact_root=str(paths.asset_root),config_hash=resolved.resolved_sha256,mode=mode))],
                              capture_output=True,text=True,timeout=30)
        if child.returncode:
            raise ValueError("isolated Gym consumer preflight rejected: " + child.stderr.strip())
        child_result=json.loads(child.stdout)
        env,checkpoint=child_result["environment"],child_result["checkpoint"]
    else:
        from rsl_rl.environment_profile import environment_settings, runtime_constructor_settings
        env=environment_settings(resolved,**env_options)
        env['constructor_settings']=runtime_constructor_settings(resolved,vars(parsed))
        checkpoint=checkpoint_preflight(paths.checkpoint_manifest,paths.asset_root,resolved.resolved_sha256,mode)
    env["asset_prerequisites"] = asset_receipt(paths,runtime_stack)
    for key,name in (("result","result.json"),("manifest_out","manifest.json"),("log_dir","training")):
        value=getattr(parsed,key,"") or str(paths.run_root/name)
        setattr(parsed,key,str(contained_output(paths.run_root,value)))
    trace=getattr(parsed,"trace",None)
    if entrypoint=="smoke":
        trace=trace or str(paths.run_root/"trace.jsonl")
    elif trace=="":
        raise ValueError("trace must be an explicit nonempty output path when enabled")
    parsed.trace=None if trace is None else str(contained_output(paths.run_root,trace))
    parsed.trace_enabled=parsed.trace is not None
    validate_output_targets(paths,parsed,(config_path,paths.launcher))
    parsed.preflight_only=base.preflight_only
    parsed.checkpoint_mode=mode
    parsed.producer_commit=base.producer_commit
    return RuntimeRequest(resolved,paths,json.dumps(vars(parsed),sort_keys=True),json.dumps(env,sort_keys=True),
                          tuple(remaining),json.dumps(checkpoint,sort_keys=True))

def bind_runtime_result(result, resolved, paths, environment, trace=None, arguments=None):
    """Bind only current evidence; a dependency/projection is never a runtime pass."""
    from rsl_rl.experiment_config import resolved_config_to_dict
    payload=resolved_config_to_dict(resolved)
    stack=resolved.identity.runtime_stack
    if result.get("runtime_stack",stack)!=stack:
        raise ValueError("runtime stack mismatch")
    if environment.get("resolved_config_sha256")!=resolved.resolved_sha256:
        raise ValueError("effective configuration hash mismatch")
    if environment.get("runtime_stack")!=stack:
        raise ValueError("effective runtime stack mismatch")
    from rsl_rl.environment_profile import environment_settings, runtime_constructor_settings
    replay=environment["replay"]
    expected=environment_settings(resolved,policy_dt_s=environment["policy_dt_s"],
        timeout_seconds=environment["actual_horizon_s"],replay_enabled=replay["enabled"],
        capacity=replay["capacity"],undo=tuple(replay["undo"]),max_level=replay["stored_level_bounds"][1],
        command_filter_alpha=environment["command_filter_alpha"])
    actual={k:v for k,v in environment.items() if k not in ("asset_prerequisites","constructor_settings")}
    if actual!=expected:
        raise ValueError("effective environment receipt differs from profile")
    if arguments is not None and stack=="isaaclab_adapter":
        constructor=runtime_constructor_settings(resolved,arguments)
        if environment.get("constructor_settings")!=constructor:
            raise ValueError("constructor receipt differs from actual arguments")
        if result.get("effective_constructor",constructor)!=constructor:
            raise ValueError("actual constructor differs from preflight")
    if result.get("status") not in ("blocked","failed"):
        raise ValueError("runtime passed needs a separate real lifecycle acceptance gate")
    if arguments is not None and trace is not None:
        requested_trace=arguments.get("trace")
        if arguments.get("trace_enabled") is not True or requested_trace is None:
            raise ValueError("trace logger was not requested")
        if _canonical(trace.path)!=contained_output(paths.run_root,requested_trace):
            raise ValueError("trace logger differs from requested output path")
    trace_error=None
    try:
        count=0 if trace is None else trace.verified_row_count()
    except Exception as exc:
        count=None
        trace_error=str(exc)
        result=dict(result,status="failed",trace_validation_error=trace_error)
        if "trace_rows" in result:
            result["claimed_trace_rows"]=result["trace_rows"]
    if trace_error is None and "trace_rows" in result and result["trace_rows"]!=count:
        raise ValueError("physical trace row count mismatch")
    bound=dict(result,runtime_stack=stack,resolved_config=payload,
        resolved_config_sha256=resolved.resolved_sha256,effective_environment=environment,
        runtime_paths={k:str(v) if v is not None else None for k,v in asdict(paths).items()},
        trace_rows=count,trace_path=str(trace.path) if trace is not None else None,
        trace_verified=trace is not None and trace_error is None,
        validation_rung=0,runtime_verified=False,
        controller_provenance="blocked_unverified_interface_and_provenance")
    import hashlib
    bound["effective_environment_sha256"]=hashlib.sha256(
        json.dumps(environment,sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()
    if arguments is not None:
        bound["effective_arguments"]=arguments
        bound["effective_arguments_sha256"]=hashlib.sha256(
            json.dumps(arguments,sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()
    return bound

def close_runtime_resources(resources):
    """Attempt every independent close; never hide the primary runtime error."""
    errors=[]
    for name,resource in resources:
        if resource is not None:
            try:
                resource.close()
            except Exception as exc:
                errors.append(dict(resource=name,error=str(exc)))
    return errors

def publish_runtime_result(output,request,trace=None):
    """Exclusive evidence publication; paths were checked before application startup."""
    environment=request.environment
    if "effective_environment" in output:
        from rsl_rl.environment_profile import reconcile_environment_receipt
        reconcile_environment_receipt(output["effective_environment"],environment)
    bound=bind_runtime_result(output,request.resolved_config,request.paths,environment,trace,
                              arguments=vars(request.arguments))
    bound["input_checkpoint"]=request.checkpoint
    encoded=json.dumps(bound,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    for name in ("result","manifest_out"):
        path=Path(getattr(request.arguments,name))
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as handle:
            handle.write(encoded)
    return bound
