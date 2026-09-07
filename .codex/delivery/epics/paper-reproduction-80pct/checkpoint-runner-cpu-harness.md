# Task 7 real CPU runner preparation

Inspected source: `65dcbbc2b13c92af99a9e7d4cba4110880f9b509` (runner unchanged since Task 1). Controller-only bounded diagnostic while Task 6 was the sole source writer. No production/test edits, simulator imports, legacy deserialization or checkpoint writes occurred.

## Observed contract failure

The actual OnPolicyRunner performed 102 real PPO updates over a two-environment/two-step CPU toy transition fixture. An observation hook replaced only save, and log output/directory creation were suppressed. The first intermediate save requested `model_101.pt` after 102 successful updates while `current_learning_iteration` still equaled 0; the final requested `model_102.pt` had field 102. `learn` returned None. Nonempty real Adam state contained 13 integer-keyed parameter entries with finite moments.

This confirms the already recorded Task 7 iteration defect and provides a viable actual-runner fixture. It does not test persisted bytes, manifest publication, safe loading, continuation, or simulation. Task 7 must replace the save hook with real v2 persistence in a test-owned directory and prove every manifest/payload/filename iteration matches completed updates; preserve the logging start threshold independently. Keep a real nonempty Adam round trip and N+M continuation test. Do not relabel this historical failure as Task 7 RED for a test that has not yet been written.

## Exact command

Working directory: primary repository root. Exit status 0 means the diagnostic assertions confirmed the historical defect, not that the checkpoint contract passed.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python - <<'PY'
import contextlib, io, json
from pathlib import Path
from types import SimpleNamespace
import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
torch.set_num_threads(1)
torch.manual_seed(59)
class CpuFixture:
    num_envs=2
    num_props=12
    num_obs=38
    num_nav_actions=3
    max_episode_length=1000
    cfg=SimpleNamespace(env=SimpleNamespace(his_len=2))
    def __init__(self):
        self.rays=torch.ones(2,5)
        self.t=0
        self.episode_length_buf=torch.zeros(2,dtype=torch.long)
    def get_observations(self):
        x=torch.linspace(-.2,.3,38).repeat(2,1)
        x[1]+=.1
        x+=(self.t%3)*.01
        return x
    def get_privileged_observations(self): return None
    def get_extras(self): return {}
    def reset(self): return self.get_observations(),None
    def step(self,actions):
        self.t+=1
        return self.get_observations(),None,torch.tensor([.2,-.1]),torch.zeros(2,dtype=torch.bool),{}
cfg={"runner":{"policy_class_name":"ActorCritic","algorithm_class_name":"PPO","num_steps_per_env":2,"save_interval":1},
     "policy":{"actor_hidden_dims":[8],"critic_hidden_dims":[8],"encoder_hidden_dims":[8],"init_noise_std":.3},
     "algorithm":{"num_learning_epochs":1,"num_mini_batches":1,"schedule":"fixed"}}
records=[]
with contextlib.redirect_stdout(io.StringIO()):
    runner=OnPolicyRunner(CpuFixture(),cfg,log_dir="/unused-cpu-probe-save-observer",device="cpu")
    runner.print_log=lambda *args,**kwargs: None
    # Intercept only persistence/log-directory creation; real rollout, PPO updates and Adam run.
    from unittest.mock import patch
    updates=[0]
    original_update=runner.alg.update
    def update():
        value=original_update()
        updates[0]+=1
        return value
    runner.alg.update=update
    runner.save=lambda path,infos=None: records.append({"filename":Path(path).name,"field":runner.current_learning_iteration,"successful_updates":updates[0]})
    with patch("rsl_rl.runners.on_policy_runner.os.makedirs") as mkdir:
        result=runner.learn(102)
    assert mkdir.call_count==1
assert records==[
    {"filename":"model_101.pt","field":0,"successful_updates":102},
    {"filename":"model_102.pt","field":102,"successful_updates":102}]
state=runner.alg.optimizer.state_dict()
assert state["state"] and all(type(k) is int for k in state["state"])
assert all(torch.isfinite(v["exp_avg"]).all() for v in state["state"].values())
print(json.dumps({"updates":updates[0],"save_observations":records,"learn_return":result,
"adam_parameter_states":len(state["state"]),"adam_integer_keys":True,"moments_finite":True,
"boundary":"CPU toy transition fixture; no simulator, files, checkpoint load or resume proof"}))
PY
```

## Output

```json
{"updates": 102, "save_observations": [{"filename": "model_101.pt", "field": 0, "successful_updates": 102}, {"filename": "model_102.pt", "field": 102, "successful_updates": 102}], "learn_return": null, "adam_parameter_states": 13, "adam_integer_keys": true, "moments_finite": true, "boundary": "CPU toy transition fixture; no simulator, files, checkpoint load or resume proof"}
```

## Converter tooling discovery (not a sandbox pass)

`command -v bwrap`, `bwrap --version`, and `bwrap --help` found `/usr/bin/bwrap`, version 0.6.1, with `--unshare-all`, `--clearenv`, `--ro-bind-fd`, `--bind-fd`, `--cap-drop`, `--new-session` and `--die-with-parent`. `/usr/bin/timeout` and `/usr/bin/prlimit` exist. No converter or sandbox isolation probe was run in this continuation. An earlier minimal namespace probe in checkpoint-contract-audit.md does not establish a correctly confined Torch converter. Task 7 must validate its actual sandbox invocation or fail closed.
