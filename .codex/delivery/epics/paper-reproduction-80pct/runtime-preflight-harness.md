# Task 6 runtime/preflight harness recommendation

Date: 2026-09-07. Inspected checkout: `5b99f456fbfb9eb2e6fc8eae5a03a1aa584e2b43`. This is a bounded read-only preparation note. No source, test, index, ref, dependency, remote, or simulator state was changed. The Task 5 worktree was not inspected. Scientific formula findings already recorded in `scientific-wiring-audit.md` are intentionally not repeated here.

## Evidence from the inspected commit

- `training/rsl_rl/rsl_rl/experiment_config.py:762-851` is already the simulator-free identity boundary. `resolve_run_config` produces an integrity-checked `ResolvedRunConfig`; `build_policy_kwargs`, `build_ppo_kwargs`, `build_env_profile_values`, and `build_replay_kwargs` return the four real projections. `training/rsl_rl/rsl_rl/policy_factory.py:91-99` is the real CPU policy/PPO constructor consumer.
- Gym startup is not CPU-importable. `scripts/train.py:34-39` imports `legged_gym.envs` and `legged_gym.utils` at module scope; `scripts/play.py:36-46` imports Isaac Gym and the environment stack at module scope. `utils/helpers.py:36-37` imports `gymapi/gymutil`, and `utils/task_registry.py:40` imports that helper before `TaskRegistry` can be imported.
- `TaskRegistry.get_cfgs` returns and mutates registered singleton configs (`task_registry.py:57-62`). `make_env` performs CLI mutation at `:91` and constructs the simulator environment at `:97`; `make_alg_runner` performs CLI mutation at `:141`, converts the config at `:152`, and constructs the runner at `:156`. These are the minimum environment and algorithm application seams.
- `play.py:50-79` mutates the obtained configs, including an actual 40 s horizon at `:76`, before `make_env` at `:82`; `train.py:52-54` goes directly through both registry constructors. The final effective config must therefore be serialized after the permitted play/train overrides, not when the profile is merely resolved.
- All three IsaacLab scripts import `AppLauncher`, parse arguments, and instantiate the application at module scope: smoke `:9,12-50,52`; PPO `:10,13-29,31`; ACSI PPO `:10,13-69,79`. Their real `main` functions begin only at `:118`, `:450`, and `:1359`, respectively. Importing these files is therefore forbidden in CPU tests.
- The adapter constructor seams are smoke actor/shield construction at `full_method_runtime_smoke.py:260-298`, PPO runner/post-construction layer replacement at `train_full_method_ppo.py:554-606`, and ACSI PPO runner/post-construction layer replacement at `train_full_method_acsi_replay_ppo.py:1663-1715`. The current manual dicts and layer replacements can bypass profile values.
- The adapter scripts execute `main()` again at module scope and write/close in a module-level `finally` (smoke `:1266-1275`, PPO `:691-700`, ACSI PPO `:1833-1842`). This boundary must move under `if __name__ == "__main__"` so importing a pure parser never launches or writes.
- `tools/gate_a.py:242-253` currently names an import check `isaac_gym_runtime` and returns `passed` when the package imports. `run_cpu_gate` has no independent IsaacLab row (`:256-270`). This is dependency evidence, not runtime execution evidence.
- `tests/test_cbf_import_boundary.py` is the right precedent: execute an extracted real, simulator-free production block in a fresh isolated process, then use AST only to prove its position relative to simulator calls. It does not inject fake Isaac modules.

## Minimal interfaces to add or extend

One small shared surface is necessary because Gym must not depend on the IsaacLab adapter. Extend the existing simulator-free `rsl_rl.experiment_config` module (or a one-file sibling in the same package) with:

```text
@dataclass(frozen=True)
ProfileCLI(registry_path: Path, algorithm_profile: str,
           implementation_delta: Tuple[str, ...])

parse_profile_cli(argv: Sequence[str], *, runtime_stack: str)
    -> (ProfileCLI, remaining_argv)
resolve_profile_cli(request: ProfileCLI, *, runtime_stack: str)
    -> ResolvedRunConfig
```

`runtime_stack` is fixed by the entrypoint (`isaac_gym_preview4` or `isaaclab_adapter`), not user-selectable. The parser must consume only the shared profile flags and return simulator-specific arguments untouched. It must require the explicit repair delta, canonicalize the registry path, use the existing resolver, and reject `paper_v1` before any proprietary import. A Task 6 scope extension to this shared file is smaller and safer than duplicating profile parsing or making Gym import `adapters`.

Keep the adapter-specific contract in the planned `adapters/runtime_commands.py`:

```text
@dataclass(frozen=True)
RuntimePaths(launcher: Path, run_root: Path,
             asset_root: Optional[Path], checkpoint_manifest: Optional[Path])

validate_runtime_paths(paths) -> RuntimePaths       # canonical, containment checked, no writes
validate_command_update(mapping) -> immutable command values
blocked_result(stack, reason, remediation, *, dependency_status) -> dict
count_physical_jsonl_rows(path) -> int
bind_runtime_result(result, resolved_config, paths, trace_path) -> dict
```

`bind_runtime_result` should reject a stack/hash mismatch, a historical `passed` payload, malformed JSONL, or a claimed trace count different from the physical row count. It should bind the exact resolved payload/hash, effective post-override consumer values, launcher/assets/checkpoint manifest identities, validation rung, and result status. Validation occurs before directory creation.

## Exact startup ordering to implement and prove

### Gym train/play/task registry

1. In `scripts/train.py` and `scripts/play.py`, retain only standard-library plus `rsl_rl` preflight imports at module scope. `main(argv)` first calls `parse_profile_cli` and `resolve_profile_cli`; only then does it import `isaacgym`, `legged_gym.envs`, and `legged_gym.utils`. A missing/broken dependency is converted to the stack-specific blocked/failed dependency result, never to runtime success.
2. Pass the resolved object explicitly through `train/play -> task_registry.make_env(..., resolved_config=...)` and `make_alg_runner(..., resolved_config=...)`; do not recover identity from globals or a manifest.
3. Change `TaskRegistry.get_cfgs` to return per-run deep copies before setting the seed. Apply the environment projection after allowed CLI/play overrides but before `set_seed`, `parse_sim_params`, and the `task_class(...)` call at current `:97`. Validate protected scientific fields and serialize the effective values at this point. This guarantees reward/config consumers see the profile before `LeggedRobot._parse_cfg` captures scales and before `_prepare_reward_function` applies dt (`legged_robot.py:73-79,796-804,603-613`).
4. In `make_alg_runner`, merge the real `policy_constructor_kwargs` and `ppo_constructor_kwargs` into the per-run `train_cfg_dict` after allowed operational overrides and before `runner_class(...)` at current `:156`. Reject unsupported or conflicting projected keys. `OnPolicyRunner` then consumes those exact dicts at `on_policy_runner.py:83-94`.
5. In `play.py`, record the explicit 40 s local play horizon (or reject it if the selected contract disallows it) in the effective runtime section. Do not serialize it as a formal paper-evaluation horizon.

### Three IsaacLab entrypoints

1. Replace each module-scope parser/launcher block with `build_parser(add_launcher_args)`, `preflight(argv)`, and `main(argv)`. `preflight` uses the real shared parser/resolver plus `validate_runtime_paths` and is importable without IsaacLab. After preflight succeeds, `main` imports `AppLauncher`, adds its arguments, performs the full parse, and only then constructs `AppLauncher`. Put execution/result write/application close under one `if __name__ == "__main__"` guarded `try/finally`.
2. Feed the same resolved object to environment, actor/PPO, reward/perception, replay, and manifest consumers. Smoke must replace manual actor/shield kwargs at current `:260-298`; trainers must replace their manual policy/algorithm dicts and remove post-runner layer replacement at current PPO `:554-606` and ACSI PPO `:1663-1715`. Construction must go through the Task 4 converters/factory so unsupported profile fields fail before `gym.make` or `OnPolicyRunner`.
3. Build and validate the effective environment/reward/perception settings before `parse_env_cfg`/`gym.make`; serialize only after allowed operational overrides. Reconcile against the public pure Task 5 APIs after Task 5 is integrated—do not inspect or copy its in-progress tree.
4. Update `run_full_method_runtime_smoke.sh` so all launcher/run/asset/checkpoint/profile inputs pass the pure preflight before its current `mkdir -p`. The wrapper may report portable preflight success, but only the launched script may emit actual IsaacLab runtime evidence.

## Behavioral acceptance matrix

| Case | Fresh-process executable assertion | AST-only ordering assertion | Honest result |
|---|---|---|---|
| Valid upstream Gym CLI | Real `parse_profile_cli` + resolver + all four projection accessors; separately exercise `build_actor_critic`/`build_ppo` and the pure environment/replay applicators; remainder preserves Gym flags; no Isaac/Omni module loaded | `resolve` precedes delayed Gym import; env projection precedes `task_class`; policy/PPO projection precedes `runner_class` | portable preflight passed; Gym runtime blocked/not executed |
| Valid upstream IsaacLab CLI, each of 3 scripts | Import/call each real pure `preflight`; exact resolved hash and paths; no Isaac/Omni module loaded | full parse precedes `AppLauncher`; preflight precedes both; consumer application precedes `gym.make`/runner | portable preflight passed; IsaacLab runtime blocked/not executed |
| `paper_v1`, missing repair delta, unknown profile/delta, or protected CLI conflict | Real parser/resolver exits nonzero before proprietary import and before output-tree creation | rejection call dominates simulator import/constructor | rejected/blocked, never passed |
| Missing dependency | Actual isolated host probe or injected probe boundary raises `ModuleNotFoundError`; no fake module | no simulator constructor is reachable | dependency blocked; runtime blocked/not attempted |
| Broken installed dependency | Isolated probe boundary raises a non-missing import exception | no simulator constructor is reachable | dependency failed; runtime blocked/not attempted; overall gate failed |
| Dependency import succeeds only | Probe returns package identity/version/path | no runtime pass producer exists in CPU gate | dependency available/passed; runtime remains blocked/not executed |
| Real lifecycle fails after launch | Deferred Rung 3 job only | n/a | dependency available; runtime failed with phase/evidence |
| Real lifecycle completes | Deferred job proves application start, environment create/reset/step/close, exact stack/hash, and physical artifact counts | n/a | runtime passed at the appropriate Gym/Lab rung only |
| Config isolation | Real pure applicator receives two separately materialized configs; mutation of one cannot affect the other | `deepcopy` precedes all mutations | CPU contract passed, not simulator evidence |
| Manifest/trace | Real `bind_runtime_result` checks exact hash/stack/effective values and physical JSONL rows | final binding follows trace close and precedes manifest publication | passed only when status and evidence agree |

Put the fresh-process cases in `tests/test_runtime_cli_contract.py`, path/command cases in `tests/test_runtime_commands.py`, and binding/count/stale-pass cases in `tests/test_runtime_manifest_contract.py`. Extend `tests/test_gate_a_portable.py` for four distinct rows: `isaac_gym_dependency`, `isaac_gym_runtime`, `isaaclab_dependency`, and `isaaclab_runtime`. Dependency rows may isolate the import/discovery callable; runtime rows must never become passed from that callable. Do not import the current simulator-starting entrypoint modules, inject `sys.modules`, create fake Isaac packages, or call `AppLauncher` in CPU tests. After the recommended refactor, importing and executing the deliberately simulator-free `preflight` surface from those modules is the intended fresh-process test.

## Actually executed bounded probes

All commands ran from the inspected repository root and used `../sea-nav-cpu-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, and isolated mode; no simulator entrypoint was imported. The raw command/output transcripts are below.

1. Real resolver plus the four projection **accessors** (not actor/PPO constructors and not environment/replay application):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 ../sea-nav-cpu-venv/bin/python -I -c 'import json,sys; from pathlib import Path; root=Path.cwd(); sys.path.insert(0,str(root/"training/rsl_rl")); from rsl_rl.experiment_config import resolve_run_config,build_policy_kwargs,build_ppo_kwargs,build_env_profile_values,build_replay_kwargs; rows={};
for stack in ("isaac_gym_preview4","isaaclab_adapter"):
 c=resolve_run_config(registry_path=root/"configs/parity_registry.yaml",algorithm_profile="upstream_fbce672c",runtime_stack=stack,implementation_delta=("ppo_state_identity_repair",)); rows[stack]={"sha":c.resolved_sha256,"consumers":[p.consumer for p in (build_policy_kwargs(c),build_ppo_kwargs(c),build_env_profile_values(c),build_replay_kwargs(c))],"statuses":[p.application_status for p in c.application_projections]}
assert not any(n.startswith(("isaacgym","isaaclab","omni")) for n in sys.modules); print(json.dumps(rows,sort_keys=True))'
```

```text
{"isaac_gym_preview4": {"consumers": ["policy_factory", "ppo_constructor", "environment_profile", "replay_reset"], "sha": "6d90936678c80a531ef425b9b994fc4a0f1b610dc976946cc3919d364806953e", "statuses": ["future_task_4_and_6", "future_task_3_and_4", "future_task_5_and_6", "future_task_5"]}, "isaaclab_adapter": {"consumers": ["policy_factory", "ppo_constructor", "environment_profile", "replay_reset"], "sha": "a93012f2a13b409975dbec24ae36a801ec113ae9f86d8fb036352abe8d02823a", "statuses": ["future_task_4_and_6", "future_task_3_and_4", "future_task_5_and_6", "future_task_5"]}}
```

This proves real profile resolution, integrity checking, and projection access without simulator imports. It does **not** prove `build_actor_critic`, `build_ppo`, environment/replay applicators, or runtime application; all returned projections still advertise `future_task_*` statuses.

2. Dependency discovery only:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 ../sea-nav-cpu-venv/bin/python -I -c 'import importlib.util,json; print(json.dumps({name:(importlib.util.find_spec(name) is not None) for name in ("isaacgym","isaaclab")},sort_keys=True))'
```

```text
{"isaacgym": false, "isaaclab": false}
```

This is dependency discovery evidence only; both real runtime stacks remain blocked.

3. No-import AST ordering probe:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 ../sea-nav-cpu-venv/bin/python -I -c 'import ast,json; from pathlib import Path; root=Path.cwd(); rows={};
for rel in ("sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py","sea_nav_current_isaaclab_full_method/train_full_method_ppo.py","sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py"):
 t=ast.parse((root/rel).read_text()); main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="main"); app_import=next(n.lineno for n in t.body if isinstance(n,ast.ImportFrom) and n.module=="isaaclab.app"); launch=next(n.lineno for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="AppLauncher"); rows[rel]={"isaaclab_import":app_import,"app_launch":launch,"main":main.lineno}
print(json.dumps(rows,sort_keys=True))'
```

```text
{"sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py": {"app_launch": 52, "isaaclab_import": 9, "main": 118}, "sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py": {"app_launch": 79, "isaaclab_import": 10, "main": 1359}, "sea_nav_current_isaaclab_full_method/train_full_method_ppo.py": {"app_launch": 31, "isaaclab_import": 10, "main": 450}}
```

This proves the current ordering defect; it is not runtime evidence.

## Recommendation versus evidence ceiling

Everything under “Minimal interfaces”, “Exact startup ordering”, and the acceptance matrix is a Task 6 recommendation. The only executed evidence is the three probes above plus direct source inspection at the named commit. No Gym/IsaacLab runtime, environment construction, controller, reset, checkpoint, metric, or accepted paper result was verified. A dependency being installed or importable must never raise that evidence ceiling.
