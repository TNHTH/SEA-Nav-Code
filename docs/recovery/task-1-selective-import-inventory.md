# Selective Later-Snapshot Import Inventory v1

Task 1 starts from the complete tree at `1259bae1b2e2635e410a315ab6bf92451df8762b`. The later sparse snapshots `origin/test@92896ba1b39087fc8cad633001a4d0dc3f313623` and `b53d3feb98a287b5888a18e3dd67aeb530664fe0` are evidence sources only; neither tree is imported wholesale.

| Candidate paths reviewed | Task 1 disposition |
|---|---|
| `training/rsl_rl/rsl_rl/runners/on_policy_runner.py` | Later performance, TensorBoard, and checkpoint changes are deferred to their owning batches. Task 1 independently adds only the tested lazy optional-W&B boundary. |
| `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`, `sea_nav_current_isaaclab_full_method/adapter_manifest.json` | Later files mix machine-bound paths with replay/training changes. No blob import; Task 1 independently ports preview output and manifest examples. |
| `sea_nav_current_isaaclab_full_method/adapters/collision_replay.py`, `adapters/command_delay.py`, `full_method_runtime_smoke.py`, `train_full_method_acsi_replay_ppo.py`, `train_full_method_ppo.py` | Deferred to the replay/runtime batches; no Task 1 import. |
| `training/legged_gym/legged_gym/utils/grid2ray.py` | Deferred to runtime recovery; no Task 1 import. The complete baseline package and Go2 assets are retained and gated. |
| `training/rsl_rl/rsl_rl/algorithms/ppo.py`, `modules/actor_critic.py`, `modules/cbf_actor_critic.py` | Deferred to PPO/state-identity work; no Task 1 import. |
| `sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py` | Deferred to checkpoint/runtime work; no Task 1 import. |

Rejected snapshot-wide changes include deletion of the complete package/assets, the machine-specific repository guidance mutation, and machine-bound live-test scripts. Task 1 reimplements only its scoped behavior behind repository-relative paths or explicit launcher/run-directory inputs.
