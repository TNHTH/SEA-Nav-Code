# Task 7 fix2 current truth

| Field | Current value | Evidence / limitation |
|---|---|---|
| Fixed version | Functional fix2 commit containing this record, based on `3b6e97d0004eac6ff3f04be4e7b56acb693db414` | Detached batch7-v2; independent rereview pending |
| Active evidence | Focused182 passed/2 skipped; full529 passed/2 skipped; Gate A five pass/four blocked | Fresh fix2 precommit commands below; historical fix1 receipt is not fix2 acceptance |
| Confirmed repairs | No disposable optimizer execution; bounded tensor/gradient/optimizer/metadata rollback; RNG and gradients preserved on both exits | Real native-v2 consumers, actual PPO continuation, hook faults |
| Configuration | Torch 2.6.0+cpu isolated venv | Real CUDA/simulator unavailable |
| Highest rung | CPU/static only | Isaac Gym/Lab, formal metrics, hardware, rights deferred |
| Delivery scope | Two production files, one existing regression file, three local ledger files | Original root registration files and all their old content remain excluded from commit |

Fix1 disposable-probe safety and complete rollback claims are superseded historical assertions. Native target load hooks remain legitimate application behavior; arbitrary external hook side effects are explicitly not rollback-guaranteed.

Fresh tests-first RED on unchanged production3b6: 33 failed/44 passed2.71s. All new failing cases reproduce the specified defects: epsilon/beta-zero rejection, dummy Adam construction/step, global hooks, gradient presence/value (four consumer modes), runner optimizer/defaults/metadata, CPU plus simulated two-device initialized-CUDA RNG. Backward/zero_grad prohibitions already pass. Nonpersistent model buffers are included in the strengthened rollback contract. RNG tests cover successful and failing legitimate load hooks; absent CUDA must never be initialized by bookkeeping.

Pure validation uses native metadata-only fused/device support checks (no dummy constructor/load/step). Legal zero beta/epsilon, foreach, fused, fused+capturable CPU, maximize/weight_decay and existing AMSGrad checkpoints all complete actual PPO continuation. Existing geometry identity, model compatibility, payload-byte receipts, lazy model-only import and converter absence remain unchanged.

Transaction scope: existing model parameter/buffer values (including nonpersistent buffers) on failure; model/optimizer parameter-union gradients and caller CPU/initialized-CUDA RNG on all exits; supplied optimizer state/groups/defaults and runner iteration/algorithm LR on failure. Only resume loads optimizer payload/iteration/LR. External hook lists, replaced module structure, process/device failure, producer RNG and physical trajectories are explicitly not rollback claims.

Gate A receipt `/tmp/sea-nav-task7-fix2.GM5pcX/gate-a-precommit.json`:90 tracked Python syntax,65 baseline files,20 static package files/9 simulator-only,7 CPU package imports, actual actor/value/PPO/storage smoke pass; IsaacGym/Lab dependency/runtime remain four blockers. Temporary JSON is an evidence receipt, not a durable recovery point. Two real CUDA RNG tests skip on this CPU-only host; all16 CPU/simulated dual-device success/failure/mode RNG cases execute.
