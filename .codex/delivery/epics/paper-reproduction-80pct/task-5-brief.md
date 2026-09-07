## Binding execution corrections (2026-09-07)

Read the updated design and the named audit report for this task. These corrections supersede conflicting earlier examples. Use the dedicated CPU interpreter at `../sea-nav-cpu-venv/bin/python`; system Torch 1.8 cannot validate these contracts. Keep training/shared code Python-3.8-compatible. No simulator stubs or claims above CPU/static Rung 2.

Read scientific-wiring-audit.md. Add a pure shared rsl_rl curriculum/replay module if needed to keep Gym independent of IsaacLab. Implement literal Eq.1 as a tested mathematical contract and upstream level/decision semantics with explicit masks and separate two Bernoulli stages. Accepted paper_v1 stays blocked by config. Implement adapter level update (currently constant) and exclude success/timeout replays. partition_reset_env_ids must expose all three normal/replay/fallback sets, not just two for already-requested replay rows. Actual root/DOF/history/filter/controller snapshots need coherent schemas and derived-state reconstruction; CPU fixtures test pure policy, real simulator write/refresh remains blocked. Curriculum learning state is cross-episode and must not be rewound with physical snapshots. Reserve/cancel/ack must retain epoch/token identity and report short-history clamp. Include O(active-envs) push regression and no device sync in hot path. Task 6 owns reward/delay consumers but coordinate replay snapshot fields for them.

### Replay schema and lifecycle ruling

Read `replay-schema-audit.md` in full. Use its compact `new_replay_episode_v1` policy and complete reconstruction table. Add `training/legged_gym/legged_gym/envs/base/legged_robot.py` and the adapter filter's public masked reset to owned paths, plus pure helpers under `rsl_rl/replay/`. Record `replay_reset_reconstruction_v1` explicitly in repaired run deltas and the selected policy; never silently select a full-history mode or advertise exact continuation. Snapshot root/DOFs/task geometry at the terminal accounting boundary before reset, with explicit step/episode/task-generation IDs. No slot contains entire observation histories.

The transaction ends only after masked observation reconstruction AND base post-reset epilogue. Acknowledge replay last. Normal, committed replay, and fallback sets cover exactly the requested IDs. Preserve the old reward/done/timeout transition and cross-episode learning state. On write/reconstruction/epilogue failure, complete a normal reset before cancelling; if fallback fails propagate without observation or ack. Add a pure lifecycle harness for all four failure points and AST wiring checks; no simulated physics result is inferred from it.

For the CPU hot-path test, read `replay-performance-harness.md`, especially its corrected capacity-relative recommendation. Use scoped TorchDispatchMode operator evidence with positive controls for selected-row history copies, full-buffer materializers and scalar extraction; a view's large metadata is not a copy. Compare fixed active-row/payload work across capacities and account for bounded per-row metadata writes. Do not use only a whole-storage-size threshold (it misses `storage[env_ids].clone()`), payload-only exact write counts, or wall-clock thresholds. Inventory the actual emitted operators and use a sufficiently large inactive population or a second environment-count comparison to substantiate O(active rows), not merely capacity independence. This is CPU semantic evidence, not CUDA performance or simulator profiling.

Rebuild velocities/gravity, geometry/rays, zero commands/previous actions, restored velocity baselines, repeated fresh SLR/navigation/ray/goal/position histories, timers and per-row bootstrap. No terminal-state data may leak into reset rows, and untouched neighbors must remain unchanged. Do not invoke mutating termination/reward callbacks for reconstruction. Validate scene/controller/frame context; unsupported carrier auto-reset/cache/contact behavior is a runtime blocker, not a guessed success. Task 6 owns current-goal reward timing and timestamped real perception consumption; coordinate their pure helper interfaces. Budget compact ring push by active rows and fixed payload width, without device synchronization or full-buffer shifts.

## Global Constraints

- Working branches must remain exactly `main`, `stable`, and `test`; all repair commits land only on `test`.
- Integrate the dependency graph strictly as `1→2→3→4→5→6→7`; do not create concurrent detached commits touching PPO/CBF/runner/reset/trainer files.
- `main` and `stable` remain at `1c5675bbedf1dcbe5a4c1a91830cae528c780793` during this recovery.
- `paper_v1` uses Eq. 4 with `epsilon_d=1.0` and describes the result as a damped safety bias, never a hard-safe projection.
- Action stages are `distribution_mean`, `policy_action`, `clipped_policy_action`, and `executed_command`.
- Every repaired run explicitly records `implementation_delta=["ppo_state_identity_repair"]`; the resolver never adds it silently.
- CPU tests may import `rsl_rl`; `legged_gym` receives syntax/AST/tree checks only because importing its task stack requires unavailable `isaacgym`.
- Missing Isaac Gym/IsaacLab, formal 100-trial metrics, publication rights, and real hardware are `blocked`, never mocked into a pass.
- Use exact-path staging. Each task ends with an independently reviewable commit and a clean focused test run.

---

### Task 5: Make collision replay a CPU-tested per-environment state machine

**Files:**
- Create: `tests/test_collision_replay_cpu.py`
- Create: `tests/test_replay_reset_partition.py`
- Modify: `sea_nav_current_isaaclab_full_method/adapters/collision_replay.py`
- Modify: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py`
- Modify: `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`
- Modify: `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

**Interfaces:**
- Add immutable `ReplayTensorSpec`, `ReplayBatch`, and `ReplaySelection` value objects.
- Add `validate_replay_config`, `partition_reset_env_ids`, `reserve_pre_collision`, `acknowledge_restore`, and `cancel_restore`.
- Preserve `legged_robot_pos.reset_idx(env_ids)` and adapter `reset(env_ids=None, replay_sample=None)` public signatures.

- [ ] **Step 1: Add replay state-machine tests**

```python
def test_capacity_covers_inclusive_undo_maximum():
    with pytest.raises(ValueError, match="ring_buffer_steps"):
        CollisionReplayBuffer(
            CollisionReplayConfig(ring_buffer_steps=150, undo_steps_range=(100, 150)),
            num_envs=2,
        )


def test_reset_partition_is_disjoint_and_complete():
    env_ids = torch.tensor([2, 4, 7])
    replay_ids, fallback_ids = partition_reset_env_ids(env_ids, torch.tensor([True, False, True]))
    assert replay_ids.tolist() == [2, 7]
    assert fallback_ids.tolist() == [4]
    assert sorted(replay_ids.tolist() + fallback_ids.tolist()) == env_ids.tolist()
```

Create deterministic fixtures that push distinguishable records for two environments through more than one wrap. Assert independent write positions, episode isolation, explicit short-history clamping, reachability of undo 100 and 150, reservation without consumption, committed-row consumption, cancelled-row retryability, and preservation of restored history/filter/task tensors.

- [ ] **Step 2: Prove the replay tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_collision_replay_cpu.py tests/test_replay_reset_partition.py`

Expected: new validation/reservation/partition APIs are missing and the old capacity silently caps the range.

- [ ] **Step 3: Implement O(selected-envs) ring writes and transactional sampling**

Preallocate tensors `[num_envs, capacity, *field_shape]`. Track per-environment write index, valid length, episode ID, collision onset, and reservation token. Sample inclusive undo values with `high=max_undo + 1`; clamp only against explicitly recorded short history. `reserve_pre_collision` does not consume metadata. `acknowledge_restore` consumes only committed rows; `cancel_restore` records a reason and keeps an eligible row retryable. The per-step `push` path contains no `.item()`, `stack`, `cat`, or full-buffer `where`.

- [ ] **Step 4: Wire the reset partition without claiming simulator proof**

Partition requested IDs before reset mutations. Apply normal resets only to `normal_ids` and `fallback_ids`; preserve or reconstruct replay-owned episode length, observation/ray/position/goal histories, delay/filter/controller state, task timers, and collision metadata for committed `replay_ids`. Recompute root-derived values after physical commit and before first observation. Mirror the policy in the IsaacLab adapter, but keep actual carrier writes behind its runtime boundary.

- [ ] **Step 5: Run CPU/static tests and record runtime blockers**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_collision_replay_cpu.py tests/test_replay_reset_partition.py
PYTHONDONTWRITEBYTECODE=1 python3 sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile training/legged_gym/legged_gym/envs/base/legged_robot_pos.py
```

Expected: CPU/static tests pass. Actual Gym root/DOF writes, refresh, and multi-env first-observation consistency remain Rung 3G `blocked`; IsaacLab carrier reset remains Rung 3L `blocked`.

- [ ] **Step 6: Commit Task 5**

```bash
git add tests/test_collision_replay_cpu.py tests/test_replay_reset_partition.py sea_nav_current_isaaclab_full_method/adapters/collision_replay.py training/legged_gym/legged_gym/envs/base/legged_robot_pos.py sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
git commit -m "feat: make collision replay reset-safe"
```

---
