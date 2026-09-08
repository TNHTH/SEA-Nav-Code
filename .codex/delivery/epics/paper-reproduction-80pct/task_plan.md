# Task Plan: SEA-Nav reproduction recovery

## Goal
Complete Batches 1–7 on `test`, verify Rungs 0–2 from a frozen clean commit, publish the authorized three-branch remote state safely, and provide a separately packaged pure-Torch differential-drive SEA adaptation for the DashGo ablation consumer.

## Current Phase
Phase 3/6/7 — review and integrate Task 7, review and integrate the differential-drive core, then implement the DashGo consumer serially

## Acceptance Criteria
- Local working branches remain exactly `main`, `stable`, and `test`; no temporary named branches are created.
- Batches land on `test` strictly in order `1→2→3→4→5→6→7`, each with red/green evidence, review, and a focused commit.
- `test` remains descended from `main@1c5675b` and retains the complete baseline tree.
- Rungs 0–2 pass from a clean detached checkout of the frozen candidate OID; Isaac Gym and IsaacLab remain explicit `blocked` results when absent.
- Remote writes use the exact leases recorded in the approved spec, are read back after every mutation, and never push `upstream/11chens-fbce672c`.
- Every locally accepted checkpoint is committed and pushed to `origin/test` after a committed-tree secret scan; `main` and `stable` remain unchanged without the simulator/deployment evidence required for promotion.
- DashGo integration keeps `algorithm_profile` and `runtime_stack` independent, preserves the 246-D policy-observation contract, supplies a separate metric `safety` observation group, and records the pinned SEA-Nav core commit in every run/export manifest.
- Ablations are configuration-only variants of one implementation: `full`, `without_acsi`, `without_shield`, and `without_lreg`; no profile may silently change observation/action geometry, normalization, seeds, episode budgets, or evaluation rules.

## Phases

### Phase 1: Requirements, evidence, and written contract
- [x] Preserve all three unrelated historical root snapshots with local recovery refs.
- [x] Rebuild local `test` from complete `main` and retain only the three requested local working branches.
- [x] Repair and commit the written recovery contract and portable repository guidance.
- [x] Map Batches 1–7 to concrete files, interfaces, tests, and blocked runtime boundaries.
- **Status:** complete

### Phase 2: Executable implementation plan
- [x] Save `docs/superpowers/plans/2026-09-04-sea-nav-reproduction-recovery.md` with exact TDD steps and commit boundaries.
- [x] Run spec-coverage, placeholder, and interface-consistency self-review.
- [x] Commit the implementation plan and this coordination state on `test` (`1259bae`).
- **Status:** complete

### Phase 3: Strict serial implementation
- [x] Batch 1 — portable Gate A and lazy optional `wandb` (`484f682`, reviewed, 31 CPU tests).
- [x] Batch 2 — parity registry, profiles, typed resolved config, and two-axis identity (`6f5e544`, reviewed, 71 CPU tests).
- [x] Batch 3 — PPO action/likelihood and auxiliary-state identity (`37f0763`, reviewed, 85 CPU tests).
- [x] Batch 4 — paper-damped CBF semantics, diagnostics, and golden vectors (`a660d74`, independent spec/quality PASS, 169 CPU tests).
- [x] Batch 5 — replay ring/reservation/reset partition and CPU policy evidence (`97252ed`, `1ae9187`; scoped review PASS, fresh integrated 212 CPU tests).
- [x] Batch 6 — scientific runtime consumers, checked startup/evidence, corrected filter and opt-in training trace (`dbc609d`, `b52808d`, `4183d2b`, `b5b9455`; scoped Spec/Quality PASS; fresh integrated342 CPU tests).
- [x] Batch 7 — checkpoint schema v2, safe loading, and explicit class registries (`a1de6e3`, `caf0f25`, `406a6ec`; integrated 529 passed/2 real-CUDA skips and Gate A five passed/four dependency blockers).
- **Status:** complete on `test`. The old `2387cf0→c7b9aa3→774027d` chain remains only under an explicitly unaccepted checkpoint tag; `92ab65d→3b6e97d→64e2706` supplied the reviewed v2-only implementation and two corrective rounds.

### Phase 4: Frozen-candidate verification
- [ ] Freeze one candidate OID and create a clean detached verification checkout.
- [ ] Run Rung 0 tree/ref/syntax/asset checks and Rung 1 CPU behavior/security tests.
- [ ] Run Rung 2 Gate A with reports outside the checkout; verify no dirty output.
- [ ] Record Isaac Gym, IsaacLab, formal metrics, licensing, and hardware as `blocked`/deferred where prerequisites are absent.
- [ ] Obtain independent final code review and resolve every P1/P2 finding.
- **Status:** pending

### Phase 5: Remote transaction and delivery
- [x] Fetch and re-check exact remote head leases and authenticated destination identity.
- [ ] Scan the frozen committed tree for credentials/secrets and record the candidate OID.
- [x] Push authorized recovery tags; read back exact objects before any branch deletion.
- [x] Create `stable@1c5675b` as an unqualified bootstrap.
- [x] Force-update only `origin/test` from exact old `92896ba…` to checkpoint `bf6e4eb…` and read back.
- [x] Delete only the long historical branch with its full expected-old SHA after recovery refs are remote-visible.
- [x] Verify remote working branches are exactly `main`, `stable`, and `test` at the published Tasks 1–6 checkpoint.
- [ ] Fast-forward each subsequently accepted repair slice to `test`; do not advance `stable` without real Isaac smoke/short-train evidence.
- **Status:** checkpoint publication complete; final repaired candidate publication remains pending

### Phase 6: Differential-drive public core
- [x] Build a standalone `sea_nav_core` candidate with versioned platform/observation/action contracts and a pure-Torch unicycle-lookahead damped LSE-CBF (`bf409c65`; detached candidate only).
- [x] Preserve the existing Go2 `ExactLSECBFLayer` behavior and label the new layer `cross_platform_method_adaptation`, never `paper-exact`.
- [x] Add golden output/gradient/invalid-input/TorchScript tests and an ablation profile resolver for `full`, `without_acsi`, `without_shield`, and `without_lreg`.
- [ ] Publish the accepted package commit to `test`; DashGo pins that full commit and records both repository identities in every run manifest.
- **Status:** fixed candidate `2cd8105` passed independent source/CPU and artifact review; primary ordered integration and publication are pending

### Phase 7: DashGo differential-drive consumer
- [ ] Adapt the fixed Isaac Lab v2.0.2 wrapper contract to the pinned RSL-RL v3.0.1 TensorDict API without importing Isaac in CPU tests.
- [ ] Add pure-query bounded-policy/CBF interfaces, next-observation storage, terminal masks, and normalization that leaves the metric safety group raw.
- [ ] Add the 72-range/angle/validity/age safety group and replay-aware environment reset transaction, including scan-cache invalidation.
- [ ] Inject the four named ablation profiles into train/play/export/evaluation, bind manifests to both repository commits, and prohibit replay during formal evaluation.
- [ ] Repair terminal-state evaluation snapshots and scale the formal matrix to seeds 42/43/44 with 100 episodes per difficulty after the 100k/5M gates.
- [ ] Keep ROS 2 Humble and real-robot `/cmd_vel` paths unchanged until the simulation/export contract is accepted; then add only offline/launch/config validation with hardware execution explicitly blocked.
- [ ] Commit, review, test, and push each accepted thin slice on DashGo `test`; do not advance DashGo `main` or `stable` without their own promotion evidence.
- **Status:** exact implementation mapping under independent read-only audit; no DashGo source adaptation integrated yet

## Worktree Registration

| Worktree | Branch/commit | Owner | Owned paths | Dependencies | Status |
|---|---|---|---|---|---|
| `work/SEA-Nav-Code` | `test@169aa1f7448ae3900a0dbcd3eb99595177d6003e` plus current coordination diff | controller | plan/coordination now; differential-drive commits integrated serially | approved spec and two independent D0 reviews | active, Tasks1–7 integrated; primary/remote read back at same commit before D0 integration |
| `work/SEA-Nav-Code-latest-review` | detached `b53d3fe` | read-only source audit | none | none | clean/read-only |
| `work/SEA-Nav-Code-batch1` | detached `4dec41b` | task1 implementer | Task 1 exact paths | reviewed and integrated as `a58ad21`, `eac2657`, `484f682` | finished, clean and retained; historical worker registration preserved in task-1-registration-history.md |
| `work/SEA-Nav-Code-batch2` | detached `3359215` | `/root/implement_batch2` | Task 2 exact paths | reviewed and integrated as `ed8e7e7`, `6f5e544` | finished, clean and retained; local registration preserved in task-2-registration-history.md |
| `work/SEA-Nav-Code-batch3` | detached `9e91c72` | `/root/implement_batch3` | Task 3 exact paths | reviewed and integrated as `37f0763` | finished, porcelain clean and retained; registration preserved in task-3-registration-history.md; ignored verification caches are not final-candidate evidence |
| `work/SEA-Nav-Code-batch4` | detached `ac0567e` | `/root/implement_batch4` | Task 4 registered paths | reviewed and integrated as `a660d74` | finished, porcelain clean and retained; registration preserved in task-4-registration-history.md |
| `work/SEA-Nav-Code-batch5` | detached `a7fe32ecbff4aae420dd073a8e2a032d941a0a10` | `/root/implement_batch5` | Task 5 registered replay/reset/ACSI paths | independently reviewed; integrated as `97252ed`, `1ae9187`; primary 212 passed | finished, porcelain clean and retained; registration preserved in task-5-registration-history.md |
| `work/SEA-Nav-Code-batch6` | detached `0efc1934cbefc1a9780baced7846484f29eb776e` | `/root/implement_batch6` | Task6 registered paths and two narrow extensions | reviewed; four ordered commits integrated through b5b9455; primary342 passed | complete; clean including ignored, registration preserved in task-6-registration-history.md, all commits/worktree retained |
| `work/SEA-Nav-Code-batch6-review-slice1` | detached `7945a9e6037fee5045cbc12ac59a054ad958b43b` (previous fixed slice `6e097ac`) | `/root/review_batch6_slice1` | no source writes; only primary coordination task-6-slice1-review.md and scoped rereview report | fixed Task 6 clock correction; subsequent Task 6 source remains separate | initial one-P2 review complete; scoped fix re-review, not Task 6 whole integration acceptance |
| `work/SEA-Nav-Code-batch6-review-full` | detached `844514929726a1cade3303867cc11702b710a04d`; BASE `65dcbbc2b13c92af99a9e7d4cba4110880f9b509` | `/root/review_batch6_full` | only primary coordination task-6-review.md and scoped rereview reports; no source/ref writes | corrected Task 6 brief (29 paths plus two narrow extensions); full fixed diff and final worker report | whole Spec/Quality FAIL, 1 P1 and 4 P2; review tree retained clean; original writer fix round 1/5, no integration |
| `work/SEA-Nav-Code-batch6-review-fix1` | detached `0efc1934cbefc1a9780baced7846484f29eb776e`; fix BASE `844514929726a1cade3303867cc11702b710a04d` | `/root/review_batch6_full` | only primary task-6-rereview-1.md; no source/ref writes | complete 11-file fix diff and worker postcommit addendum | complete, all5 findings closed; Spec/Quality PASS; clean and retained |
| `work/SEA-Nav-Code-batch7` | detached `774027d1a975e318ad2f577b457951f51c82bfad`; BASE `562d4ae0b56ca977ea433def6dbd05607284e38b`; fix BASE c7b9aa3 | `/root/implement_batch7` | exact26 inventory; no further writes active; two local registration diffs retained | fixed postcommit433/GateA report read; caller P2 independently closed; full core/converter review incomplete | writer idle, source/log committed and retained, no integration |
| `work/SEA-Nav-Code-batch7-v2` | detached `64e270628eb290e27b94167c5b34e538003be03d`; BASE candidate `92ab65d`, fix1 `3b6e97d`, project BASE `399ce2b` | `/root/review_batch7_callers` fix2 writer | Task7 application/checkpoint tests and local fix ledgers only; two worker registration diffs retained | fix2 independently read and retested by controller | complete and integrated through `test@406a6ec`; unreviewed fix1 tag retained only as recovery evidence |
| `work/SEA-Nav-Code-diffdrive-core` | detached `2cd810569008fa923bda088f0f0988292e0c809c`; BASE `399ce2b08eac40865fd6496d19324f73a3e6cc7e` | `/root/dashgo_adapter_audit/diffdrive_core_fix` | only `packages/sea_nav_core/**` and its existing focused tests/docs; retain two local registration diffs outside the functional commit | independent source/CPU PASS in `diffdrive-core-rereview-2.md`; artifact PASS in `diffdrive-core-artifact-review-2.md` | accepted candidate tagged; no overlap with Task 7, awaiting ordered `test` integration |
| `work/dashgo-rl-navigation` | `test@53bd0f0ab0100e295aed1f726c2c3c9f4ffb3b06` | controller | coordination/docs and serial integration only | accepted SEA-Nav differential-drive core commit | clean and synchronized with `origin/test`; D1 preflight published |
| `work/dashgo-rl-navigation-sea-adapter` | detached `78023cb41984a6af98b9b31b383aed2609beb47e` | future sole DashGo adapter writer | `src/dashgo_rl/sea_nav/**`, focused env/train/eval/export/config/test callers; ROS 2 source excluded initially | current DashGo `test`, accepted integrated SEA-Nav core OID, completed audit mapping | retained for the next thin implementation slice; exact base/status must be rechecked before edits |
| `work/SEA-Nav-Code-batch7-review-callers-fix1` | detached `774027d1a975e318ad2f577b457951f51c82bfad`; fix BASE `c7b9aa371b8ab3800adea378a7024f043fb58580` | original `/root/review_batch7_callers` | completed primary task-7-callers-rereview-1.md only; no further writes | original terrain P2 ADDRESSED, bounded Spec/Quality PASS; independent119 passed/1 deselected35.89s | complete, retained clean including ignored; not full Task7 acceptance |
| `work/SEA-Nav-Code-batch7-review-core` | detached `2387cf02d85a1a9a52b0e85da54c7ebad09b0cec`; BASE `562d4ae0b56ca977ea433def6dbd05607284e38b` | `/root/review_batch7_core`; controller owns only bounded functional continuation | controller may write primary task-7-core-functional-review.md and coordination failure record; no source/ref writes | fixed eight-file core snapshot and log; non-adversarial runner/persistence scope only | original reviewer errored at service safety boundary, no report/verdict; full review incomplete, no bypass or second writer |
| `work/SEA-Nav-Code-batch7-review-callers` | detached `c7b9aa371b8ab3800adea378a7024f043fb58580`; BASE `562d4ae0b56ca977ea433def6dbd05607284e38b` | `/root/review_batch7_callers`; controller read-only ordinary core follow-through | only primary task-7-callers-review.md and controller functional-report addendum; no source/ref writes | fixed caller/README diff and final worker report when available | bounded ordinary caller/documentation review, explicitly excludes converter/security review; original complete core review remains incomplete |
| Primary read-only CBF hotpath review | source through Task 5 | `/root/cbf_hotpath_review` | only coordination cbf-hotpath-review.md | unchanged CBF source since Task 4 | completed with one open final-review P2; no source mutation |
| Primary read-only CBF checked-path preparation | fixed source `12e4fff0714f5753e30bf2575e740dc12a32c051`, unchanged at coordination successor `c3db30d` | `/root/cbf_hotpath_review` | only coordination cbf-checked-path-probe.md | prior open CBF P2; Task 6 remains sole source writer | completed bounded in-memory CPU prototype, valid extraction 5→1; no production fix or acceptance |
| Primary read-only publication preflight | `test` | `/root/publication_preflight` | only coordination publication-preflight.md | no source dependency | read-only remote inspection; no Git/remote mutation |
| Primary read-only replay performance preparation | `test` | `/root/replay_perf_probe` | only coordination replay-performance-harness.md | no implementation dependency | synthetic CPU instrumentation probe; no production or simulator changes |
| Primary read-only runtime preflight preparation | `test@5b99f45` | `/root/runtime_preflight_probe` | only coordination runtime-preflight-harness.md | no Task 5 implementation dependency | bounded parser/dependency-boundary CPU test preparation; no source or simulator changes |
| Primary read-only reward oracle preparation | source through Task 4 | `/root/reward_cpu_oracles` | only coordination reward-cpu-oracles.md | no Task 5 implementation dependency | bounded independent scalar expectations and Torch bool semantics; no source/test/simulator changes |

Task 4 scope extension: only package-bootstrap root/import-order blocks in the three existing adapter entrypoints (`full_method_runtime_smoke.py`, `train_full_method_ppo.py`, `train_full_method_acsi_replay_ppo.py`) and a focused `tests/test_cbf_import_boundary.py`, to retain a single packaged CBF module identity. No runtime profile/CLI/reset behavior is assigned early. Register this extension in the worker before edits; exact rationale is in task-4-brief.md/sdd-progress.md.

## Task 6 exact scope registration

Narrow additional source path: `sea_nav_current_isaaclab_full_method/adapters/command_delay.py`, only alpha state/output separation to match Gym's retained un-clipped filter state; tests/consumer updates stay in already-owned paths. Preserve Task 5 masked reset/mirrors. Exact evidence and constraints in task-6-brief.md. This extends the 29-path inventory by one source file (plus the previously approved narrow historical YAML test migration), not by a parallel implementation batch.

Narrow additional test path: only the historical adapter-YAML assertions in `tests/test_experiment_config.py::test_committed_adapter_examples_are_identity_bound_and_not_original_claims`; migrate to real loader/explicit-delta behavior after executable YAML conversion. The adjacent historical manifest assertions and all unrelated integrity tests remain unchanged. Exact rationale in Task 6 brief.

Task6 historical scope is recorded in task-6-brief.md and preserved worker registration history. Final worker0efc193 independently accepted and integrated through primaryb5b9455; fresh342 tests and GateA verified. All five whole-review findings closed; physics/runtime and accepted-paper boundaries remain blocked. Worker source/log retained and two exact registration files restored after lossless diff preservation; no broader cleanup.

## Task 7 exact scope registration

Owner `/root/implement_batch7`; detached BASE will be the coordination-only successor of b5b9455. The explicit26-path inventory in task-7-brief.md covers checkpoint core/registry/runner, every Gym/adapter/initializer manifest caller, isolated operator tool, focused tests and the three existing README usage sections. Preserve Task6 actual shape/reset/trace/seed/runtime boundaries; no CBF/PPO/replay-core or deferred evaluator/CI/deployment ownership. Only worker task-7-log may be committed from its local coordination changes; registration remains uncommitted and primary task-7-report is its sole primary write exception. Must verify clean fresh342 baseline before editing; original checkpoint-contract-audit and real runner/sandbox capability reports are preparation, not Task7 acceptance.

## Decisions Made

| Decision | Rationale |
|---|---|
| Use `TNHTH <174231229+TNHTH@users.noreply.github.com>` at repository scope | Matches historical GitHub attribution selected for the two existing recovery commits. |
| Keep `main` and `stable` unchanged this round | The user required repairs on `test`; simulator/deployment promotion gates are unavailable. |
| Treat the latest sparse commit as a semantic import source only | It omits 53 complete-tree paths and has unrelated history. |
| Serialize all seven implementation batches | PPO/CBF/runner and replay/runtime/checkpoint files overlap; path ownership cannot make stale-parent commits safe. |
| Run CPU-only imports for `rsl_rl`; inspect `legged_gym` statically | `legged_gym` imports proprietary `isaacgym` at module load in this environment. |
| Bind any force-push to a frozen candidate OID and explicit expected-old remote SHA | Prevents both local-moving-ref and remote-race ambiguity. |
| Remove the arbitrary legacy checkpoint converter from the accepted Task 7 contract | The user selected native v2 checkpoints plus explicit old-format rejection; executable legacy pickle conversion is not required for reproduction. |
| Package DashGo support outside vendored `rsl_rl` | Avoids a package-name/version collision with DashGo's RSL-RL 3.0.1 while keeping the algorithm dependency one-way. |
| Use paper-v1 scientific losses with a named differential-drive adaptation | The user selected paper-v1 as the formal ablation basis; the kinematic change must remain explicit and cannot be relabeled as original Go2 reproduction. |

## Errors Encountered

| Error | Resolution |
|---|---|
| HTTPS origin previously lacked usable interactive credentials | Preserve remote state; later use an explicitly verified SSH destination only within the approved transaction. |
| System Torch import has a NumPy ABI conflict | Use a task-scoped isolated dependency path for CPU verification and record the exact environment; do not modify system packages. |
