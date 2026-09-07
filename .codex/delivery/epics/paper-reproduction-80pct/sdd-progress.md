# SDD ledger — plan: docs/superpowers/plans/2026-09-04-sea-nav-reproduction-recovery.md

## 2026-09-07 recovery

Verified `test@1259bae1b2e2635e410a315ab6bf92451df8762b`, clean; Tasks 1–7 are not implemented. This ledger is inside the AGENTS-mandated coordination directory, overriding the skill's default scratch path.

| Pair/task | Producer / consumer or internal check | Resolution |
|---|---|---|
| 1→2 | portable manifest / identity-bound manifest | preserve compatible callers until Task 2 wiring |
| 1→5→6→7 | legacy Gate A and new CPU gate | serial updates and inherited regression suite |
| 1→7 | runner lazy logging / checkpoint registry | serial, no concurrent runner writes |
| 2→3→4 | profile selection / pure action means / CBF geometry | explicit delta and configuration passed through to consumers |
| 2→5→6→7 | typed config / replay policy / entry points / hash-bound checkpoints | configuration must be applied, not merely serialized |
| 3→4→7 | actor, PPO / diagnostics / safe state persistence | Task 4 retains tensor forward API; Task 7 keeps action behavior |
| 5→6→7 | trainer reset / runtime inputs / checkpoint wiring | strictly sequential |
| Task 1 | gate tests vs lazy import, root discovery, output | implement behavioral tests; no fake Isaac modules |
| Task 2 | typed registry vs profile selection | explicit repair delta; simulator integration completed in Task 6 |
| Task 3 | tests vs pure actor helper | complete small-network fixtures before RED; no post-sample transform |
| Task 4 | diagnostics vs damped formula | clamped correction eta and raw eta must be unambiguous; validate real export compatibility |
| Task 5 | CPU partition vs actual simulator resets | three sets normal/replay/fallback; CPU tests cannot prove physical writes |
| Task 6 | runtime inputs vs real consumers | every resolved value must reach the relevant consumer or be blocked explicitly |
| Task 7 | safe recursive schema vs optimizer format | standard optimizer integer keys need reversible schema encoding; no unsafe fallback |

Ruling: use the repository coordination path for all SDD artifacts — AGENTS.md explicitly overrides the skill default — costs only adaptation of helper paths.
Ruling: continue from the documented broken baseline while repairing it — the user already authorized repairs and continuation — baseline failures remain recorded, never counted as new regressions.

Task 1: active; BASE=1259bae1b2e2635e410a315ab6bf92451df8762b; detached checkout `../SEA-Nav-Code-batch1`; scope is Task 1 exact file list plus local task record.

Task 1: review of `0dd407e` found three issues; fix round 1/5 active. Historical passed records must become explicitly unverified/blocked; complete tree must include all 65 baseline legged_gym files; W&B subprocess must reject W&B independently of host installation and contain no /tmp override.

Ruling: validate CPU contracts in the dedicated Python 3.10 / Torch 2.6.0+cpu environment — system Torch 1.8 lacks safe loading and the old NumPy override was temporary — this does not establish compatibility with the unavailable frozen Gym runtime.

2026-09-07: environment probe passed for Torch 2.6.0+cpu, NumPy 1.26.4, pytest 8.4.2, PyYAML 6.0.2. Task 2 dependency pins must describe this verified CPU environment, superseding the plan's machine-inherited pytest 6.2.5 / PyYAML 5.4.1 pins.

Ruling: block accepted paper_v1 runs on literal Table V action-bound ambiguity — PDF and upstream disagree and an inferred sign correction is not authoritative evidence — CPU repair continues, but paper-labeled training waits for resolution.

Ruling: extend Tasks 2/4/5/6 application contracts to cover reward equations/dt, two ACSI draws, shared CBF fixtures and perception timestamps — otherwise the planned registry would only describe unapplied values — additional focused CPU tests are required.

Ruling: allow nonnegative integer IDs only in validated optimizer state and publish immutable checkpoint generations manifest-last — real Adam state and interrupted two-file saves contradict the original sketch — adds schema-specific validation and fault tests.

2026-09-07 resumed: primary `test@aa34400`; Task 1 fix head `9fdb33e` and its covering 29-test report verified present; `/root/review_batch1` assigned scoped re-review round 1. No remote write.

Ruling: select explicit compact `new_replay_episode_v1` reconstruction and extend Task 5 through base epilogue — a root/DOF-only restore plus blanket reset mixes time, while nested full histories cost at least 1.77 GiB — this changes episode semantics and must carry `replay_reset_reconstruction_v1`; exact historical continuation remains unsupported. Full field/ordering evidence is in replay-schema-audit.md.

Task 1: fix round 1/5 (3 addressed, 1 new P2 open — complete-tree symlink substitution; commits 0dd407e..9fdb33e). Round 2/5 assigned to `/root/implement_batch1`; focused coverage `tests/test_gate_a_portable.py`, leaf and ancestor links included.

Task 1: fix round 2/5 (1 addressed, 0 open; commits 9fdb33e..4dec41b), spec PASS and quality PASS in task-1-rereview-2.md.
Task 1: complete (detached commits 1259bae..4dec41b, review clean; integrated as a58ad21/eac2657/484f682). Primary test verification: 31 passed; portable Gate A five passed with Gym blocked. Final candidate not frozen.

Task 2: registered; BASE is the next coordination-only commit following source 484f682; exact requirements in task-2-brief.md; detached checkout ../SEA-Nav-Code-batch2 is next.

Task 2: active; BASE=e22493377065c445cfd2b1ac43c01aa4d99c5120; clean detached checkout ../SEA-Nav-Code-batch2 baseline verified (12 portable-gate tests passed); implementer `/root/implement_batch2`. Only one source writer. `/root/ppo_update_audit` performs independent read-only fixture/update-boundary preparation and owns only its main-checkout audit report.

2026-09-07: ppo-update-contract-audit.md supplies a real CPU update harness and classifies alias/bad-mask/singleton risks without inventing a current simulator bug. Task 3 brief now requires the real four-sample multi-epoch update assertions; no extra storage redesign assigned.

Ruling: Task 6 must distinguish dependency discovery from simulator runtime success and report both stacks — Task 1's `isaac_gym_runtime` currently says passed after import alone, which cannot certify Rung 3 — expands only gate/probe tests, not simulator support. Missing Gym remains blocked on all current evidence.

Task 2: implemented detached HEAD=7367aa872619dfd8eccc4821ab25c4608bb3ee85, full range e224933..7367aa8. Report contains 58 passing selected CPU tests and explicit unavailable Gym boundary. Review package generated at task-2-review.diff; independent `/root/review_batch2` reviewing spec and quality. Not yet integrated.

Task 2: review changes requested (five P2: shallow frozen/hash-bound data, absent field validation, implicit legacy delta, eager manifest import boundary, conflated acquisition/refresh; one P3 horizon status). Controller read complete feedback and confirmed the first four at the actual new data/manifest boundary and the cadence against scientific-wiring-audit.md. Round 1/5 assigned to original implementer; tests/test_gate_a_portable.py ownership may expand only for its explicit manifest caller migration.

Ruling: change the horizon parity row from resolved to profile_fork — paper evaluation 30 s and upstream unsupported are not directly matched under the spec's own status definition — fixes a plan-mandated P3 without inventing an upstream evaluator, and requires regenerated configuration hashes. No simulation semantics changed.

Task 2: fix round 1/5 implementation received at `3359215`; report contains exact covering tests and outputs (71 passed). Scoped re-review assigned to original `/root/review_batch2` with `7367aa8..3359215` diff; all five P2 and the horizon P3 require verdicts before integration.

Task 2: fix round 1/5 (five P2 and one P3 addressed, 0 open; commits 7367aa8..3359215), spec PASS and quality PASS in task-2-rereview-1.md.
Task 2: complete (detached commits e224933..3359215, review clean; integrated as ed8e7e7/6f5e544). Fresh primary 71 passed, five Gate A CPU/static cases passed with Gym blocked; no deleted baseline path. Consumer migration remains Tasks 3–7, not claimed as applied here.
Task 3: registered; detached BASE will be the coordination-only successor of 6f5e544; owned paths and exact corrected fixture/update requirements are in task_plan.md and task-3-brief.md.

Task 3: active; BASE=71c2e77d41a46e67927b6de95643adf0fada8c77; detached ../SEA-Nav-Code-batch3 baseline 71 passed in 3.99s, clean before registration. Fresh implementer `/root/implement_batch3` owns only corrected brief paths. No Task 4 source writer is active.

Ruling: name a small packaged policy_factory module as Task 4's projection-to-constructor boundary — Task 2 names that consumer but no factory exists, and adapter-only construction would reverse shared dependency direction — costs one focused module/test surface; runtime application remains Task 6. No plugin framework or concurrent source changes authorized.

Ruling: migrate the inherited static gate's post-sample-projection case within Task 3 — its fixed-distribution expectation at gate_a_static_contract.py:452 explicitly requires the forbidden regression and fails after the atomic repair — retain noise-std coverage and replace it with real seeded Normal sample/likelihood evidence; costs one tightly scoped extra test file. No runtime change or skipped test.
Task 3: 14 focused tests GREEN, inherited combined suite 84 passed/1 expected obsolete-contract failure before this scope correction. Original implementer updating the single inherited case; no task review yet.

Task 3: implementation received at `9e91c722dcd0633c353d0d3c1610af7428c508f9`, range 71c2e77..9e91c72. Fresh postcommit report records 85 CPU tests passed. Independent `/root/review_batch3` assigned complete task-3-review.diff, corrected brief and full report; not yet integrated.

Ruling: final clean-checkout syntax validation uses Gate A's tracked-source compile(), pytest cache disabled, and ignored-inventory comparison — explicit compileall writes bytecode despite PYTHONDONTWRITEBYTECODE, and ordinary Git clean status hides ignored output — costs no production change and makes the no-output claim testable. Earlier compileall results remain syntax evidence only, never final clean-candidate evidence.

Task 3: complete (detached commits 71c2e77..9e91c72, review clean; integrated as 37f0763). Independent spec/quality PASS, no actionable findings. Fresh primary 85 passed and five Gate A CPU/static cases passed with Gym blocked. Remaining profile/runtime obligations assigned Tasks 4–7.
Task 4: registered; detached BASE will be the coordination-only successor of 37f0763. Exact scope in task_plan.md and corrected requirements in task-4-brief.md; no Task 4 source edits yet.

Task 4: active; BASE=17e53cf1e0b4ce476b43bdc932baa699cecf3571; detached ../SEA-Nav-Code-batch4 clean before registration and complete explicit CPU baseline 85 passed in 4.07s. Fresh implementer `/root/implement_batch4` owns only registered CBF/factory/loss/test paths; Task 3 worker retained, no concurrent source writer.

Ruling: include only the three adapter entrypoint bootstrap blocks in Task 4 if reusing the packaged CBF core — observed imports precede bundled rsl_rl selection and subsequent module purge in all three, causing import failure or split module identity for the new shared dependency — costs three tightly scoped path/import-order edits plus CPU/static boundary tests; full runtime behavior/CLI wiring stays Task 6. The original simulator and controller prerequisites remain blocked.

Task 4: implementation received at `ac0567eed6cf88a17f5178663f0324cffaaab357`; full range 17e53cf..ac0567e, fifteen registered paths. Report includes fresh postcommit 169 passed, real TorchScript and bootstrap identity regressions, and staged Gate A with Gym blocked. Complete task-4-review.diff assigned to independent `/root/review_batch4`; not integrated yet.

Task 4: complete (detached commits 17e53cf..ac0567e, review clean; integrated as a660d74). Independent spec/quality PASS; primary fresh 169 passed in 10.70s and five Gate A CPU/static cases passed with Gym blocked. Reviewed source tree matches integrated source. Task 6 owns the explicitly documented startup/geometry/reward/perception handoff; relevant actual simulator smoke remains blocked.
Task 5: registered; BASE will be the coordination-only successor of a660d74, detached ../SEA-Nav-Code-batch5. Corrected brief, replay-schema-audit.md and corrected replay-performance-harness.md govern exact compact reconstruction and active-row hot-path evidence. Only one source implementer will be active.

Task 5: active; BASE=5b99f456fbfb9eb2e6fc8eae5a03a1aa584e2b43; detached ../SEA-Nav-Code-batch5 clean before registration and full CPU baseline 169 passed in 10.71s. Fresh implementer `/root/implement_batch5` owns the registered compact replay/ACSI/lifecycle/reset/filter paths. Source implementation remains serial.

Final-review input (not pre-graded/parked): cbf-hotpath-observation.md confirms five eager-CPU scalar extractions in ordinary core/adapter CBF forward versus zero in _compute. Final whole-branch review must judge deterministic validation versus hot-path synchronization constraints; no GPU timing claim or concurrent source change.

Task 5: worker milestone, not completion — 19 pure tests reported GREEN after missing-module and lifecycle RED; actual Gym/base/adapter consumers still in progress. Runtime capability hook and missing-context behavior must be documented for Task 6; missing prerequisites cannot be relabeled applied replay.

Task 5: implementation received at 53c479dea2a2ff43fcfe5aa718da29395c5df191, full range 5b99f45..53c479d, 17 registered paths. Report records fresh 202 passed, staged portable Gate A five CPU/static passes with Gym blocked, eight Python 3.8 grammar files and explicit unsupported real-physics contracts. Complete task-5-review.diff assigned to independent `/root/review_batch5`; no integration or Task 6 source work yet.

Task 5: review changes requested, 4 P2 open — adapter normal/fallback missing prepare/refresh hooks; carrier truncation omitted from replay exclusion; device collision count cannot serialize to JSON; cancellation token dictionary unbounded. Controller read complete report and checked the actual consumers. Fix round 1/5 assigned to original /root/implement_batch5; covering files test_replay_lifecycle.py, test_replay_runtime_wiring.py, test_collision_replay_cpu.py. Correct the report's trainer_contract_locks location assertion as well. No integration.

Ruling: include checkpoint-only Gym registry/helpers/train/play and Task 6 shared-preflight caller migration in Task 7 — current task_registry.py:160-163 still derives a legacy path and calls runner.load, while helper CLI/play select old numeric/run inputs; merely repairing the runner/adapter loaders leaves this mandatory manifest boundary unwired — costs a small additional caller/static-test surface, no simulator or broader runtime redesign.

Task 5: fix round 1 implementation received at a7fe32ecbff4aae420dd073a8e2a032d941a0a10. Full fix addendum names covering tests/commands/output: 7 expected RED failures + 1 already-correct case, 8 targeted GREEN, 59 covering GREEN and staged portable Gate A. 53c479d..a7fe32e scoped package assigned to original /root/review_batch5; four verdicts required before integration. Original 202-test run belongs to 53c479d, not the fix commit.

Task 5: scoped fix review received and read completely; Spec PASS / Quality PASS, all four P2 ADDRESSED, zero new findings. Ready to integrate the two reviewed commits, then independently run the complete primary CPU/static selection. Physics/runtime acceptance remains blocked.

Task 5: complete for source/CPU/static scope; integrated `53c479d`/`a7fe32e` as `97252ed`/`1ae9187`, fresh primary **212 passed in 13.17s**, five portable Gate A CPU/static passes with Gym blocked. No baseline deletion or source/test/tools mismatch against reviewed worker. Task 6 may now start after exact path registration and fresh detached baseline. Independent whole-branch review and final frozen Rung 2 remain pending.

Task 6: registered exact 29-path scope in task-6-brief.md after read-only handoff check. Rulings: shared Gym preflight must not depend on adapter; preserve bridge's explicit reconstruction delta for disabled/enabled replay callers and validate early; checkpoint loading stays explicitly blocked until Task 7; ordinary PPO adapter cannot silently bypass projected reward/perception; trace schema must enforce four action names and physical row counts; new perception clocks participate in masked reset. No extra core CBF/PPO/runner/ring edits assigned.

Task 6: active at detached BASE `65dcbbc2b13c92af99a9e7d4cba4110880f9b509`, clean no-ignored-output baseline **212 passed in 13.06s**. Sole source implementer /root/implement_batch6. Bounded CBF hotpath review writes only its primary coordination report; no second source writer or early Task 7 implementation.

Task 6 narrow scope ruling: migrate only the inherited test_experiment_config.py adapter-YAML assertions that demand the old inactive metadata format; actual Task 6 YAML must be executable typed input. Preserve adjacent historical adapter_manifest.json and all independent integrity tests. Controller verified old lines 598–600 before authorizing; worker must register before edits.

Task 6 slice1 review: limited Spec/Quality NEEDS FIX, one P2 float32-clock cadence drift at fixed `6e097ac3d87554a8baf537aa733d72b3c7e70b33`. Same implementer fixes with meaningful long-horizon clock/reset RED→GREEN; no new owned source path. Original reviewer performs scoped re-review once fixed; later full Task 6 consumer review remains required.

Task 6 slice1 fix round1: `6e097ac..7945a9e`, original P2 ADDRESSED, scoped Spec/Quality PASS, independently 26 passed and original clock reproduction 0/0 for float32/64. Subsequent source slices and full Task 6 application/startup/capability review remain active/pending; not integrated yet.

Final-review open input CBF-HOTPATH (P2): independent bounded review established five batch-wide scalar extractions per ordinary forward and real PPO reachability, plus discarded diagnostic work. Preserve deterministic checked APIs and script behavior. Removing diagnostics alone or writing a budget exception does not establish closure; controller has not authorized unchecked dynamic inputs. Reconcile and repair/review serially after Task 7, without a concurrent core writer. See cbf-hotpath-review.md.

Task 6 narrow source ruling: additionally own adapters/command_delay.py only for alpha recurrence state/output separation. Gym retains un-clipped nav_actions_filtered, but adapter body-clips the stored recurrence despite source_alpha_only. Preserve input/body bounds, no queue and Task 5 masked reset/mirrors; require saturating recurrence and reset regressions within existing runtime-command tests and full independent review. No other core paths assigned.
