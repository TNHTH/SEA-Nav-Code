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
