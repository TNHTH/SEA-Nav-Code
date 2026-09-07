# Findings & Decisions

## Requirements
- Preserve only `main`, `stable`, and `test` as working branches; place every repair on `test`.
- Recover a complete, scientifically labeled SEA-Nav reproduction baseline for later same-robot comparison and ablation work.
- Save, commit, verify, and push the completed candidate without falsifying unavailable simulator or robot evidence.

## Current Truth Table

| Field | Current value | Evidence | Status / limits |
|---|---|---|---|
| Artifact / version identity | integrated source `test@b5b945513acb4d3e48c401653198442654a387e9`; `main=stable=1c5675bbedf1dcbe5a4c1a91830cae528c780793` | 2026-09-07 Task6 four serial cherry-picks and fresh verification | coordination-only successor does not change tested source |
| Active evidence instance | Task6 primary342 tests passed in34.53s; five GateA CPU/static passes, Gym/Lab dependency/runtime four blockers | `../task-6-integrated-gate-a.json`; progress.md and independent rereview | non-final integrated evidence; existing ignored inventory unchanged, not frozen empty-checkout evidence |
| Declared configuration | 2026-09-07 evidence-corrected design and task briefs | scientific/checkpoint audit reports; original 252f3f85 design is historical | action-bound row blocks accepted paper_v1; upstream diagnostic repair proceeds |
| Confirmed facts | remote snapshots are unrelated roots; only `main` is a complete tree; post-sample CBF and PPO smoothness state mutation are real regressions | Git object analysis and CPU/source comparison | confirmed |
| Candidate hypothesis | repairing packaging/config/PPO/CBF/replay/runtime/checkpoint contracts will make Rungs 0–2 reliable; simulator viability remains unknown | approved spec and batch mappings | implementation pending |
| Highest validation rung | Rung1 for Tasks1–6; non-final GateA passed_with_blockers | Task6 scoped Spec/Quality PASS and fresh primary full run | frozen final-candidate Rung2 still pending |
| Unpublished/unverified work | Tasks1–6 integrated through b5b9455; Task7 exact26-path scope registered, no source work yet; final CBF P2 remains open | Task6 all5 review findings closed, source tree equals reviewed0efc193, worker clean/registration history retained | local only; no remote write; Task7, final review/frozen verification, runtime and accepted-paper blockers remain |

## Research Findings

The following baseline defects are historical findings; Tasks 1–3 now repair the portable gate/W&B and PPO action/state items with the current evidence above. Remaining CBF/profile/replay/runtime/checkpoint work is not inferred complete from them.
- `main` retains the full training/deployment/adapter tree; `origin/test` and `b53d3fe` each omit 53 baseline paths and must never replace it wholesale.
- The authoritative actor uses the CBF output as the Normal mean and samples once. Local `main` applies CBF again after sampling, breaking PPO action/log-probability identity.
- PPO smoothness calls the stateful actor on interpolation observations, overwriting `alpha`, rays, nominal/safe actions, and distribution before shield losses.
- Paper Eq. 4 intentionally uses `epsilon_d=1.0`; a negative post-transform residual is compatible with its damped inductive-bias claim and is not proof of a hard-safety defect.
- Replay buffer capacity 100 cannot represent the configured inclusive 100–150 undo range; current per-step history shifting is O(num_envs×capacity), and reset clears replay-restored histories/state indiscriminately.
- `wandb` is imported eagerly even when disabled; Gate A includes machine paths and may leave preview files in the checkout.
- Runner class lookup uses `eval`, while checkpoint loading accepts ordinary pickle; both are unnecessary trust boundaries.
- This host lacks Isaac Gym/IsaacLab. CPU/static evidence can reach Rung 2 only; simulator, metrics, publication rights, and robot gates remain separate blockers.
- 2026-09-07 source recheck: Table V literally prints positive lateral/yaw lower bounds; upstream uses negatives. Record unresolved paper action bounds as blocked instead of silently correcting them.
- Gym integrates reward coefficients over policy dt; adapter omitted this factor. Equal raw weights differ by 50x at dt=.02. Task 6 must apply it once and preserve formula/profile identity.
- Upstream ACSI uses two distinct random gates, and the adapter currently never advances its goal curriculum. Task 5 owns explicit decision and update semantics.
- Checkpoint optimizer schema must handle real integer Adam state keys/None, exact completed-update counts, and publication of immutable payload generations through an atomic manifest.
- Replay reset is a transaction through the base post-reset observation/epilogue, not merely the physical setters. The compact `new_replay_episode_v1` policy restores physical/task geometry then rebuilds all histories and controller/filter state; it is an explicit repair delta, not exact historical continuation. At 2048×151 slots it needs about 54–58 MiB rather than at least 1.77 GiB for nested histories. Full evidence and field ownership are in `replay-schema-audit.md`.
- The Task 3 CPU harness must use complete flattened history and multiple rollout samples; a singleton update produces NaN from sample-standard-deviation normalization. The four-sample real update harness is verified viable, not evidence that unrepaired PPO is correct. Generic alias/stale-mask risks are not demonstrated current environment defects. See `ppo-update-contract-audit.md`.
- Final gate runtime identity needs Task 6 follow-through: package import/discovery is dependency evidence, not actual simulation. Both Gym and IsaacLab must have separate non-fabricated runtime results.

## Technical Decisions

| Decision | Rationale |
|---|---|
| Use versioned parity data and typed resolution | Prevents paper, upstream, engineering repair, and runtime adapter identities from being silently mixed. |
| Use four action-stage names | Separates distribution, sampled policy action, environment clipping, and actual controller command. |
| Require replay reserve/commit/cancel and normal/replay/fallback partitions | Avoids consuming failed samples or erasing a restored Markov state. |
| Use checkpoint schema v2 plus adjacent hash-bound manifest and `weights_only=True` capability gate | Rejects arbitrary pickle execution and binds the artifact to its identity. |

## Issues Encountered

| Issue | Resolution |
|---|---|
| Earlier external ledger still described pre-commit state | Repository-local truth table above supersedes it for execution; old ledger remains historical audit context. |
| Remote protection and archive-tag publication rights are not yet proven | Treat remote cleanup as a fail-closed transaction after candidate verification. |

2026-09-07 publication preflight confirms all three remote heads unchanged and no remote stable/archive tags. Local archive objects are exact; SSH authenticates TNHTH and only a no-op receive-pack dry-run was tested. No authenticated hosting API/browser administration path was found (nor a GitHub connector in current tool metadata); public empty rulesets are not proof of absent classic protection. Exact archive publication/transport authorization and stable protection remain separate final-transaction gates. See publication-preflight.md. These facts do not block serial local CPU repairs.

## Resources

- `cbf-checked-path-probe.md`: complete independent in-memory CPU report read; aggregate validity retains one host decision while preserving tested output/gradient/error and TorchScript behavior (core/adapter/footprint B=2/2048: 5→1). Ordinary diagnostics remain unoptimized. A direct _assert_async substitution loses the check in this actual scripted save/load experiment and is not an acceptable CBF replacement. No source fix, performance-policy amendment, CUDA speedup or P2 closure is claimed; serial post-Task7 implementation/review still required.

- `legacy-sandbox-capability.md`: trusted fresh bwrap namespace probe actually imported task Torch 2.6.0+cpu with weights_only support, only loopback visible and no /home mounted. A separate later system-Python probe verified pinned-FD read-only mount and child limit read-back, including its initial EACCES-vs-EROFS test calibration failure. No checkpoint/untrusted code, Torch-under-those-limits, limit-trigger/race/output acceptance or completed converter is claimed.

- Task 6 moving-tree output-edge self-review found candidate-path overwrites, profile stay-threshold drift, cleanup ordering and nested-manifest output defects. Sole implementer is addressing them; fixed-commit whole Task 6 review and full suite remain mandatory. The prior in-memory CBF fused-validation diagnostic produced an unusable tool result, so no improvement is established; a separate bounded read-only CPU probe is registered for preparation only.

- `gym-import-order-boundary.md`: current in-progress shared preflight eagerly loads Torch before Gym. A public IsaacGymEnvs issue provides the known gymdeps import-order error and official train source imports Gym first. Task 6 must preserve early pure rejection without poisoning the Gym parent import order; fresh-process tests and runtime capability blockers are required, with no simulator claim. This warning was issued during implementation, not a final review verdict.

- Final caller-documentation check found root README train/play examples without the new identity/runtime inputs, inherited Gym README recommending raw/numeric checkpoints and latest-run discovery (plus historical Preview 3), and adapter README describing only first-wave static work. Task 7 brief now assigns narrow usage migration in those three existing READMEs, preserving provenance and blocked simulation status; actual final parser validation is required, not guessed commands.

- `checkpoint-runner-cpu-harness.md`: real runner/PPO CPU probe completed 102 updates and observed the old intermediate save requesting model_101.pt with iteration field 0; final field was 102 and learn returned None. Real Adam had 13 integer-keyed parameter states. This is historical failure/harness feasibility, not persistence/resume or Task 7 completion; no checkpoint/simulator files were written.

- Final-review input `cbf-hotpath-observation.md` and independent `cbf-hotpath-review.md`: one open P2, five scalar extractions per ordinary core/adapter CBF forward; real T=2/E=2/M=2 actor/PPO probe observed 10 collection and 40 update extractions. Ordinary eager calls also compute unused diagnostics. This is CPU operator evidence only, no CUDA timing. Preserve dynamic invalid-input rejection and script behavior; removing diagnostics alone does not close the extraction finding. A documented budget exception or unchecked route has NOT been approved as closure. Resolve the boundary and fix/review serially after Tasks 6–7; no concurrent CBF edit.

- Approved design: `docs/superpowers/specs/2026-09-04-sea-nav-reproduction-recovery-design.md`
- Authoritative source: `11chens/SEA-Nav-Code@fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96`
- Paper: arXiv `2603.09460v1`, PDF SHA-256 `600a5040b6579fe63615d87a70f174f3fa0b0d018f74440b6707d23d36dfc2e9`
