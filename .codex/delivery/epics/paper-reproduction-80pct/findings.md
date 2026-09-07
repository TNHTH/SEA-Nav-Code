# Findings & Decisions

## Requirements
- Preserve only `main`, `stable`, and `test` as working branches; place every repair on `test`.
- Recover a complete, scientifically labeled SEA-Nav reproduction baseline for later same-robot comparison and ablation work.
- Save, commit, verify, and push the completed candidate without falsifying unavailable simulator or robot evidence.

## Current Truth Table

| Field | Current value | Evidence | Status / limits |
|---|---|---|---|
| Artifact / version identity | integrated source `test@1ae9187e293bd13550a70f7858207a62ea6f6da5`; `main=stable=1c5675bbedf1dcbe5a4c1a91830cae528c780793` | 2026-09-07 Task 5 serial cherry-picks and verification | coordination-only successor does not change the tested source |
| Active evidence instance | Task 5 primary checkout run: 212 tests passed in 13.17s; Gate A five CPU/static passes and real Gym blocked | `../task-5-integrated-gate-a.json`; summary in progress.md | non-final batch evidence; not a frozen-candidate artifact |
| Declared configuration | 2026-09-07 evidence-corrected design and task briefs | scientific/checkpoint audit reports; original 252f3f85 design is historical | action-bound row blocks accepted paper_v1; upstream diagnostic repair proceeds |
| Confirmed facts | remote snapshots are unrelated roots; only `main` is a complete tree; post-sample CBF and PPO smoothness state mutation are real regressions | Git object analysis and CPU/source comparison | confirmed |
| Candidate hypothesis | repairing packaging/config/PPO/CBF/replay/runtime/checkpoint contracts will make Rungs 0–2 reliable; simulator viability remains unknown | approved spec and batch mappings | implementation pending |
| Highest validation rung | Rung 1 for Tasks 1–5; non-final Gate A passed_with_blockers | independent Task 5 scoped review and fresh primary run | frozen final-candidate Rung 2 still pending |
| Unpublished/unverified work | Tasks 1–5 integrated; Task 6 registration/preparation and Task 7 pending; final whole-branch review still required | task-5-rereview-1.md closes the four initial P2; source/test tree equals reviewed worker, no baseline deletion | local only; no remote write; real runtime and accepted paper identity remain blocked |

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

- Final-review input `cbf-hotpath-observation.md`: a focused real TorchDispatch probe at source a660d74 records five scalar extractions in ordinary core/adapter CBF forward, zero in the pure mathematical core. This is CPU operator evidence only; CUDA impact and the safe reconciliation of deterministic validation with the no-host-sync hot-path contract require final review. No concurrent CBF fix has been made.

- Approved design: `docs/superpowers/specs/2026-09-04-sea-nav-reproduction-recovery-design.md`
- Authoritative source: `11chens/SEA-Nav-Code@fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96`
- Paper: arXiv `2603.09460v1`, PDF SHA-256 `600a5040b6579fe63615d87a70f174f3fa0b0d018f74440b6707d23d36dfc2e9`
