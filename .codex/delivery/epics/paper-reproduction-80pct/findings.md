# Findings & Decisions

## Requirements
- Preserve only `main`, `stable`, and `test` as working branches; place every repair on `test`.
- Recover a complete, scientifically labeled SEA-Nav reproduction baseline for later same-robot comparison and ablation work.
- Save, commit, verify, and push the completed candidate without falsifying unavailable simulator or robot evidence.

## Current Truth Table

| Field | Current value | Evidence | Status / limits |
|---|---|---|---|
| Artifact / version identity | `test@1259bae1b2e2635e410a315ab6bf92451df8762b`; `main=stable=1c5675bbedf1dcbe5a4c1a91830cae528c780793` | 2026-09-07 Git preflight | current local state |
| Active evidence instance | complete local clone, detached read-only `b53d3fe`, and Task 1 detached checkout from `1259bae` | 2026-09-07 worktree inventory | source unchanged; Task 1 next |
| Declared configuration | 2026-09-07 evidence-corrected design and task briefs | scientific/checkpoint audit reports; original 252f3f85 design is historical | action-bound row blocks accepted paper_v1; upstream diagnostic repair proceeds |
| Confirmed facts | remote snapshots are unrelated roots; only `main` is a complete tree; post-sample CBF and PPO smoothness state mutation are real regressions | Git object analysis and CPU/source comparison | confirmed |
| Candidate hypothesis | repairing packaging/config/PPO/CBF/replay/runtime/checkpoint contracts will make Rungs 0–2 reliable; simulator viability remains unknown | approved spec and batch mappings | implementation pending |
| Highest validation rung | Rung 0 on historical source; final candidate has not yet been built | prior syntax/Git/Gate A evidence | do not claim final pass yet |
| Unpublished/unverified work | plan committed as `1259bae`; Batches 1–7 not yet implemented; current coordination updates pending | working tree status | local only |

## Research Findings
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

## Resources
- Approved design: `docs/superpowers/specs/2026-09-04-sea-nav-reproduction-recovery-design.md`
- Authoritative source: `11chens/SEA-Nav-Code@fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96`
- Paper: arXiv `2603.09460v1`, PDF SHA-256 `600a5040b6579fe63615d87a70f174f3fa0b0d018f74440b6707d23d36dfc2e9`
