# Task 5 fix round 1 — independent scoped re-review

Base: `53c479dea2a2ff43fcfe5aa718da29395c5df191`.
Reviewed HEAD: `a7fe32ecbff4aae420dd073a8e2a032d941a0a10` in `work/SEA-Nav-Code-batch5`.

**Final Spec: PASS. Final Quality: PASS.** All four original P2 findings are **ADDRESSED**. No new actionable finding was established in this fix diff: P0=0, P1=0, P2=0, P3=0 outstanding. These verdicts close the scoped Task 5 source/CPU/static gate; they are not simulator acceptance or the later whole-branch review.

## Per-finding verdicts

| Original finding | Verdict | Evidence in the fix |
|---|---|---|
| Normal/fallback adapter resets omitted preparation and refresh | **ADDRESSED** | Trainer `_normal_reset_rows` obtains the active contract at line 997, calls `prepare_rows` before carrier reset/manual writes and `refresh_rows` after `_place_robot_at_start` at line 1016. The existing transaction next calls reconstruction and then finish; the fix does not move ack/cancel earlier. Replay-disabled operation does not acquire a contract. |
| Selected carrier timeouts remained replay eligible | **ADDRESSED** | Trainer `step`, line 1303, unions `carrier_truncated` into `_terminal_timeout` in the same ordinary-mode branch that selects carrier done signals, before capture/reset. The explicit play/evaluation branch continues to ignore carrier signals. The saved mask is the input to the previously reviewed reset selection consumer. |
| Device collision counter broke both result JSON mappings | **ADDRESSED** | Both trainer result entries, lines 1616 and 1743, convert the counter with `int(...)`. Accumulation remains device-side and unchanged, so the repair moves scalar extraction to reporting rather than the capture hot path. |
| Cancellation provenance retained every lifetime token | **ADDRESSED** | Ring `cancel_restore`, line 223, replaces `last_cancellations[env_id]` with one frozen event containing source episode, task generation, token and reason. The line 227 compatibility property returns a bounded snapshot of those latest events. Validated environment IDs bound storage to `num_envs`; stale-token validation still precedes clearing/replacing state. |

Paths in the table refer to `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py` and `training/rsl_rl/rsl_rl/replay/ring.py` in the reviewed checkout.

The added regression bodies substantiate each closure:

- Hook AST assertions bind preparation/write/refresh order for both physical paths and the actual normal/write/reconstruct/epilogue callback sequence. Pure lifecycle cases reject replay before preparation, then fail normal preparation or refresh; neither case allows reconstruction/finish, cancellation or acknowledgment.
- The timeout test executes the actual pure terminal-source selection block in both modes with carrier-only timeout, overlapping reasons, semantic-only timeout and carrier termination. It supplies the resulting mask to the shared replay decision.
- Reporting tests execute the actual capture accumulation expression under scalar-extraction tracing, then both actual result-value expressions for zero and nonzero collisions. They require a native integer and plain JSON round-trip success.
- The cancellation test exercises 1,000 retries, a neighboring row, a subsequent episode/task generation, stale cancellation rejection and a still-committable current retry. It asserts two retained entries for two environments and verifies episode/token/task provenance.

No new breakage was established from the changed source and these tests. No unrelated observation is being added as another Task 5 blocker.

## Scope and evidence boundary

Read the same binding Task 5 brief, the original four-finding review, the complete appended fix report starting at line 131, and the complete supplied 435-line `task-5-fix-1-review.diff` including log/stat and all six file diffs. The previously read code-review skill and schema/lifecycle contracts remain the review standard. No untouched source was reopened or re-reviewed.

The report distinguishes the original commit's 202-test result from this fix's evidence. It records the exact targeted RED result of 7 failures and 1 pre-existing pass, matching GREEN of 8 passes, covering suite of 59 passes, Python 3.8 grammar checks and staged portable Gate A `passed_with_blockers`. The controller confirmed that evidence; the regression bodies and fixes were independently inspected here. No covering/full suite, Gate A or additional probe was rerun because the narrow diff resolved all four concrete concerns without an unanswered issue requiring execution.

The report-location correction is accurate: policy/delta/bootstrap are top-level final-result fields, not `trainer_contract_locks` fields. This fix does not claim to complete Task 6 manifest propagation.

Read-only checks confirmed the assigned HEAD and showed only the previously preserved local task_plan/resume_state changes. No source, index, ref, branch, dependency or simulator state was mutated; no subagent was used. This report is the sole re-review write.

## Remaining blocked and cross-task obligations

Real Gym Rung 3G and IsaacLab Rung 3L remain **blocked**. Calling the correct hooks and exercising pure failure timing does not prove indexed physical readback, contact/sensor/actuator clearing, FK/cache validity, stateless controllers, terminal-before-autoreset ordering, unaffected neighbors or first/next observations in a real carrier. The retired capture-only forced smoke remains blocked.

Task 6 still owns final startup/profile propagation, applied identity/manifest serialization, current-goal reward timing and timestamped real perception. Accepted paper identity, formal 100-trial metrics, publication rights and real hardware retain their existing blocked boundaries. Final whole-branch review remains a later gate.
