# Task 3 independent spec and quality review

Date: 2026-09-07. Reviewer: `/root/review_batch3`.

No actionable findings. No P0–P3 defect was established in the Task 3 change.

**Spec compliance verdict: PASS for Task 3.** **Code quality verdict: PASS.** These verdicts cover the reviewed atomic actor/PPO repair and its authorized tests; they do not certify future profile/runtime wiring, simulator behavior, or whole-branch readiness.

## Scope and evidence

Reviewed the complete 727-line supplied `task-3-review.diff`, corrected `task-3-brief.md`, complete `task-3-report.md`, complete `ppo-update-contract-audit.md`, repository `AGENTS.md`, and updated design section 6 plus its relevant evidence corrections/CBF boundary. Applied `/home/twyc/.codex/skills/code-review/SKILL.md`.

Verified commit `9e91c722dcd0633c353d0d3c1610af7428c508f9` has parent `71c2e77d41a46e67927b6de95643adf0fada8c77`; its six changed paths match the three actor/PPO sources, new identity tests, authorized inherited Gate test migration, and task log. No unrelated source changes occur in this diff. Read-only ref inspection found exactly `main`, `stable`, and `test`; both baseline branch heads remain `1c5675bbedf1dcbe5a4c1a91830cae528c780793`.

No source, index, HEAD, branch, or remote changes were made by this reviewer. The only review write is this report. No subagents or simulator substitutes were used. The reported 85-test suite was not rerun: inspection established no concrete unanswered implementation doubt requiring a focused execution probe.

## Requirement assessment

| Requirement | Assessment and evidence in reviewed diff |
| --- | --- |
| One mean-stage CBF call and an unchanged Normal draw | PASS. `cbf_actor_critic.py:167` returns `self.distribution.sample()` directly after distribution creation. The additive/counting shield test asserts exact seeded sample equality and one call. The authorized Gate migration additionally exercises the real CBF and real Normal, retains noise-std initialization, independently verifies likelihood, and ensures a second pass would visibly change the chosen draw. |
| Pure CBF helper returns `(u_s, alpha, rays_real, u_bar)` without owning active state | PASS. `_compute_safe_action_mean` computes locals; `forward` exclusively assigns the four active auxiliaries; `action_mean_for` returns the computed mean without replacing the distribution, `mean` alias, or auxiliaries. Existing policy encoder detachment is preserved. |
| Ordinary actor pure query preserves distribution ownership | PASS. `ActorCritic.action_mean_for` computes encoder/actor output directly. `update_distribution` retains ownership of `distribution` and `mean`. Tests cover both an active distribution and a first query leaving `distribution is None` and no `mean` attribute. |
| Smoothness uses pure queries and supplied current minibatch originals | PASS. `compute_smoothness_loss` accepts keyword-only `orig_mu`/`orig_values`, computes missing originals without `act`, and compares interpolated pure means/critic values with the supplied live tensors. `update` passes `mu_batch`/`value_batch` without detach or clone. Supplied-original gradient assertions separately prove both originals remain connected. |
| Preserve current minibatch state through range, alpha, intervention, and optimizer use | PASS. The actor/PPO changes are atomic. Existing range loss uses `mu_batch`; no smoothness actor call mutates active distribution or auxiliaries. Tests capture object references and exact value snapshots for distribution, `mean`, mean/std and all four CBF fields, check them after smoothness and before every real optimizer step, observe the current alpha argument, and reconcile reported range/smoothness/intervention metrics. |
| Real collection → storage → update sample and likelihood identity | PASS. Both real actor types run two environments × two steps, two minibatches × two epochs, positive learning rate, and a second complete rollout/update. Generator rows match independent observation/sample/Normal-likelihood/mean/std records. The actual likelihood wrapper requires the yielded stored action object and excludes the freshly drawn update sample. Detached targets, valid masks, first-batch ratio one, changed later ratios, finite/nonzero gradients, parameter changes, all four steps, and storage reset are asserted. |
| Meaningful corrected fixtures | PASS. Tests use flattened 38-value histories with 12 props, five encoded rays, two goals, and two frames. Real CBF integration has active intervention and distinct current/interpolated auxiliary values. No singleton update fixture, fake simulator, constant-output policy, zero-rate-only update, or all-invalid-mask shortcut is used. |
| Unsupported actor contracts fail clearly; no recurrence work | PASS. PPO rejects recurrent actors and actors without a callable pure query at construction, with targeted error tests. There is no partial-state compatibility fallback or new recurrent implementation. |
| Authorized inherited gate migration only | PASS. The obsolete post-sample projection test and its direct-main registration alone change. Noise initialization remains covered; seeded real sampling and likelihood replace the fake distribution assertion. Other inherited Gate cases are unchanged. |
| Action-stage separation | PASS within Task 3. The actor returns the distribution sample; collection stores/scores it; the test's out-of-place `clipped_policy_action` cannot replace it. The diff introduces no post-sample CBF or executed-command transformation. Full runtime trace naming for `distribution_mean`, `policy_action`, `clipped_policy_action`, and `executed_command` remains a cross-task obligation. |
| Scope, compatibility, and evidence ceiling | PASS within Task 3. Only authorized paths change. Python 3.8 syntax compatibility and CPU/static checks are reported; no new dependency or simulator task-stack import is introduced. Geometry, profile coefficients, storage, runner, reset, and trainer implementation are outside the diff. |

## RED/GREEN and quality assessment

The report records an initial 14-case RED on actual sampling mismatch, absent pure queries/keyword arguments, state overwrite, missing rejection, and missing current-original identity. It then records 14 focused passes after the atomic repair. Strengthening gradient assertions exposed a legitimate zero-std-gradient case after surrogate clipping; requiring finite std gradients at every step and at least one nonzero std gradient per rollout is appropriate and preserves meaningful learning checks.

The inherited suite initially produced 84 passes and one obsolete projection expectation failure. The approved replacement was then strengthened when a mutation showed its first seed did not expose a second CBF pass. The final forbidden-pass mutation failed both exact-sampling tests. This is concrete regression sensitivity, not a removed or skipped assertion.

Reported final evidence is 85 passes precommit and a fresh postcommit 85 passes in 4.34 seconds, 16 standalone Gate checks, compileall, Python 3.8 AST parsing of all five changed Python files, and whitespace checks. These are implementation-reported command results, reviewed for consistency with the committed test code, not independently re-executed results. The reviewer found no reason to repeat the suite.

The production change is small and maintains clear state ownership: pure computation is separate from active policy-state assignment, and PPO reuses its existing current-batch graphs. The tests wrap real actor, generator, likelihood, smoothness, and optimizer methods instead of replacing the behavior under test. No supported-consumer correctness regression, unjustified abstraction, new dependency-direction problem, or demonstrated performance issue was found.

## Cannot verify here / retained cross-task obligations

- Integration is controller-owned: this review verifies the candidate parent and current baseline refs, but cannot certify the complete historical serial execution record or future integration exclusively onto `test`. It makes no integration or whole-branch completion claim.
- Task 4 owns `paper_v1` Eq. 4, `epsilon_d=1.0`, geometry/loss profile consumers and matching golden vectors. This diff intentionally does not establish those semantics. Its helper name does not certify hard safety; the intended paper result remains a damped safety bias.
- Task 2 owns mandatory explicit `implementation_delta=["ppo_state_identity_repair"]` declaration and no silent resolver addition; Task 6 owns actual runtime activation/recording and complete action-stage traces. Those configuration/runtime consumers were not re-audited or proven by this Task 3 diff.
- CPU tests verify tensor contracts and learning mechanics only. Isaac Gym/IsaacLab behavior, relevant simulator smoke, formal 100-trial metrics, publication rights, real hardware, and accepted paper-reproduction claims remain blocked or unverified as specified. The authorized Gate test migration changes no adapter runtime behavior.
- Generic action/observation alias mutation, mixed bad-mask reuse, privileged critic observations, recurrent interfaces, and unrelated profile behavior are not established reachable Task 3 defects and were not expanded into this review.

Task 3 is suitable for controller integration subject to the existing serial workflow and the explicitly retained later-task/runtime boundaries above.
