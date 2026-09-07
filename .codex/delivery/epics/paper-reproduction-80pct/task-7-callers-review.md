# Task 7 bounded caller and README review

Date: 2026-09-07. Reviewer `/root/review_batch7_callers`. This bounded ordinary review is complete.

## Actionable findings in the Task 7 caller/README scope

### [P2] Reconcile Gym play's terrain override with the accepted request before make_env

File: `training/legged_gym/legged_gym/scripts/play.py:106-110`, especially `env_cfg.terrain.num_rows = 1` at line 107; rejecting consumer: `training/legged_gym/legged_gym/utils/task_registry.py:110-111`.

With the current documented YAML, the real play preflight accepts an inference manifest and `source_max_goal_level=10.0`. Play then forces the actual terrain to one row. Its call to `make_env` therefore reaches the existing mandatory equality guard with requested level bound 10 and actual row bound 1 and raises `ValueError: source_max_goal_level must match the actual Gym terrain row bound`, before reaching the newly wired inference runner. This deterministic configuration failure was reproduced with actual source/config AST statements on CPU; no Gym substitute or runtime-gate override was used. The separate current `runtime_ready=False` gate still stops ordinary execution earlier, so this is not claimed as an observed simulator failure.

The two conflicting statements are inherited from BASE, but closure belongs to this Task 7 caller migration: `task-6-report.md:115` explicitly hands off **"Legacy play's overrides are unreachable until that task resolves inference and post-override effective configuration."** Task 7's binding brief requires consuming that actual handoff. The new caller remains unable to apply its accepted request even after its independent runtime prerequisites are eventually satisfied; parser and model-load tests alone do not close that contract.

Minimal correction: keep play's final terrain/configuration consistent with its already bound request and effective receipt before `make_env`, preserving the existing equality/receipt checks and runtime blocker. Retain the selected terrain row bound, or resolve any intentionally different play configuration before accepting the request; do not silently overwrite the accepted value or remove the guard. Add an ordinary CPU regression spanning actual play configuration preparation into the registry's preconstruction guard/receipt. This does not require terrain-algorithm changes, installing Gym, or claiming runtime acceptance.

**Bounded Spec: FAIL. Bounded Quality: FAIL** for this one unresolved ordinary caller contract. No additional actionable finding was established in the remaining checkpoint callers or current README command contracts. **Full Task 7 review remains INCOMPLETE**, including the service-restricted original full core review and unreviewed converter/security scope. Neither these results nor the passing tests approve Task 7 integration, simulator execution or final-project acceptance.

## Fixed identity and scope

- Read-only checkout: `../SEA-Nav-Code-batch7-review-callers`.
- HEAD: `c7b9aa371b8ab3800adea378a7024f043fb58580`.
- BASE: `562d4ae0b56ca977ea433def6dbd05607284e38b`.
- Sole write scope: this primary coordination report. No source, tests, config, index, commit, ref, dependency or remote mutation. Initial and final `git status --porcelain --untracked-files=all --ignored` are empty, and final HEAD matches the fixed identity.
- Read checkout AGENTS; full code-review/planning-with-files/git-guru SKILL instructions; binding Task 7 brief and exact inventory; Task 6 fix1 report and scoped rereview; Gym import-order boundary; shared registration/current state; fixed worker log; and complete final primary `task-7-report.md`.
- Read complete BASE..HEAD diff for all nine caller/preflight/initializer production files, the three READMEs, changed ordinary caller/shape/CLI/Gate A assertions, and ordinary runner functional tests. Read relevant surrounding producer, runner load/return, policy factory, environment-profile, Gym config, argument and lifecycle code. This is not a complete checkpoint-core implementation review.
- No source writer was started. Converter implementation and checkpoint-security tests were not reviewed or executed; no adversarial fixture was constructed or run. The expression/marker registry test embedded in the otherwise ordinary caller test file was explicitly deselected.

## Contracts checked

1. Shared preflight accepts distinct mutually exclusive init/resume/inference manifests, validates the caller-supplied producer commit, rejects mismatched mode/flag combinations and legacy raw/numeric/latest-run flags, and binds resolved-config identity and the manifest receipt before proprietary startup. Input preflight and final helper application remain separate checks.
2. Both adapter trainers and Gym task_registry pass explicit producer/config metadata into their actual shape-adapted OnPolicyRunner construction. The real init/resume helper calls apply model/optimizer/iteration semantics. The ordinary runner tests exercise real PPO updates, nonempty Adam restoration, failed-update counts, N+M continuation, and 102-save/adaptive-resume behavior.
3. Both adapter trainers and Gym train use the exact manifest returned by `learn()` and report completed PPO updates. No lexical checkpoint glob or numeric/latest-run discovery remains on these callers. Gym inference uses the same explicit request; play no longer forces legacy resume/run/checkpoint configuration.
4. Smoke applies model weights through the shared inference helper into its actual profile-built actor, retaining the CBF layer and config identity. The initializer produces actual model-only iteration-zero manifests plus separate metadata for the explicitly selected stack, and labels runtime verification false.
5. Actual fresh-process Gym train/play preflight remains Torch-free in the parent while its real child executes CPU consumer/checkpoint validation and verifies the resolved configuration hash. Prerequisite checks still precede real Gym import. Actual blocked Gym play emits bound input-checkpoint evidence and exit 3; preflight-only creates no run output.
6. Task 6 shape adaptation/applied_shapes, initial/operator/terminal-once reset behavior, smoke seed/horizon readback and no-replay first trace, disabled-by-default training trace, enabled physical row/path binding, output guards and attempt-all cleanup assertions remain unchanged except required checkpoint metadata/seam migrations, and pass in the independent selection.
7. All eight marked root/Gym/adapter README commands pass their actual pure parsers with real ordinary temporary manifests and required inputs. Current text preserves upstream attribution, labels historical installation and raw/numeric/latest-run guidance, and distinguishes CPU checks from blocked simulator/controller/paper acceptance. It does not claim the proprietary Gym parser or real simulator ran.

## Independent execution evidence

All checks ran from the fixed checkout with the dedicated Python 3.10.12 / Torch 2.6.0+cpu environment. Bytecode and pytest cache output were disabled.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_checkpoint_runtime_wiring.py tests/test_environment_profile.py \
  tests/test_runtime_cli_contract.py tests/test_runtime_manifest_contract.py \
  tests/test_runner_checkpoint.py tests/test_runner_optional_wandb.py \
  -k 'not rejects_expressions_without_execution'
```

**124 passed, 1 deselected in 36.13 s**, exit 0. The deselected test is `test_actual_gym_runner_registry_rejects_expressions_without_execution`. No converter or checkpoint-security test was selected. Ordinary helper imports from the test modules do not invoke their unselected test functions.

An additional independent producer-to-consumer probe used the actual initializer for each selected stack, actual Gym/two-trainer runner-construction AST statements from `test_environment_profile.execute_actual_runner_caller`, smoke's actual `high_level` construction statement, and the real shared checkpoint application helpers. Four routes passed: Gym, ordinary adapter PPO, ACSI adapter PPO and smoke. Each consumer's 32 state tensors exactly equals the initializer payload, iteration is zero, and both trainer/Gym warm-start targets retain empty optimizer state. No run output or simulator was created. This closes a concrete gap beyond parser-only README checks; it does not substitute a simulator environment.

Python 3.8 `ast.parse(feature_version=(3,8))` plus in-memory `compile()` passed for the nine scoped production Python files, with no bytecode emitted. Scoped full-range `git diff --check` passed. The fixed HEAD and empty final status including ignored files were read back. Test/probe stdout remains in tool outputs; no separate raw log file is claimed.

The worker's final 431-test and Gate A results were read as handoff inputs. This review did not repeat or independently certify that full selection or Gate A; its own exact evidence is the selection and probes above.

## Detailed evidence for the inherited Gym play finding

`training/legged_gym/legged_gym/scripts/play.py:106-110` forces one terrain row for its one-environment setup. `training/legged_gym/legged_gym/utils/task_registry.py:110-111` requires `request.arguments.source_max_goal_level == env_cfg.terrain.num_rows` before environment construction. The documented YAML has `source_max_goal_level: 10.0` at `sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml:14`.

A real pure preflight with that documented Gym play config passes. Executing the actual Go2 configuration classes, play's config-preparation AST statements and the actual registry guard gives `requested_source_max_goal_level=10.0`, `actual_play_terrain_rows=1`, then `ValueError: source_max_goal_level must match the actual Gym terrain row bound`. No environment or run directory was created. The first harness attempt failed before this check with `NameError: LeggedRobotPosCfg is not defined`; adding the actual intermediate parent class and NumPy to the AST extraction produced the confirmed result. That failed setup is not a production failure or extra passing test.

BASE readback confirms both conflicting statements already existed at `562d4ae`. The current normal main route stops earlier at the explicit `runtime_ready=False` controller/runtime prerequisite gate. These facts constrain the failure attribution: it is an inherited caller defect with an explicit Task 7 handoff obligation, not a newly introduced terrain algorithm defect or observed simulator failure. Task 6's complete report line 115 expressly leaves post-override effective configuration to Task 7. Therefore preserving the unresolved contradiction while adding the manifest inference route is a scoped incomplete caller migration. The README's present preflight-only/blocked claims remain accurate; parser PASS must not be promoted to play execution evidence.

## Omissions and handoff

The complete checkpoint core contract, converter isolation/behavior, checkpoint security suite and adversarial inputs are outside this task. The earlier full core review remains incomplete after its service restriction; this bounded review neither replaces nor retries it. Real Isaac Gym/IsaacLab import, proprietary parser, lifecycle, checkpoint optimizer continuation on those stacks, controller/provenance, replay/reset physics, formal metrics, accepted paper identity, redistribution rights and hardware are unverified/blocked. Model/optimizer continuation does not restore RNG or physical trajectories.

The sole Task 7 writer should address the narrow play configuration/receipt contradiction and add the actual CPU caller regression, followed by a scoped re-review. No source was changed by this reviewer. Preserve the runtime blocker, the open CBF hotpath finding for its separate owner, and the full Task 7 review/integration/frozen-candidate gates as outstanding.

## Supplemental bounded first-observation and return checks

After the initial report, the controller requested a narrow first-frame observation check on the same fixed `c7b9aa3` checkout. **The stale-first-observation candidate is rejected; there is no additional finding.** The original single terrain P2 and bounded FAIL verdict remain unchanged.

The alias concern itself is valid: `BaseTask.__init__` allocates a separate zero `obs_buf` (`base_task.py:72`), `get_observations` returns that object (`:103-104`), and `reset` returns the observations produced by `step` (`:116-121`). `LeggedRobotPos.step` delegates to the base step (`legged_robot_pos.py:251-257`), whose non-in-place `torch.clip` rebinds `obs_buf` (`legged_robot.py:112-115`). An old local observation therefore must not be assumed to follow all later environment buffers.

However, the full actual play sequence includes `env.reset()` at `play.py:153` and **`obs, _ = env.reset()` at `:154`**, after the runner constructor/load and before the first `policy(obs.detach())` at `:160`. The proposed `:126 → :129 → :160` sequence omitted these decisive statements. The local variable is explicitly rebound to the latest returned reset frame.

One additional ordinary CPU probe executed the actual play observation/policy/reset/first-action AST statements in source order and the actual Gym config-to-`OnPolicyRunner` construction statements. Its tensor fixture deliberately allocated a new observation tensor on every reset, preserving the hypothesized alias hazard without importing or simulating Isaac. The real runner and real `DifferentiableSafeActorCritic` were used; instrumentation captured the argument then invoked the real inference policy. Result: three resets in total (one runner-owned plus play's two), initial observation value `0.0`, first policy observation value `0.75`, first policy input shares the latest reset storage and not the initial tensor, finite actions of shape `[2, 3]`. Exit 0. No long rollout, environment algorithm change, runtime-gate override or output artifact was involved.

Static return-contract inspection also rejects a normal-return-None concern: all normally terminating play paths reach `play.py:206-210`, construct `blocked_result(...)`, attach `applied_shapes` and return the dictionary. The loop's `break` does not return from the function. Exceptions propagate into main's existing exception/result handling; main's preflight-only branch also returns a dictionary. No inference loop was run to establish this static conclusion.

Final supplemental HEAD readback remains `c7b9aa371b8ab3800adea378a7024f043fb58580`; status including ignored/untracked files is empty. Only this permitted report was appended. No production/test/ref changes and no converter/security expansion occurred.
