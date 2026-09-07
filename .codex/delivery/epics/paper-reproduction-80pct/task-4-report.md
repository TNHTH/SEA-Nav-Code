# Task 4 report — damped CBF diagnostics and actual profile consumers

Date: 2026-09-07. Worker: /root/implement_batch4. Worktree: ../SEA-Nav-Code-batch4. Base: 17e53cf1e0b4ce476b43bdc932baa699cecf3571. Commit: **ac0567eed6cf88a17f5178663f0324cffaaab357** (`feat(cbf): bind damped diagnostics and profile consumers`).

## Result and acceptance boundary

Task 4 CPU/static implementation is complete and self-reviewed. Full requested suite: **169 passed**, exit 0. Staged-file Gate A: **passed_with_blockers**, five passed cases and one unavailable Isaac Gym runtime blocker. No IsaacLab smoke, simulator result, formal evaluation, paper acceptance, hardware validation, or publication-rights claim.

The binding corrections, scientific-wiring-audit.md, design sections 5–7, root AGENTS.md, and TDD/verification skills were read. Exact local ownership was registered in task_plan.md/resume_state.json before source edits. After shared-core reuse exposed the adapter bootstrap ordering issue, the controller explicitly extended scope to three main bootstrap blocks plus tests/test_cbf_import_boundary.py; that extension was registered before edits. No subagents or reviewer were dispatched.

## Interfaces and scientific behavior

1. **Core layer:** forward(u_bar, lidar_dists, alpha) remains tensor-only. Exported forward_with_diagnostics returns tensor plus tensor dictionary: h_comp, Lg_h, Lg_norm_sq, r, eta_raw, nonnegative eta, correction_norm, residual_before, residual_after. It validates configured count/FOV, positive kappa/damping, radii, batch shapes, finiteness, positive raw distances and already-positive alpha. Diagnostics retain gradient graphs and do not mutate actor state. Eq.4 uses damping in the denominator; active residual_after = -eta * damping. Yaw passes through exactly.
2. **Adapter:** FootprintAwareLSECBFLayer inherits the packaged core and shares its numerical implementation. Scalar preprocessing attributes preserve real torch.jit.script compatibility; state_dict still uses ray_unit_vectors. ExactLSECBFShield.apply(..., gamma) preserves the historical keyword but explicitly documents gamma as RAW alpha logits. apply_alpha accepts already-transformed positive alpha. Canonical diagnostics and historical debug aliases coexist. Nonzero footprint requires a nonempty ablation/ name, and validation precedes clipping. Zero footprint leaves subminimum positive raw rays unchanged.
3. **Golden fixture:** tests/fixtures/cbf_paper_damped_v1.json stores schema/equation version, ray order, dtype, tolerances, operational parameters, provenance, inputs and expected output/diagnostics. Seven cases cover close/asymmetric/inactive hazards at 180/240 degrees and near-opposing 180-degree hazards. Expected values were independently calculated with scalar ECMAScript Math exp/log/trigonometry, never either tensor implementation. Both core and adapter tests consume this same file. Positive alpha=1 corresponds to raw alpha=0.5413248546129181.
4. **Factory:** packaged rsl_rl.policy_factory exposes policy_constructor_kwargs(projection, implementation_delta, **overrides), ppo_constructor_kwargs(...), build_actor_critic(resolved_config, **overrides), and build_ppo(resolved_config, actor_critic, **overrides). It uses materialize_values(), enforces resolved integrity through Task 2 accessors, rejects unknown/missing/conflicting fields, checks all fixed semantic fields, and rejects missing activation deltas. Lower-level projection converters support explicitly labelled CPU diagnostics only; they do not create run identity.
5. **Actor application:** history_frames→his_len; num_rays; observation_fov_deg fixed to the supported 240-degree external observation contract; independent cbf_fov_deg; epsilon_d→cbf_damping_factor; kappa; safe radius; safety margin all reach the actual actor/layer constructor. Normal-profile footprint/clipping, mode and classification are explicitly checked instead of silently ignored. Direct unprofiled actor default remains CBF FOV240 for compatibility, while the upstream resolved factory creates CBF FOV180 with observation FOV240.
6. **PPO application:** explicit alpha_min, intervention_coefficient, alpha_penalty_coefficient, effective actor_smoothness_coefficient and critic_smoothness_coefficient, action_range_low/high. Four actual PPO-update optimizer-gradient comparisons test zero against nondefault coefficient .37 and compare the gradient difference with independent auxiliary-loss gradients. Custom range bounds are checked against actual regularization metrics; paper floor .1/weight .1 has a scalar gradient diagnostic. SMOOTHNESS_HELPER_SCALE=.05 preserves historical helper/metric units; normalized weights inside the helper and one outer application yield effective .05/.005 defaults without double weighting. Untouched Task 3 real rollout/update and current-state graph regressions pass.
7. **Bootstrap repair:** the three adapter main functions now derive sea_root=adapter_root.parent and select/purge bundled rsl_rl before importing dependent adapters. No post-adapter duplicate purge remains. AppLauncher ordering is unchanged. Six real isolated-process bootstrap tests cover fresh/preloaded real core and verify adapter/core class identity; three AST tests bind production ordering without importing the applications or simulator modules.

## Commands and outputs

All commands below run in ../SEA-Nav-Code-batch4 with ../sea-nav-cpu-venv/bin/python. Tests disable bytecode and pytest cache.

CBF RED:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider training/rsl_rl/tests/test_cbf_lse_layer.py tests/test_cbf_shield.py --tb=short
```

Definitive result: 48 failed in 1.24 s, from missing diagnostics/apply_alpha and validation/ablation behavior. Initial fixture integer dtype construction failures were corrected before this definitive RED. CBF+Task3 GREEN: 62 passed in 1.47 s.

Factory RED command substitutes training/rsl_rl/tests/test_policy_factory.py: 24 failed, 1 passed in 1.59 s. Factory+Task3 GREEN: 39 passed in 2.16 s. Self-review added a focused ignored-value-loss-override regression: 1 failed (DID NOT RAISE), followed by 1 passed after explicit factory rejection.

The bootstrap scope-extension command substitutes tests/test_cbf_import_boundary.py: RED 9 failed in 5.98 s, GREEN 9 passed in 6.61 s. Failures included actual ModuleNotFoundError in isolated processes and incorrect AST purge/import ordering.

Final entire explicit suite:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
```

Output: **169 passed in 11.33s**. Exit 0. Repeated after commit ac0567e: **169 passed in 11.05s**, exit 0; only local plan/resume remained modified afterward. Earlier full-suite failures were five inherited CBF constructions with nonzero footprint and no ablation identity; only those constructions gained explicit ablation identity.

Staged-file Gate A:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" --report ../task-4-gate-a.json
git diff --cached --check
```

Output: {"report":"../task-4-gate-a.json","status":"passed_with_blockers"}. Exit 0. Report is outside the worktree at work/task-4-gate-a.json. Python syntax: 62 tracked files compiled; complete baseline: 65 files present; static boundary: 20 package files, 8 simulator-bound static-only; CPU imports: 7 packages; actual actor/value/PPO/storage smoke passed; Isaac Gym Preview 4 absent/blocked. Whitespace check produced no findings.

Python 3.8 grammar: ast.parse(..., feature_version=(3,8)) passed for core layer, actor, PPO, packaged factory, and adapter shield. This is grammar evidence on the dedicated Python3.10/Torch2.6 CPU environment, not a claim of executing the old system Torch.

## Review, limitations and Task 6 handoff

Self-review covered every source diff, mapping and rejection, default numerical equivalence, gradients, buffer names and scriptability, inherited CBF-only gate edits, and bootstrap ordering. No Task 3 test or graph/state behavior was weakened. Fifteen exact owned paths are committed, including the local task-4-log.md; only local task_plan.md and resume_state.json remain modified and unstaged for controller preservation. Git reports no source/test dirt or untracked outputs.

Accepted paper_v1 is still blocked by paper_table_action_bounds and perception_timing_semantics; no registry/profile/hash change was made. Diagnostic CPU math does not resolve those scientific ambiguities.

The new factory explicitly rejects value_loss_coef !=1 because inherited PPO's main value loss is hardcoded 1.0; silently accepting a changed override would be false application evidence. Direct legacy PPO behavior is otherwise unchanged for that older option.

Full startup profile application, CLI/manifest binding, reward/perception consumers, and the layer-replacement sites belong to Task 6. Current smoke default footprint .55 now correctly fails unless supplied a named ablation identity; no automatic identity or hidden delta is inserted. Standard profile startup should use zero footprint, and a true ablation path must carry its own declared identity. The newly introduced shared-package import dependency itself is repaired now in all three existing bootstrap blocks.

AGENTS.md requires the relevant IsaacLab smoke for adapter completion; it cannot be executed because the simulator is unavailable. This delivery is explicitly CPU/static scope only, with runtime smoke blocked. Formal 100-trial metrics, real hardware and publication-rights evidence remain blocked/deferred. No simulator mocks, named branches, remote mutations, QP mode, or extensible ablation framework were introduced.
