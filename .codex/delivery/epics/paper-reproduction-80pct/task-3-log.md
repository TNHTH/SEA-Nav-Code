# Task 3 implementation evidence

Date: 2026-09-07. Owner: `/root/implement_batch3`. Detached base: `71c2e77d41a46e67927b6de95643adf0fada8c77`.

Read root AGENTS, corrected Task 3 brief, complete PPO audit, design section 6, TDD skill plus writing-good-tests reference, and verification-before-completion skill. Registered write scope in local task_plan/resume before source edits. No subagents dispatched.

Original scope: three actor/PPO source files and `training/rsl_rl/tests/test_ppo_action_state_identity.py`, this log, and the explicitly authorized primary-checkout Task 3 report. Controller approved one extension after the inherited suite exposed its obsolete post-sample projection requirement: replace only that test and its direct-main registration in `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`. Local plan/resume register the extension and remain unstaged for controller preservation/restoration.

## Changes and coverage

- Both actors expose differentiable pure `action_mean_for(observations, **kwargs)`. Ordinary queries leave a first distribution uncreated and preserve an existing distribution/mean alias.
- CBF `_compute_safe_action_mean` returns `(u_s, alpha, rays_real, u_bar)` without assigning actor state; `forward` owns assignments. `act` returns its Normal draw unchanged after the one mean-stage CBF evaluation.
- PPO smoothness accepts keyword-only `orig_mu`/`orig_values`, uses pure mean queries, and preserves distribution/mean/std and all four CBF auxiliary references and exact values. Update supplies the live primary minibatch mean/value objects. Unsupported recurrent or missing-pure-query actors fail at construction without an incomplete fallback.
- Fourteen focused cases cover exact seeded sampling, ordinary first-query ownership, both actor pure-query state/gradients, supplied/default smoothness state and supplied-original graph reachability, unsupported contracts, and real repeated collection/storage/update loops.
- Real integration uses two environments, two steps, 38-value observations, two minibatches, two epochs, learning rate 0.001, then collects and updates a second rollout. All generator tuples are matched by observation content to independently saved samples/means/std/Normal likelihood. Actual likelihood calls receive the yielded stored action object, never the newly drawn update sample. Current distribution, mean alias, mu/std and CBF fields retain object identity and values through smoothness and immediately before every real optimizer step. Alpha loss gets the current alpha object; reported intervention and regularization match captured current-state terms. Stored targets are detached; live finite nonzero gradients and changed parameters prove learning, with first-batch ratio one and later ratios changed. CBF current/interpolated outputs differ and intervention is active.
- The updated inherited test keeps noise initialization coverage and uses a real actor, real mean-stage CBF, and a seeded Normal likelihood oracle. Its draw is explicitly chosen to make a forbidden second CBF pass observable.

## Commands and results

All commands run from `work/SEA-Nav-Code-batch3`. Interpreter: `../sea-nav-cpu-venv/bin/python`, Python 3.10.12 / Torch 2.6.0+cpu; target source grammar remains Python 3.8.

1. RED before source edits:

   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest -q training/rsl_rl/tests/test_ppo_action_state_identity.py`

   Exit 1, 14 failed in 1.52s: actual sampled action mismatch; missing pure actor methods; active distribution overwritten; unsupported smoothness keywords; missing constructor rejection; update omitted original mean/value identity. No simulator collection/import errors.

2. Same focused command after atomic source repair: exit 0, 14 passed in 1.34s. Adding supplied-original gradient assertions and std-gradient checks exposed an overstrict test assumption: a later fully clipped surrogate can legitimately yield zero std gradient (13 passed / 1 failed). Corrected to finite std gradients every step and at least one nonzero std gradient per rollout, preserving all policy-module nonzero checks; same focused command then passed 14 in 1.35s. No source change was needed for that harness correction.

3. Inherited plus focused suite:

   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py training/rsl_rl/tests/test_ppo_action_state_identity.py`

   Initial exit 1, 84 passed / 1 failed in 4.26s: inherited `test_actor_noise_std_and_post_sample_projection` demanded the now-forbidden post-sample transformation. Requested controller scope extension; received explicit approval and updated local scope. No skip/deletion or simulator substitute.

4. New inherited test targeted command:

   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest -q sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py::test_actor_noise_std_and_mean_stage_sampling`

   Exit 0, 1 passed in 0.84s. Self-review mutation then reintroduced the second CBF pass with apply_patch and ran:

   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest -q --tb=short training/rsl_rl/tests/test_ppo_action_state_identity.py::test_sample_is_exact_draw_from_mean_stage_normal sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py::test_actor_noise_std_and_mean_stage_sampling`

   First mutation run: 1 failed / 1 passed because the Gate seed happened to draw an action on which a second CBF pass was inactive. Strengthened the Gate fixture to seed 0 and assert the draw makes a second pass observable. Repeat mutation: exit 1, 2 failed in 0.89s, exact sampling mismatch in both tests. Restored the production fix with apply_patch.

5. Final precommit full suite command from step 3: exit 0, **85 passed in 4.13s**.

6. Standalone direct-main Gate registration:

   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

   Exit 0; JSON status `passed`, all 16 checks including renamed real-actor sampling test.

7. Syntax and whitespace:

   `PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m compileall -q training/rsl_rl/rsl_rl`

   Exit 0. (An earlier chained attempt did not reach compileall because its preceding overstrict std-gradient test failed; final execution above succeeded.)

   `../sea-nav-cpu-venv/bin/python -c 'import ast, pathlib; paths=[pathlib.Path("training/rsl_rl/rsl_rl/modules/actor_critic.py"), pathlib.Path("training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py"), pathlib.Path("training/rsl_rl/rsl_rl/algorithms/ppo.py"), pathlib.Path("training/rsl_rl/tests/test_ppo_action_state_identity.py"), pathlib.Path("sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py")]; [ast.parse(p.read_text(), filename=str(p), feature_version=(3, 8)) for p in paths]; print("Python 3.8 grammar: 5 files passed")'`

   Exit 0, all 5 changed Python files parse with Python 3.8 grammar. `git diff --check` exit 0.

## Self-review and boundaries

Read the complete source diff and authorized Gate-test diff. Original smoothness coefficients, CBF geometry, rollout storage, runner, immutable configuration, and runtime code remain unchanged. The graph tests respect the existing CBF policy latent detach while checking ordinary actor+encoder, CBF nav/backbone/alpha, and independent critic+encoder paths. The existing `enable_shield` attribute remains for compatibility but cannot activate a forbidden post-sample pass. No simulator imports/stubs, recurrent implementation, storage redesign, branch creation, or remote writes. Evidence is CPU/static only; Isaac Gym/IsaacLab, formal metrics, hardware, and publication rights remain blocked. Mandatory implementation delta is enforced by Task 2 configuration; runtime wiring remains Task 6 and geometry/loss profile knobs remain Task 4.

Commit scope: the five source/test paths and this log. Local task_plan/resume edits intentionally remain unstaged per controller instruction; no unrelated files are cleaned. Full commit SHA and fresh postcommit verification are recorded in the primary Task 3 report.
