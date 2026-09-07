# Task 4 implementation log

Date: 2026-09-07. Worker: /root/implement_batch4. Detached base: 17e53cf1e0b4ce476b43bdc932baa699cecf3571. Scope registered in local task_plan.md/resume_state.json before source edits. The controller later authorized only the three adapter main bootstrap blocks and tests/test_cbf_import_boundary.py, recorded before those edits.

## Delivered contracts

- Core Eq.4 diagnostics retain differentiable graphs: h_comp, Lg_h, Lg_norm_sq, r, eta_raw, clamped nonnegative eta, correction_norm, residual_before, residual_after. Positive rays and already-positive alpha are validated. An active residual is negative; no hard-safety guarantee.
- Both adapter interfaces use the same packaged core; apply(..., gamma) preserves historical raw-alpha keyword, apply_alpha accepts transformed alpha. Footprint preprocessing requires an explicit nonempty ablation/ name. Zero footprint preserves all positive raw distances.
- One versioned fixture stores seven independently calculated scalar golden cases, with 180/240-degree FOV, uniform close/asymmetric/inactive hazards, and nearly cancelling opposing hazards.
- Immutable Task 2 projections materialize detached constructor values. Real factory entrypoints verify resolved integrity. Unknown/missing/conflicting fields, unsupported semantics, and missing activation deltas fail. The observation contract stays 240 degrees independently of upstream CBF 180.
- PPO uses explicit effective shield/smoothness weights, alpha floor, and range bounds. SMOOTHNESS_HELPER_SCALE preserves legacy helper/metric units and is applied exactly once. Four actual optimizer-gradient comparisons test zero and nondefault coefficients. Range loss executes configured bounds. The inherited ignored non-1 value_loss_coef option is explicitly rejected by the new factory.
- Three main bootstrap blocks choose the bundled package before dependent adapter imports, derive sea_root from adapter_root.parent, and preserve AppLauncher order. Six isolated-process real-package tests prove identity with/without preloaded real core; three AST checks bind production order.

## RED/GREEN evidence

Interpreter: ../sea-nav-cpu-venv/bin/python. All pytest runs used PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=$PWD/training/rsl_rl, -q -p no:cacheprovider.

1. CBF/adapter definitive RED: 48 failed (missing diagnostics/apply_alpha, validation and ablation enforcement). An initial integer-dtype fixture construction error was corrected before this definitive RED.
2. CBF/adapter plus untouched Task 3 regressions GREEN: 62 passed in 1.47 s, including real torch.jit.script on core and adapter layers.
3. Factory/PPO RED: 24 failed, 1 passed (missing factory and unsupported constructor coefficients).
4. Factory/PPO plus untouched Task 3 regressions GREEN: 39 passed in 2.16 s.
5. First full suite: 153 passed, 5 inherited footprint tests failed. Only their CBF constructions gained explicit ablation identity; subsequent full suite: 158 passed.
6. Self-review found a legacy ignored value_loss_coef override exposed by the factory: focused RED 1 failed, then GREEN 1 passed after explicit rejection. Added scalar paper-floor diagnostic and actual configured range-loss checks.
7. Bootstrap extension RED: 9 failed (three AST ordering failures, six isolated-process import failures); GREEN: 9 passed in 6.61 s.
8. Final full CPU command:

   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py

   Result: 169 passed in 11.33 s; exit 0.

9. Python 3.8 grammar check on the five shared changed source files: passed. git diff --check: passed. Gate A uses ../task-4-gate-a.json outside the checkout; staged-file result is recorded in the primary task-4-report.md.

## Self-review and remaining boundaries

Reviewed every source diff, projection field mapping, class/state_dict compatibility, and inherited test edit. No Task 3 test or graph/state ownership changed. New layer buffers preserve ray_unit_vectors key. No simulator import or stand-in was used in CPU tests; only the extracted real package bootstrap executes.

Accepted paper_v1 remains blocked by unresolved registry rows. Diagnostic converters and paper math tests do not create accepted run identity. Full runtime profile/CLI/reward/perception/manifest wiring remains Task 6. In particular, the old smoke default footprint .55 now requires explicit ablation identity and must not be called an upstream/paper run. IsaacLab smoke and Isaac Gym execution remain blocked by unavailable runtimes; formal metrics, publication rights and hardware remain unverified/blocked. No remote mutation or named branch creation.
