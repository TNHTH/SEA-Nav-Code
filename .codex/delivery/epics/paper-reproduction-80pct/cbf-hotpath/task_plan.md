# Checked CBF hot-path implementation

## Identity and scope

Detached BASE: `a453b15c7b4e84f97bc37308f682fdde79e6fbbc`; worktree `work/SEA-Nav-Code-cbf-hotpath`; owner `/root/implement_batch7/v2_only_absence_review`. Scratch directory is this repository-local `cbf-hotpath/` directory, not the repository root. Shared coordination revision 12 read and local registration completed before source edits.

Owned production: `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py`, `sea_nav_current_isaaclab_full_method/adapters/cbf_shield.py`. Owned tests: `tests/test_cbf_hotpath.py`. Owned records: this directory's task_plan/findings/progress. Root task_plan/resume_state registration edits stay outside the functional commit. All existing actor/PPO/exporter/fixture/runtime/configuration/standalone differential-drive package paths are read-only.

## Stages

1. [done] Establish focused RED operator/routing tests against accepted BASE (nine expected failures).
2. [done] Implement one aggregate checked decision with original-error fallback and shared command/intermediate computation; preserve adapter raw-ray preprocessing.
3. [done_cpu_scope] Verify error order/messages, golden outputs and gradients, actual actor/PPO and script/save/load/export routes; full CPU/static 579 passed/2 CUDA skips, Gate A five passed/four blocked, Python 3.8 grammar and tree hygiene checked.
4. [in_progress] Six-path candidate ready for one commit; the actual candidate OID and postcommit results will be returned to the controller and recorded in excluded local registration to avoid a self-referential evidence commit. Independent fixed-commit review and real runtime gates remain pending/blocked.

## Acceptance and limitations

Valid ordinary core/zero-footprint/named .55 m ablation calls at B=2/2048: one scalar extraction, three mathematical sums, no norm/minimum/post-output-residual work. Invalid shapes precede dynamic checks; unsupported sparse/quantized/complex/meta/mixed-device cases go directly to the prior short-circuit validator. Every dynamic public/actor/export call stays checked. No unchecked/async/startup-only shortcut or zero-sync claim. Ordinary and diagnostic outputs/gradients share Eq.4, preserve state_dict and yaw, and adapter raw rays are validated before clipping. Independent review, real IsaacLab smoke and legacy Torch/CUDA/hardware remain separate unverified gates.
