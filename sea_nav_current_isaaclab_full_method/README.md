# SEA-Nav Current IsaacLab Full-Method Adapter

This directory holds the tracked IsaacLab adapter for SEA-Nav. Its results have the separate `isaaclab_adapter` runtime identity.

It is separate from `training/**`; upstream commit `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96` remains the source reference. The earlier first-wave scope below is historical context; current CPU contracts also cover real PPO checkpoint continuation and explicit startup/manifest boundaries:

- observation/history shape;
- 41 ray angle/range contract;
- LSE-CBF shield behavior with upstream `softplus(alpha_raw)` semantics and no additive alpha floor;
- per-env ACSI-style collision replay state sampling;
- source_alpha_only alpha filter with no queue delay;
- upstream terminal defaults (`stay_time=150`, contact termination enabled);
- trace schema;
- formal eval manifest with replay disabled.

Gate A passes do not establish simulator viability. IsaacLab/controller prerequisites, physical reset/replay, simulator optimizer continuation, formal metrics and accepted paper identity remain blocked.

Run the following from the repository root with explicit `SEA_NAV_*` variables as documented in the root README. Create an initializer for this stack in an empty target path:

<!-- checkpoint-example: lab-init -->
```bash
"$SEA_NAV_PYTHON" -B sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py \
  --config sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml \
  --runtime-stack isaaclab_adapter --producer-commit "$SEA_NAV_COMMIT" \
  --output-manifest "$SEA_NAV_ASSET_ROOT/init.manifest.json" --seed 42
```

Pure trainer preflight accepts model-only iteration-zero initialization:

<!-- checkpoint-example: lab-warm-start -->
```bash
"$SEA_NAV_PYTHON" -B sea_nav_current_isaaclab_full_method/train_full_method_ppo.py \
  --config sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml \
  --producer-commit "$SEA_NAV_COMMIT" --launcher "$SEA_NAV_LAUNCHER" \
  --asset-root "$SEA_NAV_ASSET_ROOT" --run-root "$SEA_NAV_RUN_ROOT" \
  --init-checkpoint-manifest "$SEA_NAV_ASSET_ROOT/init.manifest.json" \
  --iterations 4 --rollout-steps 4 --preflight-only
```

The ACSI trainer has the same mutually exclusive init/resume interface. For continuation, set `SEA_NAV_RESUME_MANIFEST` to the explicit manifest returned by a completed training run and keep it with its adjacent payload under the asset root. Its profile/stack hash must match; the input must contain real Adam state.

<!-- checkpoint-example: lab-resume -->
```bash
"$SEA_NAV_PYTHON" -B sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py \
  --config sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml \
  --producer-commit "$SEA_NAV_COMMIT" --launcher "$SEA_NAV_LAUNCHER" \
  --asset-root "$SEA_NAV_ASSET_ROOT" --run-root "$SEA_NAV_RUN_ROOT" \
  --resume-checkpoint-manifest "$SEA_NAV_RESUME_MANIFEST" \
  --iterations 4 --rollout-steps 4 --preflight-only
```

Smoke inference loads only model weights and remains a single-environment, no-replay diagnostic:

<!-- checkpoint-example: lab-inference -->
```bash
"$SEA_NAV_PYTHON" -B sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py \
  --config sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml \
  --producer-commit "$SEA_NAV_COMMIT" --launcher "$SEA_NAV_LAUNCHER" \
  --asset-root "$SEA_NAV_ASSET_ROOT" --run-root "$SEA_NAV_RUN_ROOT" \
  --checkpoint-manifest "$SEA_NAV_ASSET_ROOT/init.manifest.json" --steps 16 --preflight-only
```

Trainers disable trace by default; `--trace "$SEA_NAV_RUN_ROOT/trace.jsonl"` enables the existing four-action-stage JSONL evidence. Smoke tracing remains enabled. All output paths must be new, mutually distinct and inside the selected run root. `--preflight-only` creates no output and starts no simulator. Removing it still reaches the real controller/runtime prerequisite blocker in this checkout.

Checkpoints use completed PPO updates, immutable payload generations and a manifest published last. `learn()` returns the final manifest directly. Continuation restores model, optimizer and iteration only; it does not restore RNG or physical trajectories. Load capability is checked in the actual runtime interpreter with `weights_only=True` and a real v2 round trip, including kernel-sealed snapshots. Legacy raw files are rejected and legacy checkpoint conversion is not provided. Use native v2 initialization above or a native v2 checkpoint from a validated run.
