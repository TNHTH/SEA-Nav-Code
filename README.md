# SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments


**Project Website**: [https://11chens.github.io/sea-nav](https://11chens.github.io/sea-nav/)

<p align="center">
  <img src="imgs/terser.jpg" width="80%">
</p>

---

## Installation

The installation notes below preserve the upstream stack description. This repaired checkout has CPU/static verification only; they are not an exact tested environment lock. Isaac Gym Preview 4, IsaacLab, controller interfaces/provenance and simulator checkpoint continuation remain blocked until separately validated. The upstream source reference is `11chens/SEA-Nav-Code@fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96`.

### 1. Environment Setup
Create a new Python virtual environment with Python 3.8:
```bash
conda create -n sea_nav python=3.8
conda activate sea_nav
```

### 2. Install Isaac Gym
- Download and install Isaac Gym Preview 4 from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym).
- Install the python package:
```bash
cd isaacgym/python && pip install -e .
```

### 3. Install rsl_rl
- Clone this repository
- Install the package:
```bash
cd training/rsl_rl && pip install -e .
```

### 4. Install legged_gym
```bash
cd training/legged_gym && pip install -e .
```

---

## Usage

### Current CPU checks and startup

Run from the repository root. Select an existing dedicated CPU Python with Torch 2.6 and the CPU test dependencies; this does not establish compatibility with a simulator's older Torch. `SEA_NAV_LAUNCHER` must name the real executable selected for that runtime. `SEA_NAV_ASSET_ROOT` is an existing local asset directory, disjoint from the new `SEA_NAV_RUN_ROOT`. Set `SEA_NAV_COMMIT` to the full commit that produced the code; the resolved configuration hash is computed from the selected typed profile and passed explicitly to persistence.

```bash
SEA_NAV_PYTHON=/absolute/path/to/cpu-venv/bin/python
SEA_NAV_LAUNCHER=/absolute/path/to/runtime/launcher
SEA_NAV_ASSET_ROOT=/absolute/path/to/assets
SEA_NAV_RUN_ROOT=/absolute/path/to/new-run
SEA_NAV_COMMIT=$(git rev-parse HEAD)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" "$SEA_NAV_PYTHON" -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
```

The following command validates the actual Gym parser, inputs and CPU consumers in an isolated child, without importing Gym or Torch into the Gym parent or launching simulation. Controller/runtime prerequisites remain independently blocked.

<!-- checkpoint-example: gym-fresh -->
```bash
"$SEA_NAV_PYTHON" -B training/legged_gym/legged_gym/scripts/train.py \
  --config sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml \
  --producer-commit "$SEA_NAV_COMMIT" --launcher "$SEA_NAV_LAUNCHER" \
  --asset-root "$SEA_NAV_ASSET_ROOT" --run-root "$SEA_NAV_RUN_ROOT" \
  --task go2_pos_rough --headless --preflight-only
```

### Checkpoint initialization, continuation and inference

CPU initialization creates model-only weights at iteration zero, an adjacent schema-v2 manifest, immutable payload and a separate metadata JSON. Choose the stack that will consume them; stack/profile identity hashes must match.

<!-- checkpoint-example: gym-init -->
```bash
"$SEA_NAV_PYTHON" -B sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py \
  --config sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml \
  --runtime-stack isaac_gym_preview4 --producer-commit "$SEA_NAV_COMMIT" \
  --output-manifest "$SEA_NAV_ASSET_ROOT/init.manifest.json" --seed 42
```

Training accepts mutually exclusive `--init-checkpoint-manifest PATH` (model-only, iteration zero) or `--resume-checkpoint-manifest PATH` (model plus real Adam state and completed PPO update count). With no input manifest it starts fresh. `learn()` returns the exact final `model_N.manifest.json`; use that returned path, without latest-run or lexical filename selection. Continuation restores model/optimizer/iteration, without RNG or physical trajectory restoration.

Gym play accepts only inference manifests and applies model weights. This parser check uses the initialization above; replace it with the explicitly selected trained manifest for an eventual validated runtime:

<!-- checkpoint-example: gym-play -->
```bash
"$SEA_NAV_PYTHON" -B training/legged_gym/legged_gym/scripts/play.py \
  --config sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml \
  --producer-commit "$SEA_NAV_COMMIT" --launcher "$SEA_NAV_LAUNCHER" \
  --asset-root "$SEA_NAV_ASSET_ROOT" --run-root "$SEA_NAV_RUN_ROOT" \
  --checkpoint-manifest "$SEA_NAV_ASSET_ROOT/init.manifest.json" --preflight-only
```

Loading requires the actual runtime's explicit `weights_only=True` support plus a successful v2 round trip and kernel-sealed file snapshots. An older Torch lacking this API is blocked; the CPU Torch result cannot authorize simulator resume. Historical raw `.pt`, numeric `--checkpoint`, `--resume`, and automatic latest-run selection are rejected. Legacy checkpoint conversion is not provided. Create native v2 model-only initialization with the command above, or use an explicitly selected native v2 checkpoint returned by a validated training run; old files cannot be passed to these entry points.

Adapter commands and opt-in trace behavior are documented in [the adapter README](sea_nav_current_isaaclab_full_method/README.md). Formal paper metrics, accepted `paper_v1`, public redistribution/provenance closure and hardware deployment remain separate blocked work.

---

## Deployment (Coming soon)
For instructions on deploying to real-world robots, please refer to the [deployment README](deployment/README.md).
