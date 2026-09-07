# Task 6 Gym import-order boundary

Date: 2026-09-07. This is a controller integration warning from bounded local inspection of the in-progress Task 6 shared preflight, not a simulator run or an additional claimed completed review.

## Local observation

At worker committed slice `7945a9e6037fee5045cbc12ac59a054ad958b43b` (runtime_preflight/environment_profile are still uncommitted work), importing rsl_rl.runtime_preflight loads Torch via environment_profile's eager policy_factory/replay/reward/perception imports. A fresh process run from the worker root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -c 'import sys; import rsl_rl.runtime_preflight; print("torch_loaded_after_preflight_import="+str("torch" in sys.modules)); print("isaacgym_loaded="+str("isaacgym" in sys.modules))'
```

Exit 0:

```text
torch_loaded_after_preflight_import=True
isaacgym_loaded=False
```

This proves the transitive import ordering, not an installed-Gym failure on this host (Gym is absent).

## External corroboration and source boundary

- [IsaacGymEnvs issue 175](https://github.com/isaac-sim/IsaacGymEnvs/issues/175), read through GitHub's public issues API, reports the actual gymdeps.py:21 error: `PyTorch was imported before isaacgym modules. Please import torch after isaacgym modules.` This is a public user reproduction in the official project, not independently executed proprietary code or maintainer confirmation here.
- [Official IsaacGymEnvs train.py](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/train.py), retrieved from its raw main URL on this date, imports isaacgym at the beginning of launch_rlg_hydra before its Torch-dependent task/RL imports. This is supporting import-order evidence; the moving URL is not a locked simulator artifact.

No proprietary simulator was downloaded, installed or mocked. External text was treated only as evidence.

## Task 6 acceptance correction

Keep the Gym parent launch process free of Torch before its real isaacgym import. Invalid/blocked identity and pure input validation must still happen before proprietary startup and output creation. A Torch-free metadata phase or a separate CPU preflight subprocess can satisfy both contracts; do not solve this by silently importing proprietary Gym first in the CPU-only entry or by weakening the identity gate. Full constructor-value validation can occur in an isolated CPU preflight process, while the actual runtime applies the same verified settings after Gym import. Any two-phase path must bind/recheck config identity rather than trust stale values.

Use real fresh-process/AST boundary tests without fake Isaac modules to prove the refactored Gym entry does not transitively preload Torch. Simulator success remains blocked until actually exercised. Any extra source path needed for this correction must be registered; the current shared preflight/Gym entries are already Task 6 owned. This warning was sent while those callers were still being implemented, so their absence at this intermediate snapshot is not a final finding.

Also keep newly used tensor APIs capability-gated: the slice1 clock fix uses torch._assert_async on tested CPU Torch2.6. That is not evidence for old-Gym Torch compatibility. Unsupported actual runtime dependencies must produce precise startup blockers, not late AttributeError or an unsafe fallback.
