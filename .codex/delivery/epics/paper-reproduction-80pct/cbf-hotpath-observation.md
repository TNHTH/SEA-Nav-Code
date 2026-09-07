# CBF validation hot-path observation for final review

Date: 2026-09-07. Inspected integrated source `a660d74af257d252b734eec38ae769312c6e90f7`, executed in its coordination-only successor `5b99f456fbfb9eb2e6fc8eae5a03a1aa584e2b43`. This bounded controller diagnostic did not modify source/tests, start a simulator, or supersede the completed Task 4 spec/quality review.

## Observation

`training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:40` evaluates three tensor finiteness conditions as Python booleans, and lines 42/44 do the same for positive rays/alpha. `forward` calls the diagnostic path and this validation on every ordinary invocation. The adapter inherits the same validation before footprint preprocessing.

A scoped real eager-CPU TorchDispatchMode observes **five `aten._local_scalar_dense` extractions** for each ordinary core/adapter forward; the same mathematical `_compute` call observes **zero** and produces an equal command. This establishes scalar extraction in the current hot path. It does not measure CUDA latency or establish a robot/simulator performance regression.

The complete Task 4 contract requires deterministic invalid-input rejection, so bypassing checks without a reviewed validation boundary is not an acceptable automatic fix. Final whole-branch review must reconcile that validation contract with the design's device-to-host hot-path restriction and decide the smallest safe treatment; this note does not pre-grade severity. Consider the tensor diagnostics unconditionally calculated by ordinary forward in the same bounded review, without assuming their CPU/GPU cost from source alone. Preserve Eq. 4, all golden vectors, invalid-input checks, Task 3 state identity and TorchScript behavior if a fix is required.

## Executed command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python - <<'PY'
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
from sea_nav_current_isaaclab_full_method.adapters.cbf_shield import FootprintAwareLSECBFLayer

class ScalarProbe(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.scalars = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) == "aten._local_scalar_dense.default":
            self.scalars += 1
        return func(*args, **(kwargs or {}))

u, rays, alpha = torch.zeros(2, 3), torch.ones(2, 41), torch.ones(2, 1)
for layer in (ExactLSECBFLayer(), FootprintAwareLSECBFLayer()):
    probe = ScalarProbe()
    with probe:
        actual = layer(u, rays, alpha)
    reference, _ = layer._compute(u, rays, alpha)
    assert torch.equal(actual, reference)
    print(type(layer).__name__ + " scalar_extractions=" + str(probe.scalars))
    control = ScalarProbe()
    with control:
        layer._compute(u, rays, alpha)
    print("math_core_scalar_extractions=" + str(control.scalars))
print("torch=" + torch.__version__ + "; CPU only, no CUDA timing")
PY
```

Exit 0. Exact output:

```text
ExactLSECBFLayer scalar_extractions=5
math_core_scalar_extractions=0
FootprintAwareLSECBFLayer scalar_extractions=5
math_core_scalar_extractions=0
torch=2.6.0+cpu; CPU only, no CUDA timing
```

No performance fix is assigned to Task 5; it remains the sole active source writer. This observation is an explicit input to the final whole-branch review, not a new concurrent CBF batch.
