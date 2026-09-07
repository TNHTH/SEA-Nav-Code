# Deferred serial CBF hot-path repair

Status: prepared after Task6 acceptance while Task7 is the sole source writer.
**Not active, not an ownership grant, not a performance exception, not P2 closure.**
Activate only after Task7 independent whole acceptance, ordered integration and
fresh primary CPU/static verification. Register the exact resulting BASE and
one detached implementation owner in task_plan/resume_state before source edits.

## Existing evidence and boundary

Read cbf-hotpath-review.md and cbf-checked-path-probe.md completely. The former
establishes actual core/adapter and PPO reachability; the latter is an in-memory
prototype, not a checked-in fix. Controller rechecked unchanged production
core/adapter source at primary f1daa125c9aec41fa741431200b9b07f3e250341.

Ordinary forward currently takes the diagnostic route: five batch-wide scalar
extractions, a discarded post-command residual, a correction norm, and two
additional discarded adapter minima. The tested aggregate predicate is a
5-to-1 valid-input extraction candidate, with original precise error fallback.
It still performs validity reductions and a host decision. A direct async
assertion substitution loses its check after actual Torch2.6 script/save/load;
do not use it, tracing, a private unchecked actor route, or startup-only checks.

The goal is an actual minimal checked-path optimization plus removal of
discarded diagnostics. The residual host decision remains explicit. Neither a
documentation-only budget exception nor removal of diagnostics alone closes
the established finding. Independent review must evaluate the implemented
improvement and remaining design Section7/10 tension; do not pre-grade it.

## Proposed narrow inventory (pending activation)

Production:

- training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py
- sea_nav_current_isaaclab_full_method/adapters/cbf_shield.py

Tests:

- tests/test_cbf_hotpath.py (new focused operator/routing regression)
- training/rsl_rl/tests/test_cbf_lse_layer.py (only necessary checked/export coverage)
- tests/test_cbf_shield.py (only necessary adapter checked/export coverage)

All actor/PPO/runner/runtime/configuration/exporter code stays read-only. The
real Gym exporter can be exercised through an isolated source/AST boundary
without importing proprietary Gym; needing to change it requires a separate
evidence-backed scope decision after Task7. Golden fixture values and previous
PPO identity tests must not be weakened or rewritten to fit the optimization.

## Implementation and test contracts

1. Keep one Eq.4 implementation. A shared mathematical-intermediate helper may
   feed ordinary output and optional diagnostic construction; returning index0
   of an already constructed diagnostic tuple is not an optimization. The
   adapter must preserve raw-ray validation before footprint preprocessing and
   apply its named ablation identically in both methods. Do not accidentally
   inherit a new base forward that skips adapter preprocessing.
2. Retain every dynamic check, exact relevant error precedence, public return
   types, diagnostic keys/gradients, FOV/ray buffers, yaw, damping=1, state_dict
   names, raw-logit versus positive-alpha wrappers, and TorchScript export.
   Ordinary integer/float64 inputs must not acquire an invented dtype ban.
   Keep metadata checks before reductions; unsupported dtype/device/layout
   cases must fail consistently instead of bypassing validation. An aggregate
   predicate must not run a later unsupported comparison ahead of an earlier
   nonfinite-input error.
3. RED then GREEN at B=2 and B=2048 for core, zero-footprint adapter and a named
   .55m footprint ablation. Count actual Torch operators, not wall-time ratios:
   a normal supported valid call should require one scalar extraction, no
   diagnostic norm/minimum work, and no post-output residual reduction. Detail
   failure/fallback costs separately. Diagnostics remain an explicit API whose
   fields still match golden numeric/gradient contracts.
4. Real actor/PPO routing: T=2, E=2, M=2, only real-layer hooks counted. Expect
   two collection and eight update layer calls, with the new per-call budget;
   no fake additive shield or mocked optimizer. Re-run Task3 distribution,
   action-likelihood and auxiliary-state ownership regressions unchanged.
5. Compare checked ordinary/diagnostic command and input gradients with the
   seven committed golden fixtures (180/240 degree coverage). Check footprint
   preprocessing independently and retain zero-footprint subminimum rays.
   Preserve existing state_dict loading and wrapper alpha transformed once.
6. Real script -> save -> load for core/adapter/ablation, both public APIs,
   dynamic NaN/Inf/nonpositive and malformed-shape rejection. Cover transform
   underflow/overflow; exp2/Softplus are not proofs of runtime validity. Check
   the actual exporter-selected ordinary route without a simulator import.
7. Fresh complete CPU/static suite, Gate A, Python3.8 grammar, no-baseline-
   deletion and ignored-output inventory. Use the dedicated Torch2.6 CPU
   interpreter and cache-disabled commands in final-verification-protocol.md.
   CPU counts do not certify CUDA timing, legacy Torch, simulator or hardware.

Provide focused commits and a full report with actual failed/green commands,
exact OIDs/paths, remaining costs and blockers. Independent fixed-commit review
is required before integration; the controller does not edit production code.

## Additional candidate error-precedence probe

Controller, primary cb24dc6520a9b8dd4d7e99f8afaafbd82534e4bc, Torch2.6 CPU.
An in-memory simplified aggregate validator (same dynamic expression and
complex/device fallback as the earlier candidate) was compared with the
unchanged real layer on metadata-valid shapes. Inputs were NaN u_bar[2,3],
positive alpha[2,1], and either native sparse-COO rays[2,41] or native quantized
QUInt8 rays[2,41]. Production source was not changed.

| Rays | Actual baseline | Aggregate candidate |
|---|---|---|
| sparse COO | ValueError: CBF inputs must be finite | NotImplementedError: aten::ne.Scalar unavailable on SparseCPU |
| quantized QUInt8 | ValueError: CBF inputs must be finite | RuntimeError: isfinite not implemented for QUInt8 |

Both assertions confirming a changed exception type passed (exit0, .93s).
This is a candidate flaw, **not a new production defect**: aggregation eagerly
evaluates a later unsupported input before the existing short-circuit error.
The future implementation must use metadata-based fallback for unsupported
layout/quantization as well as complex/device cases, without a new global dtype
ban. Merely preserving the earlier prototype's tested cases is insufficient.
No new supported-layout claim or TorchScript/CUDA evidence comes from this probe.

Exact command (the printed backend list is summarized in the table above):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -I -B - <<'PY'
import sys
sys.path.insert(0, 'training/rsl_rl')
import torch
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
class AggregateProbe(ExactLSECBFLayer):
    def _validate_inputs(self, u, rays, alpha):
        if (u.is_complex() or rays.is_complex() or alpha.is_complex()
                or u.device != rays.device or u.device != alpha.device):
            return super()._validate_inputs(u, rays, alpha)
        valid = (torch.isfinite(u).all() & torch.isfinite(rays).all()
                 & torch.isfinite(alpha).all() & ~(rays <= 0).any()
                 & ~(alpha <= 0).any())
        if not valid:
            super()._validate_inputs(u, rays, alpha)
def result(layer, args):
    try:
        layer(*args)
    except Exception as exc:
        return type(exc).__name__, str(exc).splitlines()[0]
    return 'accepted', ''
for label, rays in [('sparse_coo', torch.ones(2,41).to_sparse()),
                    ('quantized', torch.quantize_per_tensor(torch.ones(2,41), .1, 0, torch.quint8))]:
    args=(torch.full((2,3),float('nan')), rays, torch.ones(2,1))
    old=result(ExactLSECBFLayer(),args)
    new=result(AggregateProbe(),args)
    print(label, 'baseline=', old, 'aggregate=', new)
    assert old[0]=='ValueError' and new[0]!=old[0]
print('Bounded candidate-precedence failure confirmed; production unchanged; no CUDA or simulation.')
PY
```

## Actual exporter CPU harness feasibility

Controller at primary5d0aa490b0567cc9a8166e0d88834aadfaad7352 also executed
the actual unchanged `helpers.py::export_policy_as_jit` definition, isolated by
AST into a module containing only real Torch/copy/os dependencies. No fake Gym
module or reimplemented exporter was used. The source filename/line locations
were retained for TorchScript inspection. With seed421 and a real small CBF
actor (all hidden dimensions[8], observations zeros[2,550]), the function's
actual script/save path produced policy.pt in a disposable CPU probe directory.
After real torch.jit.load, output matched actor.act_inference with atol=rtol=0
for core, zero-footprint adapter and named .55m footprint ablation. Setting the
first current-frame ray observation to -1000 made exp2 underflow; every loaded
export rejected it with the original positive-ray error. isaacgym/legged_gym
were absent from sys.modules at completion.

Command used the dedicated CPU interpreter with PYTHONDONTWRITEBYTECODE=1,
PYTHONPATH="$PWD/training/rsl_rl", and -B, via an inline Python AST harness.
Exit0 in1.07s; three output rows each reported
`actual_export_function_script_save_load_exact_output_and_invalid_rejection=passed`.
These are baseline harness observations, not a CBF repair test or simulation
acceptance. The generated .pt files were disposable and are not delivery or
recovery artifacts. Implement the corresponding actual-function regression
within the proposed test inventory; do not replace it with a copied wrapper.
