# CBF checked-path CPU prototype

Date: 2026-09-07. Reviewer: `/root/cbf_hotpath_review`. This is bounded preparation for the existing open P2, not a source change, a design ruling, a full hotspot fix, or a new acceptance gate.

Initial primary HEAD: `12e4fff0714f5753e30bf2575e740dc12a32c051`. Primary advanced during this read-only experiment to coordination successor `c3db30d661f685c6c756b2957ba62a09d375849e`; `git diff --exit-code 12e4fff0714f5753e30bf2575e740dc12a32c051 -- training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py sea_nav_current_isaaclab_full_method/adapters/cbf_shield.py` returned 0 with no diff. Input blobs: core `ddf0b59973fdcbeab9a508eb3a230ab04fd18a13`; adapter `8578a614ea228c6827570b16b9a78fa48f87c91b`; shared fixture `b0031db566a5f82b86ddf2577cbdcadd60153626`.

## Result and limits

An in-memory checked prototype reduced **valid eager CPU scalar extractions from 5 to 1**, at batch sizes 2 and 2048, for the shared core, zero-footprint adapter, and named .55 m footprint ablation. It retained the tested invalid-input behavior and real TorchScript save/load compatibility. This supports a small **5→1 checked-path optimization candidate**, not an unchecked fast path or a zero-synchronization claim. No GPU performance measurements were made.

The original validator was copied verbatim at runtime with `inspect.getsource`, renamed `_original_validate_inputs`, and retained as the detailed error path. The prototype first runs the same shape/metadata checks, falls back immediately for complex or mixed-device inputs, then combines all five dynamic predicates into one scalar boolean tensor. Only that combined result reaches a Python `if`. On failure it runs the original validator, retaining error precedence/message. Inheritance retains constructor checks, the mathematical core, ordinary/diagnostic APIs, and adapter preprocessing. The duplicate methods exist only in memory to make a bounded prototype scriptable; this is not a recommendation to duplicate production validators.

Coverage, separately exercised for all three layer variants:

- Ten dynamic invalid cases: NaN/Inf nominal command, NaN/Inf rays, zero/negative rays, NaN/Inf alpha, zero/negative alpha. Exact eager exception class and full message matched the baseline.
- Eight shape errors: action width/rank/empty batch, ray width/batch/rank, alpha rank/batch. Exact eager exceptions matched. Metadata checks precede tensor-value reductions.
- Seven dtype/precedence cases: complex action/rays/alpha, boolean rays, integer rays, float64 rays, and NaN action with complex rays. The last case proves aggregation must not run an unsupported comparison before the original nonfinite-input error. The baseline has **no explicit dtype whitelist**: integer and float64 rays succeed; complex/bool cases retain their original backend failures. The prototype preserves, rather than silently tightens, these existing boundaries.
- Four native Torch meta input placements and CPU inputs against a native meta layer buffer produced the same error as the baseline. These are real Torch error-path diagnostics, not simulator stand-ins; they **do not test a CUDA/CPU device mismatch**. Same-device CUDA, other devices, sparse/quantized/nested layouts, mixed precision, and the legacy PyTorch runtime were not certified.
- Seven existing shared-fixture input sets at 180/240 degrees yielded bit-for-bit identical command, every diagnostic tensor, and output-loss input gradients between baseline and prototype. This comparison establishes validator-refactor equivalence; it is **not a fresh independent re-derivation of Eq. 4**, and the .55 m ablation is not covered by the fixture's ordinary-profile expected values.
- Real `torch.jit.script` → in-memory save → load passed for each variant. Both public methods matched valid outputs/diagnostics, and all 18 dynamic/shape invalid cases preserved the error kind and final error message. Scripted traceback locations necessarily differ. The prototype validator graph contains one `aten::Bool`; its failure/fallback validator intentionally retains the original checks.
- Five representative invalid constructor configurations preserved exact errors. Construction is inherited, not reimplemented.

**Ordinary forward still constructs unused diagnostics.** This prototype does not touch `cbf_lse_layer.py:47–65,73–74` or adapter `cbf_shield.py:59–61`: correction norm, post-command residual, and adapter ray minima are still built and discarded by an ordinary eager call. Predicate reductions also remain five reductions; aggregation adds boolean tensor operations. Invalid input may cost more because it evaluates the combined predicate and then reruns detailed checks. No throughput improvement or total operator-work reduction is asserted. Real actor/PPO routing was established in `cbf-hotpath-review.md` and was not rerun with this prototype.

## Immediate exceptions and the remaining decision

For the current ordinary Python/TorchScript checked interface, returning successfully or raising based on an arbitrary dynamically computed device tensor requires a host-observable decision before the call completes. The aggregate boolean makes that dependency **one decision on valid input**, not zero. Startup validation cannot decide validity of later observations/network outputs. Merely hiding the extraction inside an assertion operator is not evidence that the host/device dependency disappeared.

A second bounded CPU experiment compared two possible assertion substitutions:

- `torch._assert` keeps the check, but eager dispatch exposes one `_local_scalar_dense` and the saved/loaded script has `aten::Bool` plus `prim::If`/`RaiseException`. It also raises `AssertionError`, not the existing precise `ValueError`, so it is not a drop-in boundary-preserving replacement.
- `torch._assert_async` eager CPU dispatch exposes `_assert_async.msg` with no separate `_local_scalar_dense`; this does not prove its C++ implementation makes no host decision. More importantly, in this exact Torch 2.6 CPU experiment, **script/save/load removes the assertion from the graph**. The loaded module returns a NaN output for invalid input without an exception. Its graph contains only the output multiplication. Thus this direct substitution fails the current export/error contract even before CUDA is considered.
- The installed `_assert_async` documentation explicitly says CUDA errors may surface only at a later kernel launch and a failed assertion trashes the CUDA context. That behavior is not an immediate, recoverable, precise invalid-input exception. This statement is documentation evidence; no CUDA execution was attempted.

These experiments exclude the tested zero-exposed-extraction substitution. They do not prove that no future compiler/runtime could provide a different genuinely synchronous checked mechanism. A new runtime, delayed failure, device-only validity masks, unchecked inputs, or an asynchronous abort cannot be substituted silently for the current contract.

**Recommendation for the controller's later boundary ruling:** the 5→1 aggregate predicate with original failure path is the smallest tested candidate that preserves this checked behavior. Keep the remaining host decision explicit, and treat removing unused diagnostics as a separate required optimization. No performance-budget exception is approved here, the specification is unchanged, and the P2 remains **open** for a serial fix and independent review after Task 7.

## Exact experiment 1 command/output

Both commands ran in the primary repository with the dedicated `../sea-nav-cpu-venv/bin/python`, no bytecode or pytest cache, no persistent prototype source, and no simulator imports. Virtual source filenames were registered only in Python `linecache` so inspect/TorchScript could retrieve the in-memory class. The previous lost-output controller attempt is not used as evidence.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python - <<'PY'
import inspect, io, json, linecache, sys, textwrap, types
from collections import Counter
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
from sea_nav_current_isaaclab_full_method.adapters.cbf_shield import FootprintAwareLSECBFLayer

# No file is created at this virtual source name. linecache supports inspect/script.
original = textwrap.dedent(inspect.getsource(ExactLSECBFLayer._validate_inputs))
validation = original.split('    if not torch.isfinite')[0] + '''    if (u_bar.is_complex() or lidar_dists.is_complex() or alpha.is_complex()
            or u_bar.device != lidar_dists.device or u_bar.device != alpha.device):
        self._original_validate_inputs(u_bar, lidar_dists, alpha)
        return
    valid = (torch.isfinite(u_bar).all() & torch.isfinite(lidar_dists).all()
             & torch.isfinite(alpha).all() & ~(lidar_dists <= 0).any()
             & ~(alpha <= 0).any())
    if not valid:
        self._original_validate_inputs(u_bar, lidar_dists, alpha)
'''
methods = textwrap.indent(validation, '    ') + textwrap.indent(
    original.replace('def _validate_inputs(', 'def _original_validate_inputs('), '    ')
source = ('import torch\nfrom torch import Tensor\n'
          'from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer\n'
          'from sea_nav_current_isaaclab_full_method.adapters.cbf_shield import FootprintAwareLSECBFLayer\n'
          'class AggregateCore(ExactLSECBFLayer):\n' + methods +
          'class AggregateAdapter(FootprintAwareLSECBFLayer):\n' + methods)
filename = '/virtual/sea_nav_cbf_checked_probe.py'
module = types.ModuleType('sea_nav_cbf_checked_probe')
module.__file__ = filename
sys.modules[module.__name__] = module
linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
exec(compile(source, filename, 'exec'), module.__dict__)

class Probe(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))
    def scalars(self):
        return self.counts['aten._local_scalar_dense.default']

def valid(batch=2):
    return [torch.zeros(batch, 3), torch.ones(batch, 41), torch.ones(batch, 1)]

def replace(index, value):
    args = valid()
    args[index] = value
    return args

def outcome(layer, args, api='forward'):
    try:
        result = getattr(layer, api)(*args)
        return ('ok', result)
    except Exception as error:
        return (type(error).__name__, str(error))

def same_outcome(left, right):
    assert left[0] == right[0], (left, right)
    if left[0] == 'ok':
        assert torch.equal(left[1], right[1])
    else:
        assert left[1] == right[1], (left, right)

dynamic = [(name, replace(index, torch.full(shape, value))) for name, index, shape, value in (
    ('u_nan', 0, (2, 3), float('nan')), ('u_inf', 0, (2, 3), float('inf')),
    ('rays_nan', 1, (2, 41), float('nan')), ('rays_inf', 1, (2, 41), float('inf')),
    ('rays_zero', 1, (2, 41), 0.), ('rays_negative', 1, (2, 41), -1.),
    ('alpha_nan', 2, (2, 1), float('nan')), ('alpha_inf', 2, (2, 1), float('inf')),
    ('alpha_zero', 2, (2, 1), 0.), ('alpha_negative', 2, (2, 1), -1.))]
shapes = [
    ('u_width', replace(0, torch.zeros(2, 2))), ('u_rank', replace(0, torch.zeros(3))),
    ('u_empty', [torch.zeros(0, 3), torch.ones(0, 41), torch.ones(0, 1)]),
    ('rays_width', replace(1, torch.ones(2, 40))), ('rays_batch', replace(1, torch.ones(1, 41))),
    ('rays_rank', replace(1, torch.ones(41))), ('alpha_rank', replace(2, torch.ones(2))),
    ('alpha_batch', replace(2, torch.ones(1, 1)))]
dtype_cases = [
    ('complex_u', replace(0, torch.zeros(2, 3, dtype=torch.complex64))),
    ('complex_rays', replace(1, torch.ones(2, 41, dtype=torch.complex64))),
    ('complex_alpha', replace(2, torch.ones(2, 1, dtype=torch.complex64))),
    ('bool_rays', replace(1, torch.ones(2, 41, dtype=torch.bool))),
    ('integer_rays', replace(1, torch.ones(2, 41, dtype=torch.int64))),
    ('double_rays', replace(1, torch.ones(2, 41, dtype=torch.float64))),
    ('nan_u_complex_rays', [torch.full((2, 3), float('nan')),
                            torch.ones(2, 41, dtype=torch.complex64), torch.ones(2, 1)])]
device_cases = [(f'meta_{i}', replace(i, item.to('meta'))) for i, item in enumerate(valid())]
device_cases.append(('all_meta', [item.to('meta') for item in valid()]))

fixture = json.loads(Path('tests/fixtures/cbf_paper_damped_v1.json').read_text())
pairs = [('core', ExactLSECBFLayer, module.AggregateCore, {}),
         ('adapter', FootprintAwareLSECBFLayer, module.AggregateAdapter, {}),
         ('footprint_ablation', FootprintAwareLSECBFLayer, module.AggregateAdapter,
          {'footprint_radius_m': .55, 'algorithm_profile': 'ablation/probe'})]
for label, Baseline, Prototype, options in pairs:
    baseline, prototype = Baseline(**options), Prototype(**options)
    for batch in (2, 2048):
        counts = []
        for layer in (baseline, prototype):
            probe = Probe()
            with probe:
                actual = layer(*valid(batch))
            counts.append(probe.scalars())
        assert torch.equal(baseline(*valid(batch)), actual)
        assert counts == [5, 1], counts
        print(f'{label} batch={batch} valid_scalar_extractions={counts} exact_output=passed')
    for category, cases in [('dynamic', dynamic), ('shapes', shapes),
                            ('dtype', dtype_cases), ('device', device_cases)]:
        for name, args in cases:
            left, right = outcome(baseline, args), outcome(prototype, args)
            same_outcome(left, right)
            if category == 'dynamic':
                assert left[0] == 'ValueError'
            if label == 'core':
                print(f'core {category}/{name}: {left[0]}' +
                      ('' if left[0] == 'ok' else ' | ' + left[1]))
        print(f'{label} {category}_exact_outcomes={len(cases)} passed')
    # A CPU input / native meta layer-buffer mismatch; not a CUDA experiment.
    same_outcome(outcome(Baseline(**options).to('meta'), valid()),
                 outcome(Prototype(**options).to('meta'), valid()))
    print(f'{label} cpu_inputs_meta_buffer_exact_error=passed')
    for case in fixture['cases']:
        layers = [Baseline(fov_deg=case['fov_deg'], **fixture['parameters'], **options),
                  Prototype(fov_deg=case['fov_deg'], **fixture['parameters'], **options)]
        rows = []
        for layer in layers:
            args = [torch.tensor([value], dtype=torch.float32, requires_grad=True)
                    for value in (case['u_bar'], case['rays_m'], [case['alpha']])]
            output, diag = layer.forward_with_diagnostics(*args)
            output.square().sum().backward()
            rows.append((output, diag, [arg.grad for arg in args]))
        assert torch.equal(rows[0][0], rows[1][0])
        assert rows[0][1].keys() == rows[1][1].keys()
        assert all(torch.equal(rows[0][1][key], rows[1][1][key]) for key in rows[0][1])
        assert all(torch.equal(a, b) for a, b in zip(rows[0][2], rows[1][2]))
    print(f'{label} fixture_inputs_output_diagnostics_gradients_exact={len(fixture["cases"])} passed')
    # Compare original and prototype loaded scripts, including precise error suffix.
    loaded = []
    for layer in (baseline, prototype):
        scripted = torch.jit.script(layer)
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        buffer.seek(0)
        loaded.append(torch.jit.load(buffer))
    for api in ('forward', 'forward_with_diagnostics'):
        before, after = [getattr(layer, api)(*valid()) for layer in loaded]
        if api == 'forward':
            assert torch.equal(before, after)
        else:
            assert torch.equal(before[0], after[0])
            assert before[1].keys() == after[1].keys()
            assert all(torch.equal(before[1][key], after[1][key]) for key in before[1])
        for name, args in dynamic + shapes:
            rows = [outcome(layer, args, api) for layer in loaded]
            assert rows[0][0] == rows[1][0] == 'Error', rows
            assert rows[0][1].splitlines()[-1] == rows[1][1].splitlines()[-1], rows
    print(f'{label} script_save_load_public_apis_valid_and_18_invalid=passed')
    graph = str(loaded[1]._c._get_method('_validate_inputs').graph)
    print(f'{label} prototype_validation_graph_aten_bool={graph.count("aten::Bool")}')

for options in ({'num_rays': 0}, {'fov_deg': 361}, {'kappa': 0},
                {'damping_factor': 0}, {'safety_margin': -1}):
    results = []
    for Layer in (ExactLSECBFLayer, module.AggregateCore):
        try:
            Layer(**options)
        except Exception as error:
            results.append((type(error).__name__, str(error)))
    assert len(results) == 2 and results[0] == results[1]
print('inherited_invalid_constructor_checks_exact=5 passed')
print('torch='+torch.__version__+'; CPU + native meta error-path diagnostics only; no CUDA/simulator')
PY
```

Exit 0. Exact output:

```text
core batch=2 valid_scalar_extractions=[5, 1] exact_output=passed
core batch=2048 valid_scalar_extractions=[5, 1] exact_output=passed
core dynamic/u_nan: ValueError | CBF inputs must be finite
core dynamic/u_inf: ValueError | CBF inputs must be finite
core dynamic/rays_nan: ValueError | CBF inputs must be finite
core dynamic/rays_inf: ValueError | CBF inputs must be finite
core dynamic/rays_zero: ValueError | lidar_dists must be positive before preprocessing
core dynamic/rays_negative: ValueError | lidar_dists must be positive before preprocessing
core dynamic/alpha_nan: ValueError | CBF inputs must be finite
core dynamic/alpha_inf: ValueError | CBF inputs must be finite
core dynamic/alpha_zero: ValueError | alpha must be already positive; transform raw logits once
core dynamic/alpha_negative: ValueError | alpha must be already positive; transform raw logits once
core dynamic_exact_outcomes=10 passed
core shapes/u_width: ValueError | u_bar must be nonempty [B,3]
core shapes/u_rank: ValueError | u_bar must be nonempty [B,3]
core shapes/u_empty: ValueError | u_bar must be nonempty [B,3]
core shapes/rays_width: ValueError | lidar_dists must be [B,num_rays] with matching batch
core shapes/rays_batch: ValueError | lidar_dists must be [B,num_rays] with matching batch
core shapes/rays_rank: ValueError | lidar_dists must be [B,num_rays] with matching batch
core shapes/alpha_rank: ValueError | alpha must be [B,1] with matching batch
core shapes/alpha_batch: ValueError | alpha must be [B,1] with matching batch
core shapes_exact_outcomes=8 passed
core dtype/complex_u: RuntimeError | clamp is not supported for complex types
core dtype/complex_rays: RuntimeError | "le_cpu" not implemented for 'ComplexFloat'
core dtype/complex_alpha: RuntimeError | "le_cpu" not implemented for 'ComplexFloat'
core dtype/bool_rays: RuntimeError | Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.
core dtype/integer_rays: ok
core dtype/double_rays: ok
core dtype/nan_u_complex_rays: ValueError | CBF inputs must be finite
core dtype_exact_outcomes=7 passed
core device/meta_0: RuntimeError | Tensor.item() cannot be called on meta tensors
core device/meta_1: RuntimeError | Tensor.item() cannot be called on meta tensors
core device/meta_2: RuntimeError | Tensor.item() cannot be called on meta tensors
core device/all_meta: RuntimeError | Tensor.item() cannot be called on meta tensors
core device_exact_outcomes=4 passed
core cpu_inputs_meta_buffer_exact_error=passed
core fixture_inputs_output_diagnostics_gradients_exact=7 passed
core script_save_load_public_apis_valid_and_18_invalid=passed
core prototype_validation_graph_aten_bool=1
adapter batch=2 valid_scalar_extractions=[5, 1] exact_output=passed
adapter batch=2048 valid_scalar_extractions=[5, 1] exact_output=passed
adapter dynamic_exact_outcomes=10 passed
adapter shapes_exact_outcomes=8 passed
adapter dtype_exact_outcomes=7 passed
adapter device_exact_outcomes=4 passed
adapter cpu_inputs_meta_buffer_exact_error=passed
adapter fixture_inputs_output_diagnostics_gradients_exact=7 passed
adapter script_save_load_public_apis_valid_and_18_invalid=passed
adapter prototype_validation_graph_aten_bool=1
footprint_ablation batch=2 valid_scalar_extractions=[5, 1] exact_output=passed
footprint_ablation batch=2048 valid_scalar_extractions=[5, 1] exact_output=passed
footprint_ablation dynamic_exact_outcomes=10 passed
footprint_ablation shapes_exact_outcomes=8 passed
footprint_ablation dtype_exact_outcomes=7 passed
footprint_ablation device_exact_outcomes=4 passed
footprint_ablation cpu_inputs_meta_buffer_exact_error=passed
footprint_ablation fixture_inputs_output_diagnostics_gradients_exact=7 passed
footprint_ablation script_save_load_public_apis_valid_and_18_invalid=passed
footprint_ablation prototype_validation_graph_aten_bool=1
inherited_invalid_constructor_checks_exact=5 passed
torch=2.6.0+cpu; CPU + native meta error-path diagnostics only; no CUDA/simulator
```

## Exact experiment 2 command/output

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python - <<'PY'
import io, linecache, sys, types
from collections import Counter
import torch
from torch.utils._python_dispatch import TorchDispatchMode

source = '''import torch
class SynchronousAssert(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        torch._assert(torch.isfinite(value).all(), "nonfinite")
        return value * 2
class AsyncAssert(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        torch._assert_async(torch.isfinite(value).all(), "nonfinite")
        return value * 2
'''
filename = '/virtual/sea_nav_cbf_assert_probe.py'
module = types.ModuleType('sea_nav_cbf_assert_probe')
module.__file__ = filename
sys.modules[module.__name__] = module
linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
exec(compile(source, filename, 'exec'), module.__dict__)

class Probe(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))

for Layer in (module.SynchronousAssert, module.AsyncAssert):
    layer = Layer()
    probe = Probe()
    with probe:
        result = layer(torch.ones(2))
    print(Layer.__name__, 'valid_eager_ops='+str(dict(probe.counts)))
    scripted = torch.jit.script(layer)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    loaded = torch.jit.load(buffer)
    for label, model in [('eager', layer), ('script_loaded', loaded)]:
        try:
            result = model(torch.tensor([float('nan'), 1.]))
        except Exception as error:
            print(Layer.__name__, label, 'invalid='+type(error).__name__+': '+str(error).splitlines()[-1])
        else:
            print(Layer.__name__, label, 'invalid=RETURNED; output_nonfinite='+str(not torch.isfinite(result).all().item()))
    print(Layer.__name__, 'loaded_graph='+str(loaded.graph).replace('\n', '\\n'))
print('torch._assert_async docstring:')
print(torch._assert_async.__doc__)
print('torch='+torch.__version__+'; CPU only; CUDA behavior was not executed')
PY
```

Exit 0. Exact output:

```text
SynchronousAssert valid_eager_ops={'aten.ones.default': 1, 'aten.abs.default': 1, 'aten.ne.Scalar': 1, 'aten.eq.Tensor': 1, 'aten.mul.Tensor': 2, 'aten.all.default': 1, 'aten._local_scalar_dense.default': 1}
SynchronousAssert eager invalid=AssertionError: nonfinite
SynchronousAssert script_loaded invalid=Error: RuntimeError: AssertionError: nonfinite
SynchronousAssert loaded_graph=graph(%self : __torch__.sea_nav_cbf_assert_probe.SynchronousAssert,\n      %value.1 : Tensor):\n  %10 : NoneType = prim::Constant()\n  %9 : str = prim::Constant[value="AssertionError: nonfinite"]() # :0:0\n  %13 : int = prim::Constant[value=2]() # /virtual/sea_nav_cbf_assert_probe.py:5:23\n  %3 : Tensor = aten::isfinite(%value.1) # /virtual/sea_nav_cbf_assert_probe.py:4:22\n  %4 : Tensor = aten::all(%3) # /virtual/sea_nav_cbf_assert_probe.py:4:22\n  %6 : bool = aten::Bool(%4) # <string>:3:9\n   = prim::If(%6) # <string>:3:2\n    block0():\n      -> ()\n    block1():\n       = prim::RaiseException(%9, %10) # <string>:3:2\n      -> ()\n  %14 : Tensor = aten::mul(%value.1, %13) # /virtual/sea_nav_cbf_assert_probe.py:5:15\n  return (%14)\n
AsyncAssert valid_eager_ops={'aten.ones.default': 1, 'aten.abs.default': 1, 'aten.ne.Scalar': 1, 'aten.eq.Tensor': 1, 'aten.mul.Tensor': 2, 'aten.all.default': 1, 'aten._assert_async.msg': 1}
AsyncAssert eager invalid=RuntimeError: nonfinite
AsyncAssert script_loaded invalid=RETURNED; output_nonfinite=True
AsyncAssert loaded_graph=graph(%self : __torch__.sea_nav_cbf_assert_probe.AsyncAssert,\n      %value.1 : Tensor):\n  %3 : int = prim::Constant[value=2]() # /virtual/sea_nav_cbf_assert_probe.py:9:23\n  %4 : Tensor = aten::mul(%value.1, %3) # /virtual/sea_nav_cbf_assert_probe.py:9:15\n  return (%4)\n
torch._assert_async docstring:

_assert_async(tensor) -> void

Asynchronously assert that the contents of tensor are nonzero.  For CPU tensors,
this is equivalent to ``assert tensor`` or ``assert tensor.is_nonzero()``; for
CUDA tensors, we DO NOT synchronize and you may only find out the assertion
failed at a later CUDA kernel launch.  Asynchronous assertion can be helpful for
testing invariants in CUDA tensors without giving up performance.  This function
is NOT intended to be used for regular error checking, as it will trash your CUDA
context if the assert fails (forcing you to restart your PyTorch process.)

Args:
    tensor (Tensor): a one element tensor to test to see if it is nonzero.  Zero
        elements (including False for boolean tensors) cause an assertion failure
        to be raised.

torch=2.6.0+cpu; CPU only; CUDA behavior was not executed
```

## Write boundary

Read root AGENTS.md and code-review SKILL.md before task actions; the exact report path was already registered by the controller. This report is the reviewer's only filesystem write. Source/test/config files, index, refs, runtime state and design were not changed. Concurrent coordination-file changes and HEAD advancement belong to the controller/other registered work and were preserved. Task 6 remained the sole source implementation writer.
