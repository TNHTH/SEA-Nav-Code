# Bounded CBF hot-path review

Date: 2026-09-07. Reviewer: `/root/cbf_hotpath_review`.
Inspected primary HEAD: `65dcbbc2b13c92af99a9e7d4cba4110880f9b509`.
Scope: the CBF scalar-extraction observation and its real actor/PPO/export consumers only. Task 6 has the sole parallel source writer in another worktree. This report is the reviewer's only write; no source, tests, index, refs, configuration or simulator state were changed.

## Finding

### [P2] Ordinary CBF mean evaluation always takes the synchronous diagnostic-validation path

Primary location: `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:40–45`, reached unconditionally from `forward` at lines 73–74 via `forward_with_diagnostics` at lines 68–71. The adapter repeats that boundary at `sea_nav_current_isaaclab_full_method/adapters/cbf_shield.py:55`.

On every valid eager call the three finiteness reductions and two positivity reductions are converted into five Python booleans. A real CPU operator probe establishes five `aten._local_scalar_dense` calls for both layer implementations, at batch sizes 2 and 2048. These are batch-wide reductions, **not five synchronizations per environment**. On a CUDA tensor a Python branch on a computed tensor value must obtain the value on the host; the probe establishes the reachable extraction sites, not GPU timings, throughput loss or a simulator regression.

The cost is on the actual policy training path, not only on an operator's diagnostic endpoint:

- `modules/cbf_actor_critic.py:142–161` constructs the nominal action, positive-alpha representation and ray distances, then invokes the layer. `forward`, pure `action_mean_for`, and `act_inference` all reach this helper (lines 165–199).
- `algorithms/ppo.py:173` calls that actor once per collected step. `runners/on_policy_runner.py:135–136` loops over `num_steps_per_env`.
- `algorithms/ppo.py:219` invokes it again for each current minibatch. The smoothness path at lines 269–270 passes the current mean, so lines 150–152 make exactly one additional CBF call for interpolated observations, not two. Current-state ownership is correct and must remain so.
- Therefore normal CBF-layer validation alone contributes `5 × (T + 2 × E × M)` scalar extractions per PPO iteration, for rollout steps `T`, epochs `E`, minibatches `M`, before unrelated PPO logging/adaptive-KL conversions. The inherited Go2 source defaults are `T=48`, `E=5`, `M=4`: **88 CBF evaluations / 440 CBF-local extractions**, not a measured run. Sources: `envs/base/legged_robot_config.py:251–263`, `envs/go2/go2_pos_config.py:288–299`. Go2's 2048 environments at line 57 make each validation reduction batch-wide; overrides may change these numbers.
- A real CPU actor/PPO run with two environments and `T=2,E=2,M=2` independently observed 2 layer calls / 10 extractions during collection and 8 layer calls / 40 extractions during update, for both core and adapter layers. Hooks limited counting to the real layer; PPO's later metric `.item()` calls were excluded. No fake environment or shield was used.

This is actionable P2 because it is a reachable repeated host-value dependency on the intended accelerator training path and conflicts with design Section 10's requirement that debug materialization/device-to-host conversions stay outside the training hot path unless explicitly enabled. It is not evidence for a P1 correctness failure or a quantified speedup. The existing Task 4 CPU/spec review remains valid for its reported scope; this bounded performance finding is a new input to whole-branch review, not that future gate itself.

Ordinary eager forward also computes diagnostics that its caller discards: `cbf_lse_layer.py:59,63` performs the post-command residual reduction and correction norm; the adapter at `cbf_shield.py:60–61` additionally performs two ray minima. The same probe sees four sum reductions and one norm per core forward, plus two minima per adapter forward. These tensor operations are a bounded secondary consequence of the same routing defect, not a separate severity item. `_compute(...)[0]` is not a complete diagnostic-cost fix: `_compute` itself still constructs the diagnostic tensors. No claim is made that an optimizing TorchScript executor cannot eliminate discarded tensor work.

## Compatibility boundary and smallest safe direction

Do **not** simply delete `_validate_inputs`, call the current private `_compute` from the actor, clamp invalid values into apparent validity, or cache one successful value check for future minibatches. Design Section 7 and Task 4 require deterministic rejection of malformed shapes/nonfinite inputs/nonpositive rays and alpha; the current tests intentionally exercise public `forward`, not only diagnostic calls.

**Recommended minimum boundary decision:** preserve dynamic checked rejection on the actor and public/scripted APIs for this recovery, explicitly identify it as an enabled correctness-validation mode and a narrowly accepted synchronization-budget exception, and optimize only the discarded diagnostic work. Do not add an unchecked training route in this fix. This preserves the stronger existing input contract and requires fewer new semantics. It is a recommendation for the controller's recorded ruling, not a design amendment made by this reviewer. The five extractions must then be reported as an accepted residual cost, not "fixed" or "zero sync."

The implementation boundaries for that ruling are:

1. Keep constructor/metadata checks and public checked `forward(u_bar, rays, alpha) -> Tensor` / exported `forward_with_diagnostics(...) -> (Tensor, Dict[str, Tensor])` behavior backward compatible. Build a single shared tensor mathematical core that can return the ordinary command without computing unused diagnostics. Checked/debug endpoints can derive the extra diagnostics from the same mathematical intermediates; do not create a second Eq. 4 implementation.
2. Keep synchronous invalid-value rejection at **every dynamic actor call** under the recommended checked mode. Trace its enabled state in configuration/reporting; do not silently select behavior from `train()`/`eval()` or describe dynamic checks as construction-only. An eventual zero-extraction requirement would need a separate approved boundary contract and explicit invalid-data behavior; preserving only checked public API tests would not preserve actor fail-fast behavior after bypassing them. That alternative is deferred, not added to this patch's scope.
3. `exp2(rays)` and `Softplus(alpha_raw)` do not establish a numerical proof of finite positive runtime input. The dedicated CPU interpreter produces `inf` for `exp2(1000)` and `0.0` for `Softplus(-1000)` in float32. Arbitrary network outputs or nonfinite observations can also fail. A trusted-input route therefore cannot be justified merely by the algebraic sign of those transforms. Changing their clamp/formula would additionally be a scientific behavior change outside this optimization.
4. Preserve the adapter's **raw-ray validation before footprint subtraction/clipping**, zero-footprint subminimum-ray identity, and the named-ablation requirement for nonzero footprint. Preserve the historical wrapper distinction: `apply(..., gamma)` receives raw logits and transforms once; `apply_alpha` receives already-positive alpha. Wrapper-side raw-logit validation at `cbf_shield.py:93` is an additional checked-boundary extraction, not included in the actor counts above.
5. Preserve Eq. 4/damping, FOV and ray-order buffers, yaw passthrough, existing `state_dict` keys, public return types, differentiable diagnostics, and Task 3's pure mean/distribution/alpha/rays/u_bar/u_s ownership. Neither a post-sample shield nor altered action likelihood is a performance fix.
6. Preserve real `torch.jit.script` / save / load for both layers and exported diagnostics. The existing Gym exporter at `training/legged_gym/legged_gym/utils/helpers.py:205–240` deep-copies the layer into a wrapper and invokes ordinary `forward`; an implicit mode change would reach that consumer too. In the current loaded scripted modules, the validator graph contains five `aten::Bool` nodes and both public methods reject zero rays. Switching to tracing to erase data-dependent checks, using Python-only flags ignored by the scripted layer, or relying on an unverified newer-PyTorch async assertion is not an acceptable compatibility substitute. Python 3.8 grammar and the original-stack PyTorch compatibility remain independent from this Torch 2.6 CPU evidence.

The minimum immediate optimization with no boundary-policy change is to stop computing unused diagnostics in ordinary eager forward while retaining checked behavior. That removes confirmed extra reductions but **does not eliminate the five extractions**. Disposition of the finding requires the explicit validation-budget exception above and corresponding implementation/test evidence; without that ruling it remains open. Queue any source fix serially after Task 7, with exact ownership registration; do not concurrently amend actor/PPO/CBF files.

## Independent regression tests recommended for the eventual serial fix

- CPU TorchDispatch operator counts around ordinary core/adapter calls at batch sizes 2 and 2048; distinguish the accepted checked/debug mode from any approved tensor-only mode. Assert the intended scalar-extraction budget and absence of unused norm/minimum diagnostics in the ordinary tensor-only route, rather than timing thresholds. An unchanged checked five-extraction endpoint must not be advertised as a zero-sync fast path.
- Real actor collection and PPO update with `T=2,E=2,M=2`, counting **only** layer activity via hooks. Prove routing reaches the intended mode and still makes one CBF collection call / two calls per minibatch; run unchanged likelihood and auxiliary-state identity regressions. Do not replace the layer with an additive fake for this performance test.
- Shared golden fixtures at 180/240 degrees; checked/ordinary/diagnostic command equality and gradient equality with the same upstream inputs. Preserve active residual `-eta * epsilon_d`, yaw, nonnegative eta and all diagnostic keys.
- Invalid cases on each checked API and round-tripped script: NaN/Inf nominal command, rays and alpha; zero/negative raw rays before ablation clipping; zero/negative positive-alpha input; incompatible shapes. Add float32 transform underflow/overflow sentinels to prohibit an invalid "transforms guarantee validity" assumption. Tests for any approved unchecked mode must state its failure policy explicitly, not accidentally inherit checked claims.
- Round-trip script for core, zero-footprint adapter and named footprint ablation, exercising both exported methods and the actual exporter-selected ordinary mode. Preserve state-dict loading and canonical raw-logit/positive-alpha wrapper behavior.
- Accelerator timing, simulator stepping and deployment behavior remain blocked here. If later profiled, warm up/synchronize at measurement boundaries and record exact hardware/runtime/configuration; do not turn these CPU operator counts into an estimated millisecond or percentage improvement.

## Executed CPU evidence

Working directory was the primary repository. Exact invocation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python - <<'PY'
import io
from collections import Counter
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
from rsl_rl.algorithms.ppo import PPO
from sea_nav_current_isaaclab_full_method.adapters.cbf_shield import FootprintAwareLSECBFLayer

class Probe(TorchDispatchMode):
    def __init__(self, active=True):
        super().__init__()
        self.active = active
        self.counts = Counter()
        self.calls = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.active:
            self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))
    def start(self, module, inputs):
        self.active = True
        self.calls += 1
    def stop(self, module, inputs, output):
        self.active = False
    def row(self):
        return {'calls': self.calls, 'scalar_extractions': self.counts['aten._local_scalar_dense.default'],
                'sum_reductions': self.counts['aten.sum.dim_IntList'],
                'linalg_vector_norm': self.counts['aten.linalg_vector_norm.default'],
                'min_dim': self.counts['aten.min.dim']}

for kind in (ExactLSECBFLayer, FootprintAwareLSECBFLayer):
    for batch in (2, 2048):
        layer = kind()
        args = (torch.zeros(batch, 3), torch.ones(batch, 41), torch.ones(batch, 1))
        p = Probe()
        with p:
            actual = layer(*args)
        reference, _ = layer._compute(*args)
        assert torch.equal(actual, reference)
        print(kind.__name__, 'B='+str(batch), p.row())
    scripted = torch.jit.script(layer)
    saved = io.BytesIO()
    torch.jit.save(scripted, saved)
    saved.seek(0)
    loaded = torch.jit.load(saved)
    graph = str(loaded._c._get_method('_validate_inputs').graph)
    invalid = [args[0], torch.zeros_like(args[1]), args[2]]
    for api in ('forward', 'forward_with_diagnostics'):
        try:
            getattr(loaded, api)(*invalid)
        except torch.jit.Error as error:
            assert 'positive before preprocessing' in str(error)
        else:
            raise AssertionError('scripted invalid rays were accepted')
    print(kind.__name__, 'script_save_load_invalid_rejection=passed',
          'validation_graph_aten_bool='+str(graph.count('aten::Bool')))
    print('cpu_numerical_alpha_underflow='+str(torch.nn.functional.softplus(torch.tensor(-1000.)).item()),
          'cpu_numerical_ray_overflow='+str(torch.exp2(torch.tensor(1000.)).item()))

for layer_kind in (ExactLSECBFLayer, FootprintAwareLSECBFLayer):
    torch.manual_seed(1729)
    actor = DifferentiableSafeActorCritic(3, actor_hidden_dims=[16], critic_hidden_dims=[16], encoder_hidden_dims=[16])
    actor.cbf_layer = layer_kind()
    ppo = PPO(actor, num_learning_epochs=2, num_mini_batches=2, schedule='fixed', desired_kl=None)
    ppo.init_storage(2, 2, (550,), (3,))
    probe = Probe(active=False)
    h1 = actor.cbf_layer.register_forward_pre_hook(probe.start)
    h2 = actor.cbf_layer.register_forward_hook(probe.stop)
    obs = torch.zeros(2, 550)
    with probe, torch.no_grad():
        for step in range(2):
            action = ppo.act(obs, obs)
            ppo.process_env_step(obs, torch.tensor([1., -1.]), torch.zeros(2, dtype=torch.bool), {})
        ppo.compute_returns(obs)
    print(layer_kind.__name__, 'collection_T2', probe.row())
    probe.calls = 0
    probe.counts.clear()
    with probe:
        metrics = ppo.update()
    assert all(torch.isfinite(torch.tensor(metrics)))
    print(layer_kind.__name__, 'update_E2_M2', probe.row())
    h1.remove()
    h2.remove()
print('torch='+torch.__version__+'; CPU only; no simulator or CUDA timing')
PY
```

Exit 0, output:

```text
ExactLSECBFLayer B=2 {'calls': 0, 'scalar_extractions': 5, 'sum_reductions': 4, 'linalg_vector_norm': 1, 'min_dim': 0}
ExactLSECBFLayer B=2048 {'calls': 0, 'scalar_extractions': 5, 'sum_reductions': 4, 'linalg_vector_norm': 1, 'min_dim': 0}
ExactLSECBFLayer script_save_load_invalid_rejection=passed validation_graph_aten_bool=5
cpu_numerical_alpha_underflow=0.0 cpu_numerical_ray_overflow=inf
FootprintAwareLSECBFLayer B=2 {'calls': 0, 'scalar_extractions': 5, 'sum_reductions': 4, 'linalg_vector_norm': 1, 'min_dim': 2}
FootprintAwareLSECBFLayer B=2048 {'calls': 0, 'scalar_extractions': 5, 'sum_reductions': 4, 'linalg_vector_norm': 1, 'min_dim': 2}
FootprintAwareLSECBFLayer script_save_load_invalid_rejection=passed validation_graph_aten_bool=5
cpu_numerical_alpha_underflow=0.0 cpu_numerical_ray_overflow=inf
ExactLSECBFLayer collection_T2 {'calls': 2, 'scalar_extractions': 10, 'sum_reductions': 8, 'linalg_vector_norm': 2, 'min_dim': 0}
ExactLSECBFLayer update_E2_M2 {'calls': 8, 'scalar_extractions': 40, 'sum_reductions': 32, 'linalg_vector_norm': 8, 'min_dim': 0}
FootprintAwareLSECBFLayer collection_T2 {'calls': 2, 'scalar_extractions': 10, 'sum_reductions': 8, 'linalg_vector_norm': 2, 'min_dim': 4}
FootprintAwareLSECBFLayer update_E2_M2 {'calls': 8, 'scalar_extractions': 40, 'sum_reductions': 32, 'linalg_vector_norm': 8, 'min_dim': 16}
torch=2.6.0+cpu; CPU only; no simulator or CUDA timing
```

The standalone `B=...` rows have `calls: 0` because only the actor/PPO part installs call-count hooks; each standalone row measures one actual layer invocation. The self-reference to `_compute` checks command identity only; it is not an independent mathematical oracle. Existing shared scalar golden fixtures remain the required independent numerical oracle.

Read in full: root AGENTS.md, code-review SKILL.md, cbf-hotpath-observation.md, Task 4 brief/report/review, and design Sections 7–10. Read the named source/test/export contexts to establish reachability and compatibility. No complete-suite or new whole-branch PASS is claimed by this report.
