# CBF hot-path fixed-commit rereview

Date: 2026-09-08

Reviewed commit: `8303dfec9325d2ce4ef76704ed171d768a3f2912`

Parent: `a453b15c7b4e84f97bc37308f682fdde79e6fbbc`

Verdict: **Spec PASS / Quality PASS** for the bounded CPU checked-path optimization. No actionable P0-P3 finding remains in the reviewed range. This closes the earlier hot-path P2 for its approved contract: valid supported dense inputs now require one host-observable checked decision, not zero synchronization, and ordinary inference no longer constructs diagnostics that its caller discards.

This review is independent of the implementer's own ledger. It inspected the complete six-path range, compared behavior directly with the exact parent implementation, and changed no production source, test, ref, branch or remote. This report is the reviewer's only added file. The pre-existing uncommitted registration edits to `task_plan.md` and `resume_state.json` were preserved.

## What was accepted

- `ExactLSECBFLayer._validate_inputs` retains shape checks before dynamic value work. For historically supported dense real numeric dtypes on one non-meta device, the five value predicates are combined into one tensor condition and only that aggregate reaches Python (`cbf_lse_layer.py:33-67`). A failed aggregate reruns the original ordered validator, preserving precise exception type, message and precedence (`:69-75`).
- Aggregate eligibility is deliberately closed, but it is an acceleration eligibility set rather than a public dtype whitelist. Sparse, quantized, complex, meta, Float8, extended unsigned, backend-limited and mixed-device cases go through the old validator. This avoids changing whether an unusual input succeeds or which earlier error wins.
- The shared Eq. 4 implementation computes command-required intermediates once (`cbf_lse_layer.py:77-90`). Ordinary `forward` returns the command from those intermediates and does not construct `residual_after` or `correction_norm`; the explicit diagnostic endpoint retains all prior differentiable fields (`:92-110`). There is no second mathematical implementation.
- The footprint adapter validates raw rays before subtraction/clipping on both APIs. Its ordinary route uses the shared command core without calculating diagnostic minima, while `forward_with_diagnostics` still emits both raw and effective minima (`cbf_shield.py:52-71`). Zero footprint remains an identity for positive subminimum rays; nonzero footprint remains restricted to a named ablation.
- State-dict identity, ray buffers, positive-alpha semantics, raw-logit wrapper behavior, yaw passthrough, Eq. 4 damping, diagnostic keys/gradients, TorchScript export, and actor/PPO action-state ownership are unchanged.

## Independent evidence

Fresh tests in the fixed checkout used Python 3.10.12, Torch 2.6.0+cpu, disabled bytecode and pytest cache output.

1. Focused hot-path suite:

   `tests/test_cbf_hotpath.py` — **50 passed in 2.06 s**.

2. Combined CBF, adapter and PPO identity selection:

   `tests/test_cbf_hotpath.py`, `training/rsl_rl/tests/test_cbf_lse_layer.py`, `tests/test_cbf_shield.py`, and `training/rsl_rl/tests/test_ppo_action_state_identity.py` — **112 passed in 2.39 s**.

3. Complete available CPU/static suite:

   `tests`, `training/rsl_rl/tests`, and `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py` — **579 passed, 2 skipped in 56.34 s**. The two skips are inherited tests that require a real CUDA device; no failure or deselection was hidden.

4. Actual exporter/TorchScript selection:

   **9 passed, 41 deselected**. The real Gym exporter definition was AST-isolated from its source file, then used to script, save and load core, zero-footprint adapter and named 0.55 m ablation policies. Both checked APIs preserved valid results and earlier-error behavior; no fake Gym package or tracing substitution was used.

5. Python 3.8 grammar:

   All three changed Python files parsed with `ast.parse(..., feature_version=(3, 8))`: the core layer, adapter and focused test. The reviewed range also passed `git diff --check`.

### Operator budget

An independent TorchDispatch probe measured actual calls at batches 2 and 2048 for the core, zero-footprint adapter and 0.55 m named ablation:

| Path | Scalar extractions | Sum reductions | Norms | Min reductions |
|---|---:|---:|---:|---:|
| Ordinary core/adapter/ablation | 1 | 3 | 0 | 0 |
| Core diagnostics | 1 | 4 | 1 | 0 |
| Adapter/ablation diagnostics | 1 | 4 | 1 | 2 |

The result is batch-independent at the two checked sizes. It proves an operator-count reduction from five scalar extractions to one on valid supported inputs and removal of unused diagnostic reductions. It is not a CUDA timing or throughput measurement.

The real actor/PPO probe retained the expected routing:

- Collection with two steps: two CBF calls, two scalar extractions and six sums.
- Update with two epochs and two minibatches: eight CBF calls, eight scalar extractions and 24 sums.
- Neither ordinary route performed a diagnostic norm or adapter minimum.

Invalid supported inputs intentionally pay for the aggregate decision before the detailed fallback: nonfinite `u_bar`, zero rays and zero alpha measured 2, 5 and 6 scalar extractions respectively. Unsupported metadata routes directly to the legacy validator; an earlier nonfinite nominal command still measured one extraction. This is an explicit compatibility cost on failure paths, not an ordinary-path regression.

### Parent equivalence and error precedence

The exact parent core source was loaded directly from Git, and the exact parent adapter class was reconstructed from its parent AST with the old core bound as its base. For three variants, seven committed fixtures and three loss modes (command, diagnostics and joint), all **63** groups of commands, diagnostic tensors and input gradients were bit-exact (`atol=0`, `rtol=0`). The existing golden fixture remained the independent Eq. 4 oracle.

An additional direct-parent matrix compared **186** outcomes across sparse, quantized, complex, meta, Float8, extended integer, malformed and precedence-sensitive inputs. Returned tensors or exception types/messages agreed with the parent in every case. This specifically guards against a newer dtype/backend failure masking the parent's earlier finite-input error.

Saved-and-loaded TorchScript modules were also exercised with 12 earlier-error cases for each of the three variants: **36/36 passed**. The validation graph contains one `aten::Bool`; ordinary graphs contain neither diagnostic norm nor adapter minimum. Dynamic rejection remains present after export.

## Limits and blocked validation

- `torch.cuda.is_available()` was false. Same-device CUDA execution, CPU-to-CUDA mixed-device behavior, stream synchronization, kernel scheduling and accelerator throughput were not tested. Meta-device error paths do not substitute for CUDA evidence.
- No legacy PyTorch/Isaac Gym environment was available. Compatibility was checked with Torch 2.6 CPU and Python 3.8 grammar only; it was not executed under Isaac Gym Preview 4's historical Torch stack.
- Isaac Gym, IsaacLab, simulator lifecycle, physical replay/reset behavior, formal paper metrics and robot deployment were not exercised. They remain blocked/independent higher rungs and are not implied by this PASS.
- Other non-CPU/non-CUDA backends were not certified. Their present fallback behavior is preserved by the tested parent-equivalence matrix, but future backend capability is not asserted.
- The remaining one host decision is intentional for the dynamic checked API. This commit must not be described as a zero-sync or unchecked fast path, and CPU operator counts must not be converted into claimed GPU speedups.

Within those boundaries, the implementation is minimal, preserves the scientific and exported API contracts, closes the confirmed repeated diagnostic work, and has adequate regression coverage. No follow-up source correction is required for this reviewed range.
