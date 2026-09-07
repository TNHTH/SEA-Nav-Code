# Task 4 independent spec and quality review

Date: 2026-09-07. Reviewer: `/root/review_batch4`.
Reviewed range: `17e53cf1e0b4ce476b43bdc932baa699cecf3571..ac0567eed6cf88a17f5178663f0324cffaaab357`, in `../SEA-Nav-Code-batch4`.

## Findings and verdicts

No actionable P0–P3 findings were established in the reviewed change. There are therefore no defect file/line annotations.

- **Spec verdict: PASS for the corrected Task 4 CPU/static scope.** The small packaged factory and the three narrow bootstrap fixes are explicitly authorized scope, not scope violations. This verdict does not mean adapter runtime completion or accepted paper reproduction.
- **Quality verdict: PASS for this range.** The shared mathematical core, explicit constructor boundary, validation, tests, and preserved actor/PPO state ownership meet the reviewed contracts. No simulator, performance, formal-metric, or hardware pass is inferred.

## Evidence and contract assessment

Read the corrected task-4-brief.md, task-4-report.md, the complete 2,106-line task-4-review.diff, root AGENTS.md, scientific-wiring-audit.md, and design sections 5–7. Applied the code-review skill's evidence and severity rules. The range contains the reported 15 paths; no Task 3 regression test is modified.

1. `cbf_lse_layer.py` implements the damped Eq. 4 denominator, nonnegative `eta` with separate `eta_raw`, yaw passthrough, and differentiable diagnostics. Validation covers configured count/FOV, positive kappa/damping/radius, nonnegative margin, shapes, finite inputs, positive raw rays, and positive alpha. The active residual remains `-eta * epsilon_d`; the implementation does not claim hard safety. The resolved profile routes epsilon_d=1.0 into the layer; deliberately changed diagnostic projections remain separate from accepted identity.
2. The adapter inherits that core, validates before footprint clipping, and permits nonzero footprint only with a nonempty `ablation/` name. `apply(..., gamma)` explicitly accepts raw logits and applies Softplus once; `apply_alpha` accepts positive alpha. One versioned stored fixture is consumed by core and adapter tests, covering 180/240 degrees, active/inactive/asymmetric/opposing hazards and the raw/positive boundary. Historic aliases do not change canonical nonnegative eta semantics.
3. `policy_factory.py` materializes detached values, rejects missing/unknown/conflicting fields and unsupported fixed semantics, checks activation deltas, and uses Task 2 integrity-checked accessors for resolved entry points. Observation FOV and CBF FOV remain independent. All projected numeric actor values reach the constructor/layer; all projected PPO coefficients and action bounds reach actual loss expressions. Nondefault coefficient tests compare real update gradients against independent auxiliary gradients, including zero and one-time weighting. The ignored inherited non-1 value-loss override is explicitly rejected by this new boundary.
4. The actor's pure mean query, sampled Normal action, current-minibatch auxiliary ownership and gradient-bearing outputs remain intact. PPO still passes current mean/value to smoothness, uses pure interpolation queries, and applies the alpha/intervention terms to current state. The coefficient refactor preserves effective default smoothness weights .05/.005.
5. All three bootstrap blocks derive the shared root from `adapter_root.parent`, select/purge bundled rsl_rl before importing the now-dependent adapter, and remove the later duplicate selection. Added tests execute extracted real-package bootstrap statements in isolated processes and check class identity; AST tests preserve AppLauncher ordering without starting the application.

## Reviewer verification and outside-diff checks

The reported **169 passed** suite and Gate A result are worker evidence from task-4-report.md, not reviewer executions. The reviewer did not repeat that suite, start a simulator/application, or inject simulator substitutes.

One specific unanswered export question was checked with the dedicated `../sea-nav-cpu-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and bundled rsl_rl on PYTHONPATH: real `torch.jit.script` → `torch.jit.save` to BytesIO → `torch.jit.load`, for the 180-degree core and a named .55 m footprint ablation adapter. **Both passed**: loaded `forward`, exported `forward_with_diagnostics`, diagnostic key sets, and every diagnostic tensor matched eager outputs exactly. No artifact was written.

Outside-diff inspection was limited to these named interface risks:

- Task 2 `experiment_config.py`: `ConfigProjection.materialize_values`, policy/PPO projection fields and activation deltas, and resolved accessor integrity checks, to verify that the new factory's inputs and mutation assumptions match the existing boundary.
- Unchanged portions of `cbf_actor_critic.py` and PPO option uses: constructor option consumption and Task 3 pure-state/sampling/gradient paths, to check whether new mappings or weights expose ignored values or overwrite current state.
- Import/AppLauncher references in the three affected entry-point files, to check for earlier dependent CBF imports outside the changed bootstrap blocks. The additional ACSI import is inside an instance method, not an earlier startup import.
- Local Git status, HEAD, branch references and range statistics: the reviewed checkout is at ac0567e with only worker coordination-file dirt. The three named local branches are main, stable and test; main/stable remain at `1c5675bbedf1dcbe5a4c1a91830cae528c780793`. Hosting-side protection was not established or tested.

## Cross-task obligations and cannot-verify boundaries

- **Task 6:** bind these constructors to the actual Gym/IsaacLab startup paths, resolved CLI values, layer-replacement sites and final manifests; apply reward and perception consumers; retain explicit implementation deltas. Projection converters and constructor tests alone do not prove startup application.
- **Task 6:** resolve the existing .55 m smoke default through a genuine named ablation path or select zero footprint for ordinary paper/upstream profiles. Current runtime call sites do not yet carry that identity. This is an acknowledged Task 6 handoff under the corrected scope, not a Task 4 import-order deferral.
- **Task 6:** align action-stage/runtime residual diagnostics with the shared layer and actual preprocessing, and carry actual geometry through checkpoint loading/reconstruction. Startup and post-load behavior cannot be certified from CPU constructor evidence.
- **Tasks 5–6:** ACSI state/decision/replay reconstruction, reward formula/time integration and perception clock semantics require their assigned implementations and integration evidence; this CBF/PPO review does not validate them.
- **Blocked external evidence:** relevant IsaacLab smoke required by AGENTS.md, Isaac Gym execution, formal 100-trial metrics, hardware behavior and publication rights remain unverified/blocked. No simulator availability was manufactured. Accepted `paper_v1` remains blocked by the unresolved `paper_table_action_bounds` and `perception_timing_semantics` rows.
- Only local branch existence and positions were observed. No remote query/change or evidence of hosting branch protection was produced.

Only this review report was written. Source, index, HEAD, branch references, remotes and worker coordination files were not changed by the reviewer.
