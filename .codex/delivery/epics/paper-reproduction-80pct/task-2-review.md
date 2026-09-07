# Task 2 independent review

Verdict: **Changes requested** for configuration integrity, explicit identity, and compatibility defects before integration.

Reviewed range: `e22493377065c445cfd2b1ac43c01aa4d99c5120..7367aa872619dfd8eccc4821ab25c4608bb3ee85`, the supplied complete 13-file diff. Implementation checkout: `work/SEA-Nav-Code-batch2`, HEAD `7367aa872619dfd8eccc4821ab25c4608bb3ee85`. Binding Task 2 corrections were read first, followed by the report, updated design, scientific wiring audit, and code-review skill. Historical examples were subordinated to the binding corrections.

## Actionable findings

### [P2] Freeze the data bound to the resolved hash, not only the dataclass attributes

`training/rsl_rl/rsl_rl/experiment_config.py:468` and `:505`.

The resolver exposes ordinary mutable dictionaries/lists in `selected_contracts`, algorithm sections, and projection values. Projection getters return those same objects. The frozen dataclasses prevent attribute replacement but do not protect their contents; serialization subsequently trusts the original hash. A consumer can change `build_policy_kwargs(config).values['epsilon_d']` to `7.0`: the selected registry value remains `1.0`, the projected value becomes `7.0`, and `resolved_config_to_dict` still publishes the original `resolved_sha256`. Nested values also alias between sections and serialization outputs. This breaks the central identity-bound configuration contract before runtime wiring even begins.

Focused CPU probe confirmed that recomputing the canonical payload hash after that mutation no longer equals the stored hash. The existing frozen-type test only checks dataclass metadata and misses this case. Recursively freeze internal resolved data and return detached materializations where consumers need mutable constructor inputs; verify integrity before serialization/manifest binding. Add a regression covering nested projection/selected-value mutation and mutation through `resolved_config_to_dict`.

### [P2] Validate selected fields and values before producing typed consumer projections

`training/rsl_rl/rsl_rl/experiment_config.py:250`.

`_section` accepts every mapping key/value and passes it through; validation checks outer row structure and profile equality, but not the selected contract schema. Supplying matching registry/profile data with `damped_cbf.epsilon_d: not-a-number` and `invented_parameter: true` successfully resolves an `upstream_fbce672c` run and exposes both in `build_policy_kwargs`. A scalar string where a numeric damping value is required, unsupported keys, and unsupported modes therefore receive an accepted identity/hash rather than an actionable configuration error. Contract-level coverage does not establish field-level consumption.

A focused probe substituted only in-memory registry/profile text through `Path.read_text`, then called the real resolver and projection getter; both invalid fields survived. No source file, fake module, or simulator was involved. Add small explicit per-contract schemas/type/domain/enum checks, reject unknown selected keys, and check finite numeric values and required fields. This can remain a small mapping implementation; a general framework is unnecessary. Tests must alter matching registry and profile inputs, since profile-mismatch tests alone cannot establish this boundary.

### [P2] Do not insert the mandatory implementation delta in the legacy manifest path

`sea_nav_current_isaaclab_full_method/adapters/manifest.py:72`.

The positional compatibility form resolves a profile and supplies `implementation_delta=('ppo_state_identity_repair',)` even though its caller provided no identity or repair declaration. Calling `default_manifest(adapter_root)` produces that delta today, contradicting the binding requirement that every caller explicitly declare it. Marking the resulting manifest `unverified` limits evidence claims but does not make the declaration explicit.

The concrete outside-diff interface check found this is also a runtime caller, not only the inherited static test: `full_method_runtime_smoke.py:1026` calls the positional form and writes the result at `:1030`. Remove automatic resolution from this path. Require explicit resolved identity/deltas from callers; update the inherited portable test accordingly. If runtime migration must remain Task 6 work, fail with a precise migration message at the legacy call rather than synthesize a repaired identity. The focused CPU probe confirmed the undeclared delta is currently added.

### [P2] Preserve the adapter import boundary when adding the packaged loader dependency

`sea_nav_current_isaaclab_full_method/adapters/manifest.py:8`.

The new eager `rsl_rl.experiment_config` import runs before the existing smoke entry point places the repository's bundled `rsl_rl` on `sys.path`. The named outside-diff interface check inspected `full_method_runtime_smoke.py:131–145`: it imports `adapters.manifest` at `:137`, then selects the bundled package at `:144`. On a launcher with no installed `rsl_rl`, importing the adapter now raises `ModuleNotFoundError`; with a different installed `rsl_rl`, that package need not contain this new module. The reported suite's explicit `PYTHONPATH` masks this ordering regression.

A clean subprocess using the dedicated CPU interpreter, only the adapter directory on `sys.path`, and no `PYTHONPATH` reproduced `ModuleNotFoundError: No module named 'rsl_rl'`. This is an import-boundary probe, not simulator execution. Delay the runtime dependency until explicit manifest construction (use type-checking-only imports for annotations), or move the existing caller's bundled-package setup before the adapter import within an explicitly registered compatibility fix. Do not add a machine-specific path or another loader implementation. Add a focused import-order regression.

### [P2] Separate upstream acquisition cadence from output refresh cadence

`configs/parity_registry.yaml:234` (also upstream effective value at `:221` and the repeated upstream profile).

The upstream selection declares `acquisition_period_s: 0.1`, while also declaring history indices `[-3,-4]` and refresh ages `[0.04,0.06]`. The binding scientific audit's ray-delay section states that upstream appends a newly acquired sample every 20 ms policy tick and refreshes the held output every 100 ms. A consumer honoring the declared acquisition period would obtain 200/300 ms history ages at those indices, not the recorded 40/60 ms ages. The projection has no separate numeric refresh period to resolve that contradiction.

Record upstream acquisition as the policy-tick cadence (currently 0.02 s) and output refresh as a distinct 0.1 s field, keeping paper acquisition at 0.1 s. Update the profile validation input and affected hashes/examples. Add a pure contract assertion linking history indices, acquisition dt, refresh interval, and maximum held age. This finding uses the named audit's authoritative source evidence; no broad environment algorithm rereview was performed.

### [P3] Do not mark the differing evaluation horizons as directly matched

`configs/parity_registry.yaml:333`.

`time_horizons` is marked `resolved`, whose binding meaning is directly matched, but the row records paper evaluation timeout `30.0` and upstream `null`/unsupported. Its selected values differ too. A status-based parity report would classify this unresolved source difference as a direct match. Preserve the honest upstream unsupported value and use a status consistent with the fork, or split the matched training horizon from the differing evaluation horizon. Add a status/selection consistency assertion; do not supply an invented upstream evaluation value.

## Verification and scope

- Did not rerun the reported green suite. The report records 58 passing CPU/inherited cases and Gate A `passed_with_blockers`; those are implementation evidence, not a fresh reviewer run.
- Ran only targeted CPU probes for mutable hash integrity, unsupported selection acceptance, implicit legacy delta, and manifest import order with `../sea-nav-cpu-venv/bin/python` and bytecode writing disabled.
- Outside-diff searches were limited to the named manifest call/import compatibility risks in existing adapter entry points and the inherited portable manifest test. No PPO/CBF/environment algorithm rereview or simulator-bound import was performed.
- Branch readback showed exactly `main`, `stable`, `test`; main/stable remained at `1c5675b`. Existing coordination dirt in both checkouts was preserved. Source, index, and HEAD were not changed by this review; the only written artifact is this review.

## Cannot verify across tasks

Task 2 projections remain explicitly future/inactive. Actual constructor consumption, PPO state repair, CBF equations/gradients, reward computation and one-time dt application, ACSI event ordering, replay activation-delta enforcement, sensor scheduler execution, and checkpoint load identity checks belong to Tasks 3–7 and are not established here. The import-order finding is a statically located and CPU-reproduced prerequisite failure, not a claim that IsaacLab otherwise runs.

Isaac Gym/IsaacLab execution and first-observation evidence, formal 100-trial metrics, publication rights, and real hardware remain blocked. Accepted `paper_v1` resolution correctly remains blocked by literal Table V bounds and unresolved perception timing; no passing paper identity was fabricated. CPU dependency contents were inspected, but a fresh environment installation and external Torch package source were not exercised.

## Quality assessment

The packaged shared-module placement, thin adapter re-export, canonical JSON encoding, explicit future projection statuses, separated scientific/runtime axes, literal paper bounds, and complete-environment dependency inventory are sound directions. Scope correctly avoids PPO/CBF implementation. The main weakness is that the advertised typed, frozen, identity-bound boundary currently relies on shallow mapping conventions and happy-path tests. The compatibility fallback also weakens a mandatory identity rule to preserve an older call shape. Correct these boundaries before later tasks build on them; the findings do not require a broader framework or simulator claims.
