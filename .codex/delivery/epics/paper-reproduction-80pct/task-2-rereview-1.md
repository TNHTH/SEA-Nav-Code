# Task 2 scoped re-review — fix round 1

Verdict: **Approved within Task 2's CPU/static configuration scope.** All five P2 findings and the horizon-status P3 are resolved. No new actionable defect was found in the fix diff.

Reviewed `7367aa872619dfd8eccc4821ab25c4608bb3ee85..33592153f1f8ac4a54361f6254b8168544e04124` using the complete supplied `task-2-rereview-1.diff` (9 files, 625 insertions, 65 deletions). The implementation checkout HEAD was read back as `33592153f1f8ac4a54361f6254b8168544e04124`. Read the main checkout's binding brief, original review, appended round-1 implementation evidence, and the recorded design ruling that `time_horizons` is a `profile_fork`. This re-review did not reopen unrelated source or algorithm scope.

## Finding-by-finding verdicts

1. **P2 — Mutable hash-bound data: resolved.** `training/rsl_rl/rsl_rl/experiment_config.py:450` recursively copies and freezes mappings as mapping proxies and sequences as tuples. The resolver, sections, and projections use that boundary. `ConfigProjection.materialize_values()` at `:268` recursively detaches constructor values. Serialization at `:854` detaches output and recomputes the payload hash, rejecting a stale stored hash; projection access and explicit manifest construction invoke this integrity check. Covering tests: `test_resolved_data_is_deeply_immutable`, `test_materialized_consumer_values_and_serialization_are_detached`, and `test_serialization_and_manifest_reject_broken_resolved_hash`.

2. **P2 — Unsupported selected fields/values: resolved.** The module declares explicit schemas for all 22 known contracts and checks both profile selections at normalization. `_validate_selected_contract` at `:402` enforces exact fields, numeric/type/domain checks, known enum values, nested reward schemas, vectors, and relevant cross-field relationships. The original matching-registry/profile bypass now fails for unknown fields, string/NaN damping, invalid FOV, and an invented scheduler. Canonical JSON also rejects nonfinite numbers. Covering test: `test_matching_registry_and_profile_invalid_selections_are_rejected` (six cases, including the additional FOV 361-degree regression). These tests modify registry and profile consistently, so they exercise the selected schema boundary rather than only redundant-file equality.

3. **P2 — Silently inserted mandatory repair delta: resolved.** `sea_nav_current_isaaclab_full_method/adapters/manifest.py:74` rejects the identity-free legacy call with an explicit migration error. No fallback resolves a profile or supplies a delta. The inherited portable test now passes a resolved identity with an explicitly declared `ppo_state_identity_repair`. Covering tests: `test_default_manifest_rejects_legacy_identity_free_call` and `test_default_manifest_uses_repository_relative_paths`. The existing smoke caller's Task 6 migration remains a declared pending consumer, as permitted by the original remedy.

4. **P2 — Eager packaged import breaks adapter import order: resolved.** `adapters/manifest.py:8` imports the types only under `TYPE_CHECKING`; its runtime dependency is delayed until explicit construction after argument/path validation. Merely importing the manifest no longer depends on the runtime having selected the bundled `rsl_rl` yet. Covering test: `test_manifest_module_imports_without_rsl_rl_on_path`, which launches a clean subprocess without `PYTHONPATH` and makes `rsl_rl` unavailable. It validates the adapter import boundary and supplies no fake simulator.

5. **P2 — Upstream acquisition versus refresh cadence: resolved.** `configs/parity_registry.yaml:222` and `:236` now carry the separate 0.1 s output refresh period alongside 0.02 s upstream acquisition; the upstream profile repeats those values. Paper acquisition remains 0.1 s. The schema checks integer refresh steps, history-derived sample ages, and maximum held age. Covering test: `test_upstream_ray_acquisition_and_output_refresh_cadences_are_distinct`, which establishes 40/60 ms ages and 140 ms maximum held age from the declared clocks. The registry hash, canonical hash fixture, and both adapter examples were updated consistently.

6. **P3 — Differing horizons labeled directly matched: resolved.** The `time_horizons` row at `configs/parity_registry.yaml:326` now has `profile_fork`, consistent with the main design's recorded ruling. Paper evaluation stays 30 s and upstream stays null/unsupported. Row normalization checks that `resolved` selections match and `profile_fork` selections differ. Covering test: `test_differing_evaluation_horizons_are_a_profile_fork`.

## Validation evidence assessed

The appended report records the named regression selection initially failing with `12 failed in 0.79s`, plus the additional FOV-domain case failing before its fix. Incremental green results were `3 passed` for explicit manifest/legacy rejection/import boundary, `2 passed` for cadence/horizons, `8 passed` for immutability/hash/schema cases, and `1 passed` for the FOV upper bound.

The reported fresh post-commit command was:

```bash
PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
```

Reported output: exit 0, **71 passed in 3.94s**. This includes the tests named above and the inherited adapter/static contracts. The post-commit Gate A command was:

```bash
PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" --report /tmp/sea-nav-gate-a-task2-fix1-postcommit.RDqVd3/gate-a.json
```

Reported output: exit 0, `passed_with_blockers`; five CPU/static cases passed, 56 tracked Python files compiled, and Isaac Gym remained blocked. The report also records Python 3.8 grammar parsing, YAML/JSON parsing, and Git whitespace checks. These are implementation-reported executions assessed alongside their test/code changes; they were not rerun by this reviewer. No specific unanswered executable doubt remained after reading the fix, so no extra CPU probe or suite repetition was needed.

## New defects and out-of-scope observations

No new actionable fix defect found. The additional schema machinery is bounded to the named contracts, and immutable internals plus explicit mutable materialization give later consumers a clear ownership contract. The change stays in configuration, manifests, data, and their tests; it does not implement PPO/CBF/replay/runtime algorithms.

The legacy smoke manifest call intentionally fails with a migration instruction until Task 6 supplies an explicit resolved identity. This is an acknowledged future consumer obligation, not a runtime-pass claim. Tasks 3–7 must still apply the projections and enforce activation deltas at their actual consumer boundaries. Projection statuses remain future/inactive. Real reward, CBF, PPO, ACSI, sensor scheduler, replay, and checkpoint behavior was not re-reviewed or validated in this round.

Accepted `paper_v1` remains blocked by Table V and unresolved perception timing. Isaac Gym/IsaacLab execution, formal metrics, publication rights, and hardware remain blocked. No environment import or simulator behavior was exercised.

## Spec and quality verdicts

**Spec compliance: pass for this scoped fix and Task 2 configuration boundary.** Mandatory deltas remain explicit, the scientific/runtime axes remain separate, the corrected source timing and horizon status are recorded, and inactive projections do not claim applied behavior.

**Code quality: pass.** The fixes address the demonstrated failure paths with explicit data ownership, validation, and regression coverage. No broader framework or runtime implementation was introduced. Canonical serialization remains detached and hash-checked; the adapter retains a single packaged configuration implementation.

Source, index, and HEAD were untouched by this reviewer. The implementation checkout retained only its two pre-existing local registration edits. Main/stable remained at `1c5675b`; branch names remained exactly main/stable/test. The only review write is this report in the main coordination directory.
