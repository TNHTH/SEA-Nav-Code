# Task 7 caller fix1 scoped re-review

Date: 2026-09-07. Original caller reviewer `/root/review_batch7_callers`. Status: COMPLETE for this narrow re-review.

## Findings and verdict

**No new actionable finding. Original terrain/receipt P2: ADDRESSED. Bounded Spec: PASS. Bounded Quality: PASS.** These verdicts cover only the fixed Gym play configuration boundary and its ordinary regression tests. The original failed caller report remains historical evidence; this report closes its single P2 at the exact fixed revision below.

**Full Task 7 review remains INCOMPLETE.** This scoped re-review neither replaces nor retries the service-restricted original core/converter/security review and does not approve Task 7 integration, simulator execution or final-project acceptance.

## Fixed identity and reviewed scope

- Read-only checkout: `../SEA-Nav-Code-batch7-review-callers-fix1`.
- HEAD: `774027d1a975e318ad2f577b457951f51c82bfad`.
- Fix BASE: `c7b9aa371b8ab3800adea378a7024f043fb58580`.
- Complete three-path diff read: **94 insertions, 1 deletion**. The sole production change removes `terrain.num_rows = 1` from `training/legged_gym/legged_gym/scripts/play.py`; the other paths add 77 lines of ordinary tests in `tests/test_checkpoint_runtime_wiring.py` and 17 worker-log lines.
- Read checkout AGENTS, current registered ownership, original P2/supplemental report, complete fix diff and necessary actual Go2/play/registry/environment-receipt context. The code-review/planning-with-files/git-guru instructions read earlier in this same review remain applicable.
- Read the final fix1 addendum in primary `task-7-report.md`, including exact postcommit 433-test and Gate A evidence, before issuing this report. Those are worker results, not relabeled independent executions.
- Sole write was this permitted primary coordination report. No production/test/config/index/ref/dependency/remote changes or sub-writer. Initial and final status including ignored/untracked files are empty; final HEAD is unchanged.

## Why the P2 is closed

The real `Go2PosRoughCfg.terrain.num_rows` is 10 (`go2_pos_config.py:154`), matching the accepted default YAML's `source_max_goal_level=10.0`. Fixed play retains that value while preserving its one-column, curriculum and maximum-initial-level settings (`play.py:106-109`). The original one-row mutation was the direct cause of the mismatch; its deletion allows the already bound configuration to reach the existing registry check unchanged.

The new positive regression runs real pure play preflight with an ordinary v2 inference manifest, actual BaseConfig → LeggedRobotCfg → LeggedRobotPosCfg → Go2PosRoughCfg definitions, the continuous actual play preparation AST slice, and the actual `make_env` preconstruction slice through `apply_gym_environment` and `reconcile_environment_receipt`. It confirms rows 10, columns 1, maximum initial level 3, request-matching receipt with stored level bounds `[0.0, 10.0]`, requested environment count and explicit controller-root binding. No copied environment implementation or simulator construction is used.

The negative parameter case changes the prepared row count to 9 and still reaches the unchanged equality guard at `task_registry.py:110-111`, rejecting before any receipt is created. The regression therefore proves both successful application of the accepted request and continued rejection of inconsistent actual configuration. Both cases retain `runtime_ready=False`, create no run root and import no Isaac module.

The complete production diff contains only the single deleted assignment. Registry equality/application/receipt checks, environment/config algorithms and runtime preflight are byte-for-byte unchanged by this fix. Static context confirms `require_runtime_prerequisites` remains before the real Gym import (`play.py:49-50`), cleanup/publication remain, and the post-runner reset/observation refresh (`:152-153`) and dictionary return (`:205-209`) remain. The previously rejected stale-first-observation and None-return candidates were not reopened or assigned new tests.

## Fresh independent verification

The following exact selection ran from the fixed review checkout with the dedicated CPU interpreter, disabled bytecode and disabled pytest cache:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_checkpoint_runtime_wiring.py tests/test_environment_profile.py \
  tests/test_runtime_cli_contract.py tests/test_runtime_manifest_contract.py \
  -k 'not rejects_expressions_without_execution'
```

**119 passed, 1 deselected in 35.89 s**, exit 0. This includes both new terrain/receipt cases, related real caller/model-load/README checks, Gym parent import-order and actual blocked-play behavior, and inherited shape/reset/trace/output contracts. The explicitly deselected case is `test_actual_gym_runner_registry_rejects_expressions_without_execution`; no converter or checkpoint-security test was selected. This is a fresh independent run on `774027d`, not the worker's 123- or 433-test run.

Python 3.8 grammar and in-memory compilation passed for the changed play and caller-test Python files without bytecode output. The actual check used:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python -B - <<'PY'
import ast,json
from pathlib import Path
paths=['training/legged_gym/legged_gym/scripts/play.py','tests/test_checkpoint_runtime_wiring.py']
for rel in paths:
    source=Path(rel).read_text()
    ast.parse(source,filename=rel,feature_version=(3,8))
    compile(source,rel,'exec')
print(json.dumps({'checked_files':paths,'python38_grammar':'passed','compile':'passed','bytecode_emitted':False}))
PY
git diff --check c7b9aa371b8ab3800adea378a7024f043fb58580..HEAD
git diff --stat c7b9aa371b8ab3800adea378a7024f043fb58580..HEAD
git rev-parse HEAD
git status --porcelain --untracked-files=all --ignored
git diff --quiet
git diff --cached --quiet
```

All checks exited 0. HEAD readback is `774027d1a975e318ad2f577b457951f51c82bfad`; final porcelain including ignored/untracked is empty, and source/index diffs are empty. Grammar acceptance does not assert Python 3.8 optional-dependency/API compatibility. Test stdout remains in tool outputs; no separate raw pytest-log file is claimed.

## Worker handoff and remaining boundaries

The final worker addendum reports genuine pre-fix RED at the terrain-row guard; focused 2-case GREEN; full precommit 433 passed in 65.61 s; fresh fixed postcommit 433 passed in 65.70 s; and five CPU/static Gate A passes plus four separate Gym/Lab dependency/runtime blockers. I read that fix1 addendum and kept its evidence attribution separate. I did not rerun the full suite or Gate A in this narrow review.

No simulator was imported, constructed or mocked; no adversarial fixture was constructed or executed. Full checkpoint-core, converter and security review remain excluded and incomplete after the earlier service restriction. Real Gym/Lab lifecycle and optimizer continuation, controller/provenance, physical replay/reset, accepted paper identity, formal metrics, rights and hardware remain unverified/blocked. The separate CBF hotpath finding is unchanged. The controller owns subsequent review/integration/frozen-candidate decisions; this report closes only the recorded caller P2.
