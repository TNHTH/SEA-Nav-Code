# Task 7 — schema-v2 checkpoint and caller recovery

Status: fixed implementation candidate, complete postcommit CPU/static verification passed. Independent whole review is **INCOMPLETE**, not PASS; the core reviewer encountered a service-side safety restriction. No integration, remote publication, simulator acceptance or final-project completion is claimed.

## Fixed revisions and ownership

- Sole source writer: `/root/implement_batch7`; worker `../SEA-Nav-Code-batch7`, detached.
- BASE: `562d4ae0b56ca977ea433def6dbd05607284e38b`; registered shared coordination revision8 / local sync8 before source edits. The controller verified this exact fresh worker baseline:342 passed33.94s; worker read back exact HEAD and empty starting porcelain.
- First slice: `2387cf02d85a1a9a52b0e85da54c7ebad09b0cec` — safe schema/registry/real runner persistence,8 paths including worker log.
- Second slice: `c7b9aa371b8ab3800adea378a7024f043fb58580` — every manifest caller, isolated operator conversion, stronger sealed-byte identity and current README examples,23 paths.
- Full review range: `562d4ae0b56ca977ea433def6dbd05607284e38b..c7b9aa371b8ab3800adea378a7024f043fb58580`. Second-slice range: `2387cf02d85a1a9a52b0e85da54c7ebad09b0cec..c7b9aa371b8ab3800adea378a7024f043fb58580`.
- Full range changes25 source/test/document paths from the26-path registered inventory plus the committed local `task-7-log.md`. `tests/test_runtime_manifest_contract.py` did not require edits. No additional path was requested, no baseline path deleted, and no CBF/PPO/replay core was changed.
- Author/committer: existing repository identity `TNHTH <174231229+TNHTH@users.noreply.github.com>`; no configuration, named branch, remote or primary source write occurred. This report is the sole authorized primary-checkout write.
- Only local task_plan.md/resume_state.json registration remains unstaged; it is excluded from both commits. No untracked/ignored output remained at commit readback. All source commits/worktrees are retained.

## Behavior delivered

### Checkpoint bytes, schema and identity

`rsl_rl.utils.checkpoint` exposes frozen manifest/result records and explicit CheckpointError. Payload keys are exactly model_state_dict, optional optimizer_state_dict, and iteration. Ordinary nested dictionaries require string keys; tensors, None and finite primitive/container values are validated. Only Adam state's top-level parameter IDs may be nonnegative integers. Adam state requires real step/first/second moments, optional maximum moment and well-formed parameter groups; integer keys are retained through load, never stringified. Arbitrary infos are excluded.

Producer commit and resolved-config SHA-256 are mandatory save inputs supplied by actual callers. The manifest's schema/keys, adjacent basename, size/hash, nonnegative completed iteration and exact allowed_sections are checked before returning data. Unknown fields, duplicate JSON keys, path escape, symlinks, malformed metadata, inconsistent sections/iteration and raw legacy input fail closed.

Save writes/fsyncs a0600 same-directory temporary payload, publishes a content-addressed generation without replacing an existing generation, fsyncs its directory, then writes/fsyncs and atomically replaces the public manifest last. Older generations remain available. Four fault boundaries verify either the old or newly published manifest stays readable, and an old manifest copy continues to reference its retained generation. This is manifest publication, with directory sync; no garbage collection or deletion is added.

All root/path components are pinned with no-follow directory descriptors; manifest/artifact reads are no-follow regular-file opens. A concrete pure-Tensor regression showed that a pinned source inode could still be overwritten in place after hashing, yielding777 weights under the old digest. The final loader therefore copies from that pinned descriptor into a private Linux memfd, applies WRITE/GROW/SHRINK/SEAL seals, checks them, and performs size/hash/torch.load on the same sealed snapshot descriptor. Missing kernel sealing support is an explicit blocker. Parent-directory swaps, artifact pathname swaps and equal-size same-inode writes cannot change the loaded verified bytes. Failure paths close raw descriptors if fstat/fdopen fails.

`torch.load` must explicitly expose weights_only, and every ordinary load supplies True; no fallback or allowlist broadening exists. Runtime manifest preflight additionally exercises a real benign v2 save/load round trip in that runtime's interpreter. These API/platform checks are behavioral gates, not a Torch version comparison.

### Actual runner and class consumers

Both policy choices and PPO use exact registries, replacing runner eval calls. The Gym task registry's runner-class eval is independently replaced by the single allowed OnPolicyRunner mapping. Unknown expressions raise without executing.

Iteration now counts successfully completed PPO updates. The runner advances it immediately after each successful update, saves filenames/payloads/manifests with that count, and preserves the original start iteration separately for logging thresholds. `learn()` returns its actual final model_N.manifest.json path. The trainers consume that return value; lexical globs and latest-run lookup are removed/rejected.

Warm start requires model-only/iteration-zero input and a fresh runner. Resume requires model plus optimizer, restores the completed iteration, and synchronizes PPO's learning_rate scalar from the restored optimizer group. Inference applies only the model; it does not restore optimizer or iteration. Tests perform real OnPolicyRunner/PPO N=3 plus M=2 continuation with nonempty Adam and unchanged restored moments. Another test runs102 real updates, saves and independently reloads every publication (counts1..102 plus final102), then resumes and executes an actual adaptive update at iteration103: restored .0023 becomes .00345 in both scalar and optimizer group. A failing second update leaves completed count1 and a readable model_1 manifest.

### Every startup caller and current usage

Shared preflight accepts mutually exclusive --init-checkpoint-manifest, --resume-checkpoint-manifest or inference --checkpoint-manifest, plus explicit --producer-commit. Mode/manifest inconsistencies and old raw/numeric/loading flags reject before proprietary startup or output creation. The resolved config hash is supplied explicitly at save and required on load. The request retains checkpoint mode, exact pinned manifest-byte SHA, payload identity and actual-runtime capability receipt; actual loading checks the same manifest-byte hash again before state mutation.

Gym's parent remains Torch-free through real preflight. Its existing isolated CPU consumer child also resolves and checks the same configuration hash, performs manifest/capability checks and returns a receipt. The real Gym import remains after prerequisite checks. Train/play and task_registry consume the request; play now has the same preflight-only, bound-blocked-result and prerequisite/cleanup path as train. No fake Gym module is introduced. Existing controller/interface/provenance readiness stays false.

Both adapter trainers pass metadata through the actual shape-adapted runner constructor and load via the shared helper. Smoke loads verified model weights into its profile-built actor without replacing the CBF layer. The CPU initializer is import-safe, resolves the actual selected stack/profile and emits model-only iteration-zero v2 files; seed/source/contract metadata stays in a separate JSON sidecar. It no longer uses a developer-home path or mixed-metadata pickle.

Task6's runner_config_for_environment and applied_shapes remain; only necessary producer/request fixtures and checkpoint-specific GateA expectations were migrated. Shape conflicts, initialization/operator/terminal-once reset tests, smoke seed/horizon readback and no-replay behavior, trainer default-disabled tracing and explicit --trace PATH, four distinct action stages, physical row/path binding, attempt-all cleanup and exclusive output checks remain in the full passing suite. All independent mathematical/CBF/PPO/replay/identity assertions are preserved.

The root, Gym and adapter READMEs describe actual current startup/checkpoint usage. Upstream attribution and old installation material are retained; Preview3/Torch1.10/raw-number/latest-run guidance is explicitly historical. Eight new examples run through their actual pure parsers with real temporary manifests and required inputs, without simulator launch. The text distinguishes CPU2.6 verification from unknown old-simulator capability and states the blocked runtime/paper/provenance boundary.

### Operator-only legacy conversion

`tools/convert_legacy_checkpoint.py` is a separately invoked operator tool, not exported by rsl_rl and unreachable from training flags. Its ordinary entry requires a preapproved input SHA, explicit producer/config hashes, a dedicated CPU interpreter, nonroot Linux/bubblewrap/timeout, canonical no-follow input and an already empty no-symlink output directory. Input has a bounded size and is copied into the same sealed private snapshot used for hash approval and actual sandbox input.

The tool uses real bubblewrap unshare-all namespaces, no inherited credential environment/home mounts, no effective capabilities, an isolated network, pinned read-only runtime/code mounts and only the selected output directory as writable host storage. The input transfers from its approved sealed FD through bubblewrap ro-bind-data into a private read-only bind mount. This is necessary because installed bwrap0.6 cannot resolve an anonymous memfd as the source of ro-bind-fd; a genuine failed run and the correction are recorded. Host input path/inode mutation cannot change the approved bytes. Runtime/code/output are still mounted through pinned descriptors.

Before unsafe conversion, the sandbox verifies namespace identities, no credential/home visibility, read-only input mount, unprivileged identity, dropped capabilities and exact CPU/memory/file-size/open-file/process limits. Defaults are20CPU seconds,4GiB address space,64MiB per-file/input limit,64open files/processes and30wall seconds with external timeout termination. Worker stdout/stderr are discarded, so untrusted code cannot flood capture buffers or inject terminal output. Failed sandbox output is retained untrusted, with no acceptance receipt.

A second independently invoked OS-sandboxed process mounts output read-only and uses only the safe v2 loader; it never traverses the unsafe load branch. It checks exact output inventory, schema, producer/config identity, payload hash, pinned manifest hash and the separate input-hash receipt. Only after this succeeds does the outside parent write conversion-receipt.json exclusively and fsync it. The retained conversion-input.json is untrusted worker provenance until compared against the independently supplied approved hash.

Actual tests cover nonempty Adam conversion and external safe acceptance, hash mismatch, nonempty output preservation, missing sandbox failure, symlink rejection, untrusted host-side-effect denial, no inherited test credential/home/network interface, pinned pathname and same-inode input mutation, low-memory failure, wall-time termination and rejection of hostile generated output without a validation-side host marker. Earlier trusted namespace/Torch probes are only historical preparation, not substituted for these real converter runs. No claim of simulator compatibility or arbitrary unbounded resource safety is made; the tool enforces the explicit budgets above.

## Verification commands and history

All pytest commands ran from the worker using Python3.10.12/Torch2.6.0+cpu at the dedicated absolute interpreter. No system dependencies were installed or changed.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python \
  -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests \
  sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
```

Precommit final source: **431 passed in 63.28s**, exit0. Postcommit exact `c7b9aa371b8ab3800adea378a7024f043fb58580`: **431 passed in 63.46s**, exit0. This is the18th executed pytest invocation; it is a fresh complete run on the committed source, not a relabeled precommit result. Source/test/config files did not change between the two runs. The final report alone was written in primary while postcommit verification ran.

The committed task-7-log.md records all17 executed pytest invocations before commit, their selectors/results, one separate shell exit127 that never ran pytest, and two trusted converter diagnostics. Important retained failures include absent core APIs (four collection errors), eight caller RED failures, four absent converter failures, two converter root-path failures, the test-only misplaced inference assertion, same-inode pure-Tensor identity RED, and six bwrap anonymous-FD transfer failures. Those are not relabeled successful behavior. Focused final caller/security run:50 passed14.83s; corrected converter/security run:32 passed15.60s; preceding real102-update runner run:4 passed2.35s.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python \
  tools/gate_a.py --repo-root "$PWD" \
  --report /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/task-7-worker-gate-a-c7b9aa3.json
```

Precommit GateA report at sibling task-7-worker-gate-a-precommit.json was read back: five CPU/static passes, four separate Gym/Lab dependency/runtime blockers. The postcommit GateA command above also exited0 with passed_with_blockers; its actual physical JSON was read back and independently reported the same nine case statuses. It compiled91 Python files without output, retained65baseline paths, parsed20Gym files/nine static-only simulator-bound files, imported7CPU-safe packages and executed real actor/value/PPO/storage CPU smoke. Gym/Lab dependencies are absent; Gym/Lab runtime rows separately say not executed. The external report filename and this record bind the result to c7b9aa3; the report schema itself contains no commit field.

Postcommit Python3.8 `ast.parse(feature_version=(3,8))` and compile() both passed for91 tracked Python files without emitted bytecode; bash -n passed for the one tracked shell script. Full-range git diff --check and empty index checks passed. Exact full-range inventory was read back; no baseline deletion occurred. These checks do not imply optional libraries support Python3.8.

Raw GateA reports are external persistent local siblings, not committed or remotely visible artifacts. Pytest stdout exists in tool outputs; no separate raw pytest-log file is claimed. Worker source/log are committed, two local registration edits are not. Final HEAD readback remains c7b9aa3; final full ignored/untracked porcelain contains exactly those two registration changes and nothing else, and the index is empty. The full-range diff for PPO/CBF/replay core files is empty.

## Review and evidence limits

Independent core review did not finish because of a service-side safety limitation, and no core review PASS exists. The controller independently supplemented ordinary persistence/102-update/adaptive evidence on fixed2387cf0; that is not a whole-contract review and predates the final snapshot/caller/converter fixes. Independent whole review on the complete fixed range, controller serial integration and frozen final-candidate verification remain required.

This candidate closes local Task7 implementation/test scope only. Gym/Lab runtime and simulator optimizer continuation, exact controller interface/provenance, physical replay/reset, accepted paper_v1, formal100-trial metrics, public redistribution rights and hardware remain blocked/deferred. Model/optimizer continuation restores no RNG or physical state. The separately recorded CBF hotpath finding remains outside Task7, unchanged. Neither main/stable nor any remote ref was advanced.

## Bounded caller fix round 1 addendum — 2026-09-07

Fixed worker HEAD is now `774027d1a975e318ad2f577b457951f51c82bfad`, still detached. Focused fix range: `c7b9aa371b8ab3800adea378a7024f043fb58580..774027d1a975e318ad2f577b457951f51c82bfad`; original full Task7 range is `562d4ae0b56ca977ea433def6dbd05607284e38b..774027d1a975e318ad2f577b457951f51c82bfad`. The earlier431-test and c7b9aa3 evidence above stays bound to that older candidate; it is not relabeled as fix1 evidence.

The original bounded caller review found one P2: accepted Gym play inference preflight binds source_max_goal_level10, but actual play preparation overwrote terrain.num_rows to1 before the existing make_env equality guard. This inherited conflict was explicitly handed off by Task6. The sole production change removes that one row override from `training/legged_gym/legged_gym/scripts/play.py`. It does not remove or alter the registry equality/receipt checks, terrain/config algorithms, other playback/reset behavior, or runtime/controller/paper prerequisites.

The focused commit contains exactly three paths: play.py, `tests/test_checkpoint_runtime_wiring.py` and worker `.codex/delivery/epics/paper-reproduction-80pct/task-7-log.md`. Its source/test scope matches the controller's narrow fix1 registration. No other source, test, config or README path changed in this round; both local registration files remain unstaged. A read-only helper checked test-construction boundaries only; this is not the original reviewer's independent rereview.

The new ordinary parametrized regression executes the actual BaseConfig → LeggedRobotCfg → LeggedRobotPosCfg → Go2PosRoughCfg class definitions with real inspect/NumPy, the complete continuous play configuration-preparation AST slice including its single-env branch, and the actual registry preconstruction slice through apply_gym_environment/reconcile_environment_receipt. It stops before seeding/simulator construction. The positive case consumes a real v2 manifest through real play preflight and observes rows10, cols1, max_init3, exact request-matching receipt with stored_level_bounds[0.,10.], controller-root binding and requested num_envs restoration. The negative case changes prepared rows to9 and still gets the actual equality-guard error before a receipt exists. Both keep runtime_ready=False and create no run-root output or Isaac import. No simulator/environment stand-in is used.

### Fix1 executed verification

All five new pytest invocations used the dedicated absolute CPU interpreter already shown above, `PYTHONDONTWRITEBYTECODE=1`, worker-local `PYTHONPATH="$PWD/training/rsl_rl"`, and `-m pytest -q -p no:cacheprovider`.

| Task7 invocation | Actual selection and revision | Result |
|---|---|---|
| 19, genuine RED | new `-k actual_gym_play_preparation` on unchanged c7b9aa3 play source | 1 failed,1 passed,28 deselected in3.75s; exit1 at actual task_registry.py:111 terrain-row guard |
| 20, GREEN | same selection after the single-line fix | 2 passed,28 deselected in3.66s; exit0 |
| 21, ordinary related coverage | checkpoint_runtime_wiring, environment_profile, runtime_cli_contract, runtime_manifest_contract and runner_checkpoint test files; `-k 'not expressions'` | 123 passed,1 deselected in36.97s; exit0 |
| 22, complete precommit suite | `tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py` | 433 passed in65.61s; exit0 |
| 23, fresh complete postcommit suite | identical full selection on exact774027d1a975e318ad2f577b457951f51c82bfad | 433 passed in65.70s; exit0 |

The RED was the expected real consumer contradiction, not an AST/import harness failure. Source/test/config files did not change between GREEN, full precommit and fresh full postcommit. The committed worker log records commands19–22 and retained earlier failures; this addendum records actual postcommit command23. No separate raw pytest-log file is claimed; stdout remains in tool outputs. The full existing test suite is implementation regression coverage, not a retry, bypass or replacement of the service-restricted independent review.

Fresh precommit and postcommit GateA commands both exited0 with passed_with_blockers. Exact external reports, each physically read back:

- `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/task-7-worker-gate-a-fix1-precommit.json`
- `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/task-7-worker-gate-a-fix1-774027d.json`

Both used `tools/gate_a.py --repo-root "$PWD" --report PATH` with the same dedicated interpreter/environment. Five CPU/static cases passed:91 Python files compiled without output,65 retained baseline paths,20Gym files parsed/nine static-only,seven CPU-safe imports and real actor/value/PPO/storage smoke. Four distinct blockers persist: missing isaacgym, unexecuted real Gym runtime, missing isaaclab and unexecuted real Lab runtime. The report schema has no commit field; the postcommit invocation, filename and this record bind its evidence to774027d. These durable local sibling reports are not committed or remotely visible artifacts.

Python3.8 grammar plus in-memory compile and bash -n passed both precommit and freshly on fixed774027d:91 Python files and one shell script, no bytecode. Exact commit inventory and parent were checked; index is empty, no baseline path was deleted, and final ignored/untracked status contains only the two expected unstaged worker registration files. All other production behavior remains inherited from reviewed/recorded earlier slices.

### Remaining acceptance boundaries

This addendum hands the fixed range to the original caller reviewer for a bounded rereview; no rereview verdict is presumed. The separate full core/converter review remains **INCOMPLETE**, not PASS. The rejected first-observation and None-return conjectures were excluded from this fix. Original runtime/controller/provenance, accepted paper_v1, formal metrics, rights and hardware blockers remain unchanged; the CBF hotpath P2 remains separate and untouched. No integration, named-branch/ref/config change, remote publication, simulator acceptance or final-project completion was performed or authorized by these ordinary results.

## Native-v2-only rebuild addendum — 2026-09-08

Current candidate: `92ab65d23590256f3165d8cad6e64698cf11bfe8`, a single detached feature commit with sole parent `399ce2b08eac40865fd6496d19324f73a3e6cc7e`. This addendum is worker/self-verification evidence, **not independent acceptance**. New fixed-range review of `399ce2b08eac40865fd6496d19324f73a3e6cc7e..92ab65d23590256f3165d8cad6e64698cf11bfe8` remains required. All earlier candidate, converter and 431/433-test claims above are preserved as historical evidence only; they do not describe or accept this new native-v2-only distribution.

### Fixed identity, scope and retained history

- Sole writer: `/root/implement_batch7`; detached worker `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/SEA-Nav-Code-batch7-v2`; coordination/sync revision 9/9.
- Commit subject: `feat: add native v2 checkpoints and explicit manifest callers`. Existing approved author/committer remains `TNHTH <174231229+TNHTH@users.noreply.github.com>`; no identity/configuration change.
- Fresh baseline was exact `399ce2b`, with empty index and complete ignored/untracked porcelain. The old final `774027d1a975e318ad2f577b457951f51c82bfad` was read-only semantic reference; source replay used `apply_patch`, not cherry-pick, history import or `git apply`.
- Each old commit `2387cf02d85a1a9a52b0e85da54c7ebad09b0cec`, `c7b9aa371b8ab3800adea378a7024f043fb58580`, and `774027d1a975e318ad2f577b457951f51c82bfad` is not an ancestor of the new candidate (three actual `git merge-base --is-ancestor OLD HEAD` checks returned 1). Old commits, refs and worktrees are retained unchanged by this worker.
- Exact committed inventory is 25 paths: the current binding brief's 23 retained source/test/document paths plus that brief and the new worker log. `tests/test_runtime_manifest_contract.py` remains unchanged and was executed. No baseline file was deleted. No CBF/PPO/actor/replay core, adapter-shield, differential-drive package, controller asset or unrelated configuration was modified.
- The current binding section explicitly supersedes historical converter and two-commit clauses. The two excluded converter files were never copied to the new worker and are absent both on disk and in the committed tree. No replacement converter/API, hidden worker CLI or unsafe-load fallback was added.

Exact changed paths, relative to this fixed candidate's repository root:

```text
.codex/delivery/epics/paper-reproduction-80pct/task-7-brief.md
.codex/delivery/epics/paper-reproduction-80pct/task-7-log.md
README.md
sea_nav_current_isaaclab_full_method/README.md
sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py
sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py
sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py
sea_nav_current_isaaclab_full_method/train_full_method_ppo.py
tests/test_checkpoint_runtime_wiring.py
tests/test_checkpoint_security.py
tests/test_checkpoint_v2.py
tests/test_environment_profile.py
tests/test_runner_checkpoint.py
tests/test_runner_registry.py
tests/test_runtime_cli_contract.py
training/legged_gym/README.md
training/legged_gym/legged_gym/scripts/play.py
training/legged_gym/legged_gym/scripts/train.py
training/legged_gym/legged_gym/utils/helpers.py
training/legged_gym/legged_gym/utils/task_registry.py
training/rsl_rl/rsl_rl/runners/on_policy_runner.py
training/rsl_rl/rsl_rl/runtime_preflight.py
training/rsl_rl/rsl_rl/utils/__init__.py
training/rsl_rl/rsl_rl/utils/checkpoint.py
```

### Native behavior and new regressions

All 13 retained production-code paths, all retained native checkpoint/security/runner tests, the environment/CLI test migrations and Gym README are blob-identical to old `774027d`: 20 paths checked against both worktree bytes and committed blob IDs. Within the 23 source/test/document paths, the only differences from that old final semantic source are two README conversion promises replaced by explicit native-v2-only guidance and 58 added lines in `tests/test_checkpoint_runtime_wiring.py` (11 cases). This preserves sealed same-byte verification, immutable manifest-last publication, exact native schema/Adam state, explicit registries/producer/config identity, real completed-update/LR continuation, all manifest callers and the accepted play terrain-row correction.

The new distribution test checks absence of `tools/convert_legacy_checkpoint.py` and `tests/test_checkpoint_legacy_converter.py`; absence of `convert_legacy_checkpoint_once` from checkpoint/public utility APIs and production function definitions; no exact hidden converter flag constants; no current README tool/receipt/operator promise; and explicit literal `weights_only=True` at every statically identified production Torch load call, including import aliases. This is a known-call-site AST regression, not a whole-program proof against arbitrary dynamic reflection. The native loader security regressions, including benign raw-format rejection and duplicate-key validation, are retained and executed; no converter executable test is part of the new suite.

Ten parametrized cases exercise actual shared Gym train preflight with split/equal forms of `--checkpoint`, `--init-checkpoint`, `--resume`, `--load_run`, and `--load-run`. Every case receives the exact `legacy raw/numeric checkpoint flags are unsupported` rejection before run-root creation. The readonly helper supported test construction only, not independent acceptance; a later optional final-check helper call was refused by agent capacity and did not run or supply evidence.

### Fresh executed verification

The new worker's invocation numbering starts at 1. Earlier old-worker invocations 1–23 remain historical and are not reused. Every actual pytest command used the dedicated absolute CPU interpreter, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH="$PWD/training/rsl_rl"`, and `-m pytest -q -p no:cacheprovider` from the new worker.

| New invocation | Selection / revision | Actual result |
|---|---|---|
| 1 | Full suite on clean exact `399ce2b`, before source edits | 342 passed in 34.53s; exit 0 |
| 2 | Five replayed native files: checkpoint_v2, checkpoint_security, runner_registry, runner_checkpoint, checkpoint_runtime_wiring; production APIs not yet replayed | 5 collection errors in 0.94s; exit 2, genuine missing-checkpoint/resolver RED |
| 3 | checkpoint_runtime_wiring `-k native_v2`, after production replay but before README corrections | 1 failed, 10 passed, 30 deselected in 1.55s; exit 1 at obsolete README conversion promise |
| 4 | Five native files plus environment_profile, runtime_cli_contract and runtime_manifest_contract | 180 passed in 39.25s; exit 0 |
| 5 | Complete precommit source | 432 passed in 52.49s; exit 0 |
| 6 | Fresh complete postcommit source at exact `92ab65d23590256f3165d8cad6e64698cf11bfe8` | 432 passed in 52.85s; exit 0 |

Full command for invocations 1, 5 and 6:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python \
  -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests \
  sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
```

Source/test/config files did not change between the final precommit and postcommit runs. The committed worker log records invocations 1–5; this authorized primary addendum records actual postcommit invocation 6 without creating a second feature commit. Two pre-action orchestration JavaScript syntax failures executed no tool or pytest action and were retried successfully; their records remain in the worker log. No separate raw pytest log is claimed; actual stdout is retained in tool outputs.

Fresh precommit and postcommit Gate A commands both exited 0 with `passed_with_blockers`; both physical external JSON reports were read back completely:

- `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/task-7-v2-worker-gate-a-precommit.json`
- `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/task-7-v2-worker-gate-a-92ab65d23590256f3165d8cad6e64698cf11bfe8.json`

Postcommit command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python \
  tools/gate_a.py --repo-root "$PWD" \
  --report /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/task-7-v2-worker-gate-a-92ab65d23590256f3165d8cad6e64698cf11bfe8.json
```

Both reports contain five CPU/static passes: 89 tracked Python files compiled without output, 65 complete baseline files, 20 Gym package files parsed with 9 static-only, 7 CPU-safe package imports, and real actor/value/PPO/storage CPU smoke. Four separate blockers remain: missing isaacgym, unexecuted real Gym runtime, missing isaaclab, and unexecuted real Lab runtime. Report schema has no commit field; the exact postcommit invocation, full-SHA filename and this record bind that evidence. External local reports are not committed or remotely visible artifacts.

Separate precommit and fresh postcommit `ast.parse(feature_version=(3,8))` plus in-memory `compile()` passed for 89 tracked Python files, with `bash -n` passing for the one shell file. Postcommit fixed HEAD/sole-parent readback, exact 25-path inventory, full-range `git diff --check`, no baseline deletions, old-three non-ancestry, disk/tree converter absence, no protected-core/package diff, and empty index all passed. Each staged path was checked as not ignored before commit. Named branches remain exactly `main`, `stable`, and `test`; this detached worker changed no named branch, recovery ref, remote ref or dependency environment.

### Handoff state and remaining blockers

Worker source/log are committed at the exact candidate above. Full ignored/untracked porcelain contains only these two intentionally unstaged local registration files, with no ignored caches or generated output:

```text
.codex/delivery/epics/paper-reproduction-80pct/task_plan.md
.codex/delivery/epics/paper-reproduction-80pct/resume_state.json
```

This appended primary `task-7-report.md` is an authorized **uncommitted coordination change**, not a worker feature-commit path and not a source mutation. Old report evidence is preserved. New independent whole fixed-range review, controller serial integration and frozen-candidate verification remain pending; old caller PASS and this worker's self-checks cannot substitute.

Gym/Lab execution, simulator optimizer continuation, controller interface/provenance, physical replay/reset, accepted `paper_v1`, formal metrics, public redistribution rights and real hardware remain blocked/deferred. Continuation does not restore RNG or exact physical trajectories, and `runtime_ready=False` remains. The separate CBF hotpath P2 is untouched. No integration, push, stable promotion, simulator acceptance or final-project completion is claimed. Source work stops at this fixed boundary pending controller/reviewer direction.

## Independent native-v2 review verdict — 2026-09-08

The independent whole review of `399ce2b08eac40865fd6496d19324f73a3e6cc7e..92ab65d23590256f3165d8cad6e64698cf11bfe8` is **FAIL**, not acceptance. The reviewer changed no file or ref. Its focused CPU selection completed with **173 passed, 4 deselected in 37.58s** after including the established Gate A static import setup; simulator execution remained excluded.

Three reproducible P2 defects require a new fixed revision:

- `OnPolicyRunner.load()` and `apply_model_checkpoint()` mutate model parameters before every later check can succeed. An unexpected model key changes `std` from `0.3` to `123` before strict loading raises; a resume optimizer-group mismatch likewise leaves the model changed. Inference, warm-start, resume and the model-only helper therefore need prevalidation plus all-or-nothing state application.
- Native-v2 validation accepts Adam states whose moments have the wrong target-parameter shape, whose `amsgrad=True` state lacks `max_exp_avg_sq`, or whose parameter group lacks required fields such as `betas`. Loading succeeds and the next real `learn(1)` fails. Validation must bind group/order/count and conditional state to the target optimizer parameters before mutating anything.
- A hash-valid checkpoint with the declared/resolved 180-degree configuration can still replace the actor's persistent `cbf_layer.ray_unit_vectors` with a 240-degree buffer. Preflight and apply succeed while the configuration receipt remains unchanged. Config-derived fixed geometry must be compared with the freshly constructed target and rejected before application.

One P3 evidence defect is also assigned: both adapter trainers compute `checkpoint_bytes` from the manifest path (474 bytes in the probe) instead of the manifest-declared payload size (27,002 bytes). They must report the validated payload `byte_size`, or use a separately named manifest-size field.

Benign paths retained positive evidence: same-generation reuse, corrupt-generation refusal without changing the prior manifest, no-dirfd fail-closed before `torch.load`, converter absence, explicit `weights_only=True`, sealed snapshot loading, manifest-last publication, completed-update numbering and existing documentation/parser cases. These successes do not close the four findings above. A single fix writer is active; a fresh independent rereview is mandatory before integration or push.
