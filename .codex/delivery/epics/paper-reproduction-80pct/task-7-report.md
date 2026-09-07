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
