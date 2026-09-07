# Task 7 checkpoint contract audit

Scope: design Section 11, implementation-plan Task 7, `OnPolicyRunner` save/load, the three adapter loaders, and the initializer. This is a pre-implementation contract audit, not a general security review.

## Required corrections

### [P1] The stated recursive schema rejects real optimizer state

Design line 230 and plan line 635 permit only string-keyed dictionaries. After one Adam step, `optimizer.state_dict()` has the form `{"state": {0: {...}}, "param_groups": [{..., "params": [0]}]}`: keys under `optimizer_state_dict.state` are integer parameter IDs. Stringifying those keys without decoding them before `Optimizer.load_state_dict` is incorrect: PyTorch's integer `id_map` will not match them, so optimizer state is not attached to parameters. PyTorch 2.6 Adam parameter groups also contain `None` values.

Correct the payload grammar, not the optimizer output:

- payload keys are exactly `model_state_dict`, optional `optimizer_state_dict`, and `iteration`;
- ordinary dictionaries are string-keyed;
- `optimizer_state_dict` has exact top-level keys `state` and `param_groups`;
- only `optimizer_state_dict.state` may have non-negative integer keys; nested per-parameter dictionaries remain string-keyed;
- `param_groups` is a list of string-keyed dictionaries and each `params` value is a list of non-negative integers;
- primitive values explicitly include `None`, `bool`, `int`, finite `float`, and `str`.

The round-trip test must perform at least one optimizer step and assert integer state keys and restored Adam moments. An empty optimizer state would miss the defect.

### [P1] Two `os.replace` calls are not an atomic two-file transaction

Design line 234 and plan line 635 are underspecified. Replacing a stable payload name and then its manifest leaves a window in which the old manifest names new bytes and fails hash validation. Reversing the order can expose a new manifest naming old bytes. This fails availability even if validation fails closed.

Use immutable, content-addressed payload generations and make the manifest the sole commit point:

1. Serialize to a same-directory `0600` temporary regular file; flush and `fsync` it.
2. Hash and size that same open file descriptor.
3. Publish it as a basename such as `<manifest-stem>.<sha256>.pt`. Never overwrite or delete an older referenced generation in the save transaction.
4. Write the canonical manifest to another same-directory temporary file, flush and `fsync`, then replace the public manifest name last.
5. `fsync` the containing directory after payload publication and again after manifest replacement if crash durability is claimed.

A crash may leave an unreferenced payload, which is safe and can be garbage-collected separately. A reader that has opened either the old or new manifest can still open the immutable generation it names. Add fault-injection tests at every publication boundary; accepted reads must be wholly old or wholly new.

### [P1] Resolve/check/hash/load by pathname leaves a TOCTOU gap

“Canonicalize containment and reject symlinks before hashing” (plan line 635) is insufficient if hashing and `torch.load` reopen the path. The path or a parent directory can be exchanged after validation, and `Path.resolve()` itself does not pin the checked object. The manifest itself also needs the same protection.

Make loading descriptor-based:

- constrain the manifest's `artifact_path` to a canonical adjacent basename (no slash, `.` or `..`), matching the design's adjacent-manifest requirement;
- open the artifact root/manifest directory and path components with directory file descriptors plus `O_DIRECTORY|O_NOFOLLOW`; open manifest and artifact with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`;
- require `fstat` to report regular files;
- parse the manifest from its already-open descriptor;
- size-check, hash, rewind, and call `torch.load(open_file_object, ..., weights_only=True)` on the *same artifact descriptor*; never reopen by pathname;
- publish with dirfd-relative basenames (`os.replace`/`os.rename` dir-fd arguments) in the already-open directory.

If a platform cannot provide these no-follow/dirfd guarantees, return an explicit unsupported-platform `CheckpointError`; do not silently fall back to resolve-then-open. Tests should cover symlinked root components, manifest, artifact, and a deterministic path swap between validation and deserialization.

### [P1] The legacy converter is not isolated merely because it is a separate function

Plan lines 604 and 658 put `convert_legacy_checkpoint_once` next to the safe runtime API, while legacy `torch.load` necessarily executes pickle code. A pre-hash establishes identity, not safety; “offline” and “no credentials” cannot be guaranteed by an in-process function.

Keep unsafe conversion out of `rsl_rl.utils` exports and out of every training/runtime parser. Use a separate operator-invoked executable plus an OS-sandbox launcher. On this host, Bubblewrap 0.6.1 is installed and a minimal `--unshare-all` probe succeeds, so isolation is feasible in principle. The launcher contract should:

- require an operator-approved SHA-256 and hash an `O_NOFOLLOW` input descriptor before any legacy load;
- pass that pinned input read-only (Bubblewrap supports `--ro-bind-fd`), unshare networking, clear the environment, bind no home/credential locations, run as an unprivileged identity, and apply CPU/memory/time/file-size limits;
- bind only an empty, no-symlink output directory as writable;
- treat the converter process and all its output as untrusted;
- after the sandbox exits, independently validate the emitted v2 manifest/payload with the safe PyTorch 2.6 loader and require the expected input hash in a separate conversion receipt.

If the sandbox or a required isolation control is unavailable, conversion is `blocked`. Unit tests can prove argument/hash/output-root gates; they must not describe ordinary Python subprocess separation as a security boundary.

### [P1] Iteration currently means different things at save time, in filenames, and at resume

`OnPolicyRunner.learn` iterates from `current_learning_iteration` but updates that field only after the loop (runner lines 110–165). Intermediate `model_<it>.pt` files therefore contain the stale pre-loop value at runner line 252. The two trainers then use lexical `sorted(model_*.pt)[-1]` (trainer lines 612–613 and 1721–1722), which can select `model_90.pt` over `model_100.pt`. Their current `--init-checkpoint` paths load only model weights and do not restore optimizer or iteration, so they are warm starts, not resumes.

Define `iteration` as the number of successfully completed PPO update iterations (equivalently, the next zero-based loop index). Capture `start_iteration` for existing logging thresholds; after each successful `alg.update()`, set `current_learning_iteration = it + 1`, then save using exactly that value in payload, manifest, and filename. Payload and manifest iteration must match and be a non-negative integer (reject `bool`). `learn` should return the final manifest path; callers must not glob filenames.

Expose distinct, mutually exclusive training semantics:

- `--init-checkpoint-manifest`: require model-only, iteration-zero input; load model weights and begin at iteration 0;
- `--resume-checkpoint-manifest`: require model plus optimizer; restore both and set `current_learning_iteration` from the verified manifest/payload.

The runtime smoke accepts only `--checkpoint-manifest` and loads model weights for inference. Add a CPU runner test for train N, resume M, and final iteration `N + M`, plus an intermediate-save test that checks filename/payload/manifest equality.

After optimizer load, also synchronize `self.alg.learning_rate` from the restored optimizer parameter group; otherwise the selectable adaptive schedule resumes with the constructor's initial scalar and can overwrite the restored learning rate on its next KL adjustment. Because schema v2 deliberately excludes RNG, environment, and rollout state, describe this as model/optimizer continuation at an accurate iteration, not bit-for-bit trajectory continuation.

### [P1] Safe-load capability must be split by runtime, not inferred from the CPU venv

The recorded CPU environment is Python 3.10/Torch `2.6.0+cpu`, whose `torch.load` has an explicit keyword-only `weights_only`; it is suitable for Task 7 CPU/security tests and independent conversion-output validation. System Torch is `1.8.0a0`; its source signature is `load(f, map_location=None, pickle_module=pickle, **pickle_load_args)`, and it also currently has a NumPy ABI import conflict. It cannot provide safe resume. The CPU venv has neither simulator stack and cannot make simulator resume pass.

Use a behavior gate, not a version comparison:

```python
def require_weights_only(torch_module) -> None:
    try:
        supported = "weights_only" in inspect.signature(torch_module.load).parameters
    except (TypeError, ValueError):
        supported = False
    if not supported:
        raise CheckpointError("safe checkpoint load blocked: torch.load lacks explicit weights_only")
```

Run this before artifact deserialization and still pass `weights_only=True` explicitly. The environment gate also needs a benign schema-v2 round trip. Task 7 test commands must use the recorded CPU venv rather than bare `python3`; simulator entry points must report resume `blocked` under their old Torch instead of substituting the CPU environment or falling back to pickle. Fresh training/save may remain a separate capability.

## Recommended API contract

```python
@dataclass(frozen=True)
class CheckpointManifest:
    schema: str
    artifact_path: str          # adjacent basename only
    byte_size: int
    sha256: str
    iteration: int              # completed update count
    producer_commit: str
    resolved_config_sha256: str
    allowed_sections: tuple[str, ...]

@dataclass(frozen=True)
class LoadedCheckpoint:
    manifest: CheckpointManifest
    model_state_dict: dict
    optimizer_state_dict: dict | None
    iteration: int

def save_checkpoint_v2(
    manifest_path: Path,
    *,
    model_state_dict: Mapping[str, object],
    optimizer_state_dict: Mapping[str, object] | None,
    iteration: int,
    producer_commit: str,
    resolved_config_sha256: str,
) -> CheckpointManifest: ...

def load_checkpoint_v2(
    manifest_path: Path,
    *,
    artifact_root: Path,
    map_location: str | torch.device = "cpu",
    expected_resolved_config_sha256: str | None = None,
) -> LoadedCheckpoint: ...
```

`allowed_sections` must equal the actual state sections exactly; it is not a caller-controlled allowlist. Validate manifest iteration against payload iteration before returning. `OnPolicyRunner.save` should delegate to this API with an explicit completed iteration and no `infos`; `OnPolicyRunner.load` should accept only a manifest and return no arbitrary serialized object.

## Call-site wiring

| Call site | Required Task 7 behavior |
|---|---|
| `init_full_method_checkpoint.py` | Produce a model-only, iteration-0 v2 artifact through `save_checkpoint_v2`; move route/seed/contract metadata outside the pickle; require explicit producer commit and resolved-config hash. |
| `full_method_runtime_smoke.py:273-289` | Replace raw-path fallback loading with `--checkpoint-manifest` and `load_checkpoint_v2`; record manifest hash, payload hash, and iteration. |
| `train_full_method_ppo.py:589-599` | Replace raw model-only fallback loading with the distinct init/resume manifest modes above; use runner resume for optimizer+iteration. |
| `train_full_method_acsi_replay_ppo.py:1698-1708` | Same as the PPO trainer; do not duplicate payload-shape heuristics. |

Plan Task 7 should add the sandbox launcher/conversion receipt to its file list, replace bare `python3` verification with the recorded CPU interpreter, add crash/race and non-empty-Adam tests, and state explicitly that simulator optimizer resume/post-resume training remain blocked until exercised in a locked simulator runtime with an explicit `weights_only` capability.
