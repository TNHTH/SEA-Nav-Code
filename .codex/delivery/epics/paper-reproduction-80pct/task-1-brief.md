## Global Constraints

- Working branches must remain exactly `main`, `stable`, and `test`; all repair commits land only on `test`.
- Integrate the dependency graph strictly as `1→2→3→4→5→6→7`; do not create concurrent detached commits touching PPO/CBF/runner/reset/trainer files.
- `main` and `stable` remain at `1c5675bbedf1dcbe5a4c1a91830cae528c780793` during this recovery.
- `paper_v1` uses Eq. 4 with `epsilon_d=1.0` and describes the result as a damped safety bias, never a hard-safe projection.
- Action stages are `distribution_mean`, `policy_action`, `clipped_policy_action`, and `executed_command`.
- Every repaired run explicitly records `implementation_delta=["ppo_state_identity_repair"]`; the resolver never adds it silently.
- CPU tests may import `rsl_rl`; `legged_gym` receives syntax/AST/tree checks only because importing its task stack requires unavailable `isaacgym`.
- Missing Isaac Gym/IsaacLab, formal 100-trial metrics, publication rights, and real hardware are `blocked`, never mocked into a pass.
- Use exact-path staging. Each task ends with an independently reviewable commit and a clean focused test run.

---

### Task 1: Establish a portable CPU Gate A and lazy optional W&B

**Files:**
- Create: `tools/gate_a.py`
- Create: `tests/test_runner_optional_wandb.py`
- Create: `tests/test_gate_a_portable.py`
- Modify: `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Modify: `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
- Modify: `sea_nav_current_isaaclab_full_method/adapters/manifest.py`
- Modify: `sea_nav_current_isaaclab_full_method/adapter_manifest.json`
- Modify: `sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml`
- Modify: `sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh`

**Interfaces:**
- Preserve `OnPolicyRunner.__init__(env, train_cfg, log_dir=None, args=None, device="cpu")`.
- Add `_wandb_enabled(args: object | None) -> bool` and `_require_wandb() -> ModuleType`.
- Add immutable `GateCase(name, status, detail)`, `discover_repo_root`, `run_cpu_gate`, `probe_isaac_gym`, `write_report`, and `main` in `tools/gate_a.py`.
- `probe_isaac_gym` returns `blocked` on `ModuleNotFoundError`; it does not make the CPU gate fail.
- The shell wrapper requires launcher and run directory from flags or named environment variables before creating output.

- [ ] **Step 1: Add W&B tests that fail against the eager import**

```python
def test_runners_import_without_wandb_installed():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(RSL_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", "import rsl_rl.runners; print('runner-imported')"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "runner-imported"


def test_missing_enabled_wandb_has_a_precise_error(monkeypatch):
    def missing(name):
        assert name == "wandb"
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="wandb logging requested"):
        on_policy_runner._require_wandb()
```

- [ ] **Step 2: Prove the W&B tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_runner_optional_wandb.py`

Expected: runner import fails with `ModuleNotFoundError: No module named 'wandb'` before the production patch.

- [ ] **Step 3: Implement the lazy dependency boundary**

```python
def _wandb_enabled(args: object | None) -> bool:
    return bool(getattr(args, "wandb", False))


def _require_wandb() -> ModuleType:
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb logging requested but the optional 'wandb' dependency is not installed"
        ) from exc
```

Remove the top-level W&B import. Resolve the client inside `wandb_log`, and call it only when `_wandb_enabled(self.args)` is true.

- [ ] **Step 4: Add portable Gate A tests**

```python
def test_gate_writes_only_to_the_requested_path(tmp_path):
    cases = run_cpu_gate(ROOT)
    assert all(case.status != "failed" for case in cases)
    report = tmp_path / "gate-a.json"
    write_report(report, cases)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["cases"]
    assert {row["status"] for row in payload["cases"]} <= {"passed", "blocked", "failed"}


def test_missing_isaac_gym_is_blocked(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))
    case = probe_isaac_gym()
    assert (case.name, case.status) == ("isaac_gym_runtime", "blocked")
```

- [ ] **Step 5: Prove the Gate A tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_gate_a_portable.py`

Expected: import fails because `tools.gate_a` does not exist.

- [ ] **Step 6: Implement Gate A and remove machine-bound output**

`run_cpu_gate` compiles tracked Python files, checks the complete `legged_gym` package/assets/entry points without importing its simulator-bound modules, then imports only the CPU-safe `rsl_rl` packages. It temporarily adds `training/rsl_rl` to `sys.path` and restores the previous list in `finally`. `write_report` writes sorted JSON only to the caller path. Change the legacy adapter Gate A preview to use `TemporaryDirectory`. Replace `/home/gwh` values in manifest/config examples, and make the shell wrapper reject absent launcher/run-directory inputs with exit 2.

- [ ] **Step 7: Run the green foundation gate**

```bash
RUN_AUDIT_DIR="$(mktemp -d /tmp/sea-nav-gate-a.XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/gate_a.py --repo-root "$PWD" --report "$RUN_AUDIT_DIR/gate-a.json"
bash -n sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh
git status --short
```

Expected: tests and gate pass; `isaac_gym_runtime` is `blocked`; the checkout contains no generated preview/report.

- [ ] **Step 8: Commit the foundation**

```bash
git add tools/gate_a.py tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py training/rsl_rl/rsl_rl/runners/on_policy_runner.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py sea_nav_current_isaaclab_full_method/adapters/manifest.py sea_nav_current_isaaclab_full_method/adapter_manifest.json sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh
git commit -m "test: add portable CPU gate foundation"
```

---
