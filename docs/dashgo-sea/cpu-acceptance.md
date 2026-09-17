# CPU / static acceptance

## Required selection (verification.md V2)

```bash
cd "$CANDIDATE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS
export PYTHONPATH="$CANDIDATE_ROOT/training:$CANDIDATE_ROOT/training/rsl_rl:$CANDIDATE_ROOT/packages/sea_nav_core/src:$CANDIDATE_ROOT/training/sea_nav_diffdrive_isaaclab"
"$CPU_PY" -B -m pytest -q -rs -p no:cacheprovider \
  tests training/rsl_rl/tests \
  sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py \
  packages/sea_nav_core/tests training/sea_nav_diffdrive_isaaclab/tests
git diff --check
```

## Validator

```bash
"$CPU_PY" tools/sea_validate_dashgo.py --level cpu --output-dir "$OUT" --python "$CPU_PY"
```

`runtime_status` must remain `NOT RUN` until an Isaac target stack gate is executed.

## Function → consumer → test (summary)

| Symbol area | Consumer | Tests |
| --- | --- | --- |
| common-t CBF | `execution.py`, `cbf_adapter.py` | `test_cbf.py`, `test_execution.py` |
| 550D history | `observation_pipeline.py` | `test_observation_pipeline.py` |
| delay queue | `perception_delay.py` | `test_perception_delay.py` |
| wrapper/bootstrap | `wrapper.py`, PPO | `test_wrapper.py`, rsl PPO tests |
| 2D actor | `actor_critic.py`, policy_factory | `test_actor_critic.py` |
| rewards | `rewards.py` | `test_rewards.py` |
| profiles | `configs/*`, core profiles | `test_profiles.py` |
| ACSI/snapshot | `acsi.py`, `snapshot.py` | `test_acsi.py`, `test_snapshot.py` |
| RNG/checkpoint | `rng.py`, `checkpoint.py` | `test_rng.py`, `test_checkpoint.py` |
| eval/export | `evaluation.py`, `export.py` | `test_evaluation.py`, `test_export.py` |
| CLI | `tools/sea_*.py` | `test_cli.py`, `test_static_api.py` |
