# Task 6 slice 1 independent re-review, fix round 1

Date: 2026-09-07. Reviewer: `/root/review_batch6_slice1`.

Fixed BASE: `6e097ac3d87554a8baf537aa733d72b3c7e70b33`.
Fixed HEAD: `7945a9e6037fee5045cbc12ac59a054ad958b43b`.
Read-only checkout: `../SEA-Nav-Code-batch6-review-slice1`.
Scope: complete three-file BASE-to-HEAD diff (`perception_delay.py`, its tests,
and worker log), original P2, and related scheduling regressions. This is not
Task 6 whole-consumer or simulator acceptance.

## Findings and disposition

**Original P2: ADDRESSED. No new actionable finding in this scoped fix.**

The implementation now converts finite supplied time to an epoch-relative
integer policy tick, uses integral acquisition/refresh deadlines, and advances
deadlines on the reset epoch's cadence. It no longer rebases the next deadline
on rounded float32 actual time. Supplied physical sample timestamps remain
unchanged for age and diagnostic transport-arrival comparisons.

The original 1000-tick reproduction, using its exact seed and inputs, now
reports zero skipped acquisitions and zero incorrect nominal upstream refresh
ages for both float32 and float64. The prior float32 output was `420 189`.

The added tests exercise both formula modes, dt `.02`/`.01`, both timestamp
dtypes, 1000 policy ticks, and a masked row reset at tick 337. They assert
acquisition counts, upstream ray/goal sample identities and held ages, and
preservation of the other row's stored timestamps. These directly cover the
original defect and its nonzero-epoch variant, not merely manifest fields.

## Fresh independent evidence

Executed from the fixed checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_navigation_reward.py tests/test_perception_delay.py
```

Output: **26 passed in 2.83s**. This includes the original 17 slice tests and
the nine new clock/finite-input cases. The worker's RED history was read in
full but was not recreated by modifying the immutable review tree.

The exact upstream reproduction script from `task-6-slice1-review.md` was
rerun on the fixed tree, with unchanged inputs. Output:

```text
original_probe torch.float64 0 0
original_probe torch.float32 0 0
```

An additional seeded diagnostic probe checked acquisition counts, output
refresh counts, and the actual retained timestamp plus sampled latency, not
just the rounded scheduling tick:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python - <<'PY'
import torch
from rsl_rl.perception_delay import PerceptionDelayConfig, TimestampedPerception

for dt in (.02, .01):
    for dtype in (torch.float64, torch.float32):
        cfg = PerceptionDelayConfig(
            mode='paper_diagnostic', policy_dt_s=dt,
            acquisition_period_s=.1, refresh_period_s=dt)
        p = TimestampedPerception(cfg, 1, 1)
        ids = torch.tensor([0])
        generator = torch.Generator().manual_seed(12)
        p.reset(ids, 0., torch.zeros(1, 1), torch.zeros(1, 2))
        acquired = 0
        skipped_refresh = 0
        future = 0
        for tick in range(1, 1001):
            now = torch.tensor([tick], dtype=dtype) * dt
            write = p.write.clone()
            refresh = p.next_refresh.clone()
            p.push(ids, now, torch.ones(1, 1), torch.zeros(1, 2),
                   generator=generator)
            acquired += int(not torch.equal(write, p.write))
            out = p.observe(ids, now, generator=generator)
            skipped_refresh += int(torch.equal(refresh, p.next_refresh))
            future += int(bool((out.sample_timestamp + out.sampled_latency
                                > now.double() + 1e-8).any()))
        print('diagnostic_probe', dt, dtype, acquired, skipped_refresh, future)
PY
```

Output columns after dtype are acquisition count, skipped policy refreshes,
and observations published before their sampled transport arrival:

```text
diagnostic_probe 0.02 torch.float64 200 0 0
diagnostic_probe 0.02 torch.float32 200 0 0
diagnostic_probe 0.01 torch.float64 100 0 0
diagnostic_probe 0.01 torch.float32 100 0 0
```

Additional checks:

- `ast.parse(..., feature_version=(3, 8))` passes for the changed source.
- `git diff --check BASE..HEAD` returns zero with no output.
- Review HEAD is exactly the fixed OID. Full porcelain including ignored
  paths remains empty after probes.
- No simulator module was mocked or loaded; all execution used the dedicated
  Torch 2.6.0+cpu interpreter.

## Contract and evidence limits retained

The scheduling conversion explicitly requires policy-tick input clocks;
nearest-tick rounding is not a general asynchronous wall-clock scheduler.
Under that contract, rounding only selects a deadline, while the original
time controls physical age and diagnostic packet arrival. It does not create
intermediate packets to fill gaps. A caller that omits policy acquisitions
cannot claim the upstream uninterrupted acquisition profile merely because
this helper subsequently advances to the next epoch deadline. Arbitrary
off-grid/backward caller clocks are not validated by this scoped pass; final
caller wiring must preserve the declared clock contract.

`torch._assert_async` is actually present and exercised in the locked CPU
runtime, and rejects the tested nonfinite time input. Its availability in old
Isaac Gym Torch combinations is **not** established by Python 3.8 syntax
success. Per the controller's existing Task 6 handoff, capability preflight
must report unsupported runtime stacks blocked before environment creation;
that later wiring is not accepted here and is not counted as a new missing
slice feature.

The TorchDispatch test establishes absence of `_local_scalar_dense` in its
instrumented push/observe path. It is not a GPU timing result or proof that
all dynamic tensor indexing is free of host synchronization.

## Limited-scope verdict

**Spec: PASS. Quality: PASS.** Original P2 closed; no additional actionable
finding established in the three-file fix. Preserve the independent whole
Task 6 review for actual callers, capability/startup ordering, reward/perception
application, masked reconstruction and final identity/trace receipts.
Simulator, accepted paper and hardware evidence remain blocked.

Only this primary-coordination report was written. No source, worker moving
tree, commit, branch, ref or remote was changed by the reviewer.
