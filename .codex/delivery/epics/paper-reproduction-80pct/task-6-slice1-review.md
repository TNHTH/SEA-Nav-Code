# Task 6 slice 1 independent review

Date: 2026-09-07. Reviewer: `/root/review_batch6_slice1`.

Fixed BASE: `65dcbbc2b13c92af99a9e7d4cba4110880f9b509`.
Fixed HEAD: `6e097ac3d87554a8baf537aa733d72b3c7e70b33`.
Read-only review checkout: `../SEA-Nav-Code-batch6-review-slice1`.
Scope: complete BASE-to-HEAD diff (five files, 315 added lines): two shared
reward/perception modules, their two test files, and the worker log. This is an
incremental slice review, **not** Task 6 final consumer-wiring acceptance.

## Actionable finding

### [P2] Float32 per-environment timestamps silently change the sensor clock

Location: `training/rsl_rl/rsl_rl/perception_delay.py:95`, `:105`, `:118`,
`:134` (input conversion at `:66`–`:67`).

Failure input: the normal tensor expression
`now = torch.tensor([tick], dtype=torch.float32) * .02`, passed to `push` and
`observe` every policy tick after reset at zero. This API accepts tensor time
and converts it to float64, but that conversion cannot recover float32
precision already lost. The fixed `1e-8` due tolerance is smaller than that
loss from approximately 0.3 seconds onward. Advancing each deadline from the
rounded actual time (`times + period`, `now + period`) then skips scheduled
acquisitions/refreshes and permanently shifts their phase. There is no
documented float64-only input requirement or rejecting validation.

Observed CPU evidence with default upstream config, one environment, one ray,
seed 9, ticks 1–1000 (20 seconds):

| Time input | Skipped acquisitions | Nominal 100 ms refreshes with age outside .04/.06 s |
|---|---:|---:|
| float64 tensor | 0 | 0 / 200 |
| float32 tensor | 420 | 189 / 200 |

The first missed acquisition is tick 15: prior deadline
`0.3000000011920929`, supplied time `0.29999998211860657`, so even after
adding `1e-8` the sample is dropped. With seed 9, tick 20 reports age `.12`
seconds, tick 30 `.16`, and tick 35 `.14` at nominal refresh boundaries. Thus
the error affects the actual rays/goals and the claimed upstream discrete
delay distribution, not just timestamp display.

The separately named diagnostic mode has the same defect: with 1000 ticks and
seed 12, dt `.02` produces 178 acquisitions instead of 200 and skips 420
policy-output refreshes; dt `.01` produces 94 acquisitions instead of 100 and
skips 240 refreshes. Float64 inputs give the expected 200/100 acquisitions and
no skipped refreshes.

Minimal repair: use a stable policy-tick/epoch schedule for these already
integral-period clocks, with a defined conversion for finite per-env time
inputs and no device-to-host scalar extraction. Do not repeatedly rebase the
schedule on rounded actual arrival time. Alternatively an explicitly enforced
float64-only timestamp contract must cover every real caller before startup;
silently accepting float32 is not safe. Add long-enough float32 and float64
regressions at `.02` and `.01`, including masked reset with a nonzero epoch,
and assert actual sample sequence/cadence and held ages rather than only
`sample_timestamp <= now`. Raising the fixed epsilon without a precision and
horizon contract only moves the failure point.

## Independent verification

From the fixed review checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_navigation_reward.py tests/test_perception_delay.py
```

Result: **17 passed in 0.87s**. Both changed source modules also parse with
Python 3.8 grammar through `ast.parse(..., feature_version=(3, 8))`.
`git diff --check BASE..HEAD` passes. Fixed checkout HEAD remains the reviewed
OID; full porcelain including ignored paths is empty after all probes.

Exact upstream clock reproduction (real Torch 2.6.0+cpu; no simulator mocks):

```python
import torch
from rsl_rl.perception_delay import PerceptionDelayConfig, TimestampedPerception

for dtype in (torch.float64, torch.float32):
    p = TimestampedPerception(PerceptionDelayConfig(), 1, 1)
    ids = torch.tensor([0])
    p.reset(ids, 0., torch.zeros(1, 1), torch.zeros(1, 2))
    generator = torch.Generator().manual_seed(9)
    skipped = []
    mismatches = []
    for tick in range(1, 1001):
        now = torch.tensor([tick], dtype=dtype) * .02
        before = p.write.clone()
        p.push(ids, now, torch.full((1, 1), float(tick)),
               torch.full((1, 2), float(tick)), generator=generator)
        if torch.equal(before, p.write):
            skipped.append(tick)
        out = p.observe(ids, now, generator=generator)
        if tick % 5 == 0 and round(out.actual_age.item(), 4) not in (.04, .06):
            mismatches.append(tick)
    print(dtype, len(skipped), len(mismatches))
```

Output:

```text
torch.float64 0 0
torch.float32 420 189
```

## Limited-scope verdict

- **Spec: NEEDS FIX (one P2)** for the accepted tensor timestamp contract and
  its resulting sensor cadence. The existing tests do not expose it because
  `tick * dt` is computed as a Python double.
- **Quality: NEEDS FIX (same P2)**. No additional actionable defect was
  established in this slice. No P0/P1 finding.
- The eight independent reward oracles, profile-specific equations,
  once-only dt weighting, recorded operational contact fallback, default
  float64 upstream history indices/hold ages, seeded diagnostic latency and
  masked noise-free reset checks pass within their tested scope.
- Caller mapping, actual Gym/adapter reward weighting, real clock dtype,
  post-reset geometry reconstruction, profile rejection, manifest/trace
  receipts and runtime preflight remain for the final whole Task 6 review.
  Their absence from this pure-module slice is not an additional finding.
- No simulator, controller, GPU timing, paper acceptance or hardware claim is
  made. Only this review report was written; no source, branch, commit or
  remote was mutated.
