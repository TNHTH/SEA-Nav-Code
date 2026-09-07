# Task7 fixed-core ordinary functional verification

2026-09-07; controller, independent of the source implementer.

**Limited functional checks PASS; full core Spec/Quality review INCOMPLETE.**
No actionable ordinary runner/persistence defect was established in the
specific paths/checks below. This is not a replacement full review or a
security verdict, and does not authorize Task7 integration.

## Identity and service failure

- BASE: 562d4ae0b56ca977ea433def6dbd05607284e38b.
- Fixed core: 2387cf02d85a1a9a52b0e85da54c7ebad09b0cec.
- Checkout: ../SEA-Nav-Code-batch7-review-core, detached and clean including
  ignored before and after these checks. Production/source/test/ref unchanged.
- Read complete checkpoint.py, runner/utils diff, committed worker log, and
  the three selected functional test files plus actual PPO adaptive-LR code.
- Original independent reviewer ended with the service error "This content
  was flagged for possible cybersecurity risk". No task-7-core-review.md exists
  in primary or the fixed checkout. This is a service failure, not a failed
  repository test. Its partial observations, if returned, remain separate.
- No retry/bypass of that restriction, changed safety control or new adversarial
  fixture was used here. Only the ordinary functional subset below ran; do not
  infer the remaining full review from it or relabel the worker's387 tests.

## Fresh ordinary functional checks

From the fixed checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_runner_checkpoint.py tests/test_checkpoint_v2.py \
  tests/test_runner_registry.py::test_exact_registry
```

Exit0, **24 passed in1.48s**. Covers real nonempty Adam/None/integer-key
roundtrip, strict ordinary data grammar and manifest binding, distinct
warm-start/resume/inference effects, successful-versus-failed update count,
3+2 continuation, and valid exact registry entries. This selection excludes
test_checkpoint_security.py, converter work and registry-expression probes.

An additional real102-update run crossed the old100-update threshold, with
actual v2 publication/loading after every save. A wrapper observed the real
update and real save methods but did not substitute their work. Every manifest,
payload and filename agreed with the actual completed-update count. The
periodic/final save sequence was exactly1..102 followed by102. After setting
the persisted optimizer LR to.0023, a new runner restored all13 Adam entries
including exp_avg, exp_avg_sq and step, plus iteration102. Its next actual
adaptive update produced LR.00345 and a real iteration103 artifact. Thus the
restored scalar is used by the adaptive consumer, not merely assigned.

Exact command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl:$PWD/tests" ../sea-nav-cpu-venv/bin/python -B - <<'PY'
import contextlib, io, json, tempfile
from pathlib import Path
import torch
from test_runner_checkpoint import runner
from rsl_rl.utils.checkpoint import load_checkpoint_v2
with tempfile.TemporaryDirectory(prefix='sea-nav-runner-functional-') as probe_dir:
    root=Path(probe_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        first=runner(root/'first')
        actual_updates=[0]
        original_update=first.alg.update
        def update():
            result=original_update()
            actual_updates[0]+=1
            return result
        first.alg.update=update
        records=[]
        original_save=first.save
        def save(path):
            result=original_save(path)
            loaded=load_checkpoint_v2(path,artifact_root=root)
            assert loaded.iteration==first.current_learning_iteration==actual_updates[0]
            assert Path(path).name=='model_%d.manifest.json'%loaded.iteration
            records.append(loaded.iteration)
            return result
        first.save=save
        final=first.learn(102)
        assert records==list(range(1,103))+[102]
        first.alg.optimizer.param_groups[0]['lr']=0.0023
        first.save(final)
        second=runner(root/'second')
        second.load(final,artifact_root=root,mode='resume')
        assert second.current_learning_iteration==102 and second.alg.learning_rate==0.0023
        for old,new in zip(first.alg.optimizer.state.values(),second.alg.optimizer.state.values()):
            for key in ('exp_avg','exp_avg_sq','step'):
                assert torch.equal(old[key],new[key])
        second.alg.schedule='adaptive'
        second.alg.desired_kl=0.01
        resumed=second.learn(1)
        assert second.alg.learning_rate==second.alg.optimizer.param_groups[0]['lr']
        assert abs(second.alg.learning_rate-0.00345)<1e-12
        assert load_checkpoint_v2(resumed,artifact_root=root).iteration==103
    print(json.dumps({'fixed_core':'2387cf02d85a1a9a52b0e85da54c7ebad09b0cec','real_updates_before_resume':actual_updates[0],'first_saved_iteration':records[0],'last_saved_iteration':records[-1],'restored_adam_entries':len(second.alg.optimizer.state),'resumed_iteration':second.current_learning_iteration,'adaptive_lr_after_first_update':second.alg.learning_rate,'boundary':'ordinary CPU fixture; no RNG/physical trajectory or complete security review'}))
PY
git status --porcelain=v1 --untracked-files=all --ignored
```

Exit0 in2.52s; output values: updates102, first saved1, last saved102,
restored Adam entries13, resumed iteration103, adaptive LR.00345. Final full
porcelain empty. Generated artifacts were disposable CPU fixtures, not delivery
or recovery data; the report records their executed behavior, not retained files.

## Remaining scope

The original complete core review remains incomplete. No crash/race/adversarial
probe or converter verification was added by this controller subset. Actual
Gym/adapter/initializer manifest callers, converter isolation and README
commands are still the writer's second slice and require their own complete
acceptance. Suggested promoting the102-update/adaptive assertions into the
already-owned test_runner_checkpoint.py, without assigning another writer.

CPU model/optimizer continuation does not restore RNG, environments or
trajectories, and is not simulator optimizer-resume evidence. All existing
runtime, accepted-paper, CBF follow-up, frozen-candidate and remote gates remain.

## Partial helper summary and failed handoff

The original reviewer's helper `checkpoint_fd_edge` had a completed summary
visible through list_agents before the controller requested results handoff.
It reported an externally writable same-inode payload changing after hashing
could yield loaded weight9.0 while the manifest still carried the digest of
weight1.0; weights_only remained enabled. The necessary precondition is an
external writer to that inode, not merely renaming its path. The summary also
mentioned a synthetic fstat exception leaking a descriptor (no real OS failure
demonstrated),90 ordinary rejected loads without FD growth, and four publication
fault points keeping a readable old generation with no temporary leftovers.

Its detailed commands had been sent to its failed parent, not this controller.
A request to relay only those completed observations also ended with the same
service safety error. No further retry was attempted. The controller therefore
has only the earlier compact summary, not the full command/output needed to
independently certify it. These are **reported partial observations/candidates**,
not additional executed controller results or a substitute review verdict.

The sole writer was told to examine the ordinary concurrent-write identity
boundary and descriptor exception cleanup inside the existing checkpoint path.
If it applies a fix, that fix needs its own ordinary regression evidence and
fixed review. No new adversarial artifact or restriction bypass is requested;
the complete core review remains incomplete.
