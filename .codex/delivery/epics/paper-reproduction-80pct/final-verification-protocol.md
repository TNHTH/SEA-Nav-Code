# Frozen-candidate evidence protocol

Status: execution checklist, **not validation evidence**. No final candidate exists while Tasks 5–7 and whole-branch review remain incomplete.

## Distinct identities

1. **Implementation candidate A:** all serial source batches and required review fixes are integrated. Run the full CPU/static suite and preserve raw results with A's full OID.
2. **Evidence commit E:** commit the A-bound results and coordination metadata by exact paths, explicitly identifying them as results of A. Do not relabel them E. Confirm any A→E change is confined to the exact evidence/coordination paths; a source/config/test change restarts implementation verification.
3. **Frozen publication candidate E:** freeze the full E OID, create a fresh detached checkout at E, and perform the definitive complete run there. E's own result cannot be embedded into the same commit with E's hash without a self-reference cycle. Save its raw result outside the checkout in a durable, E-named directory in the task workspace, and report that local artifact separately from the A-bound evidence present in Git.

The fresh complete E run is still mandatory under the design: executable-tree equivalence does not replace it. Any subsequent commit changes the frozen publication candidate and requires a new clean-checkout validation. Do not enter an endless loop of committing each candidate's own hash; keep the definitive external E-bound report immutable. A report's local presence, committed presence and remote visibility are different facts.

## Definitive checkout and output hygiene

- Use a new detached worktree under the authorized workspace work/ area; no new branch name. Record full E OID and worktree inventory.
- Use the existing dedicated CPU Python via an absolute path when the worktree's parent differs. Do not install/upgrade dependencies during this validation.
- Store reports outside the verification checkout. A durable task-workspace directory is required; /tmp alone is not delivery.
- Compare complete before/after `git status --porcelain --untracked-files=all --ignored` output. The initial inventory must be empty. Run pytest with `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`; Gate A uses tracked-source `compile()`. Do not use compileall or py_compile for the no-output claim.
- Full selection: `tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`. Do not collect simulator-dependent upstream tests by using an unqualified repository-wide pytest.
- Capture full commands, stdout/stderr, return codes, versions, exact OID, baseline ancestry, no-deletion/tree inventories, and independent Gate A dependency/runtime statuses.
- Record Isaac Gym, IsaacLab, simulator optimizer resume, formal metrics, rights and hardware blockers precisely. Neither installed packages nor CPU probes establish simulation execution.

## Frozen committed-tree privacy scan

- Before the scan, verify the working `secret-patterns.regex` blob equals the blob at E. Read the full committed path inventory from E.
- Run the existing file-name-only Git grep scan against E, not HEAD's moving name, the index, or working files. Preserve candidate OID, pattern blob OID, inventory and exact exit code.
- Exit 1 means no text-pattern match; exit 0 requires local assessment without displaying matched contents; other exits block publication.
- Include any resolution of benign fixture matches in the candidate-bound record. Do not claim binary safety, an entire-history audit or completed supply-chain/license clearance.

## Publication is a separate gate

Remote authorization, exact archive objects, full expected-old branch leases, SSH identity and stable-protection decision must still be satisfied. No part of this local checklist authorizes a remote change. If those gates remain unresolved, retain the completed local result and report the exact blocked operation, rather than advancing main/stable or silently dropping protection.

Read back any authorized remote E ref and exact committed artifact paths separately. A synchronized branch does not prove an external final-E report was uploaded. Never claim an external-only report is in the remote tree.
