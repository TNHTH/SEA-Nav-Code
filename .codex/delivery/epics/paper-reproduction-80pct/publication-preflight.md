# Remote publication preflight

Checked at `2026-09-07T10:59:06+08:00`. This was a bounded, read-only remote inspection. No Git configuration, credentials, local or remote refs, index entries, protection settings, or source files were changed. The only write is this report.

## Current ref state

`git ls-remote` reports that `origin` still has exactly these branch heads:

| Remote branch | Object ID |
|---|---|
| `main` | `1c5675bbedf1dcbe5a4c1a91830cae528c780793` |
| `test` | `92896ba1b39087fc8cad633001a4d0dc3f313623` |
| `sea-nav-training-slow-steps0-15-20260602` | `b53d3feb98a287b5888a18e3dd67aeb530664fe0` |

Remote `stable` is absent. All four queried recovery tags are absent from `origin`:

- `archive/pre-recovery-main-20260904`
- `archive/pre-recovery-test-20260904`
- `archive/pre-recovery-slow-20260904`
- `upstream/11chens-fbce672c`

The corresponding local refs exist and are lightweight tags (their ref object type is `commit`):

| Local tag | Ref/commit object ID |
|---|---|
| `upstream/11chens-fbce672c` | `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96` |
| `archive/pre-recovery-main-20260904` | `1c5675bbedf1dcbe5a4c1a91830cae528c780793` |
| `archive/pre-recovery-test-20260904` | `92896ba1b39087fc8cad633001a4d0dc3f313623` |
| `archive/pre-recovery-slow-20260904` | `b53d3feb98a287b5888a18e3dd67aeb530664fe0` |

Local working branches are exactly `main`, `stable`, and `test`; `main` and `stable` remain at `1c5675bbedf1dcbe5a4c1a91830cae528c780793`. At inspection time the active local branch was `test@6f5e544f5d0a6e8a6e397e141a8e432157e88146` with no upstream. The six porcelain entries were confined to the coordination tree and were left untouched.

`origin` fetch/push remains HTTPS at `https://github.com/TNHTH/SEA-Nav-Code.git`. The authoritative `upstream` fetch URL remains `https://github.com/11chens/SEA-Nav-Code.git` and its push URL is `DISABLED`.

## Authentication, push, and hosting administration are separate

| Capability | Evidence | Ruling |
|---|---|---|
| SSH account authentication | A non-interactive, strict-host-key `ssh -T git@github.com` probe authenticated as `TNHTH`. No SSH URL is configured on `origin`. | **Available for account identity only.** It is not publication authorization. |
| Git receive-pack access | An explicit SSH `git push --dry-run` of the already-current `main` object to `refs/heads/main` exited 0 and reported `[up to date]`. No update command or ref mutation occurred. | **Only a no-op transport check.** It does not establish permission for tag creation, branch creation, forced replacement, deletion, or a protected-ref update. |
| Repository permission readback | GitHub CLI is not installed. The Codex browser session was unauthenticated and the repository settings URL was unavailable without login; no login was attempted. Anonymous repository metadata exposes no viewer permission object. | **Blocked with current tooling/authentication.** Push/admin role cannot be read back. |
| Ruleset/protection readback | Anonymous REST readback returned no repository rulesets and no evaluated rules for `main`, but the classic `main` branch-protection endpoint requires authentication. `stable` does not yet exist. | **Incomplete.** The empty public ruleset result does not prove classic branch protection is absent, and no `stable` protection can be read back. |
| Protection administration | SSH Git transport cannot configure hosting rules, and no authenticated API/browser administration path is available. | **Not demonstrated and currently unavailable.** |

No token, credential, or authorization-scope material was inspected or emitted.

## Blocking conditions and specification/capability conflicts

1. There is no frozen final candidate. `resume_state.json` records `candidate_oid: null`, only Tasks 1–2 complete, Phase `serial_implementation`, and highest verified Rung 1. The inspected `test@6f5e544` is explicitly not final-candidate evidence.
2. The design requires explicit publication authorization for each of the three `archive/*` tags before they are pushed. The coordination record contains a general statement that pushing the final reviewed result is authorized, but it does not record archive-tag publication authorization; it separately records `public_redistribution_rights` as blocked. Generic repository access or prior reachability of the objects is not substituted for this rights gate.
3. The design requires `stable` protection to be configured and read back immediately after creating the bootstrap branch. Current tooling can reach Git refs over SSH but cannot read repository permissions or administer/read back classic branch protection. Therefore the specified final transaction cannot presently be completed end to end. Creating `stable` before this capability is ready risks the design-defined partial state: unprotected and unqualified, with later unrelated cleanup requiring a fresh controller decision.
4. The design permits an explicit SSH URL or SSH push configuration only after authorization and destination-identity recheck. This preflight did not change `origin`; the controller must choose and authorize the exact transport for the final transaction.

## Safe prerequisites for the final remote transaction

Before the first remote write, the controller must have all of the following recorded and verified:

1. Completion/review of Batches 1–7 and a frozen, immutable `SEA_NAV_CANDIDATE_OID`; local `test` must equal it immediately before the `test` push.
2. Fresh Rungs 0–2 reports from a clean detached checkout of that exact object; proof that `main` is an ancestor, no complete-baseline path was deleted, and all required-tree paths exist.
3. A fail-closed sensitive-data scan of the committed tree at that exact object with the versioned pattern set, recording the candidate OID and scanned path set. Index/worktree scanning is supplementary only.
4. Explicit publication authorization for each of the three `archive/*` tags plus resolved rights sufficient for their publication. The `upstream/*` tag remains local unless its separate rights-holder condition is satisfied.
5. An exact destination/transport decision. If SSH is selected, authorize use of the exact `git@github.com:TNHTH/SEA-Nav-Code.git` destination (or a deliberate configuration change) and recheck that it authenticates as `TNHTH`. Transport access must not be treated as authorization for the content or operations.
6. An already-authenticated hosting API or browser principal whose repository role/permissions can be read back and that can configure and verify the intended `stable` protections. The exact required checks, review restrictions, direct-push restrictions, force-push/deletion prohibitions, and administrator-bypass expectation must be fixed before branch creation so the readback has a deterministic expected result.
7. A last-moment `ls-remote` lease check showing the exact three pre-transaction heads above and confirming that the archive tags and `stable` remain absent. Any change aborts the stale plan for controller review.

Only after those gates pass may the design's serialized transaction run: publish only the authorized archive tags and verify all three OIDs; create `stable` at `1c5675b...` as unqualified and configure/read back protection; force-update only `test` from the frozen object with expected-old `92896ba...`; verify the exact remote candidate and tree; delete only the long branch with expected-old `b53d3fe...`; then read back exactly `main`, `stable`, and `test` plus the three expected archive tags. `main` is not advanced. Any failed authorization, lease, validation, protection, or readback stops dependent steps; no remote write is currently authorized by this report.
