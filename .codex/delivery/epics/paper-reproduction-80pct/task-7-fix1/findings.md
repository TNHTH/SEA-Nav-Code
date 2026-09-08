# Current truth and findings

- Fixed starting source: `92ab65d23590256f3165d8cad6e64698cf11bfe8` in detached SEA-Nav-Code-batch7-v2.
- Initially exactly two unstaged root coordination registrations; no source/index changes. Repository attribution matches explicit TNHTH noreply identity.
- Independent reviewer confirmed strict load_state_dict can partially copy before throwing; both live runner and model-only consumers call it without rollback.
- Existing Adam schema checks moments only against one another, not actual parameters; missing betas, invalid amsgrad and wrong moment shapes can pass load and break the next update.
- ray_unit_vectors is a config-derived registered CBF buffer, yet current load permits overriding its values under an unchanged resolved-config receipt.
- Both trainers report stat(final_manifest) as checkpoint_bytes, rather than returned manifest.byte_size.
- Fresh RED evidence: 34 failures on fixed92ab65d (2.47s); 33 were actual defect/invariance failures, one was an overly broad ACSI AST test selector subsequently fixed. Independent reviewer also established both byte-stat defects. Fresh focused GREEN:126 passed20.27s; final whole suite474 passed54.33s (original432 plus42 new cases).
- Exact model dtype/layout checking must occur before native strict load; optimizer validation must use actual parameter IDs-to-group-position mapping. A disposable zero-gradient Adam step checks target execution without advancing live state or RNG.
- New current candidate is the functional fix commit containing this record, based on92ab65d. Five production files plus one new regression file; no existing converter-absence test or CBF/PPO/replay mathematical source changed.
- Failure-atomicity is verified for malformed artifacts and injected late registered-model/Adam load-hook exceptions; it restores model tensors, optimizer state/groups/defaults, and preserves runner iteration/algorithm LR. This does not claim recovery from process termination, hardware loss, arbitrary external hook side effects or a restored physical trajectory.
- Final precommit Gate A at `/tmp/sea-nav-task7-fix1.7JI9rk/gate-a-precommit.json`: five CPU/static passes, four explicit dependency/runtime blockers; tracked-syntax count89 excludes the new unstaged test, which the full suite executed. Postcommit Gate A will count its committed source.
- Remaining: independent fix review; Isaac Gym/Lab prerequisites and runtime, formal metrics, hardware and redistribution-rights validation remain blocked/deferred. Temporary Gate A reports are evidence receipts, not durable backups.
