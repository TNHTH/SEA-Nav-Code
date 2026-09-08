# CBF hot-path findings

## Current truth table

| Field | Current value | Evidence / boundary |
|---|---|---|
| Artifact identity | Uncommitted six-path hotpath candidate on detached BASE a453b15c7b4e84f97bc37308f682fdde79e6fbbc | Git verified; primary test untouched except separately authorized fix2 review receipt |
| Active evidence | Final precommit full CPU/static 579 passed / 2 CUDA skips in 56.65s; Gate A five passed / four blocked | Includes final ordinary and explicit diagnostic operator-budget assertions; commit-bound verification follows |
| Declared/applied configuration | Dynamic checked core and adapter APIs; Torch 2.6 CPU interpreter | No simulator, CUDA timing or legacy Torch claim |
| Confirmed source facts | Ordinary forward: one scalar extraction, three mathematical sums, zero diagnostic norm/minimum; real PPO 2 collection and 8 update calls | Fresh actual operator hooks; invalid aggregation separately costs 2/5/6 extractions for nonfinite-u/zero-rays/zero-alpha |
| Candidate direction | Implemented closed aggregate eligibility for historical real numeric types; other layouts/dtypes/meta/mixed devices go directly to unchanged ordered value checks | Float8 counterexample regression added and repaired, no new public dtype ban |
| Highest validation | Fresh full Torch 2.6 CPU/static, real script/save/load/export, non-final precommit Gate A | 63 actual-BASE output/diagnostic/gradient comparisons bit-exact; no timing, legacy Torch or simulator acceptance |
| Unpublished work | Two production edits, one new focused test, three ledgers | Root registration remains uncommitted; no branch/ref/push action |

Historical BASE adapter inherited ordinary forward, which dispatched to its diagnostic override. The implemented candidate instead has an explicit adapter ordinary override and shared effective-ray preprocessing; both public methods validate raw rays first. Zero-footprint retains positive subminimum ranges unchanged. The shared mathematical-intermediate tuple contains only quantities required to compute the command; diagnostic reductions are built only by the diagnostic endpoint.

The genuine Gym exporter can be extracted from its actual AST definition and executed with Torch/copy/os dependencies without importing proprietary Gym. Its script/save output, not a copied substitute wrapper, is the export test boundary. Existing committed golden fixtures and PPO state-identity tests remain unchanged.

Fresh dtype extension: Float8_e4m3fn is real, dense, same-device and nonquantized, yet Torch 2.6 CPU isfinite is unsupported. Therefore structural metadata checks alone are insufficient to preserve earlier-input error precedence. The accelerated dtype set must be closed over supported historical real dtypes, with all remaining dtypes delegated unchanged to the legacy validator. This is eligibility for aggregation, not a new public dtype ban. Optional Float8 and unsigned-wide dtype tests are conditional on availability and do not import new dtype names into production.
