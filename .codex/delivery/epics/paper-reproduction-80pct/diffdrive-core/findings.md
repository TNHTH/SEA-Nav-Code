# Findings and current truth

- Current artifact: final package candidate delivered by the functional commit containing this record, based on detached BASE `399ce2b08eac40865fd6496d19324f73a3e6cc7e`; original tree was clean before registration. Resolve its fixed OID with Git, not an embedded self-referential hash. No Go2/rsl_rl/Task 7 changes.
- CPU environment: sibling `sea-nav-cpu-venv`, Python 3.10.12, Torch 2.6.0+cpu, pytest 8.4.2.
- Existing SEA actor and CBF require [B,3] body commands and fixed Go2 observation layout; they remain unchanged.
- New package must not use the `rsl_rl` import/distribution name; DashGo has its own RSL-RL implementation.
- New CBF works at lookahead point [l,0], with q=[v,l*omega], metric obstacle points transformed from LiDAR to base, and conservative footprint+l+margin envelope.
- State/controls are instantaneous geometry, not a dynamics or hard-safety certificate. Damping leaves an active residual negative.
- Paper-v1 source resolution verified: shield=.1*(intervention+mean(relu(.1-alpha)^2)); Lreg=range+.05*actor_MSE+.005*critic_MSE, lambda_reg=1. No alpha penalty coefficient of one is imported from upstream.
- Highest validation rung: CPU plus real Python packaging. Final package155 cases (36 functions with parametrization); combined SEA+package497 passed36.28s; wheel-isolated153 focused cases1.32s; extracted sdist full155 cases2.35s. Initial152-case result is historical and superseded.
- All-invalid rows are rejected; masked invalid range/angle placeholders are sanitized before geometry so pruned-ray outputs and finite masked gradients agree.
- Active constraint and actual nonzero correction are distinct: symmetric zero composite gradient can have eta>0 with identical output and still-negative residual. Both flags are explicit and tested.
- Final packaging artifacts are temporary receipts only at `/tmp/sea-nav-core-packaging.MeXwyx/final-dist/`; they are not committed or long-term backups. sdist SHA256 `3910737eaec9a399556df2ffe7d6a65e64ccb83fbec275108f456e8b4d3d8546`; wheel SHA256 `8c46d6622ba22be8e32053d61b42e6d89635d94821ce3362d63715a47dcb044b`.
- Remaining: independent fixed-commit review/integration by parent; differential-drive consumer integration and all simulator/hardware/performance/evaluation evidence remain unverified. No hard safety or original-paper claim.
