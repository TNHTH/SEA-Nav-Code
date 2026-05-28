# SEA-Nav Current IsaacLab Full-Method Adapter

This directory holds the tracked current-IsaacLab adapter for the SEA-Nav original-method reproduction route.

It is intentionally separate from upstream `training/**` so the cloned baseline at commit `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96` remains readable as the source reference. First-wave work covers Gate A static contracts only:

- observation/history shape;
- 41 ray angle/range contract;
- LSE-CBF shield behavior with upstream `softplus(alpha_raw)` semantics and no additive alpha floor;
- per-env ACSI-style collision replay state sampling;
- source_alpha_only alpha filter with no queue delay;
- upstream terminal defaults (`stay_time=150`, contact termination enabled);
- trace schema;
- formal eval manifest with replay disabled.

Gate B/Gate C training and evaluation must not start until `tests/gate_a_static_contract.py` passes.
