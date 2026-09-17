# DashGo SEA adaptation (Milestone A)

| Field | Value |
| --- | --- |
| candidate_head | `100f579115f06baac94d4a92da6cd2405f7c2f18` |
| candidate_tree | `fe1cbe0c7914b6e6804a8ab629c19770dad7eede` |
| branch | `issue/1-sea-dashgo-milestone-a` |
| implementation | `CPU_STATIC_IMPLEMENTATION_CANDIDATE` |
| TRAIN_READY | `False` (hard reject) |
| models | `[]` |
| generated_utc | 2026-09-17T02:06:36Z |

## Layers (do not confuse)

1. **CPU/static** — unit tests, CLI `--help/--describe/--validate-only`, static API map.
2. **Git delivery** — merge to SEA `test` via governance `merge_to_test` (not direct push).
3. **Isaac runtime** — construct/reset/step/close on target stack — **NOT RUN**.
4. **Training / 4060 smoke / formal** — **NOT RUN** (`TRAIN_READY=False`).
5. **Real robot** — **NOT RUN**.

## CLI entrypoints

Under `tools/`: `sea_train_dashgo`, `sea_play_dashgo`, `sea_smoke_dashgo`,
`sea_evaluate_dashgo`, `sea_summarize_dashgo`, `sea_export_dashgo`,
`sea_validate_dashgo`, `sea_run_supervisor`, `sea_runtime_preflight`.

All support CPU `--help` / `--describe` / `--validate-only` without launching Isaac.
`sea_train_dashgo` real execution exits 3 while `TRAIN_READY` is false.

## Ablation profiles

Only four: `full`, `without_acsi`, `without_shield`, `without_lreg`.
Canonical flag order: ACSI / shield / Lreg. Network, reward, budget, maps, and
eval fixtures are shared; do not invent a fifth method.
