# SPDX-License-Identifier: MIT
"""CPU/static API inventory: CLI surfaces and source→consumer→test map (A9.1)."""

from __future__ import annotations

from typing import Any, Dict, List

from cli import TOOL_SPECS, describe

# Delivery inventory helper — models populated only after formal training.
MODELS: List[str] = []


FUNCTION_CONSUMER_TEST_MAP: List[Dict[str, Any]] = [
    {
        "symbol": "acsi.acsi_probability",
        "consumers": ["AcsiManager.draw", "env reset/replay"],
        "tests": ["tests/test_acsi.py"],
    },
    {
        "symbol": "acsi.update_l_goal_from_terminal_distance",
        "consumers": ["AcsiManager.apply_terminal_curriculum"],
        "tests": ["tests/test_acsi.py"],
    },
    {
        "symbol": "snapshot.SnapshotStore.reserve/restore/ack",
        "consumers": ["env replay transaction"],
        "tests": ["tests/test_snapshot.py"],
    },
    {
        "symbol": "rng.derive_stream_seed",
        "consumers": ["NamedRngBank"],
        "tests": ["tests/test_rng.py"],
    },
    {
        "symbol": "checkpoint.save_checkpoint/load_checkpoint/resume_from_checkpoint",
        "consumers": ["runner", "tools/sea_train_dashgo"],
        "tests": ["tests/test_checkpoint.py"],
    },
    {
        "symbol": "runner.SeaNavRunner",
        "consumers": ["tools/sea_train_dashgo"],
        "tests": ["tests/test_runner.py"],
    },
    {
        "symbol": "cli.main/describe/validate_only",
        "consumers": ["tools/sea_*_dashgo.py"],
        "tests": ["tests/test_cli.py"],
    },
    {
        "symbol": "supervisor.RunSupervisor",
        "consumers": ["tools/sea_run_supervisor.py"],
        "tests": ["tests/test_supervisor.py"],
    },
    {
        "symbol": "evaluation.EvalEpisodeState",
        "consumers": ["tools/sea_evaluate_dashgo"],
        "tests": ["tests/test_evaluation.py"],
    },
    {
        "symbol": "episode_schema.validate_record/validate_corpus",
        "consumers": ["tools/sea_evaluate_dashgo", "tools/sea_summarize_dashgo"],
        "tests": ["tests/test_episode_schema.py"],
    },
    {
        "symbol": "export.export_torchscript/parity_eager_vs_script",
        "consumers": ["tools/sea_export_dashgo"],
        "tests": ["tests/test_export.py"],
    },
    {
        "symbol": "statistics.aggregate_profile_seeds/hierarchical_paired_bootstrap_stub",
        "consumers": ["tools/sea_summarize_dashgo"],
        "tests": ["tests/test_statistics.py"],
    },
]


def inventory() -> Dict[str, Any]:
    return {
        "models": list(MODELS),
        "cli": describe(),
        "function_consumer_test_map": FUNCTION_CONSUMER_TEST_MAP,
    }


def list_checks() -> List[str]:
    return sorted({entry["symbol"] for entry in FUNCTION_CONSUMER_TEST_MAP})
