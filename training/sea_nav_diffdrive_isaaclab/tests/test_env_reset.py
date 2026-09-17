# SPDX-License-Identifier: MIT
"""CPU tests for terminal capture vs reset isolation (A3.4)."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PACKAGE_ROOT / "env.py"
SOURCE = ENV_FILE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _method(name):
    cls = next(node for node in TREE.body
               if isinstance(node, ast.ClassDef) and node.name == "SeaNavDiffDriveEnv")
    return next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def test_terminal_payload_includes_pre_reset_policy_obs_field():
    capture = ast.unparse(_method("_capture_terminal_payload"))
    assert "terminal_policy_obs" in capture
    assert "terminal_root_state" in capture
    assert "terminal_executed_command" in capture


def test_reset_idx_clears_only_requested_rows_via_index_ops():
    text = ast.unparse(_method("_reset_idx"))
    assert "clear_rows" in text
    assert "index_fill_" in text


def test_close_does_not_blanket_swallow_without_idempotent_guard():
    text = ast.unparse(_method("close"))
    assert "if self._closed" in text
    # May catch teardown errors after marking closed, but must set _closed first.
    assert text.index("self._closed = True") < text.index("super().close")
