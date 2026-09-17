# SPDX-License-Identifier: MIT
"""TorchScript export parity and tamper detection tests (A8.3)."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from export import (  # noqa: E402
    ExportMeanPolicy,
    export_torchscript,
    load_and_verify,
    parity_eager_vs_script,
)


def _example(batch=2):
    return (
        torch.randn(batch, 550),
        torch.full((batch, 41), 1.5),
        torch.ones(batch, 41, dtype=torch.bool),
        torch.full((batch, 1), 0.05),
        torch.zeros(batch, 2),
    )


def test_export_parity_and_tamper_detection():
    model = ExportMeanPolicy()
    model.eval()
    ex = _example(3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.pt"
        manifest = export_torchscript(model, ex, path)
        scripted = torch.jit.load(str(path), map_location="cpu")
        parity_eager_vs_script(model, scripted, ex)
        load_and_verify(path, manifest)
        # Tamper file
        path.write_bytes(path.read_bytes() + b"x")
        with pytest.raises(ValueError, match="tamper"):
            load_and_verify(path, manifest)


def test_bad_input_status():
    model = ExportMeanPolicy()
    ex = _example(1)
    ex_list = list(ex)
    ex_list[0] = torch.full((1, 550), float("nan"))
    cmd, status = model(*ex_list)
    assert status.item() == 128
    assert cmd.abs().max().item() == 0.0
