"""Shared scalar golden vectors catch sign, FOV, damping and alpha drift."""
import json
from pathlib import Path

import pytest
import torch

from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer

FIXTURE = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/cbf_paper_damped_v1.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["id"])
def test_paper_damped_shared_golden_and_gradients(case):
    layer = ExactLSECBFLayer(fov_deg=case["fov_deg"], **FIXTURE["parameters"])
    u, rays, alpha = [torch.tensor([value], dtype=torch.float32, requires_grad=True) for value in
                      (case["u_bar"], case["rays_m"], [case["alpha"]])]
    output, diag = layer.forward_with_diagnostics(u, rays, alpha)
    for key, want in case["expected"].items():
        actual = output if key == "u_s" else diag[key]
        torch.testing.assert_close(actual, torch.tensor([want], dtype=torch.float32), atol=FIXTURE["atol"], rtol=FIXTURE["rtol"])
    torch.testing.assert_close(layer(u, rays, alpha), output, atol=0, rtol=0)
    assert torch.equal(output[:, 2], u[:, 2])
    assert (diag["eta"] >= 0).all()
    expected_residual = torch.where(diag["r"] < 0, -diag["eta"] * layer.damping_factor, diag["r"])
    torch.testing.assert_close(diag["residual_after"], expected_residual)
    output.square().sum().backward()
    for tensor in (u, rays, alpha):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("fov,edge", [(180.0, [0., -1.]), (240.0, [-.5, -.8660254])])
def test_geometry_and_real_torchscript(fov, edge):
    layer = ExactLSECBFLayer(fov_deg=fov)
    torch.testing.assert_close(layer.ray_unit_vectors[0], torch.tensor(edge), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(layer.ray_unit_vectors[20], torch.tensor([1., 0.]), atol=1e-6, rtol=1e-6)
    scripted = torch.jit.script(layer)
    args = (torch.zeros(2, 3), torch.full((2, 41), .1), torch.ones(2, 1))
    torch.testing.assert_close(scripted(*args), layer(*args))
    torch.testing.assert_close(scripted.forward_with_diagnostics(*args)[1]["eta"], layer.forward_with_diagnostics(*args)[1]["eta"])


@pytest.mark.parametrize("kwargs", [
    {"num_rays": 0}, {"num_rays": 1.5}, {"num_rays": True},
    {"fov_deg": 0}, {"fov_deg": 361}, {"fov_deg": float("nan")},
    {"kappa": 0}, {"kappa": -1}, {"kappa": float("inf")},
    {"damping_factor": 0}, {"damping_factor": -1}, {"safe_radius": -1},
    {"safety_margin": -1},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ExactLSECBFLayer(**kwargs)


@pytest.mark.parametrize("which,value", [
    (0, torch.zeros(2, 2)), (0, torch.zeros(3)),
    (1, torch.ones(2, 40)), (1, torch.ones(1, 41)),
    (2, torch.ones(2)), (2, torch.ones(1, 1)),
    (0, torch.full((2, 3), float("nan"))),
    (1, torch.full((2, 41), float("inf"))),
    (1, torch.zeros(2, 41)), (1, -torch.ones(2, 41)),
    (2, torch.full((2, 1), float("nan"))), (2, -torch.ones(2, 1)),
    (2, torch.zeros(2, 1)),
])
def test_invalid_tensor_boundary_rejected(which, value):
    args = [torch.zeros(2, 3), torch.ones(2, 41), torch.ones(2, 1)]
    args[which] = value
    with pytest.raises(ValueError):
        ExactLSECBFLayer()(*args)
