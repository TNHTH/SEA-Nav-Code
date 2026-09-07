"""Adapter uses the same versioned numerical contract as the training layer."""
import json
from pathlib import Path

import pytest
import torch

from sea_nav_current_isaaclab_full_method.adapters.cbf_shield import (
    CBFShieldConfig, ExactLSECBFShield, FootprintAwareLSECBFLayer)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cbf_paper_damped_v1.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["id"])
def test_adapter_shared_golden_raw_and_positive_alpha(case):
    u, rays, alpha = [torch.tensor([x], dtype=torch.float32) for x in (case["u_bar"], case["rays_m"], [case["alpha"]])]
    wrapper = ExactLSECBFShield(CBFShieldConfig(fov_deg=case["fov_deg"]))
    layer = FootprintAwareLSECBFLayer(fov_deg=case["fov_deg"])
    results = [wrapper.apply(u, rays, gamma=torch.tensor([[case["alpha_raw"]]])),
               wrapper.apply_alpha(u, rays, alpha), layer.forward_with_diagnostics(u, rays, alpha)]
    for output, diag in results:
        for key, want in case["expected"].items():
            torch.testing.assert_close(output if key == "u_s" else diag[key],
                                       torch.tensor([want], dtype=torch.float32), atol=FIXTURE["atol"], rtol=FIXTURE["rtol"])


def test_footprint_is_explicit_ablation_and_scriptable():
    with pytest.raises(ValueError, match="ablation"):
        FootprintAwareLSECBFLayer(footprint_radius_m=.55)
    with pytest.raises(ValueError, match="ablation"):
        ExactLSECBFShield(CBFShieldConfig(footprint_radius_m=.55))
    layer = FootprintAwareLSECBFLayer(footprint_radius_m=.55, min_effective_clearance_m=.01,
                                    algorithm_profile="ablation/footprint")
    scripted = torch.jit.script(layer)
    u, rays, alpha = torch.zeros(1, 3), torch.full((1, 41), .1), torch.ones(1, 1)
    output, diag = scripted.forward_with_diagnostics(u, rays, alpha)
    torch.testing.assert_close(scripted(u, rays, alpha), output)
    torch.testing.assert_close(diag["ray_min_effective"], torch.tensor([[.01]]))
    with pytest.raises(ValueError):
        layer(u, torch.zeros_like(rays), alpha)


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf")])
def test_wrapper_rejects_invalid_rays_before_clipping(bad):
    wrapper = ExactLSECBFShield()
    with pytest.raises(ValueError):
        wrapper.apply(torch.zeros(1, 3), torch.full((1, 41), bad), gamma=1.)


def test_zero_footprint_preserves_positive_subminimum_rays():
    layer = FootprintAwareLSECBFLayer(min_effective_clearance_m=.01)
    _, diag = layer.forward_with_diagnostics(torch.zeros(1, 3), torch.full((1, 41), .001), torch.ones(1, 1))
    torch.testing.assert_close(diag["ray_min_effective"], torch.tensor([[.001]]))
