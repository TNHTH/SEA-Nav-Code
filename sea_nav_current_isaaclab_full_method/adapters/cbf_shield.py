"""Adapter boundaries for the shared damped CBF, with explicit alpha semantics."""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer

from .footprint_clearance import FootprintClearanceConfig


@dataclass(frozen=True)
class CBFShieldConfig:
    num_rays: int = 41
    fov_deg: float = 240.0
    safe_radius_m: float = 0.15
    safety_margin_m: float = 0.05
    kappa: float = 10.0
    damping_factor: float = 1.0
    gamma_min: float = 0.0
    footprint_radius_m: float = 0.0
    min_effective_clearance_m: float = 1.0e-4
    algorithm_profile: str = "upstream_fbce672c"

    @property
    def d_safe(self) -> float:
        return self.safe_radius_m + self.safety_margin_m

    @property
    def footprint(self) -> FootprintClearanceConfig:
        return FootprintClearanceConfig(self.footprint_radius_m, self.min_effective_clearance_m)


class FootprintAwareLSECBFLayer(ExactLSECBFLayer):
    """Positive-alpha layer; nonzero footprint is a declared ablation only."""
    def __init__(self, num_rays=41, fov_deg=240.0, safe_radius=0.15,
                 safety_margin=0.05, kappa=10.0, damping_factor=1.0,
                 footprint_radius_m=0.0, min_effective_clearance_m=1.0e-4,
                 algorithm_profile="upstream_fbce672c"):
        super().__init__(num_rays, fov_deg, safe_radius, safety_margin, kappa, damping_factor)
        if not math.isfinite(footprint_radius_m) or footprint_radius_m < 0:
            raise ValueError("footprint_radius_m must be finite and nonnegative")
        if not math.isfinite(min_effective_clearance_m) or min_effective_clearance_m <= 0:
            raise ValueError("min_effective_clearance_m must be finite and positive")
        if footprint_radius_m and (not algorithm_profile.startswith("ablation/") or not algorithm_profile[9:].strip()):
            raise ValueError("nonzero footprint requires a named ablation profile")
        # Scalar attributes keep the exported layer compatible with TorchScript.
        self.footprint_radius_m = float(footprint_radius_m)
        self.min_effective_clearance_m = float(min_effective_clearance_m)

    def _effective_rays(self, lidar_dists: torch.Tensor) -> torch.Tensor:
        if self.footprint_radius_m > 0:
            return (lidar_dists - self.footprint_radius_m).clamp_min(self.min_effective_clearance_m)
        return lidar_dists

    def forward(self, u_bar: torch.Tensor, lidar_dists: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        # Check raw rays before ablation clipping, on every ordinary actor call.
        self._validate_inputs(u_bar, lidar_dists, alpha)
        return self._compute_intermediates(u_bar, self._effective_rays(lidar_dists), alpha)[0]

    @torch.jit.export
    def forward_with_diagnostics(self, u_bar: torch.Tensor, lidar_dists: torch.Tensor,
                                 alpha: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_inputs(u_bar, lidar_dists, alpha)
        effective = self._effective_rays(lidar_dists)
        output, diag = self._compute(u_bar, effective, alpha)
        diag["ray_min_raw"] = lidar_dists.min(dim=1, keepdim=True).values
        diag["ray_min_effective"] = effective.min(dim=1, keepdim=True).values
        return output, diag


class ExactLSECBFShield:
    """Compatibility wrapper: apply(..., gamma) accepts RAW alpha logits.

    Use apply_alpha for already-positive actor alpha. gamma_min is retained as
    inactive metadata for old callers; it does not clamp or shift alpha.
    """
    def __init__(self, config: Optional[CBFShieldConfig] = None,
                 device: Optional[Union[torch.device, str]] = None):
        self.config = config or CBFShieldConfig()
        self.device = torch.device(device or "cpu")
        c = self.config
        self.layer = FootprintAwareLSECBFLayer(
            num_rays=c.num_rays, fov_deg=c.fov_deg, safe_radius=c.safe_radius_m,
            safety_margin=c.safety_margin_m, kappa=c.kappa, damping_factor=c.damping_factor,
            footprint_radius_m=c.footprint_radius_m, min_effective_clearance_m=c.min_effective_clearance_m,
            algorithm_profile=c.algorithm_profile).to(self.device)
        self.ray_unit_vectors = self.layer.ray_unit_vectors
        self.ray_angles = torch.linspace(-math.radians(c.fov_deg) / 2, math.radians(c.fov_deg) / 2,
                                         c.num_rays, device=self.device)

    def apply(self, u_nominal: torch.Tensor, rays_m: torch.Tensor,
              gamma: Union[torch.Tensor, float]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """gamma is the historical name of alpha_raw, never positive alpha."""
        alpha_raw = torch.as_tensor(gamma, dtype=u_nominal.dtype, device=self.device)
        if alpha_raw.ndim == 0:
            alpha_raw = alpha_raw.expand(u_nominal.shape[0], 1)
        elif alpha_raw.ndim == 1:
            alpha_raw = alpha_raw.unsqueeze(-1)
        if not torch.isfinite(alpha_raw).all():
            raise ValueError("alpha_raw must be finite")
        return self.apply_alpha(u_nominal, rays_m, F.softplus(alpha_raw))

    def apply_alpha(self, u_nominal: torch.Tensor, rays_m: torch.Tensor,
                    alpha: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        u_nominal, rays_m, alpha = u_nominal.to(self.device), rays_m.to(self.device), alpha.to(self.device)
        output, diag = self.layer.forward_with_diagnostics(u_nominal, rays_m, alpha)
        # Preserve historical debug aliases; canonical Eq.4 fields stay differentiable.
        h_min = diag["ray_min_effective"] - self.config.d_safe
        diag.update({
            "gamma": alpha.detach(),
            "gamma_min_compat_inactive": torch.full_like(h_min, self.config.gamma_min),
            "h_min": h_min.detach(), "lse_h": diag["h_comp"].detach(),
            "shield_delta": diag["correction_norm"].detach(),
            "footprint_radius_m": torch.full_like(h_min, self.config.footprint_radius_m),
        })
        return output, diag


def clip_body_command(command: torch.Tensor) -> torch.Tensor:
    lows = torch.tensor([-0.5, -1.0, -1.0], dtype=command.dtype, device=command.device)
    highs = torch.tensor([2.0, 1.0, 1.0], dtype=command.dtype, device=command.device)
    return torch.minimum(torch.maximum(command, lows), highs)
