from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FootprintClearanceConfig:
    """Conservative center-ray clearance adjustment for finite robot footprint."""

    footprint_radius_m: float = 0.0
    min_effective_clearance_m: float = 1.0e-4

    @property
    def enabled(self) -> bool:
        return self.footprint_radius_m > 0.0


def apply_footprint_clearance(
    rays_m: torch.Tensor,
    config: FootprintClearanceConfig | None = None,
) -> torch.Tensor:
    config = config or FootprintClearanceConfig()
    if config.footprint_radius_m < 0.0:
        raise ValueError("footprint_radius_m must be non-negative")
    if config.min_effective_clearance_m <= 0.0:
        raise ValueError("min_effective_clearance_m must be positive")
    rays_m = rays_m.clamp_min(config.min_effective_clearance_m)
    if not config.enabled:
        return rays_m
    return (rays_m - float(config.footprint_radius_m)).clamp_min(config.min_effective_clearance_m)
