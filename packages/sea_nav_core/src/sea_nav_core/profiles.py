# SPDX-License-Identifier: MIT
"""The four paper-style leave-one-component-out profiles, not a 2x2 grid."""

from dataclasses import dataclass
from typing import ClassVar

from .contracts import _ManifestSpec

_NAMES = ("full", "without_acsi", "without_shield", "without_lreg")


@dataclass(frozen=True)
class AblationProfile(_ManifestSpec):
    name: str
    shield_loss_weight_owner: str = "sea_nav_core.paper_v1_shield_loss"
    lreg_loss_weight_owner: str = "sea_nav_core.paper_v1_lreg_loss"
    kind: ClassVar[str] = "ablation_profile_v1"

    def __post_init__(self):
        if type(self.name) is not str or self.name not in _NAMES:
            raise ValueError("profile must be full/without_acsi/without_shield/without_lreg")
        if self.shield_loss_weight_owner != "sea_nav_core.paper_v1_shield_loss":
            raise ValueError("lambda_shield is owned only by paper_v1_shield_loss")
        if self.lreg_loss_weight_owner != "sea_nav_core.paper_v1_lreg_loss":
            raise ValueError("Lreg coefficients are owned only by paper_v1_lreg_loss")

    @property
    def acsi_enabled(self):
        return self.name != "without_acsi"

    @property
    def shield_enabled(self):
        return self.name != "without_shield"

    @property
    def lreg_enabled(self):
        return self.name != "without_lreg"

    @property
    def alpha_min(self):
        return 0.1


def resolve_ablation_profile(name: str) -> AblationProfile:
    return AblationProfile(name)
