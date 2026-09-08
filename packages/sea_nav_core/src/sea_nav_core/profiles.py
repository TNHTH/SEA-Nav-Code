"""The four paper-style leave-one-component-out profiles, not a 2x2 grid."""

from dataclasses import dataclass
from typing import ClassVar

from .contracts import _ManifestSpec

_NAMES = ("full", "without_acsi", "without_shield", "without_lreg")


@dataclass(frozen=True)
class AblationProfile(_ManifestSpec):
    name: str
    kind: ClassVar[str] = "ablation_profile_v1"

    def __post_init__(self):
        if type(self.name) is not str or self.name not in _NAMES:
            raise ValueError("profile must be full/without_acsi/without_shield/without_lreg")

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
    def lambda_shield(self):
        return 0.1 if self.shield_enabled else 0.0

    @property
    def alpha_min(self):
        return 0.1

    @property
    def lambda_reg(self):
        return 1.0 if self.lreg_enabled else 0.0

    @property
    def lambda_pi(self):
        return 0.05

    @property
    def lambda_v(self):
        return 0.005


def resolve_ablation_profile(name: str) -> AblationProfile:
    return AblationProfile(name)
