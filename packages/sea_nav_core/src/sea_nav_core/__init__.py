"""Cross-platform method adaptation, not an original-paper or safety certificate."""

from .contracts import (
    ADAPTATION_ID,
    RESULT_CLASSIFICATION,
    SAFETY_SEMANTICS,
    ActionSpec,
    DifferentialDrivePlatformSpec,
    ObservationSpec,
)
from .profiles import AblationProfile, resolve_ablation_profile
from .cbf import UnicycleLookaheadLSECBFLayer
from .losses import paper_v1_lreg_loss, paper_v1_shield_loss

__version__ = "0.1.0"
__all__ = [
    "ADAPTATION_ID", "RESULT_CLASSIFICATION", "SAFETY_SEMANTICS", "ActionSpec",
    "DifferentialDrivePlatformSpec", "ObservationSpec", "AblationProfile",
    "resolve_ablation_profile", "UnicycleLookaheadLSECBFLayer",
    "paper_v1_lreg_loss", "paper_v1_shield_loss",
]
