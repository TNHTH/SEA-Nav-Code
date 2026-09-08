# SPDX-License-Identifier: MIT
"""Cross-platform method adaptation, not an original-paper or safety certificate."""

from .contracts import (
    ADAPTATION_ID,
    DASHGO_PLATFORM_PROVENANCE,
    RESULT_CLASSIFICATION,
    SAFETY_SEMANTICS,
    ActionSpec,
    DifferentialDrivePlatformSpec,
    ObservationSpec,
    RawSafetyObservationSpec,
)
from .profiles import AblationProfile, resolve_ablation_profile
from .cbf import UnicycleLookaheadLSECBFLayer
from .losses import (
    PAPER_V1_ALPHA_MIN,
    PAPER_V1_LAMBDA_PI,
    PAPER_V1_LAMBDA_REG,
    PAPER_V1_LAMBDA_SHIELD,
    PAPER_V1_LAMBDA_V,
    paper_v1_lreg_loss,
    paper_v1_shield_loss,
)

__version__ = "0.2.0"
__all__ = [
    "ADAPTATION_ID", "DASHGO_PLATFORM_PROVENANCE", "RESULT_CLASSIFICATION",
    "SAFETY_SEMANTICS", "ActionSpec", "DifferentialDrivePlatformSpec",
    "ObservationSpec", "RawSafetyObservationSpec", "AblationProfile",
    "resolve_ablation_profile", "UnicycleLookaheadLSECBFLayer",
    "PAPER_V1_ALPHA_MIN", "PAPER_V1_LAMBDA_PI", "PAPER_V1_LAMBDA_REG",
    "PAPER_V1_LAMBDA_SHIELD", "PAPER_V1_LAMBDA_V",
    "paper_v1_lreg_loss", "paper_v1_shield_loss",
]
