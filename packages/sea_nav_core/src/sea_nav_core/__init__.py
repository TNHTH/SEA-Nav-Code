# SPDX-License-Identifier: MIT
"""Cross-platform method adaptation, not an original-paper or safety certificate."""

from .contracts import (
    ADAPTATION_ID,
    DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE,
    DASHGO_PLATFORM_PROVENANCE,
    DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE,
    DASHGO_SIM_CAMERA_RANGE_PROVENANCE,
    ISAACLAB_CAMERA_DISTANCE_TO_CAMERA,
    ISAACLAB_CAMERA_DISTANCE_TO_IMAGE_PLANE,
    ROS_LASERSCAN_RADIAL_RANGE,
    RESULT_CLASSIFICATION,
    SAFETY_SEMANTICS,
    ActionSpec,
    DifferentialDrivePlatformSpec,
    EffectiveCommandEnvelopeSpec,
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

__version__ = "0.3.0"
__all__ = [
    "ADAPTATION_ID", "DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE",
    "DASHGO_PLATFORM_PROVENANCE",
    "DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE",
    "DASHGO_SIM_CAMERA_RANGE_PROVENANCE",
    "ISAACLAB_CAMERA_DISTANCE_TO_CAMERA",
    "ISAACLAB_CAMERA_DISTANCE_TO_IMAGE_PLANE", "ROS_LASERSCAN_RADIAL_RANGE",
    "RESULT_CLASSIFICATION",
    "SAFETY_SEMANTICS", "ActionSpec", "DifferentialDrivePlatformSpec",
    "EffectiveCommandEnvelopeSpec",
    "ObservationSpec", "RawSafetyObservationSpec", "AblationProfile",
    "resolve_ablation_profile", "UnicycleLookaheadLSECBFLayer",
    "PAPER_V1_ALPHA_MIN", "PAPER_V1_LAMBDA_PI", "PAPER_V1_LAMBDA_REG",
    "PAPER_V1_LAMBDA_SHIELD", "PAPER_V1_LAMBDA_V",
    "paper_v1_lreg_loss", "paper_v1_shield_loss",
]
