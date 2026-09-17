# SPDX-License-Identifier: MIT
"""Thin adapter re-export of the four canonical SEA ablation profiles."""

from sea_nav_core.profiles import AblationProfile, resolve_ablation_profile

ABLATION_PROFILE_NAMES = ("full", "without_acsi", "without_shield", "without_lreg")

__all__ = [
    "ABLATION_PROFILE_NAMES",
    "AblationProfile",
    "resolve_ablation_profile",
]
