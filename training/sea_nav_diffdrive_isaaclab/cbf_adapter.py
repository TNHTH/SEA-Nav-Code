# SPDX-License-Identifier: MIT
"""CBF mean filtering for DashGo 2D navigation (distribution_mean stage only).

Pure CPU Torch; imports neither Isaac nor ROS.  Samples are never re-shielded.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from sea_nav_core import (
    EffectiveCommandEnvelopeSpec,
    RawSafetyObservationSpec,
    SeaNavActorInput,
    UnicycleLookaheadLSECBFLayer,
    raw_safety_row_status,
    sea_nav_dashgo_candidate_platform_spec,
    sea_nav_dashgo_raw_safety_spec,
    sea_ray_angles_rad,
)
from sea_nav_core.observation import POLICY_OBS_DIM


ActorContext = Union[SeaNavActorInput, Mapping[str, Tensor]]


@dataclass(frozen=True)
class MeanStages:
    """Body- and q-space mean stages retained for shield diagnostics."""

    nominal_body_twist: Tensor   # u_bar  [B, 2]
    distribution_mean: Tensor    # u_s    [B, 2]
    nominal_q: Tensor            # q_bar  [B, 2]
    biased_q: Tensor             # q_s    [B, 2]
    alpha: Tensor                # [B, 1]


def _as_actor_input(context: ActorContext) -> SeaNavActorInput:
    if isinstance(context, SeaNavActorInput):
        context.validate()
        return context
    required = ("safety_ranges_m", "safety_valid", "safety_age_s")
    missing = [key for key in required if key not in context]
    if missing:
        raise ValueError("actor_context missing keys: " + ", ".join(missing))
    if "policy_obs" not in context:
        raise ValueError("actor_context must include policy_obs")
    actor_input = SeaNavActorInput(
        policy_obs=context["policy_obs"],
        safety_ranges_m=context["safety_ranges_m"],
        safety_valid=context["safety_valid"],
        safety_age_s=context["safety_age_s"],
    )
    actor_input.validate()
    return actor_input


def _default_cbf_layer() -> tuple[UnicycleLookaheadLSECBFLayer, RawSafetyObservationSpec]:
    safety = sea_nav_dashgo_raw_safety_spec()
    envelope = EffectiveCommandEnvelopeSpec(
        profile_id="dashgo_d1_primitive_candidate_v1",
        min_linear_velocity_m_s=-0.15,
        max_linear_velocity_m_s=0.30,
        max_abs_yaw_rate_rad_s=1.0,
        envelope_provenance="cbf_adapter_default",
    )
    layer = UnicycleLookaheadLSECBFLayer(
        sea_nav_dashgo_candidate_platform_spec(), safety, envelope,
    )
    return layer, safety


class CBFMeanAdapter:
    """Filter nominal body means through the frozen unicycle LSE CBF layer."""

    def __init__(
        self,
        cbf_layer: Optional[UnicycleLookaheadLSECBFLayer] = None,
        safety_spec: Optional[RawSafetyObservationSpec] = None,
    ):
        if cbf_layer is None or safety_spec is None:
            default_layer, default_spec = _default_cbf_layer()
            cbf_layer = cbf_layer or default_layer
            safety_spec = safety_spec or default_spec
        if not isinstance(cbf_layer, UnicycleLookaheadLSECBFLayer):
            raise ValueError("cbf_layer must be UnicycleLookaheadLSECBFLayer")
        if not isinstance(safety_spec, RawSafetyObservationSpec):
            raise ValueError("safety_spec must be RawSafetyObservationSpec")
        if safety_spec.manifest_sha256 != cbf_layer._safety_manifest_sha256_value:
            raise ValueError("safety spec differs from the CBF layer binding")
        self._cbf = cbf_layer
        self._safety_spec = safety_spec
        self._ray_angles_rad = torch.tensor(sea_ray_angles_rad(), dtype=torch.float64)

    @property
    def safety_manifest_sha256(self) -> str:
        return self._safety_spec.manifest_sha256

    @property
    def lookahead_distance_m(self) -> float:
        return float(self._cbf.lookahead_distance_m)

    def _ray_angles(self, batch: int, device, dtype) -> Tensor:
        return self._ray_angles_rad.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)

    def _validate_real_context(self, context: ActorContext) -> SeaNavActorInput:
        actor_input = _as_actor_input(context)
        status = raw_safety_row_status(
            self._safety_spec,
            actor_input.safety_ranges_m,
            actor_input.safety_valid,
            actor_input.safety_age_s,
        )
        if not bool(status.ok.all()):
            raise ValueError(
                "real actor_context failed raw safety row status; "
                "use action_mean_for_aux for synthetic smoothness queries"
            )
        return actor_input

    def filter_mean(
        self,
        nominal_body_twist: Tensor,
        alpha: Tensor,
        context: ActorContext,
    ) -> MeanStages:
        """Apply CBF to the nominal mean only; never mutate exploration samples."""
        actor_input = self._validate_real_context(context)
        batch = nominal_body_twist.shape[0]
        if nominal_body_twist.shape != (batch, 2):
            raise ValueError("nominal_body_twist must be [B, 2]")
        if alpha.shape != (batch, 1):
            raise ValueError("alpha must be [B, 1]")
        distribution_mean, diagnostics = self._cbf.forward(
            nominal_body_twist,
            actor_input.safety_ranges_m,
            self._ray_angles(batch, nominal_body_twist.device, nominal_body_twist.dtype),
            actor_input.safety_valid,
            actor_input.safety_age_s,
            alpha,
            self._safety_spec.manifest_sha256,
        )
        return MeanStages(
            nominal_body_twist=diagnostics["nominal_body_twist"],
            distribution_mean=distribution_mean,
            nominal_q=diagnostics["nominal_q"],
            biased_q=diagnostics["biased_q"],
            alpha=alpha,
        )

    @staticmethod
    def interpolate_aux_context(
        context_current: ActorContext,
        context_next: ActorContext,
        beta: Tensor,
    ) -> SeaNavActorInput:
        """Geometric range interpolation for PPO smoothness aux queries only."""
        current = _as_actor_input(context_current)
        nxt = _as_actor_input(context_next)
        if current.policy_obs.shape != nxt.policy_obs.shape:
            raise ValueError("policy observations must match between aux endpoints")
        batch = current.policy_obs.shape[0]
        if beta.shape != (batch, 1):
            raise ValueError("beta must be [B, 1]")
        if current.policy_obs.shape[-1] != POLICY_OBS_DIM:
            raise ValueError("policy observation must be 550-D")

        policy_obs = current.policy_obs + beta * (nxt.policy_obs - current.policy_obs)

        # Invalid rays keep finite placeholders; CBF masks them out.
        placeholder = torch.full_like(current.safety_ranges_m, 1.0)
        ranges_current = torch.where(
            current.safety_valid, current.safety_ranges_m, placeholder,
        )
        ranges_next = torch.where(
            nxt.safety_valid, nxt.safety_ranges_m, placeholder,
        )
        log_current = torch.log(ranges_current.clamp(min=1e-6))
        log_next = torch.log(ranges_next.clamp(min=1e-6))
        ranges_interp = torch.exp(log_current + beta * (log_next - log_current))
        valid_interp = current.safety_valid & nxt.safety_valid
        age_interp = torch.maximum(current.safety_age_s, nxt.safety_age_s)

        return SeaNavActorInput(
            policy_obs=policy_obs,
            safety_ranges_m=ranges_interp,
            safety_valid=valid_interp,
            safety_age_s=age_interp,
        )

    def action_mean_for_aux(
        self,
        nominal_body_twist: Tensor,
        alpha: Tensor,
        context_current: ActorContext,
        context_next: ActorContext,
        beta: Tensor,
    ) -> Tensor:
        """Pure aux mean query; synthetic ranges may leave [0.1, 3.0] by design."""
        interp = self.interpolate_aux_context(context_current, context_next, beta)
        if not bool(interp.safety_valid.any(dim=1).any()):
            raise ValueError("aux pair has empty valid-ray intersection for every row")
        batch = nominal_body_twist.shape[0]
        distribution_mean, _ = self._cbf.forward(
            nominal_body_twist,
            interp.safety_ranges_m,
            self._ray_angles(batch, nominal_body_twist.device, nominal_body_twist.dtype),
            interp.safety_valid,
            interp.safety_age_s,
            alpha,
            self._safety_spec.manifest_sha256,
        )
        return distribution_mean


__all__ = [
    "ActorContext",
    "CBFMeanAdapter",
    "MeanStages",
]
