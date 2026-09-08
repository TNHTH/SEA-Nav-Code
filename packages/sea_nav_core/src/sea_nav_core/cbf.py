"""Unicycle lookahead adaptation of paper Eq. 4, with damping fixed at one.

This is an instantaneous static-obstacle safety bias, not a constrained QP,
post-sampling guard, wheel/acceleration controller or hard-safety guarantee.
"""

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from .contracts import DifferentialDrivePlatformSpec, _number


class UnicycleLookaheadLSECBFLayer(nn.Module):
    """Map [v,omega] through q=[v,l*omega] at the base-frame point [l,0].

    Explicit ray angles are measured in the sensor frame, radians, CCW from
    sensor +x. Extrinsics map that frame into the unicycle reference frame.
    Invalid rays may carry nonfinite placeholders only when masked out; every
    batch row must contain at least one valid positive finite measurement.
    """

    def __init__(self, platform: DifferentialDrivePlatformSpec, kappa: float = 10.0,
                 damping_factor: float = 1.0):
        super().__init__()
        if not isinstance(platform, DifferentialDrivePlatformSpec):
            raise ValueError("platform must be DifferentialDrivePlatformSpec")
        _number("kappa", kappa, strictly_positive=True)
        _number("damping_factor", damping_factor, strictly_positive=True)
        if damping_factor != 1.0:
            raise ValueError("this adaptation fixes paper Eq.4 damping_factor=1.0")
        self.lookahead_distance_m = float(platform.lookahead_distance_m)
        self.envelope_radius_m = float(platform.footprint_radius_m + platform.lookahead_distance_m
                                       + platform.safety_margin_m)
        if not math.isfinite(self.envelope_radius_m):
            raise ValueError("footprint/lookahead/margin envelope must remain finite")
        self.sensor_x_m = float(platform.sensor_x_m)
        self.sensor_y_m = float(platform.sensor_y_m)
        self.sensor_yaw_rad = float(platform.sensor_yaw_rad)
        self.kappa = float(kappa)
        self.damping_factor = 1.0

    def _validate(self, nominal_twist: Tensor, ranges_m: Tensor, ray_angles_rad: Tensor,
                  valid_mask: Tensor, positive_alpha: Tensor) -> None:
        if (nominal_twist.dim() != 2 or nominal_twist.size(0) == 0
                or nominal_twist.size(1) != 2):
            raise ValueError("nominal_twist must be nonempty [B,2]")
        if (ranges_m.dim() != 2 or ranges_m.size(0) != nominal_twist.size(0)
                or ranges_m.size(1) == 0):
            raise ValueError("ranges_m must be [B,N], N>0")
        if valid_mask.shape != ranges_m.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,N]")
        if (positive_alpha.dim() != 2 or positive_alpha.size(0) != nominal_twist.size(0)
                or positive_alpha.size(1) != 1):
            raise ValueError("positive_alpha must be [B,1]")
        if ray_angles_rad.dim() == 1:
            if ray_angles_rad.size(0) != ranges_m.size(1):
                raise ValueError("ray_angles_rad must be [N] or [B,N]")
        elif ray_angles_rad.dim() == 2:
            if ray_angles_rad.shape != ranges_m.shape:
                raise ValueError("ray_angles_rad must be [N] or [B,N]")
        else:
            raise ValueError("ray_angles_rad must be [N] or [B,N]")
        for tensor in (nominal_twist, ranges_m, ray_angles_rad, positive_alpha):
            if tensor.layout != torch.strided or tensor.is_quantized:
                raise ValueError("CBF inputs must be dense strided floating tensors")
            if tensor.dtype != torch.float32 and tensor.dtype != torch.float64:
                raise ValueError("CBF supports float32 or float64")
            if tensor.dtype != nominal_twist.dtype or tensor.device != nominal_twist.device:
                raise ValueError("CBF tensors must share dtype and device")
        if (valid_mask.layout != torch.strided or valid_mask.is_quantized
                or valid_mask.device != nominal_twist.device):
            raise ValueError("valid_mask must be dense and on the same device")
        valid = (torch.isfinite(nominal_twist).all()
                 & torch.isfinite(positive_alpha).all() & (positive_alpha > 0).all()
                 & ((torch.isfinite(ranges_m) & (ranges_m > 0)) | ~valid_mask).all()
                 & (torch.isfinite(ray_angles_rad) | ~valid_mask).all()
                 & valid_mask.any(dim=1).all())
        if not bool(valid):
            raise ValueError("finite commands, positive alpha and >=1 valid finite positive ray per row required")

    def forward(self, nominal_twist: Tensor, ranges_m: Tensor, ray_angles_rad: Tensor,
                valid_mask: Tensor, positive_alpha: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        self._validate(nominal_twist, ranges_m, ray_angles_rad, valid_mask, positive_alpha)
        # Sanitize before trig/norm: masked NaN/Inf cannot contaminate gradients.
        ranges = torch.where(valid_mask, ranges_m, torch.ones_like(ranges_m))
        angles = torch.where(valid_mask, ray_angles_rad, torch.zeros_like(ranges_m))
        base_angles = angles + self.sensor_yaw_rad
        dx = self.sensor_x_m + ranges * torch.cos(base_angles) - self.lookahead_distance_m
        dy = self.sensor_y_m + ranges * torch.sin(base_angles)
        dx = torch.where(valid_mask, dx, torch.ones_like(dx))
        dy = torch.where(valid_mask, dy, torch.zeros_like(dy))
        distance = torch.sqrt(dx.square() + dy.square())
        if not bool((torch.isfinite(distance) & (distance > 0)).all()):
            raise ValueError("valid obstacle point must have finite nonzero distance from lookahead point")
        h_i = distance - self.envelope_radius_m
        logits = torch.where(valid_mask, -self.kappa * h_i, torch.full_like(h_i, -float("inf")))
        h_comp = -torch.logsumexp(logits, dim=1, keepdim=True) / self.kappa
        weights = torch.softmax(logits, dim=1)
        grad_x = -(weights * dx / distance).sum(dim=1, keepdim=True)
        grad_y = -(weights * dy / distance).sum(dim=1, keepdim=True)
        lg_h = torch.cat((grad_x, grad_y), dim=1)
        q = torch.stack((nominal_twist[:, 0], self.lookahead_distance_m * nominal_twist[:, 1]), dim=1)
        norm_sq = lg_h.square().sum(dim=1, keepdim=True)
        residual_before = (lg_h * q).sum(dim=1, keepdim=True) + positive_alpha * h_comp
        eta = torch.relu(-residual_before / (norm_sq + self.damping_factor))
        correction_q = eta * lg_h
        biased_q = q + correction_q
        biased_twist = torch.stack((biased_q[:, 0], biased_q[:, 1] / self.lookahead_distance_m), dim=1)
        biased_twist = torch.where(eta > 0, biased_twist, nominal_twist)
        residual_after = (lg_h * biased_q).sum(dim=1, keepdim=True) + positive_alpha * h_comp
        correction_norm = torch.linalg.vector_norm(correction_q, dim=1, keepdim=True)
        if not bool(torch.isfinite(biased_twist).all() & torch.isfinite(h_comp).all()
                    & torch.isfinite(residual_before).all() & torch.isfinite(residual_after).all()
                    & torch.isfinite(correction_norm).all()):
            raise ValueError("CBF arithmetic overflow: rescale inputs or use float64")
        return biased_twist, {
            "h_comp": h_comp, "Lg_h_q": lg_h, "Lg_norm_sq": norm_sq,
            "nominal_q": q, "biased_q": biased_q, "correction_q": correction_q,
            "correction_norm_q": correction_norm,
            "residual_before": residual_before, "residual_after": residual_after,
            "eta": eta, "valid_ray_count": valid_mask.sum(dim=1, keepdim=True),
            "constraint_active": eta > 0, "intervened": correction_norm > 0,
        }
