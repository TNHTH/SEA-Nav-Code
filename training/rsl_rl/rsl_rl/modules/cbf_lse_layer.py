"""Eq. 4 damped CBF safety bias; an active residual remains negative."""
import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class ExactLSECBFLayer(nn.Module):
    def __init__(self, num_rays=41, fov_deg=240.0, safe_radius=0.15,
                 safety_margin=0.05, kappa=10.0, damping_factor=1.0):
        super().__init__()
        if isinstance(num_rays, bool) or not isinstance(num_rays, int) or num_rays <= 0:
            raise ValueError("num_rays must be a positive integer")
        for name, value in (("fov_deg", fov_deg), ("safe_radius", safe_radius),
                            ("kappa", kappa), ("damping_factor", damping_factor)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if fov_deg > 360:
            raise ValueError("fov_deg must be at most 360")
        if not math.isfinite(safety_margin) or safety_margin < 0:
            raise ValueError("safety_margin must be finite and nonnegative")
        self.num_rays = num_rays
        self.fov_deg = float(fov_deg)
        self.safe_radius = float(safe_radius)
        self.safety_margin = float(safety_margin)
        self.d_safe = float(safe_radius + safety_margin)
        self.kappa = float(kappa)
        self.damping_factor = float(damping_factor)
        angles = torch.linspace(-math.radians(fov_deg) / 2, math.radians(fov_deg) / 2, num_rays)
        self.register_buffer("ray_unit_vectors", torch.stack((torch.cos(angles), torch.sin(angles)), dim=1))

    def _supports_aggregate_validation(self, value: Tensor) -> bool:
        if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
            return False
        # An acceleration eligibility set, NOT a public input dtype ban.
        # Complex/newer backend-limited dtypes (e.g. Float8) keep the original
        # short-circuit checks even when they are dense and on one device.
        return (value.dtype == torch.float16 or value.dtype == torch.float32
                or value.dtype == torch.float64 or value.dtype == torch.bfloat16
                or value.dtype == torch.uint8 or value.dtype == torch.int8
                or value.dtype == torch.int16 or value.dtype == torch.int32
                or value.dtype == torch.int64 or value.dtype == torch.bool)

    def _validate_inputs(self, u_bar: Tensor, lidar_dists: Tensor, alpha: Tensor) -> None:
        if u_bar.dim() != 2 or u_bar.size(1) != 3 or u_bar.size(0) == 0:
            raise ValueError("u_bar must be nonempty [B,3]")
        if lidar_dists.dim() != 2 or lidar_dists.size(1) != self.num_rays or lidar_dists.size(0) != u_bar.size(0):
            raise ValueError("lidar_dists must be [B,num_rays] with matching batch")
        if alpha.dim() != 2 or alpha.size(1) != 1 or alpha.size(0) != u_bar.size(0):
            raise ValueError("alpha must be [B,1] with matching batch")
        # Metadata checks must precede aggregation: an unsupported later input
        # must not mask the original short-circuit error from an earlier input.
        if (not self._supports_aggregate_validation(u_bar)
                or not self._supports_aggregate_validation(lidar_dists)
                or not self._supports_aggregate_validation(alpha)
                or u_bar.device != lidar_dists.device or u_bar.device != alpha.device):
            self._validate_values(u_bar, lidar_dists, alpha)
            return
        # This remains a dynamic checked API, not a zero-sync/unchecked path.
        # Valid dense input needs one host decision; invalid input reruns the
        # original ordered checks below to retain precise exceptions/messages.
        valid = (torch.isfinite(u_bar).all() & torch.isfinite(lidar_dists).all()
                 & torch.isfinite(alpha).all() & ~(lidar_dists <= 0).any()
                 & ~(alpha <= 0).any())
        if not valid:
            self._validate_values(u_bar, lidar_dists, alpha)

    def _validate_values(self, u_bar: Tensor, lidar_dists: Tensor, alpha: Tensor) -> None:
        if not torch.isfinite(u_bar).all() or not torch.isfinite(lidar_dists).all() or not torch.isfinite(alpha).all():
            raise ValueError("CBF inputs must be finite")
        if (lidar_dists <= 0).any():
            raise ValueError("lidar_dists must be positive before preprocessing")
        if (alpha <= 0).any():
            raise ValueError("alpha must be already positive; transform raw logits once")

    def _compute_intermediates(self, u_bar: Tensor, lidar_dists: Tensor, alpha: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Single Eq.4 implementation; no optional diagnostic reductions."""
        h_i = lidar_dists - self.d_safe
        h_comp = -torch.logsumexp(-self.kappa * h_i, dim=1, keepdim=True) / self.kappa
        weights = torch.softmax(-self.kappa * h_i, dim=1)
        lg_h = -torch.sum(weights.unsqueeze(-1) * self.ray_unit_vectors.unsqueeze(0), dim=1)
        norm_sq = torch.sum(lg_h.square(), dim=1, keepdim=True)
        r = torch.sum(lg_h * u_bar[:, :2], dim=1, keepdim=True) + alpha * h_comp
        eta_raw = -r / (norm_sq + self.damping_factor)
        eta = torch.relu(eta_raw)
        correction = eta * lg_h
        xy = u_bar[:, :2] + correction
        output = torch.cat((xy, u_bar[:, 2:]), dim=1)
        return output, h_comp, lg_h, norm_sq, r, eta_raw, eta, correction

    def _compute(self, u_bar: Tensor, lidar_dists: Tensor, alpha: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        output, h_comp, lg_h, norm_sq, r, eta_raw, eta, correction = self._compute_intermediates(u_bar, lidar_dists, alpha)
        residual_after = torch.sum(lg_h * output[:, :2], dim=1, keepdim=True) + alpha * h_comp
        return output, {
            "h_comp": h_comp, "Lg_h": lg_h, "Lg_norm_sq": norm_sq,
            "r": r, "eta_raw": eta_raw, "eta": eta,
            "correction_norm": torch.norm(correction, dim=1, keepdim=True),
            "residual_before": r, "residual_after": residual_after,
        }

    @torch.jit.export
    def forward_with_diagnostics(self, u_bar: Tensor, lidar_dists: Tensor, alpha: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Accept positive alpha, return differentiable tensors without state mutation."""
        self._validate_inputs(u_bar, lidar_dists, alpha)
        return self._compute(u_bar, lidar_dists, alpha)

    def forward(self, u_bar: Tensor, lidar_dists: Tensor, alpha: Tensor) -> Tensor:
        self._validate_inputs(u_bar, lidar_dists, alpha)
        return self._compute_intermediates(u_bar, lidar_dists, alpha)[0]
