# SPDX-License-Identifier: MIT
"""Damped unicycle-lookahead CBF adaptation with explicit physical identity.

This is an instantaneous static-obstacle differentiable safety bias. Platform
projection and residuals are diagnostics, not a constrained QP, hard-safety
guarantee, driver acknowledgement, or evidence that a robot executed a command.
"""

import hashlib
import json
import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from .contracts import (
    ADAPTATION_ID,
    RESULT_CLASSIFICATION,
    SAFETY_SEMANTICS,
    DifferentialDrivePlatformSpec,
    RawSafetyObservationSpec,
    _number,
)


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _utf8_tensor(value: str) -> Tensor:
    return torch.tensor(list(value.encode("utf-8")), dtype=torch.uint8)


def _digest_tensor(value: str) -> Tensor:
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8)


class UnicycleLookaheadLSECBFLayer(nn.Module):
    """Bias ``[v, omega]`` using a versioned raw metric LiDAR contract.

    The constructor binds complete platform, ray geometry, freshness, ``kappa``
    and damping identity. Every tensor call also supplies the raw-safety
    manifest SHA-256, preventing an anonymous policy-observation tensor from
    satisfying the API merely because one dimension happens to match.
    """

    __constants__ = [
        "lookahead_distance_m", "envelope_radius_m", "sensor_x_m", "sensor_y_m",
        "sensor_yaw_rad", "kappa", "damping_factor", "wheel_radius_m",
        "track_width_m", "max_wheel_velocity_rad_s", "max_forward_m_s",
        "max_reverse_m_s", "max_yaw_rad_s",
        "max_linear_acceleration_mps2", "max_angular_acceleration_radps2",
        "range_max_m", "max_sensor_age_s", "num_rays",
        "_platform_manifest_sha256_value", "_safety_manifest_sha256_value",
        "_configuration_sha256_value", "_configuration_manifest_json_value",
    ]

    def __init__(self, platform: DifferentialDrivePlatformSpec,
                 safety_observation: RawSafetyObservationSpec,
                 kappa: float = 10.0, damping_factor: float = 1.0):
        super().__init__()
        if not isinstance(platform, DifferentialDrivePlatformSpec):
            raise ValueError("platform must be DifferentialDrivePlatformSpec")
        if not isinstance(safety_observation, RawSafetyObservationSpec):
            raise ValueError("safety_observation must be RawSafetyObservationSpec")
        _number("kappa", kappa, strictly_positive=True)
        _number("damping_factor", damping_factor, strictly_positive=True)
        if damping_factor != 1.0:
            raise ValueError("this adaptation fixes paper Eq.4 damping_factor=1.0")

        self.lookahead_distance_m = float(platform.lookahead_distance_m)
        self.envelope_radius_m = float(
            platform.footprint_radius_m + platform.lookahead_distance_m
            + platform.safety_margin_m
        )
        if not math.isfinite(self.envelope_radius_m):
            raise ValueError("footprint/lookahead/margin envelope must remain finite")
        self.sensor_x_m = float(platform.sensor_x_m)
        self.sensor_y_m = float(platform.sensor_y_m)
        self.sensor_yaw_rad = float(platform.sensor_yaw_rad)
        self.kappa = float(kappa)
        self.damping_factor = 1.0
        self.wheel_radius_m = float(platform.wheel_radius_m)
        self.track_width_m = float(platform.track_width_m)
        self.max_wheel_velocity_rad_s = float(platform.max_wheel_velocity_rad_s)
        self.max_forward_m_s = float(platform.max_forward_m_s)
        self.max_reverse_m_s = float(platform.max_reverse_m_s)
        self.max_yaw_rad_s = float(platform.max_yaw_rad_s)
        self.max_linear_acceleration_mps2 = float(platform.max_linear_acceleration_mps2)
        self.max_angular_acceleration_radps2 = float(platform.max_angular_acceleration_radps2)
        self.range_max_m = float(safety_observation.range_max_m)
        self.max_sensor_age_s = float(safety_observation.max_sensor_age_s)
        self.num_rays = int(safety_observation.num_rays)

        algorithm = {
            "kind": "unicycle_lookahead_lse_cbf_v1",
            "kappa": self.kappa,
            "damping_factor": self.damping_factor,
            "damping_semantics": "paper_eq4_epsilon_d_fixed_one",
        }
        identity_payload = {
            "schema_version": 1,
            "kind": "sea_nav_diffdrive_cbf_configuration_v1",
            "adaptation_id": ADAPTATION_ID,
            "result_classification": RESULT_CLASSIFICATION,
            "safety_semantics": SAFETY_SEMANTICS,
            "algorithm": algorithm,
            "platform_projection": {
                "kind": "differential_drive_feasible_segment_v1",
                "order": [
                    "body_speed_clamp", "per_axis_acceleration_limit",
                    "wheel_increment_segment_limit", "wheel_to_body_inverse",
                ],
                "final_residual_source": "inverse_converted_platform_projected_command",
            },
            "platform": platform.to_manifest(),
            "raw_safety_observation": safety_observation.to_manifest(),
            "command_stages": [
                "nominal_body_twist", "cbf_body_twist",
                "platform_projected_command", "executed_command",
            ],
        }
        identity_json = _canonical_json(identity_payload)
        configuration_sha256 = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
        receipt = dict(identity_payload)
        receipt["configuration_sha256"] = configuration_sha256
        receipt_json = _canonical_json(receipt)

        self._platform_manifest_sha256_value = platform.manifest_sha256
        self._safety_manifest_sha256_value = safety_observation.manifest_sha256
        self._configuration_sha256_value = configuration_sha256
        self._configuration_manifest_json_value = receipt_json

        # Persistent buffers make state_dict non-anonymous. The load hook below
        # rejects mismatches instead of overwriting immutable configuration.
        self.register_buffer(
            "_identity_configuration_sha256",
            _digest_tensor(self._configuration_sha256_value), persistent=True,
        )
        self.register_buffer(
            "_identity_platform_sha256",
            _digest_tensor(self._platform_manifest_sha256_value), persistent=True,
        )
        self.register_buffer(
            "_identity_safety_sha256",
            _digest_tensor(self._safety_manifest_sha256_value), persistent=True,
        )
        self.register_buffer("_identity_manifest_utf8", _utf8_tensor(receipt_json), persistent=True)
        self.register_buffer(
            "_identity_numeric_configuration",
            torch.tensor([
                self.lookahead_distance_m, self.envelope_radius_m, self.sensor_x_m,
                self.sensor_y_m, self.sensor_yaw_rad, self.kappa, self.damping_factor,
                self.wheel_radius_m, self.track_width_m, self.max_forward_m_s,
                self.max_reverse_m_s, self.max_yaw_rad_s,
                self.max_wheel_velocity_rad_s,
                self.max_linear_acceleration_mps2,
                self.max_angular_acceleration_radps2, self.range_max_m,
                self.max_sensor_age_s,
            ], dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "_expected_ray_angles_rad",
            torch.tensor(safety_observation.ray_angles_rad, dtype=torch.float64),
            persistent=True,
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        protected = (
            "_identity_configuration_sha256", "_identity_platform_sha256",
            "_identity_safety_sha256", "_identity_manifest_utf8",
            "_identity_numeric_configuration", "_expected_ray_angles_rad",
        )
        for name in protected:
            key = prefix + name
            incoming = state_dict.get(key)
            expected = getattr(self, name)
            if incoming is not None:
                compatible = (
                    isinstance(incoming, Tensor)
                    and incoming.shape == expected.shape
                    and incoming.layout == expected.layout
                    and torch.equal(
                        incoming.detach().to(device="cpu", dtype=expected.dtype),
                        expected.detach().to(device="cpu"),
                    )
                )
                if not compatible:
                    error_msgs.append("incompatible immutable SEA-Nav CBF identity at " + key)
                    # load_state_dict owns this shallow mapping; replacement
                    # prevents a failed restore from mutating layer identity.
                    state_dict[key] = expected.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    @torch.jit.unused
    def manifest_receipt(self):
        """Return the complete eager-mode configuration receipt."""
        return json.loads(self._configuration_manifest_json_value)

    @torch.jit.unused
    def bind_runtime_safety_spec(self, actual: RawSafetyObservationSpec) -> str:
        """Validate Python-side metadata and return the mandatory call token."""
        if not isinstance(actual, RawSafetyObservationSpec):
            raise ValueError("runtime safety metadata is not RawSafetyObservationSpec")
        if actual.manifest_sha256 != self._safety_manifest_sha256_value:
            raise ValueError("runtime raw safety observation identity mismatch")
        return actual.manifest_sha256

    @torch.jit.export
    def configuration_manifest_json(self) -> str:
        return self._configuration_manifest_json_value

    @torch.jit.export
    def configuration_sha256(self) -> str:
        return self._configuration_sha256_value

    @torch.jit.export
    def platform_manifest_sha256(self) -> str:
        return self._platform_manifest_sha256_value

    @torch.jit.export
    def safety_manifest_sha256(self) -> str:
        return self._safety_manifest_sha256_value

    @torch.jit.export
    def assert_configuration_identity(self, expected_sha256: str) -> None:
        if expected_sha256 != self._configuration_sha256_value:
            raise ValueError("SEA-Nav CBF configuration identity mismatch")

    def _validate_matrix(self, value: Tensor, name: str, columns: int) -> None:
        if value.dim() != 2 or value.size(0) == 0 or value.size(1) != columns:
            raise ValueError(name + " must have the declared nonempty matrix shape")
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError(name + " must be a dense strided floating tensor")
        if value.dtype != torch.float32 and value.dtype != torch.float64:
            raise ValueError(name + " must be float32 or float64")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(name + " must be finite")

    def _validate(self, body_twist: Tensor, ranges_m: Tensor, ray_angles_rad: Tensor,
                  valid_mask: Tensor, sensor_age_s: Tensor, positive_alpha: Tensor,
                  safety_manifest_sha256: str) -> None:
        if safety_manifest_sha256 != self._safety_manifest_sha256_value:
            raise ValueError("runtime raw safety observation identity mismatch")
        self._validate_matrix(body_twist, "body_twist", 2)
        if (ranges_m.dim() != 2 or ranges_m.size(0) != body_twist.size(0)
                or ranges_m.size(1) != self.num_rays):
            raise ValueError("ranges_m must match the declared raw safety [B,N] shape")
        if valid_mask.shape != ranges_m.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,N]")
        if (positive_alpha.dim() != 2 or positive_alpha.size(0) != body_twist.size(0)
                or positive_alpha.size(1) != 1):
            raise ValueError("positive_alpha must be [B,1]")
        if (sensor_age_s.dim() != 2 or sensor_age_s.size(0) != body_twist.size(0)
                or sensor_age_s.size(1) != 1):
            raise ValueError("sensor_age_s must be [B,1]")
        if ray_angles_rad.dim() == 1:
            if ray_angles_rad.size(0) != self.num_rays:
                raise ValueError("ray_angles_rad must match declared [N] or [B,N]")
        elif ray_angles_rad.dim() == 2:
            if ray_angles_rad.shape != ranges_m.shape:
                raise ValueError("ray_angles_rad must match declared [N] or [B,N]")
        else:
            raise ValueError("ray_angles_rad must match declared [N] or [B,N]")
        for tensor in (ranges_m, ray_angles_rad, positive_alpha, sensor_age_s):
            if tensor.layout != torch.strided or tensor.is_quantized:
                raise ValueError("CBF inputs must be dense strided floating tensors")
            if tensor.dtype != torch.float32 and tensor.dtype != torch.float64:
                raise ValueError("CBF supports float32 or float64")
            if tensor.dtype != body_twist.dtype or tensor.device != body_twist.device:
                raise ValueError("CBF tensors must share dtype and device")
        if (valid_mask.layout != torch.strided or valid_mask.is_quantized
                or valid_mask.device != body_twist.device):
            raise ValueError("valid_mask must be dense and on the same device")

        expected_angles = self._expected_ray_angles_rad.to(
            device=body_twist.device, dtype=body_twist.dtype
        )
        if ray_angles_rad.dim() == 1:
            angles_match = torch.equal(ray_angles_rad, expected_angles)
        else:
            angles_match = torch.equal(
                ray_angles_rad,
                expected_angles.unsqueeze(0).expand(body_twist.size(0), -1),
            )
        if not bool(angles_match):
            raise ValueError("ray_angles_rad differs from the manifest-bound sensor geometry")

        valid = (
            bool(torch.isfinite(positive_alpha).all())
            and bool((positive_alpha > 0).all())
            and bool(torch.isfinite(sensor_age_s).all())
            and bool((sensor_age_s >= 0).all())
            and bool((sensor_age_s <= self.max_sensor_age_s).all())
            and bool(torch.isfinite(ray_angles_rad).all())
            and bool(((torch.isfinite(ranges_m) & (ranges_m > 0)
                       & (ranges_m <= self.range_max_m)) | ~valid_mask).all())
            and bool(valid_mask.any(dim=1).all())
        )
        if not valid:
            raise ValueError(
                "positive alpha, fresh raw metric data and >=1 valid finite positive ray per row required"
            )

    def _geometry(self, ranges_m: Tensor, ray_angles_rad: Tensor,
                  valid_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ranges = torch.where(valid_mask, ranges_m, torch.ones_like(ranges_m))
        angles = ray_angles_rad
        if angles.dim() == 1:
            angles = angles.unsqueeze(0).expand(ranges_m.size(0), -1)
        base_angles = angles + self.sensor_yaw_rad
        dx = self.sensor_x_m + ranges * torch.cos(base_angles) - self.lookahead_distance_m
        dy = self.sensor_y_m + ranges * torch.sin(base_angles)
        dx = torch.where(valid_mask, dx, torch.ones_like(dx))
        dy = torch.where(valid_mask, dy, torch.zeros_like(dy))
        distance = torch.sqrt(dx.square() + dy.square())
        if not bool(((torch.isfinite(distance) & (distance > 0)) | ~valid_mask).all()):
            raise ValueError("valid obstacle point must have finite nonzero distance from lookahead point")
        distance = torch.where(valid_mask, distance, torch.ones_like(distance))
        h_i = distance - self.envelope_radius_m
        logits = torch.where(
            valid_mask, -self.kappa * h_i, torch.full_like(h_i, -float("inf"))
        )
        h_comp = -torch.logsumexp(logits, dim=1, keepdim=True) / self.kappa
        weights = torch.softmax(logits, dim=1)
        grad_x = -(weights * dx / distance).sum(dim=1, keepdim=True)
        grad_y = -(weights * dy / distance).sum(dim=1, keepdim=True)
        lg_h = torch.cat((grad_x, grad_y), dim=1)
        norm_sq = lg_h.square().sum(dim=1, keepdim=True)
        return h_comp, lg_h, norm_sq

    def _q_for_twist(self, body_twist: Tensor) -> Tensor:
        return torch.stack(
            (body_twist[:, 0], self.lookahead_distance_m * body_twist[:, 1]), dim=1
        )

    def _residual(self, body_twist: Tensor, h_comp: Tensor, lg_h: Tensor,
                  positive_alpha: Tensor) -> Tensor:
        return (lg_h * self._q_for_twist(body_twist)).sum(dim=1, keepdim=True) + positive_alpha * h_comp

    def forward(self, nominal_body_twist: Tensor, ranges_m: Tensor,
                ray_angles_rad: Tensor, valid_mask: Tensor, sensor_age_s: Tensor,
                positive_alpha: Tensor, safety_manifest_sha256: str
                ) -> Tuple[Tensor, Dict[str, Tensor]]:
        self._validate(
            nominal_body_twist, ranges_m, ray_angles_rad, valid_mask,
            sensor_age_s, positive_alpha, safety_manifest_sha256,
        )
        h_comp, lg_h, norm_sq = self._geometry(ranges_m, ray_angles_rad, valid_mask)
        nominal_q = self._q_for_twist(nominal_body_twist)
        residual_nominal = (lg_h * nominal_q).sum(dim=1, keepdim=True) + positive_alpha * h_comp
        eta = torch.relu(-residual_nominal / (norm_sq + self.damping_factor))
        correction_q = eta * lg_h
        cbf_q = nominal_q + correction_q
        cbf_body_twist = torch.stack(
            (cbf_q[:, 0], cbf_q[:, 1] / self.lookahead_distance_m), dim=1
        )
        cbf_body_twist = torch.where(eta > 0, cbf_body_twist, nominal_body_twist)
        residual_cbf = (lg_h * cbf_q).sum(dim=1, keepdim=True) + positive_alpha * h_comp
        correction_norm = torch.linalg.vector_norm(correction_q, dim=1, keepdim=True)
        if not bool(
            torch.isfinite(cbf_body_twist).all() & torch.isfinite(h_comp).all()
            & torch.isfinite(residual_nominal).all() & torch.isfinite(residual_cbf).all()
            & torch.isfinite(correction_norm).all()
        ):
            raise ValueError("CBF arithmetic overflow: rescale inputs or use float64")
        return cbf_body_twist, {
            "nominal_body_twist": nominal_body_twist,
            "cbf_body_twist": cbf_body_twist,
            "h_comp": h_comp,
            "Lg_h_q": lg_h,
            "Lg_norm_sq": norm_sq,
            "nominal_q": nominal_q,
            "biased_q": cbf_q,
            "correction_q": correction_q,
            "correction_norm_q": correction_norm,
            "residual_nominal": residual_nominal,
            "residual_cbf": residual_cbf,
            # Compatibility aliases are stage-local, never an applied receipt.
            "residual_before": residual_nominal,
            "residual_after": residual_cbf,
            "eta": eta,
            "sensor_age_s": sensor_age_s,
            "valid_ray_count": valid_mask.sum(dim=1, keepdim=True),
            "constraint_active": eta > 0,
            "intervened": correction_norm > 0,
        }

    @torch.jit.export
    def body_twist_to_wheel_angular_velocity(self, body_twist: Tensor) -> Tensor:
        """Convert ``[v m/s, omega rad/s]`` to ``[left,right] rad/s``."""
        self._validate_matrix(body_twist, "body_twist", 2)
        half_track_omega = 0.5 * self.track_width_m * body_twist[:, 1]
        wheel_rad_s = torch.stack(
            ((body_twist[:, 0] - half_track_omega) / self.wheel_radius_m,
             (body_twist[:, 0] + half_track_omega) / self.wheel_radius_m),
            dim=1,
        )
        if not bool(torch.isfinite(wheel_rad_s).all()):
            raise ValueError("wheel conversion arithmetic overflow")
        return wheel_rad_s

    @torch.jit.export
    def wheel_angular_velocity_to_body_twist(self, wheel_rad_s: Tensor) -> Tensor:
        """Convert ``[left,right] rad/s`` to ``[v m/s, omega rad/s]``."""
        self._validate_matrix(wheel_rad_s, "wheel_rad_s", 2)
        left = wheel_rad_s[:, 0]
        right = wheel_rad_s[:, 1]
        body_twist = torch.stack(
            (0.5 * self.wheel_radius_m * (left + right),
             self.wheel_radius_m * (right - left) / self.track_width_m),
            dim=1,
        )
        if not bool(torch.isfinite(body_twist).all()):
            raise ValueError("body twist conversion arithmetic overflow")
        return body_twist

    def _project_platform_command(self, cbf_body_twist: Tensor,
                                  previous_executed_command: Tensor,
                                  dt_s: float
                                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._validate_matrix(cbf_body_twist, "cbf_body_twist", 2)
        self._validate_matrix(previous_executed_command, "previous_executed_command", 2)
        if (previous_executed_command.dtype != cbf_body_twist.dtype
                or previous_executed_command.device != cbf_body_twist.device
                or previous_executed_command.size(0) != cbf_body_twist.size(0)):
            raise ValueError("command stages must share batch, dtype and device")
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive seconds")
        previous_wheel_rad_s = self.body_twist_to_wheel_angular_velocity(
            previous_executed_command
        )
        previous_is_bounded = (
            (previous_executed_command[:, 0] <= self.max_forward_m_s).all()
            & (previous_executed_command[:, 0] >= -self.max_reverse_m_s).all()
            & (previous_executed_command[:, 1].abs() <= self.max_yaw_rad_s).all()
            & (previous_wheel_rad_s.abs() <= self.max_wheel_velocity_rad_s).all()
        )
        if not bool(previous_is_bounded):
            raise ValueError("previous_executed_command is outside declared platform limits")

        speed_limited = torch.stack(
            (torch.clamp(cbf_body_twist[:, 0], -self.max_reverse_m_s, self.max_forward_m_s),
             torch.clamp(cbf_body_twist[:, 1], -self.max_yaw_rad_s, self.max_yaw_rad_s)),
            dim=1,
        )
        max_delta = torch.stack(
            (torch.full_like(cbf_body_twist[:, 0], self.max_linear_acceleration_mps2 * dt_s),
             torch.full_like(cbf_body_twist[:, 1], self.max_angular_acceleration_radps2 * dt_s)),
            dim=1,
        )
        delta = torch.maximum(
            torch.minimum(speed_limited - previous_executed_command, max_delta),
            -max_delta,
        )
        acceleration_limited = previous_executed_command + delta
        pre_wheel_limit_rad_s = self.body_twist_to_wheel_angular_velocity(
            acceleration_limited
        )
        wheel_delta_rad_s = pre_wheel_limit_rad_s - previous_wheel_rad_s
        safe_wheel_delta_rad_s = torch.where(
            wheel_delta_rad_s != 0,
            wheel_delta_rad_s,
            torch.ones_like(wheel_delta_rad_s),
        )
        scale_to_upper = (
            self.max_wheel_velocity_rad_s - previous_wheel_rad_s
        ) / safe_wheel_delta_rad_s
        scale_to_lower = (
            -self.max_wheel_velocity_rad_s - previous_wheel_rad_s
        ) / safe_wheel_delta_rad_s
        per_wheel_scale = torch.where(
            wheel_delta_rad_s > 0,
            scale_to_upper,
            torch.where(
                wheel_delta_rad_s < 0,
                scale_to_lower,
                torch.ones_like(wheel_delta_rad_s),
            ),
        )
        wheel_limit_scale = torch.clamp(
            per_wheel_scale.amin(dim=1, keepdim=True), min=0.0, max=1.0
        )
        wheel_rad_s = previous_wheel_rad_s + wheel_limit_scale * wheel_delta_rad_s
        projected = self.wheel_angular_velocity_to_body_twist(wheel_rad_s)
        return (
            projected, speed_limited, acceleration_limited,
            previous_wheel_rad_s, pre_wheel_limit_rad_s,
            wheel_limit_scale, wheel_rad_s,
        )

    @torch.jit.export
    def project_cbf_command_with_residual(
            self, cbf_body_twist: Tensor, previous_executed_command: Tensor,
            dt_s: float, ranges_m: Tensor, ray_angles_rad: Tensor,
            valid_mask: Tensor, sensor_age_s: Tensor, positive_alpha: Tensor,
            safety_manifest_sha256: str) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Apply body/acceleration limits and recompute the projected residual.

        The returned command is still only ``platform_projected_command``; a
        driver-observed command must be passed to ``diagnose_executed_command``.
        """
        self._validate(
            cbf_body_twist, ranges_m, ray_angles_rad, valid_mask,
            sensor_age_s, positive_alpha, safety_manifest_sha256,
        )
        h_comp, lg_h, _ = self._geometry(ranges_m, ray_angles_rad, valid_mask)
        (projected, speed_limited, acceleration_limited,
         previous_wheel_rad_s, pre_wheel_limit_rad_s, wheel_limit_scale,
         wheel_rad_s) = self._project_platform_command(
             cbf_body_twist, previous_executed_command, dt_s
         )
        residual_cbf = self._residual(cbf_body_twist, h_comp, lg_h, positive_alpha)
        residual_projected = self._residual(projected, h_comp, lg_h, positive_alpha)
        if not bool(
            torch.isfinite(projected).all() & torch.isfinite(wheel_rad_s).all()
            & torch.isfinite(residual_cbf).all()
            & torch.isfinite(residual_projected).all()
        ):
            raise ValueError("platform projection diagnostic arithmetic overflow")
        dt = torch.full(
            (cbf_body_twist.size(0), 1), dt_s,
            dtype=cbf_body_twist.dtype, device=cbf_body_twist.device,
        )
        return projected, {
            "cbf_body_twist": cbf_body_twist,
            "previous_executed_command": previous_executed_command,
            "speed_limited_body_twist": speed_limited,
            "acceleration_limited_body_twist": acceleration_limited,
            "previous_wheel_angular_velocity_rad_s": previous_wheel_rad_s,
            "pre_wheel_limit_angular_velocity_rad_s": pre_wheel_limit_rad_s,
            "wheel_limit_scale": wheel_limit_scale,
            "platform_projected_command": projected,
            "projected_wheel_angular_velocity_rad_s": wheel_rad_s,
            "projection_delta_body_twist": projected - previous_executed_command,
            "projection_dt_s": dt,
            "residual_cbf": residual_cbf,
            "residual_platform_projected": residual_projected,
        }

    @torch.jit.export
    def forward_with_platform_projection(
            self, nominal_body_twist: Tensor, previous_executed_command: Tensor,
            dt_s: float, ranges_m: Tensor, ray_angles_rad: Tensor,
            valid_mask: Tensor, sensor_age_s: Tensor, positive_alpha: Tensor,
            safety_manifest_sha256: str) -> Tuple[Tensor, Dict[str, Tensor]]:
        cbf_body_twist, cbf_diagnostics = self.forward(
            nominal_body_twist, ranges_m, ray_angles_rad, valid_mask,
            sensor_age_s, positive_alpha, safety_manifest_sha256,
        )
        projected, projection_diagnostics = self.project_cbf_command_with_residual(
            cbf_body_twist, previous_executed_command, dt_s, ranges_m,
            ray_angles_rad, valid_mask, sensor_age_s, positive_alpha,
            safety_manifest_sha256,
        )
        for key, value in projection_diagnostics.items():
            cbf_diagnostics[key] = value
        return projected, cbf_diagnostics

    @torch.jit.export
    def diagnose_executed_command(
            self, executed_command: Tensor, ranges_m: Tensor,
            ray_angles_rad: Tensor, valid_mask: Tensor, sensor_age_s: Tensor,
            positive_alpha: Tensor, safety_manifest_sha256: str
            ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Recompute a residual for a separately observed executed command."""
        self._validate(
            executed_command, ranges_m, ray_angles_rad, valid_mask,
            sensor_age_s, positive_alpha, safety_manifest_sha256,
        )
        h_comp, lg_h, _ = self._geometry(ranges_m, ray_angles_rad, valid_mask)
        residual = self._residual(executed_command, h_comp, lg_h, positive_alpha)
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("executed command diagnostic arithmetic overflow")
        return residual, {
            "executed_command": executed_command,
            "residual_executed_command": residual,
            "sensor_age_s": sensor_age_s,
        }
