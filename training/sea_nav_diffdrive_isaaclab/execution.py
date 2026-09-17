# SPDX-License-Identifier: MIT
"""Current-tick command execution: joint wheel projection with fail-closed gating.

Each policy tick takes the CURRENT ``clipped_policy_action`` and projects it
along the segment from the previous executed command, jointly satisfying the
body velocity envelope, acceleration limits, and both wheel limits through
the frozen sea_nav_core projection (single geometric segment; never
independent per-wheel clamps).  Rows whose raw safety inputs fail closed
receive an exact zero command and zero wheel targets and are marked
ineligible for the transition.

Pure CPU Torch; loads neither Isaac, ROS, nor DashGo packages.
"""

import torch
from torch import Tensor

from sea_nav_core import (
    RawSafetyObservationSpec,
    UnicycleLookaheadLSECBFLayer,
    raw_safety_row_status,
    sea_ray_angles_rad,
)


class WheelTargetExecutor:
    """Platform projection and wheel-target conversion for one policy tick."""

    def __init__(self, cbf_layer: UnicycleLookaheadLSECBFLayer,
                 safety_spec: RawSafetyObservationSpec, policy_dt_s: float):
        if not isinstance(cbf_layer, UnicycleLookaheadLSECBFLayer):
            raise ValueError("cbf_layer must be UnicycleLookaheadLSECBFLayer")
        if not isinstance(safety_spec, RawSafetyObservationSpec):
            raise ValueError("safety_spec must be RawSafetyObservationSpec")
        if safety_spec.manifest_sha256 != cbf_layer._safety_manifest_sha256_value:
            raise ValueError("safety spec differs from the CBF layer binding")
        if policy_dt_s <= 0:
            raise ValueError("policy_dt_s must be positive")
        self._cbf = cbf_layer
        self._safety_spec = safety_spec
        self._policy_dt_s = float(policy_dt_s)
        self._angles = torch.tensor(sea_ray_angles_rad())

    @property
    def safety_manifest_sha256(self) -> str:
        return self._safety_spec.manifest_sha256

    def execute_tick(self, clipped_policy_action: Tensor,
                     previous_executed_command: Tensor,
                     ranges_m: Tensor, valid: Tensor, age_s: Tensor) -> dict:
        """Project the current action and return the executed command package.

        The residual diagnostics returned here describe the projected command
        only; the actor-side CBF residual on ``distribution_mean`` is a G7
        concern and is never recomputed in this module (no second post-sample
        CBF pass on the sampled action).
        """
        status = raw_safety_row_status(self._safety_spec, ranges_m, valid, age_s)
        batch = clipped_policy_action.shape[0]
        device = clipped_policy_action.device
        dtype = clipped_policy_action.dtype
        if previous_executed_command.shape != (batch, 2) \
                or previous_executed_command.device != device \
                or previous_executed_command.dtype != dtype:
            raise ValueError(
                "previous_executed_command must match the action batch/dtype/device"
            )

        executed = torch.zeros((batch, 2), dtype=dtype, device=device)
        wheels = torch.zeros((batch, 2), dtype=dtype, device=device)
        projection_active = torch.zeros((batch, 3), dtype=torch.bool, device=device)

        if bool(status.ok.any()):
            ok_rows = status.ok.nonzero(as_tuple=False).flatten()
            safe_action = clipped_policy_action.index_select(0, ok_rows)
            safe_previous = previous_executed_command.index_select(0, ok_rows)
            safe_ranges = ranges_m.index_select(0, ok_rows)
            safe_valid = valid.index_select(0, ok_rows)
            safe_age = age_s.index_select(0, ok_rows)
            safe_angles = self._angles.to(device=device, dtype=dtype) \
                .unsqueeze(0).expand(ok_rows.numel(), -1)
            # The projection itself never consumes alpha; the diagnostics-only
            # residual argument receives ones and is not recorded as actor
            # evidence.
            alpha_diagnostics = torch.ones((ok_rows.numel(), 1),
                                           dtype=dtype, device=device)
            projected, diagnostics = self._cbf.project_cbf_command_with_residual(
                safe_action, safe_previous, self._policy_dt_s,
                safe_ranges, safe_angles, safe_valid, safe_age,
                alpha_diagnostics, self._safety_spec.manifest_sha256,
            )
            executed.index_copy_(0, ok_rows, projected)
            wheel_targets = diagnostics["projected_wheel_angular_velocity_rad_s"]
            wheels.index_copy_(0, ok_rows, wheel_targets)
            # Executed body command is the IK inverse of the true wheel targets
            # (not a copy of an unrelated stage).
            recovered = self._cbf.wheel_angular_velocity_to_body_twist(wheel_targets)
            executed.index_copy_(0, ok_rows, recovered)
            projection_active.index_copy_(
                0, ok_rows, diagnostics["projection_active"],
            )

        return {
            "executed_command": executed,
            "wheel_targets_radps": wheels,
            "safety_ok": status.ok,
            "safety_reason_code": status.reason_code,
            "projection_active": projection_active,
            "fail_closed": ~status.ok,
        }
