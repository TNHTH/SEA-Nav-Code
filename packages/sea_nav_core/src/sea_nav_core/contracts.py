"""Small immutable contracts: exact manifests, units and one explicit adaptation."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import ClassVar, Mapping

ADAPTATION_ID = "dashgo_diffdrive_transfer_v1"
RESULT_CLASSIFICATION = "cross_platform_method_adaptation"
SAFETY_SEMANTICS = "differentiable_safety_bias"


def _number(name, value, *, minimum=None, strictly_positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite real number, not boolean")
    if strictly_positive and value <= 0:
        raise ValueError(name + " must be positive")
    if minimum is not None and value < minimum:
        raise ValueError(name + " is below its minimum")


def _integer(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")


def _choice(name, value, expected):
    if type(value) is not str or value != expected:
        raise ValueError(name + " must be " + expected)


class _ManifestSpec:
    kind: ClassVar[str]

    def to_manifest(self):
        return {
            "schema_version": 1,
            "kind": self.kind,
            "adaptation_id": ADAPTATION_ID,
            "result_classification": RESULT_CLASSIFICATION,
            "safety_semantics": SAFETY_SEMANTICS,
            "parameters": asdict(self),
        }

    @classmethod
    def from_manifest(cls, manifest):
        keys = {"schema_version", "kind", "adaptation_id", "result_classification",
                "safety_semantics", "parameters"}
        if not isinstance(manifest, Mapping) or set(manifest) != keys:
            raise ValueError("manifest fields must exactly match the schema")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("unsupported manifest schema_version")
        for name, expected in (("kind", cls.kind), ("adaptation_id", ADAPTATION_ID),
                               ("result_classification", RESULT_CLASSIFICATION),
                               ("safety_semantics", SAFETY_SEMANTICS)):
            _choice(name, manifest[name], expected)
        parameters = manifest["parameters"]
        if not isinstance(parameters, Mapping) or set(parameters) != {f.name for f in fields(cls)}:
            raise ValueError("parameters must exactly match the spec fields")
        return cls(**parameters)

    @property
    def manifest_sha256(self):
        payload = json.dumps(self.to_manifest(), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DifferentialDrivePlatformSpec(_ManifestSpec):
    """Caller-supplied geometry; no implicit claim of hardware calibration.

    Planar base: +x forward, +y left, +yaw counter-clockwise. Sensor x/y/yaw
    transform LiDAR-frame points into this base frame. Radius encloses the
    entire robot around its unicycle reference point, not only its chassis.
    """

    footprint_radius_m: float
    lookahead_distance_m: float
    safety_margin_m: float = 0.05
    sensor_x_m: float = 0.0
    sensor_y_m: float = 0.0
    sensor_yaw_rad: float = 0.0
    max_forward_m_s: float = 0.3
    max_reverse_m_s: float = 0.15
    max_yaw_rad_s: float = 1.0
    policy_dt_s: float = 0.05
    base_frame: str = "base_link"
    kind: ClassVar[str] = "differential_drive_platform_v1"

    def __post_init__(self):
        for name in ("footprint_radius_m", "lookahead_distance_m", "max_forward_m_s",
                     "max_reverse_m_s", "max_yaw_rad_s", "policy_dt_s"):
            _number(name, getattr(self, name), strictly_positive=True)
        _number("safety_margin_m", self.safety_margin_m, minimum=0)
        for name in ("sensor_x_m", "sensor_y_m", "sensor_yaw_rad"):
            _number(name, getattr(self, name))
        # Canonicalize equal integer/float inputs to the same manifest identity.
        for field in fields(self):
            if field.name != "base_frame":
                object.__setattr__(self, field.name, float(getattr(self, field.name)))
        if type(self.base_frame) is not str or not self.base_frame.strip() or self.base_frame != self.base_frame.strip():
            raise ValueError("base_frame must be a nonempty stripped string")


@dataclass(frozen=True)
class ObservationSpec(_ManifestSpec):
    """The existing DashGo front-180 term-major policy ABI, not CBF ranges.

    The normalizer's actual fitted state and last_action meaning are explicit
    consumer-owned identities: no Go2 padding or invisible-sector synthesis.
    """

    last_action_stage: str
    normalizer_id: str
    contract_id: str = "dashgo_front_180_history_v1"
    num_rays: int = 72
    history_frames: int = 3
    fov_deg: float = 180.0
    range_max_m: float = 12.0
    history_layout: str = "term_major"
    term_order: str = "lidar,waypoint,goal,forward_velocity,yaw_rate,last_action"
    kind: ClassVar[str] = "observation_v1"

    def __post_init__(self):
        _choice("contract_id", self.contract_id, "dashgo_front_180_history_v1")
        for name, expected in (("num_rays", 72), ("history_frames", 3)):
            _integer(name, getattr(self, name))
            if getattr(self, name) != expected:
                raise ValueError(name + " changes the declared DashGo observation ABI")
        for name, expected in (("fov_deg", 180.0), ("range_max_m", 12.0)):
            _number(name, getattr(self, name), strictly_positive=True)
            if getattr(self, name) != expected:
                raise ValueError(name + " changes the declared DashGo observation ABI")
            object.__setattr__(self, name, float(getattr(self, name)))
        _choice("history_layout", self.history_layout, "term_major")
        _choice("term_order", self.term_order, "lidar,waypoint,goal,forward_velocity,yaw_rate,last_action")
        if type(self.last_action_stage) is not str or self.last_action_stage not in ("policy_action", "clipped_policy_action", "executed_command"):
            raise ValueError("last_action_stage must name its actual producer")
        if type(self.normalizer_id) is not str or not self.normalizer_id.strip() or self.normalizer_id != self.normalizer_id.strip():
            raise ValueError("normalizer_id must be a nonempty stripped identity")

    @property
    def obs_dim(self):
        return (self.num_rays + 3 + 3 + 1 + 1 + 2) * self.history_frames


@dataclass(frozen=True)
class ActionSpec(_ManifestSpec):
    distribution: str = "bounded_tanh_gaussian"
    policy_space: str = "normalized_twist2"
    command_space: str = "body_v_omega"
    command_units: str = "m/s,rad/s"
    command_order: str = "v,omega"
    log_prob_stage: str = "policy_action"
    kl_space: str = "latent_gaussian"
    cbf_stage: str = "nominal_mean_body_twist"
    metric_space: str = "q=[v,lookahead_distance_m*omega]"
    kind: ClassVar[str] = "action_v1"

    def __post_init__(self):
        expected = {
            "distribution": "bounded_tanh_gaussian", "policy_space": "normalized_twist2",
            "command_space": "body_v_omega", "command_units": "m/s,rad/s", "command_order": "v,omega",
            "log_prob_stage": "policy_action", "kl_space": "latent_gaussian",
            "cbf_stage": "nominal_mean_body_twist", "metric_space": "q=[v,lookahead_distance_m*omega]",
        }
        for name, value in expected.items():
            _choice(name, getattr(self, name), value)

    @property
    def action_dim(self):
        return 2
