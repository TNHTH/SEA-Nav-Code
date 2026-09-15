# SPDX-License-Identifier: MIT
"""Small immutable contracts: exact manifests, units and one explicit adaptation."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import ClassVar, Mapping

ADAPTATION_ID = "dashgo_diffdrive_transfer_v1"
RESULT_CLASSIFICATION = "cross_platform_method_adaptation"
SAFETY_SEMANTICS = "differentiable_safety_bias"
DASHGO_PLATFORM_PROVENANCE = (
    "TNHTH/dashgo-rl-navigation@10023c294f34dc32a97005103bc30e6aa0f09bf5:"
    "src/dashgo_rl/dashgo_config.py;src/dashgo_rl/dashgo_env_v2.py;"
    "src/dashgo_rl/control/differential_drive.py;"
    "configs/robot/dashgo.urdf;drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml;"
    "workspaces/ros2_ws/src/dashgo_driver_ros2/config/dashgo_driver.yaml;"
    "workspaces/ros1_catkin_ws/src/dashgo_rl/urdf/dashgo_d1_sim.urdf.xacro"
)
DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE = (
    "TNHTH/dashgo-rl-navigation@10023c294f34dc32a97005103bc30e6aa0f09bf5:"
    "src/dashgo_rl/dashgo_config.py#LIDAR_CONFIG,DashGoLidarSpecs"
)
DASHGO_SIM_CAMERA_RANGE_PROVENANCE = (
    "TNHTH/dashgo-rl-navigation@10023c294f34dc32a97005103bc30e6aa0f09bf5:"
    "src/dashgo_rl/dashgo_env_v2.py#camera_front_left,camera_front_right;"
    "data_type=distance_to_image_plane;clipping_range"
)
ROS_LASERSCAN_RADIAL_RANGE = "ros_laserscan_radial_range"
ISAACLAB_CAMERA_DISTANCE_TO_CAMERA = "isaaclab_camera_distance_to_camera"
ISAACLAB_CAMERA_DISTANCE_TO_IMAGE_PLANE = "isaaclab_camera_distance_to_image_plane"
DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE = (
    "TNHTH/SEA-Nav-Code:dashgo_forward_sensor_experiment_v1;"
    "front_180_sensor;effective_reverse_disabled"
)


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


def _one_of(name, value, expected):
    if type(value) is not str or value not in expected:
        raise ValueError(name + " must be one of " + ",".join(expected))


def _nonempty_string(name, value):
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(name + " must be a nonempty stripped string")


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
    wheel_radius_m: float = 0.0632
    track_width_m: float = 0.342
    max_wheel_velocity_rad_s: float = 5.0
    max_linear_acceleration_mps2: float = 1.0
    max_angular_acceleration_radps2: float = 0.6
    base_frame: str = "base_link"
    plant_parameter_provenance: str = DASHGO_PLATFORM_PROVENANCE
    kind: ClassVar[str] = "differential_drive_platform_v2"

    def __post_init__(self):
        for name in ("footprint_radius_m", "lookahead_distance_m", "max_forward_m_s",
                     "max_reverse_m_s", "max_yaw_rad_s", "policy_dt_s",
                     "wheel_radius_m", "track_width_m", "max_wheel_velocity_rad_s",
                     "max_linear_acceleration_mps2",
                     "max_angular_acceleration_radps2"):
            _number(name, getattr(self, name), strictly_positive=True)
        _number("safety_margin_m", self.safety_margin_m, minimum=0)
        for name in ("sensor_x_m", "sensor_y_m", "sensor_yaw_rad"):
            _number(name, getattr(self, name))
        # Canonicalize equal integer/float inputs to the same manifest identity.
        for field in fields(self):
            if field.name not in ("base_frame", "plant_parameter_provenance"):
                object.__setattr__(self, field.name, float(getattr(self, field.name)))
        _nonempty_string("base_frame", self.base_frame)
        _nonempty_string("plant_parameter_provenance", self.plant_parameter_provenance)


@dataclass(frozen=True)
class EffectiveCommandEnvelopeSpec(_ManifestSpec):
    """Experiment/runtime command bounds, separate from plant capability."""

    profile_id: str
    min_linear_velocity_m_s: float
    max_linear_velocity_m_s: float
    max_abs_yaw_rate_rad_s: float
    envelope_provenance: str
    kind: ClassVar[str] = "effective_command_envelope_v1"

    def __post_init__(self):
        _nonempty_string("profile_id", self.profile_id)
        _number("min_linear_velocity_m_s", self.min_linear_velocity_m_s)
        _number(
            "max_linear_velocity_m_s", self.max_linear_velocity_m_s,
            strictly_positive=True,
        )
        _number(
            "max_abs_yaw_rate_rad_s", self.max_abs_yaw_rate_rad_s,
            strictly_positive=True,
        )
        if self.min_linear_velocity_m_s > self.max_linear_velocity_m_s:
            raise ValueError(
                "min_linear_velocity_m_s must not exceed max_linear_velocity_m_s"
            )
        for name in (
            "min_linear_velocity_m_s", "max_linear_velocity_m_s",
            "max_abs_yaw_rate_rad_s",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        _nonempty_string("envelope_provenance", self.envelope_provenance)


@dataclass(frozen=True)
class RawSafetyObservationSpec(_ManifestSpec):
    """Unnormalized metric LiDAR ABI used only by the safety layer.

    ``ray_angles_rad`` is the actual index-ordered sensor geometry, not an FOV
    from which a consumer may synthesize rays. The manifest hash is a required
    argument to the CBF call so this contract cannot be silently replaced by
    the normalized policy observation.
    """

    sensor_frame: str
    ray_angles_rad: tuple[float, ...]
    max_sensor_age_s: float
    range_min_m: float
    range_definition: str
    range_parameter_provenance: str
    range_max_m: float = 12.0
    contract_id: str = "dashgo_raw_metric_range_v2"
    range_units: str = "m"
    angle_units: str = "rad"
    age_units: str = "s"
    ranges_shape: str = "[B,N]"
    angles_shape: str = "[N] or [B,N]"
    validity_shape: str = "[B,N]"
    sensor_age_shape: str = "[B,1]"
    ray_order: str = "manifest_ray_angles_index_order"
    angle_semantics: str = "sensor_frame:+x_zero,ccw_positive"
    validity_semantics: str = (
        "true=finite_metric_range_within_inclusive_declared_min_max_including_max_clear_return"
    )
    age_semantics: str = "seconds_since_measurement_at_policy_evaluation"
    normalized: bool = False
    kind: ClassVar[str] = "raw_safety_observation_v2"

    def __post_init__(self):
        _nonempty_string("sensor_frame", self.sensor_frame)
        if not isinstance(self.ray_angles_rad, (tuple, list)) or not self.ray_angles_rad:
            raise ValueError("ray_angles_rad must be a nonempty ordered sequence")
        angles = []
        for angle in self.ray_angles_rad:
            _number("ray_angles_rad", angle)
            angles.append(float(angle))
        object.__setattr__(self, "ray_angles_rad", tuple(angles))
        for name in ("max_sensor_age_s", "range_min_m", "range_max_m"):
            _number(name, getattr(self, name), strictly_positive=True)
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.range_min_m >= self.range_max_m:
            raise ValueError("range_min_m must be less than range_max_m")
        _one_of(
            "range_definition", self.range_definition,
            (ROS_LASERSCAN_RADIAL_RANGE, ISAACLAB_CAMERA_DISTANCE_TO_CAMERA),
        )
        _nonempty_string("range_parameter_provenance", self.range_parameter_provenance)
        expected = {
            "contract_id": "dashgo_raw_metric_range_v2",
            "range_units": "m", "angle_units": "rad", "age_units": "s",
            "ranges_shape": "[B,N]", "angles_shape": "[N] or [B,N]",
            "validity_shape": "[B,N]", "sensor_age_shape": "[B,1]",
            "ray_order": "manifest_ray_angles_index_order",
            "angle_semantics": "sensor_frame:+x_zero,ccw_positive",
            "validity_semantics": (
                "true=finite_metric_range_within_inclusive_declared_min_max_including_max_clear_return"
            ),
            "age_semantics": "seconds_since_measurement_at_policy_evaluation",
        }
        for name, value in expected.items():
            _choice(name, getattr(self, name), value)
        if type(self.normalized) is not bool or self.normalized:
            raise ValueError("raw safety ranges must be explicitly unnormalized")

    @property
    def num_rays(self):
        return len(self.ray_angles_rad)

    @property
    def flattened_dim(self):
        return 3 * self.num_rays + 1


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


# ---------------------------------------------------------------------------
# SEA 550-D contracts (G2).  These supersede the legacy 246-D/tanh public ABI
# above as the implementation basis; the legacy specs stay available only as
# historical evidence and are rejected by SeaNavDashgoContractSet below.
# ---------------------------------------------------------------------------
SEA_ALGORITHM_PROFILE = "sea_nav_paper_method_operational_v1"
SEA_SOURCE_SEMANTICS = "upstream_fbce672c_550d"
SEA_PLATFORM_PROFILE = "dashgo_d1_primitive_candidate_v1"
SEA_RUNTIME_STACK = "isaaclab_2_0_2_rsl_rl_1_0_2"
SEA_RESULT_CLASSIFICATION = "cross_platform_method_adaptation"
SEA_VALIDATION_IDENTITY = "simulation_surrogate_candidate"

SEA_FRAME_FIELD_ORDER = (
    "projected_gravity[3],previous_executed_command_v_zero_omega[3],"
    "measured_base_linear_velocity[3],measured_base_angular_velocity[3],"
    "log2_clamped_delayed_ranges[41],delayed_local_goal[2]"
)
SEA_ACTION_STAGES = (
    "nominal_body_twist",
    "distribution_mean",
    "policy_action",
    "clipped_policy_action",
    "executed_command",
)


@dataclass(frozen=True)
class SeaNavObservationSpec(_ManifestSpec):
    """SEA 550-D frame-major policy observation ABI.

    Ten oldest-to-newest frames of 55 values.  The LiDAR channels are delayed
    ``log2(clamp(range_m, 0.1, 3.0))`` values; raw metric safety inputs are a
    separate contract and can never be reconstructed from this observation.
    """

    contract_id: str = "sea_nav_550d_frame_history_v1"
    frame_dim: int = 55
    history_frames: int = 10
    delayed_range_transform: str = "log2_clamp_0.1_3.0_m"
    flatten_order: str = "frame_major_oldest_to_newest"
    frame_field_order: str = SEA_FRAME_FIELD_ORDER
    ray_count: int = 41
    safety_inputs_isolated_from_policy_obs: bool = True
    raw_safety_reconstruction_forbidden: bool = True
    kind: ClassVar[str] = "sea_nav_observation_v1"

    def __post_init__(self):
        _choice("contract_id", self.contract_id, "sea_nav_550d_frame_history_v1")
        for name, expected in (
            ("frame_dim", 55), ("history_frames", 10), ("ray_count", 41),
        ):
            _integer(name, getattr(self, name))
            if getattr(self, name) != expected:
                raise ValueError(name + " changes the SEA 550-D observation ABI")
        _choice("delayed_range_transform", self.delayed_range_transform,
                "log2_clamp_0.1_3.0_m")
        _choice("flatten_order", self.flatten_order,
                "frame_major_oldest_to_newest")
        _choice("frame_field_order", self.frame_field_order, SEA_FRAME_FIELD_ORDER)
        if type(self.safety_inputs_isolated_from_policy_obs) is not bool or not self.safety_inputs_isolated_from_policy_obs:
            raise ValueError("raw safety inputs must stay isolated from the policy observation")
        if type(self.raw_safety_reconstruction_forbidden) is not bool or not self.raw_safety_reconstruction_forbidden:
            raise ValueError("raw safety reconstruction from the policy observation is forbidden")

    @property
    def obs_dim(self):
        return self.frame_dim * self.history_frames


@dataclass(frozen=True)
class SeaNavActionSpec(_ManifestSpec):
    """SEA diagonal-Normal body-twist action contract.

    Distribution is a diagonal Normal with initial std 1.5 on the raw sampled
    action.  Tanh squashing, Jacobian log-prob correction and any second
    post-sample CBF pass are forbidden; PPO scores the same ``policy_action``.
    The CBF applies to ``distribution_mean`` only.
    """

    contract_id: str = "sea_nav_normal_body_twist_v1"
    distribution: str = "diagonal_normal"
    initial_std: float = 1.5
    command_space: str = "body_v_omega"
    command_order: str = "v_mps,omega_radps"
    log_prob_stage: str = "policy_action"
    cbf_applies_to: str = "distribution_mean"
    action_stages: tuple = SEA_ACTION_STAGES
    range_loss_lower: tuple = (-0.15, -1.0)
    range_loss_upper: tuple = (0.3, 1.0)
    tanh_forbidden: bool = True
    jacobian_correction_forbidden: bool = True
    second_post_sample_cbf_forbidden: bool = True
    kind: ClassVar[str] = "sea_nav_action_v1"

    def __post_init__(self):
        _choice("contract_id", self.contract_id, "sea_nav_normal_body_twist_v1")
        _choice("distribution", self.distribution, "diagonal_normal")
        _number("initial_std", self.initial_std, strictly_positive=True)
        if self.initial_std != 1.5:
            raise ValueError("initial_std changes the SEA Normal action ABI")
        _choice("command_space", self.command_space, "body_v_omega")
        _choice("command_order", self.command_order, "v_mps,omega_radps")
        _choice("log_prob_stage", self.log_prob_stage, "policy_action")
        _choice("cbf_applies_to", self.cbf_applies_to, "distribution_mean")
        if tuple(self.action_stages) != SEA_ACTION_STAGES:
            raise ValueError("action stages change the SEA action ABI")
        object.__setattr__(self, "action_stages", tuple(self.action_stages))
        for name in ("range_loss_lower", "range_loss_upper"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(name + " must be a [v, omega] pair")
            for entry in value:
                _number(name, entry)
            object.__setattr__(self, name, (float(value[0]), float(value[1])))
        if any(low >= high for low, high in zip(self.range_loss_lower, self.range_loss_upper)):
            raise ValueError("range_loss_lower must be below range_loss_upper")
        for name in ("tanh_forbidden", "jacobian_correction_forbidden",
                     "second_post_sample_cbf_forbidden"):
            if type(getattr(self, name)) is not bool or not getattr(self, name):
                raise ValueError(name + " is part of the SEA action ABI")

    @property
    def action_dim(self):
        return 2


def sea_nav_dashgo_raw_safety_spec() -> RawSafetyObservationSpec:
    """The frozen 41-ray raw safety geometry for the DashGo surrogate."""
    from .observation import sea_ray_angles_rad
    return RawSafetyObservationSpec(
        sensor_frame="dashgo_sim_lidar",
        ray_angles_rad=sea_ray_angles_rad(),
        max_sensor_age_s=0.18,
        range_min_m=0.1,
        range_max_m=3.0,
        range_definition=ROS_LASERSCAN_RADIAL_RANGE,
        range_parameter_provenance=(
            "SEA simulation operational assumption dashgo_d1_primitive_candidate_v1; "
            "valid no-hit is the 3.0 m in-domain return"
        ),
    )


SEA_DASHGO_CANDIDATE_PROVENANCE = (
    "TNHTH/dashgo-rl-navigation@98018dd09923495db321a09920dccc09f796f805:"
    "configs/robot/dashgo.urdf#sha256=51cb52cc60176405ed24735a4cc648f12fd1924d46f77022ae0e730f4346d892;"
    "drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml#sha256=e1cc89d2220a01e07323c3395b1c675a125c25d4547b9d8f497be07af7cdbd1f;"
    "workspaces/ros2_ws/src/dashgo_rl_ros2/urdf/dashgo_d1_sim.urdf.xacro#sha256=9857b0c4006a8d942ecade413baa2df8b54b6947f5088e5c9ab8f5e925682bb6;"
    "status=candidate_not_calibrated"
)

SEA_DASHGO_CANDIDATE_GEOMETRY = {
    "footprint_radius_m": 0.203,
    "lookahead_distance_m": 0.20,
    "safety_margin_m": 0.05,
    "max_forward_m_s": 0.30,
    "max_reverse_m_s": 0.15,
    "max_yaw_rad_s": 1.0,
    "policy_dt_s": 0.02,
    "wheel_radius_m": 0.0632,
    "track_width_m": 0.342,
    "max_wheel_velocity_rad_s": 5.0,
    "max_linear_acceleration_mps2": 1.0,
    "max_angular_acceleration_radps2": 0.6,
}


def sea_nav_dashgo_candidate_platform_spec() -> DifferentialDrivePlatformSpec:
    """The frozen DashGo primitive candidate platform values (not calibrated)."""
    return DifferentialDrivePlatformSpec(
        plant_parameter_provenance=SEA_DASHGO_CANDIDATE_PROVENANCE,
        **SEA_DASHGO_CANDIDATE_GEOMETRY,
    )


@dataclass(frozen=True)
class SeaNavDashgoContractSet(_ManifestSpec):
    """Binding manifest for the SEA DashGo adaptation.

    Accepts only the SEA 550-D/Normal/41-ray contracts above plus the frozen
    DashGo candidate platform.  The legacy 246-D ``ObservationSpec`` and the
    bounded-tanh ``ActionSpec`` are explicitly rejected as an implementation
    basis, and the raw safety geometry must be exactly the 41-ray pattern.
    """

    algorithm_profile: str = SEA_ALGORITHM_PROFILE
    source_semantics: str = SEA_SOURCE_SEMANTICS
    platform_profile: str = SEA_PLATFORM_PROFILE
    runtime_stack: str = SEA_RUNTIME_STACK
    result_classification: str = SEA_RESULT_CLASSIFICATION
    validation_identity: str = SEA_VALIDATION_IDENTITY
    kind: ClassVar[str] = "sea_nav_dashgo_contract_set_v1"

    def __post_init__(self):
        for name, expected in (
            ("algorithm_profile", SEA_ALGORITHM_PROFILE),
            ("source_semantics", SEA_SOURCE_SEMANTICS),
            ("platform_profile", SEA_PLATFORM_PROFILE),
            ("runtime_stack", SEA_RUNTIME_STACK),
            ("result_classification", SEA_RESULT_CLASSIFICATION),
            ("validation_identity", SEA_VALIDATION_IDENTITY),
        ):
            _choice(name, getattr(self, name), expected)

    @classmethod
    def bind(cls, observation, action, safety_observation, platform) -> "SeaNavDashgoContractSet":
        """Validate the four contracts and return the immutable binding."""
        if not isinstance(observation, SeaNavObservationSpec):
            raise ValueError(
                "the legacy 246-D ObservationSpec is not the implementation basis; "
                "use SeaNavObservationSpec"
            )
        if not isinstance(action, SeaNavActionSpec):
            raise ValueError(
                "the legacy bounded-tanh ActionSpec is not the implementation basis; "
                "use SeaNavActionSpec"
            )
        if not isinstance(safety_observation, RawSafetyObservationSpec):
            raise ValueError("safety_observation must be RawSafetyObservationSpec")
        if not isinstance(platform, DifferentialDrivePlatformSpec):
            raise ValueError("platform must be DifferentialDrivePlatformSpec")
        expected_safety = sea_nav_dashgo_raw_safety_spec()
        if safety_observation.manifest_sha256 != expected_safety.manifest_sha256:
            raise ValueError(
                "safety_observation must be exactly the frozen 41-ray "
                "[-120..120] degree 6-degree-spacing [0.1,3.0] m 0.18 s contract"
            )
        expected_platform = sea_nav_dashgo_candidate_platform_spec()
        if platform.manifest_sha256 != expected_platform.manifest_sha256:
            raise ValueError(
                "platform must be exactly the frozen DashGo candidate geometry"
            )
        return cls()

    @classmethod
    def bound_contract_sha256(cls, observation, action, safety_observation,
                              platform) -> str:
        """Canonical hash over the four bound contract manifests."""
        import hashlib
        cls.bind(observation, action, safety_observation, platform)
        payload = json.dumps(
            {
                "contract_set": cls().to_manifest(),
                "observation": observation.to_manifest(),
                "action": action.to_manifest(),
                "safety_observation": safety_observation.to_manifest(),
                "platform": platform.to_manifest(),
            },
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
