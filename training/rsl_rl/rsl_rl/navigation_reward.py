"""Shared unweighted equations; paper equations are diagnostic, not run identity."""
import math
import torch

REWARD_NAMES = ("termination", "reach", "velocity", "clearance", "stuck", "collision", "angular")
GYM_REWARD_NAMES = dict(zip(REWARD_NAMES, ("termination", "reach_pos_target_tight", "velo_dir",
                                         "close_obst_vel", "stuck", "collision", "ang_vel_xy")))

def compute_navigation_reward_terms(x, formula_mode):
    """Contact counts/opening selection are explicit caller-owned physical inputs.

    Counts follow the recorded generic/head-base/leg grouping and XY threshold;
    no claim of body equivalence is made by these equations.
    """
    if formula_mode not in ("upstream_fbce672c", "paper_diagnostic"):
        raise ValueError("unsupported reward formula mode: " + str(formula_mode))
    d, vx, vy, wz = x["distance"], x["vx"], x["vy"], x["wz"]
    b = 1 / (1 + 2 * d.square())
    if formula_mode == "paper_diagnostic":
        velocity = x["cos_theta"] * vx + b
        clearance = torch.where(d > 1, x["cos_phi"] * vx, b)
        delta = torch.linalg.vector_norm(x["position_history"] - x["position_history"][:, :1], dim=-1).amax(-1)
        stuck = ((d > 1) & (delta < .1) & (vx > 0) & (wz.abs() < 1)).to(d.dtype)
        angular = torch.sqrt(x["wx"].square() + x["wy"].square())
    else:
        vplus = vx.clamp(min=0)
        velocity = torch.minimum((x["goal_x"] / (d + 1e-4)).clamp(min=0) * vplus, d.clamp(max=.5)) + b
        vlim = x["min_ray"].clamp(max=.5)
        clear = (x["cos_phi"].clamp(min=0) * torch.minimum(vplus, vlim) - .2 * (vplus - vlim).clamp(min=0)).clamp(min=0)
        clearance = torch.where(d > .5, clear, b)
        movement = torch.linalg.vector_norm(x["position_history"] - x["position"][:, None], dim=-1).amax(-1)
        dead = x["dead"]
        # Preserve Torch bool-addition (OR), not an integer double penalty.
        escape = dead & ((vx > 0) | (wz.abs() < 1))
        stuck = (x["not_just_reset"] & (d > .5) & (escape | dead) & (movement < .1)).to(d.dtype)
        stuck = stuck + (wz.abs() + vy.clamp(max=.5).abs() + vx.clamp(max=.5).abs()) * (d <= .5)
        angular = x["wx"].square() + x["wy"].square()
    contacts = x["generic"] + 10 * x["head_base"] + 10 * x["leg"]
    collision = (1 + 4 * (vx.square() + vy.square() + wz.square())) * contacts * (~x["initial"])
    return dict(zip(REWARD_NAMES, (x["terminated"].to(d.dtype), b * (d < .5),
                                  velocity, clearance, stuck, collision, angular)))

def weight_reward_terms(terms, raw_weights, policy_dt_s):
    """The adapter's sole dt integration point; Gym instead pre-scales weights."""
    if isinstance(policy_dt_s, bool) or not math.isfinite(policy_dt_s) or policy_dt_s <= 0:
        raise ValueError("policy_dt_s must be finite and positive")
    if set(terms) != set(REWARD_NAMES) or set(raw_weights) != set(REWARD_NAMES):
        raise ValueError("reward fields must match the seven named terms")
    if any(isinstance(v, bool) or not math.isfinite(v) for v in raw_weights.values()):
        raise ValueError("reward weights must be finite numbers")
    return {k: terms[k] * (raw_weights[k] * policy_dt_s) for k in REWARD_NAMES}
