"""Simulator-independent reset partition and lifecycle policy."""
import torch
from dataclasses import dataclass


def validate_reset_reward_terms(names):
    supported = {"termination", "collision", "close_obst_vel", "stuck", "velo_dir",
                 "reach_pos_target_tight", "ang_vel_xy", "action_rate", "dof_acc", "feet_air_time"}
    unknown = set(names) - supported
    if unknown:
        raise ValueError("unsupported reset consumer reward terms: " + ", ".join(sorted(unknown)))


def require_runtime_contract(contract, runtime_stack):
    """Require a runtime integration, never infer simulator support from tensors.

    The integration must validate scene generation, immutable terrain/map,
    joint/frame and controller identity; defer carrier auto-reset until capture;
    reset contacts/actuators and refresh caches with masked effects. Its hooks
    are exercised only by real runtimes, separately from CPU policy evidence.
    """
    capabilities = ("terminal_capture_before_autoreset", "masked_contact_reset",
                    "masked_cache_refresh", "stateless_controller", "stable_scene_context")
    if (contract is None or getattr(contract, "runtime_stack", None) != runtime_stack
            or not all(getattr(contract, name, False) is True for name in capabilities)
            or not all(callable(getattr(contract, name, None)) for name in
                       ("validate", "prepare_rows", "refresh_rows", "finish_rows"))):
        raise RuntimeError("blocked: replay needs a verified " + runtime_stack + " scene/controller/frame/contact/auto-reset contract")
    return contract


def partition_reset_env_ids(env_ids, wants_replay, reserved_mask):
    if env_ids.ndim != 1 or len(torch.unique(env_ids)) != len(env_ids):
        raise ValueError("requested IDs must be unique")
    if wants_replay.shape != env_ids.shape or reserved_mask.shape != env_ids.shape:
        raise ValueError("partition mask shape mismatch")
    if wants_replay.dtype != torch.bool or reserved_mask.dtype != torch.bool or bool((reserved_mask & ~wants_replay).any()):
        raise ValueError("invalid replay partition masks")
    return env_ids[~wants_replay], env_ids[reserved_mask], env_ids[wants_replay & ~reserved_mask]


@dataclass(frozen=True)
class ResetResult:
    normal_ids: torch.Tensor
    replay_ids: torch.Tensor
    fallback_ids: torch.Tensor
    fallback_reason: str = ""


def execute_reset_transaction(env_ids, wants_replay, selection, ring, normal_reset,
                              write_replay, reconstruct, epilogue, validate=None):
    """Publish only after all runtime hooks succeed; failed fallback propagates.

    Runtime batch setters cannot report partial physical success, so any failed
    replay stage makes the whole attempted subset uncertain. Normal reset is
    completed (including reconstruction and epilogue) before cancellation.
    Episode accounting belongs to the caller and must not live in these hooks.
    """
    if not bool(torch.isin(selection.env_ids, env_ids).all()) or len(torch.unique(selection.env_ids)) != len(selection.env_ids):
        raise ValueError("replay reservation must be a unique subset of requested reset IDs")
    reserved = torch.isin(env_ids, selection.env_ids)
    normal, replay, fallback = partition_reset_env_ids(env_ids, wants_replay, reserved)
    reason = "no_eligible_history" if len(fallback) else ""
    if len(replay):
        try:
            (validate or ring.validate_selection)(selection)
            write_replay(selection)
            reconstruct(replay)
            epilogue(replay)
        except Exception as exc:
            normal_reset(replay)
            reconstruct(replay)
            epilogue(replay)
            fallback = torch.cat((fallback, replay))
            replay = replay[:0]
            reason = type(exc).__name__ + ": " + str(exc)
    for ids in (normal, env_ids[wants_replay & ~reserved]):
        if len(ids):
            normal_reset(ids)
            reconstruct(ids)
            epilogue(ids)
    # No ack precedes an observation/bootstrap/base-epilogue failure.
    if len(replay):
        ring.acknowledge_restore(selection)
    elif len(selection.env_ids):
        ring.cancel_restore(selection, reason)
    return ResetResult(normal, replay, fallback, reason)


def bootstrap_history(history, env_ids, frame):
    history[env_ids] = frame[:, None, :].expand(-1, history.shape[1], -1)


def bootstrap_episode_buffers(buffers, ids, frames, root, dof_velocity, rays, goal, positions):
    """Masked compact-policy reconstruction shared by both runtime bindings.

    Buffers use canonical names; optional physical baselines exist only on Gym.
    Runtime-specific caches and terminal-output ownership stay in the binding.
    """
    zero = ("command", "action", "filter", "episode_length", "goal_timer", "stay_timer",
            "collision", "previous_collision", "static")
    for key in zero:
        if key in buffers:
            buffers[key][ids] = 0
    if "initial" in buffers:
        buffers["initial"][ids] = True
    for key,frame in (("navigation_history",frames["navigation"]),("slr_history",frames["slr"]),
                      ("ray_history",rays),("goal_history",goal),("position_history",positions)):
        bootstrap_history(buffers[key],ids,frame)
    restored={"held_rays":rays,"held_goal":goal,"last_dof_velocity":dof_velocity,
              "last_root_velocity":root[:,7:13],
              "last_body_twist":torch.cat((frames["body_linear"],frames["body_angular"]),dim=-1)}
    for key,value in restored.items():
        if key in buffers:
            buffers[key][ids]=value


def advance_history(history, frame, initial_mask, env_ids=None):
    ids = torch.arange(len(history), device=history.device) if env_ids is None else env_ids
    initial = initial_mask[ids]
    bootstrap_history(history, ids[initial], frame[ids[initial]])
    running = ids[~initial]
    history[running, :-1] = history[running, 1:].clone()
    history[running, -1] = frame[running]


def inverse_quaternion_rotate(quaternion, vector, order="xyzw"):
    q = quaternion if order == "xyzw" else quaternion[:, [1,2,3,0]]
    xyz, w = q[:, :3], q[:, 3:4]
    return vector - 2 * w * torch.cross(xyz, vector, dim=-1) + 2 * torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)


def local_goal(root, targets, order="xyzw"):
    q = root[:, 3:7] if order == "xyzw" else root[:, [4,5,6,3]]
    yaw = torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]), 1-2*(q[:,1]**2+q[:,2]**2))
    delta = targets[:, :2] - root[:, :2]
    return torch.stack((torch.cos(yaw)*delta[:,0]+torch.sin(yaw)*delta[:,1],
                        -torch.sin(yaw)*delta[:,0]+torch.cos(yaw)*delta[:,1]), dim=-1)


def build_reset_frames(root, dof_pos, dof_vel, default_dof_pos, rays, goal, quaternion_order,
                       joint_order=None, angular_scale=.25, dof_position_scale=1., dof_velocity_scale=.05):
    """Noise-free synthetic step-0 frames; no physics, timers or RNG advanced."""
    body_linear = inverse_quaternion_rotate(root[:,3:7],root[:,7:10],quaternion_order)
    body_angular = inverse_quaternion_rotate(root[:,3:7],root[:,10:13],quaternion_order)
    gravity = inverse_quaternion_rotate(root[:,3:7],root.new_tensor([0.,0.,-1.]).expand(len(root),-1),quaternion_order)
    zero = torch.zeros_like(body_linear)
    pos = (dof_pos-default_dof_pos)*dof_position_scale
    vel = dof_vel*dof_velocity_scale
    if joint_order is not None:
        pos,vel=pos[:,joint_order],vel[:,joint_order]
    slr = torch.cat((body_angular*angular_scale,gravity,zero,pos,vel,torch.zeros_like(dof_pos)),dim=-1)
    navigation = torch.cat((gravity,zero,body_linear,body_angular,torch.log2(rays.clamp(.1,5)),goal),dim=-1)
    return {"body_linear":body_linear,"body_angular":body_angular,"gravity":gravity,"slr":slr,"navigation":navigation}
