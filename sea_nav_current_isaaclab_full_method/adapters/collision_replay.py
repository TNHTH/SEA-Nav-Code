"""Compatibility boundary for the packaged compact replay implementation.

Legacy history arguments remain accepted but are deliberately not retained:
new_replay_episode_v1 starts fresh histories. Samples carry a reservation that
must be acknowledged or cancelled; they are not exact-continuation snapshots.
"""
import torch
from rsl_rl.replay import (
    CollisionReplayBuffer as _Ring, CollisionReplayConfig, ReplayTensorSpec,
    ReplayBatch, ReplaySelection, validate_replay_config, partition_reset_env_ids,
)


class CollisionReplayBuffer(_Ring):
    def __init__(self, config=None, num_envs=1, *, spec=None, device="cpu"):
        spec = spec or ReplayTensorSpec(
            quaternion_order="wxyz",
            geometry_fields=(("start_cell",2),("map_origin_cell",2),("goal_cell",2)))
        super().__init__(config,num_envs,spec=spec,device=device)

    def push(self, *, root_state, dof_pos, dof_vel, task_state, collision=False,
             env_ids=None, record_step_ids=None, command=None, sea_obs_hist=None, slr_obs_hist=None):
        ids = torch.arange(self.num_envs,device=self.device) if env_ids is None else torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        if isinstance(root_state,dict):
            root_state = torch.cat((root_state["root_pose"],root_state["root_velocity"]),dim=-1)
        def rows(value,width):
            return torch.as_tensor(value,device=self.device,dtype=torch.float32).reshape(len(ids),width)
        # Only explicitly versioned geometry is accepted; old arbitrary task
        # dictionaries must migrate, rather than be silently certified complete.
        if set(task_state) != {key for key,_ in self.spec.geometry_fields}:
            raise ValueError("legacy task_state must migrate to compact geometry fields")
        super().push(env_ids=ids, root_state=rows(root_state,13), dof_pos=rows(dof_pos,self.spec.dof_count),
                     dof_vel=rows(dof_vel,self.spec.dof_count),
                     task_state={key:rows(task_state[key],width) for key,width in self.spec.geometry_fields},
                     collision=collision,record_step_ids=record_step_ids)

    def sample_pre_collision(self,env_id=None,undo_steps=None,*,batched=False):
        ids = list(range(self.num_envs)) if env_id is None else [env_id] if isinstance(env_id,int) else env_id
        selection = self.reserve_pre_collision(ids,undo_steps=undo_steps)
        samples=[]
        for i, idx in enumerate(selection.env_ids.tolist()):
            mask = torch.arange(len(selection.env_ids),device=self.device)==i
            row = selection.subset(mask)
            fields = row.batch.fields
            samples.append({"env_id":idx,"step_index":int(row.batch.step_ids[0]),
                            "is_replay":True,"source_collision_step":int(row.collision_steps[0]),
                            "undo_steps":int(row.requested_undo[0]),"effective_undo":int(row.effective_undo[0]),
                            "short_history":bool(row.short_history[0]),"_selection":row,
                            "root_state":{"root_pose":fields["root_state"][0,:7],"root_velocity":fields["root_state"][0,7:]},
                            "dof_pos":fields["dof_pos"][0],"dof_vel":fields["dof_vel"][0],
                            "task_state":{key:fields[key][0] for key,_ in self.spec.geometry_fields}})
        if batched or env_id is not None and not isinstance(env_id,int):
            return samples
        return samples[0] if samples else None

    @property
    def latest_collision_step(self):
        values=[None if value<0 else value for value in self.collision_onset.tolist()]
        return values[0] if self.num_envs==1 else values
