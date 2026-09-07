"""Validated four-stage JSONL trace and closed-file physical row evidence."""
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Union

ACTION_STAGES = ("distribution_mean","policy_action","clipped_policy_action","executed_command")
REQUIRED_TRACE_FIELDS = ("step","env_id") + ACTION_STAGES

def validate_trace_record(record: Dict[str, Any], required: Iterable[str] = REQUIRED_TRACE_FIELDS) -> None:
    missing=[field for field in required if field not in record]
    if missing: raise ValueError("trace record missing fields: " + repr(missing))
    for field in ACTION_STAGES:
        value=record[field]
        if not isinstance(value,(list,tuple)) or len(value)!=3 or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in value):
            raise ValueError("trace action must contain three finite numbers: "+field)
    for field in ("step","env_id"):
        if type(record[field]) is not int or record[field]<0:
            raise ValueError("trace "+field+" must be a nonnegative integer")

def count_physical_jsonl_rows(path):
    count=0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): raise ValueError("blank physical trace row")
            record=json.loads(line)
            validate_trace_record(record)
            count+=1
    return count

class JsonlTraceLogger:
    def __init__(self,path: Union[str,Path]):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._handle=self.path.open("x",encoding="utf-8")
        self.row_count=0
    def write(self,record):
        validate_trace_record(record)
        encoded=json.dumps(record,ensure_ascii=False,allow_nan=False)
        self._handle.write(encoded+"\n")
        self._handle.flush()
        self.row_count+=1
    def close(self):
        self._handle.close()
    def verified_row_count(self):
        if not self._handle.closed: raise ValueError("trace must be closed before physical count binding")
        count=count_physical_jsonl_rows(self.path)
        if count!=self.row_count: raise ValueError("physical trace row count mismatch")
        return count
    def __enter__(self): return self
    def __exit__(self,exc_type,exc,tb): self.close()

def capture_policy_stages(actor, policy_action):
    """Read the collection-owned actor state; never rerun a policy forward."""
    return {"distribution_mean":actor.action_mean.detach().clone(),
            "policy_action":policy_action.detach().clone()}

def write_transition(trace, step, stages, reward, done, perception, observation_time):
    """Explicit trace boundary; conversions are disabled when no trace is active."""
    if trace is None:
        return
    for env_id in range(reward.shape[0]):
        row={name:value[env_id].detach().cpu().tolist() for name,value in stages.items()}
        row.update(step=int(step),env_id=env_id,reward=float(reward[env_id]),done=bool(done[env_id]),
                   observation_timestamp=float(observation_time[env_id]),
                   sample_timestamp=float(perception.held_time[env_id]),
                   sampled_latency=float(perception.held_latency[env_id]),
                   actual_sample_age=float(observation_time[env_id]-perception.held_time[env_id]),
                   synthetic_bootstrap=bool(perception.synthetic[env_id]))
        trace.write(row)
