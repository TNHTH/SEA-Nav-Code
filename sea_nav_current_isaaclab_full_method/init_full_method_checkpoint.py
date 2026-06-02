#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--metadata-out", default="")
args = parser.parse_args()

SEA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEA_ROOT / "training/rsl_rl"))
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic


def main() -> dict:
    torch.manual_seed(args.seed)
    model = DifferentiableSafeActorCritic(
        num_actions=3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.5,
        num_props=12,
        num_rays=41,
        cbf_fov_deg=240.0,
        his_len=10,
    )
    created_at = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S%z")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "route_id": "sea_nav_full_method_current_isaaclab_adapter_20260517_1649",
        "checkpoint_type": "initialized_full_method_actor_critic",
        "created_at": created_at,
        "source_repo": str(SEA_ROOT),
        "source_commit": "fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96",
        "seed": args.seed,
        "model_state_dict": model.state_dict(),
        "contract": {
            "num_actions": 3,
            "num_props": 12,
            "num_rays": 41,
            "cbf_fov_deg": 240.0,
            "his_len": 10,
            "num_obs_one_step": 55,
            "num_observations": 550,
            "actor": "DifferentiableSafeActorCritic",
            "cbf": "ExactLSECBFLayer via actor graph with 240.0-degree FOV; runtime adapter also defaults to 240.0-degree CBF wrapper",
        },
    }
    torch.save(payload, output)
    metadata = {
        key: value
        for key, value in payload.items()
        if key != "model_state_dict"
    }
    metadata["checkpoint"] = str(output)
    metadata["parameter_tensors"] = len(model.state_dict())
    metadata["parameter_count"] = int(sum(t.numel() for t in model.state_dict().values()))
    if args.metadata_out:
        metadata_path = Path(args.metadata_out)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
