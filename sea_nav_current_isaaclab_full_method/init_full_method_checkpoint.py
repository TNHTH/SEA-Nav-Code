#!/usr/bin/env python3
"""CPU model-only initialization; this does not certify a simulator runtime."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

SEA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEA_ROOT / "training/rsl_rl"))


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-stack", choices=("isaaclab_adapter", "isaac_gym_preview4"), required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    import re
    import yaml
    from rsl_rl.experiment_config import resolve_run_config
    from rsl_rl.runtime_preflight import _canonical
    if not re.fullmatch(r"[0-9a-f]{40}", args.producer_commit):
        raise ValueError("producer_commit must be an explicit full lowercase commit SHA")
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("seed must be in uint32 range")
    config_path = _canonical(args.config)
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict) or set(document) != {"schema_version", "algorithm_profile", "implementation_delta", "runtime"} or type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("runtime YAML schema/unknown fields")
    resolved = resolve_run_config(registry_path=SEA_ROOT / "configs/parity_registry.yaml",
        algorithm_profile=document["algorithm_profile"], runtime_stack=args.runtime_stack,
        implementation_delta=document["implementation_delta"])
    output = _canonical(args.output_manifest)
    metadata_path = output.with_name(output.name + ".metadata.json")
    for target in (output, metadata_path):
        if target.exists() or target == config_path or target in config_path.parents:
            raise ValueError("initialization output already exists or conflicts with input")
    import torch
    from rsl_rl.policy_factory import build_actor_critic
    from rsl_rl.utils.checkpoint import save_checkpoint_v2
    torch.manual_seed(args.seed)
    model = build_actor_critic(resolved, num_actions=3, num_props=12,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[512, 256, 128], activation="elu", init_noise_std=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = save_checkpoint_v2(output, model_state_dict=model.state_dict(), iteration=0,
        producer_commit=args.producer_commit, resolved_config_sha256=resolved.resolved_sha256)
    metadata = dict(checkpoint_manifest=str(output), producer_commit=args.producer_commit,
        resolved_config_sha256=resolved.resolved_sha256, runtime_stack=args.runtime_stack,
        algorithm_profile=resolved.identity.algorithm_profile,
        implementation_delta=list(resolved.identity.implementation_delta), seed=args.seed,
        upstream_commit="fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96",
        checkpoint=asdict(manifest), runtime_verified=False,
        scope="model-only initialization at zero completed PPO updates; no simulator evidence")
    with metadata_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
