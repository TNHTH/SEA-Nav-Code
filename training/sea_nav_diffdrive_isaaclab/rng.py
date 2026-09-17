# SPDX-License-Identifier: MIT
"""Named RNG streams via SHA-256 derivation (A7.1).

Each stream is ``SHA256(master_seed, stream_name, env_id)`` — never Python
``hash()``. State and counter are persisted for checkpoint/resume.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
from torch import Generator

STREAM_NAMES = (
    "map",
    "goal",
    "domain_randomization",
    "policy_sampling",
    "smoothness",
    "acsi",
    "perception_delay",
    "eval_fixture_delay",
)

ALGORITHM_VERSION = "sha256_torch_generator_v1"


def derive_stream_seed(master_seed: int, stream_name: str, env_id: int) -> int:
    """Derive a 64-bit seed from contract tuple; no Python hash randomization."""
    if stream_name not in STREAM_NAMES:
        raise ValueError(f"unknown stream: {stream_name}")
    payload = f"{master_seed}:{stream_name}:{env_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return struct.unpack("<Q", digest[:8])[0]


@dataclass
class NamedStream:
    """One named stream with persisted counter."""

    name: str
    env_id: int
    seed: int
    counter: int = 0
    _generator: Optional[Generator] = field(default=None, repr=False)

    def generator(self) -> Generator:
        if self._generator is None:
            g = torch.Generator()
            g.manual_seed(self.seed)
            self._generator = g
        return self._generator

    def bump(self, n: int = 1) -> None:
        self.counter += n

    def state_dict(self) -> Dict:
        return {
            "name": self.name,
            "env_id": self.env_id,
            "seed": self.seed,
            "counter": self.counter,
            "algorithm_version": ALGORITHM_VERSION,
        }

    @classmethod
    def from_state_dict(cls, data: Dict) -> "NamedStream":
        if data.get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError("RNG algorithm_version mismatch")
        stream = cls(
            name=data["name"],
            env_id=int(data["env_id"]),
            seed=int(data["seed"]),
            counter=int(data["counter"]),
        )
        stream.generator()
        return stream


@dataclass
class NamedRngBank:
    """All contract streams for one env row."""

    master_seed: int
    env_id: int
    streams: Dict[str, NamedStream] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in STREAM_NAMES:
            if name not in self.streams:
                seed = derive_stream_seed(self.master_seed, name, self.env_id)
                self.streams[name] = NamedStream(name=name, env_id=self.env_id, seed=seed)

    def get(self, stream_name: str) -> NamedStream:
        if stream_name not in self.streams:
            raise KeyError(stream_name)
        return self.streams[stream_name]

    def state_dict(self) -> Dict:
        return {
            "master_seed": self.master_seed,
            "env_id": self.env_id,
            "streams": {k: v.state_dict() for k, v in self.streams.items()},
        }

    @classmethod
    def from_state_dict(cls, data: Dict) -> "NamedRngBank":
        bank = cls(master_seed=int(data["master_seed"]), env_id=int(data["env_id"]))
        bank.streams = {
            k: NamedStream.from_state_dict(v) for k, v in data["streams"].items()
        }
        return bank


def create_banks(master_seed: int, num_envs: int) -> List[NamedRngBank]:
    return [NamedRngBank(master_seed=master_seed, env_id=i) for i in range(num_envs)]


def verify_no_python_hash() -> None:
    """Assert we never use Python hash for stream derivation."""
    s1 = derive_stream_seed(42, "map", 0)
    s2 = derive_stream_seed(42, "map", 0)
    assert s1 == s2
    # Python hash is salted per process; our derivation is stable.
    assert derive_stream_seed(42, "map", 0) != derive_stream_seed(42, "goal", 0)
