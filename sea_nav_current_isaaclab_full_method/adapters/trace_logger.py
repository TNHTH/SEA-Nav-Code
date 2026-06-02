from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


REQUIRED_TRACE_FIELDS = (
    "step",
    "u_nominal",
    "alpha",
    "h_min",
    "lse_h",
    "shield_delta",
    "u_safe",
    "filtered_command",
    "is_replay",
    "reward",
    "done",
    "collision",
)


def validate_trace_record(record: Dict[str, Any], required: Iterable[str] = REQUIRED_TRACE_FIELDS) -> None:
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"trace record missing fields: {missing}")


class JsonlTraceLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        validate_trace_record(record)
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlTraceLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
