"""探針模組。"""
from __future__ import annotations

from aosr.runtime import child_process_env


def probe_env() -> dict[str, str]:
    return child_process_env(threads=1)
