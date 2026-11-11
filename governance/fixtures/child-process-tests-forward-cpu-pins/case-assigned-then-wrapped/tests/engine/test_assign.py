"""先指給名字再傳。"""
from __future__ import annotations

from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins


def test_child() -> None:
    env = child_process_env()
    pinned = forward_cpu_pins(env)
    assert pinned
