"""包在別的東西裡。"""
from __future__ import annotations

from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins


def test_child() -> None:
    env = forward_cpu_pins(dict(child_process_env()))
    assert env
