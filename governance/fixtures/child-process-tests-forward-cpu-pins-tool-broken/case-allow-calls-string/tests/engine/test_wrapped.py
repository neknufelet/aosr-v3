"""已經包好的考卷：這一支不該被咬。"""
from __future__ import annotations

from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins


def test_child() -> None:
    env = forward_cpu_pins(child_process_env(threads=1))
    assert env
