"""取別名再叫。"""
from __future__ import annotations

from aosr.runtime import child_process_env as make_env


def test_child() -> None:
    env = make_env(threads=1)
    assert env
