"""模組.函式叫法。"""
from __future__ import annotations

from aosr import runtime


def test_child() -> None:
    env = runtime.child_process_env()
    assert env
