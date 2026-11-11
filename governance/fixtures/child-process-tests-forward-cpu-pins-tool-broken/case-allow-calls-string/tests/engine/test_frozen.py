"""凍結的檔。"""
from __future__ import annotations

from aosr.runtime import child_process_env


def test_child() -> None:
    env = child_process_env(threads=1)
    assert env
