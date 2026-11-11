"""凍結的檔又多寫了一處。"""
from __future__ import annotations

from aosr.runtime import child_process_env


def test_child() -> None:
    first = child_process_env(threads=1)
    second = child_process_env()
    assert first and second
