"""方案考卷共用的重夾具：同一次 pytest 裡只算一次，平行工人之間用檔案加檔案鎖共用。

照 pytest-xdist 文件「session 級夾具只執行一次」的做法：各工人的暫存目錄都在這一次 pytest
的共同暫存根底下，共用檔就放在那裡。每一次 pytest 都開新的暫存根（給 --basetemp 時開跑也會清空），
所以故意改錯程式的那一跑一定自己重算，讀不到上一跑的結果。只存被測段落上游的東西：
控制組手拼答案，以及管線跑出的完整結果（給下游的比較、存讀與命令列考卷用）。
"""

from __future__ import annotations

import fcntl
from collections.abc import Callable

import pytest


def shared_json(tmp_path_factory: pytest.TempPathFactory, worker_id: str, name: str,
                produce: Callable[[], str]) -> str:
    """回傳這一次 pytest 裡名叫 name 的那份 JSON；還沒有就由第一個拿到鎖的工人算。"""
    if worker_id == "master":
        return produce()
    root = tmp_path_factory.getbasetemp().parent
    path = root / f"{name}.json"
    with (root / f"{name}.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            return path.read_text(encoding="utf-8")
        text = produce()
        path.write_text(text, encoding="utf-8")
        return text
