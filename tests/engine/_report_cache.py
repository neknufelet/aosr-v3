"""同一次 pytest 的報表上游資料；xdist 工人共用暫存根與檔案鎖。"""

from __future__ import annotations

import fcntl
import pickle
from collections.abc import Callable
from typing import TypeVar, cast

import pytest


_T = TypeVar("_T")


def shared_report(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str, name: str,
    produce: Callable[[], _T],
) -> _T:
    """同一跑只求解一次；每跑的新暫存根讓產品突變必須重新求解。"""
    if worker_id == "master":
        return produce()
    root = tmp_path_factory.getbasetemp().parent
    path = root / f"{name}.pickle"
    with (root / f"{name}.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            return cast(_T, pickle.loads(path.read_bytes()))
        result = produce()
        path.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
        return result
