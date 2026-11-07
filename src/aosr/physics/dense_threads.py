"""替身的稠密代數區段暫借 BLAS 緒數，離開時還原，完整解不使用。

Linux 從 sysfs 的核心兄弟清單去重，只數親和性允許的核心；讀不到拓撲時退回
可用邏輯緒數，連邏輯緒數也查不到時用一緒。限制作用於整個行程，不適合並行巢入。
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol


class _ThreadpoolAPI(Protocol):
    """threadpoolctl 3.6.0 本票使用的介面；上游沒有型別宣告。"""

    def threadpool_limits(self, *, limits: int, user_api: Literal["blas"]) -> AbstractContextManager[object]: ...


if TYPE_CHECKING:
    threadpoolctl: _ThreadpoolAPI
else:
    import threadpoolctl

_CPU_SYSFS = Path("/sys/devices/system/cpu")


def _core_siblings(cpu: int) -> frozenset[int]:
    """讀 Linux CPU 清單（逗號與閉區間），新版欄位不在時讀舊版欄位。"""
    topology = _CPU_SYSFS / f"cpu{cpu}" / "topology"
    for name in ("core_cpus_list", "thread_siblings_list"):
        try:
            siblings: set[int] = set()
            for part in (topology / name).read_text(encoding="ascii").strip().split(","):
                ends = [int(value) for value in part.split("-")]
                if len(ends) > 2 or ends[0] < 0 or ends[-1] < ends[0]:
                    raise ValueError("無效的核心兄弟清單")
                siblings.update(range(ends[0], ends[-1] + 1))
            if cpu in siblings:
                return frozenset(siblings)
        except (OSError, ValueError):
            continue
    raise ValueError("讀不到核心兄弟清單")


def default_dense_threads() -> int:
    """可用實體核數；拓撲不明時保守回報可用邏輯緒數，不猜超執行緒倍率。"""
    if hasattr(os, "sched_getaffinity"):
        try:
            allowed = os.sched_getaffinity(0)
        except OSError:
            return os.cpu_count() or 1
        try:
            return len({_core_siblings(cpu) & allowed for cpu in allowed}) or 1
        except ValueError:
            return len(allowed) or 1
    return os.cpu_count() or 1


def checked_dense_threads(n: int | None) -> int:
    """None 用本機預設；明確指定只接受正整數（布林值也拒收）。"""
    if n is None:
        return default_dense_threads()
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("dense_threads 必須是 None 或正整數")
    return n


@contextmanager
def dense_blas_threads(n: int | None = None) -> Iterator[None]:
    """只在區塊內改 BLAS 緒數，正常或例外離開都還原；預設為可用實體核數。"""
    with threadpoolctl.threadpool_limits(limits=checked_dense_threads(n), user_api="blas"):
        yield
