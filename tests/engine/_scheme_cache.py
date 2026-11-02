"""方案考卷共用的重夾具：同一次 pytest 裡只算一次，平行工人之間用檔案加檔案鎖共用。

照 pytest-xdist 文件「session 級夾具只執行一次」的做法：各工人的暫存目錄都在這一次 pytest
的共同暫存根底下，共用檔就放在那裡。每一次 pytest 都開新的暫存根（給 --basetemp 時開跑也會清空），
所以故意改錯程式的那一跑一定自己重算，讀不到上一跑的結果。只存被測段落上游的東西：
控制組手拼答案，以及管線跑出的完整結果（給下游的比較、存讀與命令列考卷用）。
"""

from __future__ import annotations

import dataclasses
import fcntl
import struct
from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import pytest
from pydantic import BaseModel


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


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def bitwise_equal(left: object, right: object) -> bool:
    """共用存檔的讀回檢查用：逐位相等，不是值相等（#996 審查：``==`` 把 +0.0 與 −0.0 判成相等）。

    兩邊型別要一樣；浮點比 IEEE 754 雙精度的八個位元組（``float.hex()`` 分不出不同的 NaN，#996 第二輪審查）、
    複數比實部與虛部各自的八個位元組、numpy 陣列比 dtype／形狀／位元組；dataclass 與 pydantic 模型逐欄、
    對照表逐鍵而且**鍵的順序也要一樣**（產品會讀順序，例如 ``next(iter(scheme.speakers))``）、tuple／list 逐項遞迴；
    其餘（字串、整數、布林、列舉、日期、None）同型而且 ``==``。這裡不為任何一處放寬：順序確定沒有語意的那一處
    由呼叫端自己明講（見 ``_fem_reduced_cache.shared_model``）。
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return _bits(left) == _bits(cast(float, right))
    if isinstance(left, complex):
        other = cast(complex, right)
        return _bits(left.real) == _bits(other.real) and _bits(left.imag) == _bits(other.imag)
    if isinstance(left, np.ndarray):
        array = cast(np.ndarray, right)
        return left.dtype == array.dtype and left.shape == array.shape and left.tobytes() == array.tobytes()
    if isinstance(left, BaseModel):
        return all(bitwise_equal(getattr(left, name), getattr(right, name)) for name in type(left).model_fields)
    if dataclasses.is_dataclass(left):
        return all(bitwise_equal(getattr(left, field.name), getattr(right, field.name))
                   for field in dataclasses.fields(left))
    if isinstance(left, Mapping):
        table = cast(Mapping[object, object], right)
        return list(left) == list(table) and all(bitwise_equal(left[key], table[key]) for key in left)
    if isinstance(left, tuple | list):
        items = cast(tuple[object, ...] | list[object], right)
        return len(left) == len(items) and all(bitwise_equal(a, b) for a, b in zip(left, items, strict=True))
    return bool(left == right)
