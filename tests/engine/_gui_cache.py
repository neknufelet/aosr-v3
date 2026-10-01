"""網頁啟動身分按完整呼叫記憶；每個工人第一次仍執行真算法。"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import ParamSpec, TypeVar

import pytest
from pydantic import BaseModel

from aosr.reporting.physics_identity import physics_identity

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _serialize(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"身分呼叫的參數不能序列化：{type(value).__name__}")


def _memoize(function: Callable[_P, _T]) -> Callable[_P, _T]:
    signature = inspect.signature(function)
    remembered: dict[str, _T] = {}
    lock = Lock()

    @wraps(function)
    def memoized(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        serialized = json.dumps(arguments.arguments, default=_serialize, sort_keys=True)
        key = hashlib.sha256(serialized.encode()).hexdigest()
        with lock:
            if key not in remembered:
                remembered[key] = function(*args, **kwargs)
            return remembered[key]

    return memoized


_startup_identity = _memoize(physics_identity)


@pytest.fixture(autouse=True)
def gui_startup_identity_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """只換 app 的名字；題內補丁晚於夾具，照舊能蓋過記憶。"""
    monkeypatch.setattr("aosr.gui.app.physics_identity", _startup_identity)
