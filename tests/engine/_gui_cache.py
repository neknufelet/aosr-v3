"""網頁啟動身分與畫面上游讀回按完整呼叫記憶；每個工人第一次仍執行真算法。"""
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
from aosr.reporting.evaluation import load_result

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _serialize(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"身分呼叫的參數不能序列化：{type(value).__name__}")


def _memoize(function: Callable[_P, _T], *,
             serializer: Callable[[object], object] = _serialize) -> Callable[_P, _T]:
    signature = inspect.signature(function)
    remembered: dict[str, _T] = {}
    lock = Lock()

    @wraps(function)
    def memoized(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        serialized = json.dumps(arguments.arguments, default=serializer, sort_keys=True)
        key = hashlib.sha256(serialized.encode()).hexdigest()
        with lock:
            if key not in remembered:
                remembered[key] = function(*args, **kwargs)
            return remembered[key]

    return memoized


_startup_identity = _memoize(physics_identity)


def _result_argument(value: object) -> object:
    if isinstance(value, Path):
        # 結果與品質登記簿都按路徑及內容分鍵；OSError 原樣冒出，命中記憶也不能藏掉讀不到檔。
        return {"path": str(value), "sha256": hashlib.sha256(value.read_bytes()).hexdigest()}
    return _serialize(value)


_loaded_result = _memoize(load_result, serializer=_result_argument)


@pytest.fixture(autouse=True)
def gui_load_result_memo(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """只供瀏覽器與畫面考卷；讀回分級考卷不掛，題內 load_result 替身仍能蓋過。

    重驗、重評與分級由 test_result_standing、test_result_reload_compatibility 及三支不掛記憶的 GUI 題守。
    """
    filename = request.node.path.name
    if filename == "test_gui_result_standing_browser.py" or not (
        filename == "test_gui_browser.py" or filename.startswith("test_gui_")
        and filename.endswith(("_browser.py", "_visual.py"))
    ):
        return
    monkeypatch.setattr("aosr.gui.app.load_result", _loaded_result)


@pytest.fixture(autouse=True)
def gui_startup_identity_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """只換 app 的名字；題內補丁晚於夾具，照舊能蓋過記憶。"""
    monkeypatch.setattr("aosr.gui.app.physics_identity", _startup_identity)
