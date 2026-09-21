"""全套考卷的數值函式庫必須單緒：拿掉 conftest 那一段，雲端會回到撞時間上限（票 #381）。"""

from __future__ import annotations

import os

from tests.conftest import NUMERIC_THREAD_ENVS


def test_numeric_libraries_are_limited_to_one_thread_per_worker() -> None:
    """每個工人行程裡這幾格都要有值；沒設就是函式庫各自開滿緒、跟別的工人搶 CPU。"""
    missing = [name for name in NUMERIC_THREAD_ENVS if not os.environ.get(name)]
    assert not missing, f"這幾格沒設，數值函式庫會各自開滿緒：{missing}"
