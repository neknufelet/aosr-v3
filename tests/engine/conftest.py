"""引擎考卷共用的兩個自動夾具：搜尋考卷的牆鐘凍住；物理身分與模態身分按完整呼叫記憶。

牆鐘：時間照樣走真的累加流程，但每段都是零秒，逐位狀態斷言不受實際時間影響。
時間的實際數值由 test_search_timings.py 的可控假時鐘驗證；這裡不關掉累加，免得上千題考卷都沒走到時間那條路。
只換時間模組自己的入口，不碰全域單調時鐘（子行程逾時還要靠它）。
"""

import sys
from collections.abc import Mapping
from functools import partial

import pytest

from aosr import runtime
from aosr.search import timings
from tests.engine._identity_memo import apply_identity_memo


@pytest.fixture(autouse=True)
def calculation_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """同程式考卷將正式入口的環境寫入普通字典，量身分時的 JAX 設定也讀那份字典；真子行程仍走原本 runtime。

    同程式的父行程（網頁、搜尋）要跟它開的正式子行程量到同一個身分，所以不管 pytest 怎麼啟動，同程式量到的都是正式設定。
    代價：同程式載入 JAX 的考卷實際算在 JAX 預設（雙精度關），跟身分記的不同；固定阻抗不走 JAX，材料走 JAX 後要用冷子行程驗（#309）。
    """
    entry_env: dict[str, str] = {}
    runtime.configure_calculation_jax(env=entry_env)
    monkeypatch.setattr(runtime, 'configure_calculation_jax', partial(runtime.configure_calculation_jax, env=entry_env))
    original = runtime.jax_settings
    # 已載入 JAX 時 jax_settings 先讀 JAX 本身；這裡要的是字典那份冷讀值，暫時藏起 JAX 只量這一次。
    with monkeypatch.context() as cold:
        cold.delitem(sys.modules, 'jax', raising=False)
        configured = original(env=entry_env)

    def settings(*, env: Mapping[str, str] | None = None) -> tuple[str | None, bool]:
        return configured if env is None else original(env=env)

    monkeypatch.setattr(runtime, 'jax_settings', settings)


@pytest.fixture(autouse=True)
def frozen_search_wall_clock(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.path.name.startswith("test_search_") and request.path.name != "test_search_timings.py":
        monkeypatch.setattr(timings, "_now", lambda: 0.0)


@pytest.fixture(autouse=True)
def identity_memo(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """同一個工人、同一組完整參數的物理身分與模態身分只真算一次（#586）；題內補丁晚於夾具，照舊蓋過記憶。

    每個工人第一次照舊真算，每一跑是新行程，產品突變照樣重新量；身分算法本身由考身分的三支真算守（名單在 _identity_memo）。
    """
    apply_identity_memo(request, monkeypatch)
