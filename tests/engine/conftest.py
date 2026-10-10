"""引擎考卷共用的兩個自動夾具：搜尋考卷的牆鐘凍住；物理身分與模態身分按完整呼叫記憶。

牆鐘：時間照樣走真的累加流程，但每段都是零秒，逐位狀態斷言不受實際時間影響。
時間的實際數值由 test_search_timings.py 的可控假時鐘驗證；這裡不關掉累加，免得上千題考卷都沒走到時間那條路。
只換時間模組自己的入口，不碰全域單調時鐘（子行程逾時還要靠它）。
"""

import pytest

from aosr.search import timings
from tests.engine._identity_memo import apply_identity_memo


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
