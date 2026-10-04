"""既有搜尋考卷把牆鐘凍住：時間照樣走真的累加流程，但每段都是零秒，逐位狀態斷言不受實際時間影響。

時間的實際數值由 test_search_timings.py 的可控假時鐘驗證；這裡不關掉累加，免得上千題考卷都沒走到時間那條路。
"""

import pytest

from aosr.search import timings


@pytest.fixture(autouse=True)
def frozen_search_wall_clock(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.path.name.startswith("test_search_") and request.path.name != "test_search_timings.py":
        monkeypatch.setattr(timings.time, "monotonic", lambda: 0.0)
