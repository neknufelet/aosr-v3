"""樣本道具：一個不寫理由的 skip 標記。

標記那一行的行尾註解只給了到期日，沒有 reason 那一格——所以看不出這支測試為什麼被關掉。
"""
from __future__ import annotations

import pytest


@pytest.mark.skip  # expires=2099-12-31
def test_we_stopped_running_this_one() -> None:
    assert True
