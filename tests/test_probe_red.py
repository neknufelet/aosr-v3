"""[勿合] #520 探針 A：故意失敗的考卷，驗「內文寫通過」蓋不掉「程式有錯」。"""

import pytest


def test_probe_must_stay_red() -> None:
    pytest.fail("#520 探針 A：這一題故意失敗，合併請求必須被擋")
