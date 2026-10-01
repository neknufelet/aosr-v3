"""後設測試看門狗：單份樣本卡住必須留下具名失敗，不能拖到整個 job 被取消。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_fixture_runner import _fake_card, _run


def test_watchdog_fails_with_card_sample_and_timeout(tmp_path: Path) -> None:
    """拿掉 subprocess.run 的 timeout，假檢查會跑完而不 fail，這一題必須紅。"""
    (tmp_path / "sleeping_check.py").write_text("import time\ntime.sleep(2)\n", encoding="utf-8")
    card = replace(_fake_card(tmp_path, scope=[]), check="sleeping_check.py")
    sample = tmp_path / "stuck-sample"
    sample.mkdir()
    timeout = 0.1

    with pytest.raises(pytest.fail.Exception) as caught:
        _run(card, sample, module_root=tmp_path, timeout=timeout)

    message = str(caught.value)
    assert card.id in message
    assert str(sample) in message
    assert str(timeout) in message
