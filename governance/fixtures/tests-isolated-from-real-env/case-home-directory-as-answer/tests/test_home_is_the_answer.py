"""樣本用的假測試：拿家目錄當答案。

壞在：``Path.home()`` 與 ``os.path.expanduser("~")`` 都是「跑在誰的機器上」的函式。
測試的答案不准取決於那個。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import os
from pathlib import Path


def test_config_is_installed():
    assert (Path.home() / ".config").is_dir()


def test_cache_is_installed():
    assert os.path.isdir(os.path.expanduser("~/.cache"))
