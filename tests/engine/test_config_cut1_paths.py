"""``aosr.config.paths``：設定檔路徑的唯一住處（這一刀**新蓋**的一支）。

這一支沒有 donor 可以對——它是新寫的，不是從上一代搬的。它的檔頭自己指名了一個安靜的
錯法：上一代用「從自己往上數兩層」算設定檔目錄，新家的檔案住得比較深，照抄只會到
``src``，而那要等**開檔那一刻**才炸。所以這裡的裁判不是「跟 donor 比」，是**把那個錯法
釘成三條會紅的斷言**：

1. 目錄就是**本套件**底下的 ``data``（不是往上數出來的別人的目錄）；
2. ``config_path(name)`` 就是那個目錄底下的那個名字（不是別的地方、也不是把名字吞掉）；
3. ``CONFIG_DIR.parent`` 的名字是 ``config``——整條路徑的**形狀**（深度少一層會讓這一條紅）。

這三條一起把「路徑指到別的地方」與「深度少一層」兩種錯法蓋住；它蓋不到的是
「``data/`` 底下真的有那幾個檔案」——那 9 個 `.toml` 是第 2 刀的事。
"""
from __future__ import annotations

from pathlib import Path

import aosr.config as config_package
from aosr.config import paths


def test_config_dir_is_this_package_data_directory() -> None:
    """``CONFIG_DIR`` 是本套件（``aosr.config``）底下的 ``data``。"""
    package_dir = Path(config_package.__file__).resolve().parent
    assert paths.CONFIG_DIR == package_dir / "data"


def test_config_dir_shape_is_one_level_under_config() -> None:
    """``CONFIG_DIR.parent`` 的名字是 ``config``——深度少一層那個錯法在這裡紅。

    上一代是「``lib/config`` 底下往上數兩層」（檔案住得淺），新家照抄那個數法會到
    ``src``；這一條量的是路徑的形狀，不量那個數法本身，所以換寫法但指對地方照樣綠。
    """
    assert paths.CONFIG_DIR.parent.name == "config"
    assert paths.CONFIG_DIR.name == "data"
    assert paths.CONFIG_DIR.parent.parent.name == "aosr"


def test_config_path_is_the_name_under_config_dir() -> None:
    """``config_path(name)`` 就是 ``CONFIG_DIR`` 底下那個名字。"""
    assert paths.config_path("perceptual.toml") == paths.CONFIG_DIR / "perceptual.toml"
    assert paths.config_path("x").name == "x"
    sub = "/".join(["a", "b.toml"])
    assert paths.config_path(sub).parent == paths.CONFIG_DIR / "a"


def test_config_path_only_computes_it_does_not_check_existence() -> None:
    """只算路徑、不檢查它在不在（找不到檔要炸在開檔那一刻，不是在這裡安靜地回替代品）。"""
    missing = paths.config_path("definitely-not-there.toml")
    assert missing.name == "definitely-not-there.toml"
    assert not missing.exists()
