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

import os
import subprocess
import sys
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


def test_config_dir_does_not_depend_on_the_working_directory(tmp_path: Path) -> None:
    """**換一個 cwd 也指到同一個地方**——「靠 cwd」那一種錯法在這裡紅（總驗收的修 3）。

    上面三條釘的是路徑的**形狀**，而形狀那一組接不住一種錯法：把 ``CONFIG_DIR`` 改成
    ``Path.cwd() / "src" / "aosr" / "config" / "data"``。在 CI 裡 cwd 就是 repo 根，
    所以那三條照樣過——但換一個 cwd 跑就找不到檔，而 `paths.py` 的檔頭自己就寫了
    「錯法很安靜」。

    做法：在**子程序**裡把 cwd 設到別的地方、載入 ``aosr.config.paths``，比它的
    ``CONFIG_DIR`` 跟這一支行程裡算出來的一不一樣。子程序的 ``PYTHONPATH`` 由
    **這個模組實際從哪裡被載入**推出來（不是寫死 repo 路徑），所以把整個套件換成突變
    版本跑（``-o pythonpath=.:<突變樹>``）時，子程序載到的也是突變版本。
    """
    package_src = Path(paths.__file__).resolve().parents[2]  # <src>（paths 模組往上三層）
    probe = tmp_path / "probe_paths_cwd.py"
    probe.write_text(
        "import aosr.config.paths as paths\n"
        "print(paths.CONFIG_DIR)\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_src)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(probe)],
        cwd=tmp_path,  # 刻意不是 repo 根
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"子行程載入 paths 失敗：{proc.stderr.strip()[:300]}"
    assert Path(proc.stdout.strip()) == paths.CONFIG_DIR, (
        "換一個 cwd 之後 CONFIG_DIR 就不一樣了——它靠 cwd（`Path.cwd()/…`）算出來的\n"
        f"  這個 cwd：{paths.CONFIG_DIR}\n"
        f"  別的 cwd：{proc.stdout.strip()}"
    )
    assert paths.CONFIG_DIR.is_dir(), f"{paths.CONFIG_DIR} 不是目錄——指到別的地方了"
