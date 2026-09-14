"""新家最底下那一層：跟機器講話的唯一一個地方。

規矩卡 ``layers-import-downward-only`` 的第二條把「碰機器設定」這件事收在這個模組裡：
JAX 的設定物件、``os.environ`` 的讀寫、``os.putenv``／``os.unsetenv``，以及
``JAX_``／``XLA_`` 開頭的環境變數名，別的模組寫出來就紅。

**為什麼要有這個模組。** v2 事故 ``import-time-global-flip-poisons-suite``：某一支模組
在**載入的時候**翻了一次 JAX 的全域設定，從此整套測試跑在另一組精度上，而紅綠上只看得到
「昨天會過今天不過」——沒有人指得出是哪一行改的。把那件事收進一個模組、一支函式，
改設定這件事就變成「誰叫了 :func:`configure_jax`」，看得見也追得到。

**這支函式不 import jax。** 平台與精度這兩件事 JAX 自己就吃環境變數，而且環境變數要在
JAX 第一次載入**之前**寫好才算數——先 ``import jax`` 再改設定，本身就是那筆事故的形狀。
所以這一層只寫環境變數，不碰函式庫；jax 進不進得了依賴清單是另一張票的事。

**預設不碰真的環境。** ``env`` 這一格交什麼就寫到哪裡，沒交才是行程自己的環境。
測試一律交一個普通的 dict 進來——測試不准改真環境（規矩卡 ``tests-isolated-from-real-env``）。
"""
from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

# JAX 讀這兩格決定「跑在哪個裝置上」與「浮點數要不要六十四位元」。
# 名字寫在這裡是這個模組的特權：別的模組寫出同樣的字串就紅。
PLATFORMS_VAR = "JAX_PLATFORMS"
ENABLE_X64_VAR = "JAX_ENABLE_X64"

# 環境變數只吃字串，布林要寫成這兩個字。
TRUE_TEXT = "true"
FALSE_TEXT = "false"


def preload_mkl(prefix: str | Path | None = None) -> Path:
    """從目前直譯器的 ``lib`` 目錄把 MKL 載入行程的全域符號表。

    pydiso 的編譯產物可能留著另一台機器的舊 RUNPATH；先從 ``sys.prefix/lib`` 載入
    ``libmkl_rt.so.3``，讓後續 import 不依賴外部環境變數。路徑缺席就直接失敗，刻意
    不搜尋其他目錄，也不退回另一套解法。

    :param prefix: 測試可注入的直譯器前綴；不交時使用目前行程的 ``sys.prefix``。
    :returns: 已載入的 MKL 動態函式庫路徑。
    :raises FileNotFoundError: 指定前綴下沒有必要的 MKL 動態函式庫。
    """
    chosen_prefix = Path(sys.prefix) if prefix is None else Path(prefix)
    library = chosen_prefix / "lib" / "libmkl_rt.so.3"
    if not library.is_file():
        raise FileNotFoundError(f"required MKL runtime library not found: {library}")
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    return library


def configure_jax(
    *,
    platforms: str,
    enable_x64: bool,
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """把 JAX 的平台與精度寫進環境，回傳這一次真的寫了哪幾格。

    :param platforms: 要 JAX 用哪個平台（``"cpu"``、``"cpu,cuda"`` 這種逗號清單）。
        空字串不准：那等於「隨便你挑」，而「隨便挑」正是同一份程式在兩台機器上算出
        兩個答案的來源。
    :param enable_x64: 要不要六十四位元浮點數。
    :param env: 要寫進哪一個環境。不交就是這個行程自己的環境
        （``os.environ``——碰它是這個模組的特權）。測試請交一個普通的 dict 進來。
    :returns: 這一次寫下去的那幾格（名字對值），照寫進去的樣子回。
    :raises ValueError: ``platforms`` 是空的或只有空白。

    這支函式**只寫環境變數、不 import jax**：那兩格要在 JAX 第一次載入之前寫好才算數，
    載入之後再改就是 v2 事故 ``import-time-global-flip-poisons-suite`` 的形狀。
    """
    chosen = platforms.strip()
    if not chosen:
        raise ValueError(
            "platforms 不准是空的——沒有指定平台就是讓函式庫當場挑一個，"
            "同一份程式在兩台機器上會算出兩個答案，而且從結果上看不出來"
        )
    target = os.environ if env is None else env
    written = {
        PLATFORMS_VAR: chosen,
        ENABLE_X64_VAR: TRUE_TEXT if enable_x64 else FALSE_TEXT,
    }
    for name, value in written.items():
        target[name] = value
    return written
