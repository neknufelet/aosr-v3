"""把 GitHub 的回應錄一次、重播給後設測試——只給後設測試，雲端那一跑不走這裡。

**為什麼要有這一層。** 兩張准上網的卡（``merge-gate-read-back`` 與
``issues-closed-only-by-merged-pr``）的後設測試每張跑六回合，其中三回合（乾淨樹、必紅樣本、
控制樣本）真的會叫 ``gh``（GitHub 的命令列工具）去問伺服器；再加上「產收據那一跑」把整套
pytest 又跑一遍，同一句問題一次 ``uv run pytest`` 會問到十幾次。問到的答案在同一跑裡本來就
是同一份，多問的那幾次只換來牆上時間與 API 額度。

**判準一點都沒放寬。** 這一層只省掉網路來回，不省「工具在不在」：重播之前一定先確認
``argv[0]`` 真的在 ``PATH`` 上（:func:`require_tool`），所以後設測試第 4 回合（把卡宣告的外部
工具從 ``PATH`` 抽掉，必須回 2）在有快取的時候一樣回 2。抽掉工具還能回綠的快取就是放水，
那正是這個 repo 最恨的形狀。

**只有後設測試會走這裡。** 開關是環境變數 ``AOSR_GH_REPLAY_DIR``（指向一個放錄音的暫存
目錄），只有 ``tests/conftest.py`` 在開跑前設它；CI 上跑檢查那幾步沒有這一格，所以雲端那一跑
每一句都真的去問伺服器。「沒設這一格就真的會上網」由 ``tests/test_gh_replay.py`` 用一支假的
``gh`` 外殼（shim，冒充那支工具的小腳本）證明，不是靠這段話宣稱。

錄音是「一句問題一個檔」：檔名是 argv（那一句指令的完整參數陣列）的 sha256，內容是那一次的
stdout。寫檔走「先寫暫存檔再原子換名」，因為 xdist（pytest 的平行外掛）底下會有好幾個工人
（worker，平行跑測試的子程序）同時錄同一句——換名是原子的，讀到的永遠是完整的一份。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from governance.exit_codes import ToolBroken, note

# 錄音目錄的環境變數。只有後設測試設它；雲端那一跑身上沒有這一格。
REPLAY_DIR_ENV = "AOSR_GH_REPLAY_DIR"
# 錄音檔的副檔名。錄的是那一次的 stdout，不是判決。
RECORD_SUFFIX = ".stdout"


def replay_dir() -> Path | None:
    """這一跑的錄音目錄；沒設環境變數、或指到的地方不是目錄，就是「不走快取」。"""
    where = os.environ.get(REPLAY_DIR_ENV, "").strip()
    if not where:
        return None
    root = Path(where)
    return root if root.is_dir() else None


def _record_path(root: Path, argv: Sequence[str]) -> Path:
    """一句指令的錄音檔。鍵是整個 argv 的 sha256——參數差一個字就是另一句問題。"""
    blob = "\x00".join(str(part) for part in argv).encode("utf-8")
    return (root / hashlib.sha256(blob).hexdigest()).with_suffix(RECORD_SUFFIX)


def require_tool(tool: str, what: str) -> None:
    """走快取也要先確認那支工具真的在 PATH 上——抽掉工具還能回綠的快取就是放水。"""
    if shutil.which(tool) is None:
        raise ToolBroken(f"{tool} 不在 PATH 上（{what}）——讀不到伺服器就不出結論")


def replay(argv: Sequence[str], what: str) -> str | None:
    """這一句問題有沒有錄過。沒開快取、沒錄過，一律 None（那就真的去問伺服器）。"""
    root = replay_dir()
    if root is None or not argv:
        return None
    recorded = _record_path(root, argv)
    if not recorded.is_file():
        return None
    require_tool(str(argv[0]), what)
    note(f"這一句走的是這一跑錄下來的回應（{what}）——只有後設測試會走這條路")
    return recorded.read_text(encoding="utf-8")


def record(argv: Sequence[str], text: str) -> None:
    """把這一次真的問到的回應錄起來。沒開快取就什麼都不做。"""
    root = replay_dir()
    if root is None or not argv:
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as handle:
        handle.write(text)
    Path(handle.name).replace(_record_path(root, argv))
