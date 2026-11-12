"""網頁開的第一、二階段搜尋（#700），在獨立服務裡真正跑的那一層：依序跑搜尋命令列的幾步（開搜尋 ``start``、自動外圈 ``auto``）。

``python -m aosr.gui.search_exec <鏈檔>``：鏈檔是一份 JSON（:class:`Chain`），列每一步的命令列與離開碼檔。每一步照
:func:`aosr.gui.gradient_exec.run_step` 開成子行程（輸出與錯誤都寫進同一個輸出檔）、等它結束、寫離開碼小檔；
上一步離開碼是 0、這一層沒收過訊號、搜尋資料夾裡也沒有停止記號，才接下一步（照主對話開 109 受控試跑那支腳本的順序：
``start`` 正常結束才接 ``auto``）。全部跑完（或停在某一步）再寫整條的離開碼檔（最後跑的那一步的紀錄）。

``auto`` 那一步要的搜尋資料夾在 ``start`` 建好之前不知道：``start`` 把新資料夾的代號寫進代號檔（``--id-file``），這一層讀它、
補在 ``auto`` 的命令列後面（``append_folder``）。接續那一條（只跑 ``auto``）資料夾已經知道，直接寫在鏈檔裡。

停止跟第三階段同一套：服務照 ``KillMode=mixed`` 只把 SIGTERM 送給這一層，這一層轉一次給正在跑的子行程、不先走，
等它收完照實寫離開碼；收過訊號就不再開下一步。
"""
from __future__ import annotations

import re
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from aosr.gui.gradient_exec import FORWARDED, StepExit, run_step, write_exit

StepName = Literal["start", "auto"]
# 搜尋資料夾裡的兩個停止記號與資料夾代號的樣子；照抄 src/aosr/search/store.py 與 src/aosr/gui/search_view.py（考卷核兩邊相同）。
# 不從那兩支載入：這一層要越早接管訊號越好，不多載一堆模組拉長「還沒接管就被 SIGTERM 打死」的空窗。
STOP_FILE: Final = "stop"
REFINE_STOP_FILE: Final = "refine-stop"
SEARCH_ID = re.compile(r"[0-9a-f]{32}\Z")


class ChainStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: StepName
    argv: tuple[str, ...] = Field(min_length=1)
    record: str
    append_folder: bool = False


class Chain(BaseModel):
    """``folder``：已知的搜尋資料夾（接續）；沒有就照 ``search_root``／``id_file`` 讀 ``start`` 寫的代號。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    log: str
    done: str
    steps: tuple[ChainStep, ...] = Field(min_length=1)
    folder: str | None = None
    search_root: str | None = None
    id_file: str | None = None


def chain_folder(chain: Chain) -> Path | None:
    """這一條的搜尋資料夾：鏈檔寫了就用；不然讀代號檔（還沒寫、或寫的不是一個名字就是還不知道）。"""
    if chain.folder is not None:
        return Path(chain.folder)
    if chain.search_root is None or chain.id_file is None:
        return None
    try:
        name = Path(chain.id_file).read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    path = Path(chain.search_root) / name
    return path if SEARCH_ID.fullmatch(name) and path.is_dir() and not path.is_symlink() else None


def _stop_marked(folder: Path | None) -> bool:
    return folder is not None and ((folder / STOP_FILE).exists() or (folder / REFINE_STOP_FILE).exists())


def run_chain(chain: Chain) -> int:
    """第一步開起來之前就先接管訊號（記進同一份），開跑那一瞬間收到的停止也不會讓這一層一聲不響地死掉。"""
    received: list[int] = []

    def remember(signum: int, frame: FrameType | None) -> None:
        received.append(signum)

    for number in FORWARDED:
        signal.signal(number, remember)
    last: StepExit | None = None
    for index, step in enumerate(chain.steps):
        if received:
            break  # 收過訊號（含第一步開之前）就不再開任何一步
        if index > 0:
            # 上一步沒正常結束、或有人放了停止記號：不開下一步。
            folder = chain_folder(chain)
            if last is None or last.exit_code != 0 or _stop_marked(folder):
                break
            if step.append_folder and folder is None:
                last = StepExit(exit_code=None, error=f"上一步沒有留下搜尋資料夾的代號，{step.name} 開不了",
                                finished_at=time.time())
                write_exit(Path(step.record), last)
                break
        argv = [*step.argv, str(chain_folder(chain))] if step.append_folder else list(step.argv)
        run_step(Path(step.record), Path(chain.log), argv, received=received)
        last = StepExit.model_validate_json(Path(step.record).read_text(encoding="utf-8"))
    if last is None:
        last = StepExit(exit_code=None, error="開第一步之前就收到停止訊號，這一條沒有開任何一步", finished_at=time.time())
    write_exit(Path(chain.done), last.model_copy(update={"signals": tuple(received), "finished_at": time.time()}))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("用法：python -m aosr.gui.search_exec <鏈檔>\n")
        return 2
    chain = Chain.model_validate_json(Path(argv[0]).read_text(encoding="utf-8"))
    return run_chain(chain)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
