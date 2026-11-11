"""網頁開的第三階段搜尋，在獨立服務裡真正跑的那一層（#1093）。

``python -m aosr.gui.gradient_exec <離開碼檔> <輸出檔> -- <命令列…>``：把命令列開成子行程（輸出與錯誤都寫進輸出檔）、
等它結束、把離開碼寫成一個小檔（先寫暫存檔再換名）。網頁只看這個小檔與服務還在不在，不記任何行程狀態在記憶體裡，
所以網頁重開也接得回來。

停止：服務照 ``KillMode=mixed`` 只把 SIGTERM 送給這一層；這一層轉一次給子行程（搜尋命令列會把它轉成中斷、
收好工作行程、回「已停止」），自己不先走，等子行程收完才照實寫離開碼。不對整組送：命令列收到兩次 SIGTERM
會在收尾一半時再被打斷。子行程被訊號殺掉時離開碼記成負的訊號號碼（照 ``subprocess`` 的慣例）。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import FrameType

from pydantic import BaseModel, ConfigDict

FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


class StepExit(BaseModel):
    """一步的結束紀錄：子行程的離開碼（開不起來是 None，原因在 ``error``）、收到過哪些訊號、結束時間（epoch 秒）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    exit_code: int | None
    error: str | None = None
    signals: tuple[int, ...] = ()
    finished_at: float


def write_exit(path: Path, record: StepExit) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(record.model_dump_json())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


POLL_S = 0.5


def run_step(record: Path, log: Path, command: list[str]) -> int:
    """處置只把訊號記下來；轉給子行程由主迴圈做、只做一次（處置裡送會在重入時送兩次）。"""
    received: list[int] = []

    def remember(signum: int, frame: FrameType | None) -> None:
        received.append(signum)

    for number in FORWARDED:
        signal.signal(number, remember)
    with log.open("ab") as stream:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        except OSError as error:
            write_exit(record, StepExit(exit_code=None, error=f"{type(error).__name__}：{error}", finished_at=time.time()))
            return 1
        forwarded = False
        while True:
            if received and not forwarded:
                process.send_signal(received[0])
                forwarded = True
            try:
                code = process.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                continue
    write_exit(record, StepExit(exit_code=code, signals=tuple(received), finished_at=time.time()))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        sys.stderr.write("用法：python -m aosr.gui.gradient_exec <離開碼檔> <輸出檔> -- <命令列…>\n")
        return 2
    return run_step(Path(argv[0]), Path(argv[1]), argv[3:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
