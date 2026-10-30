"""牆面材料梯度搜尋的雙精度子行程（#561 辛一、辛三）：``python -m aosr.gradient.worker <資料夾> start|resume|endpoint``。

由 ``aosr.gradient.cli`` 起：環境是 ``child_process_env(threads=1)`` 再寫中央處理器＋雙精度，``start_new_session``，
資料夾鎖的描述子經 ``pass_fds`` 繼承（父行程不解鎖、子行程活著鎖就在）。這裡先核 JAX 真的是中央處理器＋雙精度，
不是就失敗、不算。狀態檔只由這裡寫：開始寫 running（帶自己的行程編號），收尾寫 finished／stopped／failed／interrupted；
進度數字（帳本記的累計求值次數、做完幾輪、最後一列的值；庚三多兩欄：換樣板帳本記的累計求值次數與做完幾輪）在寫的那一刻
從帳本讀。

起它的父行程建子行程時擋著 SIGINT、SIGTERM、SIGHUP（遮罩會繼承），所以入口裝好轉換就把這三個放開；
擋著那段收到的訊號放開時才送到，照樣轉成中斷。

離開碼：0 跑完（也寫 ``result.json``）、1 失敗（錯誤原文寫標準錯誤、也記進狀態檔）、3 停止記號或中斷（SIGTERM、SIGHUP
在 ``__main__`` 轉成 KeyboardInterrupt）。停止記號在每兩次求值之間查；帳本停在最後一列寫完整的地方。

``endpoint`` 模式（辛三）做終點（``endpoint_run.run_endpoint``），不寫狀態檔（終點不是搜尋狀態機的一段）、不看停止記號：
0 寫完 ``endpoint.json``、1 失敗（錯誤原文寫標準錯誤）、3 中斷。
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Final, Literal

from aosr import runtime
from aosr.gradient.endpoint_run import run_endpoint
from aosr.gradient.material_gradient import SearchStopped, StopCheck
from aosr.gradient.material_gradient_ledger import read_ledger
from aosr.gradient.run import (
    LEDGER_FILE, STATUS_FILE, GradientStatus, State, ledger_progress, now_text, resume_search, run_search,
    stop_requested, template_ledger_progress, write_json_atomic,
)

Mode = Literal["start", "resume"]
MODES: Final[tuple[Mode, ...]] = ("start", "resume")
ENDPOINT_MODE: Final = "endpoint"
EXIT_DONE: Final = 0
EXIT_FAILED: Final = 1
EXIT_STOPPED: Final = 3


def _status(folder: Path, state: State, started_at: str, message: str | None = None) -> None:
    """寫狀態檔；進度從兩本帳讀（沒有帳本或讀不開時是零、None——帳本壞了由計算本身報錯）。

    ``stop_reason`` 照舊是庚二迴圈收尾的停止原因；換樣板那段的進度是另兩欄（停止原因在 ``result.json``）。"""
    try:
        _, rows = read_ledger(folder / LEDGER_FILE)
    except (OSError, ValueError):
        rows = ()
    evaluations, rounds, value, reason = ledger_progress(rows)
    try:
        template_evaluations, template_rounds, _ = template_ledger_progress(folder)
    except (OSError, ValueError):
        template_evaluations, template_rounds = 0, 0
    write_json_atomic(folder / STATUS_FILE, GradientStatus(
        state=state, started_at=started_at, updated_at=now_text(), evaluations=evaluations, rounds_done=rounds,
        last_value=value, stop_reason=reason, template_evaluations=template_evaluations,
        template_rounds_done=template_rounds, pid=os.getpid(), message=message))


def run_worker(folder: Path, mode: Mode, *, should_stop: StopCheck) -> int:
    """跑一次（開始或接續），寫狀態檔與結果檔（結果檔由 ``run_search``／``resume_search`` 跑完時寫），回離開碼。"""
    started = now_text()
    _status(folder, "running", started)
    try:
        if runtime.jax_settings() != ("cpu", True):
            raise RuntimeError(f"梯度子行程要中央處理器＋雙精度，生效的是 {runtime.jax_settings()!r}")
        (run_search if mode == "start" else resume_search)(folder, should_stop=should_stop)
    except SearchStopped:
        _status(folder, "stopped", started, "停止記號：在兩次求值之間停下，帳本停在最後一列寫完整的地方")
        sys.stderr.write("梯度搜尋已停止（停止記號）；接續用 resume\n")
        return EXIT_STOPPED
    except KeyboardInterrupt:
        _status(folder, "interrupted", started, "中斷（終止訊號）")
        sys.stderr.write("梯度搜尋被中斷；接續用 resume\n")
        return EXIT_STOPPED
    except Exception as error:
        message = f"梯度搜尋失敗：{type(error).__name__}：{error}"
        sys.stderr.write(message + "\n")
        _status(folder, "failed", started, message)
        return EXIT_FAILED
    _status(folder, "finished", started)
    return EXIT_DONE


def run_endpoint_worker(folder: Path) -> int:
    """終點一次：寫 ``scheme-endpoint.json`` 與 ``endpoint.json``，回離開碼；不寫狀態檔。"""
    try:
        if runtime.jax_settings() != ("cpu", True):
            raise RuntimeError(f"終點子行程要中央處理器＋雙精度，生效的是 {runtime.jax_settings()!r}")
        run_endpoint(folder)
    except KeyboardInterrupt:
        sys.stderr.write("終點被中斷；endpoint.json 沒寫（或還是上一份）\n")
        return EXIT_STOPPED
    except Exception as error:
        sys.stderr.write(f"終點失敗：{type(error).__name__}：{error}\n")
        return EXIT_FAILED
    return EXIT_DONE


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in (*MODES, ENDPOINT_MODE):
        sys.stderr.write("用法：python -m aosr.gradient.worker <資料夾> start|resume|endpoint\n")
        return 2
    folder = Path(argv[0]).resolve()
    if argv[1] == ENDPOINT_MODE:
        return run_endpoint_worker(folder)
    mode: Mode = "start" if argv[1] == "start" else "resume"
    return run_worker(folder, mode, should_stop=stop_requested(folder))


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    for termination in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(termination, _interrupt)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM, signal.SIGHUP))
    raise SystemExit(main(sys.argv[1:]))
