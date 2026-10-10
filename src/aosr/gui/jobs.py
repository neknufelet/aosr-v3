"""獨立計算行程與可重讀的狀態小檔。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aosr.gui.labels import RESULT_RUN_LABELS, RUN_EXIT_TEXT
from aosr.runtime import child_process_env


REFERENCE_SECONDS = 360
RUN_ID_ENV = "AOSR_GUI_RUN_ID"
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


class ResultMoveConflict(ValueError):
    """同代號已在目的位置，或結果仍在計算中，不能搬動。"""


@dataclass(frozen=True)
class ResultStatus:
    """結果產物的計算身分；沒有可讀紀錄時沿用舊結果的完成待遇。"""

    status: str = "none"
    exit_code: int | None = None

    @property
    def finished(self) -> bool:
        return self.status in {"done", "none"}

    @property
    def status_text(self) -> str:
        return RESULT_RUN_LABELS[self.status][0]

    @property
    def label(self) -> str:
        return RESULT_RUN_LABELS[self.status][1]

    @property
    def notice(self) -> str:
        exit_text = RUN_EXIT_TEXT.format(code=self.exit_code) if self.exit_code is not None else ""
        return RESULT_RUN_LABELS[self.status][2].format(exit_text=exit_text)

    def notice_fields(self) -> dict[str, str]:
        return {} if self.finished else {"run_status": self.status, "run_notice": self.notice}


def _proc_stat(pid: int) -> tuple[str, int, int] | None:
    """核心的狀態、組號與開始時脈；已消失或格式不完整時無法確認身分。"""
    try:
        # 行程名可以含「) 」，從最後一個右括號切；其後首欄是原始第 3 欄。
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[2]), int(fields[19])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def _leader_matches(state: dict[str, object], start_ticks: int) -> bool:
    """逐位比核心記的開始時脈。#686 之前的舊紀錄沒有這一格，認不出身分就不認領：

    重開網頁的腳本在有計算跑著時不重開，所以換版那一刻不會有舊紀錄的計算還活著。
    """
    recorded = state.get("proc_start_ticks")
    return isinstance(recorded, int) and not isinstance(recorded, bool) and recorded == start_ticks


def _live_member(pid: int, run_id: str | None = None) -> bool:
    """掃一次組內非殭屍成員；領頭不在時還必須有這筆計算的環境記號。"""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        stat = _proc_stat(int(entry.name))
        if stat is None or stat[1] != pid or stat[0] == "Z":
            continue
        if run_id is None:
            return True
        try:
            environ = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if f"{RUN_ID_ENV}={run_id}".encode() in environ:
            return True
    return False


class JobManager:
    def __init__(self, data_dir: Path, runner: tuple[str, ...], engine_commit: str,
                 capabilities: Path, *, reference_seconds: int = REFERENCE_SECONDS,
                 reference_label: str = "", result_url_template: str = "/results/{run_id}") -> None:
        self.data_dir = data_dir
        self.runner = runner
        self.engine_commit = engine_commit
        self.capabilities = capabilities
        self.reference_seconds = reference_seconds
        self.reference_label = reference_label
        self.result_url_template = result_url_template
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.process_schemes: dict[str, str] = {}
        # 查狀態（事件迴圈上）與停止（背景執行緒）都會「讀狀態檔、判、寫回」；這把鎖只包住那一小段，
        # 不包住等行程死掉的那幾秒。可重入：停止在鎖裡會再叫一次讀檔。
        self._lock = threading.RLock()
        # 每一筆最後一次看到行程還活著的時間；被悄悄殺掉的計算不留錯誤訊息，結束時間只能靠它。
        self._last_alive: dict[str, float] = {}
        (data_dir / "runs").mkdir(parents=True, exist_ok=True)
        (data_dir / "results").mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.data_dir / "runs" / f"{run_id}.json"

    def _write(self, run_id: str, state: dict[str, object]) -> None:
        path = self._path(run_id)
        # 每次寫都開一個唯一的暫存檔，兩個寫者不共用同一個暫存檔名。
        handle, name = tempfile.mkstemp(dir=path.parent, prefix=f"{run_id}.", suffix=".tmp")
        temporary = Path(name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(state, ensure_ascii=False))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load(self, run_id: str) -> dict[str, object]:
        path = self._path(run_id)
        if not path.is_file():
            raise FileNotFoundError(run_id)
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("計算狀態檔不是物件")
        return {str(key): value for key, value in loaded.items()}

    def read_state(self, run_id: str) -> dict[str, object]:
        """讀持久狀態原文；不更新行程狀態。"""
        with self._lock:
            return self._load(run_id)

    def start(self, scheme_path: Path) -> dict[str, object]:
        # 一般起算讀的是 schemes/<代號>.json，檔名就是已過代號白名單的方案代號。
        return self._start(scheme_path, uuid.uuid4().hex, scheme_path.stem)

    def start_snapshot(self, scheme_json: str, scheme_label: str, *, extra_args: tuple[str, ...] = (),
                       job_fields: dict[str, object] | None = None) -> dict[str, object]:
        """在新計算代號自己的目錄封存方案，再用該份快照起算。

        檔名固定叫 scheme.json，不拿方案代號組路徑：結果檔裡的代號可能被動過（../、絕對路徑），
        拿它當檔名就能把快照寫到資料夾外、蓋掉別的結果。``scheme_label`` 只記進狀態給畫面顯示，
        呼叫端先照代號白名單過濾過。
        """
        run_id = uuid.uuid4().hex
        folder = self.data_dir / "runs" / run_id
        folder.mkdir()
        path = folder / "scheme.json"
        path.write_text(scheme_json, encoding="utf-8")
        if not extra_args and job_fields is None:
            return self._start(path, run_id, scheme_label)
        return self._start(path, run_id, scheme_label, extra_args=extra_args, job_fields=job_fields)

    def _settle_finished(self) -> None:
        """開新計算前，把自己開過、還記成算中的每一筆走一次 get：整組真的結束才會在 _settle 收屍（#726）。

        不直接 poll：領頭先結束、孫行程還在算時，殭屍領頭要留著組號（見 _group_alive），只能交給同一條判法。
        """
        for run_id, process in tuple(self.processes.items()):
            if process.returncode is None:
                try:
                    self.get(run_id)
                except (OSError, ValueError, KeyError, TypeError):
                    continue

    def _start(self, scheme_path: Path, run_id: str, scheme_label: str, *, extra_args: tuple[str, ...] = (),
               job_fields: dict[str, object] | None = None) -> dict[str, object]:
        self._settle_finished()
        result_path = self.data_dir / "results" / f"{run_id}.json"
        stderr_path = self.data_dir / "runs" / f"{run_id}.stderr"
        command = [*self.runner, str(scheme_path), "--out", str(result_path),
                   "--engine-commit", self.engine_commit,
                   "--capabilities", str(self.capabilities), *extra_args]
        # 執行緒與數值庫開關不繼承：MKL_CBWR 會改末位數字。三個數值函式庫執行緒變數一律給 1
        # （#577）：晚期混響的線性方程解會隨執行緒數差最後一位，網頁、考卷與搜尋的工作行程
        # 用同一個值，同一個方案在這台機器上算出逐位相同的物理結果；計算入口另固定 PARDISO 單緒。
        child_env = child_process_env(threads=1)
        child_env[RUN_ID_ENV] = run_id
        with stderr_path.open("wb") as stderr:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr,
                                       start_new_session=True, env=child_env,
                                       cwd=Path(__file__).resolve().parents[3])
        self.processes[run_id] = process
        self.process_schemes[run_id] = scheme_label
        stat = _proc_stat(process.pid)
        if stat is None:
            raise RuntimeError("無法確認計算行程身分")
        state: dict[str, object] = {
            "run_id": run_id, "scheme_id": scheme_label,
            "status": "running", "started_at": time.time(),
            "pid": process.pid, "exit_code": None, "result_path": str(result_path),
            "stderr_path": str(stderr_path),
            "proc_start_ticks": stat[2], "boot_id": BOOT_ID_PATH.read_text().strip(),
        }
        state.update(job_fields or {})
        self._write(run_id, state)
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, object]:
        with self._lock:
            state = self._load(run_id)
            if state.get("status") == "running":
                if self._group_alive(state):
                    self._last_alive[run_id] = time.time()
                else:
                    self._settle(run_id, state)
        if state["status"] == "running":
            end = time.time()
        elif "finished_at" in state:
            end = float(str(state["finished_at"]))
        else:
            # 記 finished_at 以前就結束的計算：狀態檔最後一次寫入就是判結束那一次。
            end = self._path(run_id).stat().st_mtime
        state["elapsed_s"] = round(max(0.0, end - float(str(state["started_at"]))), 1)
        reference = state.get("reference_seconds", self.reference_seconds)
        reference_label = state.get("reference_label", self.reference_label)
        state["reference_s"] = reference
        stderr_path = Path(str(state["stderr_path"]))
        # 錯誤輸出檔不在（資料夾搬過家、被清掉）就沒有尾巴可印，不讓整筆查不動、卡在計算中。
        state["stderr_tail"] = (stderr_path.read_text(errors="replace").splitlines()[-8:]
                                if stderr_path.is_file() else [])
        label = {"running": state.get("running_label", "計算中"), "done": "完成", "failed": "失敗",
                 "stopped": "已停止"}[str(state["status"])]
        # 重新整理後接回時，要看得出在算哪一份；舊狀態檔沒記代號就不印。
        scheme_label = state.get("display_label", state.get("scheme_id"))
        prefix = f"「{scheme_label}」" if isinstance(scheme_label, str) else ""
        state["display_text"] = (f"{prefix}{label}；已跑 {state['elapsed_s']} 秒，"
                                 f"參考值約 {reference} 秒" + (f"（{reference_label}）" if reference_label else ""))
        if state.get("process_note"):
            state["display_text"] = f"{state['display_text']}；{state['process_note']}"
        result_url = self.result_url_template.format(run_id=run_id)
        state["next_step_note"] = (f"查看結果：{result_url}"
                                   if state["status"] == "done" else "")
        state["result_url"] = result_url if state["status"] == "done" else None
        return state

    def result_status(self, run_id: str) -> ResultStatus:
        """有效紀錄經 get 查狀態；附屬檔讀不出來時保留原文，原文 running 不自行結算。"""
        try:
            raw = self.read_state(run_id)
        except (OSError, ValueError, KeyError, TypeError):
            return ResultStatus()
        status = raw.get("status")
        if not isinstance(status, str) or status not in RESULT_RUN_LABELS:
            return ResultStatus()
        try:
            state = self.get(run_id)
        except (OSError, ValueError, KeyError, TypeError):
            # get 可能已經結算、寫回之後才出錯：重讀一次，讀不動才沿用進去前那一份。
            try:
                state = self.read_state(run_id)
            except (OSError, ValueError, KeyError, TypeError):
                state = raw
        status = str(state["status"])
        if status not in RESULT_RUN_LABELS:
            return ResultStatus()
        code = state.get("exit_code")
        return ResultStatus(status, code if isinstance(code, int) else None)

    def restore_result(self, run_id: str) -> None:
        """持鎖搬回舊的單筆產物，不覆蓋原位置的任何同代號檔案。"""
        with self._lock:
            self._move_result(run_id, self.data_dir / "archive", self.data_dir)

    def _move_result(self, run_id: str, source: Path, target: Path) -> None:
        """呼叫端持有鎖；先查所有目的位置，搬動失敗就反向復原。"""
        result = (Path("results") / run_id).with_suffix(".json")
        relatives = (result, (Path("runs") / run_id).with_suffix(".json"),
                     (Path("runs") / run_id).with_suffix(".stderr"), Path("runs") / run_id)
        if not (source / result).is_file():
            raise FileNotFoundError("封存的結果找不到" if source.name == "archive" else "結果找不到")
        if any((target / item).exists() or (target / item).is_symlink() for item in relatives):
            place = "封存區" if target.name == "archive" else "結果或計算資料夾"
            raise ResultMoveConflict(f"{place}已經有同代號，沒有搬動任何檔案")
        self._move_files([(source / item, target / item) for item in relatives
                          if (source / item).exists() or (source / item).is_symlink()])

    def _move_files(self, paths: list[tuple[Path, Path]], finish: Callable[[], object] | None = None) -> None:
        """單筆與整包共用一次交易；清單更新也算在交易裡。呼叫端持鎖且先查過所有目的位置。"""
        moved: list[tuple[Path, Path]] = []
        try:
            for origin, destination in paths:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Path.replace 底層用 os.replace：同一顆碟搬動，不複製也不改檔案內容。
                origin.replace(destination)
                moved.append((origin, destination))
            if finish is not None:
                finish()
        except Exception as exc:
            self._rollback_result(moved)
            raise OSError("檔案搬動失敗，已搬回原處；請助理檢查資料夾權限") from exc

    def _rollback_result(self, moved: list[tuple[Path, Path]]) -> None:
        """每一個已搬的都試著復原；磁碟連復原都拒絕時，明說需要處理。"""
        errors: list[OSError] = []
        for origin, destination in reversed(moved):
            try:
                destination.replace(origin)
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise OSError("檔案搬動及搬回都失敗，請助理檢查資料夾") from errors[0]

    def list_recent(self) -> dict[str, object]:
        """列出所有未結束工作與最近一筆已結束工作。"""
        active: list[dict[str, object]] = []
        latest: tuple[float, dict[str, object]] | None = None
        for path in (self.data_dir / "runs").glob("*.json"):
            try:
                state = self.get(path.stem)
            except (ValueError, OSError, KeyError, TypeError):
                continue
            if state["status"] == "running":
                active.append(state)
            else:
                ended = float(str(state.get("finished_at", state.get("started_at", 0))))
                if latest is None or ended > latest[0]:
                    latest = (ended, state)
        active.sort(key=lambda item: float(str(item["started_at"])), reverse=True)
        return {"running": active, "recent": latest[1] if latest else None}

    def _settle(self, run_id: str, state: dict[str, object]) -> None:
        """已無可確認屬於自己的活行程：結算並寫回。呼叫端持有鎖。"""
        process = self.processes.get(run_id)
        # 收可取得的主行程離開碼；身分或權限不符時不能阻塞等行程結束。
        # 有行程把手時只有離開碼 0 才算完成；重開伺服器後沒有把手、收不回離開碼，只能看結果檔。
        code = process.poll() if process else None
        succeeded = code == 0 if process else True
        state["exit_code"] = code
        if state.get("stop_requested"):
            state["status"] = "stopped"  # 使用者按了停止：整組是被砍掉的，不是失敗
        else:
            state["status"] = "done" if Path(str(state["result_path"])).is_file() \
                and succeeded else "failed"
        if state["status"] == "failed" and code is None:
            state["process_note"] = "計算行程已不在（伺服器或電腦重開過）"
        # 沒人開著網頁時，要等下一次有人查才走到這裡；記查到的時間，離開一小時再回來就會記成跑了一小時。
        # 完成用結果檔寫出的時間。失敗時取 stderr 最後寫入（錯誤訊息）與最後一次看到它活著的較晚者：
        # 計算只在開頭寫 stderr，被記憶體不夠或 SIGKILL 悄悄殺掉時不留錯誤訊息。
        # 停止是停止那支呼叫當下判的，查到的時間就是停下來的時間。
        if state["status"] == "done":
            state["finished_at"] = Path(str(state["result_path"])).stat().st_mtime
        elif state["status"] == "failed":
            # stderr 不在（資料夾搬過家、被清掉）就只剩開始時間與最後一次看到它活著的時間。
            stderr_path = Path(str(state["stderr_path"]))
            written = [stderr_path.stat().st_mtime] if stderr_path.is_file() else []
            state["finished_at"] = max(float(str(state["started_at"])), *written,
                                       self._last_alive.get(run_id, 0.0))
        else:
            state["finished_at"] = time.time()
        self._write(run_id, state)

    def _group_alive(self, state: dict[str, object]) -> bool:
        """確認組內仍有自己的計算；無法讀身分或探測權限不足都不認領。"""
        try:
            pid = int(str(state["pid"]))
            if pid <= 0:
                return False
            process = self.processes.get(str(state.get("run_id")))
            if process is not None and process.pid == pid and process.returncode is None:
                # 自己開、還沒收的子行程：編號不可能被別人拿去，身分一定對。只偷看、不收屍（WNOWAIT）：
                # - 主執行緒先走、別的執行緒還在收尾時，/proc 已把領頭標成殭屍，核心卻還不讓收離開碼，
                #   這裡回 None，當它還在算（不然會在收得到離開碼 0 之前判失敗，#686 審查實測）。
                # - 領頭已結束但還沒收：殭屍留著組號，別人拿不走，組裡任何還活著的成員都算這筆計算的
                #   （孫行程不一定帶記號，例如經 runtime.child_process_env 開的）。收屍留給 _settle。
                try:
                    peek = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    reaped = False
                except ChildProcessError:
                    # 別處已經收掉了（例如 Popen 自己）：往下走「只認記號」那一支。
                    peek = None
                    reaped = process.poll() is not None
                if not reaped:
                    if peek is None:
                        return True
                    return _live_member(pid) or _live_member(pid)
            if process is not None and process.pid == pid:
                # 領頭已經收掉：組號可能被重用，只認身上帶這筆計算記號的成員。
                own = str(state["run_id"])
                return _live_member(pid, own) or _live_member(pid, own)
            # 舊紀錄沒有開機代號，跟換過開機一樣認不出身分。
            if state.get("boot_id") != BOOT_ID_PATH.read_text().strip():
                return False
            os.killpg(pid, 0)
            leader = _proc_stat(pid)
            run_id = None
            if leader is not None:
                # 殭屍領頭仍提供可靠的身分，不能跳過比對。
                if leader[1] != pid or not _leader_matches(state, leader[2]):
                    return False
            else:
                marker = state.get("run_id")
                if not isinstance(marker, str) or not marker:
                    return False
                run_id = marker
            # 判死前再掃一次（#601）：一次掃描先拍 /proc 名單再逐筆讀，領頭行程一派生完就結束時，
            # 拍名單那時還沒出生的孫行程不在名單裡，掃到領頭那一筆它已經是殭屍，整群會被誤判成死了。
            # 第二次拍的名單一定包含第一次掃描期間出生的成員（領頭結束前就已經派生完）。
            return _live_member(pid, run_id) or _live_member(pid, run_id)
        except (ProcessLookupError, PermissionError, FileNotFoundError, KeyError, ValueError):
            return False

    def _wait_group(self, state: dict[str, object], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while self._group_alive(state):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def stop(self, run_id: str) -> dict[str, object]:
        with self._lock:
            state = self.get(run_id)
            if state["status"] != "running":
                return state
            pid = int(str(state["pid"]))
            if pid <= 0:
                raw = self._load(run_id)
                self._settle(run_id, raw)
                return self.get(run_id)
            # 先記「使用者要停」再砍：同時查狀態的那一邊看到整組死了，判的是已停止、不是失敗。
            raw = self._load(run_id)
            raw["stop_requested"] = True
            self._write(run_id, raw)
        try:
            if self._group_alive(state):
                os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not self._wait_group(state, 2.0):
            try:
                if self._group_alive(state):
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not self._wait_group(state, 2.0):
                raise RuntimeError("行程組仍在執行，停止未確認")
        with self._lock:
            raw = self._load(run_id)
            if raw.get("status") == "running":
                self._settle(run_id, raw)
        return self.get(run_id)
