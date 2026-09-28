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
from pathlib import Path


REFERENCE_SECONDS = 360


class JobManager:
    def __init__(self, data_dir: Path, runner: tuple[str, ...], engine_commit: str,
                 capabilities: Path) -> None:
        self.data_dir = data_dir
        self.runner = runner
        self.engine_commit = engine_commit
        self.capabilities = capabilities
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        # 查狀態（事件迴圈上）與停止（背景執行緒）都會「讀狀態檔、判、寫回」；這把鎖只包住那一小段，
        # 不包住等行程死掉的那幾秒。可重入：停止在鎖裡會再叫一次讀檔。
        self._lock = threading.RLock()
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

    def start_snapshot(self, scheme_json: str, scheme_label: str) -> dict[str, object]:
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
        return self._start(path, run_id, scheme_label)

    def _start(self, scheme_path: Path, run_id: str, scheme_label: str) -> dict[str, object]:
        result_path = self.data_dir / "results" / f"{run_id}.json"
        stderr_path = self.data_dir / "runs" / f"{run_id}.stderr"
        command = [*self.runner, str(scheme_path), "--out", str(result_path),
                   "--engine-commit", self.engine_commit,
                   "--capabilities", str(self.capabilities)]
        with stderr_path.open("wb") as stderr:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr,
                                       start_new_session=True)
        self.processes[run_id] = process
        state: dict[str, object] = {
            "run_id": run_id, "scheme_id": scheme_label,
            "status": "running", "started_at": time.time(),
            "pid": process.pid, "exit_code": None, "result_path": str(result_path),
            "stderr_path": str(stderr_path),
        }
        self._write(run_id, state)
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, object]:
        with self._lock:
            state = self._load(run_id)
            if state.get("status") == "running" and not self._group_alive(int(str(state["pid"]))):
                self._settle(run_id, state)
        if state["status"] == "running":
            end = time.time()
        elif "finished_at" in state:
            end = float(str(state["finished_at"]))
        else:
            # 記 finished_at 以前就結束的計算：狀態檔最後一次寫入就是判結束那一次。
            end = self._path(run_id).stat().st_mtime
        state["elapsed_s"] = round(max(0.0, end - float(str(state["started_at"]))), 1)
        state["reference_s"] = REFERENCE_SECONDS
        stderr_path = Path(str(state["stderr_path"]))
        state["stderr_tail"] = stderr_path.read_text(errors="replace").splitlines()[-8:]
        label = {"running": "計算中", "done": "完成", "failed": "失敗",
                 "stopped": "已停止"}[str(state["status"])]
        state["display_text"] = (f"{label}；已跑 {state['elapsed_s']} 秒，"
                                 f"參考值約 {REFERENCE_SECONDS} 秒")
        state["next_step_note"] = (f"查看結果：/results/{run_id}"
                                   if state["status"] == "done" else "")
        state["result_url"] = f"/results/{run_id}" if state["status"] == "done" else None
        return state

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
        """整組都沒了：判完成、失敗或已停止，寫回。呼叫端持有鎖。"""
        process = self.processes.get(run_id)
        # 整組都沒了才收主行程的離開碼（先收再查，主行程剛好在中間結束就會拿到空的）。
        # 有行程把手時只有離開碼 0 才算完成；重開伺服器後沒有把手、收不回離開碼，只能看結果檔。
        code = process.wait() if process else None
        succeeded = code == 0 if process else True
        state["exit_code"] = code
        if state.get("stop_requested"):
            state["status"] = "stopped"  # 使用者按了停止：整組是被砍掉的，不是失敗
        else:
            state["status"] = "done" if Path(str(state["result_path"])).is_file() \
                and succeeded else "failed"
        # 沒人開著網頁時，要等下一次有人查才走到這裡；完成的用結果檔寫出的時間，
        # 不然離開一小時再回來會記成跑了一小時。失敗與停止沒有這樣的檔，只能記查到的時間。
        state["finished_at"] = (Path(str(state["result_path"])).stat().st_mtime
                                if state["status"] == "done" else time.time())
        self._write(run_id, state)

    def _group_alive(self, pid: int) -> bool:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return False
        if Path("/proc").is_dir():
            for entry in Path("/proc").iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    # 行程名那一欄可以含「) 」，從最後一個右括號切；拆不動的那一個行程略過。
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    group = int(fields[2])
                except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
                    continue
                if group == pid and fields[0] != "Z":
                    return True
            return False
        return True

    def _wait_group(self, pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while self._group_alive(pid):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def stop(self, run_id: str) -> dict[str, object]:
        with self._lock:
            state = self.get(run_id)
            if state["status"] != "running":
                return state
            # 先記「使用者要停」再砍：同時查狀態的那一邊看到整組死了，判的是已停止、不是失敗。
            raw = self._load(run_id)
            raw["stop_requested"] = True
            self._write(run_id, raw)
        pid = int(str(state["pid"]))
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process = self.processes.get(run_id)
        if not self._wait_group(pid, 2.0):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not self._wait_group(pid, 2.0):
                raise RuntimeError("行程組仍在執行，停止未確認")
        with self._lock:
            raw = self._load(run_id)
            if raw.get("status") == "running":
                self._settle(run_id, raw)
        return self.get(run_id)
