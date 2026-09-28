"""獨立計算行程與可重讀的狀態小檔。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
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
        (data_dir / "runs").mkdir(parents=True, exist_ok=True)
        (data_dir / "results").mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.data_dir / "runs" / f"{run_id}.json"

    def _write(self, run_id: str, state: dict[str, object]) -> None:
        path = self._path(run_id)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def start(self, scheme_path: Path) -> dict[str, object]:
        run_id = uuid.uuid4().hex
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
            "run_id": run_id, "status": "running", "started_at": time.time(),
            "pid": process.pid, "exit_code": None, "result_path": str(result_path),
            "stderr_path": str(stderr_path),
        }
        self._write(run_id, state)
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, object]:
        path = self._path(run_id)
        if not path.is_file():
            raise FileNotFoundError(run_id)
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("計算狀態檔不是物件")
        state: dict[str, object] = {str(key): value for key, value in loaded.items()}
        if state.get("status") == "running":
            process = self.processes.get(run_id)
            code = process.poll() if process else None
            finished = code is not None or (process is None and
                                            not self._restarted_alive(state))
            if finished:
                state["exit_code"] = code
                state["status"] = "done" if Path(str(state["result_path"])).is_file() \
                    and code in (0, None) else "failed"
                self._write(run_id, state)
        state["elapsed_s"] = round(max(0.0, time.time() - float(str(state["started_at"]))), 1)
        state["reference_s"] = REFERENCE_SECONDS
        stderr_path = Path(str(state["stderr_path"]))
        state["stderr_tail"] = stderr_path.read_text(errors="replace").splitlines()[-8:]
        label = {"running": "計算中", "done": "完成", "failed": "失敗",
                 "stopped": "已停止"}[str(state["status"])]
        state["display_text"] = (f"{label}；已跑 {state['elapsed_s']} 秒，"
                                 f"參考值約 {REFERENCE_SECONDS} 秒")
        state["next_step_note"] = "結果頁在下一步" if state["status"] == "done" else ""
        return state

    def _restarted_alive(self, state: dict[str, object]) -> bool:
        pid = int(str(state["pid"]))
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        if Path("/proc").is_dir():
            try:
                status = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
            except FileNotFoundError:
                return False
            if status == "Z":
                return False
        return True

    def stop(self, run_id: str) -> dict[str, object]:
        state = self.get(run_id)
        if state["status"] != "running":
            return state
        pid = int(str(state["pid"]))
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process = self.processes.get(run_id)
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                process.wait(timeout=2)
            state["exit_code"] = process.returncode
        state["status"] = "stopped"
        self._write(run_id, state)
        return self.get(run_id)
