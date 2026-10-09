"""#726：連送計算時，前一份算完的計算行程在開下一份前就收掉，伺服器底下不留殭屍。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from aosr.gui.jobs import JobManager
COMMIT = "a" * 40

# 替身計算：寫出結果檔就結束，沒有孫行程。
RUNNER = "import sys\nout = sys.argv[sys.argv.index('--out') + 1]\nopen(out, 'w').write('{}')\n"


def _zombie_children() -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] == "Z" and int(fields[1]) == os.getpid():
            found.append(int(entry.name))
    return found


def _wait_exited(pid: int) -> None:
    # 等到行程已經結束、只剩殭屍（還沒被收）：/proc 狀態是 Z。
    for _ in range(500):
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return
        time.sleep(0.01)
    raise AssertionError("替身計算沒有結束")


def test_starting_next_run_reaps_previous_finished_run(tmp_path: Path) -> None:
    script = tmp_path / "runner.py"
    script.write_text(RUNNER)
    scheme = tmp_path / "scheme.json"
    scheme.write_text("{}")
    manager = JobManager(tmp_path, (sys.executable, str(script)), COMMIT, tmp_path / "capabilities.toml")
    first = manager.start(scheme)
    first_pid = int(str(first["pid"]))
    _wait_exited(first_pid)
    assert first_pid in _zombie_children()
    # 不查第一份，直接送第二份：開新計算前要把第一份結算並收屍。
    second = manager.start(scheme)
    try:
        assert first_pid not in _zombie_children()
        settled = manager.read_state(str(first["run_id"]))
        assert settled["status"] == "done" and settled["exit_code"] == 0
    finally:
        manager.processes[str(second["run_id"])].wait(timeout=10)
        manager.get(str(second["run_id"]))
