"""#1128：一般計算畫面的「參考值約 N 秒」照這台機器量到的，不是固定 360 秒（2-6c：109 房一場約 21 分）。

這份方案上一次算完量到的優先；沒算過看同一間房的另一份（搜尋存成方案時的搜尋起點）；都沒有才用預設值。
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from aosr.gui.jobs import REFERENCE_SECONDS, JobManager

COMMIT = "a" * 40
# 替身計算：寫出結果檔就結束。
RUNNER = "import sys\nout = sys.argv[sys.argv.index('--out') + 1]\nopen(out, 'w').write('{}')\n"


def _manager(tmp_path: Path) -> JobManager:
    script = tmp_path / "runner.py"
    script.write_text(RUNNER, encoding="utf-8")
    return JobManager(tmp_path / "data", (sys.executable, str(script)), COMMIT, tmp_path / "capabilities.toml")


def _done(manager: JobManager, scheme_id: str, started: float, seconds: float, status: str = "done") -> str:
    run_id = uuid.uuid4().hex
    state = {"run_id": run_id, "scheme_id": scheme_id, "status": status, "started_at": started,
             "finished_at": started + seconds, "stderr_path": str(manager.data_dir / "runs" / f"{run_id}.stderr")}
    (manager.data_dir / "runs" / f"{run_id}.json").write_text(json.dumps(state), encoding="utf-8")
    return run_id


def _scheme(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _finish(manager: JobManager, view: dict[str, object]) -> None:
    manager.processes[str(view["run_id"])].wait(timeout=10)
    manager.get(str(view["run_id"]))


def test_reference_time_is_what_this_machine_measured_for_the_scheme_or_its_room(tmp_path: Path) -> None:
    """抓參考值照舊寫 360 秒（實際 21 分）、拿到沒算完或較舊那一筆、或同一間房的另一份量過卻不用。"""
    manager = _manager(tmp_path)
    _done(manager, "room", 1000.0, 900.0)
    latest = _done(manager, "room", 5000.0, 1266.4)
    _done(manager, "room", 9000.0, 30.0, status="failed")
    _done(manager, "start", 2000.0, 1250.0)
    assert manager.run_seconds(latest) == 1266 and manager.measured_seconds("room") == 1266
    assert manager.measured_seconds("nothing") is None

    own = manager.start(_scheme(tmp_path, "room"))
    assert own["reference_s"] == 1266 and "上一次算這份方案量到的" in str(own["display_text"])
    _finish(manager, own)
    adopted = manager.start(_scheme(tmp_path, "start-search-0123abcd"), reference_scheme="start")
    assert adopted["reference_s"] == 1250 and "同一間房「start」上一次量到的" in str(adopted["display_text"])
    _finish(manager, adopted)
    fresh = manager.start(_scheme(tmp_path, "elsewhere"))
    assert fresh["reference_s"] == REFERENCE_SECONDS and "（" not in str(fresh["display_text"]).split("參考值約")[1]
    _finish(manager, fresh)
