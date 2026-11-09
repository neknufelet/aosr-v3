"""running 已寫入、搜尋尚未開始時中斷，仍須留下 interrupted 狀態。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aosr.gradient import run, worker
from aosr.gradient.material_gradient import StopCheck
from tests.engine.test_material_gradient_mount import _project


def test_interrupt_after_running_write_records_interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, _project())
    original_status = worker._status
    reached: list[Path] = []

    def interrupt_after_running(folder: Path, state: run.State, started: str, message: str | None = None, *,
                                unestimable_reason: str | None = None) -> None:
        original_status(folder, state, started, message, unestimable_reason=unestimable_reason)
        if state == 'running':
            raise KeyboardInterrupt

    def unexpected_search(folder: Path, *, should_stop: StopCheck | None) -> None:
        reached.append(folder)

    monkeypatch.setattr(worker, '_status', interrupt_after_running)
    monkeypatch.setattr(worker, 'run_search', unexpected_search)
    try:
        code = worker.run_worker(tmp_path, 'start', should_stop=lambda: False)
    except KeyboardInterrupt:
        pytest.fail('running 已寫入後的 KeyboardInterrupt 漏出 run_worker')
    status = run.GradientStatus.model_validate_json((tmp_path / run.STATUS_FILE).read_bytes())
    assert code == 3
    assert status.state == 'interrupted'
    assert status.message == '中斷（終止訊號）'
    assert reached == []
