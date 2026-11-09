"""running 寫入邊界與被回呼吞掉的終止訊號，都須留下 interrupted 狀態。"""
from __future__ import annotations

from pathlib import Path
import signal
from threading import Event

import pytest

from aosr.gradient import run, worker
from aosr import runtime
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


def test_swallowed_signal_reappears_at_stop_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, _project())
    latch: Event | None = getattr(worker, '_INTERRUPTED', None)

    def swallowing_search(folder: Path, *, should_stop: StopCheck | None) -> None:
        try:
            worker._interrupt(signal.SIGTERM, None)
        except KeyboardInterrupt:
            pass
        if should_stop is None:
            pytest.fail('搜尋沒有收到停止檢查')
        should_stop()

    monkeypatch.setattr(runtime, 'jax_settings', lambda: ('cpu', True))
    monkeypatch.setattr(worker, 'run_search', swallowing_search)
    try:
        try:
            code = worker.run_worker(tmp_path, 'start', should_stop=lambda: False)
        except KeyboardInterrupt:
            pytest.fail('被吞掉的終止訊號在停止檢查重拋後漏出 run_worker')
        status = run.GradientStatus.model_validate_json((tmp_path / run.STATUS_FILE).read_bytes())
        assert status.state == 'interrupted'
        assert code == 3
        assert status.message == '中斷（終止訊號）'
    finally:
        if latch is not None:
            latch.clear()
