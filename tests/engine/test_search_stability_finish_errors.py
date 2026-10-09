"""附件收尾的停止窄縫與最外層出口，走真摘要及真訊號處理。"""
from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.search import cli, crossover_record
from aosr.search.crossover_sensitivity import record_crossover_error
from aosr.search.placement_stability_attach import record_stability_error
from aosr.search.placement_stability_record import read_summary, summary_path
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore
from tests.engine._stability_attach_cases import ShiftCompute, attach, ready


def setup_auto(store: SearchStore, registry: Path, status: SearchStatus, patch: pytest.MonkeyPatch) -> list[str]:
    patch.setattr(cli, "auto_search", lambda *a, **k: status)
    patch.setattr(cli, "attach_modal", lambda *a, **k: None)
    patch.setattr(cli, "attach_crossover", lambda *a, **k: None)
    patch.setattr(cli, "_identity", lambda *a: store.identity)
    patch.setattr(cli, "config_path", lambda name: registry if name == "quality_targets.toml" else config_path(name))
    return ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(store.path.parent / "cache")]


def test_modal_recorder_propagates_stop_to_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import modal_attach
    store, _, status = ready(tmp_path)
    def stop(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt
    monkeypatch.setattr(modal_attach, "write_summary", stop)
    with pytest.raises(KeyboardInterrupt):
        modal_attach.record_attachment_error(store, status, tmp_path / "cache", RuntimeError("低頻失敗"), stopped=False)


@pytest.mark.parametrize("stage", ["modal", "crossover"])
def test_prior_stop_preserves_fresh_done_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    store, registry, status = ready(tmp_path)
    crossing = crossover_record.fresh_summary(store, status).model_copy(update={"state": "done", "completed": True})
    crossover_record.write_summary(store.path, crossing)
    attach(store, registry, status, ShiftCompute(store))
    paths = (summary_path(store.path), crossover_record.summary_path(store.path))
    before = {p: p.read_bytes() for p in paths}
    args = setup_auto(store, registry, status, monkeypatch)
    def stop(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("前段被停仍開跑")
    monkeypatch.setattr(cli, "attach_modal" if stage == "modal" else "attach_crossover", stop)
    if stage == "modal":
        monkeypatch.setattr(cli, "attach_crossover", forbidden)
    monkeypatch.setattr(cli, "attach_stability", forbidden)
    code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()))
    assert code == 0
    assert {p: p.read_bytes() for p in paths} == before


@pytest.mark.parametrize("stage", ["modal", "crossover"])
def test_sigterm_while_recording_prevents_next_attachments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         stage: str) -> None:
    store, registry, status = ready(tmp_path)
    args = setup_auto(store, registry, status, monkeypatch)
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("前段計算失敗原文")
    target = "record_attachment_error" if stage == "modal" else "record_crossover_error"
    original = getattr(cli, target)
    recording_signals: list[bool] = []
    def recording(*args: object, **kwargs: object) -> object:
        recording_signals.append(signal.getsignal(signal.SIGTERM) == signal.SIG_IGN)
        os.kill(os.getpid(), signal.SIGTERM)
        return original(*args, **kwargs)
    monkeypatch.setattr(cli, "attach_modal" if stage == "modal" else "attach_crossover", fail)
    monkeypatch.setattr(cli, target, recording)
    started: list[str] = []
    if stage == "modal":
        monkeypatch.setattr(cli, "attach_crossover", lambda *a, **k: started.append("交接"))
    monkeypatch.setattr(cli, "attach_stability", lambda *a, **k: started.append("擺位"))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        cli._interrupt_on_termination()
        code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()))
        assert code == 0
        assert all(signal.getsignal(s) == signal.SIG_IGN for s in handlers)
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)
    assert recording_signals == [False, True] and started == []
    saved = read_summary(store.path)
    assert saved is not None and saved.state == "stopped" and not saved.completed
    crossing = crossover_record.read_summary(store.path)
    assert crossing is not None and crossing.state == "stopped"
    if stage == "modal":
        from aosr.search.modal_record import read_summary as read_modal
        modal = read_modal(store.path)
        assert modal is not None and modal.completed and "已停止" in modal.reason_text


def test_real_modal_internal_recording_stop_prevents_later_attachments(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import modal_attach, modal_record
    store, registry, status = ready(tmp_path)
    args = setup_auto(store, registry, status, monkeypatch)
    monkeypatch.setattr(cli, "attach_modal", modal_attach.attach_modal)
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("低頻本身失敗原文")
    monkeypatch.setattr(modal_attach, "_attach", fail)
    original_finish = modal_attach._finish
    recording_signals: list[bool] = []
    def finish(folder: Path, summary: modal_record.ModalSummary,
               *keep: modal_record.ModalSummary | None) -> None:
        recording_signals.append(signal.getsignal(signal.SIGTERM) == signal.SIG_IGN)
        os.kill(os.getpid(), signal.SIGTERM)
        original_finish(folder, summary, *keep)
    monkeypatch.setattr(modal_attach, "_finish", finish)
    started: list[str] = []
    monkeypatch.setattr(cli, "attach_crossover", lambda *a, **k: started.append("交接"))
    monkeypatch.setattr(cli, "attach_stability", lambda *a, **k: started.append("擺位"))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        cli._interrupt_on_termination()
        code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()))
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)
    assert code == 0 and not started
    assert recording_signals == [False, True]
    modal = modal_record.read_summary(store.path)
    crossing = crossover_record.read_summary(store.path)
    stable = read_summary(store.path)
    assert modal is not None and modal.completed and "已停止" in modal.reason_text
    assert crossing is not None and stable is not None and crossing.state == stable.state == "stopped"


@pytest.mark.parametrize("stop", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_outer_error_records_each_attachment_once_and_preserves_fresh_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: bool, fresh: bool,
) -> None:
    store, registry, status = ready(tmp_path)
    if fresh:
        crossover_record.write_summary(store.path, crossover_record.fresh_summary(store, status).model_copy(
            update={"state": "done", "completed": True}))
        attach(store, registry, status, ShiftCompute(store))
    paths = (crossover_record.summary_path(store.path), summary_path(store.path))
    before = {p: p.read_bytes() for p in paths} if fresh else {}
    args = setup_auto(store, registry, status, monkeypatch)
    def leaked(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt() if stop else OSError("最外層收尾失敗原文")
    monkeypatch.setattr(cli, "_finish_attachments", leaked)
    calls: list[str] = []
    def crossing(opened: SearchStore, current: SearchStatus, error: BaseException) -> None:
        calls.append("交接")
        record_crossover_error(opened, current, error)
    def stability(opened: SearchStore, current: SearchStatus, error: BaseException) -> None:
        calls.append("擺位")
        record_stability_error(opened, current, error)
    monkeypatch.setattr(cli, "record_crossover_error", crossing)
    monkeypatch.setattr(cli, "record_stability_error", stability)
    code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()))
    assert code == 0
    assert calls == ["交接", "擺位"]
    if fresh:
        assert {p: p.read_bytes() for p in paths} == before
    else:
        cross = crossover_record.read_summary(store.path)
        stable = read_summary(store.path)
        assert cross is not None and stable is not None
        assert cross.state == stable.state == ("stopped" if stop else "failed")
        if not stop:
            for reason in (cross.reason_text, stable.reason_text):
                assert "前段附件收尾失敗" in reason and "最外層收尾失敗原文" in reason
