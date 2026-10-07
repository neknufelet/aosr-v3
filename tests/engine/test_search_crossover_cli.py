"""交接收尾在鎖裡、模態之後；離開碼照外圈，重跑只讀新摘要。"""
from __future__ import annotations

import fcntl
import os
import signal
from pathlib import Path

import pytest

from aosr.reporting.result import SchemeResult
from aosr.search import cli, crossover_sensitivity as module
from aosr.search.crossover_record import STOPPED_NOTE, STOPPED_REASON, read_summary, summary_lines, write_summary
from aosr.search.outer_status import OUTER_MESSAGES, OuterStatus, attachment_skip_reason
from aosr.search.store import SearchStore
from aosr.search.run import SearchStatus
from aosr.search.crossover_record import CrossoverSummary
from tests.engine._crossover_cases import evaluate, prepared, protected
from tests.engine._search_outer_cases import OuterCompute


@pytest.mark.parametrize("conclusion,code", [("complete", 0), ("refine_budget", 0), ("refine_failed", 1), ("refine_interrupted", 3)])
@pytest.mark.parametrize("action", ["success", "exception", "stopped"])
def test_auto_preserves_bytes_exit_and_lock_after_modal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       conclusion: str, code: int, action: str,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    store, _, status = prepared(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate({"conclusion": conclusion})})
    events = []
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *args, **kwargs: events.append("modal"))
    monkeypatch.setattr(module, "reevaluate", evaluate)
    real_attach = module.attach_crossover
    def attach(opened: SearchStore, *, status: SearchStatus, quality_targets_path: Path) -> CrossoverSummary:
        events.append("crossover")
        descriptor = opened.open_folder_lock()
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        if action != "success":
            raise KeyboardInterrupt if action == "stopped" else RuntimeError("題目錯誤原文")
        return real_attach(opened, status=status, quality_targets_path=quality_targets_path)
    monkeypatch.setattr(cli, "attach_crossover", attach)
    before = protected(store)
    actual = cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
        compute_factory=lambda opened, *_: OuterCompute(opened))
    assert actual == code and protected(store) == before
    assert events == ["modal", "crossover"]
    summary = read_summary(store.path)
    assert summary is not None
    if action != "success":
        assert not summary.completed and summary.state == ("stopped" if action == "stopped" else "failed")
        if action == "stopped":
            assert summary.reason_text == STOPPED_REASON and STOPPED_NOTE in capsys.readouterr().err
        else:
            assert "題目錯誤原文" in summary.reason_text


@pytest.mark.parametrize("conclusion", [None, *OUTER_MESSAGES])
def test_shared_attachment_eligibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conclusion: str | None) -> None:
    store, registry, status = prepared(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate({"conclusion": conclusion})})
    calls = []
    def counted(result: SchemeResult, **kwargs: object) -> object:
        calls.append(result)
        return evaluate(result)
    monkeypatch.setattr(module, "reevaluate", counted)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    reason = attachment_skip_reason(status.outer.conclusion)
    if reason:
        assert summary.state == "skipped" and summary.reason_text == reason and not calls
    else:
        assert calls and summary.state == "done"
    assert ("（暫時）" in "\n".join(summary_lines(summary))) == (conclusion != "complete")


def test_auto_reuses_complete_and_retries_changed_or_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import math
    from aosr.search.refine import RefineLedger
    store, _, status = prepared(tmp_path)
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *args, **kwargs: None)
    calls = []
    def counted(result: SchemeResult, **kwargs: object) -> object:
        calls.append(result)
        return evaluate(result)
    monkeypatch.setattr(module, "reevaluate", counted)
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    def run() -> int:
        return cli.main(args, compute_factory=lambda opened, *_: OuterCompute(opened))
    code = run()
    assert code == 0
    first = read_summary(store.path)
    assert first is not None and first.completed
    previous = tuple(calls)
    code = run()
    assert code == 0 and tuple(calls) == previous
    write_summary(store.path, first.model_copy(update={"completed": False, "state": "stopped", "reason_text": STOPPED_REASON}))
    code = run()
    assert code == 0 and tuple(calls) != previous
    previous = tuple(calls)
    header, rows = RefineLedger.read(store.refine_ledger_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header)
    for row in rows:
        book.append(row.model_copy(update={"total_cost": math.nextafter(row.total_cost, math.inf)})
            if row.trial_number == 9 and row.total_cost is not None else row)
    code = run()
    assert code == 0 and tuple(calls) != previous
    changed = read_summary(store.path)
    assert changed is not None and changed.rows_fingerprint != first.rows_fingerprint


@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGHUP])
def test_termination_during_in_process_evaluation_records_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              termination: signal.Signals,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    store, _, status = prepared(tmp_path)
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *args, **kwargs: None)
    def stopped(*args: object, **kwargs: object) -> object:
        os.kill(os.getpid(), termination)
        raise AssertionError("訊號未轉成 KeyboardInterrupt")
    monkeypatch.setattr(module, "reevaluate", stopped)
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    before = protected(store)
    try:
        cli._interrupt_on_termination()
        code = cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
            compute_factory=lambda opened, *_: OuterCompute(opened))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert code == 0 and protected(store) == before
    summary = read_summary(store.path)
    assert summary is not None and not summary.completed and summary.state == "stopped"
    assert summary.reason_text == STOPPED_REASON and STOPPED_NOTE in capsys.readouterr().err
