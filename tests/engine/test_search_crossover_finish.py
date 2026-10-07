"""交接收尾的失敗重試、停止、沿用中斷；外圈離開碼與搜尋位元組固定。"""
from __future__ import annotations

import signal
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.gui.search_view import build_search_view
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import CandidateEvaluation
from aosr.search import cli, crossover_sensitivity as module
from aosr.search.crossover_record import CrossoverSummary, STOPPED_REASON, is_stale, read_summary, summary_path, write_summary
from aosr.search.modal_attach import AttachmentRecorded, STOPPED_NOTE
from aosr.search.outer_status import OuterStatus
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore
from tests.engine._crossover_cases import evaluate, prepared, protected
from tests.engine._search_outer_cases import OuterCompute


@pytest.fixture(autouse=True)
def restore_handlers() -> Iterator[None]:
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    yield
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _run(store: SearchStore, tmp_path: Path) -> int:
    return cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
        compute_factory=lambda opened, *_: OuterCompute(opened))


@pytest.mark.parametrize("conclusion,code", [("complete", 0), ("refine_failed", 1), ("refine_interrupted", 3)])
@pytest.mark.parametrize("recorded", [False, True])
def test_modal_stop_never_starts_crossover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str], conclusion: str, code: int, recorded: bool) -> None:
    store, _, status = prepared(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate({"conclusion": conclusion})})
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    calls, errors = [], []
    def stopped(*args: object, **kwargs: object) -> object:
        raise AttachmentRecorded(KeyboardInterrupt(), stopped=True) if recorded else KeyboardInterrupt()
    def counted(result: SchemeResult, **kwargs: object) -> object:
        calls.append(result.scheme.scheme_id)
        return evaluate(result)
    monkeypatch.setattr(cli, "attach_modal", stopped)
    monkeypatch.setattr(cli, "record_attachment_error", lambda *args, **kwargs: errors.append(args[-1]))
    monkeypatch.setattr(module, "reevaluate", counted)
    before = protected(store)
    assert _run(store, tmp_path) == code
    summary = read_summary(store.path)
    assert not calls and summary is not None and summary.state == "stopped" and not summary.completed
    assert summary.reason_text == STOPPED_REASON and protected(store) == before
    assert capsys.readouterr().err == STOPPED_NOTE + "\n"
    assert bool(errors) is not recorded


def _error(kind: str) -> Exception:
    if kind == "ValidationError":
        try:
            CandidateEvaluation.model_validate({})
        except ValidationError as error:
            return error
        raise AssertionError("題目沒有造出驗證錯誤")
    return {"KeyError": KeyError("f_s_hz"), "ValueError": ValueError("重評出錯"),
            "ZeroDivisionError": ZeroDivisionError(), "AssertionError": AssertionError()}[kind]


@pytest.mark.parametrize("kind", ["ValidationError", "KeyError", "ValueError", "ZeroDivisionError", "AssertionError"])
def test_reevaluation_errors_fail_warn_and_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str], kind: str) -> None:
    store, _, status = prepared(tmp_path)
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *args, **kwargs: None)
    calls = []
    def broken(result: SchemeResult, **kwargs: object) -> object:
        calls.append(result.scheme.scheme_id)
        raise _error(kind)
    monkeypatch.setattr(module, "reevaluate", broken)
    before = protected(store)
    code = _run(store, tmp_path)
    assert code == 0
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "failed" and not summary.completed
    assert "失敗" in summary.reason_text and f"{kind}：" in summary.reason_text
    block = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "crossover")
    assert block.warning and f"{kind}：" in "\n".join(block.lines)
    previous = tuple(calls)
    code = _run(store, tmp_path)
    assert code == 0 and tuple(calls) != previous
    assert f"{kind}：" in capsys.readouterr().err and protected(store) == before


@pytest.mark.parametrize("window", ["reuse", "after_write"])
def test_interrupt_at_completion_preserves_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str], window: str) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    original_write, original_stale = write_summary, is_stale
    saved = []
    if window == "reuse":
        module.attach_crossover(store, status=status, quality_targets_path=registry)
        saved.append(summary_path(store.path).read_bytes())
        interrupt_next = True
        def stale(previous: CrossoverSummary, current: CrossoverSummary) -> bool:
            nonlocal interrupt_next
            if interrupt_next:
                interrupt_next = False
                raise KeyboardInterrupt()
            return original_stale(previous, current)
        monkeypatch.setattr(module, "is_stale", stale)
    else:
        def write(folder: Path, summary: CrossoverSummary) -> None:
            original_write(folder, summary)
            if summary.completed:
                saved.append(summary_path(folder).read_bytes())
                raise KeyboardInterrupt()
        monkeypatch.setattr(module, "write_summary", write)
    before = protected(store)
    code = _run(store, tmp_path)
    assert code == 0
    assert summary_path(store.path).read_bytes() == saved[0] and protected(store) == before
    assert "交接敏感度檢查被停止" in capsys.readouterr().err


@pytest.mark.parametrize("previous", ["broken", "incomplete", "stale"])
def test_stop_replaces_only_unusable_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous: str) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    if previous == "broken":
        summary_path(store.path).write_text("{")
    elif previous == "incomplete":
        write_summary(store.path, summary.model_copy(update={"completed": False}))
    else:
        status = status.model_copy(update={"asked": status.asked + 1})
    module.record_crossover_error(store, status, KeyboardInterrupt())
    current = read_summary(store.path)
    assert current is not None and not current.completed and current.state == "stopped"
    assert current.reason_text == STOPPED_REASON


def test_recording_failure_prints_exception_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    store, _, status = prepared(tmp_path)
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *args, **kwargs: None)
    def broken(*args: object, **kwargs: object) -> object:
        raise AssertionError()
    monkeypatch.setattr(cli, "attach_crossover", broken)
    monkeypatch.setattr(cli, "record_crossover_error", broken)
    code = _run(store, tmp_path)
    assert code == 0
    text = capsys.readouterr().err
    assert "摘要未能保存：AssertionError：" in text and "搜尋結果不受影響：AssertionError：" in text


@pytest.mark.parametrize("recorded", [False, True])
def test_in_process_stop_leaves_signal_handlers_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      recorded: bool) -> None:
    """只有命令列入口接管了終止訊號時，收尾才改成忽略後續訊號。

    考卷在同一行程直接呼叫 main：改掉的處理會留給之後的考卷，它們開的子行程也會繼承「忽略」。
    """
    store, _, status = prepared(tmp_path)
    monkeypatch.setattr(cli, "auto_search", lambda *args, **kwargs: status)

    def stopped(*args: object, **kwargs: object) -> object:
        raise AttachmentRecorded(KeyboardInterrupt(), stopped=True) if recorded else KeyboardInterrupt()

    monkeypatch.setattr(cli, "attach_modal", stopped)
    monkeypatch.setattr(cli, "record_attachment_error", lambda *args, **kwargs: None)
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    _run(store, tmp_path)
    assert {s: signal.getsignal(s) for s in before} == before
