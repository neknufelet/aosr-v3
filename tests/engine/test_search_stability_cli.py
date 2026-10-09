"""三段收尾順序、持鎖、前段停止、獨立出口與選取隔離。"""
from __future__ import annotations

import fcntl
import os
import signal
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.search import cli
from aosr.search.outer_status import OuterStatus
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.store import SearchStore
from aosr.config.paths import config_path
from aosr.search.crossover_record import STOPPED_NOTE as CROSSOVER_STOPPED_NOTE
from aosr.search.modal_attach import STOPPED_NOTE as MODAL_STOPPED_NOTE
from tests.engine._crossover_cases import protected
from tests.engine._stability_attach_cases import ready


@pytest.mark.parametrize("conclusion,code", [("complete", 0), ("refine_budget", 0), ("refine_failed", 1), ("refine_interrupted", 3)])
@pytest.mark.parametrize("action", ["success", "failure", "stop"])
def test_auto_preserves_bytes_exit_lock_and_attachment_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], conclusion: str, code: int, action: str) -> None:
    from aosr.search.placement_stability_record import read_summary, STOPPED_NOTE
    store, _, status = ready(tmp_path)
    status = status.model_copy(update={"outer": OuterStatus.model_validate(dict(conclusion=conclusion))})
    events: list[str] = []
    monkeypatch.setattr(cli, "auto_search", lambda *a, **k: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *a, **k: events.append("modal"))
    monkeypatch.setattr(cli, "attach_crossover", lambda *a, **k: events.append("crossover"))
    def stability(opened: SearchStore, **kwargs: object) -> None:
        events.append("stability")
        descriptor = opened.open_folder_lock()
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        if action == "stop":
            raise KeyboardInterrupt
        if action == "failure":
            raise RuntimeError("穩定性故障原文")
    monkeypatch.setattr(cli, "attach_stability", stability)
    before = protected(store)
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    actual = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()))
    assert actual == code and protected(store) == before
    assert events == ["modal", "crossover", "stability"]
    stderr = capsys.readouterr().err
    if action == "success":
        assert not stderr
    else:
        summary = read_summary(store.path)
        assert summary is not None and summary.state == ("stopped" if action == "stop" else "failed")
        assert stderr.splitlines() == [STOPPED_NOTE] if action == "stop" else "穩定性故障原文" in stderr
        assert "交接敏感度" not in stderr


@pytest.mark.parametrize("stage", ["modal", "crossover"])
def test_prior_stop_records_stability_without_starting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], stage: str) -> None:
    from aosr.search.placement_stability_record import read_summary
    store, _, status = ready(tmp_path)
    def stop(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "auto_search", lambda *a, **k: status)
    monkeypatch.setattr(cli, "attach_modal", stop if stage == "modal" else lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_crossover", stop)
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("前段被停仍開跑")
    monkeypatch.setattr(cli, "attach_stability", forbidden)
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()))
    assert code == 0
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "stopped" and not summary.completed
    assert summary.points == ()
    note = MODAL_STOPPED_NOTE if stage == "modal" else CROSSOVER_STOPPED_NOTE
    assert capsys.readouterr().err.splitlines() == [note]


def test_select_only_reads_refinement_even_with_shift_results(tmp_path: Path) -> None:
    from aosr.search.select import select_refined
    from tests.engine._stability_attach_cases import ShiftCompute, attach
    store, registry, status = ready(tmp_path)
    summary = attach(store, registry, status, ShiftCompute(store))
    point = next(p for p in summary.points if p.trial_number == 7)
    assert point.result_file is not None
    shifted = store.path / point.result_file
    selected = select_refined(store, 7, tmp_path / "data")
    assert selected.result_path.read_bytes() == store.refine_result_path(7).read_bytes()
    assert selected.result_path.read_bytes() != shifted.read_bytes()
    # 移位檔即使冒用細算代號也要核出處，不能混進結果清單。
    from aosr.reporting.result import SchemeResult
    result = SchemeResult.model_validate_json(store.refine_result_path(7).read_bytes())
    shift_result = SchemeResult.model_validate_json(shifted.read_bytes())
    store.refine_result_path(7).write_text(result.model_copy(update={"origin": shift_result.origin}).model_dump_json())
    with pytest.raises(ValueError, match="出處"):
        select_refined(store, 7, tmp_path / "second-data")


def test_finish_passes_factory_own_fem_root_and_lock_fd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.worker import SubprocessCompute
    from tests.engine._stability_attach_cases import ShiftCompute
    store, registry, status = ready(tmp_path)
    monkeypatch.setattr(cli, "auto_search", lambda *a, **k: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_crossover", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    # 讓命令列讀考卷自己的登記簿；模型身分與評分設定由寫入端提供。
    original_config = config_path
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name == "quality_targets.toml" else original_config(name))
    compute = ShiftCompute(store)
    observed: list[tuple[Path, int | None]] = []
    def calculated(worker: SubprocessCompute, jobs: Sequence[CandidateJob], max_workers: int) -> Iterator[ComputedCandidate]:
        observed.append((worker.fem_root, worker.lock_fd))
        assert worker.lock_fd is not None
        assert os.fstat(worker.lock_fd).st_ino == store.path.stat().st_ino
        return compute(jobs, max_workers)
    monkeypatch.setattr(SubprocessCompute, "__call__", calculated)
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()))
    assert code == 0
    assert observed and all(root.parent == store.path / "placement-stability" for root, _ in observed)


def test_broken_stability_summary_is_repaired_inside_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.placement_stability_record import read_summary, summary_path
    from tests.engine._stability_attach_cases import ShiftCompute
    store, registry, status = ready(tmp_path)
    summary_path(store.path).parent.mkdir()
    summary_path(store.path).write_text("{broken")
    monkeypatch.setattr(cli, "auto_search", lambda *a, **k: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_crossover", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    original_config = config_path
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name == "quality_targets.toml" else original_config(name))
    before = protected(store)
    args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
    code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()),
        stability_compute_factory=lambda root: ShiftCompute(store))
    assert code == 0
    summary = read_summary(store.path)
    assert summary is not None and summary.completed and protected(store) == before


def test_sigterm_during_stability_is_recorded_then_following_signals_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from aosr.search.placement_stability_record import read_summary, STOPPED_NOTE
    store, _, status = ready(tmp_path)
    monkeypatch.setattr(cli, "auto_search", lambda *a, **k: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_crossover", lambda *a, **k: None)
    def interrupted(*args: object, **kwargs: object) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("終止訊號沒有轉成中斷")
    monkeypatch.setattr(cli, "attach_stability", interrupted)
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        cli._interrupt_on_termination()
        args = ["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")]
        code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()))
        assert code == 0
        assert all(signal.getsignal(s) == signal.SIG_IGN for s in previous)
        for s in previous:
            os.kill(os.getpid(), s)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "stopped"
    assert capsys.readouterr().err.splitlines() == [STOPPED_NOTE]
