"""第二輪複查：雙停止、現任入圍進度、保留點與完成的跳過摘要。"""
from dataclasses import replace
import os
from pathlib import Path
import signal

import pytest

from aosr.search import placement_stability_attach as module
from aosr.search.outer_status import OuterStatus
from aosr.search.placement_stability_record import StabilitySummary, read_summary, summary_path
from aosr.search.refine import RefineLedger
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore
from tests.engine._search_run_cases import RUN_DATE
from tests.engine._stability_attach_cases import ShiftCompute, attach, ready
from tests.engine.test_search_stability_finish_errors import setup_auto
from tests.engine.test_search_stability_reuse import change_rows


@pytest.mark.parametrize("where", ["pinned", "prepare"])
def test_double_sigterm_keeps_previous_points_before_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           where: str) -> None:
    from aosr.search import cli
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    change_rows(store)
    with monkeypatch.context() as patch:
        args = setup_auto(store, registry, status, patch)
        def stop(*args: object, **kwargs: object) -> None:
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("第一個停止沒有走真正訊號處理")
        patch.setattr(module if where == "pinned" else module._Attacher,
                      "_pinned" if where == "pinned" else "prepare", stop)
        original = module.record_stability_error
        interrupted = False
        def recording(opened: SearchStore, current: SearchStatus, error: BaseException, *,
                      previous: StabilitySummary | None = None) -> None:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                os.kill(os.getpid(), signal.SIGTERM)
                raise AssertionError("第二個停止沒有走真正訊號處理")
            original(opened, current, error, previous=previous)
        patch.setattr(module, "record_stability_error", recording)
        handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        try:
            cli._interrupt_on_termination()
            code = cli.main(args, compute_factory=lambda *a: lambda *b: iter(()),
                stability_compute_factory=lambda root: ShiftCompute(store))
        finally:
            for s, handler in handlers.items():
                signal.signal(s, handler)
    saved = read_summary(store.path)
    assert code == 0 and interrupted
    assert saved is not None and saved.state == "stopped" and saved.points == first.points
    resumed = ShiftCompute(store)
    assert attach(store, registry, status, resumed).state == "done" and not resumed.jobs


def test_removed_finalist_is_retained_outside_progress_and_reused_on_return(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    before_book = store.refine_ledger_path.read_bytes()
    header, rows = RefineLedger.read(store.refine_ledger_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header)
    for row in rows:
        book.append(row.model_copy(update={"outcome": "not_comparable", "total_cost": None})
                    if row.trial_number == 9 else row)
    removed = tuple(p for p in first.points if p.trial_number == 9)
    files = {store.path / p.result_file: (store.path / p.result_file).read_bytes()
             for p in removed if p.result_file is not None}
    missing = tuple(p for p in first.points if p.trial_number == 7)[:2]
    for point in missing:
        assert point.result_file is not None
        (store.path / point.result_file).unlink()
    compute = ShiftCompute(store, fail_after=1)
    with pytest.raises(RuntimeError, match="某點計算失敗原文"):
        attach(store, registry, status, compute)
    failed = read_summary(store.path)
    assert failed is not None and failed.state == "failed"
    assert {f.trial_number for f in failed.selection.finalists} == {7, None}
    assert {p.trial_number for p in failed.points} == {7, None}
    current = tuple(p for p in first.points if p.trial_number in (7, None))
    assert failed.total_points == len(current)
    assert failed.computed_points == len(current) - len(missing) + len(compute.jobs)
    assert failed.retained_points == removed
    for _ in range(2):
        change_rows(store)
        complete = attach(store, registry, status, ShiftCompute(store))
        assert complete.retained_points == removed
        assert {p: p.read_bytes() for p in files} == files
        assert complete.arithmetic is not None
        assert {r.finalist.trial_number for r in complete.arithmetic.finalists} == {7, None}
    store.refine_ledger_path.write_bytes(before_book)
    returning = ShiftCompute(store)
    restored = attach(store, registry, status, returning)
    assert {p.trial_number for p in restored.points} == {7, 9, None} and not returning.jobs
    assert restored.retained_points == ()


@pytest.mark.parametrize("where", ["identity_skip", "outer_skip", "no_finalists", "pinned", "compute"])
def test_unplaceable_points_do_not_inflate_skip_or_error_progress(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    from aosr.config.capabilities import CapabilityTable
    from aosr.config.directivity_defaults import DirectivityDefaults
    from aosr.reporting.scheme import Scheme
    from aosr.search.constraints import Reason, Violation
    from aosr.search.placement_stability_geometry import Legality, Shift, check_shift
    store, registry, status = ready(tmp_path)
    def check(project: Scheme, shift: Shift, *, contact_rel: float, capabilities: CapabilityTable,
              directivity: DirectivityDefaults) -> Legality:
        return Legality("unplaceable", violations=(Violation(Reason.SEAT_OUTSIDE_ROOM, 0.01),)) if shift.name == "ear_up" else check_shift(
            project, shift, contact_rel=contact_rel, capabilities=capabilities, directivity=directivity)
    monkeypatch.setattr(module, "check_shift", check)
    first = attach(store, registry, status, ShiftCompute(store))
    assert any(p.outcome == "unplaceable" for p in first.points)
    change_rows(store)
    if where in ("identity_skip", "outer_skip", "no_finalists"):
        if where == "no_finalists":
            header, rows = RefineLedger.read(store.refine_ledger_path)
            store.refine_ledger_path.unlink()
            book = RefineLedger.create(store.refine_ledger_path, header)
            for row in rows:
                book.append(row.model_copy(update={"outcome": "not_comparable", "total_cost": None}))
        summary = module.attach_stability(store, status=status.model_copy(update={"outer": OuterStatus(conclusion="user_stopped")})
            if where == "outer_skip" else status, quality_targets_path=registry, run_date=RUN_DATE,
            probe=lambda: replace(store.identity, program_fingerprint="changed") if where == "identity_skip" else store.identity,
            compute_factory=lambda root: ShiftCompute(store))
        assert summary.points == () and summary.retained_points == first.points
        assert summary.total_points == summary.computed_points == 0
    else:
        def fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError("重建前段失敗原文")
        if where == "pinned":
            monkeypatch.setattr(module, "_pinned", fail)
        else:
            victim = next(p for p in first.points if p.result_file is not None)
            assert victim.result_file is not None
            (store.path / victim.result_file).unlink()
        with pytest.raises(RuntimeError):
            attach(store, registry, status, ShiftCompute(store, fail_after=0))
        failed = read_summary(store.path)
        assert failed is not None and failed.state == "failed"
        assert failed.total_points == first.total_points
        assert failed.computed_points == first.computed_points - (where == "compute")
        assert any(p.outcome == "unplaceable" for p in failed.points)


@pytest.mark.parametrize("kind", ["identity", "outer", "no_finalists"])
def test_fresh_skipped_summary_survives_late_stop_byte_for_byte(tmp_path: Path, kind: str) -> None:
    store, registry, status = ready(tmp_path)
    if kind == "no_finalists":
        header, rows = RefineLedger.read(store.refine_ledger_path)
        store.refine_ledger_path.unlink()
        book = RefineLedger.create(store.refine_ledger_path, header)
        for row in rows:
            book.append(row.model_copy(update={"outcome": "not_comparable", "total_cost": None}))
    if kind == "outer":
        status = status.model_copy(update={"outer": OuterStatus(conclusion="user_stopped")})
    skipped = module.attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
        probe=lambda: replace(store.identity, program_fingerprint="changed") if kind == "identity" else store.identity,
        compute_factory=lambda root: ShiftCompute(store))
    assert skipped.state == "skipped" and skipped.completed
    before = summary_path(store.path).read_bytes()
    module.record_stability_error(store, status, KeyboardInterrupt())
    assert summary_path(store.path).read_bytes() == before


def test_stale_skipped_summary_can_be_recorded_as_stopped(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    skipped = module.attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
        probe=lambda: replace(store.identity, program_fingerprint="changed"),
        compute_factory=lambda root: ShiftCompute(store))
    assert skipped.state == "skipped"
    change_rows(store)
    module.record_stability_error(store, status, KeyboardInterrupt())
    stopped = read_summary(store.path)
    assert stopped is not None and stopped.state == "stopped" and not stopped.completed
