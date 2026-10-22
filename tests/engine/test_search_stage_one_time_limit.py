"""搜尋成本上限：真子行程替身、獨立手算時間與整群清理。"""
import os
import sys
from collections.abc import Sequence
from threading import current_thread, main_thread, local
import subprocess
from pathlib import Path

import pytest

from aosr.search.stage_one_worker import StageOneCompute
from aosr.search.run import CandidateJob
from tests.engine.test_search_stage_one_worker import setup

SCRIPT = '''
import os, sys, time
from pathlib import Path
phase = sys.argv[1]
args = sys.argv[2:]
root = Path(args[args.index('--capabilities') + 1])
out = Path(args[args.index('--out') + 1])
(root / (phase + '-' + str(os.getpid()) + '.pid')).write_text(str(os.getpid()))
delay = root / (phase + '-sleep')
if phase == 'candidates':
    specific_delay = root / ('candidates-' + args[args.index('--trial-number') + 1] + '-sleep')
    if specific_delay.exists():
        delay = specific_delay
if delay.exists():
    time.sleep(float(delay.read_text()))
if phase == 'slices':
    out.write_text('{}')
else:
    number = args[args.index('--trial-number') + 1]
    specific = root / ('template-' + number)
    out.write_bytes((specific if specific.exists() else root / 'template').read_bytes())
'''


def capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: float | None) -> tuple[StageOneCompute, CandidateJob]:
    worker, job = setup(tmp_path, monkeypatch)
    script = tmp_path / "clock-child.py"
    script.write_text(SCRIPT)
    return StageOneCompute(capabilities_path=tmp_path, engine_commit="test", search_id="test",
        fem_root=tmp_path / "parts", identity=worker.identity, search_cap_s=cap, poll_s=.005,
        runner=(sys.executable, str(script), "candidates"), slice_runner=(sys.executable, str(script), "slices")), job


def assert_no_children(tmp_path: Path) -> None:
    pids = tuple(tmp_path.glob("*.pid"))
    assert pids
    for path in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.read_text()), 0)


def test_slice_overrun_stops_whole_batch_before_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_worker import CandidateTimeLimit
    from tests.engine.test_search_stage_one_worker import _second_job
    worker, job = capped(tmp_path, monkeypatch, .15)
    other = _second_job(tmp_path, job)
    other = CandidateJob(other.trial_number, other.scheme.model_copy(update={
        "scene": other.scheme.scene.model_copy(update={"reflection_order_k": 3})}), other.result_path)
    (tmp_path / "slices-sleep").write_text("10")
    with pytest.raises(CandidateTimeLimit) as caught:
        list(worker((job, other), 1))
    record = caught.value.record
    assert record.phase == "slices"
    assert [(item.trial_number, item.state) for item in record.candidates] == [(0, "超限"), (1, "沒開始")]
    assert record.batch is None
    assert not tuple(tmp_path.glob("candidates-*.pid"))
    assert_no_children(tmp_path)


@pytest.mark.parametrize("cap", [0., -1., float("inf"), float("nan")])
def test_invalid_search_cap_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: float) -> None:
    worker, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="search_cap_s"):
        StageOneCompute(capabilities_path=tmp_path, engine_commit="test", search_id="test",
            fem_root=tmp_path / "parts", identity=worker.identity, search_cap_s=cap)


@pytest.mark.parametrize("candidate_end,over", [(10., False), (10.25, True)])
def test_exact_service_boundary_and_completed_overrun_is_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               candidate_end: float, over: bool) -> None:
    from aosr.search import worker as slices, stage_one_worker as candidates
    from aosr.search.stage_one_worker import CandidateTimeLimit
    from tests.engine.test_search_stage_one_worker import _second_job
    worker, job = capped(tmp_path, monkeypatch, 10.)
    other = _second_job(tmp_path, job)
    from aosr.search.worker import _Active
    launched: list[CandidateJob] = []
    start = worker.processes._start
    def capture(job: CandidateJob, parts: Sequence[Path] = ()) -> _Active:
        launched.append(job)
        return start(job, parts)
    monkeypatch.setattr(worker.processes, "_start", capture)
    monkeypatch.setattr(slices, "_clock", lambda: 0. if current_thread() is main_thread() else 4.)
    monkeypatch.setattr(candidates, "_clock", lambda: 4. if current_thread() is main_thread() else candidate_end)
    results = worker((job, other), 1)
    first = next(results)
    assert first.job == job and first.seconds == candidate_end - 4.
    if over:
        with pytest.raises(CandidateTimeLimit) as caught:
            next(results)
        assert [(item.trial_number, item.state, item.service_s) for item in caught.value.record.candidates] == [
            (0, "交回但超限", 10.25), (1, "沒開始", 4.)]
        assert not (tmp_path / "other.json").exists()
        assert launched == [job]
    else:
        assert [result.job for result in results] == [other]
    assert_no_children(tmp_path)


@pytest.mark.parametrize("payload", [
    {"extra": "拒收"}, {"cap_s": float("inf")}, {"cap_s": float("nan")},
    {"candidates": [{"trial_number": 0, "state": "超限", "service_s": float("nan")}]},
])
def test_pause_record_rejects_extra_and_nonfinite_values(payload: dict[str, object]) -> None:
    from pydantic import ValidationError
    from aosr.search.stage_one_time_limit import StageOneTimeLimit
    with pytest.raises(ValidationError):
        StageOneTimeLimit.model_validate({"cap_s": 600., "phase": "slices", "candidates": [], "groups": []} | payload)


def test_queued_candidate_completion_time_does_not_include_consumer_wait(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from threading import Semaphore
    from aosr.search import worker as slices, stage_one_worker as candidates
    from aosr.search.stage_one_worker import _StampedQueue
    from aosr.search.worker import _Active
    from tests.engine.test_search_stage_one_worker import _second_job
    worker, job = capped(tmp_path, monkeypatch, 10.)
    other = _second_job(tmp_path, job)
    now = [4.]
    queued = Semaphore(0)
    monkeypatch.setattr(slices, "_clock", lambda: 0. if current_thread() is main_thread() else 4.)
    monkeypatch.setattr(candidates, "_clock", lambda: now[0] if current_thread() is main_thread() else 10.)
    put = _StampedQueue.put
    def counted(self: _StampedQueue, item: _Active, block: bool = True, timeout: float | None = None) -> None:
        put(self, item, block, timeout)
        queued.release()
    monkeypatch.setattr(_StampedQueue, "put", counted)
    results = worker((job, other), 2)
    first = next(results)
    assert queued.acquire(timeout=2.) and queued.acquire(timeout=2.)
    now[0] = 1000.
    remaining = list(results)
    assert {result.job.trial_number for result in (first, *remaining)} == {0, 1}
    assert all(result.seconds == 6. for result in (first, *remaining))
    assert_no_children(tmp_path)


def test_candidate_overrun_keeps_returned_stops_peers_and_never_launches_pending(tmp_path: Path,
                                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import worker as slices, stage_one_worker as candidates
    from aosr.search.stage_one_worker import CandidateTimeLimit
    from tests.engine.test_search_stage_one_worker import _second_job
    from aosr.search.worker import _Active
    worker, job = capped(tmp_path, monkeypatch, 10.)
    second = _second_job(tmp_path, job)
    third = CandidateJob(2, job.scheme, tmp_path / "third.json")
    fourth = CandidateJob(3, job.scheme, tmp_path / "fourth.json")
    now = [4.]
    monkeypatch.setattr(slices, "_clock", lambda: 0. if current_thread() is main_thread() else 4.)
    monkeypatch.setattr(candidates, "_clock", lambda: now[0] if current_thread() is main_thread() else 5.)
    for number in (1, 2):
        (tmp_path / f"candidates-{number}-sleep").write_text("10")
    results = worker((job, second, third, fourth), 2)
    assert next(results).job == job
    # 第一題交回後第二題已服務 7 秒；第三題此時才開始，服務 0 秒。
    now[0] = 8.
    original_start = worker.processes._start
    launched: list[_Active] = []
    def advance(job: CandidateJob, parts: Sequence[Path] = ()) -> _Active:
        item = original_start(job, parts)
        launched.append(item)
        now[0] = 11.
        return item
    monkeypatch.setattr(worker.processes, "_start", advance)
    with pytest.raises(CandidateTimeLimit) as caught:
        next(results)
    assert [(row.trial_number, row.state, row.service_s) for row in caught.value.record.candidates] == [
        (0, "交回", 5.), (1, "超限", 11.), (2, "一起停下", 7.), (3, "沒開始", 4.)]
    assert not fourth.result_path.exists()
    assert [item.job.trial_number for item in launched] == [third.trial_number]
    for item in launched:
        with pytest.raises(ProcessLookupError):
            os.kill(item.process.pid, 0)
    assert_no_children(tmp_path)


def test_disabled_cap_returns_identical_jobs_and_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.engine.test_search_stage_one_worker import _second_job
    worker, job = capped(tmp_path, monkeypatch, None)
    other = _second_job(tmp_path, job)
    expected = {item.job.trial_number: (item.job, item.result.model_dump_json()) for item in worker((job, other), 2)}
    worker.search_cap_s = 600.
    actual = {item.job.trial_number: (item.job, item.result.model_dump_json()) for item in worker((job, other), 2)}
    assert actual == expected


def test_two_groups_use_own_longest_slice_and_report_queue_window(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import worker as slices
    from aosr.search.worker import _Slice, _SliceStampedQueue
    from aosr.search.stage_one_time_limit import TimeLimitTracker
    from tests.engine.test_search_stage_one_worker import _second_job
    worker, job = capped(tmp_path, monkeypatch, 100.)
    other = _second_job(tmp_path, job)
    other = CandidateJob(1, other.scheme.model_copy(update={
        "scene": other.scheme.scene.model_copy(update={"reflection_order_k": 3})}), other.result_path)
    third = CandidateJob(2, job.scheme, tmp_path / "third.json")
    jobs = (job, other, third)
    workers = 2
    # 組一：0→8、2→11，最長 9、窗口 11、排隊 2。
    # 組二：20→25、25→31，最長 6、窗口 11、排隊 5。
    schedule = iter(((8., 2.), (11., 20.), (25., 25.), (31., 40.)))
    now = [0.]
    ends: dict[int, float] = {}
    thread_time = local()
    spawn = worker.processes._spawn
    def timed_spawn(command: Sequence[str], stderr: Path) -> subprocess.Popen[bytes]:
        process = spawn(command, stderr)
        end, following = next(schedule)
        ends[process.pid] = end
        now[0] = following
        return process
    put = _SliceStampedQueue.put
    def stamped(self: _SliceStampedQueue, item: _Slice, block: bool = True, timeout: float | None = None) -> None:
        thread_time.value = ends[item.process.pid]
        put(self, item, block, timeout)
    monkeypatch.setattr(worker.processes, "_spawn", timed_spawn)
    monkeypatch.setattr(_SliceStampedQueue, "put", stamped)
    monkeypatch.setattr(slices, "_clock", lambda: now[0] if current_thread() is main_thread() else float(thread_time.value))
    tracker = TimeLimitTracker(jobs, 100.)
    def wake(now: float) -> None:
        assert tracker.slices_limit(now) is None
    parts = worker.processes._parts_by_job(jobs, workers, observe=tracker.observe,
        wake=wake)
    record = tracker.snapshot("slices", 40.)
    assert parts[0] == parts[2] and parts[0] != parts[1]
    assert [(group.trial_numbers, group.longest_slice_s, group.window_s, group.queue_s) for group in record.groups] == [
        ((0, 2), 9., 11., 2.), ((1,), 6., 11., 5.)]
    assert all(group.slices == workers and group.completed_slices == workers for group in record.groups)
    assert [(row.trial_number, row.slice_s, row.candidate_s, row.service_s) for row in record.candidates] == [
        (0, 9., None, 9.), (1, 6., None, 6.), (2, 9., None, 9.)]
    assert_no_children(tmp_path)


def test_completed_slice_overrun_uses_stamp_and_starts_no_new_group(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search import worker as slices
    from aosr.search.stage_one_worker import CandidateTimeLimit
    from tests.engine.test_search_stage_one_worker import _second_job
    worker, job = capped(tmp_path, monkeypatch, 10.)
    same = _second_job(tmp_path, job)
    later = CandidateJob(2, job.scheme.model_copy(update={
        "scene": job.scheme.scene.model_copy(update={"reflection_order_k": 3})}), tmp_path / "later.json")
    monkeypatch.setattr(slices, "_clock", lambda: 0. if current_thread() is main_thread() else 10.25)
    with pytest.raises(CandidateTimeLimit) as caught:
        list(worker((job, same, later), 1))
    record = caught.value.record
    assert [(row.trial_number, row.state, row.service_s) for row in record.candidates] == [
        (0, "超限", 10.25), (1, "超限", 10.25), (2, "沒開始", None)]
    assert [group.completed_slices for group in record.groups] == [1, 0]
    assert not tuple(tmp_path.glob("candidates-*.pid"))
    assert_no_children(tmp_path)


def test_completed_earlier_group_is_not_marked_as_stopped_mid_computation(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.stage_one_time_limit import TimeLimitTracker
    from aosr.search.worker import _Slice
    worker, job = capped(tmp_path, monkeypatch, 10.)
    other = CandidateJob(1, job.scheme.model_copy(update={
        "scene": job.scheme.scene.model_copy(update={"reflection_order_k": 3})}), tmp_path / "other.json")
    tracker = TimeLimitTracker((job, other), 10.)
    with subprocess.Popen((sys.executable, "-c", "pass")) as process:
        process.wait()
        for candidate, seconds in ((job, 4.), (other, 11.)):
            tracker.observe((candidate,), 1, None)
            tracker.observe((candidate,), 1, _Slice(0, process, tmp_path / "stderr", started=0., ended=seconds))
        record = tracker.slices_limit(11.)
    assert record is not None
    assert [(row.trial_number, row.state) for row in record.candidates] == [(0, "沒開始"), (1, "超限")]


def test_queued_overrun_is_returned_when_its_completion_is_read(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from threading import Event, Semaphore
    from aosr.search import worker as slices, stage_one_worker as candidates
    from aosr.search.stage_one_worker import CandidateTimeLimit, _StampedQueue
    from aosr.search.worker import _Active
    from tests.engine.test_search_stage_one_worker import _second_job
    worker, job = capped(tmp_path, monkeypatch, 10.)
    other = _second_job(tmp_path, job)
    first_queued, queued = Event(), Semaphore(0)
    thread_time = local()
    monkeypatch.setattr(slices, "_clock", lambda: 0. if current_thread() is main_thread() else 4.)
    monkeypatch.setattr(candidates, "_clock", lambda: 4. if current_thread() is main_thread() else float(thread_time.value))
    put, get = _StampedQueue.put, _StampedQueue.get
    def stamped(self: _StampedQueue, item: _Active, block: bool = True, timeout: float | None = None) -> None:
        thread_time.value = 5. if item.job.trial_number == 0 else 10.25
        put(self, item, block, timeout)
        queued.release()
        if item.job.trial_number == 0:
            first_queued.set()
    start = worker.processes._start
    def after_first(job: CandidateJob, parts: Sequence[Path] = ()) -> _Active:
        if job.trial_number == 1:
            assert first_queued.wait(timeout=2.)
        return start(job, parts)
    ready = [False]
    def both_ready(self: _StampedQueue, block: bool = True, timeout: float | None = None) -> _Active:
        if not ready[0]:
            assert queued.acquire(timeout=2.) and queued.acquire(timeout=2.)
            ready[0] = True
        return get(self, block, timeout)
    monkeypatch.setattr(worker.processes, "_start", after_first)
    monkeypatch.setattr(_StampedQueue, "put", stamped)
    monkeypatch.setattr(_StampedQueue, "get", both_ready)
    results = worker((job, other), 2)
    assert next(results).job == job
    returned = next(results)
    assert returned.job == other and returned.seconds == 6.25
    with pytest.raises(CandidateTimeLimit) as caught:
        next(results)
    assert [(row.trial_number, row.state) for row in caught.value.record.candidates] == [(0, "交回"), (1, "交回但超限")]
    assert_no_children(tmp_path)
