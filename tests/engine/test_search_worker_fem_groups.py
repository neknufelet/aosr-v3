"""不同場景的分片隔離、全批屏障與整群清理；不解有限元素。"""
from __future__ import annotations

import json
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from queue import Queue
from threading import Lock, Thread, get_ident

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.reporting.scheme_cli import PROGRAM_CHANGED_EXIT, PROGRAM_CHANGED_MARKER
from aosr.search.run import CandidateJob, ComputeFailed, IdentityChanged
from aosr.search.store import SearchStore
from aosr.search.worker import _check_exit as real_check_exit
from aosr.search.worker import _Slice
from aosr.search.worker import _watch as real_watch
from tests.engine._search_worker_group_cases import (
    capture_commands,
    configure_slices,
    count_unreaped_slices,
    expected_candidate,
    expected_group,
    mixed_rooms,
    wait_until,
    with_axis,
)
from tests.engine.test_search_stage_one_worker import setup as setup_stage_one
from tests.engine.test_search_worker import assert_dead, events, setup_worker
from tests.engine.test_search_worker_fem import groups_are_dead, slice_events


@pytest.mark.parametrize("axis", [None, LowFrequencyAxis.SEARCH_48, LowFrequencyAxis.VERIFICATION])
def test_single_group_keeps_exact_commands_and_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, axis: LowFrequencyAxis | None,
) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = tuple(with_axis(job, axis) for job in jobs)
    commands = capture_commands(worker, monkeypatch)
    limit = 3
    list(worker(jobs, limit))
    records = slice_events(tmp_path)
    parts = expected_group(worker, jobs, limit, records, commands)
    assert list(worker.fem_root.iterdir()) == [Path(parts[0]).parent]
    assert commands == [command for command in commands if "--slice" in command] + [
        expected_candidate(worker, job, parts) for job in jobs]
    assert all(row["parts"] == list(parts) for row in events(tmp_path))
    groups_are_dead(records)


@pytest.mark.parametrize("different_axes", [False, True])
def test_interleaved_rooms_get_only_their_own_ordered_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, different_axes: bool,
) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    if different_axes:
        jobs = (with_axis(jobs[0], LowFrequencyAxis.SEARCH_48),
                with_axis(jobs[1], LowFrequencyAxis.VERIFICATION), with_axis(jobs[2], LowFrequencyAxis.SEARCH_48))
    commands = capture_commands(worker, monkeypatch)
    limit = 3
    list(worker(jobs, limit))
    records = slice_events(tmp_path)
    a = expected_group(worker, (jobs[0], jobs[2]), limit, records, commands)
    b = expected_group(worker, (jobs[1],), limit, records, commands)
    assert Path(a[0]).parent != Path(b[0]).parent
    assert set(worker.fem_root.iterdir()) == {Path(a[0]).parent, Path(b[0]).parent}
    assert [command[command.index("--out") + 1] for command in commands if "--slice" in command] == [*a, *b]
    assert [command for command in commands if "--slice" not in command] == [
        expected_candidate(worker, job, parts) for job, parts in zip(jobs, (a, b, a), strict=True)]
    by_number = {str(row["parts"]): row for row in events(tmp_path)}
    assert set(by_number) == {str(list(a)), str(list(b))}
    assert all(float(str(row["start"])) >= max(record["end"] for record in records) for row in events(tmp_path))
    groups_are_dead(records)


@pytest.mark.parametrize("field,value", [("low_frequency_axis", LowFrequencyAxis.SEARCH),
                                        ("reflection_order_k", 3), ("scattering_by_wall", {"floor": 0.2})])
def test_scene_serialization_keeps_finer_groups(tmp_path: Path, field: str, value: object) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = tuple(with_axis(job, None) for job in jobs)
    changed = replace(jobs[1], scheme=jobs[1].scheme.model_copy(update={
        "scene": jobs[1].scheme.scene.model_copy(update={field: value})}))
    jobs = (jobs[0], changed, jobs[2])
    list(worker(jobs, 1))
    expected = [[str(SearchStore.scheme_path_for(job.result_path)) for job in group]
                for group in ((jobs[0], jobs[2]), (jobs[1],))]
    assert sorted(row["schemes"] for row in slice_events(tmp_path)) == sorted(expected)


def test_all_groups_share_worker_limit_and_each_slice_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    limit = 3
    # 第一組各片等主迴圈真的停下來等通知（阻塞的 get）才准結束：正確程式滿載時一定先走到那一步，
    # 多開一支的程式會在那之前開第四片，那時前三片一定還沒收掉（複查二）。
    options: dict[str, object] = {f"job-0-slice-{index}": {"hold": "release-first-group"} for index in range(limit)}
    options.update({f"job-1-slice-{index}": {"sleep": 0.15} for index in range(limit)})
    configure_slices(tmp_path, worker, options)
    commands = capture_commands(worker, monkeypatch)
    unreaped = count_unreaped_slices(worker, monkeypatch)
    release = tmp_path / "release-first-group"
    released_after: list[int] = []

    class ReleasingQueue(Queue[_Slice]):
        def get(self, block: bool = True, timeout: float | None = None) -> _Slice:
            # 標準 Queue.get_nowait() 就是 self.get(block=False)，排水也會走到這裡；
            # 只有阻塞的 get 才放行，並記下放行時已經開了幾片（複查三）。
            if block and not release.exists():
                released_after.append(len(unreaped))
                release.touch()
            return super().get(block, timeout)

    monkeypatch.setattr("aosr.search.worker.Queue", ReleasingQueue)
    list(worker(jobs, limit))
    # 父行程開每一片之前，還沒收掉（returncode 仍是 None）的分片都算活著，這是上限的保守量法（審查）。
    assert unreaped and max(unreaped) < limit
    assert unreaped[limit - 1] == limit - 1
    # 上面兩條的前提：放行之前第一組一片都結束不了，第一次放行時恰好開滿一輪；前提不成立也紅（複查三）。
    assert released_after == [limit]
    records = slice_events(tmp_path)
    parts = (*expected_group(worker, (jobs[0], jobs[2]), limit, records, commands),
             *expected_group(worker, (jobs[1],), limit, records, commands))
    assert sorted(row["out"] for row in records) == sorted(parts)
    assert len({row["pid"] for row in records}) == len(parts)
    assert len(parts) > limit
    changes = sorted([(row["start"], 1) for row in records] + [(row["end"], -1) for row in records])
    active = 0
    for _, delta in changes:
        active += delta
        assert 0 <= active <= limit
    assert active == 0
    groups_are_dead(records)


def test_each_group_plans_own_axis_bounded_count(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    a = (with_axis(jobs[0], LowFrequencyAxis.SEARCH_48), with_axis(jobs[2], LowFrequencyAxis.SEARCH_48))
    b = (with_axis(jobs[1], LowFrequencyAxis.VERIFICATION),)
    limit = len(low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH_48)[0]) + 1
    for group in (a, b):
        parts, commands = worker._slice_commands(group, limit)
        axis = group[0].scheme.scene.low_frequency_axis or LowFrequencyAxis.SEARCH
        count = min(limit, len(low_frequency_axis_frequencies(axis)[0]))
        assert [path.name for path in parts] == [f"slice-{index}.json" for index in range(count)]
        assert [command[command.index("--slices") + 1] for command in commands] == [str(count)] * count
        assert [command[command.index("--slice") + 1] for command in commands] == [str(index) for index in range(count)]
        assert all(command[len(worker.slice_runner):command.index("--slice")] == tuple(
            str(SearchStore.scheme_path_for(job.result_path)) for job in group) for command in commands)
    assert not slice_events(tmp_path)


@pytest.mark.parametrize("group_index", [0, 1])
@pytest.mark.parametrize("changed", [False, True])
def test_group_failure_stops_all_children_before_any_candidate(
    tmp_path: Path, group_index: int, changed: bool,
) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    limit = 3
    options: dict[str, object] = {f"job-{group_index}-slice-{index}": {"sleep": 120, "child": True}
                                  for index in range(limit)}
    if group_index:
        options["job-0-slice-0"] = {"sleep": 120, "child": True}
    group_counts = (limit, limit - 1)[:group_index + 1]
    options[f"job-{group_index}-slice-0"] = {
        "wait_for": sum(group_counts), "child": True,
        "exit": PROGRAM_CHANGED_EXIT if changed else 7, "stderr": PROGRAM_CHANGED_MARKER if changed else "failed"}
    configure_slices(tmp_path, worker, options)
    with pytest.raises(IdentityChanged if changed else ComputeFailed, match="分片 0") as caught:
        list(worker(jobs, limit))
    assert isinstance(caught.value, IdentityChanged) is changed
    assert not events(tmp_path)
    records = slice_events(tmp_path)
    expected_schemes = [[str(SearchStore.scheme_path_for(job.result_path)) for job in group]
                        for group in ((jobs[0], jobs[2]), (jobs[1],))[:group_index + 1]]
    assert sorted(row["schemes"] for row in records) == sorted(
        group for group, count in zip(expected_schemes, group_counts, strict=True) for _ in range(count))
    groups_are_dead(records)


def test_queued_failure_is_checked_before_refilling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    limit = 3
    # 前兩片等第三片開出來才准成功：主迴圈被擋在第三片的監看裡，成功通知不會先被讀走（複查）。
    configure_slices(tmp_path, worker, {"job-0-slice-0": {"hold": "release-successes"},
                                        "job-0-slice-1": {"hold": "release-successes"},
                                        "job-0-slice-2": {"hold": "release-failure", "exit": 7}})
    commands = capture_commands(worker, monkeypatch)

    def wait_initial_wave(item: _Slice, completed: Queue[_Slice], order: Lock) -> None:
        real_watch(item, completed, order)
        if item.slice_index != limit - 1:
            return
        (tmp_path / "release-successes").touch()
        assert wait_until(lambda: completed.qsize() >= limit - 1), "前兩片的成功通知沒到"
        (tmp_path / "release-failure").touch()
        assert wait_until(lambda: completed.qsize() >= limit), "第三片的失敗通知沒到"

    monkeypatch.setattr("aosr.search.worker._watch", wait_initial_wave)
    with pytest.raises(ComputeFailed, match="分片 2 離開碼 7"):
        list(worker(jobs, limit))
    assert not events(tmp_path)
    first_schemes = tuple(str(SearchStore.scheme_path_for(job.result_path)) for job in (jobs[0], jobs[2]))
    assert all(command[len(worker.slice_runner):command.index("--slice")] == first_schemes for command in commands)
    groups_are_dead(slice_events(tmp_path))


def test_failure_queued_between_refills_stops_the_next_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    limit = 3
    # 第一片等放行才失敗；後兩片等第三片開出來才准成功，主迴圈才會一次空出兩個位置（複查）。
    configure_slices(tmp_path, worker, {"job-0-slice-0": {"hold": "release-failure", "exit": 7},
                                        "job-0-slice-1": {"hold": "release-successes"},
                                        "job-0-slice-2": {"hold": "release-successes"}})
    commands = capture_commands(worker, monkeypatch)
    observed = worker._spawn
    released: list[bool] = []
    queues: list[Queue[_Slice]] = []
    second = str(SearchStore.scheme_path_for(jobs[1].result_path.resolve()))

    def wait_initial_wave(item: _Slice, completed: Queue[_Slice], order: Lock) -> None:
        queues.append(completed)
        real_watch(item, completed, order)
        if item.slice_index != limit - 1 or released:
            return
        (tmp_path / "release-successes").touch()
        assert wait_until(lambda: completed.qsize() >= limit - 1), "後兩片的成功通知沒到"

    def release_while_spawning(command: Sequence[str], stderr_path: Path) -> subprocess.Popen[bytes]:
        process = observed(command, stderr_path)
        if "--slice" in command and second in command and not released:
            # 第二組第一片剛開出去，第一組那片才失敗；它的監看還沒建，佇列裡只可能是第一片的失敗。
            (tmp_path / "release-failure").touch()
            released.append(wait_until(lambda: not queues[0].empty()))
        return process

    monkeypatch.setattr("aosr.search.worker._watch", wait_initial_wave)
    monkeypatch.setattr(worker, "_spawn", release_while_spawning)
    with pytest.raises(ComputeFailed, match="分片 0 離開碼 7"):
        list(worker(jobs, limit))
    assert released == [True], "第一片的失敗通知沒到"
    assert [command[command.index("--slice") + 1] for command in commands
            if "--slice" in command and second in command] == ["0"]
    assert not events(tmp_path)
    groups_are_dead(slice_events(tmp_path))


def test_failure_stops_other_slices_before_reading_the_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    limit = 3
    configure_slices(tmp_path, worker, {"job-0-slice-0": {"wait_for": limit, "exit": 7, "child": True},
                                        "job-0-slice-1": {"sleep": 120, "child": True},
                                        "job-0-slice-2": {"sleep": 120, "child": True}})
    checked: list[int] = []

    def check_after_others_stopped(item: _Slice, label: str) -> None:
        # 讀失敗那片的錯誤尾巴時，其他片與它們派生的子行程都已經停了（審查：植錯存活補題）。
        for record in slice_events(tmp_path):
            if record["pid"] == item.process.pid:
                continue
            assert_dead(record["pid"])
            if record["child"] is not None:
                assert_dead(record["child"])
            checked.append(record["pid"])
        real_check_exit(item, label)

    monkeypatch.setattr("aosr.search.worker._check_exit", check_after_others_stopped)
    with pytest.raises(ComputeFailed, match="分片 0 離開碼 7"):
        list(worker(jobs, limit))
    records = slice_events(tmp_path)
    assert sorted(checked) == sorted(row["pid"] for row in records if row["slice"] != 0)
    assert not events(tmp_path)
    groups_are_dead(records)


def test_slice_watchers_finish_before_parts_are_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    threads: list[Thread] = []
    joined: list[Thread] = []

    class RecordingThread(Thread):
        # 監看執行緒被等到結束（join）就記下來；回傳分片之前每一條都要記到（審查：植錯存活補題）。
        def join(self, timeout: float | None = None) -> None:
            super().join(timeout)
            if not self.is_alive():
                joined.append(self)

    def watched(item: _Slice, completed: Queue[_Slice], order: Lock) -> None:
        real_watch(item, completed, order)
        assert isinstance(item.watcher, RecordingThread)
        threads.append(item.watcher)

    monkeypatch.setattr("aosr.search.worker.Thread", RecordingThread)
    monkeypatch.setattr("aosr.search.worker._watch", watched)
    try:
        parts = worker._parts_by_job(jobs, 2)
        assert parts and threads
        assert all(any(thread is done for done in joined) for thread in threads)
    finally:
        for thread in threads:
            thread.join(timeout=20)
    groups_are_dead(slice_events(tmp_path))


@pytest.mark.parametrize("interruption", ["keyboard", "close", "sigterm"])
def test_interruption_during_mixed_first_wave_stops_all_groups(tmp_path: Path, interruption: str) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    limit = 2
    configure_slices(tmp_path, worker, {
        "job-0-slice-0": {"sleep": 120, "child": True}, "job-1-slice-0": {"sleep": 120, "child": True}})
    stream = worker(jobs, limit)
    interrupt_signal = signal.SIGTERM if interruption == "sigterm" else signal.SIGUSR1
    previous = signal.getsignal(interrupt_signal)
    errors: list[str] = []
    main_thread = get_ident()

    def interrupt(signum: int, frame: object) -> None:
        if interruption == "close":
            raise GeneratorExit
        raise KeyboardInterrupt

    def send_when_started() -> None:
        deadline = time.monotonic() + 20
        while len(slice_events(tmp_path)) < limit + 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        if len(slice_events(tmp_path)) < limit + 1:
            errors.append("兩組分片沒有全部啟動")
        signal.pthread_kill(main_thread, interrupt_signal)

    signal.signal(interrupt_signal, interrupt)
    sender = Thread(target=send_when_started, daemon=True)
    sender.start()
    try:
        with pytest.raises(GeneratorExit if interruption == "close" else KeyboardInterrupt):
            next(stream)
        assert not errors
        assert not events(tmp_path)
        groups_are_dead(slice_events(tmp_path))
    finally:
        sender.join(timeout=20)
        signal.signal(interrupt_signal, previous)
        stream.close()


def test_legacy_slices_refuses_mixed_scenes_without_starting(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    mixed = mixed_rooms(jobs)
    with pytest.raises(ValueError, match="同一場景"):
        worker._slices(mixed, 2)
    assert not any(SearchStore.scheme_path_for(job.result_path.resolve()).exists() for job in mixed)
    assert not slice_events(tmp_path)
    assert not worker.fem_root.exists()


def test_stage_one_mixed_rooms_get_their_own_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, job = setup_stage_one(tmp_path, monkeypatch)
    jobs = mixed_rooms(tuple(CandidateJob(index, job.scheme.model_copy(update={"scheme_id": f"job-{index}"}),
                                         tmp_path / f"result-{index}.json") for index in range(3)))
    template = json.loads((tmp_path / "template").read_bytes())
    for item in jobs:
        document = {**template, "scheme": item.scheme.model_dump(mode="json"),
                    "evaluation": {**template["evaluation"], "candidate_id": item.scheme.scheme_id},
                    "score": {**template["score"], "candidate_id": item.scheme.scheme_id},
                    "origin": {**template["origin"], "trial_number": item.trial_number}}
        (tmp_path / f"template-{item.trial_number}").write_text(json.dumps(document))
    configure_slices(tmp_path, worker.processes, {})
    commands = capture_commands(worker.processes, monkeypatch)
    results = list(worker(jobs, 2))
    records = slice_events(tmp_path)
    a = expected_group(worker.processes, (jobs[0], jobs[2]), 2, records, commands)
    b = expected_group(worker.processes, (jobs[1],), 2, records, commands)
    assert [command for command in commands if "--slice" not in command] == [
        expected_candidate(worker.processes, item, parts) for item, parts in zip(jobs, (a, b, a), strict=True)]
    assert {result.job.trial_number for result in results} == {item.trial_number for item in jobs}
    groups_are_dead(records)
