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
from aosr.search.worker import _Slice
from aosr.search.worker import _watch as real_watch
from tests.engine._search_worker_group_cases import (
    capture_commands,
    configure_slices,
    count_unreaped_slices,
    expected_candidate,
    expected_group,
    mixed_rooms,
    with_axis,
)
from tests.engine.test_search_stage_one_worker import setup as setup_stage_one
from tests.engine.test_search_worker import events, setup_worker
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
    configure_slices(tmp_path, worker, {f"slice-{index}": {"sleep": 0.15} for index in range(limit)})
    commands = capture_commands(worker, monkeypatch)
    unreaped = count_unreaped_slices(worker, monkeypatch)
    list(worker(jobs, limit))
    # 父行程開每一片之前，還沒收掉（returncode 仍是 None）的分片都算活著，這是上限的保守量法（審查）。
    assert unreaped and max(unreaped) == limit - 1
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
    expected_slices = range(min(limit, len(low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[0])))
    configure_slices(tmp_path, worker, {"job-0-slice-2": {"hold": "release-failure", "exit": 7}})
    commands = capture_commands(worker, monkeypatch)

    def wait_initial_wave(item: _Slice, completed: Queue[_Slice], order: Lock) -> None:
        real_watch(item, completed, order)
        if item.slice_index != limit - 1:
            return
        deadline = time.monotonic() + 20
        while completed.qsize() < limit - 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert completed.qsize() == len(expected_slices) - 1
        (tmp_path / "release-failure").touch()
        while completed.qsize() < limit and time.monotonic() < deadline:
            time.sleep(0.01)
        assert completed.qsize() == len(expected_slices)

    monkeypatch.setattr("aosr.search.worker._watch", wait_initial_wave)
    with pytest.raises(ComputeFailed, match="分片 2"):
        list(worker(jobs, limit))
    assert not events(tmp_path)
    first_schemes = tuple(str(SearchStore.scheme_path_for(job.result_path)) for job in (jobs[0], jobs[2]))
    assert all(command[len(worker.slice_runner):command.index("--slice")] == first_schemes for command in commands)
    groups_are_dead(slice_events(tmp_path))


def test_failure_queued_between_refills_stops_the_next_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0, 1, 2))
    jobs = mixed_rooms(jobs)
    limit = 3
    configure_slices(tmp_path, worker, {"job-0-slice-0": {"hold": "release-failure", "exit": 7}})
    commands = capture_commands(worker, monkeypatch)
    observed = worker._spawn
    released: list[bool] = []
    queues: list[Queue[_Slice]] = []
    second = str(SearchStore.scheme_path_for(jobs[1].result_path.resolve()))

    def wait_initial_wave(item: _Slice, completed: Queue[_Slice], order: Lock) -> None:
        # 第一組三片都開了以後，等兩片成功的通知都排進來，主迴圈一次讀完、一次補兩個位置。
        queues.append(completed)
        real_watch(item, completed, order)
        if item.slice_index != limit - 1 or released:
            return
        deadline = time.monotonic() + 20
        while completed.qsize() < limit - 1 and time.monotonic() < deadline:
            time.sleep(0.01)

    def release_while_spawning(command: Sequence[str], stderr_path: Path) -> subprocess.Popen[bytes]:
        process = observed(command, stderr_path)
        if "--slice" in command and second in command and not released:
            # 第二組第一片剛開出去，第一組那片才失敗：失敗通知排進佇列之後才回到主迴圈。
            released.append(True)
            (tmp_path / "release-failure").touch()
            deadline = time.monotonic() + 20
            while queues[0].empty() and time.monotonic() < deadline:
                time.sleep(0.01)
        return process

    monkeypatch.setattr("aosr.search.worker._watch", wait_initial_wave)
    monkeypatch.setattr(worker, "_spawn", release_while_spawning)
    with pytest.raises(ComputeFailed, match="分片 0"):
        list(worker(jobs, limit))
    assert released
    assert [command[command.index("--slice") + 1] for command in commands
            if "--slice" in command and second in command] == ["0"]
    assert not events(tmp_path)
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
