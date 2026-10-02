"""真子行程驗行程上限、完成順序、環境與整群清理；不求解物理。"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aosr.reporting.result import SchemeResult
from aosr.runtime import child_process_env
from aosr.search.run import CandidateJob
from tests.engine._search_worker_cases import SCRIPT, prepared_result

if TYPE_CHECKING:
    from aosr.search.worker import SubprocessCompute


def setup_worker(tmp_path: Path, options: dict[str, object], numbers: tuple[int | None, ...]
                 ) -> tuple[SubprocessCompute, tuple[CandidateJob, ...]]:
    from aosr.search.worker import SubprocessCompute

    result = prepared_result(tmp_path)
    script = tmp_path / "runner"
    script.write_text(SCRIPT, encoding="utf-8")
    (tmp_path / "options").write_text(json.dumps(options), encoding="utf-8")
    jobs = tuple(CandidateJob(number, result.scheme.model_copy(update={"scheme_id": f"job-{number}"}),
                             tmp_path / f"result-{number}") for number in numbers)
    worker = SubprocessCompute(capabilities_path=tmp_path, engine_commit="test", search_id="search",
                               runner=(sys.executable, str(script)), poll_s=0.005)
    return worker, jobs


def events(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(path.read_text()) for path in tmp_path.glob("event-*")]


def process_state(pid: int) -> str | None:
    """行程不在回 None；讀 /proc 的途中剛好被收掉（OSError）也算不在。"""
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


def assert_dead(pid: int) -> None:
    # 被殺的孫行程由系統收養，什麼時候被收掉不歸考卷管；殭屍已不能計算，也不能再留活行程。
    deadline = time.monotonic() + 3
    state = process_state(pid)
    while state not in (None, "Z") and time.monotonic() < deadline:
        time.sleep(0.01)
        state = process_state(pid)
    assert state in (None, "Z"), f"行程 {pid} 還活著（{state}）"
    if state is None:
        # /proc 讀不到不等於行程不在（例如沒有 /proc）：再向系統確認一次；殭屍不會走到這裡。
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_bounded_workers_yield_first_finished_and_environment(tmp_path: Path) -> None:
    numbers = (0, 1, 2, 3)
    limit = 2
    worker, jobs = setup_worker(tmp_path, {"0": {"sleep": 0.7}, "1": {"sleep": 0.05}}, numbers)
    stream = worker(jobs, limit)
    first = next(stream)
    assert first.job == jobs[1]
    running = events(tmp_path)
    assert any("end" not in row for row in running)
    results = [first, *stream]
    assert {result.job.trial_number for result in results} == {job.trial_number for job in jobs}
    records = events(tmp_path)
    changes = sorted([(float(str(row["start"])), 1) for row in records]
                     + [(float(str(row["end"])), -1) for row in records])
    active = 0
    for _, delta in changes:
        active += delta
        assert active <= limit
    for row in records:
        env = row["env"]
        assert isinstance(env, dict)
        assert all(env[name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        assert env["PYTHONPATH"] == child_process_env(threads=1)["PYTHONPATH"]
        assert row["cwd"] == str(Path(__file__).resolve().parents[2])
    for result in results:
        saved = SchemeResult.model_validate_json(result.job.result_path.read_bytes())
        assert result.candidate == saved.candidate and result.seconds == saved.timings.total_s
        assert result.identity.physics_identity == saved.physics_identity
        assert result.identity.program_fingerprint == saved.program_fingerprint
        assert result.identity.purpose_settings == saved.purpose_settings


@pytest.mark.parametrize("changed", [False, True])
def test_failure_stops_other_process_groups(tmp_path: Path, changed: bool) -> None:
    from aosr.reporting.scheme_cli import PROGRAM_CHANGED_EXIT, PROGRAM_CHANGED_MARKER
    from aosr.search.run import ComputeFailed, IdentityChanged

    code = PROGRAM_CHANGED_EXIT if changed else 7
    stderr = PROGRAM_CHANGED_MARKER if changed else "first line\nerror tail"
    worker, jobs = setup_worker(tmp_path, {"0": {"sleep": 0.3, "exit": code, "stderr": stderr},
                                          "1": {"sleep": 120, "child": True}}, (0, 1))
    started = time.monotonic()
    with pytest.raises(IdentityChanged if changed else ComputeFailed) as caught:
        list(worker(jobs, 2))
    assert time.monotonic() - started < 20
    assert "0" in str(caught.value) and str(code) in str(caught.value) and stderr in str(caught.value)
    records = events(tmp_path)
    assert records
    for record in records:
        assert_dead(int(str(record["pid"])))
        if record["child"] is not None:
            assert_dead(int(str(record["child"])))


def test_change_exit_without_marker_is_general_failure(tmp_path: Path) -> None:
    from aosr.reporting.scheme_cli import PROGRAM_CHANGED_EXIT
    from aosr.search.run import ComputeFailed, IdentityChanged

    worker, jobs = setup_worker(tmp_path, {"0": {"exit": PROGRAM_CHANGED_EXIT}}, (0,))
    with pytest.raises(ComputeFailed) as caught:
        list(worker(jobs, 1))
    assert not isinstance(caught.value, IdentityChanged)


def test_closing_generator_stops_children(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {"0": {"sleep": 0.3}, "1": {"sleep": 120, "child": True}}, (0, 1))
    stream = worker(jobs, 2)
    assert next(stream).job == jobs[0]
    started = time.monotonic()
    stream.close()
    assert time.monotonic() - started < 20
    records = events(tmp_path)
    assert records
    for record in records:
        assert_dead(int(str(record["pid"])))
        if record["child"] is not None:
            assert_dead(int(str(record["child"])))


@pytest.mark.parametrize("mismatch", ["wrong_origin", "wrong_id", "wrong_trial", "wrong_kind"])
def test_result_must_belong_to_job(tmp_path: Path, mismatch: str) -> None:
    from aosr.search.run import ComputeFailed

    worker, jobs = setup_worker(tmp_path, {"0": {mismatch: True}}, (0,))
    with pytest.raises(ComputeFailed):
        list(worker(jobs, 1))


def test_baseline_result_has_search_origin(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (None,))
    computed, = worker(jobs, 1)
    saved = SchemeResult.model_validate_json(computed.job.result_path.read_bytes())
    assert saved.origin.kind == "search_baseline"
    assert saved.origin.search_id == "search" and saved.origin.trial_number is None


def test_exception_thrown_into_generator_stops_children(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {"0": {"sleep": 0.3}, "1": {"sleep": 120, "child": True}}, (0, 1))
    stream = worker(jobs, 2)
    next(stream)
    with pytest.raises(RuntimeError, match="caller failed"):
        stream.throw(RuntimeError("caller failed"))
    records = events(tmp_path)
    assert records
    for record in records:
        assert_dead(int(str(record["pid"])))
        if record["child"] is not None:
            assert_dead(int(str(record["child"])))


def test_completion_order_survives_consumer_pause(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {"0": {"sleep": 0.8}, "1": {"sleep": 0.05},
                                          "2": {"sleep": 0.25}}, (0, 1, 2))
    stream = worker(jobs, 3)
    try:
        assert next(stream).job == jobs[1]
        deadline = time.monotonic() + 3
        while any("end" not in row for row in events(tmp_path)) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert all("end" in row for row in events(tmp_path))
        records = events(tmp_path)
        assert records
        for record in records:
            assert_dead(int(str(record["pid"])))
        assert [result.job.trial_number for result in stream] == [2, 0]
    finally:
        stream.close()


def test_failure_message_keeps_stderr_tail(tmp_path: Path) -> None:
    from aosr.search.run import ComputeFailed

    lines = [f"error-line-{number}" for number in range(12)]
    worker, jobs = setup_worker(tmp_path, {"0": {"exit": 7, "stderr": "\n".join(lines)}}, (0,))
    with pytest.raises(ComputeFailed) as caught:
        list(worker(jobs, 1))
    assert lines[-1] in str(caught.value)
    assert lines[0] not in str(caught.value)


def test_change_marker_without_change_exit_is_general_failure(tmp_path: Path) -> None:
    from aosr.reporting.scheme_cli import PROGRAM_CHANGED_MARKER
    from aosr.search.run import ComputeFailed, IdentityChanged

    worker, jobs = setup_worker(tmp_path, {"0": {"exit": 7, "stderr": PROGRAM_CHANGED_MARKER}}, (0,))
    with pytest.raises(ComputeFailed) as caught:
        list(worker(jobs, 1))
    assert not isinstance(caught.value, IdentityChanged)


def test_success_stops_descendants_before_returning_result(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {"0": {"child": True}}, (0,))
    stream = worker(jobs, 1)
    try:
        assert next(stream).job == jobs[0]
        records = events(tmp_path)
        assert records
        for record in records:
            assert record["child"] is not None
            assert_dead(int(str(record["child"])))
    finally:
        stream.close()


def test_stop_all_continues_after_group_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.worker import _stop_all

    worker, jobs = setup_worker(tmp_path, {"1": {"sleep": 120, "child": True}}, (0, 1))
    finished = worker._start(jobs[0])
    finished.process.wait(timeout=20)
    sleeping = worker._start(jobs[1])
    original = os.killpg

    def killpg(pid: int, termination: int) -> None:
        if pid == finished.process.pid:
            raise PermissionError("group belongs to someone else")
        original(pid, termination)

    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "event-1").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "event-1").exists()
        monkeypatch.setattr(os, "killpg", killpg)
        started = time.monotonic()
        _stop_all((finished, sleeping))
        assert time.monotonic() - started < 20
        records = events(tmp_path)
        assert records
        for record in records:
            assert_dead(int(str(record["pid"])))
            if record["child"] is not None:
                assert_dead(int(str(record["child"])))
    finally:
        for item in (finished, sleeping):
            try:
                original(item.process.pid, 9)
            except ProcessLookupError:
                pass
            item.process.wait(timeout=20)


def test_relative_paths_reach_child_as_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """子行程在 repo 根跑：相對的能力表與結果路徑要先轉成絕對路徑（第 4b 步修補審查）。"""
    from dataclasses import replace

    from aosr.search.worker import SubprocessCompute

    worker, jobs = setup_worker(tmp_path, {}, (0,))
    monkeypatch.chdir(tmp_path)
    relative = SubprocessCompute(capabilities_path=Path(), engine_commit="test", search_id="search",
                                 runner=worker.runner, poll_s=0.005)
    job = replace(jobs[0], result_path=Path(jobs[0].result_path.name))
    results = list(relative((job,), 1))
    assert [result.job for result in results] == [job]
    assert (tmp_path / job.result_path).is_file()
