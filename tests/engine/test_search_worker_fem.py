"""兩波搜尋的子行程契約；分片替身只切設定軸，不算正式物理。"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from threading import Thread, get_ident
from typing import NotRequired, TypedDict, cast
from uuid import UUID

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme_cli import PROGRAM_CHANGED_EXIT, PROGRAM_CHANGED_MARKER
from aosr.runtime import child_process_env
from aosr.search.run import ComputeFailed, IdentityChanged
from aosr.search.store import SearchStore
from tests.engine._search_run_cases import make_store
from tests.engine.test_search_worker import assert_dead, events, setup_worker


class SliceEvent(TypedDict):
    start: float
    end: NotRequired[float]
    pid: int
    child: int | None
    env: dict[str, str]
    cwd: str
    schemes: list[str]
    indices: list[int]
    slice: int
    slices: int
    out: str
    engine_commit: str


def slice_events(tmp_path: Path) -> list[SliceEvent]:
    return [cast(SliceEvent, json.loads(path.read_text())) for path in tmp_path.glob("slice-event-*")]


def groups_are_dead(records: Sequence[SliceEvent]) -> None:
    assert records
    for record in records:
        assert_dead(int(str(record["pid"])))
        if record["child"] is not None:
            assert_dead(int(str(record["child"])))


@pytest.mark.parametrize("axis", [None, LowFrequencyAxis.SEARCH, LowFrequencyAxis.VERIFICATION])
def test_two_waves_cover_axis_and_pass_all_parts(tmp_path: Path, axis: LowFrequencyAxis | None) -> None:
    worker, jobs = setup_worker(tmp_path, {"slice-0": {"sleep": 0.2, "child": True}}, (0, 1, 2))
    jobs = tuple(replace(job, scheme=job.scheme.model_copy(update={
        "scene": job.scheme.scene.model_copy(update={"low_frequency_axis": axis})})) for job in jobs)
    limit = 3
    assert not worker.fem_root.exists()
    results = list(worker(jobs, limit))
    records = slice_events(tmp_path)
    frequencies = low_frequency_axis_frequencies(axis or LowFrequencyAxis.SEARCH)[0]
    assert len(records) == min(limit, len(frequencies))
    assert {row["slice"] for row in records} == set(range(len(records)))
    indices = [index for row in records for index in row["indices"]]
    assert len(indices) == len(set(indices))
    assert set(indices) == set(range(len(frequencies)))
    expected_schemes = {str(SearchStore.scheme_path_for(job.result_path).resolve()) for job in jobs}
    for row in records:
        assert set(row["schemes"]) == expected_schemes
        assert row["slices"] == len(records)
        assert row["indices"]
        assert row["engine_commit"] == worker.engine_commit
        env = row["env"]
        assert isinstance(env, dict)
        assert all(env[key] == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        assert env["PYTHONPATH"] == child_process_env(threads=1)["PYTHONPATH"]
        assert row["cwd"] == str(Path(__file__).resolve().parents[2])
    parts = {row["out"] for row in records}
    batch_paths = {Path(str(path)).parent for path in parts}
    assert len(batch_paths) == len({worker.fem_root})
    for path in batch_paths:
        assert path.parent == worker.fem_root
        assert UUID(path.name).hex == path.name
    for candidate_row in events(tmp_path):
        received = candidate_row["parts"]
        assert isinstance(received, list)
        assert set(received) == parts
        assert float(str(candidate_row["start"])) >= max(float(str(slice_row["end"])) for slice_row in records)
    assert {result.job.trial_number for result in results} == {job.trial_number for job in jobs}
    for result in results:
        saved = SchemeResult.model_validate_json(result.job.result_path.read_bytes())
        assert result.candidate == saved.candidate
        assert result.identity.physics_identity == saved.physics_identity
        assert result.identity.program_fingerprint == saved.program_fingerprint
        assert result.identity.purpose_settings == saved.purpose_settings
        assert result.seconds == saved.timings.total_s
    groups_are_dead(records)


@pytest.mark.parametrize("oversubscribe", [False, True])
def test_slice_count_is_bounded_by_axis(tmp_path: Path, oversubscribe: bool) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (0,))
    frequencies = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[0]
    limit = len(frequencies) + 1 if oversubscribe else 1
    list(worker(jobs, limit))
    records = slice_events(tmp_path)
    assert len(records) == min(limit, len(frequencies))
    indices = [index for row in records for index in row["indices"]]
    assert sorted(indices) == list(range(len(frequencies)))
    assert all(row["indices"] for row in records)


@pytest.mark.parametrize("code,marker,changed", [
    (7, "error tail", False),
    (PROGRAM_CHANGED_EXIT, PROGRAM_CHANGED_MARKER, True),
    (7, PROGRAM_CHANGED_MARKER, False),
    (PROGRAM_CHANGED_EXIT, "no marker", False),
    (PROGRAM_CHANGED_EXIT, "prefix " + PROGRAM_CHANGED_MARKER, False),
])
@pytest.mark.parametrize("failed_slice", [0, 1])
def test_slice_failure_stops_groups_before_candidates(
    tmp_path: Path, code: int, marker: str, changed: bool, failed_slice: int,
) -> None:
    lines = [f"diagnostic {index}" for index in range(12)]
    stderr = "\n".join([marker, *lines, marker])
    limit = 3
    options: dict[str, object] = {f"slice-{index}": {"sleep": 120, "child": True} for index in range(limit)}
    options[f"slice-{failed_slice}"] = {"sleep": 0.5, "exit": code, "stderr": stderr, "child": True}
    worker, jobs = setup_worker(tmp_path, options, (0, 1))
    started = time.monotonic()
    with pytest.raises(IdentityChanged if changed else ComputeFailed) as caught:
        list(worker(jobs, limit))
    assert isinstance(caught.value, IdentityChanged) is changed
    assert time.monotonic() - started < 20
    message = str(caught.value)
    assert f"分片 {failed_slice}" in message and str(code) in message
    assert lines[-1] in message and lines[0] not in message
    assert not events(tmp_path)
    records = slice_events(tmp_path)
    assert {row["slice"] for row in records} == set(range(limit))
    groups_are_dead(records)


@pytest.mark.parametrize("closing", [False, True])
def test_interruption_during_first_wave_stops_all_groups(tmp_path: Path, closing: bool) -> None:
    """第一波尚未 yield：用關閉協定的 GeneratorExit 或呼叫端例外中斷等待。"""
    worker, jobs = setup_worker(tmp_path, {
        "slice-0": {"sleep": 120, "child": True},
        "slice-1": {"sleep": 120, "child": True},
    }, (0, 1))
    stream = worker(jobs, 2)
    previous = signal.getsignal(signal.SIGUSR1)
    errors: list[str] = []

    def interrupt(signum: int, frame: object) -> None:
        if closing:
            raise GeneratorExit
        raise RuntimeError("caller failed")

    def send_when_started() -> None:
        deadline = time.monotonic() + 20
        while len(slice_events(tmp_path)) < len(jobs) and time.monotonic() < deadline:
            time.sleep(0.01)
        if len(slice_events(tmp_path)) < len(jobs):
            errors.append("第一波沒有全部啟動")
        signal.pthread_kill(main_thread, signal.SIGUSR1)

    main_thread = get_ident()
    signal.signal(signal.SIGUSR1, interrupt)
    sender = Thread(target=send_when_started, daemon=True)
    sender.start()
    try:
        with pytest.raises(GeneratorExit if closing else RuntimeError):
            next(stream)
        assert not errors
        assert not events(tmp_path)
        groups_are_dead(slice_events(tmp_path))
    finally:
        sender.join(timeout=20)
        signal.signal(signal.SIGUSR1, previous)
        stream.close()


@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGHUP])
def test_cli_termination_during_first_wave_stops_groups(tmp_path: Path, termination: signal.Signals) -> None:
    from tests.engine.test_search_cli_process import cleanup, launch

    process = launch(tmp_path, {"slice-0": {"sleep": 120, "child": True}})
    try:
        deadline = time.monotonic() + 20
        while not slice_events(tmp_path) and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        records = slice_events(tmp_path)
        assert records
        process.send_signal(termination)
        process.communicate(timeout=20)
        assert process.returncode != 0
        assert not events(tmp_path)
        groups_are_dead(slice_events(tmp_path))
    finally:
        cleanup(process, tmp_path)
        for record in slice_events(tmp_path):
            for pid in (record["pid"], record["child"]):
                if pid is not None:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


def test_slice_launch_exception_stops_started_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker, jobs = setup_worker(tmp_path, {"slice-0": {"sleep": 120, "child": True}}, (0, 1))
    spawn = worker._spawn

    def fail_second(command: Sequence[str], stderr_path: Path) -> subprocess.Popen[bytes]:
        if command[command.index("--slice") + 1] != "0":
            raise OSError("cannot launch")
        process = spawn(command, stderr_path)
        deadline = time.monotonic() + 20
        while not slice_events(tmp_path) and time.monotonic() < deadline:
            time.sleep(0.01)
        return process

    monkeypatch.setattr(worker, "_spawn", fail_second)
    with pytest.raises(OSError, match="cannot launch"):
        list(worker(jobs, 2))
    assert not events(tmp_path)
    groups_are_dead(slice_events(tmp_path))


def test_baseline_and_each_call_get_fresh_slice_directory(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {}, (None,))
    first, = worker(jobs, 1)
    first_records = slice_events(tmp_path)
    second, = worker(jobs, 1)
    records = slice_events(tmp_path)
    assert first.job == second.job == jobs[0]
    assert len(records) == len(first_records) * 2
    assert len({Path(str(row["out"])).parent for row in records}) == len(records)
    expected_scheme = str(SearchStore.scheme_path_for(jobs[0].result_path).resolve())
    assert all(row["schemes"] == [expected_scheme] for row in records)


def test_old_store_opens_without_fem_directory_and_computes(tmp_path: Path) -> None:
    from aosr.search.cli import _compute
    from aosr.search.worker import SubprocessCompute

    store, _ = make_store(tmp_path)
    assert not store.fem_path.exists()
    opened = SearchStore.open(store.path)
    default_worker = _compute(opened, tmp_path, "test")
    assert isinstance(default_worker, SubprocessCompute)
    assert default_worker.fem_root == opened.fem_path
    worker, jobs = setup_worker(tmp_path, {}, (None,))
    worker.fem_root = default_worker.fem_root
    job = replace(jobs[0], result_path=opened.baseline_path)
    result, = worker((job,), 1)
    assert result.job == job and opened.fem_path.is_dir()
    assert SearchStore.open(store.path).search_id == opened.search_id


def test_empty_batch_starts_no_process_and_creates_no_fem_directory(tmp_path: Path) -> None:
    worker, jobs = setup_worker(tmp_path, {}, ())
    assert not list(worker(jobs, 2))
    assert not worker.fem_root.exists()
    assert not events(tmp_path) and not slice_events(tmp_path)


def test_search_has_no_direct_physics_imports(tmp_path: Path) -> None:
    import aosr.search

    root = Path(aosr.search.__file__).parent
    forbidden = ("aosr.physics", "aosr.reporting.physics_stage", "aosr.reporting.fem_slices")
    violations: list[str] = []
    modules = list(root.rglob("*" + ".py"))
    assert modules and tmp_path.is_dir()
    for path in modules:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = ["aosr", "search", *path.relative_to(root).parts[:-1]]
                    module = ".".join(package[:len(package) - node.level + 1] + ([module] if module else []))
                names = [module, *(module + "." + alias.name for alias in node.names)]
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
                    violations.append(f"{path.name}:{node.lineno}:{name}")
    assert not violations
