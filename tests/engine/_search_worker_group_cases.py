"""混房派工的輸入與真子行程替身；答案只從輸入和頻率軸推得。"""
from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.search.run import CandidateJob
from aosr.search.store import SearchStore
from aosr.search.worker import SubprocessCompute
from tests.engine._search_worker_cases import SLICE_SCRIPT
from tests.engine.test_search_worker_fem import SliceEvent


def mixed_rooms(jobs: Sequence[CandidateJob]) -> tuple[CandidateJob, ...]:
    """輸入順序 A、B、A；只改中間候選的房長。"""
    middle = jobs[1]
    room = middle.scheme.scene.room_m
    scene = middle.scheme.scene.model_copy(update={"room_m": replace(room, Lx=room.Lx + 1)})
    return (jobs[0], replace(middle, scheme=middle.scheme.model_copy(update={"scene": scene})), *jobs[2:])


def with_axis(job: CandidateJob, axis: LowFrequencyAxis | None) -> CandidateJob:
    return replace(job, scheme=job.scheme.model_copy(update={
        "scene": job.scheme.scene.model_copy(update={"low_frequency_axis": axis})}))


def configure_slices(root: Path, worker: SubprocessCompute, options: dict[str, object]) -> None:
    """沿用分片替身，額外讓失敗條件指定組別、以事件屏障確定同伴已啟動。"""
    # 替換的兩段原字串一定要在，不然替身悄悄沒改、考卷照綠（主對話補）。
    for anchor in ('options = json.loads((root / "options").read_text()).get("slice-" + number, {})',
                   'time.sleep(options.get("sleep", 0))'):
        assert anchor in SLICE_SCRIPT, anchor
    script = SLICE_SCRIPT.replace(
        'options = json.loads((root / "options").read_text()).get("slice-" + number, {})',
        '''settings = json.loads((root / "options").read_text())
group = json.loads(Path(args.schemes[0]).read_text())["scheme_id"]
options = settings.get(group + "-slice-" + number, settings.get("slice-" + number, {}))''')
    script = script.replace('time.sleep(options.get("sleep", 0))', '''
deadline = time.monotonic() + 20
while (len(list(root.glob("slice-event-*"))) < options.get("wait_for", 0)
       or (options.get("hold") and not (root / options["hold"]).exists())):
    if time.monotonic() >= deadline:
        sys.exit(19)
    time.sleep(0.01)
time.sleep(options.get("sleep", 0))''')
    path = root / "slice-runner"
    path.write_text(script)
    (root / "options").write_text(json.dumps(options))
    worker.slice_runner = (worker.slice_runner[0], str(path))


def capture_commands(worker: SubprocessCompute, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    spawn = worker._spawn

    def observed(command: Sequence[str], stderr_path: Path) -> subprocess.Popen[bytes]:
        commands.append(tuple(command))
        return spawn(command, stderr_path)

    monkeypatch.setattr(worker, "_spawn", observed)
    return commands


def wait_until(condition: Callable[[], bool], seconds: float = 20) -> bool:
    """等到條件成立或逾時；回傳最後一次的結果，呼叫端自己斷言（逾時不准無聲通過）。"""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def count_unreaped_slices(worker: SubprocessCompute, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """每開一片分片之前，記下之前開過、父行程還沒收掉的分片有幾支。"""
    unreaped: list[int] = []
    spawned: list[subprocess.Popen[bytes]] = []
    spawn = worker._spawn

    def observed(command: Sequence[str], stderr_path: Path) -> subprocess.Popen[bytes]:
        is_slice = "--slice" in command
        if is_slice:
            unreaped.append(sum(process.returncode is None for process in spawned))
        process = spawn(command, stderr_path)
        if is_slice:
            spawned.append(process)
        return process

    monkeypatch.setattr(worker, "_spawn", observed)
    return unreaped


def expected_group(
    worker: SubprocessCompute, jobs: Sequence[CandidateJob], limit: int,
    records: Sequence[SliceEvent], commands: Sequence[tuple[str, ...]],
) -> tuple[str, ...]:
    """分片公式的獨立答案；檢查每一片完整命令及其負責的頻點索引。"""
    axis = jobs[0].scheme.scene.low_frequency_axis or LowFrequencyAxis.SEARCH
    frequencies = low_frequency_axis_frequencies(axis)[0]
    count = min(limit, len(frequencies))
    schemes = tuple(str(SearchStore.scheme_path_for(job.result_path.resolve())) for job in jobs)
    rows = sorted((row for row in records if tuple(row["schemes"]) == schemes), key=lambda row: row["slice"])
    assert [row["slice"] for row in rows] == list(range(count))
    batch, = {Path(row["out"]).parent for row in rows}
    assert batch.parent == worker.fem_root.resolve()
    assert UUID(batch.name).hex == batch.name
    parts = tuple(str(batch / f"slice-{index}.json") for index in range(count))
    assert sorted(path.name for path in batch.glob("*.json")) == sorted(Path(part).name for part in parts)
    expected = [(*worker.slice_runner, *schemes, "--slice", str(index), "--slices", str(count),
                 "--out", part, "--capabilities", str(worker.capabilities_path.resolve()),
                 "--engine-commit", worker.engine_commit) for index, part in enumerate(parts)]
    assert [command for command in commands if "--slice" in command
            and command[command.index("--out") + 1] in parts] == expected
    size, extra = divmod(len(frequencies), count)
    for index, row in enumerate(rows):
        start = index * size + min(index, extra)
        assert row["indices"] == list(range(start, start + size + (index < extra)))
        assert row["slices"] == count
        assert row["out"] == parts[index]
    return parts


def expected_candidate(worker: SubprocessCompute, job: CandidateJob, parts: Sequence[str]) -> tuple[str, ...]:
    return (*worker.runner, str(SearchStore.scheme_path_for(job.result_path.resolve())), "--out",
            str(job.result_path.resolve()), "--engine-commit", worker.engine_commit, "--search-id", worker.search_id,
            "--capabilities", str(worker.capabilities_path.resolve()), "--trial-number", str(job.trial_number),
            "--fem-parts", *parts)
