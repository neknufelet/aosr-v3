"""第一階段搜尋成本的觀測與暫停紀錄；不求解、不替未完成候選記分。"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aosr.search.run import CandidateJob
from aosr.search.store import SearchStore
from aosr.search.worker import _Active, _Slice

FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
TIME_LIMIT_FILE = "stage_one_time_limit.json"
CandidateState = Literal["交回", "超限", "一起停下", "沒開始", "交回但超限"]
LimitPhase = Literal["slices", "candidates"]


class CandidateTiming(BaseModel):
    """每題保留兩段時間；未開始的時間格為空。"""

    model_config = FROZEN
    trial_number: int
    state: CandidateState
    slice_s: float | None = Field(default=None, ge=0)
    candidate_s: float | None = Field(default=None, ge=0)
    service_s: float | None = Field(default=None, ge=0)


class SliceGroupTiming(BaseModel):
    """分片服務時間取最長一片；窗口與排隊另列，沒有分攤。"""

    model_config = FROZEN
    trial_numbers: tuple[int, ...]
    slices: int = Field(ge=1)
    completed_slices: int = Field(ge=0)
    longest_slice_s: float | None = Field(default=None, ge=0)
    window_s: float | None = Field(default=None, ge=0)
    queue_s: float | None = Field(default=None, ge=0)


class StageOneTimeLimit(BaseModel):
    """整批暫停快照；計算器不知批號，由執行迴圈補上。"""

    model_config = FROZEN
    schema_version: Literal["stage-one-time-limit-v1"] = "stage-one-time-limit-v1"
    cap_s: float = Field(gt=0)
    phase: LimitPhase
    batch: int | None = Field(default=None, ge=0)
    candidates: tuple[CandidateTiming, ...]
    groups: tuple[SliceGroupTiming, ...]


def save_time_limit(store: SearchStore, record: StageOneTimeLimit) -> None:
    """沿用細算紀錄的同目錄暫存、同步、原子替換與只清自身暫存的寫法。"""
    path = store.path / TIME_LIMIT_FILE
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class _Group:
    numbers: tuple[int, ...]
    count: int
    items: list[_Slice] = field(default_factory=list)
    timing: SliceGroupTiming | None = None
    running: bool = False

    def snapshot(self, now: float) -> SliceGroupTiming:
        if self.timing is not None:
            return self.timing
        # 看守執行緒隨時會寫 ended：只讀一次，耗時與完成片數才出自同一刻，不會把沒算完時的耗時當成最終值存下。
        ended = [item.ended for item in self.items]
        starts = [item.started for item in self.items if item.started is not None]
        ends = [end if end is not None else now for end in ended]
        self.running = any(end is None for end in ended)
        durations = [end - start for end, start in zip(ends, starts, strict=True)]
        longest = max(durations) if durations else None
        window = max(ends) - min(starts) if starts else None
        result = SliceGroupTiming(trial_numbers=self.numbers, slices=self.count,
            completed_slices=sum(end is not None for end in ended),
            longest_slice_s=longest, window_s=window,
            queue_s=window - longest if window is not None and longest is not None else None)
        if result.completed_slices == self.count:
            self.timing = result
        return result


class TimeLimitTracker:
    """只觀測啟用上限的批次，完成時間採入列戳記，呼叫端等待不增加服務秒數。"""

    def __init__(self, jobs: Sequence[CandidateJob], cap_s: float) -> None:
        self.jobs, self.cap_s = jobs, cap_s
        self.groups: dict[str, _Group] = {}
        self.starts: dict[int, float] = {}
        self.seconds: dict[int, float] = {}
        self.returned: set[int] = set()

    def observe(self, jobs: Sequence[CandidateJob], count: int, item: _Slice | None) -> None:
        key = jobs[0].scheme.scene.model_dump_json()
        if item is None:
            numbers = tuple(job.trial_number for job in jobs if job.trial_number is not None)
            self.groups[key] = _Group(numbers, count)
        else:
            self.groups[key].items.append(item)

    def _candidates(self, phase: LimitPhase, groups: Mapping[str, SliceGroupTiming]) -> tuple[CandidateTiming, ...]:
        rows = []
        for job in self.jobs:
            number = job.trial_number
            assert number is not None
            key = job.scheme.scene.model_dump_json()
            slices = groups[key].longest_slice_s
            own = self.seconds.get(number)
            service = slices + (own or 0.) if slices is not None else None
            over = service is not None and service > self.cap_s
            state: CandidateState = "交回但超限" if over else "交回"
            if number not in self.returned:
                slicing = phase == "slices" and self.groups[key].running
                started = number in self.starts or slicing
                state = "超限" if over else "一起停下" if started else "沒開始"
            rows.append(CandidateTiming(trial_number=number, state=state,
                slice_s=slices, candidate_s=own, service_s=service))
        return tuple(rows)

    def snapshot(self, phase: LimitPhase, now: float) -> StageOneTimeLimit:
        groups = {key: group.snapshot(now) for key, group in self.groups.items()}
        return StageOneTimeLimit(cap_s=self.cap_s, phase=phase,
            candidates=self._candidates(phase, groups), groups=tuple(groups.values()))

    def slices_limit(self, now: float) -> StageOneTimeLimit | None:
        record = self.snapshot("slices", now)
        return record if any(row.state == "超限" for row in record.candidates) else None

    def candidates_limit(self, active: Sequence[_Active], starts: Mapping[int, float],
                         ended: Mapping[int, float], now: float, *, finished: _Active | None = None,
                         include_queued: bool = False) -> StageOneTimeLimit | None:
        for item in active:
            number = item.job.trial_number
            assert number is not None
            self.starts[number] = starts[id(item)]
            self.seconds[number] = ended.get(id(item), now) - starts[id(item)]
        record = self.snapshot("candidates", now)
        # 已入列的完成通知要等讀到那一筆才交回；開新工作之前仍先攔住排隊的超限通知。
        observed = {item.job.trial_number for item in active
            if include_queued or id(item) not in ended or item is finished}
        return record if any(row.trial_number in observed and row.state in ("超限", "交回但超限")
            for row in record.candidates) else None

    def accept(self, item: _Active, seconds: float) -> None:
        number = item.job.trial_number
        assert number is not None
        self.returned.add(number)
        self.seconds[number] = seconds


def time_limit_message(record: StageOneTimeLimit) -> str:
    exceeded = "、".join(f"試算 {row.trial_number} {row.service_s} 秒" for row in record.candidates
        if row.state in ("超限", "交回但超限"))
    return f"單題搜尋成本超過上限：{exceeded}；上限 {record.cap_s} 秒；搜尋暫停，未完成；紀錄 {TIME_LIMIT_FILE}"


def time_limit_lines(store: SearchStore) -> tuple[str, ...]:
    """紀錄缺席不印；紀錄壞掉只影響這一段（同細算摘要），不拖垮整份報告。"""
    path = store.path / TIME_LIMIT_FILE
    if not path.exists():
        return ()
    try:
        record = StageOneTimeLimit.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        return (f"單題搜尋成本上限紀錄讀不回：{error}",)
    lines = [f"單題搜尋成本上限：{record.cap_s} 秒；搜尋暫停，未完成"]
    for row in record.candidates:
        lines.append(f"試算 {row.trial_number}：{row.state}；服務 {row.service_s} 秒"
            f"（分片 {row.slice_s} 秒／自己 {row.candidate_s} 秒）")
    for group in record.groups:
        lines.append(f"分片組 {','.join(str(number) for number in group.trial_numbers)}：排隊 {group.queue_s} 秒")
    return tuple(lines)
