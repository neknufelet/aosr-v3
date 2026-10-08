"""搜尋這一側的回饋：確定點序、逐行事件與手動重新開輪；不執行計算。"""

from __future__ import annotations

import fcntl
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, Self

from pydantic import BaseModel, Field, StrictStr, model_validator

from aosr.search import layout, ledger
from aosr.search.ledger_io import write_line as _write_line
from aosr.search.run import RoundRecord, SearchStatus, _write_status
from aosr.search.store import FROZEN, SearchStore
from aosr.search.sampler import ReplayMismatch


class FeedbackUnavailable(ValueError):
    """裁邊去重後無點，或上一輪回饋點尚未問完；外圈可如實停下。"""


class FeedbackEvent(BaseModel):
    """一列只記接著往哪裡找，不記細算分數。"""

    model_config = FROZEN
    round: int = Field(ge=2, strict=True)
    before_batch: int = Field(ge=0, strict=True)
    anchor_trial: int = Field(ge=0, strict=True)
    points: tuple[dict[StrictStr, StrictStr], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unit_points(self) -> Self:
        seen: set[tuple[str, ...]] = set()
        for point in self.points:
            if point.keys() != layout.UNIT_SPACE.keys() and point.keys() != layout.LOCKED_UNIT_SPACE.keys():
                raise ValueError("回饋點的量名與搜尋空間不同")
            for encoded in point.values():
                value = float.fromhex(encoded)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0 or value.hex() != encoded:
                    raise ValueError("回饋點必須是單位空間內的有限 float.hex（十六進位浮點表示）")
            key = tuple(point[name] for name in layout.SEARCH_QUANTITIES if name in point)
            if key in seen:
                raise ValueError("回饋點不准重複")
            seen.add(key)
        return self

    def unit_points(self) -> tuple[dict[str, float], ...]:
        return tuple({name: float.fromhex(point[name]) for name in layout.SEARCH_QUANTITIES if name in point}
                     for point in self.points)


def _check_sequence(events: Sequence[FeedbackEvent]) -> None:
    before = 0
    for expected, event in enumerate(events, start=2):
        if event.round != expected or event.before_batch < before:
            raise ValueError("回饋輪次必須連續遞增，before_batch（批界）不准倒退")
        before = event.before_batch


def _read_handle(handle: BinaryIO) -> tuple[tuple[FeedbackEvent, ...], int]:
    lines = handle.readlines()
    events: list[FeedbackEvent] = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            break  # 只捨棄缺換行的不完整末列；完整的壞列與型別錯誤一定拒絕。
        try:
            events.append(FeedbackEvent.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"回饋事件第 {index + 1} 列壞了：{error}") from error
        valid_bytes += len(line)
    _check_sequence(events)
    return tuple(events), valid_bytes


class FeedbackLedger:
    """回饋事件沒有搜尋帳表頭；排他鎖內核序、修復半列、追加並同步到磁碟。"""

    @staticmethod
    def read(path: Path) -> tuple[FeedbackEvent, ...]:
        try:
            with path.open("rb") as handle:
                return _read_handle(handle)[0]
        except FileNotFoundError:
            return ()

    @staticmethod
    def append(path: Path, event: FeedbackEvent) -> None:
        event = FeedbackEvent.model_validate(event.model_dump())
        # 先驗序；拒絕第一次追加時也不建立空檔。
        _check_sequence((*FeedbackLedger.read(path), event))
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            events, valid_bytes = _read_handle(handle)
            _check_sequence((*events, event))
            handle.seek(valid_bytes)
            handle.truncate()
            _write_line(handle, event)


def feedback_points(anchor: ledger.LedgerRow, rows: Sequence[ledger.LedgerRow],
                    offset: float) -> tuple[dict[str, float], ...]:
    """按量序、先負後正；裁邊後以 float.hex（十六進位浮點表示）逐位去重。"""
    quantities = tuple(name for name in layout.SEARCH_QUANTITIES if name in anchor.unit_params_hex)
    center = {name: float.fromhex(anchor.unit_params_hex[name]) for name in quantities}
    seen = {tuple(row.unit_params_hex[name] for name in quantities) for row in (*rows, anchor)}
    points = []
    for name in quantities:
        for direction in (-1, 1):
            point = center | {name: min(1.0, max(0.0, center[name] + direction * offset))}
            key = tuple(point[quantity].hex() for quantity in quantities)
            if key not in seen:
                seen.add(key)
                points.append(point)
    if not points:
        raise FeedbackUnavailable("沒有可回饋的點：裁邊與逐位去重後沒有剩餘")
    return tuple(points)


def comparison_trial(store: SearchStore, status: SearchStatus) -> int | None:
    """首輪比篩選第一名；之後按本輪查中心，不讀尚未落狀態的下一輪事件。"""
    if status.round == 1:
        return status.best_trial
    event = next((event for event in FeedbackLedger.read(store.feedback_path) if event.round == status.round), None)
    if event is None:
        raise ValueError("本輪回饋中心事件不存在，不能比較第一名")
    return event.anchor_trial


def _accepted_status(store: SearchStore) -> SearchStatus:
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    if status.state == "budget_exhausted":
        raise ValueError("搜尋預算已用完，回饋跑不了")
    if status.state != "converged":
        raise ValueError("搜尋尚未達到停止條件，不能回饋")
    if store.stop_path.exists():
        raise ValueError("搜尋停止記號還在：先確認要不要繼續，刪掉停止記號再回饋")
    if store.settings.feedback is None:
        raise ValueError("搜尋設定沒有 feedback（回饋），這場搜尋不准回饋")
    if status.refine.state != "stopped":
        raise ValueError("細算還沒停，不能回饋")
    if status.refine.best == "baseline":
        raise ValueError("細算第一名是原方案，沒有可回饋的點")
    if type(status.refine.best) is not int:
        raise ValueError("細算沒有試算編號作為第一名，不能回饋")
    if status.refine.round != status.round:
        raise ValueError("細算輪次跟搜尋輪次不同，不能回饋")
    if status.refine.best == comparison_trial(store, status):
        if status.round == 1:
            raise ValueError("細算第一名跟篩選第一名相同，不需要回饋")
        raise ValueError("細算第一名沒換：跟上一次回饋中心相同，不需要回饋")
    if status.asked >= store.settings.budget:
        raise ValueError("搜尋預算已用完，回饋跑不了")
    if status.asked % store.settings.batch_size:
        raise ValueError("已要題數不在完整批界，不能回饋")
    return status


def replay_enqueues(store: SearchStore, status: SearchStatus, rows: Sequence[ledger.LedgerRow],
                    completed_batches: int, *, events: Sequence[FeedbackEvent] | None = None,
                    ) -> Mapping[int, tuple[dict[str, float], ...]]:
    """核事件與輪次帳的批界；已完成批交重播，下一批交搜尋迴圈排入。events 沒給就讀事件檔。"""
    if events is None:
        events = FeedbackLedger.read(store.feedback_path)
    space = layout.space_for(store.settings.layout)
    if any(point.keys() != space.keys() for event in events for point in event.points):
        raise ReplayMismatch("回饋點的量名與搜尋空間不同")
    if status.round != len(events) + 1 or len(status.rounds) != len(events):
        raise ValueError("回饋事件與搜尋輪次紀錄不一致")
    enqueues: dict[int, tuple[dict[str, float], ...]] = {}
    by_number = {row.trial_number: row for row in rows}
    for event, record in zip(events, status.rounds, strict=True):
        boundary = event.before_batch * store.settings.batch_size
        if (record.round != event.round - 1 or record.state != "converged" or record.asked != boundary
                or event.before_batch > completed_batches or boundary >= store.settings.budget):
            raise ValueError("回饋 before_batch（批界）對不上完整搜尋批與輪次紀錄")
        anchor = by_number.get(event.anchor_trial)
        if anchor is None or event.anchor_trial >= boundary:
            raise ValueError("回饋中心試算不在停止時的搜尋帳本內")
        enqueues[event.before_batch] = (*enqueues.get(event.before_batch, ()), *event.unit_points())
    start = events[-1].before_batch * store.settings.batch_size if events else 0
    if status.round_start_trial != start:
        raise ValueError("回饋批界跟本輪起始試算編號不同")
    if events and store.settings.feedback is None:
        raise ValueError("沒有 feedback（回饋）設定卻有回饋事件")
    return enqueues


def feedback_search(store: SearchStore) -> SearchStatus:
    """先驗全部前提與帳本，再記事件與新狀態；細算子物件原封保留。"""
    status = _accepted_status(store)
    recorded = ledger.read_for(store)
    size = store.settings.batch_size
    numbers = {row.trial_number for row in recorded.rows}
    if numbers != set(range(status.asked)) or any(row.batch_index != row.trial_number // size for row in recorded.rows):
        raise ValueError("搜尋帳本與已要題數對不上完整批界")
    events = FeedbackLedger.read(store.feedback_path)
    # 上次寫完事件、還沒寫狀態就被砍：事件比狀態多一輪。重跑時核對內容相同就只補寫狀態。
    pending = events[-1] if events and events[-1].round == status.round + 1 else None
    settled = events[:-1] if pending is not None else events
    replay_enqueues(store, status, recorded.rows, status.asked // size, events=settled)
    if settled and status.asked - settled[-1].before_batch * size < len(settled[-1].points):
        raise FeedbackUnavailable("上一輪的回饋點還沒問完就停了，剩下的點會在下一輪搶先被問到，不能再回饋")
    anchor = next((row for row in recorded.rows if row.trial_number == status.refine.best), None)
    if anchor is None:
        raise ValueError("細算第一名不在搜尋帳本裡，沒有可回饋的點")
    settings = store.settings.feedback
    assert settings is not None  # _accepted_status 已驗過必設。
    points = feedback_points(anchor, recorded.rows, settings.offset)
    event = FeedbackEvent(round=status.round + 1, before_batch=status.asked // size, anchor_trial=anchor.trial_number,
                          points=tuple({name: value.hex() for name, value in point.items()} for point in points))
    record = RoundRecord(**status.model_dump(include=set(RoundRecord.model_fields)))
    changed = status.model_copy(update={
        "round": event.round, "round_start_trial": status.asked, "rounds": (*status.rounds, record),
        "state": "running", "message": f"第 {event.round} 輪搜尋：在細算第一名 {anchor.trial_number} 號附近排入 {len(points)} 個點",
    })
    if pending is None:
        FeedbackLedger.append(store.feedback_path, event)
    elif pending != event:
        raise ValueError("回饋事件已寫但狀態沒更新，而且事件內容跟這次算出來的對不上，要人看過再處理")
    return _write_status(store, changed)
