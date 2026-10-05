"""搜尋進度的唯讀投影：不拿鎖、不讀候選結果、不重評、不寫搜尋資料夾。

判斷「有沒有計算行程在跑」讀核心公布的 /proc/locks（不呼叫 flock）。前提：網頁伺服器與搜尋行程在同一台機器、
同一個行程命名空間，搜尋資料夾在本機檔案系統（例如 ext4）上；任一邊改進容器、沙箱或網路檔案系統，鎖會看不到，
進行中的搜尋會被報成中斷——那時要換判斷方法（複查）。
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from aosr.search import ledger
from aosr.search.labels import SEARCH_STATES, REFINE_STATES, REFINE_STOP_REASONS, counts_text
from aosr.search.outer_status import OUTER_MESSAGES, OuterStatus, conclusion_message
from aosr.search.refine import RefineHeader, RefineLedger, RefineRead
from aosr.search.run import RefineStatus, SearchStatus
from aosr.search.store import SearchStore
from aosr.search.timings import NO_TIMINGS, NOT_YET, PARTIAL, round_text, timings_of, total_text

SEARCH_ID = re.compile(r"[0-9a-f]{32}\Z")
PARAM_LABELS = {"front_distance": "喇叭離前牆", "spacing": "兩支喇叭間距", "listening_distance": "聆聽距離"}
INTERRUPTED = "狀態檔說還在跑，但沒有計算行程拿著這個資料夾：上次中斷了，要接續請叫助理"
_T = TypeVar("_T")


@dataclass(frozen=True)
class Read(Generic[_T]):
    value: _T | None = None
    error: str = ""


@dataclass(frozen=True)
class Process:
    held: bool | None
    text: str


class Block(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    key: str
    title: str
    lines: tuple[str, ...]
    warning: bool = False


class SearchView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    search_id: str
    name: str
    fetched_text: str
    stage_text: str
    blocks: tuple[Block, ...]


def read_proc_locks() -> str:
    """讀核心公布的鎖清單；從不呼叫 flock（資料夾排他鎖）。"""
    return Path("/proc/locks").read_text(encoding="ascii")


def folder_process(folder: Path, locks: str | None = None) -> Process:
    """有沒有行程拿著這個資料夾的鎖；locks 給了就用那一份鎖清單（清單頁整頁只讀一次）。"""
    try:
        stat = folder.stat()
        for line in (read_proc_locks() if locks is None else locks).splitlines():
            fields = line.split()
            # 等待鎖的列多一個 ->；那不是已拿到的鎖。
            if len(fields) < 6 or fields[1] != "FLOCK":
                continue
            major, minor, inode = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
                return Process(True, "有計算行程拿著這個資料夾")
        return Process(False, "沒有計算行程拿著這個資料夾")
    except (OSError, ValueError, UnicodeError) as error:
        return Process(None, f"判不出有沒有行程在跑：{_reason(error)}")


def search_path(root: Path, search_id: str) -> Path:
    """只從合法代號組路徑；拒絕連到外面的資料夾。"""
    if not SEARCH_ID.fullmatch(search_id):
        raise ValueError("搜尋代號必須是 32 位小寫十六進位")
    path = root / search_id
    if path.is_symlink():
        raise ValueError("搜尋資料夾不能是符號連結")
    if not path.is_dir():
        raise FileNotFoundError("沒有這場搜尋")
    return path


def _reason(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return f"檔案不存在（{Path(error.filename).name if error.filename else '搜尋資料'}）"
    if isinstance(error, ValidationError):
        fields = "、".join(".".join(map(str, issue["loc"])) or "資料" for issue in error.errors())
        return f"欄位不符合伺服器認得的格式（{fields}）"
    if isinstance(error, json.JSONDecodeError):
        return "JSON 資料損壞或尚未寫完"
    return str(error)


def _read(read: Callable[[], _T]) -> Read[_T]:
    try:
        return Read(value=read())
    except (OSError, ValueError, UnicodeError) as error:
        return Read(error=f"讀不到：{_reason(error)}")


def _document(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("狀態檔必須是資料物件")
    return value


def _status_parts(document: Read[dict[str, object]]) -> tuple[Read[SearchStatus], Read[RefineStatus], Read[OuterStatus]]:
    if document.value is None:
        return Read(error=document.error), Read(error=document.error), Read(error=document.error)
    value = document.value
    # 三格各自驗：細算或外圈新欄位不會遮掉能讀的搜尋資料。搜尋頂層仍 extra=forbid。
    search = _read(lambda: SearchStatus.model_validate({key: item for key, item in value.items() if key not in {"refine", "outer"}}))
    refine = _read(lambda: RefineStatus.model_validate(value.get("refine", {})))
    outer = _read(lambda: OuterStatus.model_validate(value.get("outer", {})))
    return search, refine, outer


def _refine_book(path: Path, store: SearchStore | None) -> RefineRead:
    book = RefineLedger.read_status(path / "refine.jsonl")
    if store is not None:
        snapshot = ledger.header_for(store)
        expected = RefineHeader.model_validate({name: getattr(snapshot, name) for name in RefineHeader.model_fields if name != "ledger_version"}
                                              | {"ledger_version": book.header.ledger_version})
        if book.header != expected:
            raise ValueError("細算帳表頭跟搜尋資料夾的快照對不上")
    return book


def _phase(state: str, labels: dict[str, str], process: Process, phase: str) -> str:
    if state == "running" and process.held is not True:
        return f"{phase}中斷：{INTERRUPTED}" if process.held is False else process.text
    text = labels[state]
    return text + "（暫行）" if state == "converged" else text


def _stage(search: Read[SearchStatus], refine: Read[RefineStatus], outer: Read[OuterStatus], process: Process) -> Block:
    lines = [process.text]
    if search.value is None:
        lines.append(f"搜尋：{search.error}")
    else:
        status = search.value
        lines.append(f"搜尋：{_phase(status.state, SEARCH_STATES, process, '搜尋')}；第 {status.round} 輪")
    if refine.value is None:
        lines.append(f"細算：{refine.error}")
    else:
        status_r = refine.value
        round_text_r = "" if status_r.state == "not_started" else f"；第 {status_r.round} 輪"
        lines.append(f"細算：{_phase(status_r.state, REFINE_STATES, process, '細算')}{round_text_r}")
        if status_r.stop_reason is not None:
            lines.append(f"細算停止原因：{REFINE_STOP_REASONS[status_r.stop_reason]}")
    if search.value is None or refine.value is None or outer.value is None:
        lines.append(f"外圈結論：未判定；{outer.error or search.error or refine.error}")
    else:
        combined = search.value.model_copy(update={"refine": refine.value, "outer": outer.value})
        conclusion = conclusion_message(combined)
        # 狀態說仍在跑時，舊的完成結論不能凌駕行程查核。
        if (combined.state == "running" or combined.refine.state == "running") and process.held is not True:
            conclusion = "未判定（計算行程未確認）"
        lines.append(f"外圈結論：{conclusion}")
        if outer.value.conclusion is not None and conclusion.startswith("未判定"):
            lines.append(f"先前結論（已過期）：{OUTER_MESSAGES[outer.value.conclusion]}")
    lines.append("品質合格：本頁未判定；停止或細算完成不代表品質合格，也不是最終推薦")
    warning = bool(search.error or refine.error or outer.error or process.held is None)
    return Block(key="stage", title="目前階段", lines=tuple(lines), warning=warning)


def _counts(search: Read[SearchStatus], book: Read[ledger.LedgerRead], refined: Read[RefineRead],
            refine_not_yet: bool = False) -> Block:
    lines = [f"問過 {search.value.asked} 題（到狀態最後存檔為止）" if search.value is not None else f"問過：{search.error}"]
    if book.value is None:
        lines.append(f"搜尋帳：{book.error}")
    else:
        rows = book.value.rows
        illegal = [row for row in rows if row.outcome == "illegal"]
        reasons = Counter(reason for row in illegal for reason in (row.reason or "").split("+"))
        excluded = Counter(row.reason or "未辨識" for row in rows if row.outcome == "excluded")
        lines.extend((f"算完 {sum(row.outcome != 'illegal' for row in rows)} 個；不合法 {len(illegal)} 個",
                      f"不合法各原因數：{counts_text(dict(reasons))}",
                      f"被淘汰或排除 {sum(excluded.values())} 個；各區數：{counts_text(dict(excluded))}"))
        if book.value.dropped_last_line:
            lines.append("搜尋帳末列尚未寫完，只顯示完整列")
    if refine_not_yet:
        # 細算還沒開始時細算帳本來就還不存在：照實寫「還沒有」，不當成讀不到標紅（老闆看實跑前主對話截圖抓到）。
        lines.append("細算帳：還沒有（細算還沒開始）")
    else:
        lines.append(f"細算做了 {len(refined.value.rows)} 個（細算帳完整列）" if refined.value is not None
                     else f"細算帳：{refined.error}")
    if refined.value is not None and refined.value.dropped_last_line:
        lines.append("細算帳末列尚未寫完，只顯示完整列")
    refine_problem = "" if refine_not_yet else refined.error
    return Block(key="counts", title="候選處理數量", lines=tuple(lines), warning=bool(book.error or refine_problem or search.error))


def _timings(search: Read[SearchStatus], refine: Read[RefineStatus]) -> Block:
    lines = ["到最後一次存檔為止；不是目前行程的即時計時"]
    if search.value is None:
        lines.append(f"搜尋時間：{search.error}")
        lines.append(round_text(refine.value.seconds, "細算", from_start=False) if refine.value else f"細算時間：{refine.error}")
        lines.append("合計讀不到：搜尋時間讀不到")
    else:
        status = search.value
        timings = timings_of(status.model_copy(update={"refine": refine.value or RefineStatus()}))
        if not (timings.search or timings.refine):
            lines.append(NOT_YET if timings.from_start else NO_TIMINGS)
        lines.append(round_text(timings.search, "搜尋", from_start=timings.from_start))
        if refine.value is None:
            lines.extend((f"細算時間：{refine.error}", "合計讀不到：細算時間讀不到"))
        else:
            lines.extend((round_text(timings.refine, "細算", from_start=timings.from_start), total_text(timings)))
        if not timings.from_start and (timings.search or timings.refine):
            lines.append(PARTIAL)
    return Block(key="timings", title="已用時間", lines=tuple(lines), warning=bool(search.error or refine.error))


def _updated(path: Path, now: float, not_yet: tuple[str, ...] = ()) -> Block:
    """not_yet 列的是本來就還不該存在的檔（細算還沒開始時的細算帳）：不存在就略過，不寫讀不到。"""
    times: list[float] = []
    lines: list[str] = []
    for name in ("status.json", "ledger.jsonl", "refine.jsonl"):
        if name in not_yet and not (path / name).exists():
            continue
        timestamp = _read(lambda: (path / name).stat().st_mtime)
        if timestamp.value is None:
            lines.append(f"{name}：{timestamp.error}")
        else:
            times.append(timestamp.value)
    if times:
        latest = max(times)
        age = max(0, int(now - latest))
        lines.insert(0, f"最後更新：{datetime.fromtimestamp(latest).astimezone():%Y-%m-%d %H:%M:%S %Z}；距今 {age // 60} 分 {age % 60} 秒")
        if latest > now:
            lines.append("存檔時間在伺服器現在時間之後，請核對時鐘")
    return Block(key="updated", title="最後更新時間", lines=tuple(lines), warning=bool(lines and not times))


def _best(search: Read[SearchStatus], book: Read[ledger.LedgerRead]) -> Block:
    lines: tuple[str, ...]
    if search.value is None:
        lines = (search.error,)
    else:
        status = search.value
        label = "本次預算內最佳" if status.state == "budget_exhausted" else "帳上搜尋第一名"
        lines = (f"{label}：沒有取得篩選分數的候選",)
        if status.best_trial is not None:
            score = "讀不到：狀態缺篩選分數" if status.best_score is None else f"{status.best_score:.4f}"
            lines = (f"{label}：試算 {status.best_trial}；篩選分數：{score}",)
            row = next((item for item in book.value.rows if item.trial_number == status.best_trial), None) if book.value else None
            if row is None:
                lines += (f"擺位參數：{book.error or '讀不到：搜尋帳沒有這個編號的完整列'}",)
            elif row.outcome != "scored" or row.score != status.best_score:
                lines += ("讀不到：狀態的第一名與搜尋帳不一致，可能正在更新",)
            else:
                lines += tuple(f"{PARAM_LABELS.get(name, name)}：{value:.3f} 公尺" for name, value in row.params_m.items())
    return Block(key="search-best", title="搜尋最佳", lines=lines, warning=any("讀不到" in line for line in lines))


def _refine_best(refine: Read[RefineStatus]) -> Block:
    if refine.value is None:
        lines = (refine.error,)
    else:
        status = refine.value
        name = "原方案" if status.best == "baseline" else f"試算 {status.best}"
        cost = "讀不到：狀態缺總代價" if status.best_total_cost is None else f"{status.best_total_cost:.4f}"
        lines = ("尚無可排名的細算第一名",) if status.best is None else (f"帳上細算第一名：{name}；總代價：{cost}",)
    return Block(key="refine-best", title="細算最佳", lines=lines, warning=bool(refine.error))


def _reasons(search: Read[SearchStatus], refine: Read[RefineStatus], process: Process) -> Block:
    lines = [f"搜尋訊息原文：{search.value.message}" if search.value else f"搜尋訊息：{search.error}",
             f"細算訊息原文：{refine.value.message}" if refine.value else f"細算訊息：{refine.error}"]
    phases: tuple[tuple[str, SearchStatus | RefineStatus | None], ...] = (("搜尋", search.value), ("細算", refine.value))
    for phase, status in phases:
        if status is not None and status.state in {"failed", "interrupted"}:
            lines.append(f"停止段落：{phase}{SEARCH_STATES[status.state]}")
        if status is not None and status.state == "running" and process.held is not True:
            lines.append(f"停止段落：{phase}；{INTERRUPTED if process.held is False else process.text}")
    if refine.value and refine.value.stop_reason is not None:
        lines.append(f"細算停止原因：{REFINE_STOP_REASONS[refine.value.stop_reason]}")
    return Block(key="reasons", title="停止或失敗原因", lines=tuple(lines))


def _identity(store: Read[SearchStore], physics: str, program: str) -> Block:
    lines = [f"伺服器現在的物理身分：{physics}", f"伺服器程式指紋：{program}"]
    if store.value is None:
        lines.append(f"搜尋身分與快照：{store.error}")
    else:
        identity = store.value.identity
        lines.extend((f"資料夾物理身分：{identity.physics_identity}", f"資料夾程式指紋：{identity.program_fingerprint}"))
        if identity.program_fingerprint != program:
            lines.append("這場搜尋用的是較舊的程式（指紋與伺服器不同；這是備查資訊）")
        if identity.physics_identity != physics:
            lines.append("這場搜尋的物理身分與伺服器不同（備查資訊）")
    return Block(key="identity", title="身分", lines=tuple(lines), warning=bool(store.error))


def build_search_view(path: Path, *, server_physics: str, server_program: str) -> SearchView:
    """只開快照、狀態與兩本帳；每塊讀不到都留原因，其餘照常。"""
    now = time.time()
    # 先查鎖再讀狀態：反過來的話，搜尋剛好在中間寫完狀態、放掉鎖，那一次會閃一下「中斷」（複查）。
    process = folder_process(path)
    store = _read(lambda: SearchStore.open(path))
    search, refine, outer = _status_parts(_read(lambda: _document(path / "status.json")))
    book = _read(lambda: ledger.read_for(store.value) if store.value is not None else ledger.Ledger.read_status(path / "ledger.jsonl"))
    refined = _read(lambda: _refine_book(path, store.value))
    stage = _stage(search, refine, outer, process)
    refine_not_yet = (refine.value is not None and refine.value.state == "not_started"
                      and not (path / "refine.jsonl").exists())
    return SearchView(search_id=path.name, name=store.value.project.scheme_id if store.value else path.name,
                      fetched_text=datetime.fromtimestamp(now).astimezone().strftime("%H:%M:%S"),
                      stage_text="；".join(stage.lines[1:3]),
                      blocks=(stage, _counts(search, book, refined, refine_not_yet), _timings(search, refine),
                              _updated(path, now, ("refine.jsonl",) if refine_not_yet else ()),
                              _best(search, book), _refine_best(refine), _reasons(search, refine, process),
                              _identity(store, server_physics, server_program)))


def _list_item(path: Path, locks: Read[str]) -> dict[str, str]:
    """清單頁一場一列：只讀快照名字與狀態檔，不讀兩本帳（清單每 5 秒問一次，場數多也不能拖）。"""
    process = (Process(None, f"判不出有沒有行程在跑：{locks.error.removeprefix('讀不到：')}") if locks.value is None
               else folder_process(path, locks.value))
    store = _read(lambda: SearchStore.open(path))
    search, refine, outer = _status_parts(_read(lambda: _document(path / "status.json")))
    stage = _stage(search, refine, outer, process)
    return {"search_id": path.name, "name": store.value.project.scheme_id if store.value else path.name,
            "stage_text": "；".join(stage.lines[:3]), "url": f"/searches/{path.name}"}


def list_searches(root: Path) -> dict[str, object]:
    """只列指定根目錄的合法資料夾；不往外遞迴。"""
    items: list[dict[str, str]] = []
    error = ""
    try:
        locks = _read(read_proc_locks)
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if SEARCH_ID.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
                items.append(_list_item(path, locks))
    except OSError as exc:
        error = f"讀不到：{_reason(exc)}"
    return {"searches": items, "error": error, "fetched_text": datetime.now().astimezone().strftime("%H:%M:%S")}
