"""第一階段入選清單：只讀每席 1 Hz 細算結果，攤開空間代價與分項。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from aosr.config.paths import config_path
from aosr.config.room_proportions import RoomProportions, load_room_proportions
from aosr.scoring.stage_one_score import PRINCIPAL, ROBUSTNESS
from aosr.search.room_proportions import (
    PROPORTION_EXPLANATION, RoomProportionNotice, room_proportion_notice, room_proportion_text,
)
from aosr.search.run import ComputeFailed, IdentityChanged
from aosr.search.stage_one_refine_record import RefinedSeat, StageOneRefineStatus, read_stage_one_refined_result
from aosr.search.stage_one_refine_report import _mismatch, missing_control_lines
from aosr.search.stage_one_selection_contract import CONTROL_TEXT, SelectionSeat, role_text
from aosr.search.store import SearchStore

TITLE = "入選清單（細算後；前 {top_n} 席照細算名次，對照列出來、不算入選）"
# 這兩句是主對話定的報告字樣；不是決策紙所引的老闆原話。
NOT_FINAL = "細算第一名不是最終推薦：名次只照現行低頻合分排"
RANK_AND_QUALITY = "名次是低頻合分的排序；品質可不可接受另看入選後檢查（超線共振；改善能力尚未確認），兩件事分開看"
DELTA_NOTICE = "差值都對細算第一名，用 1 Hz 驗證軸；分項是原始 dB"
FIXED_ROOM = "房間固定，下列是同一間房的不同擺法"
VARIABLE_ROOM = "房間隨候選而變，面積與容積照每一席自己的房間算"
NO_WINNER = "沒有細算第一名，不算差值"
AREA_NOTICE = "面積對照只用來比較空間代價，不宣稱已證明哪種大小或比例最好"
STATE_TEXT = {"pending": "尚未完成", "scored": "已計分", "unassessed": "不可估",
              "incomparable": "不能同表", "unreadable": "讀不回"}
HEADINGS = ("試算編號", "角色", "搜尋名次", "細算名次", "主項單獨名次", "細算狀態", "房間 長×寬×高（公尺）",
    "地板面積 m²", "容積 m³", "主項 dB", "座位穩健性 dB", "合分", "主項單獨排序鍵 dB",
    "Δ主項 dB／Δ座位穩健性 dB／合分差／Δ地板面積 m²／Δ容積 m³", "比例提醒")


@dataclass(frozen=True)
class ShortlistValues:
    """全部來自同一席細算結果；面積、容積只在報告層計算。"""

    length_m: float
    width_m: float
    height_m: float
    floor_area_m2: float
    volume_m3: float
    principal_db: float
    robustness_db: float
    screening_total: float
    principal_only_sort_key_db: float


@dataclass(frozen=True)
class ShortlistDelta:
    """本席減細算第一名；合分差無單位，分項差是原始 dB。"""

    principal_db: float
    robustness_db: float
    screening_total: float
    floor_area_m2: float
    volume_m3: float


@dataclass(frozen=True)
class ShortlistRow:
    """每席一列；不能計分或讀不回時不帶數值，也不以零補空格。"""

    trial_number: int
    roles: tuple[str, ...]
    is_top_n: bool
    search_rank: int
    refine_rank: int | None
    state: str
    reason: str
    values: ShortlistValues | None = None
    principal_rank: int | None = None
    delta: ShortlistDelta | None = None
    proportion: RoomProportionNotice | None = None


@dataclass(frozen=True)
class StageOneShortlist:
    """供命令列與網頁共用的唯讀投影；不分房間群組。"""

    top_n: int
    room_range: bool
    rows: tuple[ShortlistRow, ...]
    # 沒取到的對照各一句原因：命令列已印在細算那一段，只有網頁另外接在說明行後面。
    missing: tuple[str, ...] = ()


def _row(store: SearchStore, status: StageOneRefineStatus, seat: SelectionSeat,
         refined: RefinedSeat | None, proportions: RoomProportions) -> ShortlistRow:
    state = refined.state if refined is not None else "pending"
    reason = refined.reason if refined is not None else ""
    if seat.error is not None:
        state, reason = "unreadable", seat.error
    row = ShortlistRow(seat.trial_number, tuple(role_text(role, seat.search_rank) for role in seat.roles), "top_n" in seat.roles,
        seat.search_rank, None, state, reason)
    if state != "scored" or refined is None:
        return replace(row, reason=reason or f"細算結果{STATE_TEXT[state]}")
    try:
        result = read_stage_one_refined_result(store, seat.trial_number)
        changed = _mismatch(result, refined, status)
        if changed is not None:
            raise ValueError(f"細算結果跟摘要記的{changed}不同，要重跑細算")
        score, room = result.score, result.scheme.scene.room_m
        if score.state != "scored" or score.screening_total is None or score.principal_only_sort_key_db is None:
            raise ValueError("細算結果沒有可用分數")
        parts = {part.name: part.value_db for part in score.components}
        area = room.Lx * room.Ly
        values = ShortlistValues(room.Lx, room.Ly, room.Lz, area, area * room.Lz,
            parts[PRINCIPAL], parts[ROBUSTNESS], score.screening_total, score.principal_only_sort_key_db)
        return replace(row, refine_rank=refined.refine_rank if row.is_top_n else None, values=values,
            proportion=room_proportion_notice(values.length_m, values.width_m, values.height_m, proportions))
    except (OSError, ValueError, KeyError, ComputeFailed, IdentityChanged) as error:
        return replace(row, state="unreadable", reason=f"細算結果讀不回：{error}")


def _principal_key(row: ShortlistRow) -> tuple[float, int]:
    assert row.values is not None
    return row.values.principal_only_sort_key_db, row.trial_number


def _delta(values: ShortlistValues, base: ShortlistValues) -> ShortlistDelta:
    return ShortlistDelta(values.principal_db - base.principal_db, values.robustness_db - base.robustness_db,
        values.screening_total - base.screening_total, values.floor_area_m2 - base.floor_area_m2, values.volume_m3 - base.volume_m3)


def stage_one_shortlist(store: SearchStore) -> StageOneShortlist | None:
    """缺細算摘要回空；壞摘要由文字入口隔離，單席壞檔由列入口隔離。"""
    path = store.path / "stage_one_refine.json"
    if not path.exists():
        return None
    status = StageOneRefineStatus.model_validate_json(path.read_bytes())
    results = {seat.trial_number: seat for seat in status.results}
    proportions = load_room_proportions(config_path("room_proportions.toml"))
    rows = [_row(store, status, seat, results.get(seat.trial_number), proportions) for seat in status.selection.candidates]
    top = sorted((row for row in rows if row.is_top_n),
        key=lambda row: row.refine_rank if row.refine_rank is not None else float("inf"))
    controls = [row for row in rows if not row.is_top_n]
    principal = sorted((row for row in top if row.values is not None), key=_principal_key)
    ranks = {row.trial_number: rank for rank, row in enumerate(principal, 1)}
    base = next((row.values for row in top if row.refine_rank == 1), None)
    return StageOneShortlist(status.selection.top_n, store.settings.layout.room_range is not None,
        tuple(replace(row, principal_rank=ranks.get(row.trial_number),
            delta=_delta(row.values, base) if row.values is not None and base is not None else None)
            for row in (*top, *controls)), missing_control_lines(status.selection))


def _rank(value: int | None) -> str:
    return "無" if value is None else str(value)


def shortlist_cells(row: ShortlistRow, *, has_winner: bool) -> tuple[str, ...]:
    """命令列與網頁共用同一列已格式化的字串格子。"""
    reason = "；".join(row.reason.splitlines())
    state = STATE_TEXT[row.state] + (f"；{reason}" if row.state != "scored" else "")
    fields = [f"試算 {row.trial_number}", "、".join(row.roles), str(row.search_rank),
              _rank(row.refine_rank), _rank(row.principal_rank), state]
    values = row.values
    if values is not None:
        fields.extend([f"{values.length_m:.6f}×{values.width_m:.6f}×{values.height_m:.6f}",
            *(f"{value:.6f}" for value in (values.floor_area_m2, values.volume_m3, values.principal_db,
                values.robustness_db, values.screening_total, values.principal_only_sort_key_db))])
    else:
        fields.extend(("無", "無", "無", "無", "無", "無", "無"))
    if not has_winner:
        fields.append(NO_WINNER)
    elif row.delta is not None:
        delta = row.delta
        fields.append("／".join(f"{value:+.6f}" for value in (delta.principal_db, delta.robustness_db,
            delta.screening_total, delta.floor_area_m2, delta.volume_m3)))
    else:
        fields.append("此席沒有可用細算分數，不算差值")
    fields.append(room_proportion_text(row.proportion))
    return tuple(fields)


def shortlist_notes(shortlist: StageOneShortlist) -> tuple[str, ...]:
    """命令列與網頁共用的說明行；有面積對照那一席才多一行它只比空間代價。"""
    area = any(CONTROL_TEXT["area_control"] in row.roles for row in shortlist.rows)
    return (NOT_FINAL, RANK_AND_QUALITY, DELTA_NOTICE, VARIABLE_ROOM if shortlist.room_range else FIXED_ROOM,
            *((AREA_NOTICE,) if area else ()), PROPORTION_EXPLANATION)


def _line(row: ShortlistRow, *, has_winner: bool) -> str:
    return "｜".join(shortlist_cells(row, has_winner=has_winner))


def stage_one_shortlist_lines(store: SearchStore) -> tuple[str, ...]:
    """不重印報告開頭三項假設；讀不回只回這一段的一行錯誤。"""
    try:
        shortlist = stage_one_shortlist(store)
        if shortlist is None:
            return ()
        lines = [TITLE.format(top_n=shortlist.top_n), *shortlist_notes(shortlist), "｜".join(HEADINGS)]
        has_winner = any(row.refine_rank == 1 for row in shortlist.rows)
        return (*lines, *(_line(row, has_winner=has_winner) for row in shortlist.rows))
    except (OSError, ValueError, KeyError) as error:
        return ("入選清單讀不回：" + "；".join(str(error).splitlines()),)
