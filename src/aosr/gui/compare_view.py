"""兩份已驗結果的比較頁顯示資料。"""
from __future__ import annotations

import csv
import io
from datetime import date

import numpy as np

from aosr.config.quality_targets import QualityTargets
from aosr.gui.labels import DIRECTIONS, LOW_FREQUENCY_AXES, SOURCE_MODELS
from aosr.reporting.compare import compare_results, comparison_problems, identity_difference_groups
from aosr.reporting.display import (
    BASELINE_NOTE, LOW_FREQUENCY_DECAY_NOTE, REVERBERATION_ROOM_NOTE,
    SPATIAL_IMPRESSION_NOTE,
)
from aosr.reporting.result import SchemeResult
from aosr.gui.result_view import CategoryView, LABELS, ResultView, ViewModel
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import QualityCategory


class SideIdentity(ViewModel):
    run_id: str
    scheme_id: str
    engine_text: str
    run_date: str
    total_text: str


class SchemeChange(ViewModel):
    path: str
    label: str
    a_text: str
    b_text: str


class FingerprintCheck(ViewModel):
    label: str
    same: bool
    text: str


class OverlaySeries(ViewModel):
    key: str
    side: str
    role: str
    speaker_id: str
    receiver_id: str
    receiver_role: str
    legend_text: str
    levels_db: tuple[float | None, ...]


class OverlayPair(ViewModel):
    label: str
    a_key: str
    b_key: str


class Overlay(ViewModel):
    frequency_hz: tuple[float, ...]
    series: tuple[OverlaySeries, ...]
    default_keys: tuple[str, str]
    pairs: tuple[OverlayPair, ...]


class CategoryRow(ViewModel):
    category: str
    label: str
    a: CategoryView
    b: CategoryView
    # 兩份不同表時，身分不同的那幾類寫「兩邊身分不同，代價不能直接比」；其餘空字串。
    comparison_text: str


class TableStatus(ViewModel):
    same_table: bool
    a_text: str
    b_text: str
    reason_text: str
    calibration_text: str


class CompareView(ViewModel):
    a: SideIdentity
    b: SideIdentity
    changes: tuple[SchemeChange, ...]
    changed_keys: tuple[str, ...]
    fingerprints: tuple[FingerprintCheck, ...]
    summary_text: str
    overlay: Overlay
    categories: tuple[CategoryRow, ...]
    table: TableStatus
    notes: tuple[str, ...]
    labels: dict[str, str]
    # 頻響疊圖的音量基準但書：頁面、匯出圖片、摘要 CSV 共用這一句，匯出去的東西也帶著它。
    level_note: str


LEVEL_NOTE = ("兩邊 dB 用同一個基準（單位振幅點源、距離 1 公尺），沒有各自對齊音量；"
              "不是校準過的絕對聲壓級。")


def _csv(rows: list[list[str | float | None]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerows(rows)
    return "\ufeff" + output.getvalue()


def curves_csv(view: CompareView) -> str:
    """輸出全部疊圖線的伺服器原值，空點留空。"""
    overlay = view.overlay
    rows: list[list[str | float | None]] = [
        ["頻率 (Hz)", *(f"{series.legend_text} (dB)" for series in overlay.series)]]
    rows.extend([frequency, *(series.levels_db[index] for series in overlay.series)]
                for index, frequency in enumerate(overlay.frequency_hz))
    return _csv(rows)


def summary_csv(view: CompareView) -> str:
    """依比較頁已排好的文字輸出；每一段的表頭說清楚每一欄是什麼，不把說明塞進「A」那一欄。

    段落：兩邊身分與同表（欄位／A／B）、整體說明（欄位／內容）、改了哪裡（項目／A／B）、
    指紋（指紋／核對）、各類結果（跟頁面同一張表）、固定說明。段與段之間空一列。
    """
    rows: list[list[str | float | None]] = [
        ["欄位", "A", "B"],
        ["方案代號", view.a.scheme_id, view.b.scheme_id],
        ["引擎", view.a.engine_text, view.b.engine_text],
        ["日期", view.a.run_date, view.b.run_date],
        ["全程", view.a.total_text, view.b.total_text],
        ["同表", view.table.a_text, view.table.b_text],
        [],
        ["欄位", "內容"],
        ["摘要句", view.summary_text],
        ["原因", view.table.reason_text],
        ["校準說明", view.table.calibration_text],
        ["音量基準", view.level_note],
        [],
        ["項目", "A", "B"],
    ]
    rows.extend([change.label, change.a_text, change.b_text] for change in view.changes)
    rows.extend([[], ["指紋", "核對"]])
    rows.extend([check.label, check.text] for check in view.fingerprints)
    rows.extend([[], ["類別", "A 狀態", "A 代價", "B 狀態", "B 代價", "說明"]])
    for category in view.categories:
        notes = dict.fromkeys(note for note in (
            category.a.note, category.b.note, category.comparison_text) if note)
        rows.append([category.label, category.a.state_label, category.a.cost_text,
                     category.b.state_label, category.b.cost_text, "；".join(notes)])
    rows.extend([[], ["說明"]])
    rows.extend([note] for note in view.notes)
    return _csv(rows)


# 跟輸入頁同一套牆名（app.js 的 wallNames）；x、y 起點終點沒有前後左右的定義，不自己翻成前牆後牆。
_WALLS = {"floor": "地板", "ceiling": "天花", "x0": "x 起點牆", "xL": "x 終點牆",
          "y0": "y 起點牆", "yL": "y 終點牆"}
_FIELDS = {
    "scene.room_m.Lx": ("房間長度 Lx", "公尺"),
    "scene.room_m.Ly": ("房間寬度 Ly", "公尺"),
    "scene.room_m.Lz": ("房間高度 Lz", "公尺"),
    "scene.sound_speed_m_s": ("聲速", "公尺／秒"),
    "scene.density_kg_m3": ("密度", "公斤／立方公尺"),
    "scene.reflection_order_k": ("反射階數", "階"),
    "scene.low_frequency_axis": ("低頻軸", ""),
    "source_model": ("聲源模型", ""),
    "purpose": ("用途", ""),
    "channel_group.feature_match_tolerance_hz": ("特徵配對容差", "Hz"),
}


def _plain(value: float | int, digits: int) -> str:
    """有效位數 digits 的一般寫法；不用科學記號（1.04e+04 會被讀成 1.04）。"""
    if isinstance(value, int):
        return str(value)
    return str(np.format_float_positional(value, precision=digits, unique=False,
                                          fractional=False, trim="-"))


def _with_unit(text: str, unit: str) -> str:
    return f"{text} {unit}" if unit else text


def _number_pair(a: float | int, b: float | int, unit: str) -> tuple[str, str]:
    """兩個不相等的數：從 4 位有效數字起，印出來一樣就加位數；到 15 位還一樣就註明微小差異，
    不印出浮點原值的尾巴。"""
    for digits in range(4, 16):
        left, right = _plain(a, digits), _plain(b, digits)
        if left != right:
            return _with_unit(left, unit), _with_unit(right, unit)
    return _with_unit(left, unit), _with_unit(right, unit) + "（微小差異）"


def _text(value: object, unit: str) -> str:
    if value is None:
        return "未設定"
    if isinstance(value, bool):
        return "有" if value else "無"
    if isinstance(value, float | int):
        return _with_unit(_plain(value, 4), unit)
    if isinstance(value, tuple):
        return "、".join(_text(item, "") for item in value)
    return str(value)


def _collect_scheme(scheme: Scheme, into: dict[str, object],
                    labels: dict[str, tuple[str, str]]) -> None:
    room = scheme.scene.room_m
    into.update({f"scene.room_m.{axis}": getattr(room, axis) for axis in ("Lx", "Ly", "Lz")})
    for field in ("sound_speed_m_s", "density_kg_m3", "reflection_order_k"):
        into[f"scene.{field}"] = getattr(scheme.scene, field)
    low_axis = scheme.scene.low_frequency_axis
    into["scene.low_frequency_axis"] = (None if low_axis is None
                                        else LOW_FREQUENCY_AXES.get(low_axis.value, low_axis.value))
    for field, title, unit in (("impedance_pa_s_per_m_by_wall", "阻抗", "帕·秒／公尺"),
                               ("scattering_by_wall", "散射", "")):
        wall_values = getattr(scheme.scene, field)
        if field == "scattering_by_wall":
            into[f"scene.{field}"] = wall_values is not None
            labels[f"scene.{field}"] = ("散射整組", "")
        for wall in sorted(set(wall_values or {})):
            path = f"scene.{field}.{wall}"
            into[path] = wall_values[wall] if wall_values is not None else None
            labels[path] = (f"{_WALLS.get(wall, wall)}{title}", unit)
    into["source_model"] = SOURCE_MODELS.get(scheme.source_model, scheme.source_model)
    into["purpose"] = LABELS.get(scheme.purpose, scheme.purpose)
    for speaker_id, point in scheme.speakers.items():
        for axis in ("x", "y", "z"):
            path = f"speakers.{speaker_id}.{axis}"
            into[path] = getattr(point, axis)
            labels[path] = (f"喇叭 {speaker_id} {axis}", "公尺")
    for channel in scheme.channel_group.channels:
        path = f"channel_group.channels.{channel.role}"
        into[path] = channel.speaker_id
        labels[path] = (f"{LABELS.get(channel.role, channel.role)}喇叭", "")
    for comparison in scheme.channel_group.comparisons:
        path = f"channel_group.comparisons.{comparison.left_role}.{comparison.right_role}"
        into[path] = (comparison.left_role, comparison.right_role)
        labels[path] = ("聲道比較", "")
    into["channel_group.feature_match_tolerance_hz"] = scheme.channel_group.feature_match_tolerance_hz
    for receiver in scheme.receiver_set.points:
        root = f"receiver_set.points.{receiver.receiver_id}"
        into[root] = "有"
        labels[root] = (f"座位 {receiver.receiver_id}", "")
        for axis, value in zip(("x", "y", "z"), receiver.position_m, strict=True):
            into[f"{root}.position.{axis}"] = value
            labels[f"{root}.position.{axis}"] = (f"座位 {receiver.receiver_id} {axis}", "公尺")
        into[f"{root}.role"] = LABELS.get(receiver.role.value, receiver.role.value)
        into[f"{root}.importance"] = receiver.importance
        direction = receiver.direction_relative_to_primary
        into[f"{root}.direction"] = (DIRECTIONS[direction][1] if direction in DIRECTIONS
                                     else direction)
        labels[f"{root}.role"] = (f"座位 {receiver.receiver_id} 角色", "")
        labels[f"{root}.importance"] = (f"座位 {receiver.receiver_id} 重要度", "")
        labels[f"{root}.direction"] = (f"座位 {receiver.receiver_id} 相對方向", "")


def scheme_differences(a: Scheme, b: Scheme) -> tuple[SchemeChange, ...]:
    """按方案欄位逐一比較，與聲學評估無關。"""
    left: dict[str, object] = {}
    right: dict[str, object] = {}
    labels: dict[str, tuple[str, str]] = dict(_FIELDS)

    _collect_scheme(a, left, labels)
    _collect_scheme(b, right, labels)
    changes: list[SchemeChange] = []
    for path in sorted(left.keys() | right.keys()):
        if path not in left or path not in right:
            if path.startswith("receiver_set.points.") and path.count(".") > 2:
                continue
            if path.startswith("receiver_set.points."):
                a_text, b_text = (("只有 A 有", "無") if path in left
                                  else ("無", "只有 B 有"))
            else:
                unit = labels[path][1]
                a_text = _text(left[path], unit) if path in left else "未設定"
                b_text = _text(right[path], unit) if path in right else "未設定"
        elif left[path] == right[path]:
            continue
        else:
            unit = labels[path][1]
            aa, bb = left[path], right[path]
            if isinstance(aa, float | int) and isinstance(bb, float | int) and not isinstance(aa, bool):
                a_text, b_text = _number_pair(aa, bb, unit)
            else:
                a_text, b_text = _text(aa, unit), _text(bb, unit)
                if a_text == b_text:
                    a_text, b_text = repr(aa), repr(bb)
        changes.append(SchemeChange(path=path, label=labels[path][0],
                                    a_text=a_text, b_text=b_text))
    return tuple(changes)


def _changed_keys(changes: tuple[SchemeChange, ...]) -> tuple[str, ...]:
    keys: set[str] = set()
    for change in changes:
        parts = change.path.split(".")
        if len(parts) == 3 and parts[0] == "speakers" and parts[2] in {"x", "y", "z"}:
            keys.add(f"speaker:{parts[1]}")
        if (len(parts) >= 3 and parts[:2] == ["receiver_set", "points"]
                and (len(parts) == 3 or parts[3] in {"position", "role", "direction"})):
            keys.add(f"receiver:{parts[2]}")
    return tuple(sorted(keys))


def _overlay(view_a: ResultView, view_b: ResultView,
             seat_texts: dict[str, str]) -> tuple[Overlay, tuple[str, ...]]:
    responses = (("a", view_a.frequency_responses), ("b", view_b.frequency_responses))
    axis = tuple(sorted({point.frequency_hz for _, group in responses
                         for response in group for point in response.points}))
    series: list[OverlaySeries] = []
    defaults: list[str] = []
    notes: list[str] = []
    for side, group in responses:
        primary = [row for row in group if row.receiver_role == "primary"]
        left = [row for row in primary if row.role == "left"]
        chosen = (left or sorted(primary, key=lambda row: row.role))[0]
        if not left:
            notes.append(f"{side.upper()} 沒有左聲道，預設改用第一個聲道的主位")
        for row in group:
            key = f"{side}:{row.role}:{row.receiver_id}"
            values = {point.frequency_hz: point.level_db for point in row.points}
            series.append(OverlaySeries(
                key=key, side=side, role=row.role, speaker_id=row.speaker_id,
                receiver_id=row.receiver_id, receiver_role=row.receiver_role,
                legend_text=f"{side.upper()}・{LABELS.get(row.role, row.role)}・"
                            f"{LABELS.get(row.receiver_role, row.receiver_role)} {row.receiver_id}",
                levels_db=tuple(values.get(frequency) for frequency in axis)))
            if row is chosen:
                defaults.append(key)
    left_rows = view_a.frequency_responses
    right_rows = view_b.frequency_responses
    pairs: list[OverlayPair] = []
    for left_row in left_rows:
        if left_row.receiver_role == "primary":
            matches = [row for row in right_rows if row.role == left_row.role
                       and row.receiver_role == "primary"]
            label = f"{LABELS.get(left_row.role, left_row.role)}・主位"
        else:
            matches = [row for row in right_rows if row.role == left_row.role
                       and row.receiver_role != "primary"
                       and row.receiver_id == left_row.receiver_id]
            # 座位寫方向（主位前方）比寫角色（周圍點）說得出是哪一點；沒有方向的才寫角色。
            label = (f"{LABELS.get(left_row.role, left_row.role)}・{left_row.receiver_id}"
                     f"（{seat_texts.get(left_row.receiver_id, left_row.receiver_label)}）")
        if matches:
            right_row = matches[0]
            pairs.append(OverlayPair(label=label,
                                     a_key=f"a:{left_row.role}:{left_row.receiver_id}",
                                     b_key=f"b:{right_row.role}:{right_row.receiver_id}"))
    preferred_keys = (defaults[0], defaults[1])
    pairs.sort(key=lambda pair: (pair.a_key, pair.b_key) != preferred_keys)
    default_keys = ((pairs[0].a_key, pairs[0].b_key) if pairs else preferred_keys)
    return Overlay(frequency_hz=axis, series=tuple(series),
                   default_keys=default_keys, pairs=tuple(pairs)), tuple(notes)


def _table(a: SchemeResult, b: SchemeResult, quality_targets: QualityTargets,
           run_date: date) -> tuple[TableStatus, frozenset[str]]:
    """兩份一起排一次；只有兩份都在同一張表、都排得上、總代價又不同時才印名次與總代價。

    不同表時哪一張算主表是看比較身分排序，跟好壞無關；落在主表那份的「名次 1」只是一個人的名次，
    所以兩邊都寫「不可同表比較」，不寫一邊可排名、一邊不可（看起來像有一份壞了）。
    另外回傳身分不同的類別，分項表要標出那幾類的代價不能直接比。
    總代價相同時名次照代號排，也不代表好壞。這幾種都不印名次。
    """
    problems = comparison_problems((a, b))
    if problems:
        return TableStatus(same_table=False, a_text="不可同表", b_text="不可同表",
                           reason_text="；".join(problems), calibration_text=BASELINE_NOTE), frozenset()
    ranking = compare_results((a, b), quality_targets=quality_targets, run_date=run_date)
    names = {"A": a.scheme.scheme_id, "B": b.scheme.scheme_id}
    status = {side: LABELS.get(ranking.status_of(name).value, ranking.status_of(name).value)
              for side, name in names.items()}
    calibration = ranking.header.calibration_note or BASELINE_NOTE
    incompatible = {row.candidate_id: row for row in ranking.not_comparable.rows}
    if incompatible:
        side, other = ("A", "B") if names["A"] in incompatible else ("B", "A")
        groups = identity_difference_groups(ranking, incompatible[names[side]].identity)
        detail = "；".join(f"{label}：{'、'.join(LABELS.get(item.value, item.value) for item in items)}"
                          for label, items in groups)
        reason = (f"{side} 與 {other} 的比較身分不同" + (f"（{detail}）" if detail else "")
                  + "；兩份不在同一張表，不列名次與總代價")
        split = LABELS["not_comparable"]
        return TableStatus(same_table=False, a_text=split, b_text=split,
                           reason_text=reason, calibration_text=calibration), frozenset(
            item.value for _, items in groups for item in items)
    rows = {row.candidate_id: row for row in ranking.rankable}
    ranked_a, ranked_b = rows.get(names["A"]), rows.get(names["B"])
    if ranked_a is None or ranked_b is None:
        ranked = [side for side, row in (("A", ranked_a), ("B", ranked_b)) if row is not None]
        which = f"只有 {ranked[0]} 排得上" if ranked else "兩份都排不上"
        return TableStatus(same_table=True, a_text=status["A"], b_text=status["B"],
                           reason_text=f"同表，但{which}；不列名次與總代價",
                           calibration_text=calibration), frozenset()
    if ranked_a.total_cost == ranked_b.total_cost:
        return TableStatus(same_table=True, a_text=f"{status['A']}；總代價相同",
                           b_text=f"{status['B']}；總代價相同",
                           reason_text="同表；總代價相同，不分名次", calibration_text=calibration), frozenset()
    total_a, total_b = _number_pair(ranked_a.total_cost, ranked_b.total_cost, "")
    return TableStatus(same_table=True,
                       a_text=f"{status['A']}；名次 {ranked_a.rank}；總代價 {total_a}",
                       b_text=f"{status['B']}；名次 {ranked_b.rank}；總代價 {total_b}",
                       reason_text="同表", calibration_text=calibration), frozenset()


def build_compare_view(*, a_run_id: str, a: SchemeResult, view_a: ResultView,
                       b_run_id: str, b: SchemeResult, view_b: ResultView,
                       quality_targets: QualityTargets, run_date: date) -> CompareView:
    """將兩份結果頁資料組成可直接顯示的比較契約。"""
    changes = scheme_differences(a.scheme, b.scheme)
    fingerprints = tuple(FingerprintCheck(
        label=label, same=left == right,
        text="相同" if left == right else f"不同：{left[:7]}／{right[:7]}")
        for label, left, right in (
            ("座位組", a.scheme.receiver_set.fingerprint, b.scheme.receiver_set.fingerprint),
            ("座位相對佈局", a.scheme.receiver_set.layout_fingerprint,
             b.scheme.receiver_set.layout_fingerprint),
            ("聲道組", a.scheme.channel_group.fingerprint, b.scheme.channel_group.fingerprint)))
    def directions(result: SchemeResult) -> dict[str, str]:
        return {point.receiver_id: DIRECTIONS[point.direction_relative_to_primary][1]
                for point in result.scheme.receiver_set.points
                if point.direction_relative_to_primary in DIRECTIONS}
    a_seats, b_seats = directions(a), directions(b)
    # 同代號的座位兩邊方向不同（含一邊沒設方向）時兩個都寫，不只寫 A 的。
    seat_texts = {seat: a_seats[seat] if a_seats.get(seat) == b_seats.get(seat) else
                  f"A：{a_seats.get(seat, '沒設方向')}／B：{b_seats.get(seat, '沒設方向')}"
                  for seat in a_seats.keys() | b_seats.keys()}
    overlay, fallback_notes = _overlay(view_a, view_b, seat_texts)
    category_a = {row.category: row for row in view_a.categories}
    category_b = {row.category: row for row in view_b.categories}
    table, differing = _table(a, b, quality_targets, run_date)
    categories = tuple(CategoryRow(category=kind.value, label=LABELS[kind.value],
                                   a=category_a[kind.value], b=category_b[kind.value],
                                   comparison_text=("兩邊身分不同，代價不能直接比"
                                                    if kind.value in differing else ""))
                       for kind in QualityCategory)
    preview = "、".join(row.label for row in changes[:5]) or "無"
    extra = f"，另 {len(changes) - 5} 處" if len(changes) > 5 else ""
    summary = (f"改了 {len(changes)} 處：{preview}{extra}；"
               f"{'同表' if table.same_table else '不可同表'}；{LOW_FREQUENCY_DECAY_NOTE}")
    return CompareView(
        a=SideIdentity(run_id=a_run_id, scheme_id=a.scheme.scheme_id,
                       engine_text=a.engine_commit[:7], run_date=a.run_date.isoformat(),
                       total_text=view_a.timing_texts["total_s"]),
        b=SideIdentity(run_id=b_run_id, scheme_id=b.scheme.scheme_id,
                       engine_text=b.engine_commit[:7], run_date=b.run_date.isoformat(),
                       total_text=view_b.timing_texts["total_s"]),
        changes=changes, changed_keys=_changed_keys(changes),
        fingerprints=fingerprints, summary_text=summary,
        overlay=overlay, categories=categories, table=table,
        notes=(LOW_FREQUENCY_DECAY_NOTE, SPATIAL_IMPRESSION_NOTE,
               REVERBERATION_ROOM_NOTE, BASELINE_NOTE, *fallback_notes), labels=LABELS,
        level_note=LEVEL_NOTE)
