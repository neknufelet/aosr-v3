"""兩份已驗結果的比較頁顯示資料。"""
from __future__ import annotations

from aosr.gui.plan_objects import changed_furniture_keys
from aosr.reporting.display import LOW_FREQUENCY_DECAY_NOTE

import csv
import io
from collections import Counter
from datetime import date
from decimal import Decimal
from urllib.parse import unquote

import numpy as np

from aosr.config.quality_targets import QualityTargets
from aosr.gui.jobs import ResultStatus
from aosr.gui.labels import (
    DIRECTIONS, LOW_FREQUENCY_AXES, ROOM_LENGTHS, SOURCE_MODELS, SPEAKER_SETUP, WALLS, listening_point_label,
    speaker_label)
from aosr.reporting.calculation_fingerprint import short_fingerprint
from aosr.reporting.compare import compare_results, comparison_problems, identity_difference_groups
from aosr.reporting.display import (
    REVERBERATION_ROOM_NOTE, APPROXIMATE_TEXT, CONDITIONS_DIFFER_TEXT,
    FURNITURE_REVERBERATION_NOTE, FURNITURE_REVERBERATION_COMPARISON_NOTE, FURNITURE_COMPARISON_REASON, FURNITURE_FIELDS,
    FURNITURE_KINDS, FURNITURE_MATERIALS, FURNITURE_MODEL_NOTE, FURNITURE_REASON,
    FURNITURE_ROOM_ANGLE, NO_SCHEME_CHANGES_TEXT, OTHER_SETTINGS_LABEL,
    furniture_name, furniture_ranking_note, reflection_models_differ,
)
from aosr.reporting.result import SchemeResult
from aosr.reporting.furniture_display import furniture_material_lines
from aosr.gui.result_view import CategoryView, FrequencyResponse, LABELS, ResultView, ViewModel
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import QualityCategory
from aosr.scoring.reflections_cost import comparison_support
from aosr.scoring.ranking_models import NotComparableRow, RankableRow, RankingResult
from aosr.scoring.recommendation import RecommendationStatus, ReviewStatus


class SideIdentity(ViewModel):
    run_id: str
    scheme_id: str
    # 程式提交代號前 7 碼與物理身分前 12 碼：頁面只放在摺起來的技術細節，主畫面寫「物理：兩份相同／不同」。
    engine_text: str
    fingerprint_text: str
    run_date: str
    total_text: str
    status_text: str


class SchemeChange(ViewModel):
    path: str
    label: str
    a_text: str
    b_text: str


class FingerprintCheck(ViewModel):
    label: str
    same: bool
    # 頁面上只寫相同／不同（不同時附一句白話說哪一類改了）；兩份指紋的前 7 碼只進摘要 CSV。
    text: str
    code_text: str


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
    # 這一對屬於哪個聲道（聲道代號）、哪個位置（主位是 "primary"，其他點是 "seat:" 加座位代號）；
    # 頁面用聲道切換加位置按鈕挑一對，A、B 一起換。
    channel: str
    position: str


class OverlayChoice(ViewModel):
    key: str
    label: str


class Overlay(ViewModel):
    frequency_hz: tuple[float, ...]
    series: tuple[OverlaySeries, ...]
    default_keys: tuple[str, str]
    pairs: tuple[OverlayPair, ...]
    # 聲道切換與位置按鈕：每一對的聲道、位置都在這兩張表裡，表上每一格也至少有一對。
    channels: tuple[OverlayChoice, ...]
    positions: tuple[OverlayChoice, ...]


class CategoryRow(ViewModel):
    a_state_text: str = ""
    b_state_text: str = ""
    category: str
    label: str
    a: CategoryView
    b: CategoryView
    # 兩份不同表時，比較身分不同的那幾類寫「評分條件不同，這一類代價不能直接比」；其餘空字串。
    comparison_text: str
    # 這一類哪一份代價比較低（伺服器判）："a"、"b"、印出來一樣是 "same"；缺代價或不能直接比是空字串。
    better: str
    better_text: str
    # 說明欄：兩邊的說明與上面那句，重複的只留一次；兩份都尚未評估的類不再寫，摘要已經寫了。
    note_text: str


class TableStatus(ViewModel):
    same_table: bool
    a_text: str
    b_text: str
    # 摘要 CSV「總代價（越低越好）」那一列的格子：只放數字（不列時寫「不列」、排不上時寫狀態），
    # 列名已經寫了總代價，格子裡不再重複。
    a_cell: str
    b_cell: str
    # 兩份能不能直接比總代價，一句白話。
    reason_text: str
    calibration_text: str
    # 哪一份比較好、差多少；不列總代價時是空字串。better 同 CategoryRow。
    verdict_text: str
    better: str
    # 判勝負那句旁邊的但書：照排名列的複核狀態寫哪一份還有警戒沒確認完，並寫明還不是最終推薦；
    # 不列總代價時是空字串。
    review_text: str


class CompareView(ViewModel):
    no_changes_text: str = NO_SCHEME_CHANGES_TEXT
    overlay_note: str = ""
    ranking_approximation_text: str = ""
    not_modeled: tuple[str, ...] = ()
    manual_checks: tuple[str, ...] = ()
    a: SideIdentity
    b: SideIdentity
    # 「物理」：兩份的物理身分一樣寫「兩份相同」，不一樣寫「兩份不同」（指紋本身只進技術細節與 CSV）。
    version_text: str
    changes: tuple[SchemeChange, ...]
    changed_keys: tuple[str, ...]
    fingerprints: tuple[FingerprintCheck, ...]
    # 三項都相同時核對卡只寫這一句（不畫三列「相同」的表）；有一項不同就是空字串，頁面畫表。
    fingerprints_text: str
    summary_text: str
    # 兩份都尚未評估的類（低頻拖尾等），摘要只寫這一次；沒有就是空字串。
    pending_text: str
    overlay: Overlay
    categories: tuple[CategoryRow, ...]
    table: TableStatus
    notes: tuple[str, ...]
    labels: dict[str, str]
    # 頻響疊圖的音量基準但書：頁面、匯出圖片、摘要 CSV 共用這一句，匯出去的東西也帶著它。
    level_note: str
    run_notices: tuple[str, ...]


LEVEL_NOTE = ("兩邊 dB 用同一個基準（單位振幅點源、距離 1 公尺），沒有各自對齊音量；"
              "不是校準過的絕對聲壓級。")
COMPARABLE_TEXT = "兩份可以直接比較（同一套評分設定）"
NO_TOTAL_TEXT = "不列總代價"
NO_TOTAL_CELL = "不列"
# 評分登記簿的校準狀態（排名表頭的 calibration）對應的白話；登記簿還是暫定基線時，分數只能看相對好壞。
CALIBRATION_TEXTS = {
    "baseline": "評分尺度還沒正式校準：代價只看得出哪一份相對比較好、差多少，不代表合格或不合格",
    "calibrated": "評分尺度已正式校準；這裡只比兩份哪一份比較好，不是合格判定",
}
# 不列總代價時（不同表、有一份排不上）不判哪一份比較好，校準那句只講各類代價。
CATEGORY_CALIBRATION_TEXTS = {
    "baseline": "評分尺度還沒正式校準：各類代價只看得出相對高低，不代表合格或不合格",
    "calibrated": "評分尺度已正式校準；這裡只逐類並列兩份的代價，不是合格判定",
}
# 指紋不同時，說明是哪一類設定改了（白話），不印雜湊。
_FINGERPRINT_WORDS = {
    "座位組": "座位的位置或設定有改",
    "座位相對佈局": "周圍點相對主位的擺法或設定有改",
    "聲道組": "聲道接哪個喇叭或聲道比較方式有改",
}
# 不能同表時，排名層分的三組原因換成白話（{side} 是不在主表的那一份）。
_GROUP_WORDS = {"少了": "{side} 沒評估", "多了": "只有 {side} 評估", "同一類但身分不同": "兩份都有但條件不同"}


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
    if view.run_notices:
        rows.extend([[], ["說明"], ["計算狀態", f"A：{view.a.status_text}", f"B：{view.b.status_text}"]])
        rows.extend([notice] for notice in view.run_notices)
    if view.overlay_note:
        rows.extend([[], [view.overlay_note]])
    return _csv(rows)


CATEGORY_HEADINGS = ("類別", "A 狀態", "A 代價", "B 狀態", "B 代價", "哪一份較好", "說明")


def category_cells(row: CategoryRow) -> list[str | float | None]:
    """分項表一列：頁面與摘要 CSV 同一套欄位（CATEGORY_HEADINGS）。"""
    return [row.label, row.a_state_text or row.a.state_label, row.a.cost_text,
            row.b_state_text or row.b.state_label, row.b.cost_text,
            row.better_text, row.note_text]


def summary_csv(view: CompareView) -> str:
    """依比較頁已排好的文字輸出；每一段的表頭說清楚每一欄是什麼，不把說明塞進「A」那一欄。

    段落：兩邊身分與總代價（欄位／A／B；總代價那一列格子只放數字，列名寫一次「總代價」）、
    整體說明（欄位／內容）、改了哪裡（項目／A／B）、指紋（指紋／核對／前 7 碼，前 7 碼只在這裡）、
    各類結果（跟頁面同一張表）、固定說明。段與段之間空一列。
    """
    rows: list[list[str | float | None]] = [
        ["欄位", "A", "B"],
        ["方案代號", view.a.scheme_id, view.b.scheme_id],
        ["計算日期", view.a.run_date, view.b.run_date],
        ["計算時間（全程）", view.a.total_text, view.b.total_text],
        ["計算狀態", view.a.status_text, view.b.status_text],
        ["總代價（越低越好）", view.table.a_cell, view.table.b_cell],
        ["物理身分前 12 碼", view.a.fingerprint_text, view.b.fingerprint_text],
        ["程式提交代號", view.a.engine_text, view.b.engine_text],
        [],
        ["欄位", "內容"],
        ["物理", view.version_text],
        ["摘要句", view.summary_text],
        ["能不能直接比", view.table.reason_text],
        ["哪一份比較好", view.table.verdict_text],
        ["複核與推薦", view.table.review_text],
        ["尚未評估", view.pending_text],
        ["校準說明", view.table.calibration_text],
        ["音量基準", view.level_note],
        [],
        ["項目", "A", "B"],
    ]
    rows.extend([change.label, change.a_text, change.b_text] for change in view.changes)
    rows.extend([[], ["指紋", "核對", "指紋前 7 碼（A／B）"]])
    rows.extend([check.label, check.text, check.code_text] for check in view.fingerprints)
    rows.extend([[], list(CATEGORY_HEADINGS)])
    rows.extend(category_cells(category) for category in view.categories)
    rows.extend([[], ["說明"]])
    rows.extend([note] for note in view.notes)
    rows.extend([notice] for notice in view.run_notices)
    if view.overlay_note:
        rows.append([view.overlay_note])
    if view.ranking_approximation_text:
        rows.append([view.ranking_approximation_text])
    return _csv(rows)


# 房間長寬高與方案用途跟方案輸入頁表單（與檢查不過的訊息）同一套字；長寬高的欄名已經寫了（公尺），
# A、B 兩格只放數字，不再重複單位。
_FIELDS = {
    **{f"scene.room_m.{axis}": (name, "") for axis, name in ROOM_LENGTHS.items()},
    "scene.sound_speed_m_s": ("聲速", "公尺／秒"),
    "scene.density_kg_m3": ("密度", "公斤／立方公尺"),
    "scene.reflection_order_k": ("反射階數", "階"),
    "scene.low_frequency_axis": ("低頻軸", ""),
    "source_model": ("聲源模型", ""),
    "purpose": ("方案用途", ""),
    "channel_group.feature_match_tolerance_hz": ("峰谷配對容差", "Hz"),
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


def _speaker_name(scheme: Scheme, speaker_id: str) -> str:
    """喇叭顯示名：先找它接哪個聲道，用顯示名稱表（網頁的顯示名稱也查這一張）的名字；表上沒有就寫代號。"""
    roles = [channel.role for channel in scheme.channel_group.channels
             if channel.speaker_id == speaker_id]
    for code in (*roles, speaker_id):
        if speaker_label(code) != code:
            return speaker_label(code)
    return f"喇叭 {speaker_id}"


def _seat_name(receiver_id: str, role: str) -> str:
    """座位顯示名：顯示名稱表有這個代號就用表上的名字；沒有的主位叫主位，其他寫座位加代號。"""
    name = listening_point_label(receiver_id)
    if name != receiver_id:
        return name
    return LABELS["primary"] if role == "primary" else f"座位 {receiver_id}"


def _path_id(code: str) -> str:
    """代號可含點；拼進欄位路徑前先換掉點（與 %），路徑照點切才切得對，取回代號用 unquote。"""
    return code.replace("%", "%25").replace(".", "%2E")


def _collect_scheme(scheme: Scheme, into: dict[str, object],
                    labels: dict[str, tuple[str, str]], split_angles: frozenset[str] = frozenset()) -> None:
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
            labels[path] = (f"{WALLS.get(wall, wall)}{title}", unit)
    into["source_model"] = SOURCE_MODELS.get(scheme.source_model, scheme.source_model)
    into["purpose"] = LABELS.get(scheme.purpose, scheme.purpose)
    for speaker_id, point in scheme.speakers.items():
        for axis in ("x", "y", "z"):
            path = f"speakers.{_path_id(speaker_id)}.{axis}"
            into[path] = getattr(point, axis)
            labels[path] = (f"{_speaker_name(scheme, speaker_id)} {axis} 座標", "公尺")
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
        root = f"receiver_set.points.{_path_id(receiver.receiver_id)}"
        seat = _seat_name(receiver.receiver_id, receiver.role.value)
        into[root] = "有"
        labels[root] = (seat, "")
        for axis, value in zip(("x", "y", "z"), receiver.position_m, strict=True):
            into[f"{root}.position.{axis}"] = value
            labels[f"{root}.position.{axis}"] = (f"{seat} {axis} 座標", "公尺")
        into[f"{root}.role"] = LABELS.get(receiver.role.value, receiver.role.value)
        into[f"{root}.importance"] = receiver.importance
        direction = receiver.direction_relative_to_primary
        into[f"{root}.direction"] = (DIRECTIONS[direction][1] if direction in DIRECTIONS
                                     else direction)
        labels[f"{root}.role"] = (f"{seat} 角色", "")
        labels[f"{root}.importance"] = (f"{seat} 重要度", "")
        labels[f"{root}.direction"] = (f"{seat} 相對方向", "")

    _collect_furniture(scheme, into, labels, split_angles)
    _collect_speaker_setup(scheme, into, labels)


def _collect_speaker_setup(scheme: Scheme, into: dict[str, object], labels: dict[str, tuple[str, str]]) -> None:
    """喇叭設定逐格列出；另一邊沒設時每格對「未設定」，不落入其他設定。"""
    setup = scheme.speaker_setup
    if setup is None:
        return
    for field in ("kind", "mount"):
        path = f"speaker_setup.{field}"
        into[path] = SPEAKER_SETUP[getattr(setup, field)]
        labels[path] = (SPEAKER_SETUP[field], "")
    path = "speaker_setup.representative"
    into[path] = SPEAKER_SETUP["representative_model" if setup.representative else "actual_model"]
    labels[path] = (SPEAKER_SETUP["representative"], "")
    for field, value in setup.cabinet.model_dump().items():
        path = f"speaker_setup.cabinet.{field}"
        into[path], labels[path] = value, (SPEAKER_SETUP[field], "公尺")


def _collect_furniture(scheme: Scheme, into: dict[str, object],
                       labels: dict[str, tuple[str, str]], split_angles: frozenset[str] = frozenset()) -> None:
    for item in scheme.furniture or ():
        # 代號可含點；保留欄位路徑的分隔符，避免把新增家具誤當子欄位略過。
        root = f"furniture.{_path_id(item.furniture_id)}"
        name = furniture_name(item.kind, item.furniture_id)
        into[root], labels[root] = "有", (name, "")
        values = item.model_dump(mode="json", exclude={"furniture_id", "placement"})
        values["kind"] = FURNITURE_KINDS[item.kind]
        values["material"] = FURNITURE_MATERIALS[item.material]
        for field, value in values.items():
            title, unit = FURNITURE_FIELDS[field]
            into[f"{root}.{field}"] = value
            labels[f"{root}.{field}"] = (f"{name} {title}", unit)
        for field, value in item.placement.model_dump(mode="python").items():
            title, unit = FURNITURE_FIELDS[field]
            if field == "yaw_deg":
                room_angle = "bottom_center_m" in type(item.placement).model_fields
                if room_angle:
                    title = FURNITURE_ROOM_ANGLE
                if item.furniture_id in split_angles:
                    field = "room_yaw_deg" if room_angle else "relative_yaw_deg"
            into[f"{root}.placement.{field}"] = value
            labels[f"{root}.placement.{field}"] = (f"{name} {title}", unit)


def scheme_differences(a: Scheme, b: Scheme) -> tuple[SchemeChange, ...]:
    """按方案欄位逐一比較，與聲學評估無關。"""
    left: dict[str, object] = {}
    right: dict[str, object] = {}
    labels: dict[str, tuple[str, str]] = dict(_FIELDS)

    # 換擺法時 yaw_deg 的基準也換了，兩種角度分開列，各自對「未設定」。
    split_angles = frozenset(item.furniture_id for item in a.furniture or () for other in b.furniture or ()
                             if item.furniture_id == other.furniture_id and type(item.placement) is not type(other.placement))
    _collect_scheme(a, left, labels, split_angles)
    _collect_scheme(b, right, labels, split_angles)
    changes: list[SchemeChange] = []
    for path in sorted(left.keys() | right.keys()):
        if path not in left or path not in right:
            root = ".".join(path.split(".")[:2])
            if path.startswith("furniture.") and (root not in left or root not in right):
                if path.count(".") > 1:
                    continue
                a_text, b_text = (("只有 A 有", "無") if path in left else ("無", "只有 B 有"))
            elif path.startswith("receiver_set.points.") and path.count(".") > 2:
                continue
            elif path.startswith("receiver_set.points."):
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
    if not changes and a.model_dump(exclude={"scheme_id"}) != b.model_dump(exclude={"scheme_id"}):
        changes.append(SchemeChange(path="other_settings", label=OTHER_SETTINGS_LABEL,
                                    a_text="—", b_text="—"))
    return tuple(changes)


def _changed_keys(changes: tuple[SchemeChange, ...]) -> tuple[str, ...]:
    keys: set[str] = set()
    for change in changes:
        parts = change.path.split(".")
        if len(parts) == 3 and parts[0] == "speakers" and parts[2] in {"x", "y", "z"}:
            keys.add(f"speaker:{unquote(parts[1])}")
        if (len(parts) >= 3 and parts[:2] == ["receiver_set", "points"]
                and (len(parts) == 3 or parts[3] in {"position", "role", "direction"})):
            keys.add(f"receiver:{unquote(parts[2])}")
    return tuple(sorted(keys))


def _unique_names(names: dict[str, tuple[str, str]]) -> dict[str, str]:
    """鍵對到（名字, 代號）：同名的點補上代號，其餘照原名；兩個點叫同一個名字就分不出是哪一點。"""
    counts = Counter(name for name, _ in names.values())
    return {key: f"{name}（{code}）" if counts[name] > 1 else name
            for key, (name, code) in names.items()}


def _directions(result: SchemeResult) -> dict[str, str]:
    """座位代號對到相對主位方向的全名（主位前方…）；沒設方向的不列。"""
    return {point.receiver_id: DIRECTIONS[point.direction_relative_to_primary][1]
            for point in result.scheme.receiver_set.points
            if point.direction_relative_to_primary in DIRECTIONS}


def _side_seat_names(result: SchemeResult) -> dict[str, str]:
    """一份方案裡每個座位的顯示名（圖例用）：主位叫主位，周圍點照這一份自己的方向，沒有方向的查顯示名稱表。"""
    directions = _directions(result)
    return _unique_names({point.receiver_id: (
        LABELS["primary"] if point.role.value == "primary"
        else directions.get(point.receiver_id) or _seat_name(point.receiver_id, point.role.value),
        point.receiver_id) for point in result.scheme.receiver_set.points})


def _position_name(receiver_id: str, role: str, a_directions: dict[str, str],
                    b_directions: dict[str, str]) -> str:
    """位置按鈕的名字：兩份方向一樣寫方向全名；不一樣（含一邊沒設）兩個都寫並附座位代號，
    只寫 A 的會讓人以為兩點同一處；兩份都沒設方向就查顯示名稱表。"""
    left, right = a_directions.get(receiver_id), b_directions.get(receiver_id)
    if left is None and right is None:
        return _seat_name(receiver_id, role)
    if left == right:
        return str(left)
    return f"座位 {receiver_id}（A：{left or '沒設方向'}／B：{right or '沒設方向'}）"


def _series(responses: tuple[tuple[str, tuple[FrequencyResponse, ...]], ...],
            axis: tuple[float, ...], seat_names: dict[str, dict[str, str]]
            ) -> tuple[list[OverlaySeries], list[str], list[str]]:
    """每一條線與每一份的預設線（左聲道主位）；圖例寫哪一份・哪個聲道・哪個位置，位置用那一份自己的座位名。"""
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
            seat = (seat_names[side].get(row.receiver_id)
                    or _seat_name(row.receiver_id, row.receiver_role))
            series.append(OverlaySeries(
                key=key, side=side, role=row.role, speaker_id=row.speaker_id,
                receiver_id=row.receiver_id, receiver_role=row.receiver_role,
                legend_text=f"{side.upper()}・{LABELS.get(row.role, row.role)}・{seat}",
                levels_db=tuple(values.get(frequency) for frequency in axis)))
            if row is chosen:
                defaults.append(key)
    return series, defaults, notes


def _pairs(left_rows: tuple[FrequencyResponse, ...], right_rows: tuple[FrequencyResponse, ...],
           directions: tuple[dict[str, str], dict[str, str]]
           ) -> tuple[list[OverlayPair], dict[str, str]]:
    """A 的每一條線配 B 同聲道、同位置的那一條：主位看角色（兩份主位代號可以不同），其他點看座位代號。
    同一個聲道與位置只留第一對，按鈕挑得到的就是全部。另回傳每個位置的按鈕名字。"""
    found: dict[tuple[str, str], tuple[FrequencyResponse, FrequencyResponse]] = {}
    names: dict[str, tuple[str, str]] = {}
    for left_row in left_rows:
        primary = left_row.receiver_role == "primary"
        matches = [row for row in right_rows if row.role == left_row.role
                   and (row.receiver_role == "primary") == primary
                   and (primary or row.receiver_id == left_row.receiver_id)]
        position = "primary" if primary else f"seat:{left_row.receiver_id}"
        if not matches or (left_row.role, position) in found:
            continue
        names[position] = (LABELS["primary"] if primary else _position_name(
            left_row.receiver_id, left_row.receiver_role, *directions), left_row.receiver_id)
        found[(left_row.role, position)] = (left_row, matches[0])
    unique = _unique_names(names)
    return [OverlayPair(label=f"{LABELS.get(channel, channel)}・{unique[position]}",
                        a_key=f"a:{left.role}:{left.receiver_id}",
                        b_key=f"b:{right.role}:{right.receiver_id}",
                        channel=channel, position=position)
            for (channel, position), (left, right) in found.items()], unique


def _overlay(view_a: ResultView, view_b: ResultView, seat_names: dict[str, dict[str, str]],
             directions: tuple[dict[str, str], dict[str, str]]) -> tuple[Overlay, tuple[str, ...]]:
    responses = (("a", view_a.frequency_responses), ("b", view_b.frequency_responses))
    axis = tuple(sorted({point.frequency_hz for _, group in responses
                         for response in group for point in response.points}))
    series, defaults, notes = _series(responses, axis, seat_names)
    pairs, position_names = _pairs(view_a.frequency_responses, view_b.frequency_responses,
                                   directions)
    preferred_keys = (defaults[0], defaults[1])
    pairs.sort(key=lambda pair: (pair.a_key, pair.b_key) != preferred_keys)
    default_keys = ((pairs[0].a_key, pairs[0].b_key) if pairs else preferred_keys)
    # 聲道照配對順序（預設那一對的聲道在前）；位置主位在前，其他照 A 的順序。
    channels = tuple(OverlayChoice(key=channel, label=LABELS.get(channel, channel))
                     for channel in dict.fromkeys(pair.channel for pair in pairs))
    positions = tuple(OverlayChoice(key=key, label=position_names[key])
                      for key in sorted(position_names, key=lambda key: key != "primary"))
    return Overlay(frequency_hz=axis, series=tuple(series), default_keys=default_keys,
                   pairs=tuple(pairs), channels=channels, positions=positions), tuple(notes)


def _review_text(row_a: RankableRow, row_b: RankableRow) -> str:
    """判勝負那句旁邊的但書（決策紙：名次、複核狀態、推薦狀態分開記，讀的人不能把「比較好」當最終推薦）。

    哪一份還有複核警戒照排名列的 review_status，括號寫警戒在哪幾類；沒有警戒只能說「目前沒有產生」，
    不能說全部通過。推薦狀態今天只有「非最終」，照 recommendation_status 寫還不能當最終推薦。"""
    rows = {"A": row_a, "B": row_b}
    # 警戒在哪幾類照分項表的類別順序寫。
    where = {side: "、".join(LABELS[kind.value] for kind in QualityCategory
                             if kind in {alert.category for alert in row.review_alerts})
             for side, row in rows.items()}
    pending = [side for side, row in rows.items() if row.review_status is ReviewStatus.PENDING]
    if len(pending) == len(rows):
        head = "兩份都還有複核警戒沒確認完" + (
            f"（都在{where['A']}）" if where["A"] == where["B"]
            else f"（A：{where['A']}；B：{where['B']}）")
    elif pending:
        other = next(side for side in rows if side not in pending)
        head = (f"{pending[0]} 還有複核警戒沒確認完（{where[pending[0]]}），"
                f"{other} 目前沒有產生複核警戒")
    else:
        head = "兩份目前都沒有產生複核警戒"
    not_final = all(row.recommendation_status is RecommendationStatus.NOT_FINAL for row in rows.values())
    return head + ("；兩份都還不能當最終推薦" if not_final else "")


def _ranked_status(row_a: RankableRow, row_b: RankableRow, calibration: str) -> TableStatus:
    """兩份同表、都排得上：各印總代價並註明越低越好，伺服器判哪一份比較好、低多少。
    差距是框裡印出來的兩個數相減（讀的人自己減得出同一個數）；只差在浮點尾巴（微小差異）時不硬分高下。"""
    total_a, total_b = row_a.total_cost, row_b.total_cost
    review = _review_text(row_a, row_b)
    if total_a == total_b:
        shown = _plain(total_a, 4)
        text = f"總代價 {shown}（越低越好）"
        return TableStatus(same_table=True, a_text=text, b_text=text, a_cell=shown, b_cell=shown,
                           reason_text=COMPARABLE_TEXT,
                           calibration_text=calibration, better="same",
                           verdict_text="兩份總代價相同，分不出哪一份比較好", review_text=review)
    shown_a, shown_b = _number_pair(total_a, total_b, "")
    tiny = "（微小差異）" in shown_a + shown_b
    winner, loser = ("A", "B") if total_a < total_b else ("B", "A")
    gap = format(abs(Decimal(shown_a) - Decimal(shown_b)), "f") if not tiny else ""
    verdict = ("兩份總代價只差在很後面的小數位，可以當作相同" if tiny else
               f"{winner} 比較好：總代價比 {loser} 低 {gap}")
    return TableStatus(same_table=True, a_text=f"總代價 {shown_a}（越低越好）",
                       b_text=f"總代價 {shown_b}（越低越好）", a_cell=shown_a, b_cell=shown_b,
                       reason_text=COMPARABLE_TEXT,
                       calibration_text=calibration, verdict_text=verdict,
                       better="same" if tiny else winner.lower(), review_text=review)


def _split_status(ranking: RankingResult, names: dict[str, str],
                  incompatible: dict[str, NotComparableRow], calibration: str
                  ) -> tuple[TableStatus, frozenset[str]]:
    """不同表：哪一張算主表只看比較身分排序、跟好壞無關，落在主表那份的「名次 1」只是一個人的名次，
    所以兩邊寫一樣的字、不印總代價，也不判哪一份比較好。另回傳條件不同的類，分項表標出那幾類不能直接比。"""
    side = "A" if names["A"] in incompatible else "B"
    groups = identity_difference_groups(ranking, incompatible[names[side]].identity)
    detail = "；".join(
        f"{_GROUP_WORDS.get(label, label).format(side=side)}："
        f"{'、'.join(LABELS.get(item.value, item.value) for item in items)}"
        for label, items in groups)
    reason = ("兩份不能直接比總代價：有幾類的評分條件不一樣" + (f"（{detail}）" if detail else "")
              + "；分項表裡其他類仍可逐類比較")
    return TableStatus(same_table=False, a_text=NO_TOTAL_TEXT, b_text=NO_TOTAL_TEXT,
                       a_cell=NO_TOTAL_CELL, b_cell=NO_TOTAL_CELL,
                       reason_text=reason, calibration_text=calibration, verdict_text="",
                       better="", review_text=""), frozenset(
                           item.value for _, items in groups for item in items)


def _table(a: SchemeResult, b: SchemeResult, quality_targets: QualityTargets,
           run_date: date) -> tuple[TableStatus, frozenset[str] | None]:
    """兩份一起排一次；只有兩份都在同一張表、都排得上時才印總代價並判哪一份比較好。

    第二格是分項表不能直接比的類；None 表示一類都不能比（兩份的固定身分就對不上，排名層不收）。
    """
    problems = comparison_problems((a, b))
    if problems:
        return TableStatus(same_table=False, a_text=NO_TOTAL_TEXT, b_text=NO_TOTAL_TEXT,
                           a_cell=NO_TOTAL_CELL, b_cell=NO_TOTAL_CELL,
                           reason_text="兩份不能直接比較：" + "；".join(problems),
                           calibration_text=CATEGORY_CALIBRATION_TEXTS["baseline"], verdict_text="",
                           better="", review_text=""), None
    ranking = compare_results((a, b), quality_targets=quality_targets, run_date=run_date)
    names = {"A": a.scheme.scheme_id, "B": b.scheme.scheme_id}
    calibration = CATEGORY_CALIBRATION_TEXTS[ranking.header.calibration]
    incompatible = {row.candidate_id: row for row in ranking.not_comparable.rows}
    if incompatible:
        return _split_status(ranking, names, incompatible, calibration)
    rows = {row.candidate_id: row for row in ranking.rankable}
    ranked_a, ranked_b = rows.get(names["A"]), rows.get(names["B"])
    if ranked_a is None or ranked_b is None:
        status = {side: LABELS.get(ranking.status_of(name).value, ranking.status_of(name).value)
                  for side, name in names.items()}
        ranked = [side for side, row in (("A", ranked_a), ("B", ranked_b)) if row is not None]
        which = f"只有 {ranked[0]} 排得上" if ranked else "兩份都排不上"
        return TableStatus(same_table=True, a_text=status["A"], b_text=status["B"],
                           a_cell=status["A"], b_cell=status["B"],
                           reason_text=f"兩份用同一套評分設定，但{which}，{NO_TOTAL_TEXT}",
                           calibration_text=calibration, verdict_text="", better="",
                           review_text=""), frozenset()
    return (_ranked_status(ranked_a, ranked_b, CALIBRATION_TEXTS[ranking.header.calibration]),
            frozenset())


def _category_rows(view_a: ResultView, view_b: ResultView, differing: frozenset[str] | None,
                   furniture_difference: bool = False,
                   ) -> tuple[tuple[CategoryRow, ...], list[str]]:
    """分項並列：每一類判哪一份代價低（越低越好）。照畫面上印出來的代價比，印出來一樣就說相同，
    不拿看不到的小數位分高下；缺代價、條件不同或兩份根本不能比（differing 是 None）就不判。
    兩份都尚未評估的類另外回傳，摘要寫一次，每一列不再重複。"""
    category_a = {row.category: row for row in view_a.categories}
    category_b = {row.category: row for row in view_b.categories}
    pending = [kind.value for kind in QualityCategory
               if category_a[kind.value].state == category_b[kind.value].state == "not_evaluated"]
    rows: list[CategoryRow] = []
    for kind in QualityCategory:
        a, b = category_a[kind.value], category_b[kind.value]
        comparison = (f"{CONDITIONS_DIFFER_TEXT}，這一類代價不能直接比"
                      if differing is not None and kind.value in differing else "")
        if comparison and kind is QualityCategory.REFLECTIONS_AND_ECHO and furniture_difference:
            comparison += f"；{FURNITURE_COMPARISON_REASON}"
        better = better_text = ""
        if differing is not None and not comparison and a.cost is not None and b.cost is not None:
            better, better_text = (("same", "相同") if a.cost_text == b.cost_text
                                   else ("a", "A 較好") if a.cost < b.cost else ("b", "B 較好"))
        notes: tuple[str, ...] = (comparison,) if kind.value in pending else (a.note, b.note, comparison)
        if kind is QualityCategory.REVERBERATION and bool(view_a.furniture_reason) != bool(view_b.furniture_reason):
            side = "A" if view_a.furniture_reason else "B"
            notes = (f"{side}：{FURNITURE_REVERBERATION_NOTE}", FURNITURE_REVERBERATION_COMPARISON_NOTE, comparison)
        rows.append(CategoryRow(category=kind.value, label=LABELS[kind.value], a=a, b=b,
                                a_state_text=a.state_label + (f"；{APPROXIMATE_TEXT}" if "furniture_model_approximate" in a.flags else ""),
                                b_state_text=b.state_label + (f"；{APPROXIMATE_TEXT}" if "furniture_model_approximate" in b.flags else ""),
                                comparison_text=comparison, better=better, better_text=better_text,
                                note_text="；".join(dict.fromkeys(note for note in notes if note))))
    return tuple(rows), pending


def _pending_text(pending: list[str], totals_shown: bool) -> str:
    """兩份都尚未評估的類（低頻拖尾等）；有列總代價時註明它們不算在裡面。"""
    if not pending:
        return ""
    parts = [LOW_FREQUENCY_DECAY_NOTE] if "low_frequency_decay" in pending else []
    names = "、".join(LABELS[code] for code in pending if code != "low_frequency_decay")
    if names:
        parts.append(f"{names}：兩份都尚未評估" + ("，不算進總代價" if totals_shown else ""))
    return "；".join(parts)


def _summary(changes: tuple[SchemeChange, ...]) -> str:
    if not changes:
        return NO_SCHEME_CHANGES_TEXT
    preview = "、".join(row.label for row in changes[:5])
    extra = f"，另 {len(changes) - 5} 處" if len(changes) > 5 else ""
    return f"改了 {len(changes)} 處：{preview}{extra}"


def _fingerprints(a: SchemeResult, b: SchemeResult) -> tuple[FingerprintCheck, ...]:
    """座位與聲道設定核對：頁面寫相同或「不同（哪一類改了）」；兩份指紋前 7 碼只給摘要 CSV。"""
    return tuple(FingerprintCheck(
        label=label, same=left == right,
        text="相同" if left == right else f"不同（{_FINGERPRINT_WORDS[label]}）",
        code_text=f"{left[:7]}／{right[:7]}")
        for label, left, right in (
            ("座位組", a.scheme.receiver_set.fingerprint, b.scheme.receiver_set.fingerprint),
            ("座位相對佈局", a.scheme.receiver_set.layout_fingerprint,
             b.scheme.receiver_set.layout_fingerprint),
            ("聲道組", a.scheme.channel_group.fingerprint, b.scheme.channel_group.fingerprint)))


def _side(run_id: str, result: SchemeResult, view: ResultView, status: ResultStatus) -> SideIdentity:
    return SideIdentity(run_id=run_id, scheme_id=result.scheme.scheme_id,
                        engine_text=result.engine_commit[:7],
                        fingerprint_text=short_fingerprint(result.physics_identity),
                        run_date=result.run_date.isoformat(),
                        total_text=view.timing_texts["total_s"], status_text=status.status_text)


def compare_run_notices(a_status: ResultStatus, b_status: ResultStatus) -> tuple[str, ...]:
    """比較成功、不能比或被拒收都用同一組診斷警語。"""
    return tuple(f"{side}：{status.notice}" for side, status in (("A", a_status), ("B", b_status))
                 if not status.finished)


def _overlay_note(a: Scheme, b: Scheme) -> str:
    """第 13 條第 77 行接主詞；匯出與網頁一律讀 overlay_note。"""
    if a.furniture and b.furniture:
        return FURNITURE_MODEL_NOTE
    if a.furniture:
        return f"A：{FURNITURE_MODEL_NOTE}"
    return f"B：{FURNITURE_MODEL_NOTE}" if b.furniture else ""


def _setup_notes(a: SchemeResult, view_a: ResultView, b: SchemeResult, view_b: ResultView) -> tuple[str, ...]:
    if not any(result.scheme.furniture or result.scheme.speaker_setup for result in (a, b)):
        return ()
    lines = []
    for side, result, view in (("A", a, view_a), ("B", b, view_b)):
        parts = (*furniture_material_lines(result), *((view.speaker_setup_line,) if view.speaker_setup_line else ()))
        has_setup = bool(result.scheme.furniture or result.scheme.speaker_setup)
        lines.append(f"{side}：" + ("；".join(parts) if has_setup else "無家具或喇叭設定"))
    return tuple(lines)


def build_compare_view(*, a_run_id: str, a: SchemeResult, view_a: ResultView,
                       b_run_id: str, b: SchemeResult, view_b: ResultView,
                       quality_targets: QualityTargets, run_date: date,
                       a_status: ResultStatus = ResultStatus(),
                       b_status: ResultStatus = ResultStatus()) -> CompareView:
    """將兩份結果頁資料組成可直接顯示的比較契約。"""
    changes = scheme_differences(a.scheme, b.scheme)
    overlay, fallback_notes = _overlay(
        view_a, view_b, {"a": _side_seat_names(a), "b": _side_seat_names(b)},
        (_directions(a), _directions(b)))
    table, differing = _table(a, b, quality_targets, run_date)
    supports = tuple(comparison_support(next(item for item in result.candidate.evaluations
                     if item.category is QualityCategory.REFLECTIONS_AND_ECHO)) for result in (a, b))
    categories, pending = _category_rows(view_a, view_b, differing, reflection_models_differ(*supports))
    pending_text = _pending_text(pending, bool(table.verdict_text))
    notices = compare_run_notices(a_status, b_status)
    if notices:
        unfinished = "；".join(f"{side}：{status.label}" for side, status in
                              (("A", a_status), ("B", b_status)) if not status.finished)
        subject = "兩份計算都" if not a_status.finished and not b_status.finished else "有一份計算"
        table = table.model_copy(update={
            "better": "", "verdict_text":
            f"{subject}沒有正常完成（{unfinished}），不下哪一份比較好的結論"})
        categories = tuple(row.model_copy(update={"better": "", "better_text": ""})
                           for row in categories)
    # 校準那句在摘要（table.calibration_text），兩份都尚未評估的類也在摘要（pending_text），說明區不再重複。
    fingerprints = _fingerprints(a, b)
    return CompareView(
        overlay_note=_overlay_note(a.scheme, b.scheme),
        ranking_approximation_text=furniture_ranking_note(tuple(row.label for row in categories
            if "furniture_model_approximate" in (*row.a.flags, *row.b.flags)))
            if a.scheme.furniture and b.scheme.furniture and table.same_table and table.better else "",
        not_modeled=tuple(dict.fromkeys((*view_a.not_modeled, *view_b.not_modeled))),
        manual_checks=tuple(dict.fromkeys((*view_a.manual_checks, *view_b.manual_checks))),
        a=_side(a_run_id, a, view_a, a_status), b=_side(b_run_id, b, view_b, b_status),
        version_text=("兩份相同" if a.physics_identity == b.physics_identity
                      else "兩份不同"),
        changes=changes, changed_keys=(*_changed_keys(changes), *changed_furniture_keys(a.scheme, b.scheme)), fingerprints=fingerprints,
        fingerprints_text=("、".join(check.label for check in fingerprints) + "：兩份都相同"
                           if all(check.same for check in fingerprints) else ""),
        summary_text=_summary(changes),
        pending_text=pending_text,
        overlay=overlay, categories=categories, table=table,
        notes=(REVERBERATION_ROOM_NOTE, *fallback_notes, *_setup_notes(a, view_a, b, view_b),
               *((FURNITURE_REASON,) if a.scheme.furniture or b.scheme.furniture else ())), labels=LABELS,
        level_note=LEVEL_NOTE, run_notices=notices)
