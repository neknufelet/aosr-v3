"""兩份已驗結果的比較頁顯示資料。"""
from __future__ import annotations

from datetime import date

from aosr.config.quality_targets import QualityTargets
from aosr.reporting.compare import compare_results, comparison_problems, identity_difference
from aosr.reporting.display import (
    BASELINE_NOTE, LOW_FREQUENCY_DECAY_NOTE, REVERBERATION_ROOM_NOTE,
    SPATIAL_IMPRESSION_NOTE,
)
from aosr.reporting.result import SchemeResult
from aosr.reporting.result_view import CategoryView, LABELS, ResultView, ViewModel
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


class Overlay(ViewModel):
    frequency_hz: tuple[float, ...]
    series: tuple[OverlaySeries, ...]
    default_keys: tuple[str, str]


class CategoryRow(ViewModel):
    category: str
    label: str
    a: CategoryView
    b: CategoryView


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
    fingerprints: tuple[FingerprintCheck, ...]
    summary_text: str
    overlay: Overlay
    categories: tuple[CategoryRow, ...]
    table: TableStatus
    notes: tuple[str, ...]
    labels: dict[str, str]


_WALLS = {"x0": "左牆", "xL": "右牆", "y0": "前牆", "yL": "後牆",
          "floor": "地板", "ceiling": "天花板"}
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


def _number_pair(a: float | int, b: float | int, unit: str) -> tuple[str, str]:
    for digits in range(4, 16):
        left, right = f"{a:.{digits}g}", f"{b:.{digits}g}"
        if left != right:
            suffix = f" {unit}" if unit else ""
            return left + suffix, right + suffix
    left, right = f"{a:.15g} {unit}".strip(), f"{b:.15g} {unit}".strip()
    return left, right if left != right else right + "（微小差異）"


def _text(value: object, unit: str) -> str:
    if value is None:
        return "未設定"
    if isinstance(value, bool):
        return "有" if value else "無"
    if isinstance(value, float | int) and not isinstance(value, bool):
        return f"{value:.4g} {unit}".strip()
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, str) and value in LABELS:
        return LABELS[value]
    if isinstance(value, tuple):
        return "、".join(_text(item, "") for item in value)
    return str(value)


def _collect_scheme(scheme: Scheme, into: dict[str, object],
                    labels: dict[str, tuple[str, str]]) -> None:
    room = scheme.scene.room_m
    into.update({f"scene.room_m.{axis}": getattr(room, axis) for axis in ("Lx", "Ly", "Lz")})
    for field in ("sound_speed_m_s", "density_kg_m3", "reflection_order_k", "low_frequency_axis"):
        into[f"scene.{field}"] = getattr(scheme.scene, field)
    for field, title, unit in (("impedance_pa_s_per_m_by_wall", "阻抗", "Pa·s／m"),
                               ("scattering_by_wall", "散射", "")):
        wall_values = getattr(scheme.scene, field)
        if field == "scattering_by_wall":
            into[f"scene.{field}"] = wall_values is not None
            labels[f"scene.{field}"] = ("散射整組", "")
        for wall in sorted(set(wall_values or {})):
            path = f"scene.{field}.{wall}"
            into[path] = wall_values[wall] if wall_values is not None else None
            labels[path] = (f"{_WALLS.get(wall, wall)}{title}", unit)
    into["source_model"] = scheme.source_model
    into["purpose"] = scheme.purpose
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
        into[f"{root}.role"] = receiver.role.value
        into[f"{root}.importance"] = receiver.importance
        into[f"{root}.direction"] = receiver.direction_relative_to_primary
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


def _overlay(view_a: ResultView, view_b: ResultView) -> tuple[Overlay, tuple[str, ...]]:
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
    return Overlay(frequency_hz=axis, series=tuple(series),
                   default_keys=(defaults[0], defaults[1])), tuple(notes)


def _table(a: SchemeResult, b: SchemeResult, quality_targets: QualityTargets,
           run_date: date) -> TableStatus:
    problems = comparison_problems((a, b))
    if problems:
        return TableStatus(same_table=False, a_text="不可同表", b_text="不可同表",
                           reason_text="；".join(problems), calibration_text=BASELINE_NOTE)
    ranking = compare_results((a, b), quality_targets=quality_targets, run_date=run_date)
    rows = {row.candidate_id: row for row in ranking.rankable}
    def row_text(result: SchemeResult) -> str:
        name = result.scheme.scheme_id
        status = LABELS.get(ranking.status_of(name).value, ranking.status_of(name).value)
        ranked = rows.get(name)
        return (f"{status}；名次 {ranked.rank}；總代價 {ranked.total_cost:.4g}"
                if ranked else f"{status}；名次未列；總代價未列")
    incompatible = ranking.not_comparable.rows
    reason = (identity_difference(ranking, incompatible[0].identity)
              if incompatible else "同表")
    return TableStatus(same_table=not incompatible, a_text=row_text(a), b_text=row_text(b),
                       reason_text=reason, calibration_text=ranking.header.calibration_note or BASELINE_NOTE)


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
    overlay, fallback_notes = _overlay(view_a, view_b)
    category_a = {row.category: row for row in view_a.categories}
    category_b = {row.category: row for row in view_b.categories}
    categories = tuple(CategoryRow(category=kind.value, label=LABELS[kind.value],
                                   a=category_a[kind.value], b=category_b[kind.value])
                       for kind in QualityCategory)
    table = _table(a, b, quality_targets, run_date)
    preview = "、".join(row.label for row in changes[:5]) or "無"
    extra = f"等 {len(changes) - 5} 處" if len(changes) > 5 else ""
    summary = (f"改了 {len(changes)} 處：{preview}{extra}；"
               f"{'同表' if table.same_table else '不可同表'}；{LOW_FREQUENCY_DECAY_NOTE}")
    return CompareView(
        a=SideIdentity(run_id=a_run_id, scheme_id=a.scheme.scheme_id,
                       engine_text=a.engine_commit[:7], run_date=a.run_date.isoformat(),
                       total_text=view_a.timing_texts["total_s"]),
        b=SideIdentity(run_id=b_run_id, scheme_id=b.scheme.scheme_id,
                       engine_text=b.engine_commit[:7], run_date=b.run_date.isoformat(),
                       total_text=view_b.timing_texts["total_s"]),
        changes=changes, fingerprints=fingerprints, summary_text=summary,
        overlay=overlay, categories=categories, table=table,
        notes=(LOW_FREQUENCY_DECAY_NOTE, SPATIAL_IMPRESSION_NOTE,
               REVERBERATION_ROOM_NOTE, BASELINE_NOTE, *fallback_notes), labels=LABELS)
