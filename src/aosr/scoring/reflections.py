"""#351 反射與顫動回音的逐點評估：只量診斷，不計代價。

代價、逐區標記與顫動警戒在 ``reflections_cost``。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from aosr.config.quality_targets import QualityPurpose, SettingEntry, load_quality_targets
from aosr.physics.reflection_screen import ReflectionScreen, WallPairRow
from aosr.physics.reflection_window import ReflectionWindow
from aosr.physics.report_io import PathRow, ReportOutput
from aosr.physics.room_paths import NUMERICALLY_GUARDED_ORDER_K
from aosr.physics.third_octave_decay import (
    ThirdOctaveDecay, ThirdOctaveDecayRow,
    subband_weighted_mean, third_octave_bands,
)
from aosr.scoring.channel_matching import ChannelGroup
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CategoryEvaluation, EvaluationState, Flag, InputProvenance,
    MetricState, QualityCategory, RawQuantity, ReasonCode, in_declared_order,
)
from aosr.scoring.direction_zones import (
    DirectionZone, ListeningAxisUndefined, ZoneLimits, classify, listening_angles,
    listening_axis, zone_limits,
)
from aosr.scoring.placement import Placement, merge_or_empty, point_placement
from aosr.scoring.reflections_contract import (
    REFLECTIONS_AND_ECHO_EVALUATOR_VERSION, MetricCell, ReflectionChannel, ReflectionPath, ReflectionSource,
    ReflectionsAndEchoPayload, WallPairBandRisk, WallPairRisk, ZonePoint, ZoneResult,
)
from aosr.scoring.reverberation import _reason_code
from aosr.scoring.timbre import _octave_mean_level_db


_LISTENING_AXIS_RULE: Final[str] = "loudspeaker_base_bisector_toward_speakers.v1"
_PREFIX: Final[str] = "reflections_and_echo."


@dataclass(frozen=True)
class ReflectionInput:
    """一角色、一接收點的主報表與反射專用物理資料。"""

    role: str
    receiver_id: str
    report: ReportOutput
    screen: ReflectionScreen | None
    window: ReflectionWindow | None
    third_octave_decay: ThirdOctaveDecay | None
    report_id: str
    engine_commit: str
    speaker_id: str


@dataclass(frozen=True)
class _Settings:
    window_ms: float
    bounds_hz: tuple[float, float]
    limits: ZoneLimits
    decay_db: float
    flutter_alert_band_centers_hz: tuple[int, ...]
    baseline: bool
    fingerprint: str

    @property
    def window_s(self) -> float:
        return self.window_ms / 1000.0


def _entry(purpose: QualityPurpose, key: str, unit: str) -> SettingEntry:
    found = purpose.entry(key)
    if not isinstance(found, SettingEntry) or found.unit != unit:
        raise ValueError(f"{key} 必須是 {unit} 量法設定")
    return found


def _scalar(entry: SettingEntry) -> float:
    if isinstance(entry.value, tuple):
        raise ValueError(f"{entry.key} 必須是單一數值")
    return float(entry.value)


def _settings(path: str | Path, purpose_name: str) -> _Settings:
    purpose = load_quality_targets(path).purpose(purpose_name)
    window = _entry(purpose, _PREFIX + "window_upper_ms", "ms")
    frequency = _entry(purpose, _PREFIX + "frequency_range_hz", "Hz")
    decay = _entry(purpose, _PREFIX + "flutter_decay_db", "dB")
    alert = _entry(purpose, _PREFIX + "flutter_alert_band_centers_hz", "Hz")
    if not isinstance(frequency.value, tuple) or len(frequency.value) != 2:
        raise ValueError("frequency_range_hz 必須有兩個端點")
    if not isinstance(alert.value, tuple):
        raise ValueError("flutter_alert_band_centers_hz 必須是清單")
    if any(not float(value).is_integer() for value in alert.value):
        raise ValueError("flutter_alert_band_centers_hz 必須是整數標稱帶名")
    alert_centers = tuple(int(value) for value in alert.value)
    bounds = tuple(float(value) for value in frequency.value)
    limits, statuses = zone_limits(purpose)
    used_statuses = (window.status, frequency.status, decay.status, alert.status,
                     *statuses.values())
    canonical = json.dumps({
        "window_upper_ms": window.value,
        "frequency_range_hz": bounds,
        "zone_limits": limits.model_dump(mode="json"),
        "flutter_decay_db": decay.value,
        "flutter_alert_band_centers_hz": alert_centers,
        "listening_axis_rule": _LISTENING_AXIS_RULE,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _Settings(
        window_ms=_scalar(window), bounds_hz=(bounds[0], bounds[1]),
        limits=limits, decay_db=_scalar(decay),
        flutter_alert_band_centers_hz=alert_centers,
        baseline="baseline" in used_statuses,
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _provenance(data: ReflectionInput) -> InputProvenance:
    return InputProvenance(
        report_id=data.report_id, engine_commit=data.engine_commit,
        speaker_id=data.speaker_id, receiver_id=data.receiver_id,
    )


def _placement(data: Sequence[ReflectionInput]) -> Placement:
    return merge_or_empty(
        point_placement(
            item.speaker_id, item.report.scene.source_m.as_tuple(),
            item.receiver_id, item.report.scene.receiver_m.as_tuple(),
        ) for item in data
    )


def _flags(data: Sequence[ReflectionInput], settings: _Settings,
           payload: ReflectionsAndEchoPayload | None) -> tuple[Flag, ...]:
    found = [
        Flag.NO_DIRECTIVITY, Flag.WINDOW_ONLY_DELAY_SCREEN,
        Flag.GEOMETRY_MATERIAL_CONSERVATIVE_SCREEN,
    ]
    if settings.baseline:
        found.append(Flag.BASELINE_SETTINGS)
    measured = ({(channel.role, channel.receiver_id) for channel in payload.channels
                 if channel.state is MetricState.MEASURED} if payload is not None else set())
    if any(
        item.report.top.reflection_order_k > NUMERICALLY_GUARDED_ORDER_K
        or item.window is not None and item.window.validation == "unvalidated"
        for item in data if (item.role, item.receiver_id) in measured
    ):
        found.append(Flag.UNVALIDATED)
    return in_declared_order(found)


def _result(
    data: Sequence[ReflectionInput], candidate_id: str, settings: _Settings,
    state: EvaluationState, reason_codes: tuple[ReasonCode, ...],
    primary_receiver_id: str,
    payload: ReflectionsAndEchoPayload | None = None,
    raw: tuple[RawQuantity, ...] = (),
    used: Sequence[ReflectionInput] | None = None,
) -> CategoryEvaluation:
    ordered = sorted(data, key=lambda item: (item.role, item.receiver_id))
    first = next((item for item in ordered if item.receiver_id == primary_receiver_id), ordered[0])
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id=candidate_id,
        scene_fingerprint=first.report.scene.scene_fingerprint,
        placement=_placement(ordered if used is None else used),
        category=QualityCategory.REFLECTIONS_AND_ECHO,
        state=state, payload=payload, raw_quantities=raw, category_cost=None,
        flags=_flags(ordered, settings, payload), reason_codes=reason_codes,
        evaluator_version=REFLECTIONS_AND_ECHO_EVALUATOR_VERSION,
        settings_fingerprint=settings.fingerprint, provenance=_provenance(first),
    )


def _identity_reason(
    data: Sequence[ReflectionInput], group: ChannelGroup, primary_id: str,
) -> ReasonCode | None:
    expected = {item.role: item.speaker_id for item in group.channels}
    actual_roles = {(item.receiver_id, item.role) for item in data}
    if len(actual_roles) != len(data):
        return ReasonCode.CHANNEL_ROLE_MISMATCH
    if any(item.role not in expected or item.speaker_id != expected[item.role] for item in data):
        return ReasonCode.CHANNEL_ROLE_MISMATCH
    if {item.role for item in data} != set(expected):
        return ReasonCode.CHANNEL_ROLE_MISMATCH
    receivers = {item.receiver_id for item in data} | {primary_id}
    if any({role for receiver, role in actual_roles if receiver == point} != set(expected)
           for point in receivers):
        return ReasonCode.RECEIVER_ID_MISMATCH
    return None


def _physical_reason(data: ReflectionInput, settings: _Settings) -> ReasonCode | None:
    report, screen, window = data.report, data.screen, data.window
    decay = data.third_octave_decay
    table = report.path_table
    if table is None:
        return ReasonCode.PATH_TABLE_MISSING
    if screen is None or window is None or decay is None:
        return ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISSING
    scene = report.scene
    if (screen.scene_fingerprint != scene.scene_fingerprint
            or window.scene_fingerprint != scene.scene_fingerprint
            or decay.scene_fingerprint != scene.scene_fingerprint):
        return ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
    if tuple(row.band for row in decay.rows) != third_octave_bands():
        return ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
    if (screen.source_m != scene.source_m or screen.receiver_m != scene.receiver_m
            or window.source_m != scene.source_m or window.receiver_m != scene.receiver_m):
        return ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
    if (screen.reflection_order_k != report.top.reflection_order_k
            or window.report_order_k != report.top.reflection_order_k
            or table.reflection_order_k != report.top.reflection_order_k):
        return ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
    if table.frequencies_hz != screen.frequencies_hz or table.frequencies_hz != window.frequencies_hz:
        return ReasonCode.FREQUENCY_AXIS_MISMATCH
    if report.points is not None:
        if tuple(point.frequency_hz for point in report.points) != table.frequencies_hz:
            return ReasonCode.FREQUENCY_AXIS_MISMATCH
        if tuple(point.scattering for point in report.points) != table.scattering_coefficient:
            return ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
    if any(len(row.relative_direct_energy) != len(table.frequencies_hz)
           for row in (*table.rows, *window.rows)):
        return ReasonCode.FREQUENCY_AXIS_MISMATCH
    if table.scattering_coefficient != window.scattering_coefficient:
        return ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
    direct = next((row for row in table.rows if row.order == 0), None)
    if direct is None or window.direct_delay_s != direct.delay_s or window.window_s != settings.window_s:
        return ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
    if window.coverage != "complete":
        return ReasonCode.REFLECTION_WINDOW_INCOMPLETE
    if (table.frequencies_hz[0] > settings.bounds_hz[0]
            or table.frequencies_hz[-1] < settings.bounds_hz[1]
            or not any(settings.bounds_hz[0] <= frequency <= settings.bounds_hz[1]
                       for frequency in table.frequencies_hz)):
        return ReasonCode.INSUFFICIENT_COVERAGE
    if not report.bands:
        return ReasonCode.BAND_ROW_MISSING
    return None


def _missing(reason: ReasonCode) -> MetricCell:
    return MetricCell(value=None, state=MetricState.NOT_COMPUTABLE, reason_codes=(reason,))


def _level(energy: float, empty_reason: ReasonCode) -> MetricCell:
    if energy <= 0.0:
        return _missing(empty_reason)
    return MetricCell(value=10.0 * math.log10(energy), state=MetricState.MEASURED, reason_codes=())


def _sum_level(energy: Sequence[float], empty_reason: ReasonCode) -> MetricCell:
    peak = max(energy, default=0.0)
    if peak <= 0.0:
        return _missing(empty_reason)
    scaled = math.fsum(value / peak for value in energy)
    return MetricCell(
        value=10.0 * (math.log10(peak) + math.log10(scaled)),
        state=MetricState.MEASURED, reason_codes=(),
    )


def _broadband(frequencies: tuple[float, ...], energy: tuple[float, ...],
               bounds: tuple[float, float]) -> MetricCell:
    selected = tuple(index for index, frequency in enumerate(frequencies)
                     if bounds[0] <= frequency <= bounds[1])
    if not selected:
        return _missing(ReasonCode.INSUFFICIENT_COVERAGE)
    if not any(energy[index] > 0.0 for index in selected):
        return _missing(ReasonCode.ZERO_REFLECTION_ENERGY)
    values = np.asarray([energy[index] for index in selected], dtype=np.float64)
    axis = np.asarray([frequencies[index] for index in selected], dtype=np.float64)
    return MetricCell(
        value=_octave_mean_level_db(axis, values, bounds),
        state=MetricState.MEASURED, reason_codes=(),
    )


def _paths(data: ReflectionInput, settings: _Settings, axis: tuple[float, float]
           ) -> tuple[ReflectionPath, ...]:
    table = data.report.path_table
    window = data.window
    assert table is not None and window is not None
    direct = next(row for row in table.rows if row.order == 0)
    rows: list[tuple[ReflectionSource, int, PathRow]] = [
        (ReflectionSource.PATH_TABLE, index, row)
        for index, row in enumerate(table.rows) if row.order > 0
    ]
    rows.extend((ReflectionSource.WINDOW_EXTENSION, index, row)
                for index, row in enumerate(window.rows))
    found: list[ReflectionPath] = []
    for source, source_index, row in rows:
        relative = row.delay_s - direct.delay_s
        azimuth, elevation = listening_angles(row.direction_vector, axis)
        broadband = _broadband(table.frequencies_hz, row.relative_direct_energy,
                               settings.bounds_hz)
        found.append(ReflectionPath(
            source=source, source_index=source_index, order=row.order,
            wall_sequence=row.wall_sequence, relative_direct_delay_s=relative,
            room_azimuth_deg=row.direction_angles.azimuth_deg,
            room_elevation_deg=row.direction_angles.elevation_deg,
            listening_azimuth_deg=azimuth, listening_elevation_deg=elevation,
            zone=classify(azimuth, elevation, settings.limits),
            within_window=relative <= settings.window_s,
            broadband_level_db=broadband.value, broadband_state=broadband.state,
            broadband_reason_codes=broadband.reason_codes,
        ))
    return tuple(found)


def _point(
    frequency: float, column: int, zone: DirectionZone,
    paths: tuple[ReflectionPath, ...], rows: tuple[PathRow, ...],
) -> ZonePoint:
    candidates = [index for index, path in enumerate(paths)
                  if path.zone is zone and path.within_window]
    if not candidates:
        missing = _missing(ReasonCode.NO_REFLECTION_IN_ZONE_POINT)
        return ZonePoint(
            frequency_hz=frequency, strongest_level_db=None,
            strongest_delay_s=None, strongest_path_index=None,
            strongest_state=missing.state, strongest_reason_codes=missing.reason_codes,
            total_energy_db=missing,
        )
    winner = min(candidates, key=lambda index: (
        -rows[index].relative_direct_energy[column],
        paths[index].relative_direct_delay_s, index,
    ))
    energy = rows[winner].relative_direct_energy[column]
    strongest = _level(energy, ReasonCode.ZERO_REFLECTION_ENERGY)
    total = _sum_level(
        tuple(rows[index].relative_direct_energy[column] for index in candidates),
        ReasonCode.ZERO_REFLECTION_ENERGY,
    )
    return ZonePoint(
        frequency_hz=frequency, strongest_level_db=strongest.value,
        strongest_delay_s=paths[winner].relative_direct_delay_s,
        strongest_path_index=winner, strongest_state=strongest.state,
        strongest_reason_codes=strongest.reason_codes, total_energy_db=total,
    )


def _channel(data: ReflectionInput, settings: _Settings, axis: tuple[float, float],
             reason: ReasonCode | None) -> ReflectionChannel:
    table = data.report.path_table
    window = data.window
    if reason is None:
        assert table is not None and window is not None
        paths = _paths(data, settings, axis)
        frequency_indices = tuple(index for index, frequency in enumerate(table.frequencies_hz)
                                  if settings.bounds_hz[0] <= frequency <= settings.bounds_hz[1])
        rows = tuple(row for row in table.rows if row.order > 0) + window.rows
    else:
        paths = ()
        frequency_indices = ()
        rows = ()
    frequencies = table.frequencies_hz if table is not None else ()
    zones = tuple(ZoneResult(
        zone=zone, points=tuple(_point(frequencies[index], index, zone, paths, rows)
                                for index in frequency_indices),
    ) for zone in DirectionZone)
    totals = tuple(
        _sum_level(
            tuple(row.relative_direct_energy[index] for row, path in zip(rows, paths)
                  if path.within_window),
            ReasonCode.ZERO_REFLECTION_ENERGY if any(path.within_window for path in paths)
            else ReasonCode.NO_REFLECTION_IN_ZONE_POINT,
        )
        for index in frequency_indices
    )
    report_order = data.report.top.reflection_order_k
    window_order = window.computed_order_k if window is not None else report_order
    computed_order = max(report_order, window_order)
    return ReflectionChannel(
        role=data.role, speaker_id=data.speaker_id, receiver_id=data.receiver_id,
        is_primary=False, provenance=_provenance(data), reflections=paths,
        zones=zones, total_window_energy_db=totals,
        report_order_k=report_order, computed_order_k=computed_order,
        coverage=window.coverage if window is not None else "missing",
        validation=window.validation if window is not None and window_order == computed_order else "missing",
        state=MetricState.MEASURED if reason is None else MetricState.UNAVAILABLE,
        reason_codes=() if reason is None else (reason,),
    )


def _wall_pairs(data: ReflectionInput, settings: _Settings) -> tuple[WallPairRisk, ...]:
    screen = data.screen
    decay = data.third_octave_decay
    assert screen is not None and decay is not None
    axis = screen.frequencies_hz
    found: list[WallPairRisk] = []
    for pair in screen.pairs:
        bands = tuple(_wall_band(pair, axis, row, settings)
                      for row in decay.rows)
        found.append(WallPairRisk(
            walls=pair.faces,
            round_trip_delay_s=MetricCell(
                value=pair.round_trip_delay_s, state=MetricState.MEASURED, reason_codes=()),
            bands=bands,
        ))
    return tuple(found)


def _wall_band(pair: WallPairRow, axis: tuple[float, ...], row: ThirdOctaveDecayRow,
               settings: _Settings) -> WallPairBandRisk:
    band = row.band
    if not any(band.lower_hz <= frequency < band.upper_hz for frequency in axis):
        # 子帶裡一個逐頻點都沒有：先判，不靠比對共用函式丟出的錯誤字串
        loss = _missing(ReasonCode.INSUFFICIENT_COVERAGE)
    else:
        retained = subband_weighted_mean(axis, pair.round_trip_retained_energy, band)
        if retained <= 0.0:
            loss = _missing(ReasonCode.ZERO_RETENTION)
        elif retained >= 1.0:
            loss = _missing(ReasonCode.FULL_REFLECTION)
        else:
            loss = MetricCell(value=-10.0 * math.log10(retained),
                              state=MetricState.MEASURED, reason_codes=())
    duration = (_missing(loss.reason_codes[0]) if loss.value is None else MetricCell(
        value=settings.decay_db / loss.value * pair.round_trip_delay_s,
        state=MetricState.MEASURED, reason_codes=(),
    ))
    if row.t20_s is None:
        t20 = MetricCell(
            value=None, state=MetricState.UNAVAILABLE,
            reason_codes=(_reason_code(row.t20_unavailable_reason or ""),),
        )
    elif row.t20_s <= 0.0:
        t20 = MetricCell(
            value=None, state=MetricState.UNAVAILABLE,
            reason_codes=(ReasonCode.NON_POSITIVE_VALUE,),
        )
    else:
        t20 = MetricCell(value=row.t20_s, state=MetricState.MEASURED, reason_codes=())
    return WallPairBandRisk(
        frequency_hz=band.center_hz, nominal_center_hz=band.nominal_center_hz,
        lower_hz=band.lower_hz, upper_hz=band.upper_hz,
        round_trip_loss_db=loss,
        decay_duration_s=duration, room_t20_s=t20,
    )


def _third_t20_rows(decay: ThirdOctaveDecay) -> tuple[tuple[float | None, str | None], ...]:
    return tuple((row.t20_s, row.t20_unavailable_reason) for row in decay.rows)


def _data_reasons(
    data: tuple[ReflectionInput, ...], primary: tuple[ReflectionInput, ...],
    settings: _Settings,
) -> tuple[dict[tuple[str, str], ReasonCode | None], ReasonCode | None]:
    common_scene = primary[0].report.scene.scene_fingerprint
    if any(item.report.scene.scene_fingerprint != common_scene for item in primary):
        return {}, ReasonCode.SCENE_FINGERPRINT_MISMATCH
    reasons = {(item.role, item.receiver_id): _physical_reason(item, settings)
               for item in data}
    for item in data:
        key = (item.role, item.receiver_id)
        if item.report.scene.scene_fingerprint != common_scene:
            reasons[key] = ReasonCode.SCENE_FINGERPRINT_MISMATCH
    for item in primary:
        cause = reasons[item.role, item.receiver_id]
        if cause is not None:
            return reasons, cause
    assert primary[0].screen is not None
    assert primary[0].report.path_table is not None
    primary_axis = primary[0].report.path_table.frequencies_hz
    if any(item.report.path_table is None
           or item.report.path_table.frequencies_hz != primary_axis for item in primary):
        return reasons, ReasonCode.FREQUENCY_AXIS_MISMATCH
    for item in data:
        table = item.report.path_table
        key = (item.role, item.receiver_id)
        if reasons[key] is None and table is not None and table.frequencies_hz != primary_axis:
            reasons[key] = ReasonCode.FREQUENCY_AXIS_MISMATCH
    for item in data:
        key = (item.role, item.receiver_id)
        if (item.screen is not None and reasons[key] is None
                and (item.screen.pairs != primary[0].screen.pairs
                     or item.third_octave_decay is None
                     or primary[0].third_octave_decay is None
                     or _third_t20_rows(item.third_octave_decay)
                     != _third_t20_rows(primary[0].third_octave_decay))):
            if item.receiver_id == primary[0].receiver_id:
                return reasons, ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
            reasons[key] = ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH
    return reasons, _placement_reason(data, primary, reasons)


def _placement_reason(
    data: tuple[ReflectionInput, ...], primary: tuple[ReflectionInput, ...],
    reasons: dict[tuple[str, str], ReasonCode | None],
) -> ReasonCode | None:
    """主位自己的擺位衝突才整類不可估；周圍點對不上只記那一支（#480）。

    反射的分數只看主位，周圍點壞了只記那一支；已經因別的原因拒用的周圍點不再拿來合併擺位，
    免得一筆不用的舊座標把主位一起否決。周圍某一點跟主位同代號不同座標、或同一點兩支
    之間接收點座標不同，那一點對不上的每一支記擺位不符：那一支只記原因碼，出身欄的報表代號指回原報表，
    舊座標本身不進輸出。只管可估結果的擺位；整類不可估時照舊把全部輸入合併（衝突就帶空擺位）。
    """
    if not _placement(primary).speaker_positions_m:
        return ReasonCode.PLACEMENT_MISMATCH
    groups: dict[str, list[ReflectionInput]] = {}
    for item in data:
        if item.receiver_id != primary[0].receiver_id and reasons[item.role, item.receiver_id] is None:
            groups.setdefault(item.receiver_id, []).append(item)
    for items in groups.values():
        internal = not _placement(items).receiver_positions_m
        for item in items:
            if internal or not _placement((*primary, item)).speaker_positions_m:
                reasons[item.role, item.receiver_id] = ReasonCode.PLACEMENT_MISMATCH
    return None


def evaluate_reflections(
    channel_group: ChannelGroup, inputs: Sequence[ReflectionInput], *,
    primary_receiver_id: str, candidate_id: str, purpose: str,
    quality_targets_path: str | Path,
) -> CategoryEvaluation:
    """量一候選兩聲道在主位與周圍點的早期反射及平行牆顫動。"""
    if not inputs:
        raise ValueError("至少要有一份報表")
    data = tuple(sorted(inputs, key=lambda item: (item.role, item.receiver_id)))
    settings = _settings(quality_targets_path, purpose)
    identity = _identity_reason(data, channel_group, primary_receiver_id)
    if identity is not None:
        return _result(data, candidate_id, settings, EvaluationState.UNAVAILABLE,
                       (identity,), primary_receiver_id)
    primary = tuple(item for item in data if item.receiver_id == primary_receiver_id)
    # 聲道數不是 2 由 listening_axis 自己判（丟 ListeningAxisUndefined），這裡不再重複一道
    try:
        axis = listening_axis(tuple(item.report.scene.source_m.as_tuple() for item in primary),
                              primary[0].report.scene.receiver_m.as_tuple())
    except ListeningAxisUndefined:
        return _result(data, candidate_id, settings, EvaluationState.UNAVAILABLE,
                       (ReasonCode.LISTENING_AXIS_UNDEFINED,), primary_receiver_id)
    reasons, failure = _data_reasons(data, primary, settings)
    if failure is not None:
        return _result(data, candidate_id, settings, EvaluationState.UNAVAILABLE,
                       (failure,), primary_receiver_id)
    channels = tuple(
        _channel(item, settings, axis, reasons[item.role, item.receiver_id]).model_copy(
            update={"is_primary": item.receiver_id == primary_receiver_id})
        for item in data
    )
    pairs = _wall_pairs(primary[0], settings)
    payload = ReflectionsAndEchoPayload(
        category="reflections_and_echo", window_upper_ms=settings.window_ms,
        window_upper_s=settings.window_s, frequency_range_hz=settings.bounds_hz,
        zone_limits=settings.limits, listening_axis_xy=axis,
        listening_axis_rule=_LISTENING_AXIS_RULE,
        primary_receiver_id=primary_receiver_id, includes_speaker_directivity=False,
        channels=channels, wall_pairs=pairs,
        flutter_alert_band_centers_hz=settings.flutter_alert_band_centers_hz,
    )
    raw = tuple(RawQuantity(
        name=f"round_trip_delay_{pair.walls[0]}_{pair.walls[1]}",
        value=pair.round_trip_delay_s.value, unit="s",
    ) for pair in pairs if pair.round_trip_delay_s.value is not None)
    used = tuple(item for item in data if reasons[item.role, item.receiver_id] is None)
    return _result(data, candidate_id, settings, EvaluationState.MEASURED, (),
                   primary_receiver_id, payload, raw, used)
