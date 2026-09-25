"""從反射評估抽取逐點、逐方向區、逐頻點的有號聲道診斷。"""
from __future__ import annotations

from collections.abc import Sequence

from aosr.scoring.channel_matching_reflections_contract import (
    OneSidedReflection, ReflectionAsymmetry, ReflectionAsymmetryPoint,
    ReflectionAsymmetryState, ReflectionSide,
)
from aosr.scoring.contract import (
    CategoryEvaluation, ChannelComparisonPair, ChannelIdentity, EvaluationState,
    MetricState, QualityCategory, ReasonCode, in_declared_order,
)
from aosr.scoring.direction_zones import DirectionZone
from aosr.scoring.placement import Placement, PlacementMismatchError, merge_placements
from aosr.scoring.reflections_contract import (
    CONFIRMED_NO_REFLECTION, REFLECTIONS_AND_ECHO_EVALUATOR_VERSION,
    ReflectionChannel, ReflectionsAndEchoPayload, ZonePoint,
)


def _unavailable(reasons: tuple[ReasonCode, ...], upstream: CategoryEvaluation | None,
                 payload: ReflectionsAndEchoPayload | None,
                 pairs: tuple[tuple[str, str], ...]) -> ReflectionAsymmetry:
    return ReflectionAsymmetry(
        state=MetricState.UNAVAILABLE, reason_codes=reasons,
        reflections_evaluator_version=upstream.evaluator_version if upstream else None,
        reflections_settings_fingerprint=upstream.settings_fingerprint if upstream else None,
        source_flags=upstream.flags if upstream else (),
        frequency_range_hz=payload.frequency_range_hz if payload else None,
        window_upper_ms=payload.window_upper_ms if payload else None,
        primary_receiver_id=payload.primary_receiver_id if payload else None,
        comparison_order=pairs, points=(), one_sided=(),
    )


def _point(channel: ReflectionChannel, zone: DirectionZone, frequency: float) -> ZonePoint | None:
    for item in channel.zones:
        if item.zone is zone:
            return next((point for point in item.points if point.frequency_hz == frequency), None)
    return None


def _side(channel: ReflectionChannel, point: ZonePoint | None) -> ReflectionSide | None:
    # 聲道本身不可估、或時間窗沒證明蓋滿時，裡面即使留著點也不採信（老闆：已證明兩邊時間窗完整）。
    if channel.state is not MetricState.MEASURED or channel.coverage != "complete":
        return None
    if point is None or point.strongest_state is not MetricState.MEASURED:
        return None
    assert point.strongest_level_db is not None
    assert point.strongest_delay_s is not None
    assert point.strongest_path_index is not None
    path = channel.reflections[point.strongest_path_index]
    return ReflectionSide(
        role=channel.role, speaker_id=channel.speaker_id, level_db=point.strongest_level_db,
        delay_s=point.strongest_delay_s, path_index=point.strongest_path_index,
        source=path.source, source_index=path.source_index, order=path.order,
        wall_sequence=path.wall_sequence, provenance=channel.provenance,
    )


def _absence(channel: ReflectionChannel, point: ZonePoint | None) -> tuple[bool, tuple[ReasonCode, ...]]:
    if channel.state is not MetricState.MEASURED:
        return False, channel.reason_codes
    if channel.coverage != "complete":
        return False, (ReasonCode.REFLECTION_WINDOW_INCOMPLETE,)
    if point is None:
        return False, (ReasonCode.INSUFFICIENT_COVERAGE,)
    codes = point.strongest_reason_codes
    if codes and set(codes) <= CONFIRMED_NO_REFLECTION:
        return True, codes
    return False, codes or (ReasonCode.INSUFFICIENT_COVERAGE,)


def _cell(left: ReflectionChannel, right: ReflectionChannel, zone: DirectionZone,
          frequency: float) -> ReflectionAsymmetryPoint:
    lp, rp = _point(left, zone, frequency), _point(right, zone, frequency)
    ls, rs = _side(left, lp), _side(right, rp)
    la, lc = _absence(left, lp) if ls is None else (False, ())
    ra, rc = _absence(right, rp) if rs is None else (False, ())
    reasons: tuple[ReasonCode, ...] = ()
    if (ls is None and not la) or (rs is None and not ra):
        state = ReflectionAsymmetryState.UNAVAILABLE
        missing = (*(() if ls is not None or la else lc), *(() if rs is not None or ra else rc))
        reasons = in_declared_order(missing)
        ls = rs = None
    elif ls is not None and rs is not None:
        state = ReflectionAsymmetryState.MEASURED
    elif ls is not None or rs is not None:
        state = ReflectionAsymmetryState.ONE_SIDED
        reasons = rc if ls is not None else lc
    else:
        state = ReflectionAsymmetryState.BOTH_ABSENT
        reasons = in_declared_order((*lc, *rc))
    return ReflectionAsymmetryPoint(
        left_role=left.role, right_role=right.role, receiver_id=left.receiver_id,
        zone=zone, frequency_hz=frequency, state=state,
        left_minus_right_db=ls.level_db - rs.level_db if ls is not None and rs is not None else None,
        left=ls, right=rs, reason_codes=reasons,
    )


def _identity_reason(payload: ReflectionsAndEchoPayload, channels: Sequence[ChannelIdentity],
                     receiver_ids: Sequence[str], primary_receiver_id: str,
                     placement: Placement, upstream: CategoryEvaluation) -> ReasonCode | None:
    expected = {item.role: item.speaker_id for item in channels}
    actual = {item.role: item.speaker_id for item in payload.channels}
    if expected.keys() != actual.keys():
        return ReasonCode.CHANNEL_ROLE_MISMATCH
    if expected != actual:
        return ReasonCode.SPEAKER_ID_MISMATCH
    if ({item.receiver_id for item in payload.channels} != set(receiver_ids)
            or payload.primary_receiver_id != primary_receiver_id):
        return ReasonCode.RECEIVER_ID_MISMATCH
    try:
        merge_placements((placement, upstream.placement))
    except PlacementMismatchError:
        return ReasonCode.PLACEMENT_MISMATCH
    return None


def reflection_asymmetry(
    reflections: CategoryEvaluation | None, *, candidate_id: str,
    scene_fingerprint: str, channels: Sequence[ChannelIdentity],
    comparisons: Sequence[ChannelComparisonPair], receiver_ids: Sequence[str],
    primary_receiver_id: str, placement: Placement,
) -> ReflectionAsymmetry:
    """核對反射評估身分後逐格保留 L−R、單側及缺資料狀態。"""
    pairs = tuple((item.left_role, item.right_role) for item in comparisons)
    if reflections is None:
        return _unavailable((ReasonCode.REFLECTIONS_EVALUATION_MISSING,), None, None, pairs)
    if reflections.category is not QualityCategory.REFLECTIONS_AND_ECHO:
        raise ValueError("必須交反射類評估")
    payload = reflections.payload if isinstance(reflections.payload, ReflectionsAndEchoPayload) else None
    for wrong, reason in (
        (reflections.evaluator_version != REFLECTIONS_AND_ECHO_EVALUATOR_VERSION, ReasonCode.EVALUATOR_VERSION_MISMATCH),
        (reflections.candidate_id != candidate_id, ReasonCode.CANDIDATE_ID_MISMATCH),
        (reflections.scene_fingerprint != scene_fingerprint, ReasonCode.SCENE_FINGERPRINT_MISMATCH),
    ):
        if wrong:
            return _unavailable((reason,), reflections, payload, pairs)
    if reflections.state is EvaluationState.UNAVAILABLE:
        return _unavailable(reflections.reason_codes, reflections, payload, pairs)
    if payload is None:
        raise TypeError("反射評估 payload 必須是 ReflectionsAndEchoPayload")
    identity_reason = _identity_reason(payload, channels, receiver_ids, primary_receiver_id,
                              placement, reflections)
    if identity_reason is not None:
        return _unavailable((identity_reason,), reflections, payload, pairs)
    by_key = {(item.role, item.receiver_id): item for item in payload.channels}
    measured_channel = next(item for item in payload.channels if item.state is MetricState.MEASURED)
    frequencies = tuple(point.frequency_hz for point in measured_channel.zones[0].points)
    points = tuple(
        _cell(by_key[left, receiver], by_key[right, receiver], zone, frequency)
        for left, right in pairs for receiver in sorted(receiver_ids)
        for zone in DirectionZone for frequency in frequencies
    )
    one_sided_rows = []
    for item in points:
        if item.state is not ReflectionAsymmetryState.ONE_SIDED:
            continue
        present = item.left if item.left is not None else item.right
        assert present is not None
        one_sided_rows.append(OneSidedReflection(
            left_role=item.left_role, right_role=item.right_role,
            receiver_id=item.receiver_id, zone=item.zone, frequency_hz=item.frequency_hz,
            present=present,
            absent_role=item.right_role if item.left is not None else item.left_role,
            reason_codes=item.reason_codes,
        ))
    return ReflectionAsymmetry(
        state=MetricState.MEASURED, reason_codes=(),
        reflections_evaluator_version=reflections.evaluator_version,
        reflections_settings_fingerprint=reflections.settings_fingerprint,
        source_flags=reflections.flags, frequency_range_hz=payload.frequency_range_hz,
        window_upper_ms=payload.window_upper_ms,
        primary_receiver_id=payload.primary_receiver_id,
        comparison_order=pairs, points=points, one_sided=tuple(one_sided_rows),
    )
