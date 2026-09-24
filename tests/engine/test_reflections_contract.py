"""反射輸出契約拒收矛盾狀態與錯誤列身分的考卷。"""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from aosr.scoring.contract_base import Flag, InputProvenance, MetricState, ReasonCode
from aosr.scoring.direction_zones import DirectionZone, ZoneLimits
from aosr.scoring.reflections_contract import (
    MetricCell,
    ReflectionChannel,
    ReflectionPath,
    ReflectionSource,
    ReflectionsAndEchoPayload,
    WallPairBandRisk,
    WallPairRisk,
    ZonePoint,
    ZoneResult,
)


def _metric(value: float | None = None, reason: ReasonCode | None = ReasonCode.NO_REFLECTION_IN_ZONE_POINT) -> MetricCell:
    return MetricCell(value=value, state=MetricState.UNAVAILABLE if value is None else MetricState.MEASURED,
                        reason_codes=(reason,) if reason is not None else ())


def _payload() -> ReflectionsAndEchoPayload:
    point = ZonePoint(frequency_hz=1000.0, strongest_level_db=None, strongest_delay_ms=None,
                      strongest_path_index=None, strongest_state=MetricState.UNAVAILABLE,
                      strongest_reason_codes=(ReasonCode.NO_REFLECTION_IN_ZONE_POINT,),
                      total_energy_db=_metric())
    zones = tuple(ZoneResult(zone=zone, points=(point,)) for zone in DirectionZone)
    provenance = InputProvenance(report_id="r", engine_commit="c", speaker_id="l", receiver_id="main")
    channels = tuple(ReflectionChannel(role=role, speaker_id=role, receiver_id="main", is_primary=True,
                                       provenance=provenance.model_copy(update={"speaker_id": role}),
                                       reflections=(), zones=zones, total_window_energy_db=(_metric(),),
                                       report_order_k=1, computed_order_k=2, coverage="complete",
                                       validation="validated", state=MetricState.MEASURED,
                                       reason_codes=()) for role in ("left", "right"))
    unavailable = MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                               reason_codes=(ReasonCode.ZERO_RETENTION,))
    bands = (WallPairBandRisk(frequency_hz=1000.0, round_trip_loss_db=unavailable,
                              decay_duration_ms=unavailable,
                              room_t20_s=_metric(None, ReasonCode.T20_BAND_UNAVAILABLE)),)
    pairs = tuple(WallPairRisk(walls=walls, round_trip_delay_ms=_metric(10.0, None), bands=bands)
                  for walls in (("x0", "xL"), ("y0", "yL"), ("floor", "ceiling")))
    return ReflectionsAndEchoPayload(category="reflections_and_echo", window_upper_ms=15.0,
                                     window_upper_s=0.015, frequency_range_hz=(1000.0, 8000.0),
                                     zone_limits=ZoneLimits(vertical_min_abs_elevation_deg=30.0,
                                     front_max_abs_azimuth_deg=40.0, rear_min_abs_azimuth_deg=135.0),
                                     listening_axis_xy=(0.0, 1.0), listening_axis_rule="stereo_base_bisector_v1",
                                     primary_receiver_id="main", includes_speaker_directivity=False,
                                     channels=channels, wall_pairs=pairs)


def test_zero_and_missing_states_round_trip_without_sentinel() -> None:
    payload = _payload()
    loaded = ReflectionsAndEchoPayload.model_validate_json(payload.model_dump_json())
    point = loaded.channels[0].zones[0].points[0]
    assert point.strongest_level_db is None
    assert point.strongest_reason_codes == (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    assert point.total_energy_db.value is None
    assert set(Flag) >= {Flag.WINDOW_ONLY_DELAY_SCREEN, Flag.GEOMETRY_MATERIAL_CONSERVATIVE_SCREEN}
    assert set(ReasonCode) >= {ReasonCode.PATH_TABLE_MISSING, ReasonCode.REFLECTION_WINDOW_INCOMPLETE,
                               ReasonCode.LISTENING_AXIS_UNDEFINED}


@pytest.mark.parametrize("change,message", [
    ("metric", "值／狀態／原因碼"),
    ("length", "逐點長度"),
    ("frequency", "頻率必須遞增"),
    ("duplicate", "角色與接收點不可重複"),
    ("missing_primary", "主位每個角色"),
    ("extra", "extra_forbidden"),
    ("nonfinite", "finite_number"),
])
def test_payload_rejects_each_invalid_shape(change: str, message: str) -> None:
    document = deepcopy(_payload().model_dump(mode="python"))
    if change == "metric":
        document["channels"][0]["zones"][0]["points"][0]["total_energy_db"]["reason_codes"] = ()
    elif change == "length":
        document["channels"][0]["total_window_energy_db"] = ()
    elif change == "frequency":
        point = deepcopy(document["channels"][0]["zones"][0]["points"][0])
        document["channels"][0]["zones"][0]["points"] = (point, point)
    elif change == "duplicate":
        document["channels"] = (*document["channels"], document["channels"][0])
    elif change == "missing_primary":
        document["channels"][0]["is_primary"] = False
    elif change == "extra":
        document["unexpected"] = 1
    else:
        document["window_upper_s"] = float("inf")
    with pytest.raises(ValidationError, match=message):
        ReflectionsAndEchoPayload.model_validate(document)


def test_path_retains_raw_angles_and_zero_energy_reason() -> None:
    path = ReflectionPath(source=ReflectionSource.PATH_TABLE, source_index=0, order=1,
                          wall_sequence=("floor",), relative_direct_delay_ms=2.0,
                          room_azimuth_deg=0.0, room_elevation_deg=-45.0,
                          listening_azimuth_deg=0.0, listening_elevation_deg=-45.0,
                          zone=DirectionZone.VERTICAL, within_window=True,
                          broadband_level_db=None, broadband_state=MetricState.NOT_COMPUTABLE,
                          broadband_reason_codes=(ReasonCode.ZERO_REFLECTION_ENERGY,))
    assert ReflectionPath.model_validate_json(path.model_dump_json()).room_elevation_deg == -45.0


def test_zero_energy_strongest_retains_path_and_delay() -> None:
    point = ZonePoint(frequency_hz=1000.0, strongest_level_db=None, strongest_delay_ms=2.0,
                      strongest_path_index=0, strongest_state=MetricState.NOT_COMPUTABLE,
                      strongest_reason_codes=(ReasonCode.ZERO_REFLECTION_ENERGY,),
                      total_energy_db=_metric(None, ReasonCode.ZERO_REFLECTION_ENERGY))
    assert point.strongest_path_index == 0
    assert point.strongest_delay_ms == 2.0


def test_wall_pairs_require_three_named_axes_and_matching_bands() -> None:
    document = deepcopy(_payload().model_dump(mode="python"))
    document["wall_pairs"] = document["wall_pairs"][:-1]
    with pytest.raises(ValidationError, match="三對平行牆"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_every_receiver_has_each_primary_speaker_role() -> None:
    document = deepcopy(_payload().model_dump(mode="python"))
    surrounding = deepcopy(document["channels"][0])
    surrounding["receiver_id"] = "near"
    surrounding["is_primary"] = False
    surrounding["provenance"]["receiver_id"] = "near"
    document["channels"] = (document["channels"][0], surrounding, document["channels"][1])
    with pytest.raises(ValidationError, match="每個接收點都必須保留每個角色"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_primary_requires_exactly_two_speaker_roles() -> None:
    document = deepcopy(_payload().model_dump(mode="python"))
    document["channels"] = document["channels"][:1]
    with pytest.raises(ValidationError, match="兩支喇叭"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_wall_pair_rejects_nonpositive_measured_loss() -> None:
    unavailable = _metric(None, ReasonCode.FULL_REFLECTION)
    with pytest.raises(ValidationError, match="全反射損耗"):
        WallPairBandRisk(frequency_hz=1000.0, round_trip_loss_db=_metric(0.0, None),
                         decay_duration_ms=unavailable, room_t20_s=unavailable)


def test_wall_pair_zero_retention_is_explicitly_not_computable() -> None:
    metric = MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                          reason_codes=(ReasonCode.ZERO_RETENTION,))
    band = WallPairBandRisk(frequency_hz=1000.0, round_trip_loss_db=metric,
                            decay_duration_ms=metric, room_t20_s=_metric(0.5, None))
    assert band.decay_duration_ms.value is None
    assert band.decay_duration_ms.state is MetricState.NOT_COMPUTABLE


def test_measured_channels_use_same_fine_axis() -> None:
    document = deepcopy(_payload().model_dump(mode="python"))
    channel = document["channels"][0]
    for zone in channel["zones"]:
        point = deepcopy(zone["points"][0])
        point["frequency_hz"] = 2000.0
        zone["points"] = (point,)
    with pytest.raises(ValidationError, match="各聲道逐點頻率必須一致"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_primary_incomplete_window_cannot_claim_measured_payload() -> None:
    document = deepcopy(_payload().model_dump(mode="python"))
    document["channels"][0]["coverage"] = "not_provable"
    with pytest.raises(ValidationError, match="主位時間窗未蓋滿"):
        ReflectionsAndEchoPayload.model_validate(document)
