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


def _measured_path(source: ReflectionSource = ReflectionSource.PATH_TABLE,
                   source_index: int = 0) -> ReflectionPath:
    return ReflectionPath(source=source, source_index=source_index, order=1,
                          wall_sequence=("front",), relative_direct_delay_ms=2.0,
                          room_azimuth_deg=0.0, room_elevation_deg=0.0,
                          listening_azimuth_deg=0.0, listening_elevation_deg=0.0,
                          zone=DirectionZone.FRONT, within_window=True,
                          broadband_level_db=-12.0, broadband_state=MetricState.MEASURED,
                          broadband_reason_codes=())


def _channel_with_paths(*paths: ReflectionPath) -> ReflectionChannel:
    document = _payload().channels[0].model_dump(mode="python")
    document["reflections"] = tuple(path.model_dump(mode="python") for path in paths)
    return ReflectionChannel.model_validate(document)


def _channel_with_strongest_path() -> ReflectionChannel:
    document = _channel_with_paths(_measured_path()).model_dump(mode="python")
    front = next(zone for zone in document["zones"] if zone["zone"] is DirectionZone.FRONT)
    front["points"][0]["strongest_level_db"] = -12.0
    front["points"][0]["strongest_delay_ms"] = 2.0
    front["points"][0]["strongest_path_index"] = 0
    front["points"][0]["strongest_state"] = MetricState.MEASURED
    front["points"][0]["strongest_reason_codes"] = ()
    return ReflectionChannel.model_validate(document)


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


@pytest.mark.parametrize("value,state", [
    (1.0, MetricState.UNAVAILABLE),
    (None, MetricState.MEASURED),
])
def test_metric_cell_rejects_value_state_mismatch(value: float | None, state: MetricState) -> None:
    valid = _metric(value, None if value is not None else ReasonCode.NO_REFLECTION_IN_ZONE_POINT)
    document = valid.model_dump(mode="python")
    document["state"] = state
    with pytest.raises(ValidationError, match="值／狀態／原因碼不一致"):
        MetricCell.model_validate(document)


def test_path_rejects_wall_sequence_length_mismatch() -> None:
    document = _measured_path().model_dump(mode="python")
    document["order"] = 2
    with pytest.raises(ValidationError, match="牆序列長度必須等於反射階數"):
        ReflectionPath.model_validate(document)


@pytest.mark.parametrize("field,value", [
    ("strongest_delay_ms", 2.0),
    ("strongest_path_index", 0),
])
def test_strongest_delay_and_index_must_appear_together(field: str, value: float | int) -> None:
    document = _payload().channels[0].zones[0].points[0].model_dump(mode="python")
    document[field] = value
    with pytest.raises(ValidationError, match="最強反射的延遲與路徑索引必須同進同出"):
        ZonePoint.model_validate(document)


def test_zero_energy_path_requires_its_specific_reason() -> None:
    valid = ZonePoint(frequency_hz=1000.0, strongest_level_db=None, strongest_delay_ms=2.0,
                      strongest_path_index=0, strongest_state=MetricState.NOT_COMPUTABLE,
                      strongest_reason_codes=(ReasonCode.ZERO_REFLECTION_ENERGY,),
                      total_energy_db=_metric(None, ReasonCode.ZERO_REFLECTION_ENERGY))
    document = valid.model_dump(mode="python")
    document["strongest_reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    with pytest.raises(ValidationError, match="零能量路徑必須帶零能量原因碼"):
        ZonePoint.model_validate(document)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_channel_requires_each_zone_exactly_once(change: str) -> None:
    document = _payload().channels[0].model_dump(mode="python")
    zones = document["zones"]
    document["zones"] = zones[:-1] if change == "missing" else (*zones, zones[0])
    with pytest.raises(ValidationError, match="聲道必須保留四區結果且不可重複"):
        ReflectionChannel.model_validate(document)


@pytest.mark.parametrize("change,indices", [
    ("extension_before_table", (0, 1)),
    ("duplicate_index", (0, 1)),
    ("descending_index", (0, 2, 3)),
])
def test_channel_rejects_reflection_source_order_and_indices(change: str, indices: tuple[int, ...]) -> None:
    valid = _channel_with_paths(*(_measured_path(source_index=index) for index in indices))
    document = valid.model_dump(mode="python")
    if change == "extension_before_table":
        document["reflections"][0]["source"] = ReflectionSource.WINDOW_EXTENSION
    elif change == "duplicate_index":
        document["reflections"][1]["source_index"] = 0
    else:
        document["reflections"][2]["source_index"] = 1
    with pytest.raises(ValidationError, match="反射路徑必須先主表後補算且來源索引遞增不重複"):
        ReflectionChannel.model_validate(document)


@pytest.mark.parametrize("field,value", [
    ("zone", DirectionZone.LATERAL),
    ("within_window", False),
])
def test_strongest_index_must_refer_to_same_zone_inside_window(field: str, value: DirectionZone | bool) -> None:
    document = _channel_with_strongest_path().model_dump(mode="python")
    document["reflections"][0][field] = value
    with pytest.raises(ValidationError, match="最強反射路徑索引必須指向同區窗內路徑"):
        ReflectionChannel.model_validate(document)


@pytest.mark.parametrize("field", ["speaker_id", "receiver_id"])
def test_channel_provenance_must_match_its_identity(field: str) -> None:
    document = _payload().channels[0].model_dump(mode="python")
    document["provenance"][field] = "someone_else"
    with pytest.raises(ValidationError, match="出身與聲道身分不一致"):
        ReflectionChannel.model_validate(document)


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


@pytest.mark.parametrize("cause,wrong_cause", [
    (ReasonCode.ZERO_RETENTION, ReasonCode.FULL_REFLECTION),
    (ReasonCode.FULL_REFLECTION, ReasonCode.ZERO_RETENTION),
])
def test_wall_pair_decay_preserves_uncomputable_loss_reason(cause: ReasonCode, wrong_cause: ReasonCode) -> None:
    document = _payload().wall_pairs[0].bands[0].model_dump(mode="python")
    document["round_trip_loss_db"]["reason_codes"] = (cause,)
    document["decay_duration_ms"]["reason_codes"] = (cause,)
    valid = WallPairBandRisk.model_validate(document)
    changed = valid.model_dump(mode="python")
    changed["decay_duration_ms"]["reason_codes"] = (wrong_cause,)
    with pytest.raises(ValidationError, match="來回損耗不可計算時持續度須保留同一原因"):
        WallPairBandRisk.model_validate(changed)


def test_payload_rejects_inconsistent_window_units() -> None:
    document = _payload().model_dump(mode="python")
    document["window_upper_s"] = 0.020
    with pytest.raises(ValidationError, match="時間窗 ms 與 s 不一致"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_payload_rejects_nonunit_listening_axis() -> None:
    document = _payload().model_dump(mode="python")
    document["listening_axis_xy"] = (0.0, 2.0)
    with pytest.raises(ValidationError, match="聆聽軸必須是水平單位向量"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_payload_requires_channels_sorted_by_role_and_receiver() -> None:
    document = _payload().model_dump(mode="python")
    document["channels"] = tuple(reversed(document["channels"]))
    with pytest.raises(ValidationError, match="聲道必須按角色、接收點排序"):
        ReflectionsAndEchoPayload.model_validate(document)


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
