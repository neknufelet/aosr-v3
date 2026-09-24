"""反射輸出契約拒收矛盾狀態與錯誤列身分的考卷。"""
from __future__ import annotations

from copy import deepcopy
import math

import pytest
from pydantic import ValidationError

from aosr.scoring.contract_base import InputProvenance, MetricState, ReasonCode
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
    point = ZonePoint(frequency_hz=1000.0, strongest_level_db=None, strongest_delay_s=None,
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
    bands = (WallPairBandRisk(frequency_hz=1000.0, nominal_center_hz=1000,
                              lower_hz=890.0, upper_hz=1120.0,
                              round_trip_loss_db=unavailable,
                              decay_duration_s=unavailable,
                              room_t20_s=_metric(None, ReasonCode.T20_BAND_UNAVAILABLE)),)
    pairs = tuple(WallPairRisk(walls=walls, round_trip_delay_s=_metric(0.010, None), bands=bands)
                  for walls in (("x0", "xL"), ("y0", "yL"), ("floor", "ceiling")))
    return ReflectionsAndEchoPayload(category="reflections_and_echo", window_upper_ms=15.0,
                                     window_upper_s=0.015, frequency_range_hz=(1000.0, 8000.0),
                                     zone_limits=ZoneLimits(vertical_min_abs_elevation_deg=30.0,
                                     front_max_abs_azimuth_deg=40.0, rear_min_abs_azimuth_deg=135.0),
                                     listening_axis_xy=(0.0, 1.0), listening_axis_rule="stereo_base_bisector_v1",
                                     primary_receiver_id="main", includes_speaker_directivity=False,
                                     channels=channels, wall_pairs=pairs,
                                     flutter_alert_band_centers_hz=(1000,))


@pytest.mark.parametrize(("centers", "message"), [
    ((1000, 1000), "顫動警戒標稱帶名不可重複"),
    ((400,), "顫動警戒標稱帶名須存在於牆對子帶"),
])
def test_alert_band_list_rejects_duplicate_or_absent_wall_band(
    centers: tuple[int, ...], message: str,
) -> None:
    document = _payload().model_dump(mode="python")
    document["flutter_alert_band_centers_hz"] = centers
    with pytest.raises(ValidationError, match=message):
        ReflectionsAndEchoPayload.model_validate(document)


def test_three_wall_pairs_must_share_exact_subband_edges() -> None:
    document = _payload().model_dump(mode="python")
    document["wall_pairs"][0]["bands"][0]["upper_hz"] = 1100.0
    with pytest.raises(ValidationError, match="三對平行牆的子帶清單必須一致"):
        ReflectionsAndEchoPayload.model_validate(document)


def _measured_path(source: ReflectionSource = ReflectionSource.PATH_TABLE,
                   source_index: int = 0) -> ReflectionPath:
    return ReflectionPath(source=source, source_index=source_index, order=1,
                          wall_sequence=("front",), relative_direct_delay_s=0.002,
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
    front["points"][0]["strongest_delay_s"] = 0.002
    front["points"][0]["strongest_path_index"] = 0
    front["points"][0]["strongest_state"] = MetricState.MEASURED
    front["points"][0]["strongest_reason_codes"] = ()
    front["points"][0]["total_energy_db"] = _metric(-12.0, None).model_dump(mode="python")
    document["total_window_energy_db"] = (_metric(-12.0, None).model_dump(mode="python"),)
    return ReflectionChannel.model_validate(document)


def _channel_with_zero_energy_path() -> ReflectionChannel:
    document = _channel_with_paths(_measured_path()).model_dump(mode="python")
    path = document["reflections"][0]
    path["broadband_level_db"] = None
    path["broadband_state"] = MetricState.NOT_COMPUTABLE
    path["broadband_reason_codes"] = (ReasonCode.ZERO_REFLECTION_ENERGY,)
    front = document["zones"][0]["points"][0]
    front["strongest_delay_s"] = 0.002
    front["strongest_path_index"] = 0
    front["strongest_state"] = MetricState.NOT_COMPUTABLE
    front["strongest_reason_codes"] = (ReasonCode.ZERO_REFLECTION_ENERGY,)
    front["total_energy_db"] = _metric(None, ReasonCode.ZERO_REFLECTION_ENERGY).model_dump(mode="python")
    return ReflectionChannel.model_validate(document)


def test_zero_and_missing_states_round_trip_without_sentinel() -> None:
    payload = _payload()
    loaded = ReflectionsAndEchoPayload.model_validate_json(payload.model_dump_json())
    point = loaded.channels[0].zones[0].points[0]
    assert point.strongest_level_db is None
    assert point.strongest_reason_codes == (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    assert point.total_energy_db.value is None


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
                          wall_sequence=("floor",), relative_direct_delay_s=0.002,
                          room_azimuth_deg=0.0, room_elevation_deg=-45.0,
                          listening_azimuth_deg=0.0, listening_elevation_deg=-45.0,
                          zone=DirectionZone.VERTICAL, within_window=True,
                          broadband_level_db=None, broadband_state=MetricState.NOT_COMPUTABLE,
                          broadband_reason_codes=(ReasonCode.ZERO_REFLECTION_ENERGY,))
    assert ReflectionPath.model_validate_json(path.model_dump_json()).room_elevation_deg == -45.0


def test_zero_energy_strongest_retains_path_and_delay() -> None:
    point = ZonePoint(frequency_hz=1000.0, strongest_level_db=None, strongest_delay_s=0.002,
                      strongest_path_index=0, strongest_state=MetricState.NOT_COMPUTABLE,
                      strongest_reason_codes=(ReasonCode.ZERO_REFLECTION_ENERGY,),
                      total_energy_db=_metric(None, ReasonCode.ZERO_REFLECTION_ENERGY))
    loaded = ZonePoint.model_validate_json(point.model_dump_json())
    assert loaded.strongest_state is MetricState.NOT_COMPUTABLE
    assert loaded.strongest_reason_codes == (ReasonCode.ZERO_REFLECTION_ENERGY,)
    assert loaded.strongest_level_db is None
    assert loaded.strongest_path_index == 0
    assert loaded.strongest_delay_s == 0.002
    changed = point.model_dump(mode="python")
    changed["strongest_reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    with pytest.raises(ValidationError, match="零能量路徑必須帶零能量原因碼"):
        ZonePoint.model_validate(changed)


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
    ("strongest_delay_s", 0.002),
    ("strongest_path_index", 0),
])
def test_strongest_delay_and_index_must_appear_together(field: str, value: float | int) -> None:
    document = _payload().channels[0].zones[0].points[0].model_dump(mode="python")
    document[field] = value
    with pytest.raises(ValidationError, match="最強反射的延遲與路徑索引必須同進同出"):
        ZonePoint.model_validate(document)


def test_zero_energy_path_requires_its_specific_reason() -> None:
    valid = ZonePoint(frequency_hz=1000.0, strongest_level_db=None, strongest_delay_s=0.002,
                      strongest_path_index=0, strongest_state=MetricState.NOT_COMPUTABLE,
                      strongest_reason_codes=(ReasonCode.ZERO_REFLECTION_ENERGY,),
                      total_energy_db=_metric(None, ReasonCode.ZERO_REFLECTION_ENERGY))
    document = valid.model_dump(mode="python")
    document["strongest_reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    with pytest.raises(ValidationError, match="零能量路徑必須帶零能量原因碼"):
        ZonePoint.model_validate(document)


@pytest.mark.parametrize("change,message", [
    ("missing", "聲道必須保留四區結果"),
    ("duplicate", "聲道四區結果不可重複"),
])
def test_channel_requires_each_zone_exactly_once(change: str, message: str) -> None:
    document = _payload().channels[0].model_dump(mode="python")
    zones = document["zones"]
    document["zones"] = zones[:-1] if change == "missing" else (*zones, zones[0])
    with pytest.raises(ValidationError, match=message):
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
        WallPairBandRisk(frequency_hz=1000.0, nominal_center_hz=1000,
                         lower_hz=890.0, upper_hz=1120.0,
                         round_trip_loss_db=_metric(0.0, None),
                         decay_duration_s=unavailable, room_t20_s=unavailable)


def test_wall_pair_zero_retention_is_explicitly_not_computable() -> None:
    metric = MetricCell(value=None, state=MetricState.NOT_COMPUTABLE,
                          reason_codes=(ReasonCode.ZERO_RETENTION,))
    band = WallPairBandRisk(frequency_hz=1000.0, nominal_center_hz=1000,
                            lower_hz=890.0, upper_hz=1120.0,
                            round_trip_loss_db=metric,
                            decay_duration_s=metric, room_t20_s=_metric(0.5, None))
    loaded = WallPairBandRisk.model_validate_json(band.model_dump_json())
    assert loaded.decay_duration_s.value is None
    assert loaded.decay_duration_s.state is MetricState.NOT_COMPUTABLE
    assert loaded.decay_duration_s.reason_codes == (ReasonCode.ZERO_RETENTION,)
    changed = band.model_dump(mode="python")
    changed["decay_duration_s"]["value"] = 0.0
    with pytest.raises(ValidationError, match="值／狀態／原因碼不一致"):
        WallPairBandRisk.model_validate(changed)


@pytest.mark.parametrize("cause,wrong_cause", [
    (ReasonCode.ZERO_RETENTION, ReasonCode.FULL_REFLECTION),
    (ReasonCode.FULL_REFLECTION, ReasonCode.ZERO_RETENTION),
])
def test_wall_pair_decay_preserves_uncomputable_loss_reason(cause: ReasonCode, wrong_cause: ReasonCode) -> None:
    document = _payload().wall_pairs[0].bands[0].model_dump(mode="python")
    document["round_trip_loss_db"]["reason_codes"] = (cause,)
    document["decay_duration_s"]["reason_codes"] = (cause,)
    valid = WallPairBandRisk.model_validate(document)
    changed = valid.model_dump(mode="python")
    changed["decay_duration_s"]["reason_codes"] = (wrong_cause,)
    with pytest.raises(ValidationError, match="來回損耗不可計算時持續度須保留同一原因"):
        WallPairBandRisk.model_validate(changed)


def test_payload_rejects_inconsistent_window_units() -> None:
    document = _payload().model_dump(mode="python")
    document["window_upper_s"] = 0.020
    with pytest.raises(ValidationError, match="時間窗 ms 與 s 不一致"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_window_includes_exact_seconds_upper_bound_and_excludes_next_float() -> None:
    document = _payload().model_dump(mode="python")
    document["window_upper_ms"] = 16.1
    document["window_upper_s"] = 16.1 / 1000.0
    path = _measured_path().model_dump(mode="python")
    path["relative_direct_delay_s"] = document["window_upper_s"]
    document["channels"][0]["reflections"] = (path,)
    accepted = ReflectionsAndEchoPayload.model_validate(document)
    assert accepted.channels[0].reflections[0].within_window
    document["channels"][0]["reflections"][0]["relative_direct_delay_s"] = math.nextafter(
        document["window_upper_s"], math.inf
    )
    document["channels"][0]["reflections"][0]["within_window"] = False
    excluded = ReflectionsAndEchoPayload.model_validate(document)
    assert not excluded.channels[0].reflections[0].within_window
    document["channels"][0]["reflections"][0]["within_window"] = True
    with pytest.raises(ValidationError, match="反射路徑窗內旗標與延遲不一致"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_window_rejects_one_float_difference_between_ms_and_s() -> None:
    document = _payload().model_dump(mode="python")
    document["window_upper_ms"] = 16.1
    exact_seconds = 16.1 / 1000.0
    document["window_upper_s"] = math.nextafter(exact_seconds, math.inf)
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
    document["channels"][0]["state"] = MetricState.UNAVAILABLE
    document["channels"][0]["reason_codes"] = (ReasonCode.REFLECTION_WINDOW_INCOMPLETE,)
    with pytest.raises(ValidationError, match="主位時間窗未蓋滿"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_primary_unavailable_channel_invalidates_category() -> None:
    document = _payload().model_dump(mode="python")
    document["channels"][0]["state"] = MetricState.UNAVAILABLE
    document["channels"][0]["reason_codes"] = (ReasonCode.PATH_TABLE_MISSING,)
    with pytest.raises(ValidationError, match="主位每支聲道都必須是已量"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_all_channels_unavailable_has_own_error() -> None:
    document = _payload().model_dump(mode="python")
    for channel in document["channels"]:
        channel["state"] = MetricState.UNAVAILABLE
        channel["reason_codes"] = (ReasonCode.PATH_TABLE_MISSING,)
    with pytest.raises(ValidationError, match="全部聲道不可估"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_missing_primary_receiver_has_own_error() -> None:
    document = _payload().model_dump(mode="python")
    document["primary_receiver_id"] = "absent"
    with pytest.raises(ValidationError, match="主位接收點不存在"):
        ReflectionsAndEchoPayload.model_validate(document)


@pytest.mark.parametrize("field,value,message", [
    ("listening_azimuth_deg", 90.0, "反射路徑分區與角度不一致"),
    ("listening_elevation_deg", 45.0, "反射路徑分區與角度不一致"),
    ("within_window", False, "反射路徑窗內旗標與延遲不一致"),
])
def test_payload_checks_path_zone_and_window(field: str, value: float | bool, message: str) -> None:
    document = _payload().model_dump(mode="python")
    document["channels"][0]["reflections"] = (_measured_path().model_dump(mode="python"),)
    document["channels"][0]["reflections"][0][field] = value
    with pytest.raises(ValidationError, match=message):
        ReflectionsAndEchoPayload.model_validate(document)


def test_payload_window_uses_its_own_upper_bound() -> None:
    document = _payload().model_dump(mode="python")
    document["channels"][0]["reflections"] = (_measured_path().model_dump(mode="python"),)
    document["channels"][0]["reflections"][0]["relative_direct_delay_s"] = 0.015
    assert ReflectionsAndEchoPayload.model_validate(document).channels[0].reflections[0].within_window
    document["window_upper_ms"] = 1.0
    document["window_upper_s"] = 0.001
    with pytest.raises(ValidationError, match="反射路徑窗內旗標與延遲不一致"):
        ReflectionsAndEchoPayload.model_validate(document)


@pytest.mark.parametrize("change,message", [
    ("delay", "最強反射延遲必須等於所指路徑"),
    ("missing_energy", "沒有窗內反射時總能量不可已量"),
    ("zero_energy", "零能量反射不可寫成已量總能量"),
    ("measured_missing", "已量最強反射的總能量不可是沒有反射"),
    ("no_paths", "沒有反射路徑時窗內總能量不可已量"),
    ("index", "最強反射路徑索引必須指向同區窗內路徑"),
])
def test_channel_rejects_inconsistent_energy_or_strongest(change: str, message: str) -> None:
    if change in {"missing_energy", "no_paths"}:
        document = _payload().channels[0].model_dump(mode="python")
    elif change == "zero_energy":
        document = _channel_with_zero_energy_path().model_dump(mode="python")
    else:
        document = _channel_with_strongest_path().model_dump(mode="python")
    front = document["zones"][0]["points"][0]
    if change == "delay":
        front["strongest_delay_s"] = math.nextafter(0.002, math.inf)
    elif change == "missing_energy":
        front["total_energy_db"] = _metric(0.0, None).model_dump(mode="python")
    elif change == "zero_energy":
        front["total_energy_db"] = _metric(0.0, None).model_dump(mode="python")
    elif change == "measured_missing":
        front["total_energy_db"] = _metric().model_dump(mode="python")
    elif change == "no_paths":
        document["total_window_energy_db"] = (_metric(0.0, None).model_dump(mode="python"),)
    else:
        front["strongest_path_index"] = 100
    with pytest.raises(ValidationError, match=message):
        ReflectionChannel.model_validate(document)


@pytest.mark.parametrize("change,message", [
    ("validated_order", "驗證階數不得超過數值保證階數"),
    ("incomplete_measured", "覆蓋未完成的聲道不可已量"),
    ("table_order", "主表反射階數不得超過主報表階數"),
    ("extension_order", "補算反射階數必須高於主報表且不超過補算階數"),
])
def test_channel_checks_validation_coverage_and_source_orders(change: str, message: str) -> None:
    document = _channel_with_paths(_measured_path()).model_dump(mode="python")
    if change == "validated_order":
        document["computed_order_k"] = 4
    elif change == "incomplete_measured":
        document["coverage"] = "not_provable"
    elif change == "table_order":
        document["report_order_k"] = 0
    else:
        document["reflections"][0]["source"] = ReflectionSource.WINDOW_EXTENSION
    with pytest.raises(ValidationError, match=message):
        ReflectionChannel.model_validate(document)


def test_payload_rejects_frequency_outside_declared_range() -> None:
    document = _payload().model_dump(mode="python")
    document["frequency_range_hz"] = (2000.0, 8000.0)
    with pytest.raises(ValidationError, match="逐點頻率必須落在頻率範圍內"):
        ReflectionsAndEchoPayload.model_validate(document)


@pytest.mark.parametrize("field,value,message", [
    ("source_index", -1, "greater_than_equal"),
    ("order", 0, "greater_than_equal"),
    ("wall_sequence", (), "too_short"),
    ("relative_direct_delay_s", -1.0, "greater_than_equal"),
])
def test_path_rejects_each_field_bound(field: str, value: object, message: str) -> None:
    document = _measured_path().model_dump(mode="python")
    document[field] = value
    with pytest.raises(ValidationError, match=message):
        ReflectionPath.model_validate(document)


@pytest.mark.parametrize("field,value,message", [
    ("frequency_hz", 0.0, "greater_than"),
    ("strongest_delay_s", -1.0, "greater_than_equal"),
    ("strongest_path_index", -1, "greater_than_equal"),
])
def test_zone_point_rejects_each_field_bound(field: str, value: object, message: str) -> None:
    document = _payload().channels[0].zones[0].points[0].model_dump(mode="python")
    document[field] = value
    with pytest.raises(ValidationError, match=message):
        ZonePoint.model_validate(document)


@pytest.mark.parametrize("field,value,message", [
    ("role", "", "string_too_short"),
    ("role", "UPPER", "string_pattern_mismatch"),
    ("speaker_id", "", "string_too_short"),
    ("receiver_id", "", "string_too_short"),
    ("report_order_k", -1, "greater_than_equal"),
    ("computed_order_k", -1, "greater_than_equal"),
])
def test_channel_rejects_each_field_bound(field: str, value: object, message: str) -> None:
    document = _payload().channels[0].model_dump(mode="python")
    document[field] = value
    with pytest.raises(ValidationError, match=message):
        ReflectionChannel.model_validate(document)


@pytest.mark.parametrize("field,value,message", [
    ("window_upper_ms", 0.0, "greater_than"),
    ("window_upper_s", 0.0, "greater_than"),
    ("frequency_range_hz", (0.0, 8000.0), "greater_than"),
    ("frequency_range_hz", (1000.0, 0.0), "greater_than"),
    ("listening_axis_rule", "", "string_too_short"),
    ("primary_receiver_id", "", "string_too_short"),
    ("channels", (), "too_short"),
    ("includes_speaker_directivity", True, "literal_error"),
    ("category", "wrong", "literal_error"),
])
def test_payload_rejects_each_field_bound_or_literal(field: str, value: object, message: str) -> None:
    document = _payload().model_dump(mode="python")
    document[field] = value
    with pytest.raises(ValidationError, match=message):
        ReflectionsAndEchoPayload.model_validate(document)


@pytest.mark.parametrize("change,message", [
    ("state", "聲道狀態與原因碼不一致"),
    ("order", "補算階數不得小於主報表階數"),
    ("zone_axis", "同一聲道四區頻率軸必須一致"),
])
def test_channel_rejects_remaining_consistency_errors(change: str, message: str) -> None:
    document = _payload().channels[0].model_dump(mode="python")
    if change == "state":
        document["reason_codes"] = (ReasonCode.PATH_TABLE_MISSING,)
    elif change == "order":
        document["computed_order_k"] = 0
    else:
        document["zones"][0]["points"][0]["frequency_hz"] = 2000.0
    with pytest.raises(ValidationError, match=message):
        ReflectionChannel.model_validate(document)


def test_zone_point_rejects_level_without_path() -> None:
    document = _payload().channels[0].zones[0].points[0].model_dump(mode="python")
    document["strongest_level_db"] = -12.0
    document["strongest_state"] = MetricState.MEASURED
    document["strongest_reason_codes"] = ()
    with pytest.raises(ValidationError, match="最強反射有聲級時必須有延遲與路徑索引"):
        ZonePoint.model_validate(document)


def test_zone_point_rejects_zero_energy_without_path() -> None:
    document = _payload().channels[0].zones[0].points[0].model_dump(mode="python")
    document["strongest_state"] = MetricState.NOT_COMPUTABLE
    document["strongest_reason_codes"] = (ReasonCode.ZERO_REFLECTION_ENERGY,)
    with pytest.raises(ValidationError, match="零能量原因必須指到路徑"):
        ZonePoint.model_validate(document)


def test_path_rejects_broadband_level_state_mismatch() -> None:
    document = _measured_path().model_dump(mode="python")
    document["broadband_state"] = MetricState.UNAVAILABLE
    with pytest.raises(ValidationError, match="值／狀態／原因碼不一致"):
        ReflectionPath.model_validate(document)


def test_zone_point_rejects_measured_state_without_level() -> None:
    document = _payload().channels[0].zones[0].points[0].model_dump(mode="python")
    document["strongest_state"] = MetricState.MEASURED
    with pytest.raises(ValidationError, match="值／狀態／原因碼不一致"):
        ZonePoint.model_validate(document)


def test_zone_result_rejects_descending_frequencies() -> None:
    document = _payload().channels[0].zones[0].model_dump(mode="python")
    earlier = deepcopy(document["points"][0])
    later = deepcopy(earlier)
    earlier["frequency_hz"] = 2000.0
    document["points"] = (earlier, later)
    with pytest.raises(ValidationError, match="頻率必須遞增"):
        ZoneResult.model_validate(document)


def test_extension_path_cannot_exceed_computed_order() -> None:
    document = _channel_with_paths(_measured_path()).model_dump(mode="python")
    document["reflections"][0]["source"] = ReflectionSource.WINDOW_EXTENSION
    document["reflections"][0]["order"] = 3
    document["reflections"][0]["wall_sequence"] = ("front", "side", "rear")
    with pytest.raises(ValidationError, match="補算反射階數必須高於主報表且不超過補算階數"):
        ReflectionChannel.model_validate(document)


@pytest.mark.parametrize("field,value,message", [
    ("frequency_hz", 0.0, "greater_than"),
    ("decay_duration_s", 0.0, "衰減持續度必須為正"),
    ("room_t20_s", 0.0, "本房 t20_s 必須為正"),
])
def test_wall_band_rejects_nonphysical_values(field: str, value: float, message: str) -> None:
    document = _payload().wall_pairs[0].bands[0].model_dump(mode="python")
    if field == "frequency_hz":
        document[field] = value
    else:
        document[field] = _metric(value, None).model_dump(mode="python")
    with pytest.raises(ValidationError, match=message):
        WallPairBandRisk.model_validate(document)


@pytest.mark.parametrize("change,message", [
    ("walls", "兩面牆必須相異"),
    ("band_axis", "牆對頻率必須遞增"),
    ("delay", "來回延遲必須為正"),
])
def test_wall_pair_rejects_invalid_walls_bands_or_delay(change: str, message: str) -> None:
    document = _payload().wall_pairs[0].model_dump(mode="python")
    if change == "walls":
        document["walls"] = ("x0", "x0")
    elif change == "band_axis":
        document["bands"] = (document["bands"][0], document["bands"][0])
    else:
        document["round_trip_delay_s"] = _metric(-1.0, None).model_dump(mode="python")
    with pytest.raises(ValidationError, match=message):
        WallPairRisk.model_validate(document)


@pytest.mark.parametrize("change,message", [
    ("range", "頻率範圍必須遞增"),
    ("speaker", "每個角色必須固定對應一支不同喇叭"),
    ("pair_duplicate", "平行牆對不可重複"),
    ("pair_frequency", "三對平行牆的子帶清單必須一致"),
])
def test_payload_rejects_remaining_identity_and_band_errors(change: str, message: str) -> None:
    document = _payload().model_dump(mode="python")
    if change == "range":
        document["frequency_range_hz"] = (8000.0, 1000.0)
    elif change == "speaker":
        document["channels"][1]["speaker_id"] = "left"
        document["channels"][1]["provenance"]["speaker_id"] = "left"
    elif change == "pair_duplicate":
        document["wall_pairs"] = (*document["wall_pairs"], document["wall_pairs"][0])
    else:
        document["wall_pairs"][0]["bands"][0]["nominal_center_hz"] = 2000
    with pytest.raises(ValidationError, match=message):
        ReflectionsAndEchoPayload.model_validate(document)


def test_payload_rejects_surrounding_measured_axis_mismatch() -> None:
    document = _payload().model_dump(mode="python")
    surrounding = deepcopy(document["channels"][0])
    surrounding["receiver_id"] = "near"
    surrounding["provenance"]["receiver_id"] = "near"
    surrounding["is_primary"] = False
    for zone in surrounding["zones"]:
        zone["points"][0]["frequency_hz"] = 2000.0
    peer = deepcopy(document["channels"][1])
    peer["receiver_id"] = "near"
    peer["provenance"]["receiver_id"] = "near"
    peer["is_primary"] = False
    document["channels"] = (document["channels"][0], surrounding, document["channels"][1], peer)
    with pytest.raises(ValidationError, match="各聲道逐點頻率必須一致"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_channel_rejects_measured_total_without_any_in_window_path() -> None:
    document = _channel_with_paths(_measured_path()).model_dump(mode="python")
    document["reflections"][0]["within_window"] = False
    document["total_window_energy_db"] = (_metric(0.0, None).model_dump(mode="python"),)
    with pytest.raises(ValidationError, match="沒有反射路徑時窗內總能量不可已量"):
        ReflectionChannel.model_validate(document)


def _payload_with_surrounding_channels() -> ReflectionsAndEchoPayload:
    document = _payload().model_dump(mode="python")
    surrounding = []
    for channel in document["channels"]:
        peer = deepcopy(channel)
        peer["receiver_id"] = "near"
        peer["provenance"]["receiver_id"] = "near"
        peer["is_primary"] = False
        surrounding.append(peer)
    document["channels"] = (document["channels"][0], surrounding[0], document["channels"][1], surrounding[1])
    return ReflectionsAndEchoPayload.model_validate(document)


def test_payload_rejects_role_switching_speaker_at_surrounding_point() -> None:
    document = _payload_with_surrounding_channels().model_dump(mode="python")
    document["channels"][1]["speaker_id"] = "other"
    document["channels"][1]["provenance"]["speaker_id"] = "other"
    with pytest.raises(ValidationError, match="每個角色必須固定對應一支喇叭"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_payload_rejects_surrounding_point_masquerading_as_primary() -> None:
    document = _payload_with_surrounding_channels().model_dump(mode="python")
    document["channels"][1]["is_primary"] = True
    with pytest.raises(ValidationError, match="主位旗標必須對應主位接收點"):
        ReflectionsAndEchoPayload.model_validate(document)


def test_metric_cell_rejects_measured_value_with_reason() -> None:
    document = _metric(1.0, None).model_dump(mode="python")
    document["reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    with pytest.raises(ValidationError, match="值／狀態／原因碼不一致"):
        MetricCell.model_validate(document)


@pytest.mark.parametrize("field", ["coverage", "validation"])
def test_channel_rejects_unknown_status_literal(field: str) -> None:
    document = _payload().channels[0].model_dump(mode="python")
    document[field] = "unknown"
    with pytest.raises(ValidationError, match="literal_error"):
        ReflectionChannel.model_validate(document)


@pytest.mark.parametrize("field", ["source", "zone"])
def test_path_rejects_unknown_enum(field: str) -> None:
    document = _measured_path().model_dump(mode="python")
    document[field] = "unknown"
    with pytest.raises(ValidationError, match="enum"):
        ReflectionPath.model_validate(document)


def test_wall_pair_requires_two_wall_names() -> None:
    document = _payload().wall_pairs[0].model_dump(mode="python")
    document["walls"] = ("x0",)
    with pytest.raises(ValidationError, match="missing"):
        WallPairRisk.model_validate(document)
