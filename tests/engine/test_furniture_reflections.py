"""手組家具反射輸入，驗近似原因、零代價及漏傳家具的大聲失敗。"""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.physics.reflection_window import ReflectionWindow
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, Flag, ReasonCode, MetricState
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload, CONFIRMED_NO_REFLECTION, ZonePoint
from aosr.scoring.reflections_cost import cost_reflections_evaluation
from tests.engine import test_reflections as fixtures
from tests.engine.test_furniture_scoring_flags import furniture_report


def furniture_records(records: tuple[ReflectionInput, ...] | None = None,
                      furniture_id: str = "desk") -> tuple[ReflectionInput, ...]:
    changed = []
    for item in records or fixtures._pair():
        assert item.window is not None
        document = item.window.model_dump(mode="python")
        document.update(coverage="approximate", furniture_ids=(furniture_id,))
        changed.append(replace(item, report=furniture_report(item.report, furniture_id),
                               window=ReflectionWindow.model_validate(document)))
    return tuple(changed)


def reflection_codes(payload: ReflectionsAndEchoPayload) -> set[ReasonCode]:
    codes: set[ReasonCode] = set()
    for channel in payload.channels:
        codes.update(channel.reason_codes)
        for path in channel.reflections:
            codes.update(path.broadband_reason_codes)
        for zone in channel.zones:
            for point in zone.points:
                codes.update(point.strongest_reason_codes)
                codes.update(point.total_energy_db.reason_codes)
        for cell in channel.total_window_energy_db:
            codes.update(cell.reason_codes)
    return codes


def _zero_records(records: tuple[ReflectionInput, ...]) -> tuple[ReflectionInput, ...]:
    changed = []
    for item in records:
        table = item.report.path_table
        assert table is not None and item.window is not None
        rows = tuple(row.model_copy(update={"relative_direct_energy": tuple(
            0.0 for _ in table.frequencies_hz)}) if row.order > 0 else row for row in table.rows)
        window_rows = tuple(row.model_copy(update={"relative_direct_energy": tuple(
            0.0 for _ in table.frequencies_hz)}) for row in item.window.rows)
        changed.append(replace(item, report=item.report.model_copy(update={
            "path_table": table.model_copy(update={"rows": rows})}),
            window=item.window.model_copy(update={"rows": window_rows})))
    return tuple(changed)


@pytest.mark.parametrize("zero", [False, True])
def test_approximate_reflections_keep_absence_distinct_and_cost_identical(zero: bool) -> None:
    records = _zero_records(fixtures._pair()) if zero else fixtures._pair()
    complete = fixtures._evaluate(records)
    approximate = fixtures._evaluate(furniture_records(records))
    assert approximate.state is EvaluationState.MEASURED
    payload = approximate.payload
    assert isinstance(payload, ReflectionsAndEchoPayload)
    assert {channel.coverage for channel in payload.channels} == {"approximate"}
    codes = reflection_codes(payload)
    assert ReasonCode.APPROXIMATE_NO_REFLECTION_IN_ZONE_POINT in codes
    if zero:
        assert ReasonCode.APPROXIMATE_ZERO_REFLECTION_ENERGY in codes
    assert codes.isdisjoint(CONFIRMED_NO_REFLECTION)
    assert {Flag.FURNITURE_MODEL_APPROXIMATE, Flag.FURNITURE_PARALLEL_FLUTTER_NOT_ASSESSED} <= set(approximate.flags)
    assert set(complete.flags).isdisjoint({Flag.FURNITURE_MODEL_APPROXIMATE,
                                         Flag.FURNITURE_PARALLEL_FLUTTER_NOT_ASSESSED})
    registry = load_quality_targets(config_path("quality_targets.toml"))
    purpose = registry.purpose("dedicated_two_channel_listening_room")
    assert cost_reflections_evaluation(complete, purpose, registry.fingerprint).category_cost == (
        cost_reflections_evaluation(approximate, purpose, registry.fingerprint).category_cost)
    assert approximate.settings_fingerprint == complete.settings_fingerprint
    assert approximate.evaluator_version == complete.evaluator_version == "aosr.scoring.reflections.v3"
    assert CategoryEvaluation.model_validate(approximate.model_dump(mode="json")) == approximate


def test_reflection_contract_rejects_mixed_approximate_and_confirmed_absence() -> None:
    document = fixtures._evaluate(furniture_records()).model_dump(mode="python")
    points = [point for zone in document["payload"]["channels"][0]["zones"] for point in zone["points"]]
    point = next(point for point in points if point["strongest_path_index"] is None)
    point["strongest_reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    point["total_energy_db"]["reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    with pytest.raises(ValueError, match="近似.*確認|確認.*近似"):
        CategoryEvaluation.model_validate(document)


# 每種對不上各比自己的原句：主位那支對不上時 payload 驗證器也會擋，只比「家具」分不出是哪一道擋的；
# 周圍點那支不受 payload 驗證器管，只有窗核對擋得住。
_MISMATCH_MESSAGES = {"header_only": "表頭有家具，時間窗卻沒有傳入家具近似",
                      "window_only": "表頭沒有家具，時間窗卻帶家具代號", "different_ids": "家具清單不一致"}


def _surrounding_records() -> tuple[ReflectionInput, ...]:
    return (*fixtures._pair(), fixtures._record("left", 1.3, "s1", receiver_y=2.2),
            fixtures._record("right", 2.5, "s1", receiver_y=2.2))


def _broken(item: ReflectionInput, old: ReflectionInput, mismatch: str) -> ReflectionInput:
    if mismatch == "header_only":
        return replace(item, window=old.window)
    if mismatch == "window_only":
        return replace(item, report=old.report)
    assert item.window is not None
    return replace(item, window=item.window.model_copy(update={"furniture_ids": ("coffee",)}))


@pytest.mark.parametrize("scope", ["every_channel", "surrounding_only"])
@pytest.mark.parametrize("mismatch", sorted(_MISMATCH_MESSAGES))
def test_furniture_header_window_mismatch_raises_instead_of_becoming_unavailable(mismatch: str, scope: str) -> None:
    complete = fixtures._pair() if scope == "every_channel" else _surrounding_records()
    approximate = furniture_records(complete)
    first_broken = 0 if scope == "every_channel" else 2
    records = tuple(_broken(item, old, mismatch) if index >= first_broken else item
                    for index, (item, old) in enumerate(zip(approximate, complete, strict=True)))
    with pytest.raises(ValueError, match=_MISMATCH_MESSAGES[mismatch]):
        fixtures._evaluate(records)


def _unprovable(item: ReflectionInput) -> ReflectionInput:
    assert item.window is not None
    document = item.window.model_dump(mode="python")
    document.update(coverage="not_provable", computed_order_k=SUPPORTED_MAX_ORDER,
                    validation="unvalidated", next_uncomputed_earliest_relative_s=None)
    return replace(item, window=ReflectionWindow.model_validate(document))


@pytest.mark.parametrize(("mismatch", "message"), [
    ("missing_ids", "時間窗與路徑表表頭的家具清單不一致"),
    ("different_ids", "時間窗與路徑表表頭的家具清單不一致"),
    ("window_only", "表頭沒有家具，時間窗卻帶家具代號"),
])
def test_unprovable_window_still_checks_furniture_ids(mismatch: str, message: str) -> None:
    records = furniture_records()
    item = _unprovable(records[0])
    assert item.window is not None
    if mismatch == "window_only":
        item = replace(item, report=fixtures._pair()[0].report)
    else:
        assert item.window is not None
        item = replace(item, window=item.window.model_copy(update={
            "furniture_ids": None if mismatch == "missing_ids" else ("coffee",)}))
    with pytest.raises(ValueError, match=message):
        fixtures._evaluate((item, records[1]))


def test_furniture_unprovable_primary_is_unavailable_without_approximation_flags() -> None:
    records = furniture_records()
    result = fixtures._evaluate((_unprovable(records[0]), records[1]))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.REFLECTION_WINDOW_INCOMPLETE,)
    assert {Flag.FURNITURE_MODEL_APPROXIMATE, Flag.FURNITURE_PARALLEL_FLUTTER_NOT_ASSESSED}.isdisjoint(result.flags)


def test_furniture_unprovable_surrounding_channel_does_not_reject_primary() -> None:
    extra = (fixtures._record("left", 1.3, "s1", receiver_y=2.2),
             fixtures._record("right", 2.5, "s1", receiver_y=2.2))
    records = furniture_records((*fixtures._pair(), *extra))
    result = fixtures._evaluate((*records[:2], _unprovable(records[2]), records[3]))
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    channel = next(item for item in result.payload.channels if (item.role, item.receiver_id) == ("left", "s1"))
    assert channel.state is MetricState.UNAVAILABLE
    assert channel.reason_codes == (ReasonCode.REFLECTION_WINDOW_INCOMPLETE,)


@pytest.mark.parametrize("change, message", [
    ({"strongest_delay_s": None, "strongest_path_index": None}, "零能量原因必須指到路徑"),
    ({"total_energy_db": {"value": -20.0, "state": "measured", "reason_codes": ()}},
     "零能量反射不可寫成已量總能量"),
    ({"strongest_level_db": -20.0, "strongest_state": "measured", "strongest_reason_codes": (),
      "total_energy_db": {"value": None, "state": "unavailable",
                          "reason_codes": ("approximate_no_reflection_in_zone_point",)}},
     "已量最強反射的總能量不可是沒有反射"),
])
def test_zone_point_rejects_incoherent_approximate_absence(change: dict[str, object], message: str) -> None:
    document: dict[str, object] = {"frequency_hz": 500.0, "strongest_level_db": None,
                "strongest_delay_s": 0.005, "strongest_path_index": 0,
                "strongest_state": "unavailable", "strongest_reason_codes": ("approximate_zero_reflection_energy",),
                "total_energy_db": {"value": None, "state": "unavailable",
                                    "reason_codes": ("approximate_zero_reflection_energy",)}}
    document.update(change)
    with pytest.raises(ValueError, match=message):
        ZonePoint.model_validate(document)


_COVERAGE_CODE_MESSAGES = {
    True: "近似聲道不准使用確認沒有反射原因碼",
    False: "非近似聲道不准使用近似沒有反射原因碼",
}


@pytest.mark.parametrize("approximate", [True, False])
@pytest.mark.parametrize("layer", ["strongest", "point_total", "path", "window_total"])
def test_absence_code_matches_channel_coverage_at_each_layer(approximate: bool, layer: str) -> None:
    records = _zero_records(fixtures._pair())
    document = fixtures._evaluate(furniture_records(records) if approximate else records).model_dump(mode="python")
    channel = document["payload"]["channels"][0]
    points = [point for zone in channel["zones"] for point in zone["points"]]
    wrong_empty = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT if approximate
                   else ReasonCode.APPROXIMATE_NO_REFLECTION_IN_ZONE_POINT)
    wrong_zero = ReasonCode.ZERO_REFLECTION_ENERGY if approximate else ReasonCode.APPROXIMATE_ZERO_REFLECTION_ENERGY
    if layer == "strongest":
        point = next(point for point in points if point["strongest_path_index"] is None)
        point["strongest_reason_codes"] = (wrong_empty,)
    elif layer == "point_total":
        point = next(point for point in points if point["strongest_path_index"] is not None)
        point["total_energy_db"]["reason_codes"] = (wrong_zero,)
    elif layer == "path":
        path = next(path for path in channel["reflections"] if path["broadband_level_db"] is None)
        path["broadband_reason_codes"] = (wrong_zero,)
    else:
        cell = next(cell for cell in channel["total_window_energy_db"] if cell["value"] is None)
        cell["reason_codes"] = (wrong_zero,)
    with pytest.raises(ValueError, match=_COVERAGE_CODE_MESSAGES[approximate]):
        CategoryEvaluation.model_validate(document)


def _absence_codes_as(encoded: str, approximate: bool) -> str:
    """考卷手寫兩種沒有的對應，只換原因碼的完整字串。"""
    pairs = (("no_reflection_in_zone_point", "approximate_no_reflection_in_zone_point"),
             ("zero_reflection_energy", "approximate_zero_reflection_energy"))
    for complete, estimated in pairs:
        old, new = (complete, estimated) if approximate else (estimated, complete)
        encoded = encoded.replace(json.dumps(old), json.dumps(new))
    return encoded


@pytest.mark.parametrize("approximate", [True, False])
def test_absence_codes_cannot_all_mislabel_channel_coverage(approximate: bool) -> None:
    records = _zero_records(fixtures._pair())
    result = fixtures._evaluate(furniture_records(records) if approximate else records)
    encoded = _absence_codes_as(result.model_dump_json(), not approximate)
    assert encoded != result.model_dump_json()
    with pytest.raises(ValueError, match=_COVERAGE_CODE_MESSAGES[approximate]):
        CategoryEvaluation.model_validate_json(encoded)


@pytest.mark.parametrize("model", [None, "single_bounce_finite_size_v1"])
@pytest.mark.parametrize("role", ["left", "right"])
def test_furniture_model_checks_each_primary_channel(model: str | None, role: str) -> None:
    records = fixtures._pair()
    document = fixtures._evaluate(furniture_records(records) if model else records).model_dump(mode="json")
    payload = document["payload"]
    payload["furniture_model"] = model
    index = next(index for index, channel in enumerate(payload["channels"]) if channel["role"] == role)
    channel = payload["channels"][index]
    channel["coverage"] = "complete" if model else "approximate"
    payload["channels"][index] = json.loads(_absence_codes_as(json.dumps(channel), model is None))
    with pytest.raises(ValueError, match="家具模型有值若且唯若主位聲道覆蓋是近似"):
        CategoryEvaluation.model_validate(document)


@pytest.mark.parametrize("model, message", [
    (None, "沒有家具模型時，聲道覆蓋不准是近似"),
    ("single_bounce_finite_size_v1", "有家具模型時，聲道覆蓋不准是完整"),
])
def test_furniture_model_checks_surrounding_channel_coverage(model: str | None, message: str) -> None:
    records = _surrounding_records()
    document = fixtures._evaluate(furniture_records(records) if model else records).model_dump(mode="json")
    payload = document["payload"]
    payload["furniture_model"] = model
    index = next(index for index, channel in enumerate(payload["channels"]) if not channel["is_primary"])
    channel = payload["channels"][index]
    channel["coverage"] = "complete" if model else "approximate"
    payload["channels"][index] = json.loads(_absence_codes_as(json.dumps(channel), model is None))
    with pytest.raises(ValueError, match=message):
        CategoryEvaluation.model_validate(document)


def _unavailable_surrounding(furnished: bool, kind: str) -> dict[str, object]:
    """周圍點左聲道做成不可估：不可證明、沒窗、或有家具但沒路徑表；回評估的 JSON 文件。"""
    records = furniture_records(_surrounding_records()) if furnished else _surrounding_records()
    target = records[2]
    if kind == "unprovable":
        broken = _unprovable(target)
    elif kind == "no_window":
        broken = replace(target, window=None)
    else:
        broken = replace(target, report=target.report.model_copy(update={"path_table": None}))
    document: dict[str, object] = fixtures._evaluate((*records[:2], broken, records[3])).model_dump(mode="json")
    return document


def _surrounding_left(document: dict[str, object]) -> dict[str, object]:
    payload = document["payload"]
    assert isinstance(payload, dict)
    channel: dict[str, object] = next(item for item in payload["channels"]
                                      if (item["receiver_id"], item["role"]) == ("s1", "left"))
    assert channel["state"] == "unavailable"
    return channel


# 不可估聲道沒有分區點，綁覆蓋那條規則只剩它自己的原因碼欄可咬。
@pytest.mark.parametrize("furnished, kind", [(False, "unprovable"), (False, "no_window"), (True, "no_path_table")])
def test_unavailable_channel_reason_codes_follow_its_own_coverage(furnished: bool, kind: str) -> None:
    document = _unavailable_surrounding(furnished, kind)
    _surrounding_left(document)["reason_codes"] = [
        "no_reflection_in_zone_point" if furnished else "approximate_no_reflection_in_zone_point"]
    with pytest.raises(ValueError, match=_COVERAGE_CODE_MESSAGES[furnished]):
        CategoryEvaluation.model_validate(document)


@pytest.mark.parametrize("furnished, coverage, message", [
    (True, "complete", "有家具模型時，聲道覆蓋不准是完整"),
    (False, "approximate", "沒有家具模型時，聲道覆蓋不准是近似"),
])
def test_unavailable_surrounding_coverage_follows_furniture_model(furnished: bool, coverage: str, message: str) -> None:
    document = _unavailable_surrounding(furnished, "no_path_table" if furnished else "unprovable")
    _surrounding_left(document)["coverage"] = coverage
    with pytest.raises(ValueError, match=message):
        CategoryEvaluation.model_validate(document)


@pytest.mark.parametrize("model", ["", " ", "garbage-model"])
def test_reflection_payload_rejects_unknown_furniture_model(model: str) -> None:
    document = fixtures._evaluate(furniture_records()).model_dump(mode="python")
    document["payload"]["furniture_model"] = model
    with pytest.raises(ValueError, match="未知家具模型"):
        CategoryEvaluation.model_validate(document)


def test_approximate_channel_without_window_paths_has_zero_cost_and_both_absent() -> None:
    from tests.engine import test_channel_matching_reflections as matching
    from aosr.scoring.channel_matching_reflections_contract import ReflectionAsymmetryState
    changed = []
    for item in fixtures._pair():
        table = item.report.path_table
        assert table is not None and item.window is not None
        direct = next(row.delay_s for row in table.rows if row.order == 0)
        rows = tuple(row.model_copy(update={"delay_s": direct + 0.05}) if row.order > 0 else row
                     for row in table.rows)
        changed.append(replace(item, report=item.report.model_copy(update={
            "path_table": table.model_copy(update={"rows": rows})}),
            window=item.window.model_copy(update={"rows": ()})))
    result = fixtures._evaluate(furniture_records(tuple(changed)))
    assert result.state is EvaluationState.MEASURED
    payload = result.payload
    assert isinstance(payload, ReflectionsAndEchoPayload)
    expected = (ReasonCode.APPROXIMATE_NO_REFLECTION_IN_ZONE_POINT,)
    for channel in payload.channels:
        assert channel.state is MetricState.MEASURED and channel.coverage == "approximate"
        assert channel.reflections and all(not path.within_window for path in channel.reflections)
        assert all(point.strongest_reason_codes == expected and point.total_energy_db.reason_codes == expected
                   for zone in channel.zones for point in zone.points)
        assert all(cell.reason_codes == expected for cell in channel.total_window_energy_db)
    registry = load_quality_targets(config_path("quality_targets.toml"))
    cost = cost_reflections_evaluation(result, registry.purpose("dedicated_two_channel_listening_room"), registry.fingerprint)
    assert cost.category_cost is not None and cost.category_cost.value == 0.0
    diagnosis = matching._diagnosis(result)
    assert diagnosis.state is MetricState.MEASURED and diagnosis.points
    assert all(point.state is ReflectionAsymmetryState.BOTH_ABSENT and point.reason_codes == expected
               for point in diagnosis.points)
    assert not diagnosis.one_sided
