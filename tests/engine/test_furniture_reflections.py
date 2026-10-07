"""手組家具反射輸入，驗近似原因、零代價及漏傳家具的大聲失敗。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.physics.reflection_window import ReflectionWindow
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, Flag, ReasonCode, MetricState
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload, CONFIRMED_NO_REFLECTION
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
                      "window_only": "表頭沒有家具，時間窗卻是家具近似", "different_ids": "家具清單不一致"}


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
