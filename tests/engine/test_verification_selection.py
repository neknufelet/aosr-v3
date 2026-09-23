"""#435 驗證名單：榜首、淘汰線兩側、抽查與預算。"""

from __future__ import annotations

import math
from random import Random

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import (
    QualityTargets,
    SettingEntry,
    TargetEntry,
    load_quality_targets,
)
from aosr.scoring.category_registry import EliminationReason
from aosr.scoring.contract import CandidateEvaluation, CategoryEvaluation, ChannelMatchingPayload
from aosr.scoring.ranking import ExternalAcceptance, ExternalFloors
from aosr.scoring.verification_selection import (
    VerificationLine,
    VerificationRules,
    VerificationSelection,
    _certain_failures,
    _measurements,
    _near_reasons,
    _rules,
    select_for_verification,
)
from tests.engine.test_channel_matching_cost import _measured as _channel_measured
from tests.engine.test_listening_area_cost import _measured as _area_measured
from tests.engine.test_scoring_ranking import _candidate, _rank, _timbre, _unavailable


_REGISTRY = load_quality_targets(config_path("quality_targets.toml"))
_PURPOSE = _REGISTRY.purpose("dedicated_two_channel_listening_room")


def _area(
    candidate_id: str, *, ripple: float = 0.0, level: float = 0.0
) -> CategoryEvaluation:
    measured = _area_measured(ripple_worst=ripple, level_worst=level)
    payload = measured.payload
    assert payload is not None
    return measured.model_copy(
        update={
            "candidate_id": candidate_id,
            "payload": payload.model_copy(update={"candidate_id": candidate_id}),
        }
    )


def _item(
    candidate_id: str,
    *,
    ripple: float = 0.0,
    level: float = 0.0,
    tilt: float = 0.0,
    alert: bool = False,
) -> CandidateEvaluation:
    features: list[dict[str, object]] | None = None
    if alert:
        features = [
            {
                "kind": "dip",
                "center_frequency_hz": 60.0,
                "depth_db": -16.0,
                "width_octave": None,
                "flags": ["feature_boundary_incomplete"],
            }
        ]
    return _candidate(
        _timbre(candidate_id, tilt=tilt, features=features),
        _area(candidate_id, ripple=ripple, level=level),
    )


def _select(
    *items: CandidateEvaluation,
    spot_check_count: int | None = None,
    budget_candidates: int | None = None,
    seed: int = 7,
    external: ExternalFloors | None = None,
) -> VerificationSelection:
    ranked = _rank(*items, registry=_REGISTRY, external=external)
    return select_for_verification(
        ranked, _REGISTRY, spot_check_count, budget_candidates, seed
    )


def test_registry_keeps_every_observed_shift_and_choice_with_units() -> None:
    expected = {
        "listening_area_stability.tilt_worst_deviation.primary_to_surrounding": (
            0.06400451504130211,
            "dB/oct",
        ),
        "listening_area_stability.tilt_worst_deviation.surrounding_to_surrounding": (
            0.1204947100713113,
            "dB/oct",
        ),
        "listening_area_stability.ripple_rms_worst_deviation.primary_to_surrounding": (
            0.2918352138463858,
            "dB",
        ),
        "listening_area_stability.ripple_rms_worst_deviation.surrounding_to_surrounding": (
            0.4075937466013331,
            "dB",
        ),
        "listening_area_stability.overall_level_worst_deviation.primary_to_surrounding": (
            0.18436084752146975,
            "dB",
        ),
        "listening_area_stability.overall_level_worst_deviation.surrounding_to_surrounding": (
            0.2747873372372851,
            "dB",
        ),
        "channel_matching.tilt_difference_worst": (0.040077746272799364, "dB/oct"),
        "channel_matching.ripple_rms_difference_worst": (0.1859825252960885, "dB"),
        "channel_matching.broadband_level_difference_worst": (
            0.17677849839578919,
            "dB",
        ),
        "channel_matching.direct_time_difference_worst": (0.0, "ms"),
    }
    for suffix, (value, unit) in expected.items():
        entry = _PURPOSE.entry(f"verification.observed_axis_shift.{suffix}")
        assert isinstance(entry, SettingEntry)
        assert (entry.value, entry.unit, entry.status, entry.source_kind) == (
            value,
            unit,
            "baseline",
            "engineering_recommendation",
        )
        assert "435-measure 目錄 evaluate 輸出" in entry.source
    for key, value in (
        ("verification.range_multiplier", 2.0),
        ("verification.top_n", 5),
    ):
        entry = _PURPOSE.entry(key)
        assert isinstance(entry, SettingEntry)
        assert (entry.value, entry.unit, entry.status, entry.source_kind) == (
            value,
            "1",
            "baseline",
            "product_choice",
        )


def test_top_n_enter_with_or_without_alert_and_precede_near_line() -> None:
    top = [
        _item(f"top-{index}", tilt=index * 0.01, alert=index == 0) for index in range(5)
    ]
    near = _item("near", ripple=3.0, tilt=2.0)
    ranked = _rank(*top, near, registry=_REGISTRY)
    assert ranked.rankable[0].review_alerts
    assert not ranked.rankable[1].review_alerts
    selected = _select(*top, near)
    n = int(selected.rules.top_n.value)
    assert [item.candidate_id for item in selected.candidates[:n]] == [
        row.candidate_id for row in ranked.rankable[:n]
    ]
    assert all(item.reasons[0].kind == "top_n" for item in selected.candidates[:n])


def test_top_n_keeps_its_near_line_detail_as_well() -> None:
    items = [
        _item(f"candidate-{index}", ripple=3.0 if index == 0 else 0.0)
        for index in range(5)
    ]
    selection = _select(*items)
    marked = next(
        item for item in selection.candidates if item.candidate_id == "candidate-0"
    )
    assert {reason.kind for reason in marked.reasons} == {"top_n", "near_line"}
    assert any(reason.line_key is not None for reason in marked.reasons)


def test_both_sides_enter_and_raw_shift_is_multiplied_without_rounding() -> None:
    rule = _PURPOSE.entry(
        "verification.observed_axis_shift.listening_area_stability.ripple_rms_worst_deviation.primary_to_surrounding"
    )
    multiplier = _PURPOSE.entry("verification.range_multiplier")
    limit = _PURPOSE.entry("listening_area_stability.ripple_rms_worst_deviation")
    assert isinstance(rule, SettingEntry) and isinstance(multiplier, SettingEntry)
    assert isinstance(limit, TargetEntry)
    assert isinstance(rule.value, int | float)
    assert isinstance(multiplier.value, int | float)
    assert isinstance(limit.value, int | float)
    margin = float(rule.value) * float(multiplier.value)
    center = float(limit.value)
    top = [_item(f"top-{index}", tilt=index * 0.01) for index in range(5)]
    selection = _select(
        *top,
        _item("inside-pass", ripple=center - margin, tilt=3.0),
        _item("inside-fail", ripple=center + margin, tilt=3.0),
        _item("outside-pass", ripple=center - margin * 1.01, tilt=3.0),
        _item("outside-fail", ripple=center + margin * 1.01, tilt=3.0),
    )
    by_id = {item.candidate_id: item for item in selection.candidates}
    for candidate_id in ("inside-pass", "inside-fail"):
        near = next(
            reason
            for reason in by_id[candidate_id].reasons
            if reason.kind == "near_line"
        )
        assert near.margin == margin
        assert near.limit == center
    assert "outside-pass" not in by_id
    assert "outside-fail" not in by_id
    assert {item.candidate_id for item in selection.not_added} >= {
        "outside-pass",
        "outside-fail",
    }


def test_certain_failure_and_external_floor_are_explained() -> None:
    top = [_item(f"top-{index}", tilt=index * 0.01) for index in range(5)]
    near_and_far = _item("near-and-far", ripple=3.1, level=9.0, tilt=3.0)
    # 只有近線（起伏 3.1 dB 貼著 3 dB 的線）、沒有別的遠超過線的原因：只剩外部底線淘汰能把它排除。
    # 原本同時帶一個遠超過線的量，拿掉外部底線那一條判斷也照樣被排除，考卷咬不到（主對話突變抓到）。
    external = _item("external", ripple=3.1)
    selection = _select(
        *top,
        near_and_far,
        external,
        external=ExternalFloors(
            declared_by="fixture", verdicts={"external": ExternalAcceptance.FAILED}
        ),
    )
    excluded = {item.candidate_id: item for item in selection.not_added}
    assert "near-and-far" in excluded and "external" in excluded
    assert any(
        reason.startswith("listening_area_level_primary_to_surrounding_worst_beyond_limit")
        for reason in excluded["near-and-far"].reasons
    )
    # 名單上的排除原因要直說是外部底線淘汰，不是「沒有可核對的原始最差值」那種含糊的說法。
    assert excluded["external"].reasons == ("external_floor_failed",)


def test_spot_check_pending_then_seeded_from_outside_pool() -> None:
    items = [_item(f"candidate-{index:02}", tilt=index * 0.01) for index in range(12)]
    pending = _select(*items)
    assert pending.rules.spot_check_count is None
    assert pending.rules.spot_check_note == "抽查數量待定，這一輪沒有抽查"
    assert all(item.reasons[0].kind != "spot_check" for item in pending.candidates)
    sampled = _select(*items, spot_check_count=3, seed=7)
    repeated = _select(*items, spot_check_count=3, seed=7)
    changed = _select(*items, spot_check_count=3, seed=11)
    drawn = [
        item.candidate_id
        for item in sampled.candidates
        if item.reasons[0].kind == "spot_check"
    ]
    pool = sorted(set(item.candidate_id for item in pending.not_added))
    assert drawn == Random(7).sample(pool, 3)
    assert drawn == [
        item.candidate_id
        for item in repeated.candidates
        if item.reasons[0].kind == "spot_check"
    ]
    assert drawn != [
        item.candidate_id
        for item in changed.candidates
        if item.reasons[0].kind == "spot_check"
    ]


def test_budget_marks_remainder_and_counts_unconsidered() -> None:
    top = [_item(f"top-{index}", tilt=index * 0.01) for index in range(5)]
    near = _item("near", ripple=3.0, tilt=3.0)
    missing = _candidate(_unavailable("missing", "timbre_balance", ("missing_points",)))
    different = _candidate(
        _timbre("different", settings_fingerprint="other-settings"), _area("different")
    )
    full = _select(*top, near, missing, different)
    assert full.rules.budget_note == "預算待定，沒有截斷"
    assert all(item.status == "scheduled" for item in full.candidates)
    limited = _select(*top, near, missing, different, budget_candidates=2)
    assert [item.status for item in limited.candidates] == [
        "scheduled" if index < 2 else "not_verified_over_budget"
        for index in range(len(limited.candidates))
    ]
    assert limited.not_evaluated_count == len(
        _rank(*top, near, missing, different, registry=_REGISTRY).not_evaluated
    )
    assert limited.not_comparable_count == len(
        _rank(*top, near, missing, different, registry=_REGISTRY).not_comparable.rows
    )


def test_rules_must_come_from_the_same_registry_as_the_ranking() -> None:
    """拿另一份登記簿（任一條不同，整份指紋就不同）去挑這一份排名的人要報錯；
    名單記下這一輪用的登記簿指紋。拿掉指紋比對這題會紅（施工席提的疑慮）。"""
    top = tuple(_item(f"top-{index}") for index in range(5))
    ranked = _rank(*top, registry=_REGISTRY)
    document = _REGISTRY.model_dump(mode="json", by_alias=True)
    for purpose in document["purpose"]:
        for setting in purpose["setting"]:
            if setting["key"] == "verification.range_multiplier":
                setting["value"] = 3.0
    other = QualityTargets.model_validate(document)

    with pytest.raises(ValueError, match="指紋"):
        select_for_verification(ranked, other, None, None, 7)
    selection = select_for_verification(ranked, _REGISTRY, None, None, 7)
    assert selection.rules.registry_fingerprint == _REGISTRY.fingerprint
    assert selection.rules.registry_fingerprint == ranked.header.registry_fingerprint


def _rules_now() -> VerificationRules:
    return _rules(_PURPOSE, _REGISTRY.fingerprint, None, None, 7)


def _line_for(reason: EliminationReason) -> VerificationLine:
    return next(line for line in _rules_now().lines if line.elimination_reason is reason)


def test_zero_range_line_is_never_near_but_over_the_line_is_certain() -> None:
    """直達聲時間差只看距離與聲速、換軸不變（範圍 0）：剛好等於線不算近線、不追加；超過線就是確定不合格。
    把「範圍 0 也算近線」接回去這題會紅（找碴席抓到：剛好等於線的候選會被追加、還擠掉真正要驗的人）。"""
    reason = EliminationReason.CHANNEL_MATCHING_DIRECT_TIME_WORST_BEYOND_LIMIT
    line = _line_for(reason)
    assert line.margin == 0.0
    at_limit = {reason: (line.limit.value, line)}
    just_over = {reason: (math.nextafter(line.limit.value, math.inf), line)}

    assert _near_reasons(at_limit) == ()
    assert _certain_failures((reason,), just_over)


def test_rules_snapshot_records_the_direct_time_switch() -> None:
    """直達聲時間差開關會影響選人，這一輪的規則要記下它的值與出處（找碴席抓到漏記）。"""
    entry = _PURPOSE.entry("channel_matching.direct_time_cost_enabled")
    assert isinstance(entry, SettingEntry)
    snapshot = _rules_now().direct_time_cost_enabled

    assert (snapshot.key, snapshot.value, snapshot.source) == (entry.key, entry.value, entry.source)


def test_channel_and_peer_lines_read_their_own_worst_values() -> None:
    """聲道匹配四條線、聽音區「周圍彼此」那一組，各自讀到自己那一格的最差值；
    讓聲道匹配那一支永遠不讀、或把周圍彼此那一組當成不存在，這題會紅（找碴席抓到的盲點）。"""
    area = _area_measured(
        tilt_worst=0.2, ripple_worst=0.3, level_worst=0.4,
        peer_means=(0.1, 0.1, 0.1), peer_worsts=(0.7, 1.1, 1.3),
    )
    base = _channel_measured(direct_time_cost_enabled=True)
    assert isinstance(base.payload, ChannelMatchingPayload)
    aggregate = base.payload.aggregates[0]
    worsts = {"tilt_difference": 1.4, "ripple_rms_difference": 2.9, "broadband_level_difference": 3.8, "direct_time_difference": 0.6}
    changed = aggregate.model_copy(
        update={
            name: getattr(aggregate, name).model_copy(update={"worst_absolute_difference": value})
            for name, value in worsts.items()
        }
    )
    channel = base.model_copy(
        update={"payload": base.payload.model_copy(update={"aggregates": (changed,)})}
    )

    measured = _measurements((area, channel), _rules_now())

    expected = {
        EliminationReason.LISTENING_AREA_TILT_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT: 0.7,
        EliminationReason.LISTENING_AREA_RIPPLE_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT: 1.1,
        EliminationReason.LISTENING_AREA_LEVEL_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT: 1.3,
        EliminationReason.CHANNEL_MATCHING_TILT_WORST_BEYOND_LIMIT: 1.4,
        EliminationReason.CHANNEL_MATCHING_RIPPLE_WORST_BEYOND_LIMIT: 2.9,
        EliminationReason.CHANNEL_MATCHING_LEVEL_WORST_BEYOND_LIMIT: 3.8,
        EliminationReason.CHANNEL_MATCHING_DIRECT_TIME_WORST_BEYOND_LIMIT: 0.6,
    }
    assert {reason: measured[reason][0] for reason in expected} == expected


def test_closer_near_candidate_goes_first_and_budget_cuts_the_farther_one() -> None:
    """兩個近線候選：離線近的排前面，預算只剩一個位置時截掉遠的那個；
    近線順序排反（遠的先）這題會紅（找碴席抓到的盲點）。"""
    top = [_item(f"top-{index}", tilt=index * 0.01) for index in range(5)]
    close = _item("near-close", ripple=3.05)
    far = _item("near-far", ripple=3.4)

    selection = _select(*top, close, far, budget_candidates=len(top) + 1)

    status = {item.candidate_id: item.status for item in selection.candidates}
    assert status["near-close"] == "scheduled"
    assert status["near-far"] == "not_verified_over_budget"


def test_spot_checks_draw_only_from_outside_the_near_and_excluded_candidates() -> None:
    """抽查只抽「不是前 N、不是近線、也沒被確定不合格排除」的候選：池子剛好三個、抽三個，
    抽到的就是那三個；近線的不重複入列、被外部淘汰的不被抽中。把近線或已排除的混進抽查池，這題會紅（找碴席抓到）。"""
    top = [_item(f"top-{index}", tilt=index * 0.01) for index in range(5)]
    near = _item("near", ripple=3.1)
    external = _item("external", ripple=3.1)
    # 這組考卷工具給的候選總代價都一樣，名次照代號排；代號用 z- 開頭才會排在 top-… 後面，不會變成前五名。
    safe = [_item(f"z-safe-{index}") for index in range(3)]

    selection = _select(
        *top, near, external, *safe,
        spot_check_count=len(safe),
        external=ExternalFloors(declared_by="fixture", verdicts={"external": ExternalAcceptance.FAILED}),
    )

    drawn = [item.candidate_id for item in selection.candidates if item.reasons[0].kind == "spot_check"]
    assert sorted(drawn) == sorted(f"z-safe-{index}" for index in range(len(safe)))
    assert [item.reasons[0].kind for item in selection.candidates if item.candidate_id == "near"] == [
        "near_line"
    ]
    assert "external" not in {item.candidate_id for item in selection.candidates}
