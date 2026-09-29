"""結果頁的逐點差值與資料來源。"""
from __future__ import annotations

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.compare import compare_results
from aosr.reporting.display import level_db
from aosr.reporting.result import SchemeResult
from aosr.gui.result_view import (LABELS, FrequencyPoint, FrequencyResponse,
                                   _alerts, _excess, _fixed, _frequency_plot_data,
                                   _listening_area, build_result_view)
from aosr.scoring.contract import (EvaluationState, ListeningAreaStabilityPayload,
                                   QualityCategory, ReasonCode)
from aosr.scoring.listening_area import listening_area_pair_deviations
from aosr.scoring.review_alert import ListeningAreaReviewAlert
from tests.engine.test_scheme_pipeline import shared_control_result
from tests.engine.test_listening_area import _evaluate, _payload, _receiver_set, _results


def test_pair_deviations_reconstruct_both_payload_groups() -> None:
    receivers = _receiver_set(front_importance=2.0, back_importance=3.0)
    results = _results(receivers)
    payload: ListeningAreaStabilityPayload = _payload(_evaluate(receivers, results))
    pairs = listening_area_pair_deviations(
        receivers, results, broadband_range_hz=(20.0, 8000.0)
    )
    assert next(item.value for item in pairs if item.metric == "tilt" and
                item.receiver_id == "front" and item.reference_id == "main") == pytest.approx(2.0)
    for metric, field in (
        ("tilt", "tilt_stability"),
        ("ripple_rms", "ripple_rms_stability"),
        ("overall_level", "overall_level_stability"),
    ):
        comparison = getattr(payload, field)
        for group, aggregate in (
            ("primary_to_surrounding", comparison.primary_to_surrounding),
            ("surrounding_to_surrounding", comparison.surrounding_to_surrounding),
        ):
            assert aggregate is not None
            selected = [pair for pair in pairs if pair.metric == metric and pair.group == group]
            assert selected
            assert max(pair.value for pair in selected) == aggregate.worst_deviation.value
            assert sum(pair.value * pair.weight for pair in selected) / sum(
                pair.weight for pair in selected
            ) == aggregate.weighted_mean_deviation
            worst = aggregate.worst_deviation
            assert any(pair.receiver_id == worst.receiver.receiver_id and
                       pair.reference_id == worst.reference.receiver_id and
                       pair.value == worst.value for pair in selected)


def test_public_pair_deviations_wrap_unavailable_axis() -> None:
    receivers = _receiver_set(front_importance=2.0, back_importance=3.0)
    results = _results(receivers)
    changed_axis = (results[0].frequencies_hz[0] * 1.01,
                    *results[0].frequencies_hz[1:])
    changed = (results[0].model_copy(update={"frequencies_hz": changed_axis}), *results[1:])
    with pytest.raises(ValueError, match="frequency_axis_mismatch"):
        listening_area_pair_deviations(receivers, changed,
                                       broadband_range_hz=(20.0, 8000.0))


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def test_result_view_uses_engine_results_and_registry(result: SchemeResult) -> None:
    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    assert view.scheme_id == result.scheme.scheme_id
    assert view.engine_commit == result.engine_commit
    assert view.run_date == result.run_date
    assert view.timings == result.timings
    assert {(item.role, item.receiver_id) for item in view.frequency_responses} == {
        (pair.role, pair.receiver_id) for pair in result.pairs
    }
    for series, pair in zip(view.frequency_responses, result.pairs, strict=True):
        assert pair.report.points is not None
        assert [(point.frequency_hz, point.level_db) for point in series.points] == [
            (point.frequency_hz, level_db(point.total_energy)) for point in pair.report.points
        ]
    assert {item.category for item in view.categories} == {item.value for item in QualityCategory}
    ranking = compare_results((result,), quality_targets=load_quality_targets(
        config_path("quality_targets.toml")), run_date=result.run_date)
    row = ranking.rankable[0]
    assert {item.category: item.cost for item in view.categories if item.cost is not None} == {
        line.identity.category.value: line.category_cost for line in row.categories
    }
    assert len(view.alerts) == len(row.review_alerts)
    assert any(item.note == "低頻拖尾：尚未評估" for item in view.categories)
    assert any(item.note == "空間感：尚未評估" for item in view.categories)
    assert view.reverberation.note == "殘響是整間房的統計量，換座位不變"
    assert view.reverberation.bands
    assert all(band.target_low_text and band.target_high_text
               for band in view.reverberation.bands)
    assert view.reflections
    reflection_evaluation = next(item for item in result.candidate.evaluations
                                 if item.category is QualityCategory.REFLECTIONS_AND_ECHO)
    assert view.reflections[0].flags == tuple(flag.value for flag in reflection_evaluation.flags)
    assert view.listening_area.scope_note and view.listening_area.pairs
    assert {item.group for item in view.listening_area.pairs} >= {"primary_to_surrounding"}
    assert view.ranking_status == "rankable" and LABELS[view.ranking_status] == "可排名"


def test_result_page_script_only_renders_server_values() -> None:
    from aosr.gui.app import STATIC

    script = (STATIC / "results.js").read_text()
    html = (STATIC / "results.html").read_text()
    assert "innerHTML" not in script
    assert "Math.log" not in script and "Math.pow" not in script
    assert "uPlot" in script and "vendor/uplot" in html
    assert "/api/results/" in script
    assert "time: false" in script and "distr: 3" in script
    assert "聲級由伺服器換算" not in script
    assert "最差差距暫定線" in script
    assert "item.reason_codes.join" not in script


def test_alert_uses_structured_exact_value_not_rounded_note() -> None:
    alert = ListeningAreaReviewAlert(
        category=QualityCategory.LISTENING_AREA_STABILITY, role="left",
        speaker_id="left", metric="ripple_rms", group="primary_to_surrounding",
        receiver_id="right10", reference_id="main", deviation=3.0001,
        limit=3.0, note="差 3 dB，超過暫定線 3 dB；尚未正式校準",
    )
    view = _alerts((alert,))[0]
    assert view.receiver_id == "right10" and view.reference_id == "main"
    assert view.excess_text != "未超過"
    assert view.excess_text is not None
    assert float(view.excess_text) > 0
    assert ("差值", "3.00 dB") in view.fields
    assert ("暫定線", "3.00 dB") in view.fields
    assert alert.note not in str(view.model_dump())


def test_frequency_plot_arrays_share_sorted_axis(result: SchemeResult) -> None:
    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    for role, arrays in view.frequency_plot_data.items():
        assert role in {item.role for item in view.frequency_responses}
        axis = arrays[0]
        assert axis and all(isinstance(value, float) for value in axis)
        xs = [value for value in axis if value is not None]
        assert all(a < b for a, b in zip(xs, xs[1:]))
        assert all(len(values) == len(axis) for values in arrays[1:])
        assert len(arrays[1:]) == len([item for item in view.frequency_responses if item.role == role])


def test_frequency_plot_inserts_null_for_missing_points() -> None:
    def response(receiver: str, frequencies: tuple[float, ...]) -> FrequencyResponse:
        return FrequencyResponse(role="left", speaker_id="left", receiver_id=receiver,
                                 receiver_role="surrounding", receiver_label="周圍點",
                                 points=tuple(FrequencyPoint(frequency_hz=value, level_db=-1.0,
                                                             frequency_text=f"{value} Hz",
                                                             level_text="-1.00 dB")
                                              for value in frequencies))
    arrays = _frequency_plot_data((response("front", (20.0, 40.0)),
                                   response("back", (30.0, 40.0))))["left"]
    assert arrays == ((20.0, 30.0, 40.0), (-1.0, None, -1.0), (None, -1.0, -1.0))


def test_result_view_registry_limits_and_reverberation_intervals(result: SchemeResult) -> None:
    from aosr.scoring.reverberation_cost import target_intervals

    registry = load_quality_targets(config_path("quality_targets.toml"))
    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    purpose = registry.purpose(result.scheme.purpose)
    limits = {metric: purpose.entry(f"listening_area_stability.{key}")
              for metric, key in (("tilt", "tilt_worst_deviation"),
                                  ("ripple_rms", "ripple_rms_worst_deviation"),
                                  ("overall_level", "overall_level_worst_deviation"))}
    from aosr.config.quality_targets import TargetEntry
    assert all(isinstance(entry, TargetEntry) for entry in limits.values())
    for pair in view.listening_area.pairs:
        entry = limits[pair.metric]
        assert isinstance(entry, TargetEntry)
        assert pair.limit == entry.value
        assert pair.excess_text == ("未超過" if not pair.over_limit else pair.excess_text)
        assert pair.status_text.startswith("超過暫定線" if pair.over_limit else "未超過")
    for summary in view.listening_area.summaries:
        entry = limits[summary.metric]
        assert isinstance(entry, TargetEntry) and isinstance(entry.value, float | int)
        expected = f"{entry.value:.3f}" if summary.metric == "tilt" else f"{entry.value:.2f}"
        assert summary.limit_text == expected
        evaluation = next(item for item in result.candidate.evaluations
                          if item.category is QualityCategory.LISTENING_AREA_STABILITY)
        from aosr.scoring.contract import ListeningAreaChannelsPayload
        payload = evaluation.payload
        assert isinstance(payload, ListeningAreaChannelsPayload)
        channel = next(item for item in payload.channels if item.role == summary.role)
        field = {"tilt": "tilt_stability", "ripple_rms": "ripple_rms_stability",
                 "overall_level": "overall_level_stability"}[summary.metric]
        aggregate = getattr(getattr(channel.payload, field), summary.group)
        assert aggregate is not None
        value = aggregate.worst_deviation.value
        expected_value = f"{value:.3f}" if summary.metric == "tilt" else f"{value:.2f}"
        assert summary.worst_value_text == expected_value
    intervals = target_intervals(purpose)
    for band in view.reverberation.bands:
        lower, upper = intervals[band.center_frequency_hz]
        assert band.target_low_text == f"{lower:.3f} 秒"
        assert band.target_high_text == f"{upper:.3f} 秒"


def test_flutter_alert_is_human_readable() -> None:
    from aosr.scoring.review_alert import FlutterReviewAlert

    alert = FlutterReviewAlert(category=QualityCategory.REFLECTIONS_AND_ECHO,
                               walls=("floor", "ceiling"), nominal_center_hz=400,
                               center_frequency_hz=396.85, lower_hz=300.0, upper_hz=500.0,
                               decay_duration_s=0.06871, room_t20_s=0.06868,
                               decay_db=60.0, note="原始字不可照印")
    view = _alerts((alert,))[0]
    assert ("牆對", "地板、天花") in view.fields
    # 顫動是整間房的量，沒有聲道也沒有喇叭：標題只有類別，不留空的分隔號。
    assert view.heading_text == "牆間顫動警戒"
    assert ("持續度", "68.71 毫秒") in view.fields
    assert ("本房 T20", "68.68 毫秒") in view.fields
    assert view.excess_text == "0.03 毫秒"
    assert "walls" not in str(view.fields) and "原始字" not in str(view.fields)
    reflected = _alerts((alert.model_copy(update={"decay_duration_s": None}),))[0]
    assert ("持續度", "全反射，持續度無限長") in reflected.fields


@pytest.mark.parametrize("duration,t20", [(0.06876, 0.06874),
                                          (0.06875001, 0.06874999)])
def test_flutter_excess_never_displays_zero(duration: float, t20: float) -> None:
    from aosr.scoring.review_alert import FlutterReviewAlert

    alert = FlutterReviewAlert(category=QualityCategory.REFLECTIONS_AND_ECHO,
                               walls=("floor", "ceiling"), nominal_center_hz=400,
                               center_frequency_hz=396.85, lower_hz=300.0, upper_hz=500.0,
                               decay_duration_s=duration, room_t20_s=t20,
                               decay_db=60.0, note="警戒")
    view = _alerts((alert,))[0]
    fields = dict(view.fields)
    assert fields["持續度"] != fields["本房 T20"]
    assert view.excess_text is not None
    assert view.excess_text.startswith("小於 ") or float(view.excess_text.split()[0]) > 0


def test_flutter_uses_one_decimal_when_it_distinguishes_values() -> None:
    from aosr.scoring.review_alert import FlutterReviewAlert

    alert = FlutterReviewAlert(category=QualityCategory.REFLECTIONS_AND_ECHO,
                               walls=("floor", "ceiling"), nominal_center_hz=400,
                               center_frequency_hz=396.85, lower_hz=300.0, upper_hz=500.0,
                               decay_duration_s=0.0688, room_t20_s=0.0686,
                               decay_db=60.0, note="警戒")
    view = _alerts((alert,), roles={"None": "left"})[0]
    assert ("持續度", "68.8 毫秒") in view.fields
    assert ("本房 T20", "68.6 毫秒") in view.fields
    assert view.excess_text == "0.2 毫秒"
    assert view.role is None and view.heading_text == "牆間顫動警戒"


def test_alert_roles_and_units_follow_scheme_channels(result: SchemeResult) -> None:
    from aosr.scoring.review_alert import PeakDipReviewAlert

    channel = result.scheme.channel_group.channels[0]
    alert = PeakDipReviewAlert(category=QualityCategory.TIMBRE_BALANCE,
                               speaker_id=channel.speaker_id, receiver_id="main", kind="dip",
                               center_frequency_hz=100.0, depth_db=-33.62,
                               width_octave=0.5, limit_db=15.0,
                               narrower_than_axis=False, note="警戒")
    view = _alerts((alert,), roles={channel.speaker_id: channel.role})[0]
    assert view.role == channel.role
    assert ("峰谷量", "谷深 33.62 dB") in view.fields
    assert view.heading_text == f"谷值警戒・{LABELS[channel.role]}・{channel.speaker_id}"
    assert ("警戒線", "15.00 dB") in view.fields


def test_peak_alert_depth_keeps_db() -> None:
    from aosr.scoring.review_alert import PeakDipReviewAlert

    alert = PeakDipReviewAlert(category=QualityCategory.TIMBRE_BALANCE,
                               speaker_id="left", receiver_id="main", kind="peak",
                               center_frequency_hz=100.0, depth_db=6.0,
                               width_octave=0.5, limit_db=5.0,
                               narrower_than_axis=False, note="警戒")
    assert ("峰谷量", "6.00 dB") in _alerts((alert,))[0].fields


def test_flutter_alerts_follow_speaker_and_seat_alerts() -> None:
    from aosr.scoring.review_alert import FlutterReviewAlert, PeakDipReviewAlert

    flutter = FlutterReviewAlert(category=QualityCategory.REFLECTIONS_AND_ECHO,
                                 walls=("floor", "ceiling"), nominal_center_hz=400,
                                 center_frequency_hz=396.85, lower_hz=300.0, upper_hz=500.0,
                                 decay_duration_s=0.0688, room_t20_s=0.0686,
                                 decay_db=60.0, note="警戒")
    peak = PeakDipReviewAlert(category=QualityCategory.TIMBRE_BALANCE,
                              speaker_id="left", receiver_id="main", kind="peak",
                              center_frequency_hz=100.0, depth_db=6.0,
                              width_octave=0.5, limit_db=5.0,
                              narrower_than_axis=False, note="警戒")
    assert [item.kind for item in _alerts((flutter, peak, flutter, peak))] == [
        "peak", "peak", "flutter", "flutter"]
    # 同類照原順序：跟喇叭與座位有關的警戒之間不重排（穩定排序，只把顫動挪到後面）。
    dip = peak.model_copy(update={"kind": "dip", "depth_db": -20.0, "limit_db": 15.0})
    # 輸入故意不照字母順序（峰在谷前）：照類別名稱重排會變成谷在前，就抓得到。
    assert [item.kind for item in _alerts((peak, flutter, dip))] == ["peak", "dip", "flutter"]


def test_reflection_display_keeps_payload_classification(result: SchemeResult) -> None:
    import re
    from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload

    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    evaluation = next(item for item in result.candidate.evaluations
                      if item.category is QualityCategory.REFLECTIONS_AND_ECHO)
    payload = evaluation.payload
    assert isinstance(payload, ReflectionsAndEchoPayload)
    original = {(channel.speaker_id, channel.receiver_id): channel for channel in payload.channels
                if channel.is_primary}
    for channel in view.reflections:
        stored = original[channel.speaker_id, channel.receiver_id]
        ordered = sorted(stored.reflections, key=lambda item: item.relative_direct_delay_s)
        for path, raw in zip(channel.paths, ordered, strict=True):
            assert re.fullmatch(r"\d+\.\d{2} 毫秒", path.delay_text)
            assert re.fullmatch(r"-?\d+\.\d dB", path.level_text)
            assert re.fullmatch(r"-?\d+\.\d 度", path.azimuth_text)
            assert re.fullmatch(r"-?\d+\.\d 度", path.elevation_text)
            assert path.azimuth_text != "-0.0 度" and path.elevation_text != "-0.0 度"
            assert path.within_window == raw.within_window
            assert path.zone == raw.zone.value
            assert path.wall_sequence == tuple(LABELS[wall] for wall in raw.wall_sequence)
        assert [path.delay_text for path in channel.paths] == [
            f"{raw.relative_direct_delay_s * 1000:.2f} 毫秒"
            for raw in ordered]
    assert all(re.fullmatch(r"\d+ Hz", band.center_text)
               for band in view.reverberation.bands)


def test_fixed_display_handles_absent_and_negative_zero() -> None:
    assert _fixed(None, 1, "度") == "—"
    assert _fixed(-0.01, 1, "度") == "0.0 度"


def test_unavailable_sections_and_actual_surrounding_distance(result: SchemeResult) -> None:
    from aosr.scoring.receiver_set import ReceiverRole

    primary = result.scheme.receiver_set.primary.position_m
    points = tuple(point.model_copy(update={"importance": 0.0,
                                    "position_m": (primary[0] + 0.4, primary[1], primary[2])})
                   if point.role is ReceiverRole.SURROUNDING else point
                   for point in result.scheme.receiver_set.points)
    receivers = result.scheme.receiver_set.model_copy(update={"points": points})
    scheme = result.scheme.model_copy(update={"receiver_set": receivers})
    evaluations = tuple(item.model_copy(update={"state": EvaluationState.UNAVAILABLE,
                                         "payload": None,
                                         "reason_codes": (ReasonCode.ZERO_TOTAL_IMPORTANCE,)})
                        if item.category in (QualityCategory.LISTENING_AREA_STABILITY,
                                             QualityCategory.REFLECTIONS_AND_ECHO) else item
                        for item in result.candidate.evaluations)
    candidate = result.candidate.model_copy(update={"evaluations": evaluations})
    altered = result.model_copy(update={"scheme": scheme, "candidate": candidate})
    registry = load_quality_targets(config_path("quality_targets.toml"))
    listening = _listening_area(altered, config_path("quality_targets.toml"), registry)
    assert listening.state == "unavailable" and not listening.pairs
    assert listening.reason_codes == (ReasonCode.ZERO_TOTAL_IMPORTANCE.value,)
    assert "0.40 公尺" in listening.scope_note
    from aosr.gui.result_view import _reflections
    reflections = _reflections(altered)
    assert reflections and all(item.state == "unavailable" and not item.paths
                               for item in reflections)
    view = build_result_view(altered, quality_targets_path=config_path("quality_targets.toml"))
    assert view.listening_area.state == "unavailable"
    assert view.reflections and view.reflections[0].state == "unavailable"


def test_server_labels_cover_all_displayed_codes() -> None:
    from aosr.scoring.category_registry import EliminationReason, NotEvaluatedReason
    from aosr.scoring.contract_base import Flag
    from aosr.scoring.ranking_models import CandidateStatus

    for codes in (ReasonCode, Flag, EliminationReason, NotEvaluatedReason, CandidateStatus):
        assert all(code.value in LABELS for code in codes)
    from aosr.gui.app import STATIC
    script = (STATIC / "app.js").read_text()
    assert 'fetch("/api/plan", {method: "POST"' in script
    assert "problem.path" in script and "problem.message" in script
    assert "JSON.stringify(check.problems" not in script


def test_excess_distinguishes_at_limit_below_and_just_above() -> None:
    assert _excess(0.0, "dB") == "未超過"
    assert _excess(-0.01, "dB") == "未超過"
    assert _excess(0.0001, "dB") == "0.0001"
    assert _excess(1e-13, "dB") == "小於 0.000000000001"


def test_baseline_note_follows_registry_status(result: SchemeResult) -> None:
    from aosr.config.quality_targets import TargetEntry
    from aosr.scoring.review_alert import PeakDipReviewAlert

    registry = load_quality_targets(config_path("quality_targets.toml"))
    purpose = registry.purpose(result.scheme.purpose)
    key = "timbre_balance.peak_depth_db"
    entry = purpose.entry(key)
    assert isinstance(entry, TargetEntry)
    updated = purpose.model_copy(update={"target": tuple(
        item.model_copy(update={"status": "calibrated"}) if item.key == key else item
        for item in purpose.target)})
    changed = registry.model_copy(update={"purposes": tuple(
        updated if item.name == purpose.name else item for item in registry.purposes)})
    alert = PeakDipReviewAlert(category=QualityCategory.TIMBRE_BALANCE,
                               speaker_id="left", receiver_id="main", kind="peak",
                               center_frequency_hz=100.0, depth_db=6.0,
                               width_octave=0.5, limit_db=5.0,
                               narrower_than_axis=False, note="警戒")
    assert _alerts((alert,), registry, purpose.name)[0].baseline_note == "尚未正式校準"
    assert _alerts((alert,), changed, purpose.name)[0].baseline_note is None


def test_reverberation_display_uses_public_interval_reader(
    result: SchemeResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.gui.result_view import _reverberation
    from aosr.scoring.reverberation_cost import target_intervals

    registry = load_quality_targets(config_path("quality_targets.toml"))
    centers = target_intervals(registry.purpose(result.scheme.purpose))
    sentinel = {center: (1.123, 2.234) for center in centers}
    monkeypatch.setattr("aosr.gui.result_view.target_intervals", lambda purpose: sentinel)
    view = _reverberation(result, registry)
    assert all(band.target_low_text == "1.123 秒" and
               band.target_high_text == "2.234 秒" for band in view.bands)


def test_validation_rejection_omits_pydantic_input_and_url(result: SchemeResult) -> None:
    from pydantic import ValidationError
    from aosr.gui.app import _rejection_reason

    document = result.model_dump(mode="json")
    document["schema_version"] = "aosr.scheme_result.v1"
    with pytest.raises(ValidationError) as raised:
        SchemeResult.model_validate(document)
    reason = _rejection_reason(raised.value)
    assert reason == "結果檔格式是舊版（欄位 schema_version）"
    assert "input_value" not in reason and "http" not in reason
    document = result.model_dump(mode="json")
    del document["quality_targets_fingerprint"]
    with pytest.raises(ValidationError) as missing:
        SchemeResult.model_validate(document)
    reason = _rejection_reason(missing.value)
    assert reason == "結果檔欄位不符合現行格式（欄位 quality_targets_fingerprint）"
    assert "input_value" not in reason and "http" not in reason
