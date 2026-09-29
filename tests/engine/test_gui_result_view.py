"""結果頁的逐點差值與資料來源。"""
from __future__ import annotations

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.compare import compare_results
from aosr.reporting.display import level_db
from aosr.reporting.result import SchemeResult
from aosr.gui.labels import listening_point_label, speaker_label
from aosr.gui.result_view import (LABELS, FrequencyPoint, FrequencyResponse, PairView,
                                   _alert_sections, _alerts, _excess, _fixed, _flutter_groups,
                                   _frequency_plot_data, _listening_area, _pair_choices,
                                   build_result_view)
from aosr.scoring.contract import (EvaluationState, ListeningAreaStabilityPayload,
                                   QualityCategory, ReasonCode)
from aosr.scoring.listening_area import listening_area_pair_deviations
from aosr.scoring.review_alert import FlutterReviewAlert, ListeningAreaReviewAlert
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
    # 顫動按牆對合併成一筆，逐帶的都還在明細裡：兩區加起來一筆不少。
    assert len(view.alerts) + sum(len(group.bands) for group in view.flutter_groups) == len(
        row.review_alerts)
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
    # 圖例不跟游標印數值（不然每條線後面掛「Value」與「--」）。
    assert "legend: {live: false}" in script


def test_alert_uses_structured_exact_value_not_rounded_note() -> None:
    alert = ListeningAreaReviewAlert(
        category=QualityCategory.LISTENING_AREA_STABILITY, role="left",
        speaker_id="left", metric="ripple_rms", group="primary_to_surrounding",
        receiver_id="right10", reference_id="main", deviation=3.0001,
        limit=3.0, note="差 3 dB，超過暫定線 3 dB；尚未正式校準",
    )
    view = _alerts((alert,))[0]
    assert view.receiver_id == "right10" and view.reference_id == "main"
    # 顯示名：主位查得到；right10 不在顯示名稱表上，照原代號印，不猜。
    assert view.place_text == "位置對：主位 ↔ right10"
    assert view.heading_text == "聆聽區最差差距・左聲道喇叭"
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


def _flutter(walls: tuple[str, str], nominal: int, duration: float | None,
             t20: float = 0.0687) -> FlutterReviewAlert:
    return FlutterReviewAlert(category=QualityCategory.REFLECTIONS_AND_ECHO, walls=walls,
                              nominal_center_hz=nominal, center_frequency_hz=nominal * 0.992,
                              lower_hz=nominal * 0.89, upper_hz=nominal * 1.12,
                              decay_duration_s=duration, room_t20_s=t20, decay_db=60.0,
                              note="原始字不可照印")


def test_flutter_alerts_merge_into_one_entry_per_wall_pair() -> None:
    floor = ("floor", "ceiling")
    sides = ("x0", "xL")
    # 牆對交錯、頻帶亂序給：合併要照牆對分、帶照頻率排，不靠輸入順序。
    alerts = (_flutter(floor, 630, 0.1072), _flutter(sides, 400, 0.2144),
              _flutter(floor, 400, 0.1072), _flutter(floor, 500, 0.1072),
              _flutter(sides, 500, 0.2144))
    groups = _flutter_groups(alerts)
    assert [group.walls for group in groups] == [floor, sides]
    first = groups[0]
    assert first.heading_text == "牆間顫動警戒・地板、天花"
    assert dict(first.fields) == {"頻帶": "400–630 Hz，共 3 個三分之一八度帶",
                                  "持續度": "107.2 毫秒", "本房同帶 T20": "68.7 毫秒",
                                  "超出多少": "38.5 毫秒"}
    assert [band.nominal_text for band in first.bands] == ["400 Hz", "500 Hz", "630 Hz"]
    assert first.detail_summary_text == f"逐帶明細（{len(first.bands)} 帶）"
    assert "衰減 60 dB 所需的時間" in first.summary_text
    assert dict(groups[1].fields)["頻帶"] == "400–500 Hz，共 2 個三分之一八度帶"
    assert "原始字" not in str(groups[0].model_dump()) and "walls" not in str(first.fields)


def test_flutter_group_shows_ranges_when_bands_differ() -> None:
    walls = ("y0", "yL")
    group = _flutter_groups((_flutter(walls, 400, 0.1072), _flutter(walls, 500, 0.15, t20=0.07),
                             _flutter(walls, 630, None)))[0]
    fields = dict(group.fields)
    assert fields["持續度"] == "107.2–150.0 毫秒；另有 1 帶全反射，持續度無限長"
    assert fields["本房同帶 T20"] == "68.7–70.0 毫秒"
    assert fields["超出多少"] == "38.5–80.0 毫秒"
    reflected = group.bands[-1]
    assert (reflected.duration_text, reflected.excess_text) == ("全反射，持續度無限長", "持續度無限長")
    only_reflected = dict(_flutter_groups((_flutter(walls, 400, None),))[0].fields)
    assert only_reflected["持續度"] == "全反射，持續度無限長"
    assert only_reflected["超出多少"] == "持續度無限長"
    assert only_reflected["頻帶"] == "400 Hz，共 1 個三分之一八度帶"


def test_flutter_band_detail_is_human_readable() -> None:
    band = _flutter_groups((_flutter(("floor", "ceiling"), 400, 0.06871, t20=0.06868),))[0].bands[0]
    assert band.nominal_text == "400 Hz" and band.center_text == "396.80 Hz"
    assert band.duration_text == "68.71 毫秒"
    assert band.room_t20_text == "68.68 毫秒"
    assert band.excess_text == "0.03 毫秒"


@pytest.mark.parametrize("duration,t20", [(0.06876, 0.06874),
                                          (0.06875001, 0.06874999)])
def test_flutter_excess_never_displays_zero(duration: float, t20: float) -> None:
    band = _flutter_groups((_flutter(("floor", "ceiling"), 400, duration, t20=t20),))[0].bands[0]
    assert band.duration_text != band.room_t20_text
    assert band.excess_text.startswith("小於 ") or float(band.excess_text.split()[0]) > 0


def test_flutter_uses_one_decimal_when_it_distinguishes_values() -> None:
    band = _flutter_groups((_flutter(("floor", "ceiling"), 400, 0.0688, t20=0.0686),))[0].bands[0]
    assert (band.duration_text, band.room_t20_text, band.excess_text) == (
        "68.8 毫秒", "68.6 毫秒", "0.2 毫秒")


def test_flutter_group_rows_share_the_finest_digits_needed() -> None:
    # 一帶要兩位才分得出持續度與 T20：同一對牆的每一帶都印兩位，逐帶明細對得齊。
    group = _flutter_groups((_flutter(("floor", "ceiling"), 400, 0.06871, t20=0.06868),
                             _flutter(("floor", "ceiling"), 500, 0.1072)))[0]
    assert [band.duration_text for band in group.bands] == ["68.71 毫秒", "107.20 毫秒"]


def test_alert_sections_put_flutter_after_seat_alerts(result: SchemeResult) -> None:
    from aosr.scoring.review_alert import PeakDipReviewAlert

    registry = load_quality_targets(config_path("quality_targets.toml"))
    peak = PeakDipReviewAlert(category=QualityCategory.TIMBRE_BALANCE,
                              speaker_id="left", receiver_id="main", kind="peak",
                              center_frequency_hz=100.0, depth_db=6.0,
                              width_octave=0.5, limit_db=5.0,
                              narrower_than_axis=False, note="警戒")
    dip = peak.model_copy(update={"kind": "dip", "depth_db": -20.0, "limit_db": 15.0})
    flutter = (_flutter(("floor", "ceiling"), 400, 0.1072), _flutter(("floor", "ceiling"), 500, 0.1072))
    # 輸入故意峰在谷前、顫動夾在中間：跟座位有關的照原順序，顫動全部合併到另一區。
    seats, groups = _alert_sections((peak, flutter[0], dip, flutter[1]), registry,
                                    result.scheme.purpose, {"left": "left"})
    assert [item.kind for item in seats] == ["peak", "dip"]
    assert [group.walls for group in groups] == [("floor", "ceiling")]
    assert [band.nominal_text for band in groups[0].bands] == ["400 Hz", "500 Hz"]


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
    # 標題用喇叭顯示名、不再掛喇叭代號；座位另一行也是顯示名。
    assert view.heading_text == f"谷值警戒・{speaker_label(channel.role)}"
    assert view.place_text == f"位置：{listening_point_label('main')}"
    assert ("警戒線", "15.00 dB") in view.fields


def test_peak_alert_depth_keeps_db() -> None:
    from aosr.scoring.review_alert import PeakDipReviewAlert

    alert = PeakDipReviewAlert(category=QualityCategory.TIMBRE_BALANCE,
                               speaker_id="left", receiver_id="main", kind="peak",
                               center_frequency_hz=100.0, depth_db=6.0,
                               width_octave=0.5, limit_db=5.0,
                               narrower_than_axis=False, note="警戒")
    assert ("峰谷量", "6.00 dB") in _alerts((alert,))[0].fields
    # 不知道是哪個聲道時照喇叭代號印，不留空也不猜。
    assert _alerts((alert,))[0].heading_text == "峰值警戒・left"


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
    # 沒有路徑表就沒有時間窗那句、也沒有窗外摺疊區；標題照樣用顯示名。
    assert all(item.window_text == item.outside_summary_text == "" for item in reflections)
    assert {item.heading_text for item in reflections} == {
        f"{speaker_label(channel.role)} → {listening_point_label(item.receiver_id)}"
        for channel, item in zip(altered.scheme.channel_group.channels, reflections, strict=True)}
    view = build_result_view(altered, quality_targets_path=config_path("quality_targets.toml"))
    assert view.listening_area.state == "unavailable"
    assert view.reflections and view.reflections[0].state == "unavailable"
    # 反射不可估時注意事項照樣是白話句子，每一個旗標那一句都跟各類結果主表同一個旗標那一句一字不差。
    category = next(item for item in view.categories if item.category == "reflections_and_echo")
    for item in view.reflections:
        assert item.flags and _same_sentences(item.flags, item.flags_text, category.flags, category.flags_text)


def test_server_labels_cover_all_displayed_codes() -> None:
    from aosr.scoring.category_registry import EliminationReason, NotEvaluatedReason
    from aosr.scoring.contract_base import Flag
    from aosr.scoring.ranking_models import CandidateStatus

    for codes in (ReasonCode, Flag, EliminationReason, NotEvaluatedReason, CandidateStatus):
        assert all(code.value in LABELS for code in codes)
    from aosr.gui.app import STATIC
    script = (STATIC / "app.js").read_text()
    assert 'fetch("/api/plan", {method: "POST"' in script
    # 檢查不過的訊息：頁面印伺服器寫好的那一行（problem.text），不自己拿英文路徑拼，也不把整包倒出來。
    assert "problem.text" in script and "problem.path" not in script
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


def test_display_names_come_from_the_label_tables(result: SchemeResult) -> None:
    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    assert view.speaker_names == {channel.role: speaker_label(channel.role)
                                  for channel in result.scheme.channel_group.channels}
    assert view.point_names == {point.receiver_id: listening_point_label(point.receiver_id)
                                for point in result.scheme.receiver_set.points}
    # 範例方案的代號查得到中文：喇叭的 left 跟座位的 left 是兩個名字。
    assert view.speaker_names["left"] == "左聲道喇叭" and view.point_names["front"] == "主位前方"
    first = result.scheme.channel_group.channels[0].role
    primary = result.scheme.receiver_set.primary.receiver_id
    assert view.reverberation.caption_text == (
        f"{view.reverberation.note}；取自{speaker_label(first)} → {listening_point_label(primary)}")
    for summary in view.listening_area.summaries:
        ends = {listening_point_label(summary.worst_reference_id), listening_point_label(summary.worst_receiver_id)}
        assert set(summary.worst_pair_text.split(" ↔ ")) == ends


def test_reflections_list_window_paths_and_count_the_rest(result: SchemeResult) -> None:
    from aosr.gui.result_view import _reflection_texts
    from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload

    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    payload = next(item for item in result.candidate.evaluations
                   if item.category is QualityCategory.REFLECTIONS_AND_ECHO).payload
    assert isinstance(payload, ReflectionsAndEchoPayload)
    window = f"{payload.window_upper_ms:g} 毫秒"
    assert view.reflections
    for channel in view.reflections:
        inside = sum(path.within_window for path in channel.paths)
        outside = len(channel.paths) - inside
        assert channel.heading_text == (
            f"{speaker_label(channel.role)} → {listening_point_label(channel.receiver_id)}")
        assert f"直達音後 {window}內" in channel.window_text
        assert f"窗內有 {inside} 條" in channel.window_text
        # 考卷資料窗外有路徑：摘要行要說有幾條（數字從資料算，不寫死）。
        assert outside and f"反射路徑 {outside} 條" in channel.outside_summary_text
    # 窗外一條都沒有時不畫摺疊區：摘要是空字串。
    inside_only = tuple(path for path in view.reflections[0].paths if path.within_window)
    assert _reflection_texts(inside_only, payload.window_upper_ms)[1] == ""


def test_reverberation_verdict_follows_the_evaluators_own_directions(result: SchemeResult) -> None:
    from aosr.gui.result_view import EVALUATOR_VERDICT_NOTE, _reverberation

    registry = load_quality_targets(config_path("quality_targets.toml"))
    ranking = compare_results((result,), quality_targets=registry, run_date=result.run_date)
    costed = next(line.evaluation for line in ranking.rankable[0].categories
                  if line.identity.category is QualityCategory.REVERBERATION)
    assert costed.category_cost is not None
    directions = costed.category_cost.component_directions
    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    measured = [band for band in view.reverberation.bands if band.t20_state == "measured"]
    assert measured
    assert {band.center_frequency_hz: band.verdict for band in measured} == {
        band.center_frequency_hz: directions[f"t20_target_interval.{band.center_frequency_hz:g}Hz"]
        for band in measured}
    assert view.reverberation.compare_note == EVALUATOR_VERDICT_NOTE and "T20" in EVALUATOR_VERDICT_NOTE
    # 評分說高於上限，畫面就照印，不自己拿 T20 再比一次（把方向全改成高於上限就抓得到）。
    flipped = costed.model_copy(update={"category_cost": costed.category_cost.model_copy(
        update={"component_directions": {key: "above_range" for key in directions}})})
    shown = _reverberation(result, registry, flipped)
    assert {(band.verdict, band.verdict_text) for band in shown.bands
            if band.t20_state == "measured"} == {("above_range", "高於上限")}


def test_reverberation_verdict_without_cost_compares_t20_with_target(
    result: SchemeResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.gui.result_view import DIRECT_VERDICT_NOTE, _reverberation
    from aosr.scoring.contract import ReverberationPayload

    registry = load_quality_targets(config_path("quality_targets.toml"))
    payload = next(item for item in result.candidate.evaluations
                   if item.category is QualityCategory.REVERBERATION).payload
    assert isinstance(payload, ReverberationPayload)
    measured = [(band.center_frequency_hz, band.t20.value) for band in payload.bands
                if band.t20.value is not None]
    assert len(measured) >= 3
    (low, low_value), (same, same_value), (high, high_value) = measured[:3]
    # 目標區間繞著各帶的 T20 擺：一帶在下限之下、一帶剛好等於上下限、一帶在上限之上；其餘帶沒登記目標。
    sentinel = {low: (low_value + 0.1, low_value + 0.2), same: (same_value, same_value),
                high: (high_value - 0.2, high_value - 0.1)}
    monkeypatch.setattr("aosr.gui.result_view.target_intervals", lambda purpose: sentinel)
    view = _reverberation(result, registry)
    verdicts = {band.center_frequency_hz: (band.verdict, band.verdict_text) for band in view.bands}
    assert [verdicts[center] for center in (low, same, high)] == [
        ("below_range", "低於下限"), ("within_range", "在目標內"), ("above_range", "高於上限")]
    assert {verdicts[center] for center, _ in measured[3:]} <= {("no_target", "這個頻帶沒有登記目標")}
    assert view.compare_note == DIRECT_VERDICT_NOTE and "直接拿 T20" in DIRECT_VERDICT_NOTE


def test_reverberation_says_which_metric_was_not_measured(result: SchemeResult) -> None:
    from aosr.gui.result_view import _reverberation
    from aosr.scoring.contract import ReverberationMetric, ReverberationPayload
    from aosr.scoring.contract_base import MetricState

    registry = load_quality_targets(config_path("quality_targets.toml"))
    evaluation = next(item for item in result.candidate.evaluations
                      if item.category is QualityCategory.REVERBERATION)
    payload = evaluation.payload
    assert isinstance(payload, ReverberationPayload)
    missing = ReverberationMetric(value=None, unit="s", state=MetricState.UNAVAILABLE,
                                  reason_codes=(ReasonCode.INSUFFICIENT_DECAY_RANGE,),
                                  reason="衰減範圍不足")
    bands = (payload.bands[0].model_copy(update={"t30": missing}),
             payload.bands[1].model_copy(update={"t20": missing}), *payload.bands[2:])
    changed = evaluation.model_copy(update={"payload": payload.model_copy(update={"bands": bands})})
    candidate = result.candidate.model_copy(update={"evaluations": tuple(
        changed if item is evaluation else item for item in result.candidate.evaluations)})
    view = _reverberation(result.model_copy(update={"candidate": candidate}), registry)
    assert view.bands[0].measured_text == "T20 量到；T30 不可估（衰減範圍不足）"
    assert view.bands[1].measured_text.startswith("T20 不可估（衰減範圍不足）")
    assert (view.bands[1].verdict, view.bands[1].verdict_text) == ("unavailable", "T20 量不到，無法比")
    assert {band.measured_text for band in view.bands[2:]
            if band.t20_state == band.t30_state == "measured"} <= {"T20、T30 都量到"}


def test_categories_main_view_uses_plain_words_and_short_commit(result: SchemeResult) -> None:
    from aosr.gui.result_view import FLAG_TEXTS
    from aosr.scoring.contract_base import Flag

    # 考卷結果的提交只有 7 個字（control），「只取前 7 碼」拿它量不出來：換成四十碼的提交再建一次。
    commit = "e53bfae" + "f" * 33
    view = build_result_view(result.model_copy(update={"engine_commit": commit}),
                             quality_targets_path=config_path("quality_targets.toml"))
    assert view.engine_commit == commit and view.engine_commit_text == "e53bfae"
    assert view.cost_note == "代價越低越好，0 表示沒有扣分"
    assert any(item.flags for item in view.categories)
    for item in view.categories:
        # 主表只有白話：原代號一個都不出現，每個旗標都有它那一句。
        assert all(flag not in item.flags_text for flag in item.flags)
        assert all(FLAG_TEXTS.get(flag, LABELS[flag]) in item.flags_text for flag in item.flags)
    # 評估器版本與原因碼還在資料裡，給「技術細節」摺疊區用。
    assert any(item.evaluator_version for item in view.categories)
    assert all(flag.value in FLAG_TEXTS or flag.value in LABELS for flag in Flag)


def test_reflection_notes_use_the_same_plain_sentences_as_categories(result: SchemeResult) -> None:
    from aosr.gui.result_view import FLAG_TEXTS

    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    category = next(item for item in view.categories if item.category == "reflections_and_echo")
    assert view.reflections and category.flags
    for channel in view.reflections:
        # 反射那一段跟各類結果主表同一份白話：原代號不出現，每個旗標都有它那一句，而且同一個旗標兩處一字不差
        # （主表那一格另外帶評分時加的「某方向反射超線」，反射這一段只列反射評估自己的旗標）。
        assert channel.flags and all(flag not in channel.flags_text for flag in channel.flags)
        assert all(FLAG_TEXTS.get(flag, LABELS[flag]) in channel.flags_text for flag in channel.flags)
        assert _same_sentences(channel.flags, channel.flags_text, category.flags, category.flags_text)
        assert all(LABELS[flag] not in channel.flags_text for flag in channel.flags
                   if flag in FLAG_TEXTS and LABELS[flag] not in FLAG_TEXTS[flag])
        # 考卷結果一定有的兩個反射旗標：句子釘死，不從 FLAG_TEXTS 抄（抄的話兩邊一起改也量不出來）。
        sentences = dict(zip(channel.flags, channel.flags_text.split("；"), strict=True))
        assert sentences["window_only_delay_screen"] == "反射只看直達音後的時間窗"
        assert sentences["geometry_material_conservative_screen"] == "反射用幾何與材料做保守篩選"


def _same_sentences(flags: tuple[str, ...], text: str, table_flags: tuple[str, ...], table_text: str) -> bool:
    """一個旗標一句（句數對不上就紅），而且這裡每個旗標那一句都跟主表同一個旗標那一句一樣。"""
    mine = dict(zip(flags, text.split("；"), strict=True))
    table = dict(zip(table_flags, table_text.split("；"), strict=True))
    return all(table.get(flag) == sentence for flag, sentence in mine.items())


def _pair(role: str, group: str, reference: str, receiver: str, metric: str) -> PairView:
    return PairView(role=role, speaker_id=role, metric=metric, group=group, receiver_id=receiver,
                    reference_id=reference, value=0.1, value_text="0.10", unit="dB", limit=3.0,
                    limit_text="3.00", excess_text="未超過", over_limit=False, baseline_note="",
                    status_text="未超過")


def test_pair_choices_one_per_pair_in_display_order(result: SchemeResult) -> None:
    ends = (("surrounding_to_surrounding", "up", "front"), ("primary_to_surrounding", "main", "down"),
            ("primary_to_surrounding", "main", "front"), ("surrounding_to_surrounding", "back", "front"),
            ("primary_to_surrounding", "main", "seat9"))
    # 同一對有三種量：選項只要一個；輸入順序故意亂排。
    pairs = tuple(_pair("left", group, reference, receiver, metric)
                  for metric in ("tilt", "ripple_rms", "overall_level")
                  for group, reference, receiver in ends)
    choices = _pair_choices(pairs, "main")
    assert [(item.group, item.text) for item in choices] == [
        ("primary_to_surrounding", "主位 ↔ 主位前方"), ("primary_to_surrounding", "主位 ↔ 主位下方"),
        ("primary_to_surrounding", "主位 ↔ seat9"),
        ("surrounding_to_surrounding", "主位前方 ↔ 主位後方"),
        ("surrounding_to_surrounding", "主位前方 ↔ 主位上方")]
    # 按鈕字：主位對周圍點只寫周圍點那一端（組名另外寫一次）；周圍點彼此在選單裡，照整對的名字。
    assert [item.button_text for item in choices] == ["主位前方", "主位下方", "seat9",
                                                      "主位前方 ↔ 主位後方", "主位前方 ↔ 主位上方"]
    # 主位是哪一個照傳進來的代號，不看代號排在前面還是後面。
    reversed_pair = (_pair("left", "primary_to_surrounding", "back", "main", "tilt"),)
    assert _pair_choices(reversed_pair, "main")[0].button_text == "主位後方"
    assert _pair_choices(reversed_pair, "back")[0].button_text == "主位"
    # 代號照原樣留著：頁面拿它找曲線與明細。
    assert {(item.group, item.reference_id, item.receiver_id) for item in choices} == set(ends)
    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    assert {(item.role, item.group, item.receiver_id, item.reference_id)
            for item in view.listening_area.pair_choices} == {
        (item.role, item.group, item.receiver_id, item.reference_id) for item in view.listening_area.pairs}


def test_ranking_line_says_which_categories_are_not_counted(result: SchemeResult) -> None:
    from aosr.gui.result_view import _ranking_text
    from aosr.scoring.category_registry import NotEvaluatedReason
    from aosr.scoring.ranking_models import MissingCategory

    view = build_result_view(result, quality_targets_path=config_path("quality_targets.toml"))
    # 考卷結果可排名、低頻拖尾與空間感尚未評估：那一行不說「缺的類：無」（跟表上兩個「尚未評估」互相矛盾），
    # 而是說哪幾類還沒評、不算進總代價；沒有淘汰原因就不寫「淘汰原因：無」。
    assert view.ranking_text == "排名位置：可排名；尚未評估、不算進總代價：低頻拖尾、空間感"
    assert {LABELS[item.category] for item in view.categories if item.state == "not_evaluated"} == {"低頻拖尾", "空間感"}
    # 其他沒有代價的類照表上的狀態另列一段。
    unavailable = tuple(item.model_copy(update={"state": "unavailable", "state_label": "不可估", "cost": None,
                                                "cost_text": "—"}) if item.category == "reverberation" else item
                        for item in view.categories)
    assert _ranking_text("rankable", (), (), unavailable) == (
        "排名位置：可排名；尚未評估、不算進總代價：低頻拖尾、空間感；不可估、不算進總代價：殘響")
    # 淘汰或擋住排名時：原因與擋住排名的類照伺服器的標籤列出；沒有總代價就不提總代價。
    assert _ranking_text("eliminated", ("external_floor_failed",), (), view.categories) == (
        "排名位置：淘汰；淘汰原因：外部底線未過；尚未評估：低頻拖尾、空間感")
    missing = (MissingCategory(category=QualityCategory.REVERBERATION, reason=NotEvaluatedReason.COST_NOT_COMPUTED,
                               evaluator_reason_codes=()),)
    blocked = tuple(item.model_copy(update={"cost": None}) if item.category == "reverberation" else item
                    for item in view.categories)
    assert _ranking_text("not_evaluated", (), missing, blocked) == (
        "排名位置：未評估；擋住排名的類：殘響（代價尚未算出）；尚未評估：低頻拖尾、空間感")


def test_labels_have_no_bare_english_abbreviation() -> None:
    # 「起伏 RMS 差」只有英文縮寫、旁邊沒有中文：改成中文，均方根放括號。
    assert LABELS["ripple_rms"] == "起伏差（均方根）"
    assert not any("RMS" in value for value in LABELS.values())
