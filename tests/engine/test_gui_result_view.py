"""結果頁的逐點差值與資料來源。"""
from __future__ import annotations

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.compare import compare_results
from aosr.reporting.display import level_db
from aosr.reporting.result import SchemeResult
from aosr.reporting.result_view import _alerts, build_result_view
from aosr.scoring.contract import ListeningAreaStabilityPayload, QualityCategory
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
            assert max(pair.value for pair in selected) == pytest.approx(
                aggregate.worst_deviation.value
            )
            assert sum(pair.value * pair.weight for pair in selected) / sum(
                pair.weight for pair in selected
            ) == pytest.approx(aggregate.weighted_mean_deviation)
            worst = aggregate.worst_deviation
            assert any(pair.receiver_id == worst.receiver.receiver_id and
                       pair.reference_id == worst.reference.receiver_id and
                       pair.value == pytest.approx(worst.value) for pair in selected)


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


def test_result_page_script_only_renders_server_values() -> None:
    from aosr.gui.app import STATIC

    script = (STATIC / "results.js").read_text()
    html = (STATIC / "results.html").read_text()
    assert "innerHTML" not in script
    assert "Math.log" not in script and "Math.pow" not in script
    assert "uPlot" in script and "vendor/uplot" in html
    assert "/api/results/" in script


def test_alert_uses_structured_exact_value_not_rounded_note() -> None:
    alert = ListeningAreaReviewAlert(
        category=QualityCategory.LISTENING_AREA_STABILITY, role="left",
        speaker_id="left", metric="ripple_rms", group="primary_to_surrounding",
        receiver_id="right10", reference_id="main", deviation=3.0001,
        limit=3.0, note="差 3 dB，超過暫定線 3 dB；尚未正式校準",
    )
    view = _alerts((alert,))[0]
    assert view.receiver_id == "right10" and view.reference_id == "main"
    assert view.excess_text == "0.000100000000"
    assert ("差值", "3.000100000000") in view.fields
    assert ("暫定線", "3.000000000000") in view.fields
    assert alert.note not in str(view.model_dump())
