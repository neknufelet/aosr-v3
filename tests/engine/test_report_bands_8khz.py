"""#437 報表 8000 Hz 八度帶與殘響目標的整合考卷。"""

from __future__ import annotations

import math

import pytest

from aosr.config import frequency_axis
from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.scoring.contract import CONTRACT_SCHEMA_VERSION, CategoryEvaluation
from aosr.scoring.reverberation_cost import cost_reverberation_evaluation
from tests.engine._placement import POINT_PLACEMENT


_PURPOSE = "dedicated_two_channel_listening_room"


def _reverberation_columns() -> tuple[tuple[float, ...], ...]:
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE)
    columns = []
    for key in (
        "reverberation.target_band_centers_hz",
        "reverberation.target_t20_nominal_s_by_band",
        "reverberation.target_t20_tolerance_s_by_band",
    ):
        entry = purpose.entry(key)
        assert isinstance(entry, SettingEntry)
        assert isinstance(entry.value, tuple)
        columns.append(tuple(float(value) for value in entry.value))
    return tuple(columns)


def _cost_at_8khz(t20_s: float) -> CategoryEvaluation:
    registry = load_quality_targets(config_path("quality_targets.toml"))
    metric = {
        "value": t20_s,
        "unit": "s",
        "state": "measured",
        "reason_codes": [],
        "reason": None,
    }
    evaluation = CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": "8khz-boundary",
            "scene_fingerprint": "a" * 64,
            "placement": POINT_PLACEMENT,
            "category": "reverberation",
            "state": "measured",
            "payload": {
                "category": "reverberation",
                "bands": [
                    {
                        "center_frequency_hz": 8000.0,
                        "band_range_hz": [8000.0 / math.sqrt(2.0), 8000.0 * math.sqrt(2.0)],
                        "schroeder_position": "above",
                        "model_validation_status": "validated",
                        "t20": metric,
                        "t30": metric,
                        "fitting_difference": {**metric, "value": 1.0, "unit": "1"},
                    }
                ],
                "adjacent_band_changes": [],
                "logarithm_base": 2.0,
            },
            "raw_quantities": [{"name": "t20_8000_hz", "value": t20_s, "unit": "s"}],
            "category_cost": None,
            "flags": [],
            "reason_codes": [],
            "evaluator_version": "8khz-boundary-fixture",
            "settings_fingerprint": registry.fingerprint,
            "provenance": {
                "report_id": "8khz-boundary-report",
                "engine_commit": "fixture-engine",
                "speaker_id": "left",
                "receiver_id": "main-seat",
            },
        }
    )
    return cost_reverberation_evaluation(
        evaluation,
        registry.purpose(_PURPOSE),
        registry.fingerprint,
    )


def test_report_centers_reach_8khz_by_octave_steps() -> None:
    """拔掉 8000 Hz，或中心清單不再逐帶倍增，本題會紅。"""
    centers = frequency_axis.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ

    assert centers[0] == 125.0
    assert centers[-1] == 8000.0
    assert all(upper == 2.0 * lower for lower, upper in zip(centers, centers[1:]))


def test_dense_axis_reaches_the_8khz_band_upper_edge_without_crossing_it() -> None:
    """拿掉 8000 Hz 密軸，或把半開上緣多算一點，本題會紅。"""
    last_frequency = frequency_axis.GEOMETRIC_BAND_FREQUENCIES_HZ[-1]

    assert 11313.5 <= last_frequency < 11314.0


def test_reverberation_registry_columns_align_and_include_8khz() -> None:
    """三表錯位、容許量非正，或漏掉 8000 Hz 目標，本題會紅。"""
    centers, nominal, tolerance = _reverberation_columns()

    assert centers[-1] == 8000.0
    assert len(centers) == len(nominal) == len(tolerance)
    assert all(width > 0.0 for width in tolerance)
    index = centers.index(8000.0)
    assert nominal[index] - tolerance[index] == pytest.approx(0.25)
    assert nominal[index] + tolerance[index] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("t20_s", "expected_direction", "is_penalized"),
    (
        (0.425, "within_range", False),
        # 0.27 落在 8000 帶的 0.25–0.6 裡、卻在中頻的 0.3–0.6 外：誤用中頻區間會罰，這一例才分得出來。
        (0.27, "within_range", False),
        (0.24, "below_range", True),
        (0.61, "above_range", True),
    ),
)
def test_reverberation_cost_uses_the_8khz_quarter_to_point_six_interval(
    t20_s: float,
    expected_direction: str,
    is_penalized: bool,
) -> None:
    """8000 Hz 改用別帶區間，區間內會被罰或 0.24／0.61 秒會漏罰。"""
    result = _cost_at_8khz(t20_s)
    assert result.category_cost is not None
    component = result.category_cost.components["t20_target_interval.8000Hz"]

    assert (component > 0.0) is is_penalized
    assert result.category_cost.component_directions["t20_target_interval.8000Hz"] == (
        expected_direction
    )


def test_late_energy_capability_row_spans_exactly_the_report_band_centers() -> None:
    """晚期混響入口是逐帶的：能力表那列的頻率範圍必須是報表帶清單的首尾中心，帶清單加了一帶而表沒跟上就紅。"""
    table = load_capabilities(config_path("capabilities.toml"))
    entry = next(item for item in table.entry if item.name == "late_energy")
    centers = frequency_axis.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ

    assert entry.capability
    for capability in entry.capability:
        assert capability.frequency_hz == (centers[0], centers[-1])
