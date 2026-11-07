"""#1032：第三階段專用範圍、第一階段控制組與排除低於下界的目標。"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import pytest

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, SettingEntry, load_quality_targets
from aosr.scoring.gradient_objective_jax import (
    GradientObjectiveConstants, gradient_objective_constants, gradient_objective_from_pressures,
)
from aosr.scoring.gradient_objective_settings import gradient_objective_settings, gradient_objective_settings_from_section
from aosr.scoring.receiver_set import primary_cross
from aosr.scoring.stage_one_score import stage_one_score_settings, stage_one_settings_from_section

PURPOSE = "dedicated_two_channel_listening_room"
RANGE_KEY = "stage_three_gradient.scored_range_hz"
OLD_GRADIENT = "e43e0f93a65016ccb1320333c3f80d08d2114d54e4acd7207f7e941f5dfcf971"
OLD_STAGE_ONE = "2d9c47e3be7e3ffcb29f4c6f57fc17fb55da1c0fb77f7bb25a449cc2673f98ea"


def section_with_range(bounds: tuple[float, ...], *, status: str = "baseline", unit: str = "Hz") -> QualityPurpose:
    section = load_quality_targets(config_path("quality_targets.toml")).stage_one(PURPOSE)
    entry = SettingEntry.model_validate({"key": RANGE_KEY, "value": bounds, "unit": unit, "status": status,
                                         "source_kind": "product_choice", "source": "#1032 考卷",
                                         "source_id": "test", "source_version": "v1", "locator": "test",
                                         "conditions": "test", "frequency_range_hz": (20.0, 300.0), "context": "test",
                                         "verification_digest": "sha256:" + "a" * 64})
    return section.model_copy(update={"setting": (*[item for item in section.setting if item.key != RANGE_KEY], entry)})


def test_stage_three_range_has_fresh_identity_and_stage_one_is_unchanged() -> None:
    registry = load_quality_targets(config_path("quality_targets.toml"))
    original = stage_one_score_settings(registry, PURPOSE)
    gradient = gradient_objective_settings(registry, PURPOSE)
    assert original.fingerprint == OLD_STAGE_ONE
    assert original.scored_range_hz == (20.0, 300.0)
    assert gradient.stage_one.scored_range_hz == (30.0, 300.0)
    assert gradient.stage_one.fingerprint != original.fingerprint
    assert gradient.fingerprint != OLD_GRADIENT
    entry = registry.stage_one(PURPOSE).entry(RANGE_KEY)
    assert isinstance(entry, SettingEntry)
    assert entry.status == "baseline" and entry.source_kind == "product_choice" and entry.unit == "Hz"


@pytest.mark.parametrize("bounds", [(0.0, 300.0), (-1.0, 300.0), (30.0, 30.0), (300.0, 30.0), (19.0, 300.0)])
def test_invalid_stage_three_range_is_rejected(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="第三階段|計分範圍"):
        gradient_objective_settings_from_section(section_with_range(bounds))


def test_range_below_stage_one_names_the_lower_bound() -> None:
    with pytest.raises(ValueError, match="不得低於第一階段下界"):
        gradient_objective_settings_from_section(section_with_range((19.0, 300.0)))


@pytest.mark.parametrize("bounds,unit", [((30.0,), "Hz"), ((30.0, 300.0), "dB")])
def test_range_shape_and_unit_are_checked(bounds: tuple[float, ...], unit: str) -> None:
    with pytest.raises(ValueError, match=RANGE_KEY):
        gradient_objective_settings_from_section(section_with_range(bounds, unit=unit))


def test_range_value_and_status_both_change_stage_three_fingerprints() -> None:
    base = gradient_objective_settings_from_section(section_with_range((30.0, 300.0)))
    for section in (section_with_range((40.0, 300.0)), section_with_range((30.0, 300.0), status="calibrated")):
        changed = gradient_objective_settings_from_section(section)
        assert changed.fingerprint != base.fingerprint
        assert changed.stage_one.fingerprint != base.stage_one.fingerprint
        assert stage_one_settings_from_section(section).fingerprint == OLD_STAGE_ONE


def test_peak_below_thirty_changes_old_objective_but_not_stage_three() -> None:
    registry = load_quality_targets(config_path("quality_targets.toml"))
    settings = gradient_objective_settings(registry, PURPOSE)
    original = stage_one_score_settings(registry, PURPOSE)
    axis = np.asarray(low_frequency_axis_frequencies(LowFrequencyAxis.VERIFICATION)[0])
    receivers = primary_cross(primary_position_m=(3.0, 2.0, 1.2), offset_m=0.1)
    flat = np.ones((len(axis), 2, len(receivers.points)), dtype=np.complex128)
    peak = flat.copy()
    peak[axis == 22.0] *= 100.0
    current = gradient_objective_constants(axis, settings, receivers, 2)
    old = gradient_objective_constants(axis, dataclasses.replace(settings, stage_one=original), receivers, 2)
    def value(constants: GradientObjectiveConstants, pressure: NDArray[np.complex128]) -> float:
        return float(gradient_objective_from_pressures(constants, jnp.asarray(pressure)).objective_total)
    assert value(old, peak) > value(old, flat)
    assert value(current, peak) == value(current, flat)


def test_stage_three_reads_both_registry_endpoints_without_changing_stage_one() -> None:
    section = section_with_range((40.0, 280.0))
    changed = gradient_objective_settings_from_section(section)
    assert changed.stage_one.scored_range_hz == (40.0, 280.0)
    assert stage_one_settings_from_section(section).fingerprint == OLD_STAGE_ONE


def test_stage_three_range_is_required_and_not_hardcoded() -> None:
    import ast
    from pathlib import Path
    from aosr.scoring import gradient_objective_settings as module
    section = section_with_range((30.0, 300.0))
    without = section.model_copy(update={"setting": tuple(entry for entry in section.setting if entry.key != RANGE_KEY)})
    with pytest.raises(KeyError, match=RANGE_KEY):
        gradient_objective_settings_from_section(without)
    assert module.__file__ is not None
    nodes = ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8")))
    assert not [node.value for node in nodes if isinstance(node, ast.Constant) and isinstance(node.value, (float, int))
                and node.value in (20.0, 30.0, 300.0)]
