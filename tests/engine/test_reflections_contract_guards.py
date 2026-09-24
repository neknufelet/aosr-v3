"""#351 真評估輸出的深層格式守門。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from aosr.scoring.contract import EvaluationState
from aosr.scoring.direction_zones import DirectionZone
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from test_reflections import (  # type: ignore[import-not-found]  # expires=2026-10-24 reason=pytest-test-module-path
    _evaluate, _pair,
)


def _real_payload() -> ReflectionsAndEchoPayload:
    result = _evaluate(_pair())
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    return result.payload


def _revalidate(payload: ReflectionsAndEchoPayload) -> ReflectionsAndEchoPayload:
    return ReflectionsAndEchoPayload.model_validate(payload.model_dump(mode="python"))


def test_each_zone_last_frequency_must_fit_declared_range() -> None:
    payload = _real_payload().model_copy(update={"frequency_range_hz": (300.0, 7000.0)})
    assert all(zone.points[-1].frequency_hz > 7000.0
               for channel in payload.channels for zone in channel.zones)
    with pytest.raises(ValidationError, match="逐點頻率必須落在頻率範圍內"):
        _revalidate(payload)


def test_second_channel_outside_path_cannot_claim_inside_window() -> None:
    payload = _real_payload()
    channel = payload.channels[1]
    index = next(index for index, path in enumerate(channel.reflections) if not path.within_window)
    paths = tuple(path.model_copy(update={"within_window": True}) if position == index else path
                  for position, path in enumerate(channel.reflections))
    changed = channel.model_copy(update={"reflections": paths})
    payload = payload.model_copy(update={"channels": (payload.channels[0], changed)})
    with pytest.raises(ValidationError, match="反射路徑窗內旗標與延遲不一致"):
        _revalidate(payload)


def test_second_channel_path_zone_must_match_angle() -> None:
    payload = _real_payload()
    channel = payload.channels[1]
    index = next(index for index, path in enumerate(channel.reflections) if not path.within_window)
    original = channel.reflections[index]
    wrong = DirectionZone.REAR if original.zone is not DirectionZone.REAR else DirectionZone.FRONT
    paths = tuple(path.model_copy(update={"zone": wrong}) if position == index else path
                  for position, path in enumerate(channel.reflections))
    changed = channel.model_copy(update={"reflections": paths})
    payload = payload.model_copy(update={"channels": (payload.channels[0], changed)})
    with pytest.raises(ValidationError, match="反射路徑分區與角度不一致"):
        _revalidate(payload)


def test_second_wall_pair_exact_subband_center_must_match() -> None:
    payload = _real_payload()
    pair = payload.wall_pairs[1]
    original = pair.bands[5]
    center = original.frequency_hz + 1.0
    assert original.lower_hz < center < original.upper_hz
    bands = tuple(band.model_copy(update={"frequency_hz": center}) if index == 5 else band
                  for index, band in enumerate(pair.bands))
    changed = pair.model_copy(update={"bands": bands})
    payload = payload.model_copy(update={"wall_pairs": (
        payload.wall_pairs[0], changed, *payload.wall_pairs[2:])})
    with pytest.raises(ValidationError, match="三對平行牆的子帶清單必須一致"):
        _revalidate(payload)


def test_second_wall_pair_names_cannot_be_reversed() -> None:
    payload = _real_payload()
    pair = payload.wall_pairs[1]
    changed = pair.model_copy(update={"walls": (pair.walls[1], pair.walls[0])})
    payload = payload.model_copy(update={"wall_pairs": (
        payload.wall_pairs[0], changed, *payload.wall_pairs[2:])})
    with pytest.raises(ValidationError, match="必須保留三對平行牆"):
        _revalidate(payload)
