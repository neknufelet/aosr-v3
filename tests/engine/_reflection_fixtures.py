"""#351 反射評估器考卷的共用夾具（原樣從 test_reflections.py 搬出，騰行數；#505 第二刀）。

test_reflections.py 用同名轉出，其他考卷照舊 ``from tests.engine import test_reflections as fixtures`` 取用。
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ, LowFrequencyAxis, planned_band_points
from aosr.config.paths import config_path
from aosr.physics import report_io
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_path_table import build_path_table
from aosr.physics.report_io import (
    BandRow, CapabilitySection, PathTableSection, PointRow, ReportOutput, SceneSection,
    TopFields,
)
from aosr.physics.room_paths import NUMERICALLY_GUARDED_ORDER_K
from aosr.physics.third_octave_decay import (
    ThirdOctaveDecay, ThirdOctaveDecayRow, subband_weighted_mean, third_octave_bands,
)
from aosr.scoring.channel_matching import ChannelComparison, ChannelDefinition, ChannelGroup
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, Flag, MetricState, ReasonCode
from aosr.scoring.direction_zones import DirectionZone
import aosr.scoring.reflections as reflections
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload, ReflectionSource
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections


_AXIS = (250.0, 300.0, 500.0, 800.0, 1000.0, 1250.0,
         2000.0, 4000.0, 8000.0, 9000.0)
_SCATTERING = tuple(0.2 for _ in _AXIS)
_ROOM = {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0}
_SMALL = {"Lx": 4.5, "Ly": 3.5, "Lz": 2.6}
_WALLS = {"floor": 900.0, "ceiling": 1300.0, "x0": 1646.4,
          "xL": 2500.0, "y0": 4000.0, "yL": 6000.0}


def _band(center: float) -> BandRow:
    return BandRow(
        center_frequency_hz=center, fem_energy=1.0, fem_point_count=1,
        direct_energy=1.0, reflected_energy=0.0, interference_energy=0.0,
        late_energy=0.0, geometric_energy=1.0, fem_contribution=1.0,
        geometric_contribution=0.0, total_energy=1.0, w_fem=1.0, w_geo=0.0,
        f_s_hz=200.0, capped_by_upper_limit=False, t20_s=1.0,
        t20_unavailable_reason=None, t30_s=1.2, t30_unavailable_reason=None,
    )


def _third_decay(report: ReportOutput) -> ThirdOctaveDecay:
    parents = {band.center_frequency_hz: band for band in report.bands}
    parents.update({band.octave_center_hz: _band(band.octave_center_hz)
                    for band in third_octave_bands()
                    if band.octave_center_hz not in parents})
    return ThirdOctaveDecay(
        scene_fingerprint=report.scene.scene_fingerprint,
        rows=tuple(ThirdOctaveDecayRow(
            band=band, point_count=len(planned_band_points(
                GEOMETRIC_LANE_FREQUENCIES_HZ, band.lower_hz, band.upper_hz)),
            t20_s=parents[band.octave_center_hz].t20_s,
            t20_unavailable_reason=parents[band.octave_center_hz].t20_unavailable_reason,
            t30_s=parents[band.octave_center_hz].t30_s,
            t30_unavailable_reason=parents[band.octave_center_hz].t30_unavailable_reason,
            missing_planned_hz=(), unplanned_hz=(),
            t20_unavailable_cause=("octave_band" if parents[band.octave_center_hz].t20_unavailable_reason else None),
            t30_unavailable_cause=("octave_band" if parents[band.octave_center_hz].t30_unavailable_reason else None),
        ) for band in third_octave_bands()),
    )


def _input(source_y: float, receiver_y: float = 1.9, receiver_x: float = 3.2,
           room: dict[str, float] = _ROOM, order: int = 3) -> report_io.ReportInput:
    return report_io.load_input_document({
        "room_m": room,
        "source_m": {"x": 1.0, "y": source_y, "z": 1.2},
        "receiver_m": {"x": receiver_x, "y": receiver_y, "z": 1.2},
        "sound_speed_m_s": 343.0, "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": _WALLS, "reflection_order_k": order,
    }, load_capabilities(config_path("capabilities.toml")))


def _record(role: str, source_y: float, receiver: str = "main", *,
            room: dict[str, float] = _ROOM, receiver_y: float = 1.9,
            receiver_x: float = 3.2,
            window_s: float = 15.0 / 1000.0, order: int = 3,
            axis: tuple[float, ...] = _AXIS) -> ReflectionInput:
    inputs = _input(source_y, receiver_y, receiver_x, room, order)
    solved = report_io.solver_inputs(inputs)
    scattering = tuple(0.2 for _ in axis)
    table = build_path_table(
        room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.density_kg_m3 * solved.sound_speed_m_s,
        impedance_by_wall=solved.impedance_by_wall,
        frequencies_hz=axis, scattering_coefficient=scattering,
        reflection_order_k=inputs.reflection_order_k,
    )
    report = ReportOutput(
        scene=SceneSection(
            scene_fingerprint=report_io.scene_fingerprint(inputs),
            source_m=inputs.source_m, receiver_m=inputs.receiver_m,
        ),
        capability=CapabilitySection(
            frequency_hz=(20.0, 20000.0), outputs=("t20_s",),
            status="experimental", evidence=(),
        ),
        top=TopFields(
            f_s_hz=200.0, crossover_lower_hz=100.0,
            crossover_upper_hz=300.0, capped_by_upper_limit=False,
            reflection_order_k=inputs.reflection_order_k,
            low_frequency_axis=LowFrequencyAxis.SEARCH,
            eyring_t60_by_band_s={"1000.0": 1.0}, room_volume_m3=72.0,
            schroeder_band_count=1,
        ),
        bands=(_band(500.0), _band(1000.0), _band(2000.0), _band(4000.0), _band(8000.0)),
        points=tuple(PointRow(
            frequency_hz=frequency, fem_energy=None,
            direct_energy=1.0, reflected_energy=0.0, interference_energy=0.0,
            late_energy=0.0, scattering=scatter, geometric_energy=1.0,
            w_fem=0.0, w_geo=1.0, total_energy=1.0,
        ) for frequency, scatter in zip(axis, scattering, strict=True)),
        path_table=PathTableSection.model_validate(asdict(table)),
    )
    window = build_reflection_window(
        inputs, frequencies_hz=axis, scattering_coefficient=scattering,
        window_s=window_s,
    )
    return ReflectionInput(
        role=role, receiver_id=receiver, report=report,
        screen=build_reflection_screen(inputs, axis), window=window,
        third_octave_decay=_third_decay(report),
        report_id=f"{role}-{receiver}", engine_commit="engine-commit",
        speaker_id=role,
    )


def _group() -> ChannelGroup:
    return ChannelGroup(
        channels=(ChannelDefinition(role="left", speaker_id="left"),
                  ChannelDefinition(role="right", speaker_id="right")),
        comparisons=(ChannelComparison(left_role="left", right_role="right"),),
        feature_match_tolerance_hz=0.0,
    )


def _evaluate(records: tuple[ReflectionInput, ...], path: Path | None = None) -> CategoryEvaluation:
    return evaluate_reflections(
        _group(), records, primary_receiver_id="main", candidate_id="candidate-a",
        purpose="dedicated_two_channel_listening_room",
        quality_targets_path=path or config_path("quality_targets.toml"),
    )


def _pair() -> tuple[ReflectionInput, ReflectionInput]:
    return _record("left", 1.3), _record("right", 2.5)


def _set_entry(tmp_path: Path, key: str, value: str) -> Path:
    text = config_path("quality_targets.toml").read_text()
    pattern = rf'(key = "{re.escape(key)}"\nvalue = )[^\n]+'
    changed, count = re.subn(pattern, lambda match: match.group(1) + value, text)
    if count != 1:
        raise ValueError(f"登記簿設定 {key} 未唯一命中")
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed)
    return path


def _small_reflection_registry(tmp_path: Path, key: str, field: str, value: str) -> Path:
    source = config_path("quality_targets.toml").read_text()
    needed = (
        "reflections_and_echo.window_upper_ms", "reflections_and_echo.frequency_range_hz",
        "reflections_and_echo.flutter_decay_db",
        "reflections_and_echo.flutter_alert_band_centers_hz",
        "direction_zones.vertical_min_abs_elevation_deg",
        "direction_zones.front_max_abs_azimuth_deg",
        "direction_zones.rear_min_abs_azimuth_deg",
    )
    chunks = source.split("[[purpose.setting]]\n")
    blocks = []
    for name in needed:
        block = next(chunk for chunk in chunks[1:] if chunk.startswith(f'key = "{name}"\n'))
        block = block.split("\n[[purpose.", 1)[0].rstrip()
        if name == key:
            block, count = re.subn(rf"(?m)^{field} = [^\n]+$",
                                   f"{field} = {value}", block)
            if count != 1:
                raise ValueError(f"登記簿欄位 {key}.{field} 未唯一命中")
        blocks.append("[[purpose.setting]]\n" + block)
    header = ('schema_version = 1\n[[purpose]]\n'
              'name = "dedicated_two_channel_listening_room"\n'
              'target = []\nweight = []\nqualification = []\n')
    path = tmp_path / "quality_targets.toml"
    path.write_text(header + "\n".join(blocks) + "\n")
    return path


def _vary_pair(records: tuple[ReflectionInput, ReflectionInput], values: tuple[float, ...]
               ) -> tuple[ReflectionInput, ReflectionInput]:
    changed = []
    for item in records:
        assert item.screen is not None
        pair = item.screen.pairs[0]
        first = pair.model_copy(update={"round_trip_retained_energy": values})
        changed.append(replace(item, screen=item.screen.model_copy(update={
            "pairs": (first, *item.screen.pairs[1:])
        })))
    return changed[0], changed[1]
