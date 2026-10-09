"""#559 第七步：桌面到達方向跨登記簿仰角界線的獨立手算考卷。"""
from __future__ import annotations

import math
from dataclasses import asdict, replace

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.furniture import FaceDirection
from aosr.geometry.shoebox import Point
from aosr.physics import report_io
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_io import PathTableSection, SceneSection
from aosr.physics.report_path_table import PathTableData, build_path_table
from aosr.physics.report_source import SourceModelSection
from aosr.scoring.contract import EvaluationState
from aosr.scoring.direction_zones import DirectionZone, ZoneLimits, classify, listening_angles, zone_limits
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from tests.engine import _furniture_energy_cases as case, _reflection_fixtures as reflections
from tests.engine._directivity import DIRECTIVITY

PURPOSE = "dedicated_two_channel_listening_room"


def _limits() -> ZoneLimits:
    return zone_limits(load_quality_targets(config_path("quality_targets.toml")).purpose(PURPOSE))[0]


def _receiver(offset_m: float) -> Point:
    # 夾具：聲源與耳高都是 1.05 m；桌底 0.5 m、板厚 0.1 m，頂面 0.6 m。
    # 鏡像源高 2×0.6−1.05=0.15；耳朵到鏡像源的垂直差是 0.9 m。
    # D界線=(耳高+聲源高−2×桌頂)/tan(登記簿界線)。現在 D=1.55884572681199 m。
    # x=2.4+D±0.001；±1 mm 避開左右聲道浮點尾差造成的正好界線分區。
    desk = case.desk()
    top = desk.bottom_center_m[2] + desk.height_m
    height = case.RECEIVER.z + case.SOURCE.z - 2.0 * top
    distance = height / math.tan(math.radians(_limits().vertical_min_abs_elevation_deg))
    return Point(case.SOURCE.x + distance + offset_m, case.RECEIVER.y, case.RECEIVER.z)


def _hand_geometry(source: Point, receiver: Point) -> tuple[tuple[float, float, float], Point, float]:
    # 兩端等高，所以反射點在水平中點，z=桌頂；答案不呼叫產品反射點／方向函式。
    top = case.desk().bottom_center_m[2] + case.desk().height_m
    hit = Point((source.x + receiver.x) / 2.0, (source.y + receiver.y) / 2.0, top)
    arrival = (hit.x - receiver.x, hit.y - receiver.y, hit.z - receiver.z)
    length = math.sqrt(sum(value * value for value in arrival))
    vector = tuple(value / length for value in arrival)
    elevation = math.degrees(math.atan2(top - receiver.z, math.hypot(arrival[0], arrival[1])))
    return (vector[0], vector[1], vector[2]), hit, elevation


def _table(source: Point, receiver: Point) -> PathTableData:
    return build_path_table(source_model=case.OMNI, room=case.ROOM, source=source, receiver=receiver,
        sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO_C, impedance_by_wall=case.WALLS,
        frequencies_hz=reflections._AXIS, scattering_coefficient=reflections._SCATTERING,
        reflection_order_k=3, furniture=case.lane_inputs((case.desk(),), reflections._AXIS))


@pytest.mark.parametrize(("offset_m", "expected_zone"), (
    (-0.001, DirectionZone.VERTICAL), (0.001, DirectionZone.FRONT),
), ids=("near-vertical", "far-front"))
def test_furniture_arrival_direction_crosses_hand_boundary(offset_m: float, expected_zone: DirectionZone) -> None:
    receiver = _receiver(offset_m)
    row = next((row for row in _table(case.SOURCE, receiver).rows
                if row.furniture_id == "desk" and row.furniture_face == FaceDirection.TOP), None)
    assert row is not None, "兩側都必須保留桌面頂反射列"
    expected_vector, hit, expected_elevation = _hand_geometry(case.SOURCE, receiver)
    # 手算反射點 x=3.1789228634±0.0005，落在桌板 x=2.6..3.4、y=1.2..2.8。
    assert 2.6 < hit.x < 3.4 and 1.2 < hit.y < 2.8
    assert row.reflection_point_m == pytest.approx(hit.as_tuple(), abs=1e-12)
    assert row.direction_vector == pytest.approx(expected_vector, abs=1e-12)
    # 聆聽軸朝 −x，桌面來聲在耳下，仰角必須負：約 −30.015923／−29.984092 度。
    azimuth, elevation = listening_angles(row.direction_vector, (-1.0, 0.0))
    assert azimuth == pytest.approx(0.0, abs=1e-12)
    assert elevation == pytest.approx(expected_elevation, abs=1e-12)
    assert row.direction_angles.elevation_deg == pytest.approx(expected_elevation, abs=1e-12)
    assert elevation < 0.0
    boundary = _limits().vertical_min_abs_elevation_deg
    assert (abs(elevation) > boundary) if offset_m < 0.0 else (abs(elevation) < boundary)
    assert classify(azimuth, elevation, _limits()) is expected_zone


def _record(role: str, source: Point, receiver: Point) -> ReflectionInput:
    inputs = report_io.load_input_document({
        "room_m": {"Lx": case.ROOM.Lx, "Ly": case.ROOM.Ly, "Lz": case.ROOM.Lz},
        "source_model": {"kind": "omnidirectional"},
        "source_m": dict(zip(("x", "y", "z"), source.as_tuple(), strict=True)),
        "receiver_m": dict(zip(("x", "y", "z"), receiver.as_tuple(), strict=True)),
        "sound_speed_m_s": case.SPEED, "density_kg_m3": case.DENSITY,
        "impedance_pa_s_per_m_by_wall": {wall.wall_name(): value for wall, value in case.WALLS.items()},
        "reflection_order_k": 3, "furniture": [case.desk().model_dump(mode="json")],
    }, load_capabilities(config_path("capabilities.toml")), DIRECTIVITY)
    # 既有夾具只提供非判區的報表欄位；真正場景、路徑表、窗與覆蓋篩選都重新建。
    template = reflections._record(role, source.y, receiver_y=receiver.y, receiver_x=receiver.x)
    report = template.report.model_copy(update={
        "scene": SceneSection(source_model=SourceModelSection.from_spec(case.OMNI, None),
            scene_fingerprint=report_io.scene_fingerprint(inputs), source_m=inputs.source_m,
            receiver_m=inputs.receiver_m),
        "path_table": PathTableSection.model_validate(asdict(_table(source, receiver))),
    })
    window = build_reflection_window(inputs, frequencies_hz=reflections._AXIS,
        scattering_coefficient=reflections._SCATTERING, window_s=template.window.window_s,
        contact_rel=case.CONTACT_REL) if template.window is not None else None
    return replace(template, report=report, window=window,
        screen=build_reflection_screen(inputs, reflections._AXIS), third_octave_decay=reflections._third_decay(report))


@pytest.mark.parametrize(("offset_m", "expected_zone"), (
    (-0.001, DirectionZone.VERTICAL), (0.001, DirectionZone.FRONT),
), ids=("near-vertical", "far-front"))
def test_reflection_evaluator_uses_furniture_arrival_direction(offset_m: float, expected_zone: DirectionZone) -> None:
    receiver = _receiver(offset_m)
    # 左聲道就是 A 的幾何；右聲道只在 y 增 1 mm，讓兩聲道聆聽軸有定義、仍朝 −x。
    # 右聲道水平距離增加約 0.32 μm，遠小於界線兩側的 1 mm；仍用各自手算答案。
    sources = {"left": case.SOURCE, "right": Point(case.SOURCE.x, case.SOURCE.y + 0.001, case.SOURCE.z)}
    records = tuple(_record(role, source, receiver) for role, source in sources.items())
    result = evaluate_reflections(reflections._group(), records, primary_receiver_id="main",
        candidate_id="hand-boundary", purpose=PURPOSE, quality_targets_path=config_path("quality_targets.toml"))
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    for channel in result.payload.channels:
        path = next((path for path in channel.reflections if path.wall_sequence == ("furniture",)), None)
        assert path is not None, "兩側每聲道都必須保留桌面反射"
        _, _, expected_elevation = _hand_geometry(sources[channel.role], receiver)
        assert path.listening_elevation_deg == pytest.approx(expected_elevation, abs=1e-12)
        assert path.listening_elevation_deg < 0.0
        assert path.zone is expected_zone
        assert path.within_window
