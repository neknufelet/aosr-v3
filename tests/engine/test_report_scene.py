"""票 #415：三路接合報表的場景身分與逐份座標。"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import (
    report_io,
    report_output,
    three_lane_report,
    three_lane_report_cli,
)
from aosr.physics.report_io import ReportInput
from aosr.physics.report_source import AnalyticAxisymmetricInput, SourceModelKind, SourceModelSpec


_WALL_NAMES = tuple(wall.wall_name() for wall in Wall.all())


def test_output_assembly_has_its_own_module_without_report_io_reexport() -> None:
    """輸出組裝住 ``report_output``；``report_io`` 不准再匯出一次（兩個入口遲早各走各的）。"""
    assert callable(report_output.output_from_report)
    assert not hasattr(report_io, "output_from_report")


def test_every_report_input_field_is_either_shared_scene_or_per_report() -> None:
    """替報表輸入新增一格的人一定得決定它進不進場景指紋；漏了決定這一題就紅。"""
    declared = set(report_io.SCENE_FINGERPRINT_FIELDS) | set(
        report_io.PER_REPORT_INPUT_FIELDS
    )
    assert declared == set(ReportInput.model_fields)
    assert not set(report_io.SCENE_FINGERPRINT_FIELDS) & set(
        report_io.PER_REPORT_INPUT_FIELDS
    )


def _document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
        "source_model": {"kind": "omnidirectional"},
        "source_m": {"x": 1.2, "y": 1.3, "z": 1.1},
        "receiver_m": {"x": 4.7, "y": 2.8, "z": 1.4},
        "sound_speed_m_s": 343.0,
        "density_kg_m3": 1.2,
        "impedance_pa_s_per_m_by_wall": {
            wall: 1646.4 for wall in _WALL_NAMES
        },
        "scattering_by_wall": {wall: 0.2 for wall in _WALL_NAMES},
        "reflection_order_k": 3,
    }
    document.update(overrides)
    return document


def _inputs(**overrides: object) -> ReportInput:
    return report_io.load_input_document(
        _document(**overrides), load_capabilities(config_path("capabilities.toml"))
    )


def _fake_fem_energy(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> tuple[float, ...]:
    del room, source, receiver, wall_impedances, density_kg_m3, sound_speed_m_s
    return tuple(frequency_hz * 3.0 + 7.0 for frequency_hz in frequencies_hz)


def _solved_report(
    monkeypatch: pytest.MonkeyPatch, inputs: ReportInput
) -> three_lane_report.ThreeLaneReport:
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    solved = report_io.solver_inputs(inputs)
    return three_lane_report.solve_three_lane_report(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=solved.room,
        source=solved.source,
        receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        density_kg_m3=solved.density_kg_m3,
        impedance_by_wall=solved.impedance_by_wall,
        scattering_by_wall=solved.scattering_by_wall,
        reflection_order_k=solved.reflection_order_k,
    )


@pytest.mark.parametrize(
    ("coordinate", "changed"),
    (
        ("source_m", {"x": 1.8, "y": 1.3, "z": 1.1}),
        ("receiver_m", {"x": 4.2, "y": 2.8, "z": 1.4}),
    ),
)
def test_coordinates_change_the_scene_section_but_not_its_fingerprint(
    monkeypatch: pytest.MonkeyPatch, coordinate: str, changed: dict[str, float]
) -> None:
    baseline = _inputs()
    inputs = _inputs(**{coordinate: changed})
    report = _solved_report(monkeypatch, baseline)

    output = report_output.output_from_report(report, inputs=inputs, with_points=False)

    assert report_io.scene_fingerprint(inputs) == report_io.scene_fingerprint(baseline)
    assert output.scene.source_m == inputs.source_m
    assert output.scene.receiver_m == inputs.receiver_m


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("room_m", {"Lx": 6.5, "Ly": 4.0, "Lz": 3.0}),
        ("sound_speed_m_s", 344.0),
        ("density_kg_m3", 1.21),
        (
            "impedance_pa_s_per_m_by_wall",
            {**{wall: 1646.4 for wall in _WALL_NAMES}, "floor": 1700.0},
        ),
        ("scattering_by_wall", {wall: 0.3 for wall in _WALL_NAMES}),
        ("reflection_order_k", 4),
        ("source_model", {
            "kind": SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value,
            "parameters": {
                "beta_limit": 3.2, "beta_corner_hz": 4000.0, "beta_exponent": 0.67,
                "power_floor_limit_db": -45.0, "power_floor_corner_hz": 2550.0,
                "power_floor_exponent": 1.12,
            },
            "aim_m": {"x": 4.7, "y": 2.8, "z": 1.4},
        }),
    ),
)
def test_each_shared_input_changes_the_scene_fingerprint(
    field: str, changed: object
) -> None:
    # 第二刀的載入政策拒收解析近似；聯合型別本身仍有合法的第二種值，
    # 用 model_copy 隔離政策後只考指紋序列化，證明模型種類確實進入共享身分。
    inputs = (
        _inputs().model_copy(update={"source_model": AnalyticAxisymmetricInput.model_validate(changed)})
        if field == "source_model" else _inputs(**{field: changed})
    )
    assert report_io.scene_fingerprint(inputs) != (
        report_io.scene_fingerprint(_inputs())
    )


def test_omitted_scattering_differs_from_explicit_zero_scattering() -> None:
    assert report_io.scene_fingerprint(_inputs(scattering_by_wall=None)) != (
        report_io.scene_fingerprint(
            _inputs(scattering_by_wall={wall: 0.0 for wall in _WALL_NAMES})
        )
    )


def test_wall_key_order_does_not_change_the_scene_fingerprint() -> None:
    """載入器本來就會把牆排好；這裡用不經驗證的 ``model_copy`` 硬塞反序的字典，
    直接咬指紋自己的鍵排序——少了那一步這一題會紅。"""
    loaded = _inputs()
    reversed_walls = dict(reversed(tuple(loaded.impedance_pa_s_per_m_by_wall.items())))
    assert list(reversed_walls) != list(loaded.impedance_pa_s_per_m_by_wall)
    shuffled = loaded.model_copy(
        update={"impedance_pa_s_per_m_by_wall": reversed_walls}
    )

    assert report_io.scene_fingerprint(shuffled) == report_io.scene_fingerprint(loaded)


def test_real_report_output_carries_the_input_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    report = _solved_report(monkeypatch, inputs)

    output = report_output.output_from_report(report, inputs=inputs, with_points=True)

    assert output.scene.scene_fingerprint == report_io.scene_fingerprint(inputs)
    assert output.scene.source_m == inputs.source_m
    assert output.scene.receiver_m == inputs.receiver_m


def test_output_refuses_inputs_that_did_not_produce_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """拿別的輸入來組輸出，場景指紋就是假的；至少兩邊都有的反射階數要對得上。"""
    inputs = _inputs()
    report = _solved_report(monkeypatch, inputs)
    other = _inputs(reflection_order_k=inputs.reflection_order_k + 1)

    with pytest.raises(ValueError, match="不是產出這份報表的那一份"):
        report_output.output_from_report(report, inputs=other, with_points=False)


def test_cli_prints_the_scene_line_right_after_the_capability_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """命令列文字輸出：能力行照舊第一，場景行第二，指紋等於直接拿這一份輸入算的。"""
    input_path = tmp_path / "room.json"
    input_path.write_text(json.dumps(_document()), encoding="utf-8")
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)

    exit_code = three_lane_report_cli.main(
        [str(input_path), "--capabilities", str(config_path("capabilities.toml"))]
    )
    lines = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert lines[0].startswith("capability ")
    assert lines[1].startswith("scene scene_fingerprint=")
    assert report_io.scene_fingerprint(_inputs()) in lines[1]
