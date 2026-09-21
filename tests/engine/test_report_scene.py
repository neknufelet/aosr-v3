"""票 #415：三路接合報表的場景身分與逐份座標。"""
from __future__ import annotations

import importlib.util
from collections.abc import Mapping

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, report_output, three_lane_report
from aosr.physics.report_io import ReportInput


_WALL_NAMES = tuple(wall.wall_name() for wall in Wall.all())


def test_output_assembly_has_its_own_module_without_report_io_reexport() -> None:
    assert importlib.util.find_spec("aosr.physics.report_output") is not None
    assert not hasattr(report_io, "output_from_report")


def _document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "room_m": {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0},
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
    ),
)
def test_each_shared_input_changes_the_scene_fingerprint(
    field: str, changed: object
) -> None:
    assert report_io.scene_fingerprint(_inputs(**{field: changed})) != (
        report_io.scene_fingerprint(_inputs())
    )


def test_omitted_scattering_differs_from_explicit_zero_scattering() -> None:
    assert report_io.scene_fingerprint(_inputs(scattering_by_wall=None)) != (
        report_io.scene_fingerprint(
            _inputs(scattering_by_wall={wall: 0.0 for wall in _WALL_NAMES})
        )
    )


def test_wall_key_order_does_not_change_the_scene_fingerprint() -> None:
    ordered = {wall: 1646.4 for wall in _WALL_NAMES}
    reversed_order = dict(reversed(tuple(ordered.items())))
    assert report_io.scene_fingerprint(
        _inputs(impedance_pa_s_per_m_by_wall=ordered)
    ) == report_io.scene_fingerprint(
        _inputs(impedance_pa_s_per_m_by_wall=reversed_order)
    )


def test_real_report_output_carries_the_input_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    report = _solved_report(monkeypatch, inputs)

    output = report_output.output_from_report(report, inputs=inputs, with_points=True)

    assert output.scene.scene_fingerprint == report_io.scene_fingerprint(inputs)
    assert output.scene.source_m == inputs.source_m
    assert output.scene.receiver_m == inputs.receiver_m
