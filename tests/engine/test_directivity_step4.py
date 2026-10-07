"""#505 第四步：輸入允許範圍與方向倍率的產品接線。"""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.frequency_axis import GEOMETRIC_BAND_FREQUENCIES_HZ, GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, report_source
from aosr.physics.geometric_lane import GeometricEarlyResult, solve_geometric_early_lane
from aosr.physics.report_path_table import PathTableData, build_path_table
from aosr.physics.report_source import AnalyticAxisymmetricInput, SourceModelKind, SourceModelSpec
from aosr.physics.source_directivity import off_axis_degrees
from tests.engine import _source_model_control as control
from tests.engine._directivity import DIRECTIVITY


def _document() -> dict[str, object]:
    document = deepcopy(control.SCENE_WITHOUT_SOURCE_MODEL)
    document["source_model"] = report_source.default_source_model(
        Point(4.35, 2.45, 1.05), DIRECTIVITY,
    ).model_dump(mode="json")
    return document


def _inputs() -> report_io.ReportInput:
    return report_io.load_input_document(
        _document(), load_capabilities(config_path("capabilities.toml")), DIRECTIVITY,
    )


def _early(
    solved: report_io.SolverInputs, model: SourceModelSpec, frequencies: tuple[float, ...],
) -> GeometricEarlyResult:
    return solve_geometric_early_lane(
        source_model=model, room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.sound_speed_m_s * solved.density_kg_m3,
        frequencies_hz=frequencies,
        impedance_by_wall={wall.wall_name(): complex(solved.impedance_by_wall[wall]) for wall in Wall.all()},
        reflection_order_k=1,
    )


def _table(solved: report_io.SolverInputs, model: SourceModelSpec) -> PathTableData:
    return build_path_table(
        furniture=None,
        source_model=model, room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        rho_c_pa_s_per_m=solved.sound_speed_m_s * solved.density_kg_m3,
        frequencies_hz=(1000.0, 4000.0), impedance_by_wall=solved.impedance_by_wall,
        scattering_coefficient=(0.2, 0.2), reflection_order_k=1,
    )


def test_default_model_passes_input_and_solver_contract() -> None:
    solved = report_io.solver_inputs(_inputs())
    assert solved.source_model.kind == SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1
    assert solved.source_model.parameters == DIRECTIVITY.two_parameter
    assert solved.source_model.aim == Point(4.35, 2.45, 1.05)
    other = DIRECTIVITY.model_copy(update={
        "two_parameter": DIRECTIVITY.two_parameter.model_copy(update={"beta_limit": 2.5}),
    })
    chosen = report_source.default_source_model(Point(4.35, 2.45, 1.05), other)
    assert chosen.parameters.to_curve() == other.two_parameter
    assert chosen.parameters.to_curve() != DIRECTIVITY.two_parameter


def test_missing_directivity_registry_is_a_wiring_error() -> None:
    with pytest.raises(report_io.WiringError, match="directivity"):
        report_io.ReportInput.model_validate(
            _document(), context={"table": load_capabilities(config_path("capabilities.toml"))},
        )


@pytest.mark.parametrize(("field", "bad"), [
    ("beta_limit", DIRECTIVITY.allowed_range.beta_max + 0.01),
    ("beta_limit", DIRECTIVITY.allowed_range.beta_min - 0.01),
    ("power_floor_limit_db", DIRECTIVITY.allowed_range.power_floor_min_db - 0.01),
    ("power_floor_limit_db", DIRECTIVITY.allowed_range.power_floor_max_db + 0.01),
])
def test_input_rejects_each_curve_limit(field: str, bad: float) -> None:
    document = _document()
    model = document["source_model"]
    assert isinstance(model, dict)
    params = model["parameters"]
    assert isinstance(params, dict)
    params[field] = bad
    with pytest.raises(ValueError, match=f"source_model.parameters.{field}"):
        report_io.load_input_document(
            document, load_capabilities(config_path("capabilities.toml")), DIRECTIVITY,
        )


def test_analytic_early_lane_changes_off_axis_reflections_and_keeps_on_axis_direct() -> None:
    solved = report_io.solver_inputs(_inputs())
    omni = SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL)
    plain = _early(solved, omni, (1000.0, 4000.0))
    directed = _early(solved, solved.source_model, (1000.0, 4000.0))
    assert directed.direct_energy == plain.direct_energy
    assert directed.reflected_energy != plain.reflected_energy


def test_path_table_records_off_axis_angles_and_changes_reflection_energy() -> None:
    solved = report_io.solver_inputs(_inputs())
    plain = _table(solved, SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL))
    directed = _table(solved, solved.source_model)
    assert all(row.departure_off_axis_deg is None for row in plain.rows)
    assert all(row.departure_off_axis_deg is not None for row in directed.rows)
    # 對準點就是這一份的接收點：直達那一列正對軸線，離軸角逐位是 0。
    assert next(row for row in directed.rows if row.order == 0).departure_off_axis_deg == 0.0
    assert any(
        left.relative_direct_energy != right.relative_direct_energy
        for left, right in zip(plain.rows, directed.rows, strict=True) if left.order > 0
    )


@pytest.mark.parametrize("aim", (
    Point(0.0, 0.0, 0.0), Point(6.3, 4.1, 2.9),
))
def test_aim_on_closed_room_boundary_is_accepted(aim: Point) -> None:
    document = _document()
    model = document["source_model"]
    assert isinstance(model, dict)
    model["aim_m"] = {"x": aim.x, "y": aim.y, "z": aim.z}
    accepted = report_io.load_input_document(
        document, load_capabilities(config_path("capabilities.toml")), DIRECTIVITY,
    )
    assert isinstance(accepted.source_model, AnalyticAxisymmetricInput)
    assert accepted.source_model.aim_m == aim


@pytest.mark.parametrize("aim", (
    Point(-0.01, 2.0, 1.0), Point(6.31, 2.0, 1.0),
    Point(2.0, -0.01, 1.0), Point(2.0, 4.11, 1.0),
    Point(2.0, 2.0, -0.01), Point(2.0, 2.0, 2.91),
    Point(1.15, 0.95, 1.2),
))
def test_outside_or_coincident_aim_is_rejected(aim: Point) -> None:
    document = _document()
    model = document["source_model"]
    assert isinstance(model, dict)
    model["aim_m"] = {"x": aim.x, "y": aim.y, "z": aim.z}
    with pytest.raises(ValueError, match="source_model.aim_m"):
        report_io.load_input_document(
            document, load_capabilities(config_path("capabilities.toml")), DIRECTIVITY,
        )


@pytest.mark.parametrize("frequencies", (GEOMETRIC_LANE_FREQUENCIES_HZ, GEOMETRIC_BAND_FREQUENCIES_HZ))
def test_both_official_axes_change_reflection_and_off_axis_direct(
    frequencies: tuple[float, ...],
) -> None:
    solved = report_io.solver_inputs(_inputs())
    model = SourceModelSpec(solved.source_model.kind, solved.source_model.parameters, Point(4.0, 2.0, 1.0))
    plain = _early(solved, SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL), frequencies)
    on_axis = _early(solved, solved.source_model, frequencies)
    directed = _early(solved, model, frequencies)
    assert on_axis.direct_energy == plain.direct_energy
    assert directed.direct_energy != plain.direct_energy
    assert directed.reflected_energy != plain.reflected_energy


def test_relative_path_energy_uses_direct_and_reflected_d_independently() -> None:
    frequencies = (100.0, 200.0)
    source, receiver, aim = Point(2.0, 1.0, 1.0), Point(4.0, 1.0, 1.0), Point(4.0, 2.0, 1.0)
    model = SourceModelSpec(
        SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1, DIRECTIVITY.two_parameter, aim,
    )
    rho_c = 1.2 * 343.0
    table = build_path_table(
        furniture=None,
        source_model=model, room=Room(10.0, 10.0, 10.0), source=source, receiver=receiver,
        sound_speed_m_s=343.0, rho_c_pa_s_per_m=rho_c,
        impedance_by_wall={wall: 4.0 * rho_c for wall in Wall.all()},
        frequencies_hz=frequencies, scattering_coefficient=(0.2, 0.2), reflection_order_k=1,
    )
    x0 = next(row for row in table.rows if row.wall_sequence == ("x0",))
    direct = next(row for row in table.rows if row.order == 0)
    # 離軸角用考卷自己算的向量：軸線 (2,1,0)/√5，直達離開聲源 +x、x0 那條離開聲源 −x。
    assert direct.departure_off_axis_deg == pytest.approx(math.degrees(math.acos(2.0 / math.sqrt(5.0))), rel=1e-12)
    assert x0.departure_off_axis_deg == pytest.approx(math.degrees(math.acos(-2.0 / math.sqrt(5.0))), rel=1e-12)
    for frequency, actual in zip(frequencies, x0.relative_direct_energy, strict=True):
        curve = DIRECTIVITY.two_parameter
        beta = curve.beta_limit / (1.0 + (curve.beta_corner_hz / frequency) ** curve.beta_exponent)
        db = curve.power_floor_limit_db / (1.0 + (curve.power_floor_corner_hz / frequency) ** curve.power_floor_exponent)
        floor = 10.0 ** (db / 10.0)
        # 軸向 (2,1,0)/√5；直達 +x、x0 反射離開聲源時為 −x。
        direct_d2 = (1.0 - floor) * math.exp(-2.0 * beta * (1.0 - 2.0 / math.sqrt(5.0))) + floor
        reflected_d2 = (1.0 - floor) * math.exp(-2.0 * beta * (1.0 + 2.0 / math.sqrt(5.0))) + floor
        expected = 0.8 * ((3.0 / 5.0) / 6.0) ** 2 * reflected_d2 / ((1.0 / 2.0) ** 2 * direct_d2)
        assert actual == pytest.approx(expected, rel=1e-12)


def test_mirrored_paths_keep_each_departure_factor_bitwise() -> None:
    aim = Point(2.0, 2.0, 1.0)
    model = SourceModelSpec(SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1,
                            DIRECTIVITY.two_parameter, aim)

    def mirrored(source: Point, receiver: Point) -> PathTableData:
        return build_path_table(
            furniture=None,
            source=source, receiver=receiver, source_model=model, room=Room(4.0, 4.0, 2.0),
            sound_speed_m_s=343.0, rho_c_pa_s_per_m=1.2 * 343.0,
            impedance_by_wall={wall: 1600.0 for wall in Wall.all()},
            frequencies_hz=(1000.0,), scattering_coefficient=(0.2,), reflection_order_k=1,
        )

    left = mirrored(Point(1.0, 1.0, 1.0), Point(1.5, 2.0, 1.0))
    right = mirrored(Point(3.0, 1.0, 1.0), Point(2.5, 2.0, 1.0))
    mirror = {"x0": "xL", "xL": "x0"}
    by_path = {row.wall_sequence: row for row in right.rows}
    for row in left.rows:
        counterpart = by_path[tuple(mirror.get(wall, wall) for wall in row.wall_sequence)]
        assert row.departure_off_axis_deg == counterpart.departure_off_axis_deg
        assert row.relative_direct_energy == counterpart.relative_direct_energy


def test_off_axis_angle_keeps_small_forward_angle() -> None:
    sideways = 1e-8
    forward = math.sqrt(1.0 - sideways * sideways)
    assert off_axis_degrees((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 0.0
    assert off_axis_degrees((1.0, 0.0, 0.0), (forward, sideways, 0.0)) == pytest.approx(
        math.degrees(math.atan2(sideways, forward)), rel=1e-12,
    )
