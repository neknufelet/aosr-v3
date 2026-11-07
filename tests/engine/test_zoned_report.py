"""#562 正式 numpy 三路：入口、分區散射、牆對留存與同材控制。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from typing import TypedDict
from aosr.physics.report_source import SourceModelSpec
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics import geometric_lane, late_energy, report_io, wall_surfaces
from aosr.physics.amplitude import reflection_coefficient
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.room_paths import image_source_paths
from aosr.physics.report_surfaces import WholeWallZone
from aosr.physics.wall_reflections import zoned_reflection_sources
from aosr.reporting import scheme_cli
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import EvaluationState, QualityCategory
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._precision_contracts import contract_value
from tests.engine._zoned_report_cases import small_scheme, stable_digest


class _PathArguments(TypedDict):
    room: Room
    source: Point
    receiver: Point
    c: float
    max_order: int


class _EarlyArguments(TypedDict):
    room: Room
    source: Point
    receiver: Point
    source_model: SourceModelSpec
    sound_speed_m_s: float
    rho_c_pa_s_per_m: float
    frequencies_hz: tuple[float, ...]
    impedance_by_wall: dict[str, float]
    reflection_order_k: int


@pytest.fixture(scope="module")
def zoned_inputs() -> report_io.ReportInput:
    return report_io.load_input_document(pair(small_scheme(zoned=True)), CAPABILITIES, DIRECTIVITY)


def test_two_identical_zones_equal_whole_image_lane(zoned_inputs: report_io.ReportInput,
    record_property: Callable[[str, object], None]) -> None:
    """抓早期散射漏區／逐點反射只看首區；沿用 #1018 反射乘積絕對尺。"""
    inputs = zoned_inputs
    axis = (125.0, 503.0)
    zones = inputs.surface_zones_by_wall
    assert zones is not None
    sources = zoned_reflection_sources(inputs.room_m, zones, axis,
        density_kg_m3=inputs.density_kg_m3, sound_speed_m_s=inputs.sound_speed_m_s)
    whole = report_io.load_input_document(pair(small_scheme()), CAPABILITIES, DIRECTIVITY)
    whole_sources = zoned_reflection_sources(whole.room_m, {name: (WholeWallZone(area="whole_wall",
        material=ConstantImpedance(impedance_pa_s_m=value)),) for name, value in (whole.impedance_pa_s_per_m_by_wall or {}).items()}, axis,
        density_kg_m3=whole.density_kg_m3, sound_speed_m_s=whole.sound_speed_m_s)
    args = _PathArguments(room=inputs.room_m, source=inputs.source_m, receiver=inputs.receiver_m,
        c=inputs.sound_speed_m_s, max_order=3)
    expected = image_source_paths(**args, reflection_sources=whole_sources)
    actual = image_source_paths(**args, reflection_sources=sources)
    difference = max(abs(a-b) for old, new in zip(expected, actual, strict=True)
                     for a, b in zip(old.reflection_product, new.reflection_product, strict=True))
    assert difference <= contract_value("reflection_product_ulp")
    lane_args = _EarlyArguments(room=inputs.room_m, source=inputs.source_m, receiver=inputs.receiver_m,
        source_model=report_io.solver_inputs(whole).source_model, sound_speed_m_s=inputs.sound_speed_m_s,
        rho_c_pa_s_per_m=inputs.density_kg_m3*inputs.sound_speed_m_s,
        frequencies_hz=axis, impedance_by_wall=dict.fromkeys(Wall.wall_names(), 1646.4), reflection_order_k=3)
    reference = geometric_lane.solve_geometric_early_lane(**lane_args, reflection_sources=whole_sources)
    observed = geometric_lane.solve_geometric_early_lane(**lane_args, reflection_sources=sources)
    np.testing.assert_array_equal(observed.reflected_energy, reference.reflected_energy)
    assert observed.scattering == reference.scattering
    record_property("image_reflection_max_absolute_difference", difference)


def test_two_identical_zones_equal_whole_late_lane(zoned_inputs: report_io.ReportInput,
    record_property: Callable[[str, object], None]) -> None:
    """抓 combined lane（合成幾何路）拒收或漏傳逐區吸音；沿用 #1021 同材尺。"""
    inputs = zoned_inputs
    axis = (500.0, 1000.0)
    surfaces = wall_surfaces.zoned_wall_surface_response(inputs.room_m, inputs.surface_zones_by_wall or {}, axis,
        density_kg_m3=inputs.density_kg_m3, sound_speed_m_s=inputs.sound_speed_m_s)
    problem = late_energy.LateEnergyInputs(inputs.room_m, inputs.density_kg_m3*inputs.sound_speed_m_s,
        axis, surfaces.named_rows(axis), 6, 0.9, absorption_by_zone=surfaces.named_zone_rows(axis))
    reference = late_energy.solve_late_energy(replace(problem, absorption_by_zone=None))
    actual = late_energy.solve_late_energy(problem)
    old, new = np.array([b.late_reverberant_energy for b in reference.bands]), np.array([
        b.late_reverberant_energy for b in actual.bands])
    np.testing.assert_allclose(new, old, rtol=contract_value("late_energy_physical_property"), atol=0.0)
    sources = zoned_reflection_sources(inputs.room_m, inputs.surface_zones_by_wall or {}, axis,
        density_kg_m3=inputs.density_kg_m3, sound_speed_m_s=inputs.sound_speed_m_s)
    combined = geometric_lane.solve_geometric_lane(room=inputs.room_m, source=inputs.source_m,
        receiver=inputs.receiver_m, source_model=report_io.solver_inputs(inputs).source_model,
        sound_speed_m_s=inputs.sound_speed_m_s, rho_c_pa_s_per_m=problem.rho_c_pa_s_per_m,
        frequencies_hz=axis, impedance_by_wall=dict.fromkeys(Wall.wall_names(), 1646.4),
        reflection_sources=sources, absorption_by_wall=problem.absorption_by_wall,
        absorption_by_zone=problem.absorption_by_zone, reflection_order_k=3)
    assert all(value > 0.0 for value in combined.late_energy)
    record_property("late_max_relative_difference", float(np.max(abs((new-old)/old))))


def test_zone_scattering_uses_area_times_normal_reflected_energy() -> None:
    """1:2 非等面積、兩材、各面散射不同；抓平均區數或只看第一區。"""
    inputs = report_io.load_input_document(pair(small_scheme(zoned=True, changed=True)), CAPABILITIES, DIRECTIVITY)
    axis = (125.0, 503.0)
    rho_c = inputs.density_kg_m3*inputs.sound_speed_m_s
    sources = zoned_reflection_sources(inputs.room_m, inputs.surface_zones_by_wall or {}, axis,
        density_kg_m3=inputs.density_kg_m3, sound_speed_m_s=inputs.sound_speed_m_s)
    scattering = dict(zip(Wall.wall_names(), (0.1, 0.2, 0.4, 0.3, 0.5, 0.6), strict=True))
    def retained(z: float) -> float:
        return abs((complex(z)-rho_c)/(complex(z)+rho_c))**2
    weights = (6*retained(1646.4), 6*retained(1646.4),
               retained(950.0)+2*retained(1646.4), 3*retained(1646.4),
               4.5*retained(1646.4), 4.5*retained(1646.4))
    expected = sum(w*s for w, s in zip(weights, scattering.values(), strict=True))/sum(weights)
    actual = geometric_lane.solve_geometric_early_lane(room=inputs.room_m, source=inputs.source_m,
        receiver=inputs.receiver_m, source_model=report_io.solver_inputs(inputs).source_model,
        sound_speed_m_s=inputs.sound_speed_m_s, rho_c_pa_s_per_m=rho_c, frequencies_hz=axis,
        impedance_by_wall=dict.fromkeys(Wall.wall_names(), 1646.4), scattering_by_wall=scattering,
        reflection_sources=sources, reflection_order_k=3)
    np.testing.assert_allclose(actual.scattering, (expected, expected), rtol=0.0,
                               atol=contract_value("reflection_product_ulp"))


def test_screen_area_averages_each_face_before_pair_product() -> None:
    """主對話 11-08 指定的留存合成；抓把阻抗先平均，或兩面區數混一起。"""
    inputs = report_io.load_input_document(pair(small_scheme(zoned=True, changed=True)), CAPABILITIES, DIRECTIVITY)
    inputs = inputs.model_copy(update={"scattering_by_wall": dict.fromkeys(Wall.wall_names(), 0.25)})
    material = ConstantImpedance(impedance_pa_s_m=1646.4)
    result = build_reflection_screen(inputs, (125.0, 503.0), materials_by_wall=dict.fromkeys(Wall.all(), material))
    rho_c = inputs.density_kg_m3*inputs.sound_speed_m_s
    a = abs(reflection_coefficient(complex(950.0), 1.0, rho_c))**2*0.75
    b = abs(reflection_coefficient(complex(1646.4), 1.0, rho_c))**2*0.75
    expected = (a+2*b)/3
    x, = (row for row in result.pairs if row.pair == "x")
    assert x.face_retained_energy == ((expected, expected), (b, b))
    assert x.round_trip_retained_energy == (expected*b, expected*b)


@pytest.mark.parametrize("with_furniture", (False, True))
def test_report_late_family_keeps_nonuniform_zone_distribution(with_furniture: bool) -> None:
    """抓正式報表晚期偷換為等效整面；家具時同一 ε 必須套到每區。"""
    from aosr.physics.three_lane_report_batch import _prepare_late_family
    from aosr.physics.report_furniture import AbsoluteFurniture
    from aosr.physics.furniture_scene import furniture_absorption_area_m2
    inputs = report_io.load_input_document(pair(small_scheme(zoned=True, changed=True)), CAPABILITIES, DIRECTIVITY)
    furniture = (AbsoluteFurniture.model_validate(dict(furniture_id="table", kind="coffee_table", material="wood",
        width_m=0.2, depth_m=0.2, height_m=0.2, bottom_center_m=(1.0, 1.0, 0.0), yaw_deg=0)),) if with_furniture else None
    axis = (500.0, 1000.0)
    _surfaces, _t60, _fs, actual, _decay = _prepare_late_family(room=inputs.room_m, materials_by_wall={},
        density_kg_m3=inputs.density_kg_m3, sound_speed_m_s=inputs.sound_speed_m_s,
        report_frequencies_hz=axis, reflection_order_k=3, surface_zones_by_wall=inputs.surface_zones_by_wall,
        furniture=furniture)
    response = wall_surfaces.zoned_wall_surface_response(inputs.room_m, inputs.surface_zones_by_wall or {}, axis,
        density_kg_m3=inputs.density_kg_m3, sound_speed_m_s=inputs.sound_speed_m_s)
    rows, zones = response.named_rows(axis), response.named_zone_rows(axis)
    rho_c = inputs.density_kg_m3*inputs.sound_speed_m_s
    if furniture:
        af = furniture_absorption_area_m2(furniture, axis, rho_c)
        areas = dict(zip(Wall.wall_names(), (6.0, 6.0, 3.0, 3.0, 4.5, 4.5), strict=True))
        epsilon = tuple(af[i]/sum(areas[name]*(1-row[i]) for name, row in rows.items()) for i in range(len(axis)))
        rows = {name: tuple(a+(1-a)*e for a, e in zip(row, epsilon, strict=True)) for name, row in rows.items()}
        zones = {name: tuple(replace(cell, absorption=tuple(a+(1-a)*e for a, e in
            zip(cell.absorption, epsilon, strict=True))) for cell in cells) for name, cells in zones.items()}
    expected = geometric_lane.solve_geometric_late_energy(room=inputs.room_m, rho_c_pa_s_per_m=rho_c,
        frequencies_hz=axis, absorption_by_wall=rows, absorption_by_zone=zones, reflection_order_k=3)
    np.testing.assert_allclose([b.late_reverberant_energy for b in actual.bands],
        [b.late_reverberant_energy for b in expected.bands],
        rtol=contract_value("late_energy_physical_property"), atol=0.0)


@pytest.mark.parametrize("zoned", (False, True))
def test_scheme_cli_real_three_lanes_to_six_categories(tmp_path: Path, zoned: bool) -> None:
    """完整入口不替身物理／評分；整面釘起點，異材分區須產出六類與全部逐點報表。"""
    scheme = small_scheme(zoned=zoned, changed=zoned)
    source, out = tmp_path / "scheme.json", tmp_path / "result.json"
    source.write_text(scheme.model_dump_json())
    assert scheme_cli.main(["run", str(source), "--out", str(out), "--engine-commit", "control",
        "--run-date", "2026-11-08", "--capabilities", str(config_path("capabilities.toml"))]) == 0
    result = SchemeResult.model_validate_json(out.read_bytes())
    assert {e.category for e in result.candidate.evaluations} == {QualityCategory.TIMBRE_BALANCE, QualityCategory.LISTENING_AREA_STABILITY,
        QualityCategory.REFLECTIONS_AND_ECHO, QualityCategory.LOW_FREQUENCY_RESPONSE,
        QualityCategory.CHANNEL_MATCHING, QualityCategory.REVERBERATION}
    assert all(e.state is EvaluationState.MEASURED for e in result.candidate.evaluations)
    assert len(result.pairs) == 14
    assert all(p.report.points and p.report.path_table and p.report.bands for p in result.pairs)
    if zoned:
        assert all(p.report.measured_seams_by_wall is None for p in result.pairs)
    if not zoned:
        assert stable_digest(result.model_dump(mode="json")) == "bb4ff2a08393c857c222d188f67433e080f03d5b308a1330a4b6b7550dcea8d5"


def test_scheme_cli_second_zone_assembly_uses_real_cpu_double(tmp_path: Path) -> None:
    """冷入口真算第二區構造，抓首區固定值誤判能力與逐區法漏接；不改父行程精度。"""
    import subprocess
    import sys
    from aosr.runtime import child_process_env
    source, out = tmp_path / "scheme.json", tmp_path / "result.json"
    source.write_text(small_scheme(zoned=True, assembly=True).model_dump_json())
    run = subprocess.run([sys.executable, "-m", "aosr.reporting.scheme_cli", "run", str(source), "--out", str(out),
        "--engine-commit", "control", "--run-date", "2026-11-08", "--capabilities", str(config_path("capabilities.toml"))],
        env=child_process_env(threads=1), capture_output=True, text=True, timeout=240)
    assert run.returncode == 0, run.stdout+run.stderr
    result = SchemeResult.model_validate_json(out.read_bytes())
    assert len(result.candidate.evaluations) == 6
    assert all(e.state is EvaluationState.MEASURED for e in result.candidate.evaluations)
    assert len(set(result.pairs[0].report.top.eyring_t60_by_band_s.values())) > 1
