"""#1057 空氣在家具之後進正式晚期準備；真值獨立換算 dB/m。"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.source_reference import DIFFUSE_MONOPOLE_4PI
from aosr.config.three_lane_crossover import EYRING_COEFFICIENT_S_PER_M
from aosr.geometry.shoebox import Wall
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.physics import air_absorption, late_energy, three_lane_report_batch as batch
from aosr.physics.furniture_scene import furniture_absorption_area_m2
from aosr.physics.three_lane_report_zones import prepare_zoned_late_family
from tests.engine import _furniture_energy_cases as case
from tests.engine._furniture_absorption_cases import ITEMS, late_kwargs
from tests.engine._precision_contracts import contract_value
from tests.engine.test_late_energy_zones import zones


def _measures() -> tuple[float, float]:
    x, y, z = case.ROOM.Lx, case.ROOM.Ly, case.ROOM.Lz
    return x*y*z, 2.0*(x*y+y*z+x*z)


def _m(frequencies: tuple[float, ...]) -> NDArray[np.float64]:
    """不呼叫受驗 ε 函式；由第一段已驗的 dB/m 自行轉能量衰減。"""
    return air_absorption.air_attenuation_db_per_m(frequencies, temperature_c=20.0,
        relative_humidity_percent=30.0, pressure_pa=101325.0) * math.log(10.0)/10.0


@pytest.mark.parametrize("zoned", (False, True), ids=("whole", "zoned"))
@pytest.mark.parametrize("furnished", (False, True), ids=("air", "furniture-air"))
def test_report_uniform_eyring_and_decay_include_air(zoned: bool, furnished: bool) -> None:
    """漏牆面或任一分區、錯自由路徑、只改 Eyring 或顛倒家具順序都要紅。"""
    kwargs = late_kwargs()
    items = ITEMS if furnished else None
    if zoned:
        partition = zones(Wall.FLOOR, split_u=1.0/3.0)
        material = kwargs["materials_by_wall"][Wall.FLOOR]
        partition = {w: tuple(cell.model_copy(update={"material": material}) for cell in cells)
                     for w, cells in partition.items()}
        surfaces, t60, _fs, energy, decay = prepare_zoned_late_family(room=case.ROOM, zones=partition,
            density_kg_m3=case.DENSITY, sound_speed_m_s=case.SPEED,
            report_frequencies_hz=case.FREQUENCIES, reflection_order_k=1, furniture=items)
    else:
        surfaces, t60, _fs, energy, decay = batch._prepare_late_family(**kwargs, furniture=items)
    volume, surface = _measures()
    alpha = complex_random_incidence_absorption(4.0)
    frequencies = surfaces.frequencies_hz
    af = np.asarray(furniture_absorption_area_m2(items, frequencies, case.RHO_C)) if items else np.zeros(len(frequencies))
    wall_alpha = alpha + af/surface
    loss = -surface*np.log1p(-wall_alpha) + 4.0*_m(frequencies)*volume
    tolerance = contract_value("late_energy_physical_property")
    for frequency, actual in t60.items():
        assert actual == pytest.approx(EYRING_COEFFICIENT_S_PER_M*volume/loss[frequencies.index(frequency)],
                                      rel=tolerance, abs=0.0)
    for band in decay.result.bands:
        exact = 24.0*math.log(10.0)*volume/(case.SPEED*loss[frequencies.index(band.frequency_hz)])
        assert band.t20_s == pytest.approx(exact, rel=tolerance, abs=0.0)
        assert band.t30_s == pytest.approx(exact, rel=tolerance, abs=0.0)
    for energy_band in energy.bands:
        i = frequencies.index(energy_band.frequency_hz)
        expected = 1.0 - (1.0-wall_alpha[i])*math.exp(-_m((energy_band.frequency_hz,))[0]*4.0*volume/surface)
        exact_energy = 4.0*DIFFUSE_MONOPOLE_4PI*(1.0-expected)/(-surface*math.log1p(-expected))
        assert energy_band.late_reverberant_energy == pytest.approx(exact_energy, rel=tolerance, abs=0.0)


def test_furniture_area_is_added_before_air() -> None:
    """空氣先進會把家具 A_f 的貢獻多留一份 A_f·ε_air。"""
    surfaces, _t60, _fs, energy, _decay = batch._prepare_late_family(**late_kwargs(), furniture=ITEMS)
    volume, surface = _measures()
    af = furniture_absorption_area_m2(ITEMS, case.FREQUENCIES, case.RHO_C)
    epsilon = 1.0-np.exp(-_m(case.FREQUENCIES)*4.0*volume/surface)
    for i, band in enumerate(energy.bands):
        alpha = surfaces.absorption_by_wall[Wall.FLOOR][surfaces.frequencies_hz.index(band.frequency_hz)]
        want = alpha+af[i]/surface + (1.0-alpha-af[i]/surface)*epsilon[i]
        reversed_order = alpha+(1.0-alpha)*epsilon[i]+af[i]/surface
        tolerance = contract_value("late_energy_physical_property")
        exact_energy = 4.0*DIFFUSE_MONOPOLE_4PI*(1.0-want)/(-surface*math.log1p(-want))
        assert band.late_reverberant_energy == pytest.approx(exact_energy, rel=tolerance, abs=0.0)
        assert abs(reversed_order-want)/want > tolerance


def test_air_conditions_are_loaded_and_serialization_stays_four_fields() -> None:
    """兩個新增鍵不可消失或滲入既有四欄序列化契約。"""
    constants = load_physics_constants(config_path("physics_constants.toml"))
    assert getattr(constants, "air_temperature_c", None) == 20.0
    assert getattr(constants, "air_relative_humidity_percent", None) == 30.0
    assert constants.model_dump() == {"air_density_kg_m3": 1.2, "sound_speed_m_s": 343.0,
        "ref_pressure_pa": 2.0e-5, "air_viscosity_pa_s": 1.81e-5}
    assert constants.model_dump_json() == ('{"air_density_kg_m3":1.2,"sound_speed_m_s":343.0,'
        '"ref_pressure_pa":0.00002,"air_viscosity_pa_s":0.0000181}')


@pytest.mark.parametrize("missing", ("air_temperature_c", "air_relative_humidity_percent", "atmospheric_pressure_pa"))
def test_air_conditions_missing_key_is_rejected_without_cache(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """先讀成功再換檔；任一缺鍵須拒收，不可沿用快取或拿零代替。"""
    from aosr.config import air_absorption_environment as environment

    path = tmp_path / "physics_constants.toml"
    source = config_path("physics_constants.toml").read_text()
    path.write_text(source)
    monkeypatch.setattr(environment, "config_path", lambda _name: path)
    assert environment.air_absorption_environment() == (20.0, 30.0, 101325.0)
    path.write_text("\n".join(line for line in source.splitlines() if not line.startswith(missing+" =")))
    with pytest.raises(ValueError, match=missing):
        environment.air_absorption_environment()


def test_zero_air_adjustment_preserves_arrays_bitwise() -> None:
    """ε＝0 時不能繞道 log/exp 或改成單精度；直接比每格陣列。"""
    alpha = np.asarray([[0.0, 0.125, 0.9], [0.33, 0.75, 1.0]], dtype=np.float64)
    result = air_absorption.air_adjusted_absorption(alpha, np.zeros(3))
    assert np.all(result == alpha)


def test_steady_absorbed_wall_and_air_power_equals_source() -> None:
    """空氣乘每格剩餘反射率，牆與空氣總功率必須守恆。"""
    from tests.engine.test_late_energy_zones import synthetic

    inputs = synthetic(Wall.FLOOR, 2, 1.0/3.0, 0.5)
    base = late_energy._patch_absorption(inputs)
    epsilon = 1.0-np.exp(-_m(inputs.frequencies_hz)*4.0*_measures()[0]/_measures()[1])
    adjusted = air_absorption.air_adjusted_absorption(base, epsilon)
    geometry = late_energy._patch_geometry(inputs.room, inputs.n_per_wall)
    factors = late_energy._form_factors(geometry)
    source = np.full(len(geometry.areas), 1.0/np.sum(geometry.areas))
    for i in range(len(inputs.frequencies_hz)):
        outgoing = np.linalg.solve(np.eye(len(source))-(1.0-adjusted[:, i, None])*factors, source)
        incident = factors@outgoing
        wall_power = np.sum(geometry.areas*base[:, i]*incident)
        air_power = np.sum(geometry.areas*(1.0-base[:, i])*epsilon[i]*incident)
        assert wall_power+air_power == pytest.approx(float(geometry.areas@source),
            rel=contract_value("late_energy_physical_property"), abs=0.0)


def test_identical_zones_with_air_equal_whole_wall() -> None:
    """同阻抗分格必須共用同一個 ε_air；整牆與每格的能量都一致。"""
    from tests.engine.test_late_energy_zones import inputs

    split = inputs(zones(Wall.FLOOR, split_u=1.0/3.0))
    volume, surface = _measures()
    epsilon = 1.0-np.exp(-_m(split.frequencies_hz)*4.0*volume/surface)
    walls = {w: tuple(air_absorption.air_adjusted_absorption(row, epsilon))
             for w, row in split.absorption_by_wall.items()}
    cells = {w: tuple(replace(cell, absorption=tuple(air_absorption.air_adjusted_absorption(cell.absorption, epsilon)))
                     for cell in rows) for w, rows in (split.absorption_by_zone or {}).items()}
    adjusted = replace(split, absorption_by_wall=walls, absorption_by_zone=cells)
    whole = late_energy.solve_late_energy(replace(adjusted, absorption_by_zone=None))
    divided = late_energy.solve_late_energy(adjusted)
    np.testing.assert_allclose([b.late_reverberant_energy for b in divided.bands],
        [b.late_reverberant_energy for b in whole.bands], rtol=contract_value("late_energy_physical_property"), atol=0.0)


@pytest.mark.parametrize("zoned", (False, True), ids=("whole", "zoned"))
@pytest.mark.parametrize("furnished", (False, True), ids=("air", "furniture-air"))
def test_public_report_eyring_and_decay_follow_exact_air_loss(
    zoned: bool, furnished: bool,
) -> None:
    """公開報表入口必須把空氣帶到 Eyring 及每個衰減點；只替換無關的有限元素。"""
    from aosr.physics import three_lane_report as report
    from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ

    partition = zones(Wall.FLOOR, split_u=1.0/3.0) if zoned else None
    if partition is not None:
        material = late_kwargs()["materials_by_wall"][Wall.FLOOR]
        partition = {w: tuple(cell.model_copy(update={"material": material}) for cell in cells)
                     for w, cells in partition.items()}
    items = ITEMS if furnished else None
    result = report.solve_three_lane_reports(source_model=case.OMNI, room=case.ROOM,
        sources={"s": case.SOURCE}, receivers={"r": case.RECEIVER}, sound_speed_m_s=case.SPEED,
        density_kg_m3=case.DENSITY, impedance_by_wall=None if zoned else case.WALLS,
        materials_by_wall={} if zoned else late_kwargs()["materials_by_wall"], surface_zones_by_wall=partition,
        furniture=items, reflection_order_k=1, contact_rel=case.CONTACT_REL if items else None,
        fem_energies={("s", "r"): (1.0,) * len(FEM_LANE_FREQUENCIES_HZ)})["s", "r"]
    volume, surface = _measures()
    alpha = complex_random_incidence_absorption(4.0)
    tolerance = contract_value("late_energy_physical_property")
    for frequency, t60 in result.eyring_t60_by_band_s.items():
        af = furniture_absorption_area_m2(items, (frequency,), case.RHO_C)[0] if items else 0.0
        loss = -surface*math.log1p(-alpha-af/surface)+4.0*_m((frequency,))[0]*volume
        assert t60 == pytest.approx(EYRING_COEFFICIENT_S_PER_M*volume/loss, rel=tolerance, abs=0.0)
    for band in result.late_decay.bands:
        af = furniture_absorption_area_m2(items, (band.frequency_hz,), case.RHO_C)[0] if items else 0.0
        loss = -surface*math.log1p(-alpha-af/surface)+4.0*_m((band.frequency_hz,))[0]*volume
        exact = 24.0*math.log(10.0)*volume/(case.SPEED*loss)
        assert band.t20_s == pytest.approx(exact, rel=tolerance, abs=0.0)
        assert band.t30_s == pytest.approx(exact, rel=tolerance, abs=0.0)
