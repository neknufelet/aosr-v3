"""#561 晚期矩形：精確交集、材料來源與整面控制。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.geometry.shoebox import Room, Wall
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics import late_energy, wall_surfaces
from aosr.physics.crossover import eyring_t60_by_band, schroeder_frequency_hz
from aosr.physics.report_surfaces import RectangleZone, SurfaceZone, WholeWallZone
from tests.engine._precision_contracts import contract_value

ROOM = Room(6.0, 4.0, 3.0)
FREQUENCIES = (500.0, 1000.0)
CONSTANTS = load_physics_constants(config_path("physics_constants.toml"))
MATERIAL = ConstantImpedance(impedance_pa_s_m=8200.0)


def zones(wall: Wall, *, split_u: float = 0.5, split_v: float = 0.5) -> dict[str, tuple[SurfaceZone, ...]]:
    """考卷自行按已公布軸向給非正方形牆，不借受驗的交集或網格函式。"""
    sizes = {Wall.FLOOR: (6.0, 4.0), Wall.CEILING: (6.0, 4.0),
             Wall.X0: (4.0, 3.0), Wall.XL: (4.0, 3.0), Wall.Y0: (6.0, 3.0), Wall.YL: (6.0, 3.0)}
    width, height = sizes[wall]
    u, v = (0.0, width * split_u, width), (0.0, height * split_v, height)
    result: dict[str, tuple[SurfaceZone, ...]] = {
        w.wall_name(): (WholeWallZone(area="whole_wall", material=MATERIAL),) for w in Wall.all()}
    result[wall.wall_name()] = tuple(RectangleZone(area="rectangle", id=f"q{i}{j}",
        u_m=(u[i], u[i+1]), v_m=(v[j], v[j+1]), material=MATERIAL) for i in range(2) for j in range(2))
    return result


def inputs(partition: dict[str, tuple[SurfaceZone, ...]], n: int = 6) -> late_energy.LateEnergyInputs:
    response = wall_surfaces.zoned_wall_surface_response(ROOM, partition, FREQUENCIES,
        density_kg_m3=CONSTANTS.air_density_kg_m3, sound_speed_m_s=CONSTANTS.sound_speed_m_s)
    return late_energy.LateEnergyInputs(ROOM, CONSTANTS.rho_c, FREQUENCIES,
        response.named_rows(FREQUENCIES), n, 0.9, absorption_by_zone=response.named_zone_rows(FREQUENCIES))


def synthetic(wall: Wall, n: int, split_u: float, split_v: float) -> late_energy.LateEnergyInputs:
    partition = zones(wall, split_u=split_u, split_v=split_v)
    cells = partition[wall.wall_name()]
    by_zone = {wall.wall_name(): tuple(wall_surfaces.RectangleAbsorption(zone=zone,
        absorption=(alpha, alpha / 2.0)) for zone, alpha in zip(cells, (0.125, 0.25, 0.5, 0.75), strict=True)
        if isinstance(zone, RectangleZone))}
    return late_energy.LateEnergyInputs(ROOM, CONSTANTS.rho_c, FREQUENCIES,
        {w.wall_name(): (0.25, 0.125) for w in Wall.all()}, n, 0.9, absorption_by_zone=by_zone)


@pytest.mark.parametrize("wall", Wall.all())
def test_aligned_patches_take_zone_alpha_bitwise(wall: Wall) -> None:
    """把交集改成中心判區不會抓到；本題抓格序、牆序與對齊後多做乘除。"""
    problem = synthetic(wall, 6, 0.5, 0.5)
    actual = late_energy._patch_absorption(problem)
    expected = np.array([[(0.125, 0.25, 0.5, 0.75)[2*(i//3)+j//3]] for i in range(6) for j in range(6)])
    start = Wall.all().index(wall) * 36
    np.testing.assert_array_equal(actual[start:start+36, :1], expected)
    np.testing.assert_array_equal(actual[start:start+36, 1:], expected / 2.0)


@pytest.mark.parametrize("wall", Wall.all())
def test_unaligned_patch_uses_exact_overlap_and_correct_uv(wall: Wall) -> None:
    """u=1/3、v=1/2，2×2 小塊左下跨 u：2/3·1/8＋1/3·1/2=1/4。

    第二小塊跨上列：2/3·1/4＋1/3·3/4=5/12。中心取區與 u/v 互換都會紅。
    """
    problem = synthetic(wall, 2, 1.0/3.0, 0.5)
    actual = late_energy._patch_absorption(problem)
    start = Wall.all().index(wall) * 4
    assert actual[start, 0] == 0.25
    assert actual[start+1, 0] == pytest.approx((2.0 * 0.25 + 0.75) / 3.0,
        rel=contract_value("late_energy_physical_property"), abs=0.0)
    assert actual[start+2, 0] == 0.5
    assert actual[start+3, 0] == 0.75


def test_uniform_zones_equal_whole_wall_with_physical_property_contract() -> None:
    """四區同材必須守現有同阻抗分格契約；抓反射算子與艾林修正漏接分區。"""
    zoned = inputs(zones(Wall.FLOOR))
    whole = replace(zoned, absorption_by_zone=None)
    expected = late_energy.solve_late_energy(whole)
    actual = late_energy.solve_late_energy(zoned)
    np.testing.assert_allclose([b.late_reverberant_energy for b in actual.bands],
        [b.late_reverberant_energy for b in expected.bands],
        rtol=contract_value("late_energy_physical_property"), atol=0.0)


def test_changing_one_zone_material_changes_late_energy() -> None:
    """只改左上材料也要真進求解；抓只用第一區或只用舊整面列。"""
    partition = zones(Wall.FLOOR)
    original = late_energy.solve_late_energy(inputs(partition))
    cells = list(partition["floor"])
    cells[1] = cells[1].model_copy(update={"material": ConstantImpedance(impedance_pa_s_m=950.0)})
    partition["floor"] = tuple(cells)
    changed = late_energy.solve_late_energy(inputs(partition))
    assert changed.bands[0].late_reverberant_energy != original.bands[0].late_reverberant_energy


def test_equivalent_absorption_and_eyring_use_area_weights() -> None:
    """1/3 寬的一區 α=1/8，其餘 α=1/2：等效為 3/8；不准平均區數。"""
    problem = synthetic(Wall.FLOOR, 2, 1.0/3.0, 0.5)
    cells = problem.absorption_by_zone
    assert cells is not None
    cells = {"floor": tuple(replace(cells["floor"][i],
        zone=cells["floor"][i].zone.model_copy(update={"v_m": (0.0, 4.0)}), absorption=(alpha, alpha))
        for i, alpha in ((0, 0.125), (2, 0.5)))}
    whole = {wall: dict(zip(FREQUENCIES, row, strict=True)) for wall, row in
             ((Wall.from_name(name), row) for name, row in problem.absorption_by_wall.items())}
    expected = dict(whole)
    expected[Wall.FLOOR] = dict.fromkeys(FREQUENCIES, 0.375)
    assert eyring_t60_by_band(ROOM, whole, absorption_by_zone=cells) == eyring_t60_by_band(ROOM, expected)
    effective = wall_surfaces.equivalent_absorption_rows(ROOM, problem.absorption_by_wall, cells)
    assert effective["floor"] == (0.375, 0.375)


def test_whole_wall_control_matches_base_per_cell() -> None:
    """起點逐格回歸含 T20／T30 與家具吸音；抓超出各族容差的變動。"""
    from tests.engine._late_zone_control import assert_control, control_inputs, control_values
    assert_control(control_values(control_inputs()))


def test_whole_wall_zoned_entry_is_bitwise_unchanged_in_same_run() -> None:
    """同次執行守整面入口：分區入口沒有矩形也不可多做面積乘除而改末位。"""
    from tests.engine._late_zone_control import control_inputs, control_values, hex_snapshot
    original, zoned = control_inputs(), control_inputs(zoned=True)
    assert zoned.absorption_by_zone == {}
    assert hex_snapshot(zoned.absorption_by_wall) == hex_snapshot(original.absorption_by_wall)
    assert hex_snapshot(control_values(zoned)) == hex_snapshot(control_values(original))


def test_furniture_whole_room_absorption_uses_zone_area_weights() -> None:
    """家具整房分配須先等效，非均勻分區仍守總吸音增加量。"""
    from aosr.physics.furniture_scene import furniture_adjusted_absorption
    problem = synthetic(Wall.FLOOR, 2, 1.0/3.0, 0.5)
    cells = problem.absorption_by_zone
    assert cells is not None
    whole = {Wall.from_name(name): row for name, row in problem.absorption_by_wall.items()}
    effective = wall_surfaces.equivalent_absorption_rows(ROOM, problem.absorption_by_wall, cells)
    expected = furniture_adjusted_absorption(ROOM, FREQUENCIES,
        {Wall.from_name(name): row for name, row in effective.items()}, (1.0, 2.0))
    assert furniture_adjusted_absorption(ROOM, FREQUENCIES, whole, (1.0, 2.0), absorption_by_zone=cells) == expected


def test_geometric_late_orders_pass_zone_rows_to_radiosity() -> None:
    """幾何路的晚期子入口必須帶逐區列，不可退化成每牆一個數字。"""
    from aosr.physics.geometric_lane import solve_geometric_late_energy
    from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
    problem = replace(synthetic(Wall.FLOOR, 2, 1.0/3.0, 0.5), n_per_wall=ART_N_PER_WALL_DEFAULT)
    expected = late_energy.solve_late_energy_by_order(problem, max_order=3)
    actual = solve_geometric_late_energy(room=ROOM, rho_c_pa_s_per_m=CONSTANTS.rho_c,
        frequencies_hz=FREQUENCIES, absorption_by_wall=problem.absorption_by_wall,
        absorption_by_zone=problem.absorption_by_zone, reflection_order_k=3)
    assert actual == expected


def test_report_late_decay_preserves_rectangle_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    """報表衰減子入口須在選頻後保留矩形列，不可只傳整面平均。"""
    from aosr.physics import three_lane_report as report
    from aosr.physics.late_decay import solve_late_decay
    from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
    monkeypatch.setattr(report, "LATE_DECAY_FREQUENCIES_HZ", FREQUENCIES)
    monkeypatch.setattr(report, "GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ", (500.0, 1000.0))
    problem = replace(synthetic(Wall.Y0, 2, 1.0/3.0, 0.5), n_per_wall=ART_N_PER_WALL_DEFAULT)
    result = report._solve_report_late_decay(room=ROOM, absorption_by_wall=problem.absorption_by_wall,
        absorption_by_zone=problem.absorption_by_zone, rho_c_pa_s_per_m=CONSTANTS.rho_c,
        sound_speed_m_s=CONSTANTS.sound_speed_m_s)
    expected = solve_late_decay(problem, sound_speed_m_s=CONSTANTS.sound_speed_m_s)
    assert tuple(b.frequency_hz for b in result.result.bands) == FREQUENCIES
    for actual, reference in zip(result.result.bands, expected.bands, strict=True):
        assert actual.t30_s is not None and reference.t30_s is not None
        np.testing.assert_allclose((actual.t20_s, actual.t30_s, actual.perron_t60_s),
            (reference.t20_s, reference.t30_s, reference.perron_t60_s),
            rtol=contract_value("late_decay_t30_property"), atol=0.0)


@pytest.mark.parametrize("bad", ["empty", "bands", "nan", "negative", "above_one", "gap", "overlap"])
def test_invalid_zone_absorption_is_rejected(bad: str) -> None:
    """低層也驗分區覆蓋與吸音列；不能讓 NaN、缺區或錯頻帶靜默進解。"""
    problem = synthetic(Wall.FLOOR, 2, 1.0/3.0, 0.5)
    assert problem.absorption_by_zone is not None
    cells = list(problem.absorption_by_zone["floor"])
    if bad == "empty":
        cells = []
    elif bad == "gap":
        cells.pop()
    elif bad == "overlap":
        cells.append(cells[-1])
    else:
        row = {"bands": (0.25,), "nan": (float("nan"), 0.25), "negative": (-0.25, 0.25),
               "above_one": (1.25, 0.25)}[bad]
        cells[0] = replace(cells[0], absorption=row)
    with pytest.raises(ValueError):
        late_energy.solve_late_energy(replace(problem, absorption_by_zone={"floor": tuple(cells)}))


def test_whole_zone_factory_preserves_original_rows_bitwise() -> None:
    """整面方案的材料列不經面積乘除；整面入口與舊函式相同。"""
    partition: dict[str, tuple[SurfaceZone, ...]] = {
        w.wall_name(): (WholeWallZone(area="whole_wall", material=MATERIAL),) for w in Wall.all()}
    actual = inputs(partition)
    reference = wall_surfaces.wall_surface_response(dict.fromkeys(Wall.all(), MATERIAL), FREQUENCIES,
        density_kg_m3=CONSTANTS.air_density_kg_m3, sound_speed_m_s=CONSTANTS.sound_speed_m_s)
    assert actual.absorption_by_wall == reference.named_rows(FREQUENCIES)
    assert actual.absorption_by_zone == {}


@pytest.mark.parametrize("wall", Wall.all())
def test_unaligned_v_patch_uses_both_rows(wall: Wall) -> None:
    """3×3 小塊的中央 v 格跨 1/2 高：左右 u 區各取上下材料的一半。"""
    problem = synthetic(wall, 3, 1.0/3.0, 0.5)
    actual = late_energy._patch_absorption(problem)
    start = Wall.all().index(wall) * 9
    np.testing.assert_allclose(actual[[start+1, start+4, start+7], 0], (0.1875, 0.625, 0.625),
        rtol=contract_value("late_energy_physical_property"), atol=0.0)


def test_combined_geometric_lane_refuses_zone_absorption_before_whole_wall_scattering() -> None:
    """晚期子路支援不等於整條散射路支援；不能拿整面阻抗算早期、只替換晚期。"""
    from aosr.geometry.shoebox import Point
    from aosr.physics.geometric_lane import solve_geometric_lane
    from aosr.physics.report_source import SourceModelKind, SourceModelSpec
    from aosr.physics.report_surfaces import RectangleZonesUnsupported
    problem = synthetic(Wall.FLOOR, 2, 1.0/3.0, 0.5)
    with pytest.raises(RectangleZonesUnsupported):
        solve_geometric_lane(room=ROOM, source=Point(1.0, 1.0, 1.0), receiver=Point(2.0, 2.0, 1.0),
            source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL), sound_speed_m_s=CONSTANTS.sound_speed_m_s,
            rho_c_pa_s_per_m=CONSTANTS.rho_c, frequencies_hz=FREQUENCIES,
            impedance_by_wall=dict.fromkeys(Wall.wall_names(), 8200.0),
            absorption_by_wall=problem.absorption_by_wall, absorption_by_zone=problem.absorption_by_zone)
