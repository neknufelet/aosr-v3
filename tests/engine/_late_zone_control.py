"""727acf825736dbd2da26280fabb88e97cd16e557 實量控制的可重播題目。"""
from collections.abc import Iterator
from dataclasses import asdict

from aosr.geometry.shoebox import Room, Wall
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics import late_energy, wall_surfaces
from aosr.physics.crossover import eyring_t60_by_band, schroeder_frequency_hz
from aosr.physics.furniture_scene import furniture_adjusted_absorption
from aosr.physics.late_decay import solve_late_decay
from aosr.physics.report_surfaces import WholeWallZone
from tests.engine._cross_cpu_tolerance import GEOMETRIC_REL, LINEAR_SOLVE_REL, assert_within, cell
from tests.engine._late_zone_control_answers import CONTROL
from tests.engine.test_late_energy_zones import CONSTANTS


def control_inputs(*, zoned: bool = False) -> late_energy.LateEnergyInputs:
    """六面異材、非整數房長，抓整面也繞進矩形乘除後的末位差。"""
    frequencies = (125.0, 500.0, 1000.0)
    materials = {w: ConstantImpedance(impedance_pa_s_m=z) for w, z in
                 zip(Wall.all(), (8200.0, 1646.4, 3100.0, 950.0, 12500.0, 2300.0), strict=True)}
    kwargs = dict(density_kg_m3=CONSTANTS.air_density_kg_m3, sound_speed_m_s=CONSTANTS.sound_speed_m_s)
    if zoned:
        response = wall_surfaces.zoned_wall_surface_response(Room(6.7, 4.1, 2.9),
            {wall.wall_name(): (WholeWallZone(area="whole_wall", material=material),)
             for wall, material in materials.items()}, frequencies, **kwargs)
        return late_energy.LateEnergyInputs(Room(6.7, 4.1, 2.9), CONSTANTS.rho_c, frequencies,
            response.named_rows(frequencies), 6, 0.9, absorption_by_zone=response.named_zone_rows(frequencies))
    original = wall_surfaces.wall_surface_response(materials, frequencies, **kwargs)
    return late_energy.LateEnergyInputs(Room(6.7, 4.1, 2.9), CONSTANTS.rho_c, frequencies,
                                       original.named_rows(frequencies), 6, 0.9)


def control_values(problem: late_energy.LateEnergyInputs) -> dict[str, object]:
    """六族原始輸出；分區入口無矩形時也送出空列，跟原整面入口同次對照。"""
    bands = {Wall.from_name(name): dict(zip(problem.frequencies_hz, row, strict=True))
             for name, row in problem.absorption_by_wall.items()}
    t60 = eyring_t60_by_band(problem.room, bands, absorption_by_zone=problem.absorption_by_zone)
    return dict(late=asdict(late_energy.solve_late_energy(problem)),
        orders=asdict(late_energy.solve_late_energy_by_order(problem, max_order=3)),
        decay=asdict(solve_late_decay(problem, sound_speed_m_s=CONSTANTS.sound_speed_m_s)),
        furniture={wall.wall_name(): row for wall, row in furniture_adjusted_absorption(problem.room,
            problem.frequencies_hz, {Wall.from_name(name): row for name, row in problem.absorption_by_wall.items()},
            (1.0, 2.0, 3.0), absorption_by_zone=problem.absorption_by_zone).items()},
        t60=t60, fs=schroeder_frequency_hz(problem.room, t60))


def _leaves(value: object, path: str = "") -> Iterator[tuple[str, float | int | bool]]:
    """按完整欄名與索引逐格攤開，不分組加總；未知型別直接拒收。"""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}.{index}")
    else:
        assert isinstance(value, (float, int, bool)), (path, value)
        yield path, value


def hex_snapshot(values: object) -> dict[str, str | int | bool]:
    """每格浮點轉 hex（十六進位原值），同次對照也能分辨正負零。"""
    return {path: value.hex() if isinstance(value, float) else value for path, value in _leaves(values)}


# 逐欄分級照決策紙第 6 條，看那一格實際經過的計算，不看它掛在哪一族底下：
# 只經輸入與四則的逐位（碰撞頻率＝c·S／4V）；吸音率（Paris 積分）、面積平均 α、艾林比、家具、艾林 T60、交接是閉式①；
# 輻射度能量、逐階能量、Perron 特徵值與衰減擬合經線性代數②。沒列到的欄位直接紅，不准默默落進較寬的那一級。
_EXACT_FIELDS = frozenset({"frequency_hz", "collision_frequency_hz"})
_GEOMETRIC_FIELDS = frozenset({"alpha_by_wall", "alpha_bar", "eyring_ratio"})
_GEOMETRIC_FAMILIES = frozenset({"furniture", "t60", "fs"})
_LINEAR_FIELDS = frozenset({"raw_reverberant_energy", "late_reverberant_energy", "energy_by_order", "tail_energy",
                            "perron_t60_s", "slope_db_per_s", "soft_weight_sum", "t20_s", "t30_s",
                            "t30_slope_db_per_s", "t30_soft_weight_sum"})


def _float_tier(path: str) -> float | None:
    """那一格的相對容差；``None`` 是逐位。"""
    parts = set(path.split("."))
    if parts & _EXACT_FIELDS:
        return None
    if path.split(".", 1)[0] in _GEOMETRIC_FAMILIES or parts & _GEOMETRIC_FIELDS:
        return GEOMETRIC_REL
    if parts & _LINEAR_FIELDS:
        return LINEAR_SOLVE_REL
    raise AssertionError(f"控制組欄位沒有分級：{path}")


def assert_control(values: object) -> None:
    """鍵、頻率、旗標與只經四則的格逐位，其餘照 ``_float_tier`` 逐格比，不另造尺。"""
    actual = hex_snapshot(values)
    assert actual.keys() == CONTROL.keys(), "控制組欄位不同"
    cells = []
    for path, expected in CONTROL.items():
        value = actual[path]
        assert type(value) is type(expected), path
        rel = _float_tier(path) if isinstance(expected, str) else None
        if rel is None:
            assert value == expected, path
        else:
            assert isinstance(value, str)
            cells.append(cell(path, value, expected, rel=rel, abs_=0.0))
    assert_within(cells)
