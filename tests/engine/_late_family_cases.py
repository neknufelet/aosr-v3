"""晚期一族的小房、偏離格線分區與正式 NumPy 參考；不讀真房檔。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from aosr.config.air_absorption_environment import air_absorption_environment
from aosr.physics.air_absorption import air_absorption_fraction
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Wall
from aosr.physics import crossover, late_energy
from aosr.physics.furniture_scene import furniture_absorption_area_m2, furniture_absorption_fraction
from aosr.physics.geometric_lane import _late_share_energy, _room_scattering
from tests.engine._furniture_energy_cases import desk
from tests.engine.test_late_energy_zones import CONSTANTS as CONSTANTS, synthetic

FREQUENCIES = (80.0, 150.0, 190.0, 240.0, 300.0, 500.0, 1000.0, 2000.0)
CASES = ("whole", "zoned", "furniture", "air", "furniture_air")


def problem(name: str) -> late_energy.LateEnergyInputs:
    base = synthetic(Wall.FLOOR, 2, 1.0 / 3.0, 0.5)
    zones = {wall: tuple(replace(cell, absorption=tuple(cell.absorption[0] * (1.0 - i / 20.0)
        for i in range(len(FREQUENCIES)))) for cell in cells) for wall, cells in (base.absorption_by_zone or {}).items()}
    rows = {wall: tuple((0.12 + j / 30.0) * (1.0 + i / 20.0) for i in range(len(FREQUENCIES)))
            for j, wall in enumerate(Wall.wall_names())}
    return replace(base, frequencies_hz=FREQUENCIES, absorption_by_wall=rows,
                   absorption_by_zone=None if name == "whole" else zones)


def alpha_rows(inputs: late_energy.LateEnergyInputs) -> np.ndarray:
    return np.asarray([inputs.absorption_by_wall[w] for w in Wall.wall_names()] +
        [cell.absorption for w in Wall.wall_names() for cell in (inputs.absorption_by_zone or {}).get(w, ())])


def with_alpha(inputs: late_energy.LateEnergyInputs, alpha: np.ndarray) -> late_energy.LateEnergyInputs:
    offset = len(Wall.all())
    zones = {}
    for w in Wall.wall_names():
        cells = (inputs.absorption_by_zone or {}).get(w, ())
        if cells:
            zones[w] = tuple(replace(cell, absorption=tuple(alpha[offset+i])) for i, cell in enumerate(cells))
            offset += len(cells)
    return replace(inputs, absorption_by_wall={w: tuple(alpha[i]) for i, w in enumerate(Wall.wall_names())},
                   absorption_by_zone=zones or None)


def furniture_area() -> np.ndarray:
    return np.asarray(furniture_absorption_area_m2((desk(),), FREQUENCIES, CONSTANTS.rho_c))


def adjusted(inputs: late_energy.LateEnergyInputs, area: np.ndarray | None) -> late_energy.LateEnergyInputs:
    if area is None:
        return inputs
    walls = late_energy._wall_absorption(inputs)
    fraction = furniture_absorption_fraction(inputs.room, FREQUENCIES,
        {w: tuple(walls[i]) for i, w in enumerate(Wall.all())}, tuple(area))
    return with_alpha(inputs, alpha_rows(inputs) + (1.0 - alpha_rows(inputs)) * np.asarray(fraction))


def scattering() -> np.ndarray:
    # 正式 NumPy 入口產生 s；六面同一條曲線，含端點 0、1。JAX 晚期入口只吃結果。
    row = tuple(np.linspace(0.0, 1.0, len(FREQUENCIES)))
    impedances = {w: (complex(4.0 * CONSTANTS.rho_c),) * len(FREQUENCIES) for w in Wall.wall_names()}
    return np.asarray(_room_scattering(problem("whole").room, CONSTANTS.rho_c, impedances,
                                      dict.fromkeys(Wall.wall_names(), row), FREQUENCIES))


def air_fraction(inputs: late_energy.LateEnergyInputs) -> np.ndarray:
    room = inputs.room
    volume = room.Lx * room.Ly * room.Lz
    surface = 2.0 * (room.Lx * room.Ly + room.Ly * room.Lz + room.Lx * room.Lz)
    temperature, humidity, pressure = air_absorption_environment()
    return air_absorption_fraction(FREQUENCIES, 4.0 * volume / surface,
        temperature_c=temperature, relative_humidity_percent=humidity, pressure_pa=pressure)


def reference(inputs: late_energy.LateEnergyInputs, area: np.ndarray | None = None,
              epsilon: np.ndarray | None = None) -> dict[str, np.ndarray]:
    inputs = adjusted(inputs, area)
    if epsilon is not None:
        alpha = alpha_rows(inputs)
        inputs = with_alpha(inputs, alpha + (1.0 - alpha) * epsilon)
    result = late_energy.solve_late_energy_by_order(inputs, max_order=REFLECTION_ORDER_K)
    walls = late_energy._wall_absorption(inputs)
    bands = {w: dict(zip(FREQUENCIES, walls[i], strict=True)) for i, w in enumerate(Wall.all())}
    t60 = crossover.eyring_t60_by_band(inputs.room, bands)
    fs = crossover.schroeder_frequency_hz(inputs.room, t60)
    weights = crossover.crossover_weights(FREQUENCIES, fs)
    return {"alpha_by_wall": walls, "alpha_by_patch": late_energy._patch_absorption(inputs),
        "total_energy": np.asarray([b.late_reverberant_energy for b in result.bands]),
        "energy_by_order": np.asarray([b.energy_by_order for b in result.bands]),
        "tail_energy": np.asarray([b.tail_energy for b in result.bands]),
        "late_share_energy": np.asarray(_late_share_energy(result.bands, tuple(scattering()))),
        "t60": np.asarray(list(t60.values())), "fs": np.asarray(fs),
        "w_fem": np.asarray(weights.w_fem), "w_geo": np.asarray(weights.w_geo)}
