"""報表考卷獨立算空氣：只借已被表值守住的係數，房間幾何與逐格手算留在考卷側。"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from aosr.config.air_absorption_environment import air_absorption_environment
from aosr.geometry.shoebox import Room, Wall
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.physics.air_absorption import air_absorption_fraction


def air_epsilon(frequencies: tuple[float, ...], *, room: Room) -> NDArray[np.float64]:
    volume = room.Lx*room.Ly*room.Lz
    surface = 2.0*(room.Lx*room.Ly+room.Ly*room.Lz+room.Lx*room.Lz)
    temperature, humidity, pressure = air_absorption_environment()
    return air_absorption_fraction(frequencies, 4.0*volume/surface,
        temperature_c=temperature, relative_humidity_percent=humidity, pressure_pa=pressure)


def late_absorption(impedance: float, frequencies: tuple[float, ...], *, room: Room,
                    rho_c: float) -> dict[str, tuple[float, ...]]:
    alpha = complex_random_incidence_absorption(impedance/rho_c)
    row = tuple(alpha+(1.0-alpha)*air_epsilon(frequencies, room=room))
    return dict.fromkeys(Wall.wall_names(), row)
