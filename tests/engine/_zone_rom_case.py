"""#561 分區替身的小房間固定輸入；不使用正式示範房。"""
from __future__ import annotations

import numpy as np

from aosr.config.fem_reduced import load_reduced_model_settings
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import WallPartitions
from aosr.physics.fem_helmholtz import FrequencyWallImpedances
from aosr.physics.fem_reduced import FullOperators, export_room_operators

AXIS = (40.0, 50.0, 63.0, 80.0)
ROOM = Room(1.6, 1.2, 1.0)
PARTITIONS: WallPartitions = {Wall.X0: (("right", (0.6, 1.2), (0.0, 1.0)),
                                        ("left", (0.0, 0.6), (0.0, 1.0)))}
SETTINGS = load_reduced_model_settings(config_path("fem_reduced.toml")).model_copy(update={
    "initial_impedance_count": 8, "initial_frequency_count": 4, "max_full_solves": 32})


def operators() -> FullOperators:
    return export_room_operators(ROOM, {"source": Point(0.3, 0.3, 0.4)}, {"seat": Point(1.2, 0.7, 0.6)},
        density_kg_m3=1.2, sound_speed_m_s=343.0, physics_identity="phys-v1:" + "0" * 64,
        max_frequency_hz=80.0)


def materials(seed: int, count: int) -> list[FrequencyWallImpedances]:
    rng = np.random.default_rng(seed)
    return [{wall: tuple(complex(z) for z in 411.6 * (rng.uniform(2, 12) - 1j * rng.uniform(0.5, 2) * 100 / np.asarray(AXIS)))
             for wall in Wall.all()} for _ in range(count)]
