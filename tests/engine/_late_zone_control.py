"""727acf825736dbd2da26280fabb88e97cd16e557 實量控制的可重播題目。"""
from dataclasses import asdict
import hashlib
import json

from aosr.geometry.shoebox import Room, Wall
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics import late_energy, wall_surfaces
from aosr.physics.crossover import eyring_t60_by_band, schroeder_frequency_hz
from tests.engine.test_late_energy_zones import CONSTANTS


def control_inputs() -> late_energy.LateEnergyInputs:
    """六面異材、非整數房長，抓整面也繞進矩形乘除後的末位差。"""
    frequencies = (125.0, 500.0, 1000.0)
    materials = {w: ConstantImpedance(impedance_pa_s_m=z) for w, z in
                 zip(Wall.all(), (8200.0, 1646.4, 3100.0, 950.0, 12500.0, 2300.0), strict=True)}
    response = wall_surfaces.wall_surface_response(materials, frequencies,
        density_kg_m3=CONSTANTS.air_density_kg_m3, sound_speed_m_s=CONSTANTS.sound_speed_m_s)
    return late_energy.LateEnergyInputs(Room(6.7, 4.1, 2.9), CONSTANTS.rho_c, frequencies,
                                       response.named_rows(frequencies), 6, 0.9)


def control_digest() -> str:
    """hash 包含浮點原始數字的 JSON 表示，可抓一位元差，不對答案作四捨五入。"""
    problem = control_inputs()
    bands = {Wall.from_name(name): dict(zip(problem.frequencies_hz, row, strict=True))
             for name, row in problem.absorption_by_wall.items()}
    t60 = eyring_t60_by_band(problem.room, bands)
    values = dict(late=asdict(late_energy.solve_late_energy(problem)),
        orders=asdict(late_energy.solve_late_energy_by_order(problem, max_order=3)),
        t60=t60, fs=schroeder_frequency_hz(problem.room, t60))
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
