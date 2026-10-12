"""牆面響應考卷的隔離子行程；父行程指定精度與暫存位置。"""
from __future__ import annotations

from contextlib import ExitStack
import json
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from aosr.config.art_lane import ART_N_PER_WALL_DEFAULT
from aosr.config.frequency_axis import LATE_DECAY_FREQUENCIES_HZ, LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ
from aosr.geometry.shoebox import Room, Wall
from aosr.materials.construction import random_incidence_absorption
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, PorousLayer, SurfaceMaterial
from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.physics.crossover import eyring_t60_by_band
from aosr.physics.geometric_lane import solve_geometric_late_energy
from aosr.physics.late_energy import LateEnergyInputs, LateEnergyOrderResult
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics import three_lane_report as report
from aosr.physics import three_lane_report_batch as batch
from aosr.physics.wall_surfaces import wall_surface_response
CONTROL: dict[str, dict[str, object]] = {}


ROOM = Room(5.3, 4.1, 2.7)
BOARD = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),
                         AirLayer(thickness_m=.03)), backing="rigid")


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("固定阻抗正式準備呼叫了JAX算數函式")


def _select(shared: batch._Shared, expected: dict[str, object]) -> dict[str, object]:
    late = {b.frequency_hz: b for b in shared.late_result.bands}
    decay = {b.frequency_hz: b for b in shared.decay.result.bands}
    late_rows = cast(list[tuple[str, str, str, list[str]]], expected["late"])
    decay_rows = cast(list[tuple[str, str, str]], expected["decay"])
    return {"t60": {str(f): t.hex() for f, t in shared.t60.items()}, "fs": shared.f_s_hz.hex(),
            "late": [(f, late[float.fromhex(f)].late_reverberant_energy.hex(),
                      late[float.fromhex(f)].tail_energy.hex(),
                      [e.hex() for e in late[float.fromhex(f)].energy_by_order])
                     for f, *_rest in late_rows],
            "decay": [(f, decay[float.fromhex(f)].t20_s.hex(),
                       None if decay[float.fromhex(f)].t30_s is None else
                       cast(float, decay[float.fromhex(f)].t30_s).hex()) for f, *_rest in decay_rows]}


def _constant() -> dict[str, object]:
    answers: dict[str, object] = {}
    # 封住所有jnp可呼叫入口，不只封住單一陣列建構入口。
    overrides = {name: _forbidden for name, value in vars(jnp).items() if callable(value)}
    with ExitStack() as stack:
        for name, function in overrides.items():
            stack.enter_context(patch.object(jnp, name, function))
        for name, expected in CONTROL.items():
            walls = cast(dict[str, float], expected["walls"])
            shared = batch._prepare(source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
                room=ROOM, sound_speed_m_s=343., density_kg_m3=1.2,
                impedance_by_wall={Wall.from_name(wall): z for wall, z in walls.items()},
                scattering_by_wall=None, capability=None, reflection_order_k=3,
                low_frequency_axis=LowFrequencyAxis.SEARCH)
            assert shared.surfaces.notes_by_wall == {wall: () for wall in Wall.all()}
            answers[name] = _select(shared, expected)
    return answers


def _materials(material: SurfaceMaterial) -> dict[Wall, SurfaceMaterial]:
    walls = cast(dict[str, float], CONTROL["unequal_hard"]["walls"])
    result: dict[Wall, SurfaceMaterial] = {Wall.from_name(w): ConstantImpedance(impedance_pa_s_m=z)
                                         for w, z in walls.items()}
    result[Wall.FLOOR] = material
    return result


def _direct_rows(materials: dict[Wall, SurfaceMaterial], frequencies: tuple[float, ...]) -> dict[str, tuple[float, ...]]:
    air = load_physics_constants(config_path("physics_constants.toml")).model_copy(
        update={"air_density_kg_m3": 1.2, "sound_speed_m_s": 343.})
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    rows = {}
    for wall, material in materials.items():
        if isinstance(material, ConstantImpedance):
            value = complex_random_incidence_absorption(complex(material.impedance_pa_s_m) / (1.2 * 343.))
            rows[wall.wall_name()] = (value,) * len(frequencies)
        else:
            values, _notes = random_incidence_absorption(material, jnp.asarray(frequencies), air=air, ranges=ranges)
            rows[wall.wall_name()] = tuple(float(v) for v in values)
    return rows


def _construction() -> dict[str, object]:
    fine = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[1]
    materials = _materials(BOARD)
    surfaces, t60, _fs, late, decay = batch._prepare_late_family(room=ROOM,
        materials_by_wall=materials, density_kg_m3=1.2, sound_speed_m_s=343.,
        report_frequencies_hz=fine, reflection_order_k=3)
    # 三條軸各自直接呼叫材料核心；會抓聯集軸漏點、取值或內插錯誤。
    for axis in (fine, LATE_DECAY_FREQUENCIES_HZ, SCHROEDER_T60_BANDS_HZ):
        assert surfaces.named_rows(axis) == _direct_rows(materials, axis)
    bands = _direct_rows(materials, SCHROEDER_T60_BANDS_HZ)
    expected_t60 = eyring_t60_by_band(ROOM, {Wall.from_name(w): dict(zip(SCHROEDER_T60_BANDS_HZ, row, strict=True))
                                          for w, row in bands.items()})
    assert t60 == expected_t60
    assert t60[500.] != t60[1000.]
    expected_late = solve_geometric_late_energy(room=ROOM, rho_c_pa_s_per_m=1.2 * 343.,
        frequencies_hz=fine, absorption_by_wall=_direct_rows(materials, fine), reflection_order_k=3)
    assert late == expected_late
    expected_decay = report._solve_report_late_decay(room=ROOM, rho_c_pa_s_per_m=1.2 * 343.,
        sound_speed_m_s=343., absorption_by_wall=_direct_rows(materials, LATE_DECAY_FREQUENCIES_HZ))
    assert decay == expected_decay
    _changes_and_notes(materials, surfaces.frequencies_hz, late, decay)
    return {"verified": True, "alpha_500_1000": bands["floor"]}


def _changes_and_notes(materials: dict[Wall, SurfaceMaterial], axis: tuple[float, ...],
                       late: LateEnergyOrderResult, decay: report._ReportLateDecay) -> None:
    baseline = _materials(ConstantImpedance(impedance_pa_s_m=8200.))
    fine = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[1]
    _, _, _, old_late, old_decay = batch._prepare_late_family(room=ROOM, materials_by_wall=baseline,
        density_kg_m3=1.2, sound_speed_m_s=343., report_frequencies_hz=fine, reflection_order_k=3)
    assert late != old_late
    old_bands = {b.frequency_hz: b for b in old_decay.result.bands}
    assert any(b.t20_s != old_bands[b.frequency_hz].t20_s for b in decay.result.bands)
    assert any(b.t30_s != old_bands[b.frequency_hz].t30_s for b in decay.result.bands)
    old = wall_surface_response(materials, axis, density_kg_m3=1.2, sound_speed_m_s=343.)
    changed = wall_surface_response(materials, axis, density_kg_m3=1.2, sound_speed_m_s=330.)
    assert old.absorption_by_wall[Wall.FLOOR] != changed.absorption_by_wall[Wall.FLOOR]
    changed_air = load_physics_constants(config_path("physics_constants.toml")).model_copy(
        update={"air_density_kg_m3": 1.2, "sound_speed_m_s": 330.})
    direct_changed, _ = random_incidence_absorption(BOARD, jnp.asarray(axis), air=changed_air,
        ranges=load_porous_model_ranges(config_path("porous_model_ranges.toml")))
    np.testing.assert_array_equal(changed.absorption_by_wall[Wall.FLOOR], direct_changed)
    porous = Assembly(layers=(PorousLayer(thickness_m=.03, flow_resistivity_pa_s_m2=72900.,
                                       material_class="fibrous"), AirLayer(thickness_m=.03)), backing="rigid")
    notes_response, *_ = batch._prepare_late_family(room=ROOM, materials_by_wall=_materials(porous),
        density_kg_m3=1.2, sound_speed_m_s=343., report_frequencies_hz=fine, reflection_order_k=3)
    air = load_physics_constants(config_path("physics_constants.toml")).model_copy(
        update={"air_density_kg_m3": 1.2, "sound_speed_m_s": 343.})
    direct, notes = random_incidence_absorption(porous, jnp.asarray(axis), air=air,
        ranges=load_porous_model_ranges(config_path("porous_model_ranges.toml")))
    assert notes_response.notes_by_wall[Wall.FLOOR] == notes
    assert any(n.frequencies_hz for n in notes)
    np.testing.assert_array_equal(notes_response.absorption_by_wall[Wall.FLOOR], direct)
    # 能量底層只能消費α，不得偷偷再從阻抗重算。
    inputs = LateEnergyInputs(ROOM, 1.2 * 343., (500., 1000.), old.named_rows((500., 1000.)),
                              ART_N_PER_WALL_DEFAULT, 1.)
    from aosr.physics.late_energy import _wall_absorption
    np.testing.assert_array_equal(_wall_absorption(inputs)[Wall.all().index(Wall.FLOOR)],
                                  old.named_rows((500., 1000.))["floor"])


def main() -> None:
    CONTROL.update(json.loads(sys.stdin.read()))
    if sys.argv[1] == "constant":
        result = _constant()
    elif sys.argv[1] == "construction":
        result = _construction()
    else:
        try:
            wall_surface_response(_materials(BOARD), (500., 1000.), density_kg_m3=1.2, sound_speed_m_s=343.)
        except ValueError as exc:
            result = {"error": str(exc)}
        else:
            raise AssertionError("構造在單精度行程未拒收")
    Path(sys.argv[2]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
