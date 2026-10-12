"""複數有限元素考卷的子行程：精度與 CPU 釘選由父行程注入；跨機器釘主線與構造接線的題用替身解法器。"""
from __future__ import annotations

import json
import sys
from contextlib import AbstractContextManager, ExitStack
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from aosr import runtime
from aosr.config.construction_environment import construction_environment
from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, SurfaceMaterial
from aosr.physics import fem_helmholtz as fem, three_lane_report as report, three_lane_report_batch as batch
from aosr.physics.report_io import SolverInputs
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.reporting.fem_slices import FemInputs, FemKey, fem_inputs
from tests.engine._fem_shared_cases import small_scheme
from tests.engine._solver_stand_in import StandInSolver

CONTROL: dict[str, object] = {}
ROOM = Room(2.3, 1.7, 1.4)
RHO, SPEED = 1.2, 343.
SOURCE_MODEL = SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL)


def _walls() -> dict[Wall, float]:
    values = cast(dict[str, float], CONTROL['impedance_pa_s_per_m_by_wall'])
    return {Wall.from_name(w): z for w, z in values.items()}


def _materials() -> dict[Wall, SurfaceMaterial]:
    return {w: ConstantImpedance(impedance_pa_s_m=z) for w, z in _walls().items()}


def _inputs(materials: dict[Wall, SurfaceMaterial] | None = None) -> FemInputs:
    scheme = small_scheme()
    solved = SolverInputs(ROOM, Point(.4, .5, .6), SOURCE_MODEL, Point(1.5, .9, .7),
                          SPEED, RHO, _walls(), None, 3, LowFrequencyAxis.SEARCH)
    if materials is None:
        return fem_inputs(scheme, solved)
    return fem_inputs(scheme, solved, materials_by_wall=materials)


def _points(name: str) -> dict[str, Point]:
    return {key: Point(*point) for key, point in cast(dict[str, list[float]], CONTROL[name]).items()}


def _energy(walls: fem.FrequencyWallImpedances, axis: tuple[float, ...]) -> dict[tuple[str, str], tuple[float, ...]]:
    return report._solve_fem_energies(room=ROOM, sources=_points('sources'), receivers=_points('receivers'),
        wall_impedances=walls, frequencies_hz=axis, density_kg_m3=RHO, sound_speed_m_s=SPEED)


def _stand_in() -> AbstractContextManager[object]:
    """把 PARDISO 換成跨機器逐位相同的替身（為什麼見 tests/engine/_solver_stand_in.py）。"""
    runtime.preload_mkl()
    from pydiso import mkl_solver
    return patch.object(mkl_solver, 'MKLPardisoSolver', StandInSolver)


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError('固定阻抗有限元素碰了 JAX')


def _formal_energies(
    rows: fem.FrequencyWallImpedances | None = None, *, batched: bool = True,
) -> dict[tuple[str, str], tuple[float, ...]]:
    """保留正式準備與求解；只省略與本題無關的鏡像接合。

    Scene 尚未開放構造，故構造只注入已求值的內部有限元素欄，其餘準備照真跑。
    """
    original = batch._prepare

    def prepare(**kwargs: object) -> batch._Shared:
        shared = original(source_model=SOURCE_MODEL, room=ROOM,
            density_kg_m3=RHO, sound_speed_m_s=SPEED, impedance_by_wall=_walls(),
            scattering_by_wall=None, capability=None, reflection_order_k=3,
            low_frequency_axis=LowFrequencyAxis.SEARCH)
        return shared if rows is None else replace(shared, fem_wall_impedances=rows)

    def pair(shared: batch._Shared, source: Point, receiver: Point,
             fem_energy: tuple[float, ...]) -> tuple[float, ...]:
        return fem_energy

    with patch.object(batch, '_prepare', prepare), patch.object(batch, '_pair_report', pair):
        values = batch.solve_reports(source_model=SOURCE_MODEL, room=ROOM,
            sources=_points('sources'), receivers=_points('receivers'),
            density_kg_m3=RHO, sound_speed_m_s=SPEED, impedance_by_wall=_walls(),
            scattering_by_wall=None, capability=None, reflection_order_k=3,
            low_frequency_axis=LowFrequencyAxis.SEARCH, batch_fem=batched)
    return cast(dict[tuple[str, str], tuple[float, ...]], values)


def _constant() -> dict[str, object]:
    axis = tuple(float.fromhex(f) for f in cast(list[str], CONTROL['frequencies_hz']))
    with ExitStack() as stack:
        for name, value in vars(jnp).items():
            if callable(value):
                stack.enter_context(patch.object(jnp, name, _forbidden))
        stack.enter_context(_stand_in())
        shared = batch._prepare(source_model=SOURCE_MODEL, room=ROOM,
            density_kg_m3=RHO, sound_speed_m_s=SPEED, impedance_by_wall=_walls(),
            scattering_by_wall=None, capability=None, reflection_order_k=3,
            low_frequency_axis=LowFrequencyAxis.SEARCH)
        inputs = _inputs()
        assert shared.fem_wall_impedances == inputs.wall_impedances
        assert all(type(z) is float for z in shared.fem_wall_impedances.values()), '固定阻抗要原樣的 float'
        assert all(type(z) is float for z in inputs.wall_impedances.values()), '固定阻抗要原樣的 float'
        assert FemKey.from_inputs(inputs) == FemKey.from_inputs(replace(inputs, wall_impedances=shared.fem_wall_impedances))
        energies = _energy(shared.fem_wall_impedances, axis)
        indices = tuple(inputs.frequencies_hz.index(f) for f in axis)
        for batched in (False, True):
            formal = _formal_energies(batched=batched)
            assert {pair: tuple(row[i] for i in indices) for pair, row in formal.items()} == energies
        for (s, r), values in energies.items():
            single = report._solve_fem_energy(room=ROOM, source=_points('sources')[s], receiver=_points('receivers')[r],
                wall_impedances=inputs.wall_impedances, frequencies_hz=axis,
                density_kg_m3=RHO, sound_speed_m_s=SPEED)
            assert single == values
        # 搜尋分片讀同一條正式軸；選控制頻點的索引，錨定仍是全軸 A(f0)。
        from aosr.physics.fem_batch import solve_fem_energies_many
        sliced = solve_fem_energies_many(room=ROOM, candidates={'c': (_points('sources'), _points('receivers'))},
            wall_impedances=inputs.wall_impedances, frequencies_hz=inputs.frequencies_hz,
            frequency_indices=tuple(inputs.frequencies_hz.index(f) for f in axis),
            density_kg_m3=RHO, sound_speed_m_s=SPEED)['c']
        assert sliced == energies
    return {'energy': {f'{s}|{r}': [e.hex() for e in row] for (s, r), row in energies.items()},
            'key': FemKey.from_inputs(inputs).model_dump_json()}


def _construction() -> dict[str, object]:
    with _stand_in():
        return _construction_checked()


def _construction_checked() -> dict[str, object]:
    from aosr.materials.construction import surface_impedance
    axis = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[0]
    materials = _materials()
    keys = []
    for depth in (.03, .05):
        materials[Wall.XL] = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),
                                             AirLayer(thickness_m=depth)), backing='rigid')
        rows = batch._prepare_fem_impedances(materials, axis, density_kg_m3=RHO, sound_speed_m_s=SPEED)
        inputs = _inputs(materials)
        assert inputs.wall_impedances == rows
        key = FemKey.from_inputs(inputs)
        assert key == FemKey.from_inputs(replace(inputs, wall_impedances=rows))
        assert FemKey.model_validate_json(key.model_dump_json()) == key
        keys.append(key)
        air, ranges = construction_environment(density_kg_m3=RHO, sound_speed_m_s=SPEED)
        direct, _notes = surface_impedance(materials[Wall.XL], jnp.asarray(axis), air=air, ranges=ranges)
        expected: fem.FrequencyWallImpedances = {**_walls(), Wall.XL: tuple(complex(z) for z in direct)}
        assert rows == expected
        picks = tuple(float.fromhex(f) for f in cast(list[str], CONTROL['frequencies_hz']))
        selected: fem.FrequencyWallImpedances = {**_walls(), Wall.XL: tuple(complex(direct[axis.index(f)]) for f in picks)}
        indices = tuple(axis.index(f) for f in picks)
        from aosr.physics.fem_batch import solve_fem_energies_many
        actual = solve_fem_energies_many(room=ROOM, candidates={'c': (_points('sources'), _points('receivers'))},
            wall_impedances=rows, frequencies_hz=axis, frequency_indices=indices,
            density_kg_m3=RHO, sound_speed_m_s=SPEED)['c']
        assert actual == _energy(selected, picks)
        formal = _formal_energies(rows)
        assert {pair: tuple(row[i] for i in indices) for pair, row in formal.items()} == actual
        assert actual != _energy(_walls(), picks)
        cells = key.impedance_hex[Wall.all().index(Wall.XL)]
        assert cells == tuple((complex(z).real.hex(), complex(z).imag.hex()) for z in direct)
        # 保持實部，單獨改虛部；保持虛部，單獨改實部，都必須翻鑰匙。
        for offset in (1., 1j):
            changed = {**rows, Wall.XL: tuple(complex(z) + offset for z in direct)}
            assert FemKey.from_inputs(replace(inputs, wall_impedances=changed)) != key
    assert keys[0] != keys[1]
    assert all(key != FemKey.from_inputs(_inputs()) for key in keys)
    return {'verified': True}


def _construction_single() -> dict[str, object]:
    with _stand_in():
        return _construction_single_checked()


def _construction_single_checked() -> dict[str, object]:
    """逐對求解那條正式路（batch_fem=False）也要吃到構造；裁判是直接餵逐頻列的分片求解。"""
    from aosr.physics.fem_batch import solve_fem_energies_many
    axis = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[0]
    materials = _materials()
    materials[Wall.XL] = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),
                                         AirLayer(thickness_m=.03)), backing='rigid')
    rows = batch._prepare_fem_impedances(materials, axis, density_kg_m3=RHO, sound_speed_m_s=SPEED)
    picks = tuple(float.fromhex(f) for f in cast(list[str], CONTROL['frequencies_hz']))
    indices = tuple(axis.index(f) for f in picks)
    sliced = solve_fem_energies_many(room=ROOM, candidates={'c': (_points('sources'), _points('receivers'))},
        wall_impedances=rows, frequencies_hz=axis, frequency_indices=indices,
        density_kg_m3=RHO, sound_speed_m_s=SPEED)['c']
    formal = _formal_energies(rows, batched=False)
    assert {pair: tuple(row[i] for i in indices) for pair, row in formal.items()} == sliced
    assert sliced != _energy(_walls(), picks)
    return {'verified': True}


def _scene_air() -> dict[str, object]:
    """構造的空氣用場景的密度與聲速；場景刻意跟產品設定檔不同，一次只改一格。"""
    from aosr.config.physics_constants import load_physics_constants
    from aosr.config.paths import config_path
    from aosr.materials.construction import surface_impedance
    config = load_physics_constants(config_path("physics_constants.toml"))
    assert (config.air_density_kg_m3, config.sound_speed_m_s) == (RHO, SPEED), "這題假設控制場景等於設定檔"
    axis = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[0]
    materials = _materials()
    materials[Wall.XL] = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),
                                         AirLayer(thickness_m=.05)), backing='rigid')
    default_rows = batch._prepare_fem_impedances(materials, axis, density_kg_m3=RHO, sound_speed_m_s=SPEED)
    for rho, speed in ((RHO * 1.05, SPEED), (RHO, SPEED * 1.03)):
        air, ranges = construction_environment(density_kg_m3=rho, sound_speed_m_s=speed)
        direct, _notes = surface_impedance(materials[Wall.XL], jnp.asarray(axis), air=air, ranges=ranges)
        expected = tuple(complex(z) for z in direct)
        prepared = batch._prepare_fem_impedances(materials, axis, density_kg_m3=rho, sound_speed_m_s=speed)
        solved = SolverInputs(ROOM, Point(.4, .5, .6), SOURCE_MODEL, Point(1.5, .9, .7),
                              speed, rho, _walls(), None, 3, LowFrequencyAxis.SEARCH)
        sliced = fem_inputs(small_scheme(), solved, materials_by_wall=materials).wall_impedances
        assert prepared[Wall.XL] == expected and sliced[Wall.XL] == expected
        assert prepared[Wall.XL] != default_rows[Wall.XL]
    return {'verified': True}


def _pressure(room: Room, walls: fem.FrequencyWallImpedances, axis: tuple[float, ...]) -> np.ndarray:
    mesh = generate_shoebox_mesh(room, max_frequency_hz=300., elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
                                sound_speed_m_s=SPEED, random_seed=FEM_MESH_RANDOM_SEED)
    return fem.solve_fem_helmholtz(mesh, wall_impedances=walls, source=Point(.25, .09, .11),
        receiver=Point(1.1, .16, .13), frequencies_hz=axis, density_kg_m3=RHO, sound_speed_m_s=SPEED)


def _convention() -> dict[str, object]:
    from aosr.physics.wall_surfaces import wall_normal_impedances
    axis = (45., 95., 160., 230.)
    materials: dict[Wall, SurfaceMaterial | None] = {w: None for w in Wall.all()}
    materials[Wall.X0] = ConstantImpedance(impedance_pa_s_m=2 * RHO * SPEED)
    materials[Wall.XL] = Assembly(layers=(AirLayer(thickness_m=.3),), backing='rigid')
    rows = wall_normal_impedances(materials, axis, density_kg_m3=RHO, sound_speed_m_s=SPEED)
    reference = _pressure(Room(2.3, .25, .25), {**rows, Wall.XL: None}, axis)
    correct = _pressure(Room(2., .25, .25), rows, axis)
    gap = cast(tuple[complex, ...], rows[Wall.XL])
    conjugated = _pressure(Room(2., .25, .25), {**rows, Wall.XL: tuple(z.conjugate() for z in gap)}, axis)
    return {'frequencies': axis, 'correct': (abs(correct-reference)/abs(reference)).tolist(),
            'conjugated': (abs(conjugated-reference)/abs(reference)).tolist()}


def main() -> None:
    global CONTROL
    CONTROL = json.loads(sys.stdin.read())
    functions = {'constant': _constant, 'construction': _construction, 'convention': _convention,
                 'scene_air': _scene_air, 'construction_single': _construction_single}
    import aosr
    result = {**functions[sys.argv[1]](), 'aosr_file': str(Path(aosr.__file__).resolve())}
    Path(sys.argv[2]).write_text(json.dumps(result))


if __name__ == '__main__':
    main()
