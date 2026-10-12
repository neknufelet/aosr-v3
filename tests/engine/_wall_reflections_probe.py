"""逐路徑構造反射的隔離雙精度考卷；來源檔由父行程核對。"""
from __future__ import annotations

import math
import json
import traceback
import sys
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import cast
from unittest.mock import patch

import aosr
import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ, GEOMETRIC_BAND_FREQUENCIES_HZ, LowFrequencyAxis
from aosr.geometry.shoebox import Point, Room, Wall, wall_count_signature
from aosr.materials import construction
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, PorousLayer, SurfaceMaterial, ConstructionNote
from aosr.physics import amplitude as amp, geometric_lane as geo, report_path_table as tables, three_lane_report as report
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_io import ReportInput, PathRow, load_input_document
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.room_paths import RoomPath, image_source_paths
from aosr.physics.wall_surfaces import wall_normal_impedances
from aosr.physics.wall_reflections import WallReflectionSources
from aosr.physics.report_path_table import PathTableData

ROOM, SOURCE, RECEIVER = Room(6., 4., 3.), Point(1.1, .9, 1.2), Point(3.7, 2.1, 1.45)
RHO, C = 1.19, 347.
AXIS = (125., 251., 503., 1009., 2017., 4031.)
MODEL = SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL)
WALLS = {wall: 900. + 173. * i for i, wall in enumerate(Wall.all())}
SCATTER = {wall: .05 + i * .04 for i, wall in enumerate(Wall.all())}
BOARD = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.), AirLayer(thickness_m=.43)), backing="rigid")
POROUS = Assembly(layers=(PorousLayer(thickness_m=.05, flow_resistivity_pa_s_m2=12000., material_class="fibrous"), AirLayer(thickness_m=.17)), backing="rigid")


def _materials(two: bool = True, *, depth: float | None = None) -> dict[Wall, SurfaceMaterial]:
    materials: dict[Wall, SurfaceMaterial] = {w: ConstantImpedance(impedance_pa_s_m=z) for w, z in WALLS.items()}
    materials[Wall.X0] = BOARD if depth is None else Assembly(layers=(BOARD.layers[0], AirLayer(thickness_m=depth)), backing="rigid")
    if two:
        materials[Wall.YL] = POROUS
    return materials


def _input(*, order: int = 1) -> ReportInput:
    return load_input_document({"room_m": asdict(ROOM), "source_m": asdict(SOURCE), "receiver_m": asdict(RECEIVER),
        "source_model": {"kind": "omnidirectional"}, "density_kg_m3": RHO, "sound_speed_m_s": C,
        "impedance_pa_s_per_m_by_wall": {w.wall_name(): z for w, z in WALLS.items()},
        "scattering_by_wall": {w.wall_name(): s for w, s in SCATTER.items()}, "reflection_order_k": order}, load_capabilities(config_path("capabilities.toml")),
        load_directivity_defaults(config_path("directivity_defaults.toml")))


def _sources(materials: Mapping[Wall, SurfaceMaterial], axis: tuple[float, ...] = AXIS) -> WallReflectionSources:
    from aosr.physics.wall_reflections import wall_reflection_sources
    return wall_reflection_sources(materials, axis, density_kg_m3=RHO, sound_speed_m_s=C)


def _paths(materials: Mapping[Wall, SurfaceMaterial], axis: tuple[float, ...] = AXIS) -> list[RoomPath]:
    return image_source_paths(ROOM, SOURCE, RECEIVER, C, 3, reflection_sources=_sources(materials, axis))


def _bytes(values: tuple[complex, ...]) -> bytes:
    return np.asarray(values, dtype=np.complex128).tobytes()


def _direct() -> None:
    """獨立逐路徑算角與Z，再交原純量乘積；抓角、核心批次、次方或連乘漂移。"""
    air, ranges = construction_environment(density_kg_m3=RHO, sound_speed_m_s=C)
    geometry = image_source_paths(ROOM, SOURCE, RECEIVER, C, 3)
    for two in (False, True):
        materials, sources = _materials(two), _sources(_materials(two))
        rows = sources.impedances_for_paths(geometry, RECEIVER.as_tuple())
        actual = _paths(materials)
        for path in actual:
            counts = wall_count_signature(path.identity)
            z_rows = {w.wall_name(): (complex(z),) * len(AXIS) for w, z in WALLS.items()}
            for wall, material in materials.items():
                if isinstance(material, ConstantImpedance) or not counts[wall.wall_name()]:
                    continue
                axis = 2 if wall in (Wall.FLOOR, Wall.CEILING) else 0 if wall in (Wall.X0, Wall.XL) else 1
                cos = abs(RECEIVER.as_tuple()[axis] - path.image[axis]) / path.dist_m
                angle = math.degrees(math.acos(cos))
                z, _ = construction.surface_impedance_at_angles(material, jnp.asarray(AXIS), jnp.asarray([angle]), air=air, ranges=ranges)
                expected = np.asarray(z)[:, 0]
                assert cos in rows[wall.wall_name()], (wall, path.identity, cos)
                assert rows[wall.wall_name()][cos].tobytes() == expected.tobytes(), (wall, path.identity)
                z_rows[wall.wall_name()] = tuple(complex(value) for value in expected)
            scalar = amp.reflection_product(amp.Materials(RHO * C, AXIS, z_rows), path.dist_m, RECEIVER.as_tuple(), path.image, counts)
            assert _bytes(path.reflection_product) == _bytes(scalar), path.identity


def _table(materials: Mapping[Wall, SurfaceMaterial], order: int = 3) -> PathTableData:
    return tables.build_path_table(room=ROOM, source=SOURCE, receiver=RECEIVER, sound_speed_m_s=C,
        rho_c_pa_s_per_m=RHO * C, density_kg_m3=RHO, impedance_by_wall=WALLS, frequencies_hz=AXIS,
        scattering_coefficient=(.21,) * len(AXIS), reflection_order_k=order, source_model=MODEL,
        furniture=None, materials_by_wall=materials)


def _readers() -> None:
    """攔下四個讀者的真路徑；抓細密軸或補算悄悄讀回固定牆。"""
    materials = _materials()
    captured: list[tuple[str, list[RoomPath]]] = []

    def recording(*args: object, **kwargs: object) -> list[RoomPath]:
        paths = image_source_paths(*args, **kwargs)  # type: ignore[arg-type]  # expires=2026-12-08 reason=原樣轉交讀者的幾何參數
        captured.append(("lane", paths))
        return paths

    with patch.object(geo, "image_source_paths", recording):
        report._solve_geometric_report_lane(source_model=MODEL, room=ROOM, source=SOURCE, receiver=RECEIVER,
            wall_impedances=WALLS, scattering_by_wall=SCATTER, rho_c_pa_s_per_m=RHO*C, sound_speed_m_s=C,
            reflection_order_k=3, frequencies_hz=AXIS, density_kg_m3=RHO, materials_by_wall=materials)
        with patch.object(report, "GEOMETRIC_BAND_FREQUENCIES_HZ", AXIS):
            report._solve_dense_geometric_report_lane(source_model=MODEL, room=ROOM, source=SOURCE, receiver=RECEIVER,
                wall_impedances=WALLS, scattering_by_wall=SCATTER, rho_c_pa_s_per_m=RHO*C, sound_speed_m_s=C,
                reflection_order_k=3, density_kg_m3=RHO, materials_by_wall=materials)
    with patch.object(tables, "image_source_paths", recording):
        full = _table(materials)
    products = [{p.identity: _bytes(p.reflection_product) for p in paths} for _, paths in captured]
    assert products and all(rows == products[0] for rows in products)
    window = build_reflection_window(_input(), frequencies_hz=AXIS, scattering_coefficient=(.21,)*len(AXIS),
        window_s=.025, materials_by_wall=materials)
    full = _table(materials, window.computed_order_k)
    expected = tuple(PathRow.model_validate(asdict(row)) for row in full.rows
        if row.order > window.report_order_k and row.delay_s - window.direct_delay_s <= window.window_s)
    assert window.rows and window.rows == expected


def _normal() -> None:
    """從獨立垂直入射列手算散射與牆對，抓路徑角取代垂直入射。"""
    materials = _materials()
    rows = wall_normal_impedances(materials, AXIS, density_kg_m3=RHO, sound_speed_m_s=C)
    assert all(z is not None for z in rows.values())
    named = {w.wall_name(): ((complex(z),)*len(AXIS) if isinstance(z, float) else cast(tuple[complex, ...], z))
             for w, z in rows.items()}
    lane = geo.solve_geometric_early_lane(source_model=MODEL, room=ROOM, source=SOURCE, receiver=RECEIVER,
        sound_speed_m_s=C, rho_c_pa_s_per_m=RHO*C, frequencies_hz=AXIS,
        impedance_by_wall={w.wall_name(): z for w, z in WALLS.items()},
        scattering_by_wall={w.wall_name(): s for w, s in SCATTER.items()}, reflection_sources=_sources(materials))
    areas = geo._wall_areas(ROOM)
    expected = []
    for i in range(len(AXIS)):
        weights = {w: areas[w.wall_name()] * abs((complex(named[w.wall_name()][i])-RHO*C)/(complex(named[w.wall_name()][i])+RHO*C))**2 for w in Wall.all()}
        expected.append(sum(weights[w]*SCATTER[w] for w in Wall.all()) / sum(weights.values()))
    assert lane.scattering == tuple(expected)
    screen = build_reflection_screen(_input(), AXIS, materials_by_wall=materials)
    for pair in screen.pairs:
        retained = tuple(tuple(abs(amp.reflection_coefficient(complex(z), 1., RHO*C))**2*(1.-SCATTER[Wall.from_name(face)]) for z in named[face]) for face in pair.faces)
        assert pair.face_retained_energy == retained
        assert pair.round_trip_retained_energy == tuple(a*b for a, b in zip(*retained, strict=True))
    assert any(len(set(row)) > 1 for pair in screen.pairs for row in pair.face_retained_energy)


def _outputs() -> None:
    """兩種頻率軸、路徑表及篩檢都受單面構造及空腔深度改動影響。"""
    constant = {w: ConstantImpedance(impedance_pa_s_m=z) for w, z in WALLS.items()}
    variants: list[Mapping[Wall, SurfaceMaterial]] = [constant, _materials(False), _materials(False, depth=.61)]
    for axis in (GEOMETRIC_LANE_FREQUENCIES_HZ, GEOMETRIC_BAND_FREQUENCIES_HZ):
        energies = [geo.solve_geometric_early_lane(source_model=MODEL, room=ROOM, source=SOURCE, receiver=RECEIVER,
            sound_speed_m_s=C, rho_c_pa_s_per_m=RHO*C, frequencies_hz=axis,
            impedance_by_wall={w.wall_name(): z for w, z in WALLS.items()}, reflection_sources=_sources(m, axis)).reflected_energy for m in variants]
        assert all(a != b for a, b in zip(energies, energies[1:]))
    tables_ = [_table(m).rows for m in variants]
    screens = [build_reflection_screen(_input(), AXIS, materials_by_wall=m).pairs for m in variants]
    assert all(a != b for a, b in zip(tables_, tables_[1:]))
    assert all(a != b for a, b in zip(screens, screens[1:]))


def _reuse() -> None:
    """按牆去重並一次求核心；未打構造牆的路徑仍每面只算一次純量R。"""
    materials = _materials()
    real_core, real_r = construction.surface_impedance_at_angles, amp.reflection_coefficient
    calls: list[SurfaceMaterial] = []
    batches: list[tuple[float, ...]] = []
    coefficients: list[tuple[complex, float, float]] = []

    def core(material: SurfaceMaterial, *args: object, **kwargs: object) -> tuple[jax.Array, tuple[ConstructionNote, ...]]:
        calls.append(material)
        batches.append(tuple(float(a) for a in np.asarray(cast(jax.Array, args[1]))))
        return real_core(material, *args, **kwargs)  # type: ignore[arg-type]  # expires=2026-12-08 reason=只記錄核心呼叫並原樣轉交

    def coefficient(z: complex, cos: float, rho_c: float) -> complex:
        coefficients.append((z, cos, rho_c))
        return real_r(z, cos, rho_c)

    geometry = image_source_paths(ROOM, SOURCE, RECEIVER, C, 3)
    with patch.object(construction, "surface_impedance_at_angles", core), patch.object(amp, "reflection_coefficient", coefficient):
        actual = _paths(materials)
    assert calls == [m for w in Wall.all() if not isinstance(m := materials[w], ConstantImpedance)]
    expected_angles = [tuple(math.degrees(math.acos(cos)) for cos in dict.fromkeys(
        amp.incidence_cos(amp._wall_axis(w.wall_name()), p.dist_m, RECEIVER.as_tuple(), p.image)
        for p in geometry if wall_count_signature(p.identity)[w.wall_name()]))
        for w in Wall.all() if not isinstance(materials[w], ConstantImpedance)]
    assert all(actual[:len(expected)] == expected and set(actual) == set(expected)
               for actual, expected in zip(batches, expected_angles, strict=True))
    expected = [(complex(WALLS[Wall.from_name(w)]), amp.incidence_cos(amp._wall_axis(w), p.dist_m, RECEIVER.as_tuple(), p.image), RHO*C)
        for p in geometry for w in amp.CANONICAL_WALLS if wall_count_signature(p.identity)[w] and isinstance(materials[Wall.from_name(w)], ConstantImpedance)]
    assert coefficients == expected
    assert any(not any(wall_count_signature(p.identity)[w.wall_name()] for w, m in materials.items() if not isinstance(m, ConstantImpedance)) for p in actual if p.order)


def _constant() -> None:
    """封住所有jnp可呼叫物，兩軸、表、窗、篩檢仍能完成且等於舊材料路徑。"""
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("固定阻抗呼叫了JAX")

    materials = {w: ConstantImpedance(impedance_pa_s_m=z) for w, z in WALLS.items()}
    with ExitStack() as stack:
        for name, value in vars(jnp).items():
            if callable(value):
                stack.enter_context(patch.object(jnp, name, forbidden))
        for axis in (GEOMETRIC_LANE_FREQUENCIES_HZ, GEOMETRIC_BAND_FREQUENCIES_HZ):
            rows = {w.wall_name(): (complex(z),)*len(axis) for w, z in WALLS.items()}
            baseline = image_source_paths(ROOM, SOURCE, RECEIVER, C, 3, amp.Materials(RHO*C, axis, rows))
            assert _paths(materials, axis) == baseline
            lane = report._solve_geometric_report_lane if axis == GEOMETRIC_LANE_FREQUENCIES_HZ else report._solve_dense_geometric_report_lane
            lane(source_model=MODEL, room=ROOM, source=SOURCE, receiver=RECEIVER, wall_impedances=WALLS,
                scattering_by_wall=SCATTER, rho_c_pa_s_per_m=RHO*C, sound_speed_m_s=C, reflection_order_k=3)
        _table(materials)
        build_reflection_window(_input(), frequencies_hz=AXIS, scattering_coefficient=(.21,)*len(AXIS), window_s=.025)
        build_reflection_screen(_input(), AXIS)


def _reject_mutant(check: Callable[[], None]) -> None:
    """真數值考卷必須拒絕被植錯的產品函式；存活就是失敗。"""
    try:
        check()
    except AssertionError:
        return
    raise AssertionError("植錯存活：數值考卷沒有抓到錯誤")


def _zero_angle_mutant() -> None:
    real = WallReflectionSources._evaluate_wall

    def zero(self: WallReflectionSources, wall: str, material: SurfaceMaterial,
             cosines: tuple[float, ...]) -> dict[float, np.ndarray[tuple[int, ...], np.dtype[np.complex128]]]:
        row = real(self, wall, material, (1.,))[1.]
        return {cos: row for cos in cosines}

    with patch.object(WallReflectionSources, "_evaluate_wall", zero):
        _reject_mutant(_direct)


def _wrong_axis_mutant() -> None:
    original = amp.incidence_cos

    def wrong(axis: int, dist: float, receiver: tuple[float, float, float], image: tuple[float, float, float]) -> float:
        return original((axis + 1) % 3, dist, receiver, image)

    with patch.object(amp, "incidence_cos", wrong):
        _reject_mutant(_direct)


def _angle_normal_rows() -> dict[str, tuple[complex, ...]]:
    sources = _sources(_materials())
    geometry = image_source_paths(ROOM, SOURCE, RECEIVER, C, 3)
    rows = sources.impedances_for_paths(geometry, RECEIVER.as_tuple())
    normal = sources.normal_rows()
    return {wall: (tuple(complex(v) for v in next(iter(rows[wall].values()))) if wall in rows else z)
            for wall, z in normal.items()}


def _scattering_angle_mutant() -> None:
    rows = _angle_normal_rows()
    with patch.object(WallReflectionSources, "normal_rows", lambda self: rows):
        _reject_mutant(_normal)


def _screen_angle_mutant() -> None:
    from aosr.physics import reflection_screen
    rows = {Wall.from_name(w): z for w, z in _angle_normal_rows().items()}
    with patch.object(reflection_screen, "wall_normal_impedances", lambda *args, **kwargs: rows):
        _reject_mutant(_normal)


def _batch() -> None:
    """正式批次內部覆寫只流向鏡像法；有限元素、晚期的材料來源照舊。"""
    from aosr.physics import three_lane_report_batch as batch
    shared = batch._prepare(source_model=MODEL, room=ROOM, sound_speed_m_s=C, density_kg_m3=RHO,
        impedance_by_wall=WALLS, scattering_by_wall=SCATTER, capability=None, reflection_order_k=3,
        low_frequency_axis=LowFrequencyAxis.SEARCH, materials_by_wall=_materials())
    assert shared.geometric_materials == _materials()
    assert shared.fem_wall_impedances == WALLS
    captured: list[Mapping[Wall, SurfaceMaterial]] = []
    real = report._solve_both_geometric_report_lanes

    def recording(*args: object, **kwargs: object) -> tuple[geo.GeometricLaneResult, geo.GeometricEarlyResult]:
        from typing import cast
        captured.append(cast(Mapping[Wall, SurfaceMaterial], kwargs["materials_by_wall"]))
        return real(*args, **kwargs)  # type: ignore[arg-type]  # expires=2026-12-08 reason=原樣轉交批次鏡像法參數

    with patch.object(report, "_solve_both_geometric_report_lanes", recording):
        batch.solve_reports(source_model=MODEL, room=ROOM, sources={"S": SOURCE}, receivers={"R": RECEIVER},
            sound_speed_m_s=C, density_kg_m3=RHO, impedance_by_wall=WALLS, scattering_by_wall=SCATTER,
            capability=None, reflection_order_k=3, low_frequency_axis=LowFrequencyAxis.SEARCH, batch_fem=True,
            fem_energies={("S", "R"): (0.,)*len(shared.fem_frequencies_hz)}, materials_by_wall=_materials())
    assert captured == [_materials()]


if __name__ == "__main__":
    assert Path(str(aosr.__file__)).resolve() == Path(sys.argv[2]).resolve(), aosr.__file__
    modes = {"direct": _direct, "readers": _readers, "normal": _normal, "outputs": _outputs, "reuse": _reuse, "constant": _constant, "batch": _batch,
             "zero-angle": _zero_angle_mutant, "wrong-axis": _wrong_axis_mutant,
             "scattering-angle": _scattering_angle_mutant, "screen-angle": _screen_angle_mutant}
    if sys.argv[1] == "suite":
        results = {}
        for name, check in modes.items():
            try:
                check()
                results[name] = ""
            except Exception:
                results[name] = traceback.format_exc()
        Path(sys.argv[3]).write_text(json.dumps(results))
    else:
        modes[sys.argv[1]]()
