"""反射可微考卷的自足小房與現有 Python 裁判；不讀外部正式結果。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

import numpy as np
from numpy.typing import NDArray

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityPurpose, load_quality_targets
from aosr.geometry.shoebox import wall_count_signature
from aosr.geometry.shoebox import Point
from aosr.physics import amplitude as amp, amplitude_jax as aj
from aosr.physics.report_path_table import build_path_table
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.source_directivity import SourceModel, departure_direction, pressure_factor_for_direction
from aosr.physics.report_io import solver_inputs
from aosr.physics.room_paths import RoomPath, image_source_paths
from aosr.scoring.direction_zones import DirectionZone, listening_axis
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from aosr.scoring.reflections_cost import _zone_excess, _zone_threshold, cost_reflections_evaluation
from tests.engine import _reflection_fixtures as fixtures
from tests.engine._directivity import DIRECTIVITY

CASES = ("room", "small_extension", "complex_frequency")


@cache
def purpose() -> QualityPurpose:
    return load_quality_targets(config_path("quality_targets.toml")).purpose("dedicated_two_channel_listening_room")


@dataclass(frozen=True)
class Case:
    records: tuple[ReflectionInput, ReflectionInput]
    paths: tuple[tuple[RoomPath, ...], ...]
    tables: tuple[aj.PathTable, ...]
    directions: tuple[NDArray[np.float64], ...]
    axis: tuple[float, float]
    z: NDArray[np.complex128]
    scattering: NDArray[np.float64]
    rho_c: float
    sound_speed: float


@cache
def case(name: str) -> Case:
    room = fixtures._SMALL if name == "small_extension" else fixtures._ROOM
    order = 1 if name == "small_extension" else 3
    records = tuple(fixtures._record(role, y, room=room, order=order)
                    for role, y in (("left", 1.3), ("right", 2.5)))
    paths, tables, directions = [], [], []
    for record in records:
        assert record.window is not None
        source = record.report.scene.source_m
        inputs = fixtures._input(source.y, room=room, order=order)
        solved = solver_inputs(inputs)
        found = tuple(image_source_paths(solved.room, solved.source, solved.receiver, solved.sound_speed_m_s,
                                        max_order=record.window.computed_order_k))
        paths.append(found)
        tables.append(aj.build_path_table(found, solved.receiver.as_tuple(), fixtures._AXIS))
        directions.append(np.asarray([[(p.image[i] - solved.receiver.as_tuple()[i]) / p.dist_m
                                       for i in range(3)] for p in found], dtype=np.float64))
    z = np.asarray([[complex(fixtures._WALLS[w])] * len(fixtures._AXIS) for w in amp.CANONICAL_WALLS])
    if name == "complex_frequency":
        rng = np.random.default_rng(562)
        z *= rng.uniform(0.7, 1.5, z.shape) + 1j * rng.uniform(0.2, 0.6, z.shape)
    axis = listening_axis(tuple(r.report.scene.source_m.as_tuple() for r in records),
                          records[0].report.scene.receiver_m.as_tuple())
    return Case((records[0], records[1]), tuple(paths), tuple(tables), tuple(directions), axis,
                z, np.asarray(fixtures._SCATTERING), solved.density_kg_m3 * solved.sound_speed_m_s,
                solved.sound_speed_m_s)


def numpy_energy(data: Case, channel: int, z: NDArray[np.complex128], scatter: NDArray[np.float64]) -> NDArray[np.float64]:
    """用既有反射乘積與壓力算式，照 report_path_table 的逐路徑能量定義。"""
    table = data.tables[channel]
    materials = amp.Materials(data.rho_c, tuple(table.frequencies_hz),
                             {w: tuple(z[i]) for i, w in enumerate(amp.CANONICAL_WALLS)})
    receiver = data.records[channel].report.scene.receiver_m.as_tuple()
    pressures = []
    for path in data.paths[channel]:
        refl = amp.reflection_product(materials, path.dist_m, receiver, path.image, wall_count_signature(path.identity))
        pressures.append(amp.path_pressure(materials, path.dist_m, data.sound_speed, refl))
    power = np.abs(np.asarray(pressures)) ** 2
    return np.asarray((1.0 - scatter) ** table.order[:, None] * power / power[table.direct_index()], dtype=np.float64)


def directional_inputs(data: Case, channel: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """真實指向性含偏軸直達；答案走既有 numpy 路徑表，不從 JAX 能量乘倍率推回。"""
    record, table = data.records[channel], data.tables[channel]
    solved = solver_inputs(fixtures._input(record.report.scene.source_m.y))
    model = SourceModelSpec(SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1,
                           parameters=DIRECTIVITY.two_parameter, aim=Point(3.2, 1.9, 2.4))
    assert model.aim is not None and model.parameters is not None
    factors = np.asarray([pressure_factor_for_direction(
        departure_direction(path, solved.receiver), tuple(table.frequencies_hz), SourceModel.TWO_PARAMETER,
        source=solved.source, aim=model.aim, params=model.parameters) for path in data.paths[channel]])
    expected = build_path_table(
        room=solved.room, source=solved.source, receiver=solved.receiver, sound_speed_m_s=data.sound_speed,
        rho_c_pa_s_per_m=data.rho_c, frequencies_hz=tuple(table.frequencies_hz),
        scattering_coefficient=tuple(data.scattering), reflection_order_k=record.report.top.reflection_order_k,
        source_model=model, furniture=None, materials_by_wall=solved.materials_by_wall)
    rows = {(row.order, row.wall_sequence): row.relative_direct_energy for row in expected.rows}
    energy = np.asarray([rows[path.order, tuple(w for b in path.bounces for w in b.walls)]
                         for path in data.paths[channel]])
    return factors, energy


def numpy_score(data: Case, energies: tuple[NDArray[np.float64], ...],
                settings: QualityPurpose | None = None) -> dict[str, NDArray[np.float64]]:
    """把能量送回真的 evaluate_reflections + cost_reflections_evaluation，不重寫計分公式。"""
    records = []
    settings = purpose() if settings is None else settings
    for channel, record in enumerate(data.records):
        table, window = record.report.path_table, record.window
        assert table is not None and window is not None
        rows = {(p.order, p.delay_s, tuple(w for b in p.bounces for w in b.walls)): tuple(energies[channel][i])
                for i, p in enumerate(data.paths[channel])}
        report = record.report.model_copy(update={"path_table": table.model_copy(update={
            "rows": tuple(r.model_copy(update={"relative_direct_energy": rows[r.order, r.delay_s, r.wall_sequence]})
                          for r in table.rows)})})
        window = window.model_copy(update={"rows": tuple(r.model_copy(update={
            "relative_direct_energy": rows[r.order, r.delay_s, r.wall_sequence]}) for r in window.rows)})
        records.append(replace(record, report=report, window=window))
    measured = fixtures._evaluate(tuple(records))
    assert isinstance(measured.payload, ReflectionsAndEchoPayload)
    costed = cost_reflections_evaluation(measured, settings, "test")
    assert costed.category_cost is not None
    channels = tuple(c for c in measured.payload.channels if c.is_primary)
    excess = np.asarray([[_zone_excess(zone, _zone_threshold(settings, zone.zone),
                                     measured.payload.frequency_range_hz) for zone in c.zones] for c in channels])
    costs = np.asarray([[costed.category_cost.components[f"{c.role}.{zone.value}"]
                         for zone in DirectionZone] for c in channels])
    return {"zone_excess_db": excess, "zone_cost": costs, "category_cost": np.asarray(costed.category_cost.value)}


def reference(data: Case, z: NDArray[np.complex128] | None = None,
              scatter: NDArray[np.float64] | None = None) -> dict[str, NDArray[np.float64]]:
    energies = tuple(numpy_energy(data, i, data.z if z is None else z, data.scattering if scatter is None else scatter)
                     for i in range(2))
    return {**numpy_score(data, energies), "path_energy": np.stack(energies)}
