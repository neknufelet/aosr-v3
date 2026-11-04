"""#561 雙精度子行程：起點快照與分區構造的獨立純量對照。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

import numpy as np

from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, SurfaceMaterial
from aosr.physics import report_path_table, wall_reflections
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.report_surfaces import RectangleZone, WholeWallZone, ZonesByWall
from aosr.physics.room_paths import image_source_paths
from aosr.runtime import configure_calculation_jax

ROOM: Room = Room(6., 4., 3.)
SOURCE: Point = Point(1.1, .9, 1.2)
RECEIVER: Point = Point(3.7, 2.1, 1.45)
AXIS: tuple[float, ...] = (125., 503.)
MODEL: SourceModelSpec = SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL)


def materials(construction: bool) -> dict[Wall, SurfaceMaterial]:
    """固定與構造控制組共用的明確題目。"""
    result: dict[Wall, SurfaceMaterial] = {
        wall: ConstantImpedance(impedance_pa_s_m=900. + 173. * i) for i, wall in enumerate(Wall.all())}
    if construction:
        result[Wall.X0] = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),
            AirLayer(thickness_m=.43)), backing="rigid")
    return result


def snapshot(construction: bool, zoned: bool = False) -> tuple[str, str]:
    """反射乘積與壓力保留 IEEE 位元；路徑表序列化包含每一列全部欄位。"""
    walls = materials(construction)
    zones: ZonesByWall = {w.wall_name(): (WholeWallZone(area="whole_wall", material=m),) for w, m in walls.items()}
    sources = (wall_reflections.zoned_reflection_sources(ROOM, zones, AXIS, density_kg_m3=1.19, sound_speed_m_s=347.)
        if zoned else wall_reflections.wall_reflection_sources(walls, AXIS, density_kg_m3=1.19, sound_speed_m_s=347.))
    paths = image_source_paths(ROOM, SOURCE, RECEIVER, 347., 3, reflection_sources=sources)
    bits = b"".join(np.asarray((p.reflection_product, p.path_pressure), dtype=np.complex128).tobytes() for p in paths)
    table = report_path_table.build_path_table(room=ROOM, source=SOURCE, receiver=RECEIVER,
        sound_speed_m_s=347., rho_c_pa_s_per_m=1.19*347., density_kg_m3=1.19,
        frequencies_hz=AXIS, scattering_coefficient=(.21, .21), reflection_order_k=3,
        source_model=MODEL, furniture=None, materials_by_wall={} if zoned else walls,
        surface_zones_by_wall=zones if zoned else None)
    return hashlib.sha256(bits).hexdigest(), hashlib.sha256(
        json.dumps(asdict(table), sort_keys=True).encode()).hexdigest()


def check_partitioned_construction() -> None:
    """同面固定與構造四區，獨立按幾何點選材、核心直接求角，再用純量次方對照。"""
    import jax.numpy as jnp
    from aosr.config.construction_environment import construction_environment
    from aosr.materials.construction import surface_impedance_at_angles
    from aosr.physics import amplitude as amp
    import math

    walls = materials(True)
    zones: ZonesByWall = {w.wall_name(): (WholeWallZone(area="whole_wall", material=m),) for w, m in walls.items()}
    zones["x0"] = tuple(RectangleZone(area="rectangle", id=f"r{row}c{col}", u_m=u, v_m=v,
        material=walls[Wall.X0] if (row, col) == (1, 1) else ConstantImpedance(impedance_pa_s_m=800.+600.*(row*2+col)))
        for row, v in enumerate(((0., 1.5), (1.5, 3.))) for col, u in enumerate(((0., 1.5), (1.5, 4.))))
    sources = wall_reflections.zoned_reflection_sources(ROOM, zones, AXIS, density_kg_m3=1.19, sound_speed_m_s=347.)
    paths = image_source_paths(ROOM, SOURCE, RECEIVER, 347., 3, reflection_sources=sources)
    cosines = tuple(dict.fromkeys(amp.incidence_cos(0, p.dist_m, RECEIVER.as_tuple(), p.image) for p in paths
        if any("x0" in b.walls and b.point[1] >= 1.5 and b.point[2] >= 1.5 for b in p.bounces)))
    air, ranges = construction_environment(density_kg_m3=1.19, sound_speed_m_s=347.)
    z, _notes = surface_impedance_at_angles(walls[Wall.X0], jnp.asarray(AXIS),
        jnp.asarray(tuple(math.degrees(math.acos(c)) for c in cosines)), air=air, ranges=ranges)
    rows = dict(zip(cosines, np.asarray(z, dtype=np.complex128).T, strict=True))
    for p in paths:
        expected = []
        for band in range(len(AXIS)):
            product = 1.+0j
            for wall in Wall.all():
                name = wall.wall_name()
                cos = amp.incidence_cos(wall.axis(), p.dist_m, RECEIVER.as_tuple(), p.image)
                points = [b.point for b in p.bounces if name in b.walls]
                if wall is Wall.X0:
                    for index in range(4):
                        count = sum(2*int(point[2] >= 1.5)+int(point[1] >= 1.5) == index for point in points)
                        if count:
                            impedance = complex(rows[cos][band]) if index == 3 else complex(800.+600.*index)
                            product *= amp.reflection_coefficient(impedance, cos, 1.19*347.) ** count
                elif points:
                    material = walls[wall]
                    assert isinstance(material, ConstantImpedance)
                    product *= amp.reflection_coefficient(complex(material.impedance_pa_s_m), cos, 1.19*347.) ** len(points)
            expected.append(product)
        assert np.asarray(p.reflection_product).tobytes() == np.asarray(expected).tobytes(), p.identity


def forbidden_zone_route(*args: object, **kwargs: object) -> None:
    raise AssertionError("整面一區不得進入逐區反射路徑")


if __name__ == "__main__":
    configure_calculation_jax()
    mode, directory = sys.argv[1], Path(sys.argv[2])
    if mode == "legacy":
        from tests.engine.test_zone_reflections import _legacy_comparison
        (directory / "zone-result.json").write_text(json.dumps(_legacy_comparison(directory)))
    elif mode == "checks":
        check_partitioned_construction()
        with patch.object(wall_reflections.WallReflectionSources, "_apply_zones", forbidden_zone_route):
            result = {name: snapshot(value, zoned=True) for name, value in (("fixed", False), ("construction", True))}
        (directory / "zone-result.json").write_text(json.dumps(result))
    else:
        result = {name: snapshot(value) for name, value in (("fixed", False), ("construction", True))}
        (directory / "zone-result.json").write_text(json.dumps(result))
