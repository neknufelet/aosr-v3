"""#562 早期 JAX（可微陣列算數）考卷的自足小房與兩種分區材料。"""
from __future__ import annotations

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.construction_models import AirLayer, Assembly, ConstantImpedance, PorousLayer, SurfaceMaterial
from aosr.physics.report_surfaces import RectangleZone
from aosr.physics.report_source import SourceModelSpec
from aosr.physics.wall_reflections import WallReflectionSources, wall_reflection_sources
from aosr.physics.wall_surfaces import wall_path_impedances
import numpy as np
from numpy.typing import NDArray
from tests.engine import _furniture_energy_cases as furniture_case

ROOM = Room(6., 4., 3.)
SOURCES = (Point(1.1, .9, 1.2), Point(1.4, 2.9, 1.1))
RECEIVERS = (Point(3.7, 2.1, 1.45), Point(4.3, 1.4, 1.3))
AXIS = GEOMETRIC_LANE_FREQUENCIES_HZ
RHO, SPEED = 1.19, 347.
CONSTRUCTION = Assembly(layers=(PorousLayer(thickness_m=.05, flow_resistivity_pa_s_m2=12000.,
    material_class="fibrous"), AirLayer(thickness_m=.17)), backing="rigid")


def reflection_sources(name: str, axis: tuple[float, ...] = AXIS) -> WallReflectionSources:
    """whole：固定整面；construction：構造整面；zoned：固定與構造各一區。"""
    walls: dict[Wall, SurfaceMaterial] = {w: ConstantImpedance(impedance_pa_s_m=900.+173.*i)
                                         for i, w in enumerate(Wall.all())}
    zones = None
    if name == "construction":
        walls[Wall.X0] = CONSTRUCTION
    if name == "zoned":
        del walls[Wall.X0]
        zones = {Wall.X0: (
            RectangleZone(area="rectangle", id="low", u_m=(0., 1.5), v_m=(0., 3.),
                material=ConstantImpedance(impedance_pa_s_m=800.)),
            RectangleZone(area="rectangle", id="high", u_m=(1.5, 4.), v_m=(0., 3.), material=CONSTRUCTION))}
    return wall_reflection_sources(walls, axis, density_kg_m3=RHO, sound_speed_m_s=SPEED,
        rectangle_zones_by_wall=zones, room=ROOM if zones else None)


def scattering() -> dict[str, float]:
    return {w.wall_name(): .08+.07*i for i, w in enumerate(Wall.all())}


def models() -> tuple[SourceModelSpec, ...]:
    return (furniture_case.OMNI, furniture_case.analytic())


def patch_rows(sources: WallReflectionSources, patch_ids: tuple[str, ...],
               cosines: tuple[NDArray[np.float64], ...]) -> tuple[tuple[NDArray[np.complex128], ...], NDArray[np.complex128]]:
    """真的 numpy 參考阻抗：沿用逐角求值及垂直入射求值，沒有內插。"""
    materials = {w.wall_name(): m for w, m in sources.materials_by_wall.items()}
    materials.update({w.wall_name()+"@"+z.id: z.material
        for w, zones in (sources.rectangle_zones_by_wall or {}).items() for z in zones})
    rows, normals = [], []
    for name, cos in zip(patch_ids, cosines, strict=True):
        keys = tuple(float(c) for c in cos)
        values = wall_path_impedances(materials[name], sources.fixed_materials.frequencies_hz, keys or (1.,),
            wall_name=name.partition("@")[0], density_kg_m3=sources.density_kg_m3,
            sound_speed_m_s=sources.sound_speed_m_s)
        rows.append(np.stack([values[c] for c in keys], axis=1) if keys else
            np.empty((len(sources.fixed_materials.frequencies_hz), 0), dtype=np.complex128))
        normal = wall_path_impedances(materials[name], sources.fixed_materials.frequencies_hz, (1.,),
            wall_name=name.partition("@")[0], density_kg_m3=sources.density_kg_m3,
            sound_speed_m_s=sources.sound_speed_m_s)
        normals.append(normal[1.])
    return tuple(rows), np.asarray(normals, dtype=np.complex128)
