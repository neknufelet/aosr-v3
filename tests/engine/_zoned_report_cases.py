"""#562 小房正式方案（整面與分區）。"""
from __future__ import annotations

from aosr.materials.construction_models import Assembly, ConstantImpedance
from aosr.physics.report_surfaces import RectangleZone, WholeWallZone, ZonesByWall
from aosr.reporting.scheme import Scheme
from tests.engine._furniture_cases import reference_document
from tests.engine._surface_zone_cases import ASSEMBLY


def small_scheme(*, zoned: bool = False, changed: bool = False, assembly: bool = False) -> Scheme:
    """沿用七座位參考方案，所有幾何縮半；x0 切成面積 1:2 的兩區。"""
    original = Scheme.model_validate(reference_document())
    room = original.scene.room_m
    scene = original.scene.model_copy(update={"room_m": type(room)(room.Lx/2, room.Ly/2, room.Lz/2)})
    speakers = {name: type(point)(point.x/2, point.y/2, point.z/2) for name, point in original.speakers.items()}
    receivers = original.receiver_set.model_copy(update={"points": tuple(point.model_copy(update={
        "position_m": tuple(value/2 for value in point.position_m)}) for point in original.receiver_set.points)})
    if zoned:
        zones: ZonesByWall = {name: (WholeWallZone(area="whole_wall", material=ConstantImpedance(impedance_pa_s_m=value)),)
                 for name, value in (scene.impedance_pa_s_per_m_by_wall or {}).items()}
        material = zones["x0"][0].material
        zones["x0"] = tuple(RectangleZone(area="rectangle", id=name, u_m=u, v_m=(0.0, 1.5),
            material=Assembly.model_validate(ASSEMBLY) if assembly and name == "b" else
            material.model_copy(update={"impedance_pa_s_m": 950.0}) if changed and name == "a" else material)
            for name, u in (("a", (0.0, 2.0/3.0)), ("b", (2.0/3.0, 2.0))))
        scene = scene.model_copy(update={"surface_zones_by_wall": zones, "impedance_pa_s_per_m_by_wall": None})
    return Scheme.model_validate(original.model_copy(update={"scene": scene, "speakers": speakers,
                                                            "receiver_set": receivers}).model_dump())

