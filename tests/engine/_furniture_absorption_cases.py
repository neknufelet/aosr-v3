"""家具整件吸音量的共用輸入；尺寸抄自輸入頁預設（考卷自己的值，預設改了這裡不跟著動），位置只供場景綁定。"""
from typing import TypedDict
from aosr.geometry.furniture import FurnitureKind
from aosr.geometry.shoebox import Room, Wall
from aosr.materials.construction_models import SurfaceMaterial
from aosr.physics.report_furniture import AbsoluteFurniture
from tests.engine import _furniture_energy_cases as case
from tests.engine._material_flow_cases import fixed_materials

ITEMS = (
    AbsoluteFurniture(furniture_id="sofa", kind=FurnitureKind.SOFA, material="fabric",
        width_m=1.80, depth_m=0.75, height_m=0.65, bottom_center_m=(3.0, 3.0, 0.0), yaw_deg=0.0),
    AbsoluteFurniture(furniture_id="table", kind=FurnitureKind.COFFEE_TABLE, material="wood",
        width_m=1.20, depth_m=0.60, height_m=0.03, bottom_center_m=(3.0, 2.0, 0.47), yaw_deg=0.0),
    AbsoluteFurniture(furniture_id="cloud", kind=FurnitureKind.CEILING_CLOUD, material="absorptive_cloud",
        width_m=1.35, depth_m=1.80, height_m=0.05, bottom_center_m=(3.0, 2.0, 2.5), yaw_deg=0.0),
)


class LateKwargs(TypedDict):
    room: Room
    materials_by_wall: dict[Wall, SurfaceMaterial]
    density_kg_m3: float
    sound_speed_m_s: float
    report_frequencies_hz: tuple[float, ...]
    reflection_order_k: int


def late_kwargs() -> LateKwargs:
    return dict(room=case.ROOM, materials_by_wall=fixed_materials(case.WALLS),
        density_kg_m3=case.DENSITY, sound_speed_m_s=case.SPEED,
        report_frequencies_hz=case.FREQUENCIES, reflection_order_k=1)
