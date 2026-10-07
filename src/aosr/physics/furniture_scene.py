"""家具接入物理時共用的場景、逐頻材質與含指向性的路徑聲壓零件。

本步尚未接任何物理入口，也不供模態閉包匯入。介質、聲速與接觸契約由
呼叫端明給。材質是估計值，未知頻帶保留逐點旗標；指向性向下方向尚未
獨立驗證（家具決策紙第 27 條），桌面反射強度依賴此假設。
"""
from __future__ import annotations

from collections.abc import Sequence

from aosr.config.directivity_defaults import DirectivityDefaults, TwoParameterCurve
from aosr.config.furniture_materials import FurnitureMaterialName, load_furniture_materials
from aosr.config.paths import config_path
from aosr.geometry.furniture import FurnitureBox, Vec3, contact_margin_m
from aosr.geometry.shoebox import Point
from aosr.materials.furniture_materials import FurnitureImpedanceOnAxis, furniture_impedance_on_axis
from aosr.physics.furniture_paths import Furniture, FurniturePath, furniture_path_amplitude
from aosr.physics.report_furniture import AbsoluteFurniture, normalized_furniture
from aosr.physics.source_directivity import (
    SourceModel, TwoParameterValues, pressure_factor_for_direction, unit_vector,
)


def furniture_scene(items: Sequence[AbsoluteFurniture], room_size_m: Vec3, *,
                    contact_rel: float) -> tuple[tuple[Furniture, ...], float]:
    """絕對家具按代號換成盒子，回 (家具清單, 公尺接觸界線)；盒子錯句原樣往上丟。"""
    margin_m = contact_margin_m(room_size_m, contact_rel=contact_rel)
    ordered = normalized_furniture(tuple(items)) or ()
    furniture = tuple(Furniture(item.furniture_id, FurnitureBox(
        kind=item.kind, width_m=item.width_m, depth_m=item.depth_m, height_m=item.height_m,
        bottom_center_m=item.bottom_center_m, yaw_deg=item.yaw_deg, margin_m=margin_m,
    )) for item in ordered)
    return furniture, margin_m


def furniture_impedances(items: Sequence[AbsoluteFurniture], frequencies_hz: Sequence[float],
                         rho_c: float) -> dict[str, FurnitureImpedanceOnAxis]:
    """按家具代號回逐頻實數阻抗與未知旗標；預設曲線使用該件種類。

    快取只住在這次呼叫內，按材質共用同一個不可變結果；軸與 ρc 在一次
    呼叫裡固定。另一條軸或另一個介質的呼叫重新計算，不跨候選藏狀態。
    每件仍先核適用種類，不能因為材質已快取而借用別種家具的曲線。
    config_path 的字面值讓本模組接進物理閉包後，資料檔自動納入物理身分。
    """
    ordered = normalized_furniture(tuple(items)) or ()
    materials = dict(load_furniture_materials(config_path("furniture_materials.toml")).materials())
    by_material: dict[FurnitureMaterialName, FurnitureImpedanceOnAxis] = {}
    by_id: dict[str, FurnitureImpedanceOnAxis] = {}
    for item in ordered:
        material = materials[item.material]
        material.for_kind(item.kind.value)
        if item.material not in by_material:
            by_material[item.material] = furniture_impedance_on_axis(
                material, frequencies_hz, rho_c, curve="default", kind=item.kind.value)
        by_id[item.furniture_id] = by_material[item.material]
    return by_id


def furniture_pressure_with_directivity(
    path: FurniturePath, frequencies_hz: tuple[float, ...], impedance_per_freq: tuple[float, ...],
    *, rho_c: float, c: float, model: SourceModel,
    source: Point | None = None, aim: Point | None = None,
    params: DirectivityDefaults | TwoParameterValues | TwoParameterCurve | None = None,
    baffle_width_m: float | None = None, piston_radius_m: float | None = None,
) -> tuple[complex, ...]:
    """家具路徑聲壓乘 D；反射角、√(K₁K₂) 與兩段距離相位沿用家具振幅零件。

    不乘整房散射。全向直接回原聲壓，不做乘一；非全向沿聲源→反射點的
    departure_direction 算 D。上一代相容模型使用同一個呼叫端聲速 c。
    """
    _, pressure = furniture_path_amplitude(path, frequencies_hz, impedance_per_freq, rho_c=rho_c, c=c)
    if model == SourceModel.OMNIDIRECTIONAL:
        return pressure
    factors = pressure_factor_for_direction(
        unit_vector(path.departure_direction), frequencies_hz, model, source=source, aim=aim,
        params=params, baffle_width_m=baffle_width_m, piston_radius_m=piston_radius_m, sound_speed_m_s=c)
    return tuple(p * float(d) for p, d in zip(pressure, factors, strict=True))
