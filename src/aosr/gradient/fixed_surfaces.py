"""固定面的共用設定：現有材料庫 id，沿用方案整面材料的解析入口；空設定不改舊格式。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Annotated

from pydantic import AfterValidator

from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.material_gradient import MaterialGradientSettings, Mount
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient.material_gradient import SurfaceDesign, designs_from_presets
from aosr.materials.construction_models import SurfaceMaterial
from aosr.physics.report_surfaces import WholeWallZone


def checked_fixed_surfaces(value: Mapping[str, str]) -> dict[str, str]:
    """面名與材料 id 都必須認得，至少留一面可換；排序使保存與身分不受選項順序影響。"""
    names = {wall.wall_name() for wall in Wall.all()}
    unknown = value.keys() - names
    if unknown:
        raise ValueError(f"固定面的面名不認識：{sorted(unknown)}；只收 {sorted(names)}")
    if len(value) == len(names):
        raise ValueError("不能全部六面固定：至少留一面可換")
    ids = {preset.id for preset in load_construction_presets(config_path("construction_presets.toml"))}
    for face, material in value.items():
        if material not in ids:
            raise ValueError(f"固定面 {face} 的材料 id {material!r} 不在現有材料庫")
    return dict(sorted(value.items()))


FixedSurfaces = Annotated[dict[str, str], AfterValidator(checked_fixed_surfaces)]


def parse_fixed_surfaces(values: Sequence[str]) -> dict[str, str]:
    """可重複的 ``--fixed-surface 面=材料``；字典化前先拒收同面重複。"""
    found: dict[str, str] = {}
    for item in values:
        face, separator, material = item.partition("=")
        if not separator or not face or not material:
            raise ValueError(f"--fixed-surface 要寫成 面=材料 id，收到 {item!r}")
        if face in found:
            raise ValueError(f"--fixed-surface 同一面 {face} 給了兩次")
        found[face] = material
    return checked_fixed_surfaces(found)


def fixed_materials(fixed: Mapping[str, str]) -> dict[str, SurfaceMaterial]:
    """材料庫僅提供填表資料，驗材料與方案原牆相同，交 ``WholeWallZone``。"""
    fixed = checked_fixed_surfaces(fixed)
    table = {preset.id: preset for preset in load_construction_presets(config_path("construction_presets.toml"))}
    return {face: WholeWallZone.model_validate({"area": "whole_wall", "material": table[material].material}).material
            for face, material in fixed.items()}


def designs_with_fixed(surfaces: Sequence[tuple[str, str | None]], settings: MaterialGradientSettings,
                       presets: Sequence[ConstructionPreset], *, kept: Mapping[str, SurfaceMaterial], mount: Mount,
                       fixed_surfaces: Mapping[str, str]) -> tuple[SurfaceDesign, ...]:
    """固定面強制用無樣板、無變數的常數材料；其他面的原牆與安裝方式照舊。"""
    materials = fixed_materials(fixed_surfaces) if fixed_surfaces else {}
    selected = tuple((face, None if face in fixed_surfaces else preset) for face, preset in surfaces)
    designs = designs_from_presets(selected, settings, presets, kept={**kept, **materials}, mount=mount)
    return tuple(replace(design, fixed_material_id=fixed_surfaces.get(design.surface_id)) for design in designs)
