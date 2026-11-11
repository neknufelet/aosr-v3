"""固定面的共用設定：現有材料庫 id，沿用方案整面材料的解析入口；空設定不改舊格式。

材料除了構造預設 id，還收保留字 :data:`SCHEME_MATERIAL`：那一面（或那一區）用方案檔自己寫的材料、不帶變數、
不參與換樣板（網頁的「固定這一組材料」與「保留原牆」都走這條，#1093）。保留字含連字號，構造預設與使用者材料庫的
id 只收小寫英數與底線，永遠撞不到。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Annotated

from pydantic import AfterValidator

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import SCHEME_MATERIAL as SCHEME_MATERIAL, MaterialGradientSettings, Mount
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient.boundary import patch_key
from aosr.gradient.material_gradient import SurfaceDesign, designs_from_presets
from aosr.materials.search_library import PresetMaterial
from aosr.materials.construction_models import SurfaceMaterial
from aosr.physics.report_surfaces import WholeWallZone


def checked_fixed_surfaces(value: Mapping[str, str]) -> dict[str, str]:
    """面名與材料 id 都必須認得（或是保留字 :data:`SCHEME_MATERIAL`），至少留一面可換；排序使保存與身分不受選項順序影響。"""
    names = {wall.wall_name() for wall in Wall.all()}
    for name in value:
        patch_key(name)
        wall, separator, _ = name.partition("@")
        if separator and wall in value:
            raise ValueError(f"同一面不能同時給牆名與面片代號：{wall}")
    if names <= value.keys():
        raise ValueError("不能全部六面固定：至少留一面可換")
    ids = {preset.id for preset in load_construction_presets(config_path("construction_presets.toml"))}
    for face, material in value.items():
        if material != SCHEME_MATERIAL and material not in ids:
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
    """材料庫僅提供填表資料，驗材料與方案原牆相同，交 ``WholeWallZone``；保留字那幾面不在這裡（用方案自己的材料）。"""
    fixed = checked_fixed_surfaces(fixed)
    table = {preset.id: preset for preset in load_construction_presets(config_path("construction_presets.toml"))}
    return {face: WholeWallZone.model_validate({"area": "whole_wall", "material": table[material].material}).material
            for face, material in fixed.items() if material != SCHEME_MATERIAL}


def designs_with_fixed(surfaces: Sequence[tuple[str, str | None]], settings: MaterialGradientSettings,
                       presets: Sequence[PresetMaterial], *, kept: Mapping[str, SurfaceMaterial], mount: Mount,
                       fixed_surfaces: Mapping[str, str]) -> tuple[SurfaceDesign, ...]:
    """固定面強制用無樣板、無變數的常數材料（保留字那幾面就是 ``kept`` 那一面）；其他面的原牆與安裝方式照舊。"""
    fixed_surfaces = expanded_fixed_surfaces(fixed_surfaces, tuple(name for name, _ in surfaces))
    materials = fixed_materials(fixed_surfaces) if fixed_surfaces else {}
    selected = tuple((face, None if face in fixed_surfaces else preset) for face, preset in surfaces)
    designs = designs_from_presets(selected, settings, presets, kept={**kept, **materials}, mount=mount)
    return tuple(replace(design, fixed_material_id=fixed_surfaces.get(design.surface_id)) for design in designs)


def expanded_fixed_surfaces(value: Mapping[str, str], names: Sequence[str]) -> dict[str, str]:
    """牆名展成該牆所有面片；面片名只能命中一區，至少留一片可換。"""
    checked_fixed_surfaces(value)
    result: dict[str, str] = {}
    for face, material in value.items():
        matches = [name for name in names if name == face or ("@" not in face and name.partition("@")[0] == face)]
        if not matches:
            raise ValueError(f"固定面 {face} 不在搜尋面片清單")
        result.update((name, material) for name in matches)
    if len(result) == len(names):
        raise ValueError("不能全部面片固定：至少留一面可換")
    return dict(sorted(result.items()))


def checked_max_depths(value: Mapping[str, float]) -> dict[str, float]:
    """深度上限用公尺、有限且為正；同一牆不能混整面與面片，保存依面名排序。"""
    for name, depth in value.items():
        patch_key(name)
        wall, separator, _ = name.partition("@")
        if separator and wall in value:
            raise ValueError(f"同一面不能同時給牆名與面片代號：{wall}")
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError(f"面 {name} 的深度上限必須是有限正數（公尺），收到 {depth}")
    return dict(sorted(value.items()))


MaxDepths = Annotated[dict[str, float], AfterValidator(checked_max_depths)]


def check_fixed_depth_conflicts(fixed: Mapping[str, str], caps: Mapping[str, float]) -> None:
    """固定整面與該面的任何上限、固定面片與涵蓋它的上限，都拒收。"""
    for face in caps:
        for frozen in fixed:
            if face == frozen or (face.partition("@")[0] == frozen.partition("@")[0]
                                  and ("@" not in face or "@" not in frozen)):
                raise ValueError(f"固定面 {frozen} 不准再給深度上限（{face}）")


def parse_max_depths(values: Sequence[str]) -> dict[str, float]:
    """可重複的 ``--max-depth 面=公尺``，字典化前拒收同面重複。"""
    found: dict[str, float] = {}
    for item in values:
        face, separator, text = item.partition("=")
        if not separator or not face or not text:
            raise ValueError(f"--max-depth 要寫成 面=公尺，收到 {item!r}")
        if face in found:
            raise ValueError(f"--max-depth 同一面 {face} 給了兩次")
        try:
            found[face] = float(text)
        except ValueError as error:
            raise ValueError(f"--max-depth 面 {face} 的公尺值不是數字：{text!r}") from error
    return checked_max_depths(found)


def expanded_max_depths(value: Mapping[str, float], names: Sequence[str]) -> dict[str, float]:
    """整面上限套到那面每一片；面片必須在本次搜尋清單中。"""
    result: dict[str, float] = {}
    for face, depth in checked_max_depths(value).items():
        matches = [name for name in names if name == face or ("@" not in face and name.partition("@")[0] == face)]
        if not matches:
            raise ValueError(f"深度上限面 {face} 不在搜尋面片清單")
        result.update((name, depth) for name in matches)
    return dict(sorted(result.items()))


def designs_with_max_depths(designs: Sequence[SurfaceDesign], settings: MaterialGradientSettings,
                            value: Mapping[str, float], *, check_family_upper: bool = True) -> tuple[SurfaceDesign, ...]:
    """套入每面生效上限；不貼的面也保留選項，後續換家族繼續套用。"""
    caps = expanded_max_depths(value, tuple(d.surface_id for d in designs))
    result = []
    for design in designs:
        cap = caps.get(design.surface_id)
        if cap is None:
            result.append(design)
            continue
        if design.fixed_material_id is not None:
            raise ValueError(f"固定面 {design.surface_id} 不准再給深度上限")
        families = (settings.family_by_id(design.family_id),) if design.family_id else settings.family
        for family in families:
            if check_family_upper and cap > family.max_total_depth_m:
                raise ValueError(f"面 {design.surface_id} 的深度上限 {cap} m 超過家族 {family.id} 上限 {family.max_total_depth_m} m")
            if cap < family.thicknesses_m[0]:
                raise ValueError(f"面 {design.surface_id} 的深度上限 {cap} m 放不下家族 {family.id} 最薄棉 {family.thicknesses_m[0]} m")
        effective = None if design.max_total_depth_m is None else min(design.max_total_depth_m, cap)
        result.append(replace(design, max_depth_m=cap, max_total_depth_m=effective))
    return tuple(result)
