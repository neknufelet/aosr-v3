"""開跑那一組起跑配置的原料：方案原本每一面的材料、產品構造預設清單、每一面的起跑設計。

從 ``run`` 原樣搬出來（#1206 第 2 支修補；``run`` 照原名接出去）：命令列開跑、建替身與網頁行程核左右對稱的綁定
（``cli.start_designs``）建的是同一組起跑配置，而網頁行程不准載入 JAX。所以這一檔**不准載入 JAX**，
也不准載入會載入 JAX 的模組。只讀方案、設定與材料資料，不算數。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from aosr.config.capabilities import CapabilityTable
from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.material_gradient import MOUNT_RIGID, MaterialGradientSettings, Mount
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient.boundary import BoundaryKey, material_keys, named_materials, scheme_patch_keys, surface_ids
from aosr.gradient.fixed_surfaces import designs_with_fixed, designs_with_max_depths
from aosr.gradient.surface_design import SurfaceDesign, wall_designs_from_presets
from aosr.materials.construction_models import SurfaceMaterial
from aosr.materials.search_library import PresetMaterial
from aosr.physics.fem_reduced_boundary import boundary_patches
from aosr.physics.report_io import ReportInput, solver_inputs
from aosr.physics.report_surfaces import RectangleZone
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import checked_inputs


def presets_table() -> tuple[ConstructionPreset, ...]:
    """產品的構造預設清單（``construction_presets.toml``）。"""
    return load_construction_presets(config_path("construction_presets.toml"))


def scene_wall_materials(scheme: Scheme, *, capabilities: CapabilityTable,
                         directivity: DirectivityDefaults) -> dict[BoundaryKey, SurfaceMaterial]:
    """沿 ``export_operators`` 的 checked_inputs 第一對輸入取材料；分區直接取每片的 material。

    整面仍交 solver_inputs 取 materials_by_wall，不另轉固定阻抗；分區鍵與序由 boundary_patches 共用。
    """
    _, documents = checked_inputs(scheme, capabilities=capabilities, directivity=directivity)
    return _wall_materials_from_inputs(scheme, next(iter(documents.values()))[1])


def _wall_materials_from_inputs(scheme: Scheme, inputs: ReportInput) -> dict[BoundaryKey, SurfaceMaterial]:
    """共用已驗過的逐對輸入；材料與面片序維持 scene_wall_materials 的讀法。"""
    keys = scheme_patch_keys(scheme)
    if keys != boundary_patches():
        zones = inputs.surface_zones_by_wall or {}
        return {(wall, zone): next(z.material for z in zones[wall.wall_name()]
                                  if (isinstance(z, RectangleZone) and z.id == zone) or
                                  (not isinstance(z, RectangleZone) and not zone)) for wall, zone in keys}
    return cast(dict[BoundaryKey, SurfaceMaterial], dict(solver_inputs(inputs).materials_by_wall))


def _uncapped_designs_for[K: BoundaryKey](presets: Sequence[str | None], settings: MaterialGradientSettings, *,
                kept: Mapping[K, SurfaceMaterial], mount: Mount = MOUNT_RIGID,
                fixed_surfaces: Mapping[str, str] | None = None,
                table: Sequence[PresetMaterial] | None = None) -> tuple[SurfaceDesign, ...]:
    """有分區照共用面片序、整面照 ``Wall.all()`` 的設計；preset 不認識或不在家族清單裡就拒收。preset 是 None 的面一律帶 ``kept``
    那一面的材料（方案原本的牆，``scene_wall_materials``），是「不貼、保留原牆」，不是剛性；``mount`` 是貼在原牆上時，
    貼樣板的面背後接 ``kept`` 那一面。``table`` 是構造清單（不給是產品構造預設；有材料快照的搜尋給 :func:`project_presets`）。"""
    table = presets_table() if table is None else tuple(table)
    keys = material_keys(kept)
    if keys != boundary_patches():
        names = surface_ids(keys)
        if len(presets) != len(names):
            raise ValueError(f"面片清單要給 {len(names)} 個 preset id")
        return designs_with_fixed(tuple(zip(names, presets, strict=True)), settings, table,
                                  kept=named_materials(kept), mount=mount, fixed_surfaces=fixed_surfaces or {})
    whole_kept = cast(Mapping[Wall, SurfaceMaterial], kept)
    if fixed_surfaces:
        if len(presets) != len(Wall.all()):
            raise ValueError("六面牆要給六個 preset id")
        return designs_with_fixed(tuple(zip((wall.wall_name() for wall in Wall.all()), presets, strict=True)),
                                  settings, table, kept={w.wall_name(): m for w, m in whole_kept.items()},
                                  mount=mount, fixed_surfaces=fixed_surfaces)
    return wall_designs_from_presets(presets, settings, table,
                                     kept={wall.wall_name(): material for wall, material in whole_kept.items()}, mount=mount)


def designs_for[K: BoundaryKey](presets: Sequence[str | None], settings: MaterialGradientSettings, *,
                kept: Mapping[K, SurfaceMaterial], mount: Mount = MOUNT_RIGID,
                fixed_surfaces: Mapping[str, str] | None = None,
                max_depths: Mapping[str, float] | None = None,
                table: Sequence[PresetMaterial] | None = None) -> tuple[SurfaceDesign, ...]:
    """建立起跑設計，套入每面上限（跟起跑家族自己的上限取較嚴；起跑家族最淺排法放不進就拒收，#1082 段 2-4）；
    空設定沿用舊設計。``table`` 同 :func:`_uncapped_designs_for`。"""
    designs = _uncapped_designs_for(presets, settings, kept=kept, mount=mount, fixed_surfaces=fixed_surfaces, table=table)
    return designs_with_max_depths(designs, settings, max_depths or {})
