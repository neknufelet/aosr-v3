"""一個面的設計（:class:`SurfaceDesign`）與「從 preset 組設計」那幾支：開跑那一組起跑配置的原料。

從 ``material_gradient`` 原樣搬出來（#1206 第 2 支修補；``material_gradient`` 照原名接出去）：這幾支只讀設定與材料資料、
不算數，網頁行程核左右對稱的綁定（``aosr.gui.gradient_plan``）要建同一組起跑配置，而網頁行程不准載入 JAX
（``test_web_process_never_imports_jax``）。所以這一檔**不准載入 JAX**，也不准載入會載入 JAX 的模組。

設計變數只有每面棉板厚度（多孔層）與空腔深度（空氣層）；變數名是樣板的 ``layer<層號>.<欄位>``
（:func:`aosr.materials.variable_names.variable_name`）。純函式，沒有命令列、沒有行程、不印東西。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import TypeAdapter

from aosr.config.material_gradient import (
    MOUNT_ON_WALL, MOUNT_RIGID, MOUNTS, MaterialGradientSettings, Mount, is_fixed_target,
)
from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import (
    AirLayer, Assembly, ConstantImpedance, MeasuredImpedance, MountedOnWall, PorousLayer, SurfaceMaterial,
)
from aosr.materials.variable_names import variable_name
from aosr.materials.search_library import PresetMaterial

_THICKNESS = "thickness_m"
# 貼在原牆上時原牆收的型別（``MountedOnWall.wall``）：不收本身已是貼牆構造的材料（只有一層巢狀）。
# 用 type 陳述式：只是型別別名、不在模組層建材料（材料只有一個出生地，``test_material_flow``）。
type WallMaterial = Assembly | ConstantImpedance | MeasuredImpedance


@dataclass(frozen=True)
class SurfaceDesign:
    """一個面的設計：樣板（沒有樣板是 None）、哪些樣板變數是設計變數、各自上下界、合法厚度與總深度上限、
    保留的原牆材料。

    ``variables`` 照「厚度、空腔」的順序，名字是樣板的 ``layer<層號>.<欄位>``；
    厚度界是合法厚度的最薄到最厚，空腔界是家族的空腔範圍（公尺）。
    ``material`` 是 None 時：``kept_material`` 有值就是「不貼、保留方案原本的牆」（整條鏈用這份材料的垂直入射阻抗，
    常數、沒有變數），兩個都 None 才是剛性面。有樣板的面不收 ``kept_material``。``preset_id`` 是 ``fixed:<代號>``
    （#1082 段 2-3b，換樣板整塊換成准用清單裡的固定材料）時也沒有樣板：``kept_material`` 就是那份固定材料（整塊、常數）。
    ``mount`` 是這一次搜尋的安裝方式（#990）；``backing_material`` 是貼在原牆上時那一面原本的牆，只在 ``on_wall``
    而且有樣板時必填，其他情況拒收。``material`` 仍是 preset 那份構造（終點照舊逐欄核身分）。
    """

    surface_id: str
    preset_id: str | None
    family_id: str | None
    material: Assembly | None
    variables: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    thickness_variable: str | None
    cavity_variable: str | None
    legal_thicknesses_m: tuple[float, ...]
    max_total_depth_m: float | None
    kept_material: SurfaceMaterial | None = None
    mount: Mount = MOUNT_RIGID
    backing_material: WallMaterial | None = None
    # 固定面保留材料庫 id（跟保留原牆分開記身分），材料本身仍走 kept_material。
    fixed_material_id: str | None = None
    # 使用者的面上限：不貼也保留，帳本與換家族不能遺失它；None 不改既有存檔。
    max_depth_m: float | None = None

    def __post_init__(self) -> None:
        if not self.surface_id.strip():
            raise ValueError("面代號不可空白")
        if self.mount not in MOUNTS:
            raise ValueError(f"面 {self.surface_id} 的安裝方式只收 {list(MOUNTS)}，收到 {self.mount!r}")
        if self.backing_material is not None and (self.material is None or self.mount != MOUNT_ON_WALL):
            raise ValueError(f"面 {self.surface_id} 只有貼在原牆上而且有樣板時才收原牆")
        if self.fixed_material_id is not None and (self.material is not None or self.kept_material is None):
            raise ValueError(f"固定面 {self.surface_id} 要用無樣板的保留材料")
        if self.material is None:
            if is_fixed_target(self.preset_id) and (self.kept_material is None or self.fixed_material_id is not None):
                raise ValueError(f"面 {self.surface_id} 換成固定材料 {self.preset_id}，要帶那份材料、而且不是固定面")
            if ((self.preset_id is not None and not is_fixed_target(self.preset_id)) or self.family_id is not None
                    or self.variables or self.lower
                    or self.upper or self.thickness_variable is not None or self.cavity_variable is not None
                    or self.legal_thicknesses_m or self.max_total_depth_m is not None):
                raise ValueError(f"沒有樣板的面 {self.surface_id} 不收 preset、變數、界或規格")
            return
        if self.kept_material is not None:
            raise ValueError(f"面 {self.surface_id} 有樣板構造，不收保留原牆的材料")
        if self.mount == MOUNT_ON_WALL and not isinstance(self.backing_material, (Assembly, ConstantImpedance, MeasuredImpedance)):
            raise ValueError(f"面 {self.surface_id} 貼在原牆上，要給那一面原本的牆（固定阻抗、實測阻抗表或多層構造）")
        if self.mount == MOUNT_ON_WALL and self.material.backing != "rigid":
            raise ValueError(f"面 {self.surface_id} 的樣板背後不是剛背（{self.material.backing}），不能改成貼在原牆上")
        kinds = _variable_kinds(self.material)
        unknown = [name for name in self.variables if name not in kinds]
        if unknown:
            raise ValueError(f"設計變數名樣板沒有：{unknown}；這個樣板可當設計變數的是 {sorted(kinds)}")
        if self.thickness_variable is None or kinds.get(self.thickness_variable) is not PorousLayer:
            raise ValueError(f"面 {self.surface_id} 的厚度變數必須是樣板裡多孔層的厚度")
        if self.cavity_variable is not None and kinds.get(self.cavity_variable) is not AirLayer:
            raise ValueError(f"面 {self.surface_id} 的空腔變數必須是樣板裡空氣層的厚度")
        expected = tuple(name for name in (self.thickness_variable, self.cavity_variable) if name is not None)
        if self.variables != expected:
            raise ValueError(f"面 {self.surface_id} 的設計變數必須照厚度、空腔排：{list(expected)}")
        if len(self.lower) != len(self.variables) or len(self.upper) != len(self.variables):
            raise ValueError(f"面 {self.surface_id} 的上下界數目要等於設計變數數")
        if any(not np.isfinite(low) or not np.isfinite(high) or low > high
               for low, high in zip(self.lower, self.upper, strict=True)):
            raise ValueError(f"面 {self.surface_id} 的上下界必須有限且下界不大於上界")
        legal = self.legal_thicknesses_m
        if not legal or list(legal) != sorted(set(legal)) or self.max_total_depth_m is None:
            raise ValueError(f"面 {self.surface_id} 要有由薄到厚的合法厚度與總深度上限")
        if self.lower[0] != legal[0] or self.upper[0] != legal[-1]:
            raise ValueError(f"面 {self.surface_id} 的厚度界必須是合法厚度的最薄到最厚")


# ---- 一串面：從 preset 組設計 ----

def _variable_kinds(material: Assembly) -> dict[str, type]:
    """這個樣板可當設計變數的名字 → 那一層的型別（只有多孔層與空氣層的厚度）。"""
    return {variable_name(index, _THICKNESS): type(layer)
            for index, layer in enumerate(material.layers, start=1) if isinstance(layer, PorousLayer | AirLayer)}


def wall_surface_ids() -> tuple[str, ...]:
    """替身那六面的面代號，照 ``Wall.all()`` 的順序；給今天只有六面的呼叫端用。"""
    return tuple(wall.wall_name() for wall in Wall.all())


def _surface_design(surface_id: str, preset_id: str, settings: MaterialGradientSettings,
                    presets: Mapping[str, PresetMaterial], *, mount: Mount = MOUNT_RIGID,
                    kept: Mapping[str, SurfaceMaterial] | None = None) -> SurfaceDesign:
    preset = presets.get(preset_id)
    if preset is None:
        raise ValueError(f"preset id 不認識：{preset_id}")
    family = settings.family_of(preset_id)
    if family is None:
        raise ValueError(f"preset {preset_id} 不在家族清單裡（設定檔 [[family]]）")
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    material = adapter.validate_python(preset.material)
    if not isinstance(material, Assembly):
        raise ValueError(f"preset {preset_id} 不是多層構造，沒有厚度可當設計變數")
    porous = [index for index, layer in enumerate(material.layers, start=1) if isinstance(layer, PorousLayer)]
    air = [index for index, layer in enumerate(material.layers, start=1) if isinstance(layer, AirLayer)]
    if len(porous) != 1 or len(air) > 1 or len(porous) + len(air) != len(material.layers):
        raise ValueError(f"preset {preset_id} 必須是一層棉板、最多一層空腔，沒有別的層")
    thickness = variable_name(porous[0], _THICKNESS)
    cavity = variable_name(air[0], _THICKNESS) if air else None
    lower = [family.thicknesses_m[0]]
    upper = [family.thicknesses_m[-1]]
    if cavity is not None:
        lower.append(family.cavity_range_m[0])
        upper.append(family.cavity_range_m[1])
    return SurfaceDesign(
        surface_id=surface_id, preset_id=preset_id, family_id=family.id, material=material,
        variables=tuple(name for name in (thickness, cavity) if name is not None),
        lower=tuple(lower), upper=tuple(upper), thickness_variable=thickness, cavity_variable=cavity,
        legal_thicknesses_m=family.thicknesses_m, max_total_depth_m=family.max_total_depth_m,
        mount=mount, backing_material=_backing(surface_id, material, mount, kept))


def _backing(surface_id: str, material: Assembly, mount: Mount,
             kept: Mapping[str, SurfaceMaterial] | None) -> WallMaterial | None:
    """貼在原牆上的面背後那面牆（``kept`` 那一面）；剛背 None。preset 不是剛背、缺原牆或原牆本身已是貼牆構造都拒收。"""
    if mount != MOUNT_ON_WALL:
        return None
    if material.backing != "rigid":
        raise ValueError(f"面 {surface_id} 的樣板背後不是剛背（{material.backing}），不能改成貼在原牆上")
    wall = None if kept is None else kept.get(surface_id)
    if wall is None:
        raise ValueError(f"面 {surface_id} 貼在原牆上，但方案原本的牆的材料清單沒有這一面")
    if isinstance(wall, MountedOnWall):
        raise ValueError(f"面 {surface_id} 方案原本的牆已經是貼在原牆上的構造，不能再貼一層")
    return wall


def design_template_material(design: SurfaceDesign) -> Assembly | MountedOnWall | None:
    """替身鏈建樣板、終點摘要用的構造（換 θ 之前）：剛背是 preset 那份 ``material``；貼在原牆上是
    ``MountedOnWall(layers=那份的層, wall=原牆)``；沒有樣板 None。"""
    if design.material is None:
        return None
    if design.mount != MOUNT_ON_WALL:
        return design.material
    assert design.backing_material is not None
    return MountedOnWall(layers=design.material.layers, wall=design.backing_material)


def designs_mount(designs: Sequence[SurfaceDesign]) -> Mount:
    """一串設計的安裝方式（每次搜尋一個值）；混了兩種拒收。"""
    found = {design.mount for design in designs}
    if len(found) != 1:
        raise ValueError(f"一次搜尋只有一種安裝方式，設計清單有 {sorted(found)}")
    return found.pop()


def _plain_design(surface_id: str, kept: Mapping[str, SurfaceMaterial] | None,
                  mount: Mount = MOUNT_RIGID) -> SurfaceDesign:
    """preset 是 None 的面：給了 ``kept`` 就是保留原牆（``kept`` 一定要有這一面），沒給才是剛性。"""
    if kept is None:
        return SurfaceDesign(surface_id, None, None, None, (), (), (), None, None, (), None, mount=mount)
    material = kept.get(surface_id)
    if material is None:
        raise ValueError(f"面 {surface_id} 不貼，但保留原牆的材料清單沒有這一面")
    return SurfaceDesign(surface_id, None, None, None, (), (), (), None, None, (), None, kept_material=material,
                         mount=mount)


def designs_from_presets(surfaces: Sequence[tuple[str, str | None]], settings: MaterialGradientSettings,
                         presets: Sequence[PresetMaterial], *,
                         kept: Mapping[str, SurfaceMaterial] | None = None,
                         mount: Mount = MOUNT_RIGID) -> tuple[SurfaceDesign, ...]:
    """一串（面代號, preset id 或 None）→ 每面的設計。面數不限，面代號不可重複。

    preset 是 None 的面：給了 ``kept``（面代號 → 方案原本的材料）就是「不貼、保留原牆」，``kept`` 缺這一面就拒收；
    沒給 ``kept`` 才是剛性面（合成題用）。有 preset 的面：剛背不看 ``kept``；``mount`` 是 ``on_wall`` 時背後接
    ``kept`` 那一面（缺就拒收）。
    """
    if not surfaces:
        raise ValueError("至少要有一個面")
    ids = [surface_id for surface_id, _ in surfaces]
    if len(set(ids)) != len(ids):
        raise ValueError(f"面代號重複：{ids}")
    by_id = {preset.id: preset for preset in presets}
    return tuple(
        _plain_design(surface_id, kept, mount) if preset_id is None
        else fixed_candidate_design(surface_id, preset_id, by_id, mount) if is_fixed_target(preset_id)
        else _surface_design(surface_id, preset_id, settings, by_id, mount=mount, kept=kept)
        for surface_id, preset_id in surfaces)


def fixed_candidate_design(surface_id: str, target: str, presets: Mapping[str, PresetMaterial],
                            mount: Mount) -> SurfaceDesign:
    """``fixed:<代號>``：整塊換成那份固定材料（構造清單裡同名那一筆，由准用清單接進來），沒有變數；整塊取代那一面的牆
    （不管安裝方式，材料本身就是完整的牆面）。清單裡沒有這一筆就拒收。"""
    preset = presets.get(target)
    if preset is None:
        raise ValueError(f"面 {surface_id} 的 {target} 不在這一場的固定材料清單裡（只能是准用清單列過的固定材料）")
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    return SurfaceDesign(surface_id, target, None, None, (), (), (), None, None, (), None,
                         kept_material=adapter.validate_python(preset.material), mount=mount)


def wall_designs_from_presets(preset_ids: Sequence[str | None], settings: MaterialGradientSettings,
                              presets: Sequence[PresetMaterial], *,
                              kept: Mapping[str, SurfaceMaterial] | None = None,
                              mount: Mount = MOUNT_RIGID) -> tuple[SurfaceDesign, ...]:
    """今天的替身只有六面：照 ``Wall.all()`` 給六個 preset id（或 None），面代號是牆名；``kept``、``mount`` 同
    :func:`designs_from_presets`（鍵是牆名）。"""
    names = wall_surface_ids()
    if len(preset_ids) != len(names):
        raise ValueError(f"六面牆要給 {len(names)} 個 preset id（照 Wall.all()），收到 {len(preset_ids)} 個")
    return designs_from_presets(tuple(zip(names, preset_ids, strict=True)), settings, presets, kept=kept, mount=mount)
