"""替身訓練過哪些材料（#1082 段 2-3c）：建模把這一場的家族記進訓練說明，開跑前核這一場的家族都在裡面；不載入 JAX。

搜尋會走到的材料＝搜尋設定的家族：設定檔的在前、材料庫的可最佳化材料一筆一個接在後
（:func:`~aosr.materials.search_library.with_user_materials`），換樣板也會換到每一個。替身只在訓練與留出抽過的範圍驗過，
這一場的家族有一個沒訓練過、或範圍超出訓練過的，就拒收、要重建——不是只警告（限制一律是硬的）。

一個家族記：代號、換樣板構造的代號與內容摘要（流阻之類的內容換了、代號沒變也算不同）、厚度範圍、空腔範圍、總深度上限。
訓練只用換樣板抽（:mod:`aosr.gradient.surrogate_training`），所以記的空腔是「真的抽過的」：換樣板沒有空氣層就是 None。
開跑那一側要的是「搜尋可能走到的」：家族裡任一個構造（起跑可以選任一個成員）有空氣層就要空腔，而且每個成員的多孔層
（厚度除外）要跟換樣板相同——不同的成員訓練沒見過，點名拒收（審查 #1107 點出：換樣板沒空腔、成員有空腔會漏核）。
涵蓋＝這一場的每個家族都找得到同代號、同構造內容，而且厚度與空腔範圍都在訓練過的裡面、總深度上限不比訓練過的大。
個案的每面深度上限只會把範圍切小，不算進來（替身照家族範圍建）。

訓練說明（``training.json``）沒有 ``coverage`` 這一節的是舊替身：#1098 那一版、或手建的替身旁邊根本沒有訓練說明，
都當成只用產品設定檔那幾個家族訓練過。這一場的家族就是產品設定檔那幾個時照舊放行；有別的家族就拒收、要重建。
建模時這一場的家族就是產品設定檔那幾個（材料庫沒有可最佳化材料、也沒換家族設定）就不寫這一節，訓練說明逐位照舊。

每區准用清單裡的固定材料候選（#1082 段 2-3b 的 ``fixed:<代號>``，整塊取代那一片的牆）也一樣（主對話 11-12 定）：
建模照設計設定的准用清單把每個候選放進訓練與留出，記下「哪一個、材料內容摘要、准用在哪幾片」；開跑時這一場的每個候選
都要同代號、同內容，准用的面片也要在訓練放過的裡面，不然拒收、要重建。2-3b 開跑前每個候選的完整解留出驗證照做。

訓練說明不在替身 manifest 的雜湊裡：它跟替身同一個資料夾整份換名進格子，事後被手改閘門看不出來（照實記下，非阻塞）。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import (
    FamilySettings, MaterialGradientSettings, is_fixed_target, load_material_gradient_settings,
)
from aosr.config.paths import config_path
from aosr.materials.search_library import PresetMaterial, SearchMaterials, snapshot_bytes
from aosr.physics.fem_reduced_receipt import ReducedModelNotValidated

TRAINING_VERSION: Final = "aosr.gradient.surrogate_training.v1"
COVERAGE_VERSION: Final = "aosr.gradient.surrogate_training.v2"
NOT_TRAINED_TEXT: Final = "替身沒用這些材料訓練過，要重建"


class SurrogateNotTrainedOn(ReducedModelNotValidated):
    """這一場的家族超出替身訓練過的：拒收、要重建（是替身核不過的一種，網頁照「替身核不過、要重建」那一條走）。"""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class FamilyCoverage(_Frozen):
    """一個家族的訓練範圍：換樣板構造（代號＋內容摘要）、厚度與空腔的上下界、總深度上限（公尺）。"""

    id: str = Field(min_length=1)
    template_preset: str = Field(min_length=1)
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    thickness_m: tuple[float, float]
    cavity_m: tuple[float, float] | None
    max_total_depth_m: float


class FixedCoverage(_Frozen):
    """一個固定材料候選（``fixed:<代號>``）：材料內容摘要、訓練時放過它的面片（照名字排）。"""

    target: str = Field(pattern=r"^fixed:[a-z][a-z0-9_]*$")
    material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    surfaces: tuple[str, ...] = Field(min_length=1)


class Coverage(_Frozen):
    """訓練說明裡的 ``coverage`` 一節：訓練過的每個家族、固定材料候選、那一份搜尋設定的指紋、材料庫（接成家族的那幾筆）的指紋。"""

    families: tuple[FamilyCoverage, ...] = Field(min_length=1)
    fixed: tuple[FixedCoverage, ...] = ()
    settings_fingerprint: str
    materials_sha256: str | None


class FixedTarget(NamedTuple):
    """這一場（或這一次建模）的一個固定材料候選：目標字串、那份完整牆面材料（物理格式）、准用它的面片。"""

    target: str
    material: Mapping[str, object]
    surfaces: tuple[str, ...]

    def coverage(self) -> FixedCoverage:
        return FixedCoverage(target=self.target, material_sha256=template_sha256(self.material), surfaces=self.surfaces)


def fixed_targets(allowed: Mapping[str, Sequence[str]], table: Sequence[PresetMaterial]) -> tuple[FixedTarget, ...]:
    """准用清單（面片 → 目標）裡每個 ``fixed:<代號>`` → 它的材料（構造清單裡同名那一筆，
    :func:`~aosr.materials.search_library.fixed_candidates` 生成的）與准用它的面片；照目標字串排。"""
    by_id = {preset.id: preset for preset in table}
    found = []
    for target in sorted({item for targets in allowed.values() for item in targets if is_fixed_target(item)}):
        preset = by_id.get(target)
        if preset is None:
            raise ValueError(f"准用清單的固定材料 {target} 不在構造清單裡")
        found.append(FixedTarget(target, preset.material,
                                 tuple(sorted(name for name, targets in allowed.items() if target in targets))))
    return tuple(found)


def template_sha256(material: Mapping[str, object]) -> str:
    """構造內容的摘要：正規化 JSON（鍵排序、緊湊分隔、不轉義非 ASCII）做 SHA-256。"""
    content = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _layers(material: Mapping[str, object]) -> list[Mapping[str, object]]:
    layers = material.get("layers")
    return [layer for layer in layers if isinstance(layer, Mapping)] if isinstance(layers, list) else []


def has_air_layer(material: Mapping[str, object]) -> bool:
    """構造有空氣層（``kind = "air"``）：搜尋設計照這一層開空腔變數（``material_gradient._surface_design`` 同一條）。"""
    return any(layer.get("kind") == "air" for layer in _layers(material))


def porous_identity(material: Mapping[str, object]) -> list[dict[str, object]]:
    """多孔層去掉厚度（厚度是搜尋變數）之後的內容：同一個家族的成員要跟換樣板一樣，替身才算見過。"""
    return [{key: value for key, value in layer.items() if key != "thickness_m"}
            for layer in _layers(material) if layer.get("kind") == "porous"]


def _preset(table: Mapping[str, PresetMaterial], family: FamilySettings, preset_id: str) -> PresetMaterial:
    preset = table.get(preset_id)
    if preset is None:
        raise ValueError(f"家族 {family.id} 的構造 {preset_id} 不在構造清單裡")
    return preset


def family_coverage(settings: MaterialGradientSettings, presets: Sequence[PresetMaterial], *,
                    reachable: bool = False) -> tuple[FamilyCoverage, ...]:
    """搜尋設定的每個家族 → 它的範圍；構造照構造清單找（找不到就拒收，講是哪一個）。

    ``reachable`` 是 False：建模真的抽過的（空腔只看換樣板有沒有空氣層）；True：開跑可能走到的（家族任一成員有空氣層就算）。"""
    table = {preset.id: preset for preset in presets}
    rows = []
    for family in settings.family:
        template = _preset(table, family, family.template_preset)
        shapes = [template.material]
        if reachable:
            shapes += [_preset(table, family, member).material for member in family.presets]
        cavity = family.cavity_range_m if any(has_air_layer(shape) for shape in shapes) else None
        rows.append(FamilyCoverage(id=family.id, template_preset=family.template_preset,
                                   template_sha256=template_sha256(template.material),
                                   thickness_m=(min(family.thicknesses_m), max(family.thicknesses_m)),
                                   cavity_m=cavity, max_total_depth_m=family.max_total_depth_m))
    return tuple(rows)


def _member_problems(settings: MaterialGradientSettings, presets: Sequence[PresetMaterial]) -> list[str]:
    """家族成員的多孔層（厚度除外）跟換樣板不同：替身只用換樣板訓練過，那個成員沒見過。"""
    table = {preset.id: preset for preset in presets}
    problems = []
    for family in settings.family:
        porous = porous_identity(_preset(table, family, family.template_preset).material)
        for member in family.presets:
            if porous_identity(_preset(table, family, member).material) != porous:
                problems.append(f"家族 {family.id} 的構造 {member} 的多孔層跟換樣板 {family.template_preset} 不同"
                                "（替身只用換樣板訓練過）")
    return problems


def product_families() -> tuple[FamilySettings, ...]:
    """產品設定檔（``material_gradient.toml``）的家族：舊替身就是用這幾個訓練的。"""
    return load_material_gradient_settings(config_path("material_gradient.toml")).family


def only_product_families(materials: SearchMaterials) -> bool:
    """這一場的家族就是產品設定檔那幾個（沒有接進使用者材料、家族設定也沒換）。"""
    return not materials.entries and materials.settings.family == product_families()


def coverage_for(materials: SearchMaterials, fixed: Sequence[FixedTarget] = ()) -> Coverage | None:
    """建模要寫進訓練說明的那一節；只有產品設定檔那幾個家族、也沒有固定材料候選時是 None（訓練說明逐位照舊）。"""
    if only_product_families(materials) and not fixed:
        return None
    library = None if not materials.entries else hashlib.sha256(snapshot_bytes(materials.entries)).hexdigest()
    return Coverage(families=family_coverage(materials.settings, materials.presets),
                    fixed=tuple(target.coverage() for target in fixed),
                    settings_fingerprint=materials.settings.fingerprint, materials_sha256=library)


def read_coverage(training: Path) -> Coverage | None:
    """替身旁的訓練說明裡記的涵蓋；檔不在或沒有這一節（舊替身）是 None，檔壞了拒收。"""
    if not training.is_file():
        return None
    try:
        document = json.loads(training.read_text(encoding="utf-8"))
        section = document.get("coverage") if isinstance(document, dict) else None
        return None if section is None else Coverage.model_validate(section)
    except (OSError, ValueError, ValidationError) as error:
        raise SurrogateNotTrainedOn(f"{NOT_TRAINED_TEXT}：替身旁的訓練說明 {training} 讀不懂（{type(error).__name__}），"
                                    "看不出訓練過哪些材料") from error


def _inside(wanted: tuple[float, float], trained: tuple[float, float]) -> bool:
    return trained[0] <= wanted[0] and wanted[1] <= trained[1]


def _problems(wanted: Sequence[FamilyCoverage], trained: Sequence[FamilyCoverage]) -> list[str]:
    by_id = {family.id: family for family in trained}
    problems = []
    for family in wanted:
        seen = by_id.get(family.id)
        if seen is None:
            problems.append(f"家族 {family.id} 沒訓練過")
        elif (seen.template_preset, seen.template_sha256) != (family.template_preset, family.template_sha256):
            problems.append(f"家族 {family.id} 的材料內容跟訓練時不同")
        else:
            if not _inside(family.thickness_m, seen.thickness_m):
                problems.append(f"家族 {family.id} 的厚度範圍 {list(family.thickness_m)} 超出訓練過的 {list(seen.thickness_m)}")
            if family.cavity_m is not None and seen.cavity_m is None:
                problems.append(f"家族 {family.id} 的空腔沒訓練過（訓練用的換樣板沒有空氣層，這一場會調空腔）")
            elif family.cavity_m is not None and seen.cavity_m is not None and not _inside(family.cavity_m, seen.cavity_m):
                problems.append(f"家族 {family.id} 的空腔範圍 {list(family.cavity_m)} 超出訓練過的 {list(seen.cavity_m)}")
            if family.max_total_depth_m > seen.max_total_depth_m:
                problems.append(f"家族 {family.id} 的總深度上限 {family.max_total_depth_m:g} m 比訓練過的 "
                                f"{seen.max_total_depth_m:g} m 大")
    return problems


def _fixed_problems(wanted: Sequence[FixedTarget], trained: Sequence[FixedCoverage]) -> list[str]:
    by_target = {item.target: item for item in trained}
    problems = []
    for item in wanted:
        seen = by_target.get(item.target)
        if seen is None:
            problems.append(f"固定材料 {item.target} 沒訓練過")
        elif seen.material_sha256 != template_sha256(item.material):
            problems.append(f"固定材料 {item.target} 的內容跟訓練時不同")
        elif not set(item.surfaces) <= set(seen.surfaces):
            problems.append(f"固定材料 {item.target} 准用在 {sorted(set(item.surfaces) - set(seen.surfaces))}，"
                            f"訓練只放過 {list(seen.surfaces)}")
    return problems


def check_covers(training: Path, materials: SearchMaterials, fixed: Sequence[FixedTarget] = ()) -> None:
    """開跑前核這一場的家族與固定材料候選都在替身訓練過的範圍裡；不在就拋 :class:`SurrogateNotTrainedOn`（點名哪一個、哪一格）。

    ``training`` 是替身旁的訓練說明（``surrogate.npz`` 同資料夾的 ``training.json``）；``fixed`` 是這一場准用清單的固定材料候選。"""
    trained = read_coverage(training)
    wanted = family_coverage(materials.settings, materials.presets, reachable=True)
    if trained is None:
        if only_product_families(materials) and not fixed:
            return
        # 舊替身照產品設定檔家族＋產品構造清單訓練（建模命令列用的就是 construction_presets.toml）：基準不跟這一場的構造清單走，
        # 不然同代號樣板換了內容（流阻、多了空氣層）兩邊一起變、比不出來。
        product = family_coverage(load_material_gradient_settings(config_path("material_gradient.toml")),
                                  load_construction_presets(config_path("construction_presets.toml")))
        extra = (_problems(wanted, product) + _member_problems(materials.settings, materials.presets)
                 + _fixed_problems(fixed, ()))
        if not extra:
            # 舊替身當作照產品設定檔家族訓練過：這一場只把產品家族收窄或拿掉幾個（照 v2 同一條「相同或更窄」），照舊收下。
            return
        raise SurrogateNotTrainedOn(f"{NOT_TRAINED_TEXT}：這份替身沒記訓練過哪些材料（材料庫接進搜尋之前建的，"
                                    f"只用產品設定檔的家族訓練過），這一場{'；'.join(extra)}")
    problems = (_problems(wanted, trained.families) + _member_problems(materials.settings, materials.presets)
                + _fixed_problems(fixed, trained.fixed))
    if problems:
        raise SurrogateNotTrainedOn(f"{NOT_TRAINED_TEXT}：{'；'.join(problems)}")
