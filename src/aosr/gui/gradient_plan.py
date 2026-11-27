"""設計設定 → 第三階段梯度搜尋（gradient search，用替身算梯度找材料）命令列參數（#1093）。純函式：不開行程、不寫檔。

每一區的做法照 ``zone_design.effective_modes``（沒寫的區一律固定）：
- 交給搜尋：起跑用選定家族的樣板（``template_preset``），之後換樣板也照搜尋設定；
- 固定這一組材料、保留原牆：兩種都是「用方案檔那一區自己寫的材料、不帶變數、不換樣板」，命令列寫成固定面
  ``面=scheme-material``（方案檔裡固定區放的是那一組材料、保留原牆的區放的是原牆，所以同一條路就對）。
面片的順序照命令列自己的取法（``scheme_patch_keys``），不在這裡另排。

每區准用清單（#1082 段 2-3b，設計設定的 ``allowed``，材料庫代號或 none）轉成命令列 ``--allowed 面=目標,…``：
可最佳化材料是家族代號、固定材料是 ``fixed:<代號>``、none 照寫。有清單的區起點照順序挑：選定家族在清單裡就用它、
不在就用清單裡第一個家族（設定檔順序）、都沒有家族就不貼（清單要有 none）；清單只放了固定成品就拒收（起點要是可調材料或不貼）。

深度上限（#1082 段 1-4）：設計設定的三層（房間預設／每面牆／每一區）取最小，只帶給搜尋區；開跑請求另填的只能更嚴
（兩邊取小），``depth_sources`` 記每片是哪一層擋的（``zone``／``wall``／``room``／``request``）。設計設定有一條紅的
（``zone_limits``：固定區貼太深、分區太小或出界）就不開。

個案上限跟材料組自己的限制取較嚴（#1082 段 2-4，判法在 ``aosr.gradient.depth_fit``，立面圖標紅同一把）：有上限的搜尋區，
准用的每一組（沒寫清單就是全部可調材料）照最後生效的上限（設計設定與這一跑另填取小）判，計畫記 ``depth_targets``
（每一組的生效上限與哪一條擋住）；不可用的組從 ``--allowed`` 拿掉、起點也跳過它（沒寫清單的區由命令列換樣板時自己濾），
一組都不可用就不開（「不貼」不算救）。個案上限比材料組自己的上限大不再拒收，命令列取較嚴的那一個。

左右對稱（#1206 第 2 支）：設計設定的開關開著（缺那一格就是開）時，由程式最佳化、對著對稱軸互為鏡像的兩區配成一對
（``zone_symmetry.pair_zones``，分區頁看的也是這一支的結果），每一對在開跑參數多一個 ``--tie A=B``（A 是面序在前那一片）。
位置對上的每一對先過引擎的開跑前檢查（:func:`_engine_check`：命令列開跑那一組起跑配置 ``cli.start_designs``，再逐對叫
``ties.check_ties_against_start``；這裡不另寫一份規則），引擎拒收的不進計畫。關掉、或沒有位置對上的兩區時這一段一行都不跑，
計畫與參數跟沒有這一格之前逐位相同。``--tie`` 只加在 :func:`start_arguments`，不進 :func:`configuration_arguments`
（建替身照那一份核固定材料）。

「鋼板直接貼在棉上」的構造不進程式最佳化（#1208 第 1 支，判斷在 ``reference_only``：界線頻率以下的數字只能參考）：
准用清單原文（設計設定存的那一份，還沒拿掉任何一組之前）裡有一筆固定材料是這種構造就拒收、點名哪一區哪一筆；
不默默拿掉、不改存檔。判在限制檢查與深度之前：太深放不下的那一筆照樣是這一句，不會先被別的理由蓋掉。
沒有這種構造時這一段一個字都不產生，計畫與參數跟之前逐位相同。命令列照舊收 ``--allowed fixed:<代號>``，這裡只管網頁。

每一區自己的可調範圍（#1199，設計設定的 ``ranges``，材料代號 → 棉厚與空腔的下上）：對材料庫核（只准收窄、棉厚是產品厚度、
收窄後放得下）跟立面圖標紅同一把（``zone_limits.zone_range_problems``，紅的就不開；命令列開跑照同一條再核一次），
第三階段頁另填的上限（分區頁看不到）收窄後放不下也拒收、同一句（:func:`_depth_targets`）。
起跑那一族有範圍時起跑值（樣板的初值）不在範圍裡不拒收：引擎照換樣板那一把把起點投影進範圍（``material_gradient.project_initial``）。
計畫記 ``ranges``（面片代號 → 家族 → 範圍），開跑參數每一族多一個 ``--range``（只加在 :func:`start_arguments`：
範圍只收窄、家族代號不變，建替身那一份不帶、不用重建替身）。沒有範圍時這一段一個字都不產生，計畫與參數跟之前逐位相同。

這裡只擋網頁講得清楚的事（設計設定對不上方案、一區都沒交給搜尋、家族不認識、深度上限給在不是搜尋的區、准用清單的代號
不認識或排不出起點、有超過限制的區、准用的材料組一組都放不下）。
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import (
    MOUNT_ON_WALL, NONE_TARGET, SCHEME_MATERIAL, FamilySettings, MaterialGradientSettings, Mount, fixed_target,
)
from aosr.config.paths import config_path
from aosr.gradient.boundary import scheme_patch_keys, surface_ids
from aosr.gradient.depth_fit import ZoneRange, narrowed_family, ranged_misfits
from aosr.gui.material_surfaces import legacy_replacement
from aosr.gui.wall_words import wall_words
from aosr.gui.zone_construction import OriginalReader
from aosr.gui.zone_design import WHOLE_WALL_KEY, DesignSettings, ZoneMode, effective_modes, mismatches
from aosr.gui.zone_limits import (
    CandidateSelection, DepthCap, DepthLevel, DepthTargets, RangeCheck, depth_cap, effective_candidates, no_group_words, reference_only_refusals, saved_refusals, target_fits,
    ranged_misfit_words, target_line, zone_limits, zone_words,
)
from aosr.gui.zone_symmetry import PAIR_WITHIN_M, Frame, SettingsCheck, Symmetry, Unchecked, drawn_frames, drawn_partners, pair_zones, symmetry_on
from aosr.materials.construction_models import SurfaceMaterial
from aosr.materials.library import ConstructionEntry, LibraryEntry, builtin_entries, fixed_surface, full_library
from aosr.materials.search_library import PresetMaterial
from aosr.reporting.scheme import Scheme

if TYPE_CHECKING:
    from aosr.gradient.material_gradient import SurfaceDesign

Device = Literal["cpu", "gpu"]
_LOG = logging.getLogger(__name__)
_MATERIAL: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)


class PlanRefused(ValueError):
    """這份設計設定開不了搜尋；訊息是給畫面的白話。"""


class GradientRequest(BaseModel):
    """網頁開跑時可以選的幾格；全部有預設，什麼都不給就照預設跑。

    ``max_steps`` 不給＝不帶 ``--max-steps``，照搜尋設定檔的預設（現在 60）；``max_depths`` 是牆 → 區代號 → 公尺；
    ``frequency_batch`` 是替身每批求解幾個頻點（顯卡記憶體不夠時用；不給照命令列預設，一次全部）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    family: str | None = Field(default=None, min_length=1)
    mount: Mount = MOUNT_ON_WALL
    device: Device = "gpu"
    max_steps: int | None = Field(default=None, ge=1, strict=True)
    frequency_batch: int | None = Field(default=None, ge=1, strict=True)
    max_depths: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _positive_depths(self) -> Self:
        for wall, zones in self.max_depths.items():
            for zone, depth in zones.items():
                if depth <= 0.0:
                    raise ValueError(f"{wall} 分區 {zone} 的深度上限要大於 0 公尺，收到 {depth}")
        return self


class SearchPlan(BaseModel):
    """轉好的搜尋參數：面片代號、每片做法與起跑 preset、固定面、深度上限、家族與實際步數上限。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    surfaces: tuple[str, ...]
    modes: tuple[ZoneMode, ...]
    presets: tuple[str | None, ...]
    fixed_surfaces: dict[str, str]
    max_depths: dict[str, float]
    family: str
    max_steps: int
    request: GradientRequest
    # 面片代號 → 准用目標（命令列 --allowed 的寫法）；沒有清單的面不列，舊的計畫逐位不變。
    allowed: dict[str, tuple[str, ...]] = Field(default_factory=dict, exclude_if=lambda value: not value)
    # 面片代號 → 深度上限是哪一層擋的（zone／wall／room／request）；沒有上限的計畫不寫這一格（舊計畫逐位不變）。
    depth_sources: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    # 面片代號 → 每一組材料在那一片的判決（#1082 段 2-4）；只列有上限的搜尋區，沒有上限的計畫不寫這一格。
    depth_targets: dict[str, tuple[TargetLine, ...]] = Field(default_factory=dict, exclude_if=lambda value: not value)
    # 成對的兩片（左右對稱，#1206）：每一對面序在前的那一片在前，對與對照前面那一片的面序；沒有就不寫這一格（舊計畫逐位不變）。
    ties: tuple[tuple[str, str], ...] = Field(default=(), exclude_if=lambda value: not value)
    # 面片代號 → 家族代號 → 這一區自己的可調範圍（#1199，命令列 --range 的寫法）；沒有就不寫這一格（舊計畫逐位不變）。
    ranges: dict[str, dict[str, ZoneRange]] = Field(default_factory=dict, exclude_if=lambda value: not value)


class TargetLine(BaseModel):
    """一組材料在一片的判決：目標代號（命令列寫法）、名字、可不可用、生效上限（公尺；不可用是 None）、
    哪一條擋住（``case`` 個案上限／``material`` 材料組自己）與一句白話。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    target: str
    label: str
    usable: bool
    effective_m: float | None
    blocker: Literal["case", "material"] | None
    text: str


def surface_of(wall: str, zone: str) -> str:
    """設計設定的（牆, 區代號）→ 命令列的面片代號：整面一區是牆名，矩形是「牆@區」。"""
    return wall if zone == WHOLE_WALL_KEY else f"{wall}@{zone}"


def zone_of(surface: str) -> tuple[str, str]:
    wall, separator, zone = surface.partition("@")
    return wall, zone if separator else WHOLE_WALL_KEY


def _depths(request: GradientRequest, modes: Mapping[str, Mapping[str, ZoneMode]],
            design: DesignSettings | None) -> tuple[dict[str, float], dict[str, str]]:
    found: dict[str, tuple[float, str]] = {}
    for wall, wall_modes in modes.items():
        for zone, mode in wall_modes.items():
            cap = depth_cap(design, wall, zone)
            if mode == "search" and cap is not None:
                found[surface_of(wall, zone)] = (cap.value_m, cap.level)
    designed = bool(found)
    for wall, given in request.max_depths.items():
        for zone, depth in given.items():
            found_mode = modes.get(wall, {}).get(zone)
            if found_mode is None:
                raise PlanRefused(f"深度上限給的 {wall} 分區 {zone} 方案裡沒有")
            if found_mode != "search":
                raise PlanRefused(f"{wall} 分區 {zone} 不是交給搜尋的區，不能給深度上限（固定與保留原牆的區不加層）")
            surface = surface_of(wall, zone)
            if surface not in found or depth < found[surface][0]:
                found[surface] = (depth, "request")
    ordered = dict(sorted(found.items()))
    sources = {surface: source for surface, (_, source) in ordered.items()}
    if not designed:
        sources = {}  # 設計設定沒有給任何一區上限：計畫照舊不寫這一格
    return {surface: depth for surface, (depth, _) in ordered.items()}, sources


def _targets(wall: str, zone: str, ids: tuple[str, ...], settings: MaterialGradientSettings,
             fixed_ids: frozenset[str]) -> tuple[str, ...]:
    families = {family.id for family in settings.family}
    unknown = [item for item in ids if item != NONE_TARGET and item not in families and item not in fixed_ids]
    if unknown:
        raise PlanRefused(f"{wall} 分區 {zone} 的准用清單有材料庫裡沒有的代號 {unknown}")
    return tuple(item if item == NONE_TARGET or item in families else fixed_target(item) for item in ids)


def _start_preset(wall: str, zone: str, targets: tuple[str, ...] | None, family: FamilySettings,
                  settings: MaterialGradientSettings, unusable: frozenset[str], names: Mapping[str, str]) -> str | None:
    """起點：選定家族（在清單裡、放得下）→ 清單裡（沒清單就是全部）第一個放得下的家族（設定檔順序）→ 不貼。
    都沒有就拒收，原文跟立面圖那一區的紅框同一套白話（#1176）；牆名照 ``names``（方案的前後左右）。"""
    if (targets is None or family.id in targets) and family.id not in unusable:
        return family.template_preset
    listed = [item for item in settings.family if (targets is None or item.id in targets) and item.id not in unusable]
    if listed:
        return listed[0].template_preset
    if targets is None or NONE_TARGET in targets:
        return None
    raise PlanRefused(f"{zone_words(wall, zone, names)}只放了固定成品，程式沒得選：程式要從一個能調的東西開始，請在立面圖這一區加上「不貼」"
                      "或一種可調材料（例如 48K 玻璃棉），或把做法改成「指定貼一種」")


def _allowed(design: DesignSettings | None, targets: DepthTargets, depths: Mapping[str, float]) -> dict[tuple[str, str], tuple[str, ...]]:
    found = {}
    for wall, zones in (design.zones if design is not None else {}).items():
        for zone, item in zones.items():
            if item.allowed is not None:
                selection = CandidateSelection(allowed=item.allowed, ranges=item.ranges, cap=depths.get(surface_of(wall, zone)))
                found[(wall, zone)] = effective_candidates(selection, targets).allowed
    return found


def _ranges(design: DesignSettings | None) -> dict[tuple[str, str], dict[str, ZoneRange]]:
    """每一區自己的可調範圍（區 → 材料代號 → 範圍）。對材料庫核（只准收窄、棉厚是產品厚度、收窄後放得下）在前面
    立面圖標紅那一把（``saved_refusals`` → ``zone_limits.zone_range_problems``，判法 ``depth_fit.range_problem``）已經擋過、
    紅的就不開；命令列開跑再照同一條核一次（``fixed_surfaces.check_ranges``）。"""
    return {(wall, zone): dict(sorted(item.ranges.items()))
            for wall, zones in (design.zones if design is not None else {}).items()
            for zone, item in zones.items() if item.ranges is not None}


def used_ranges(plan: SearchPlan, settings: MaterialGradientSettings) -> dict[str, dict[str, dict[str, object]]]:
    """開跑預覽給畫面的（#1199）：有範圍的每一片每一族實際用的範圍——棉厚下上、切過的產品厚度、空腔下上
    （材料搜尋不調空腔是 None）。只排給人看，不進計畫。"""
    found: dict[str, dict[str, dict[str, object]]] = {}
    for surface, families in plan.ranges.items():
        for code, given in families.items():
            family = settings.family_by_id(code)
            narrowed = narrowed_family(family, given)
            found.setdefault(surface, {})[code] = {
                "porous_m": list(given.porous_m), "thicknesses_m": list(narrowed.thicknesses_m),
                "cavity_m": None if family.initial_cavity_m is None else list(narrowed.cavity_range_m)}
    return found


def _refuse_reference_only(scheme: Mapping[str, object], design: DesignSettings | None, targets: DepthTargets,
                           names: Mapping[str, str]) -> None:
    """准用清單原文裡有「鋼板直接貼在棉上」的固定材料就拒收（#1208）：寫排第一的那一組的整句
    （分區頁列的是同一支 ``zone_limits.reference_only_refusals`` 的每一組，順序也是它排的）。"""
    refused = reference_only_refusals(scheme, design, targets, names)
    if refused:
        raise PlanRefused(str(refused[0]["text"]))


def builtin_depth_targets(settings: MaterialGradientSettings) -> DepthTargets:
    """沒有材料庫可讀時（考卷、命令列以外的呼叫）的材料組：這份設定的家族＋內建構造預設當固定材料，名字照代號。"""
    presets = load_construction_presets(config_path("construction_presets.toml"))
    labels = {entry.id: entry.label for entry in builtin_entries()}
    return DepthTargets(settings, {preset.id: _MATERIAL.validate_python(preset.material) for preset in presets}, labels)


def library_depth_targets(settings: MaterialGradientSettings, entries: Sequence[LibraryEntry], *,
                          read_original: OriginalReader | None = None) -> DepthTargets:
    """網頁那一份：``settings`` 已接上使用者可最佳化材料（``with_user_materials``）；固定材料是內建構造加使用者的固定材料，
    名字照材料庫。"""
    entries = (*full_library(None), *entries)
    return DepthTargets(settings, {entry.id: material for entry in entries if (material := fixed_surface(entry)) is not None},
                        {entry.id: entry.label for entry in entries},
                        frozenset(entry.id for entry in entries if isinstance(entry, ConstructionEntry) and entry.backing != "zone_wall"),
                        frozenset(entry.id for entry in entries if isinstance(entry, ConstructionEntry) and entry.backing == "zone_wall"), read_original,
                        frozenset(entry.id for entry in entries if legacy_replacement(entry)))


_LEVEL_WORDS = {"zone": "這一塊", "wall": "這面牆", "room": "房間預設", "request": "這一跑另填"}


def _depth_targets(design: DesignSettings | None, pairs: list[tuple[str, str]], per_surface: tuple[ZoneMode, ...],
                   depths: Mapping[str, float], sources: Mapping[str, str], targets: DepthTargets,
                   names: Mapping[str, str]) -> dict[str, tuple[TargetLine, ...]]:
    """交給搜尋、有上限的每一區，准用的每一組在生效上限下的判決（開跑預覽那一格；准用清單照它拿掉放不下的組）。

    寫了可調範圍的組照收窄後的判（#1199，同分區頁 ``zone_limits._target_problems`` 與命令列 ``check_depth_targets``）：
    收窄後放不下的（``depth_fit.ranged_misfits``）拒收，句子同分區頁那一句（``ranged_misfit_words``）。分區頁看不到
    第三階段頁另填的上限（``GradientRequest.max_depths``），那一格只有這裡擋得到；這裡在左右對稱那一段之前，
    所以不會變成「建不出起跑配置、悄悄不綁」。准用的組一組都放不下照舊拒收。"""
    found: dict[str, tuple[TargetLine, ...]] = {}
    for (wall, zone), mode in zip(pairs, per_surface, strict=True):
        surface = surface_of(wall, zone)
        if mode != "search" or surface not in depths:
            continue
        item = design.zones.get(wall, {}).get(zone) if design is not None else None
        given = None if item is None else item.ranges
        fits = target_fits(item.allowed if item else None, depths[surface], targets, given)
        if not fits:
            continue
        source = sources.get(surface, "request")
        cap = DepthCap(depths[surface], cast(DepthLevel, source if source in ("zone", "wall", "room") else "zone"))
        lines = [target_line(fit, cap, targets, level_words=_LEVEL_WORDS.get(source, source)) for fit in fits]
        misfits = ranged_misfits(fits, given)
        if misfits:
            raise PlanRefused(f"{zone_words(wall, zone, names)}：" + "；".join(
                ranged_misfit_words(fit, cap, targets, level_words=_LEVEL_WORDS.get(source, source)) for fit in misfits))
        if not any(fit.usable for fit in fits):
            raise PlanRefused(f"{zone_words(wall, zone, names)}：{no_group_words(lines)}")
        found[surface] = tuple(TargetLine.model_validate(line) for line in lines)
    return dict(sorted(found.items()))


class TieInputs(NamedTuple):
    """引擎核綁定要的、計畫本身沒帶的那幾樣（命令列開跑時自己讀的）：構造清單（不給是產品構造預設）、使用者材料
    （准用清單點名的固定材料從這裡接進構造清單，同命令列）、能力表的路徑（不給是產品那一份）。"""

    presets: tuple[PresetMaterial, ...] | None = None
    entries: tuple[LibraryEntry, ...] = ()
    capabilities: Path | None = None


# 引擎拒收一對時說的是哪一項（``ties.TieRefused.what``）→ 給人看的一句。
_TIE_WORDS = {"准用清單": "兩區讓程式選的材料不一樣", "家族": "兩區讓程式選的材料不一樣（起點的材料不同）",
              "起跑樣板": "兩區讓程式選的材料不一樣（起點的材料不同）", "θ 上下界": "兩區可調的範圍不一樣",
              "深度上限": "兩區的最深不一樣", "可調範圍": "兩區可調的範圍不一樣", "裝法": "兩區的裝法不一樣", "fixed": "其中一區不是由程式最佳化",
              "missing": "其中一區不在這一份方案的分區裡"}


def _engine_check(scheme: Scheme, plan: SearchPlan, settings: MaterialGradientSettings, inputs: TieInputs) -> SettingsCheck:
    """位置對上的一對能不能綁：問引擎的開跑前檢查（``aosr.gradient.ties.check_ties_against_start``），這裡不另寫規則。

    起跑配置用命令列開跑同一支（``cli.start_designs``，構造清單照命令列的接法），第一次被問才建（要載入搜尋那一層，
    沒有位置對上的兩區就不載入）；起跑配置本身引擎就不收時，每一對都算沒配成，回 :class:`Unchecked`
    （不是這一對設定不一樣；例外原文只進伺服器記錄）。"""
    built: list[tuple[tuple[SurfaceDesign, ...], dict[str, tuple[str, ...]], dict[str, dict[str, ZoneRange]]] | Unchecked] = []

    def check(first: str, second: str) -> str | Unchecked | None:
        from aosr.config.capabilities import load_capabilities
        from aosr.config.directivity_defaults import load_directivity_defaults
        from aosr.config.material_gradient import fixed_target_id
        from aosr.gradient.cli import start_designs
        from aosr.gradient.fixed_surfaces import expanded_ranges, fixed_targets_of
        from aosr.gradient.start_config import presets_table
        from aosr.gradient.ties import TieRefused, check_ties_against_start
        from aosr.materials.search_library import fixed_candidates, user_fixed_entries

        if not built:
            try:
                table = presets_table() if inputs.presets is None else inputs.presets
                targets = fixed_targets_of(plan.allowed)
                if targets:
                    listed = user_fixed_entries([fixed_target_id(target) for target in targets], inputs.entries)
                    table = (*table, *fixed_candidates(targets, table, listed))
                designs, expanded = start_designs(
                    scheme, settings, table, plan.presets, mount=plan.request.mount, fixed=dict(plan.fixed_surfaces),
                    caps=dict(plan.max_depths), allowed=dict(plan.allowed),
                    capabilities=load_capabilities(inputs.capabilities or config_path("capabilities.toml")),
                    directivity=load_directivity_defaults(config_path("directivity_defaults.toml")), ranges=plan.ranges)
                built.append((designs, expanded, expanded_ranges(plan.ranges, [design.surface_id for design in designs])))
            except (ValueError, OSError) as error:  # 引擎拒收，或檢查要的檔讀不到：都不讓分區頁的限制檢查整支倒掉
                # 原文是程式的話（代號、路徑），只進伺服器記錄，不上畫面。
                _LOG.warning("左右對稱：建不出開跑那一組起跑配置，核不了綁定：%s", error)
                built.append(Unchecked("這份設定開跑前檢查沒過", "這份設定開跑前檢查沒過，先看不了兩區的設定一不一樣。"))
        start = built[0]
        if isinstance(start, Unchecked):
            return start
        try:
            check_ties_against_start(((first, second),), *start)  # 起跑設計、准用清單、可調範圍（都展開到面片）
        except TieRefused as error:
            return _TIE_WORDS.get(error.what, f"兩區的{error.what}不一樣")
        return None

    return check


def plan_symmetry(scheme: Scheme, design: DesignSettings | None, plan: SearchPlan, settings: MaterialGradientSettings,
                  inputs: TieInputs | None = None) -> Symmetry:
    """這一份計畫的配對結果（成對的、沒配成的與原因）：``zone_symmetry.pair_zones`` 配上引擎的開跑前檢查。
    ``plan`` 只拿起跑配置那幾格（起跑樣板、裝法、固定面、深度上限、准用清單），帶不帶 ``ties`` 都一樣。"""
    return pair_zones(scheme, design, _engine_check(scheme, plan, settings, inputs or TieInputs()))


def draft_symmetry(document: object, design_value: object,
                   plan: Callable[[Scheme, DesignSettings | None], tuple[SearchPlan, MaterialGradientSettings, TieInputs]]
                   ) -> dict[str, object]:
    """分區頁要的配對（表單草稿＋還沒存的設計設定）：算得出來回 ``available: True`` 加 :class:`Symmetry` 的每一格；
    草稿還不是一份合法方案、或設計設定格式不對，回 ``available: False`` 與一句原因。``plan`` 回不帶綁定的計畫
    （:func:`plan_search` 的 ``pair=False``）、搜尋設定與 :class:`TieInputs`，只在有位置對上的兩區時才叫；
    它說開不了（:class:`PlanRefused`）時那幾對照那個原因算沒配成。"""
    try:
        scheme = Scheme.model_validate(document)
    except ValidationError:
        return {"available": False, "partners": {}, "message": "方案還有沒填好的地方，先算不了哪幾區成對"}
    try:
        design = None if design_value is None else DesignSettings.model_validate(design_value)
    except ValidationError:
        return {"available": False, "partners": {}, "message": "設計設定格式不對，先算不了哪幾區成對"}
    checks: list[SettingsCheck] = []

    def check(first: str, second: str) -> str | Unchecked | None:
        if not checks:
            try:
                base, settings, inputs = plan(scheme, design)
                checks.append(_engine_check(scheme, base, settings, inputs))
            except PlanRefused as error:  # 原文是寫給畫面的白話，照寫
                refused = Unchecked("這份設定現在開不了搜尋", f"這份設定現在開不了搜尋：{str(error).rstrip('。')}。")
                checks.append(lambda first, second: refused)
        return checks[0](first, second)

    result = pair_zones(scheme, design, check)
    frame = Frame(scheme, result.axis) if result.axis else None
    return {"available": True, **result.model_dump(mode="json"), "partners": drawn_partners(scheme, frame) if frame else {},
            "projections": {wall: frame.projection(wall) for wall in frame.sizes} if frame else {},
            "placement_refusals": {z.surface: refused[1] for z in drawn_frames(scheme)
                                   if (refused := frame.mirror_refusal(z))} if frame else {},
            "pair_within_m": PAIR_WITHIN_M}


def plan_search(scheme: Scheme, design: DesignSettings | None, request: GradientRequest,
                settings: MaterialGradientSettings, fixed_ids: frozenset[str] = frozenset(), *,
                range_check: RangeCheck | None = None, depth_targets: DepthTargets | None = None,
                tie_inputs: TieInputs | None = None, pair: bool = True) -> SearchPlan:
    """``fixed_ids`` 是材料庫裡固定材料的代號（內建構造預設＋使用者的固定材料），准用清單照它認固定材料；
    ``range_check`` 給了就連可信範圍一起擋（跟立面圖標紅同一把，網頁開跑與建替身傳材料頁那一支）；
    ``depth_targets`` 是判材料組放不放得進上限的那一份（網頁傳接了材料庫的，不給就用 :func:`builtin_depth_targets`）；
    ``tie_inputs`` 是引擎核綁定要的那幾樣（網頁傳接了材料庫的）；``pair=False`` 不配對（分區頁要自己拿整份配對結果時用）。"""
    plan = _plan_without_ties(scheme, design, request, settings, fixed_ids, range_check=range_check, depth_targets=depth_targets)
    if not pair or not symmetry_on(design):
        return plan
    pairs = plan_symmetry(scheme, design, plan, settings, tie_inputs).pairs
    return plan if not pairs else plan.model_copy(update={"ties": tuple((item.first, item.second) for item in pairs)})


def _plan_without_ties(scheme: Scheme, design: DesignSettings | None, request: GradientRequest,
                       settings: MaterialGradientSettings, fixed_ids: frozenset[str], *,
                       range_check: RangeCheck | None, depth_targets: DepthTargets | None) -> SearchPlan:
    problems = mismatches(design, scheme) if design is not None else []
    if problems:
        raise PlanRefused("設計設定跟方案對不上（" + "；".join(problems) + "），先在立面圖重存每一區的做法")
    targets_for_depth = depth_targets if depth_targets is not None else builtin_depth_targets(settings)
    names = wall_words(scheme)
    document = scheme.model_dump(mode="json")
    _refuse_reference_only(document, design, targets_for_depth, names)
    refusals = saved_refusals(document, design, range_check, targets_for_depth)
    if refusals:
        raise PlanRefused("有超過限制的區（立面圖上標紅的），先改好再開：" + "；".join(refusals))
    modes = effective_modes(design, scheme)
    surfaces = surface_ids(scheme_patch_keys(scheme))
    pairs = [zone_of(surface) for surface in surfaces]
    listed = {(wall, zone) for wall, zones in modes.items() for zone in zones}
    if set(pairs) != listed or len(pairs) != len(listed):
        raise RuntimeError(f"方案的面片 {list(surfaces)} 跟設計設定的區 {sorted(listed)} 不是同一組")
    per_surface = tuple(modes[wall][zone] for wall, zone in pairs)
    if "search" not in per_surface:
        raise PlanRefused("沒有任何一區交給搜尋：每一區都定了，就用一般計算直接算，不必開第三階段搜尋")
    try:
        family = settings.family_by_id(request.family) if request.family is not None else settings.family[0]
    except ValueError as error:
        raise PlanRefused(f"{error}；可選：{'、'.join(item.id for item in settings.family)}") from error
    ranges = _ranges(design)
    depths, sources = _depths(request, modes, design)
    choices = _allowed(design, targets_for_depth, depths)
    allowed = {pair: _targets(*pair, ids, settings, fixed_ids) for pair, ids in choices.items()}
    lines = _depth_targets(design, pairs, per_surface, depths, sources, targets_for_depth, names)
    unusable = {surface: frozenset(line.target for line in found if not line.usable) for surface, found in lines.items()}
    presets = tuple(_start_preset(*pair, allowed.get(pair), family, settings, unusable.get(surface_of(*pair), frozenset()), names)
                    if mode == "search" else None for pair, mode in zip(pairs, per_surface, strict=True))
    return SearchPlan(
        surfaces=surfaces, modes=per_surface,
        presets=presets,
        fixed_surfaces={surface: SCHEME_MATERIAL for surface, mode in zip(surfaces, per_surface, strict=True)
                        if mode != "search"},
        max_depths=depths, depth_sources=sources, depth_targets=lines, family=family.id,
        max_steps=request.max_steps if request.max_steps is not None else settings.descent.max_steps, request=request,
        allowed={surface_of(*pair): targets for pair, targets in sorted(allowed.items())},
        ranges={surface_of(*pair): given for pair, given in sorted(ranges.items())})


SearchPlan.model_rebuild()


def fixed_depths(plan: SearchPlan, scheme: Scheme, design: DesignSettings | None,
                 targets: DepthTargets) -> dict[str, dict[str, object]]:
    """開跑預覽「深度上限」那一格給固定區用的（#1082 段 2-9）：跟立面圖同一次限制檢查（``zone_limits``，選了材料庫的
    實測表照底構造算總深那一條也在裡面）的生效上限（``cap``：值與哪一層，沒給是 None）與深度那幾條原句（``lines``）。
    只排給人看，不進計畫（計畫存在工作記錄裡、舊的逐位不變）；保留原牆與交給搜尋的區不列。"""
    result = zone_limits(scheme.model_dump(mode="json"), None if design is None else design.model_dump(mode="json"),
                         depth_targets=lambda: targets)
    caps = cast(dict[str, dict[str, dict[str, object]]], result["caps"])
    zones = cast(dict[str, dict[str, list[dict[str, str]]]], result["zones"])
    found: dict[str, dict[str, object]] = {}
    for surface, mode in zip(plan.surfaces, plan.modes, strict=True):
        if mode != "fixed":
            continue
        wall, zone = zone_of(surface)
        cap = caps.get(wall, {}).get(zone)
        found[surface] = {"cap": None if cap is None else {"value_m": cap["value_m"], "level": cap["level"]},
                          "lines": [{"level": line["level"], "text": line["text"]}
                                    for line in zones.get(wall, {}).get(zone, []) if line["code"].startswith("depth_")]}
    return found


def configuration_arguments(plan: SearchPlan) -> list[str]:
    """開跑那一組起跑配置的命令列參數（起跑樣板、裝法、固定面、深度上限；准用清單另交）：建替身照它核固定材料
    （#1082 段 2-8），寫法跟 :func:`start_arguments` 同一份。"""
    arguments = ["--presets", ",".join("none" if preset is None else preset for preset in plan.presets),
                 "--mount", plan.request.mount]
    for surface, material in plan.fixed_surfaces.items():
        arguments.extend(("--fixed-surface", f"{surface}={material}"))
    for surface, depth in plan.max_depths.items():
        arguments.extend(("--max-depth", f"{surface}={depth!r}"))
    return arguments


def start_arguments(plan: SearchPlan) -> list[str]:
    """``aosr.gradient.cli start`` 裡跟設計有關的那幾格（方案、替身、根目錄等由呼叫端補）。"""
    configuration = configuration_arguments(plan)
    # 順序照舊：起跑樣板、裝法、裝置、固定面、深度上限……
    arguments = [*configuration[:4], "--device", plan.request.device, *configuration[4:]]
    if plan.request.frequency_batch is not None:
        arguments.extend(("--frequency-batch", str(plan.request.frequency_batch)))
    if plan.request.max_steps is not None:
        arguments.extend(("--max-steps", str(plan.request.max_steps)))
    for surface, targets in plan.allowed.items():
        arguments.extend(("--allowed", f"{surface}={','.join(targets)}"))
    # 每一區的可調範圍（#1199）：同綁定只加在這裡（建替身那一份不帶，範圍只收窄、不用重建替身）；寫法跟命令列讀回那一支同一份。
    from aosr.gradient.fixed_surfaces import range_argument

    for surface, families in plan.ranges.items():
        for code, given in families.items():
            arguments.extend(("--range", range_argument(surface, code, given)))
    # 成對的兩片（#1206）：只加在這裡、不進 configuration_arguments（建替身照那一份核固定材料，綁定不改起跑配置）。
    for first, second in plan.ties:
        arguments.extend(("--tie", f"{first}={second}"))
    return arguments
