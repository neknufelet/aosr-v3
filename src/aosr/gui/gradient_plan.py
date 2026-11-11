"""設計設定 → 第三階段梯度搜尋（gradient search，用替身算梯度找材料）命令列參數（#1093）。純函式：不開行程、不寫檔。

每一區的做法照 ``zone_design.effective_modes``（沒寫的區一律固定）：
- 交給搜尋：起跑用選定家族的樣板（``template_preset``），之後換樣板也照搜尋設定；
- 固定這一組材料、保留原牆：兩種都是「用方案檔那一區自己寫的材料、不帶變數、不換樣板」，命令列寫成固定面
  ``面=scheme-material``（方案檔裡固定區放的是那一組材料、保留原牆的區放的是原牆，所以同一條路就對）。
面片的順序照命令列自己的取法（``scheme_patch_keys``），不在這裡另排。

每區准用清單（#1082 段 2-3b，設計設定的 ``allowed``，材料庫代號或 none）轉成命令列 ``--allowed 面=目標,…``：
可最佳化材料是家族代號、固定材料是 ``fixed:<代號>``、none 照寫。有清單的區起點照順序挑：選定家族在清單裡就用它、
不在就用清單裡第一個家族（設定檔順序）、都沒有家族就不貼（清單要有 none）；清單只有固定材料就拒收（起點要是可調材料或不貼）。

這裡只擋網頁講得清楚的事（設計設定對不上方案、一區都沒交給搜尋、家族不認識、深度上限給在不是搜尋的區、准用清單的代號
不認識或排不出起點）；
深度上限放不放得下家族的棉、超不超過家族上限，照舊由命令列在開跑前判，錯誤原文照實回給畫面。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.material_gradient import (
    MOUNT_ON_WALL, NONE_TARGET, SCHEME_MATERIAL, FamilySettings, MaterialGradientSettings, Mount, fixed_target,
)
from aosr.gradient.boundary import scheme_patch_keys, surface_ids
from aosr.gui.zone_design import WHOLE_WALL_KEY, DesignSettings, ZoneMode, effective_modes, mismatches
from aosr.reporting.scheme import Scheme

Device = Literal["cpu", "gpu"]


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


def surface_of(wall: str, zone: str) -> str:
    """設計設定的（牆, 區代號）→ 命令列的面片代號：整面一區是牆名，矩形是「牆@區」。"""
    return wall if zone == WHOLE_WALL_KEY else f"{wall}@{zone}"


def zone_of(surface: str) -> tuple[str, str]:
    wall, separator, zone = surface.partition("@")
    return wall, zone if separator else WHOLE_WALL_KEY


def _depths(request: GradientRequest, modes: Mapping[str, Mapping[str, ZoneMode]]) -> dict[str, float]:
    found = {}
    for wall, zones in request.max_depths.items():
        for zone, depth in zones.items():
            mode = modes.get(wall, {}).get(zone)
            if mode is None:
                raise PlanRefused(f"深度上限給的 {wall} 分區 {zone} 方案裡沒有")
            if mode != "search":
                raise PlanRefused(f"{wall} 分區 {zone} 不是交給搜尋的區，不能給深度上限（固定與保留原牆的區不加層）")
            found[surface_of(wall, zone)] = depth
    return dict(sorted(found.items()))


def _targets(wall: str, zone: str, ids: tuple[str, ...], settings: MaterialGradientSettings,
             fixed_ids: frozenset[str]) -> tuple[str, ...]:
    families = {family.id for family in settings.family}
    unknown = [item for item in ids if item != NONE_TARGET and item not in families and item not in fixed_ids]
    if unknown:
        raise PlanRefused(f"{wall} 分區 {zone} 的准用清單有材料庫裡沒有的代號 {unknown}")
    return tuple(item if item == NONE_TARGET or item in families else fixed_target(item) for item in ids)


def _start_preset(wall: str, zone: str, targets: tuple[str, ...] | None, family: FamilySettings,
                  settings: MaterialGradientSettings) -> str | None:
    if targets is None or family.id in targets:
        return family.template_preset
    listed = [item for item in settings.family if item.id in targets]
    if listed:
        return listed[0].template_preset
    if NONE_TARGET in targets:
        return None
    raise PlanRefused(f"{wall} 分區 {zone} 的准用清單只有固定材料：搜尋要一個起點（一種可調材料或「不貼」），"
                      "請在清單加一種可調材料或不貼")


def _allowed(design: DesignSettings | None, settings: MaterialGradientSettings,
             fixed_ids: frozenset[str]) -> dict[tuple[str, str], tuple[str, ...]]:
    found = {}
    for wall, zones in (design.zones if design is not None else {}).items():
        for zone, item in zones.items():
            if item.allowed is not None:
                found[(wall, zone)] = _targets(wall, zone, item.allowed, settings, fixed_ids)
    return found


def plan_search(scheme: Scheme, design: DesignSettings | None, request: GradientRequest,
                settings: MaterialGradientSettings, fixed_ids: frozenset[str] = frozenset()) -> SearchPlan:
    """``fixed_ids`` 是材料庫裡固定材料的代號（內建構造預設＋使用者的固定材料），准用清單照它認固定材料。"""
    problems = mismatches(design, scheme) if design is not None else []
    if problems:
        raise PlanRefused("設計設定跟方案對不上（" + "；".join(problems) + "），先在立面圖重存每一區的做法")
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
    allowed = _allowed(design, settings, fixed_ids)
    return SearchPlan(
        surfaces=surfaces, modes=per_surface,
        presets=tuple(_start_preset(*pair, allowed.get(pair), family, settings) if mode == "search" else None
                      for pair, mode in zip(pairs, per_surface, strict=True)),
        fixed_surfaces={surface: SCHEME_MATERIAL for surface, mode in zip(surfaces, per_surface, strict=True)
                        if mode != "search"},
        max_depths=_depths(request, modes), family=family.id,
        max_steps=request.max_steps if request.max_steps is not None else settings.descent.max_steps, request=request,
        allowed={surface_of(*pair): targets for pair, targets in sorted(allowed.items())})


def start_arguments(plan: SearchPlan) -> list[str]:
    """``aosr.gradient.cli start`` 裡跟設計有關的那幾格（方案、替身、根目錄等由呼叫端補）。"""
    arguments = ["--presets", ",".join("none" if preset is None else preset for preset in plan.presets),
                 "--mount", plan.request.mount, "--device", plan.request.device]
    for surface, material in plan.fixed_surfaces.items():
        arguments.extend(("--fixed-surface", f"{surface}={material}"))
    for surface, depth in plan.max_depths.items():
        arguments.extend(("--max-depth", f"{surface}={depth!r}"))
    if plan.request.frequency_batch is not None:
        arguments.extend(("--frequency-batch", str(plan.request.frequency_batch)))
    if plan.request.max_steps is not None:
        arguments.extend(("--max-steps", str(plan.request.max_steps)))
    for surface, targets in plan.allowed.items():
        arguments.extend(("--allowed", f"{surface}={','.join(targets)}"))
    return arguments
