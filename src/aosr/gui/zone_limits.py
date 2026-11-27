"""超過限制的區塊標紅、寫出是哪一條擋住（#1111，#1082 第 1-4 段）。

一份方案（可以是網頁上還沒存的草稿）加上它的設計設定，逐面逐區列出哪幾條限制沒過：

- 位置：超出牆面（房間尺寸改小以後最常見）、太小（寬或高不到最小邊）、跟別區重疊；一面沒蓋滿另列在那一面。
  判法跟切牆面（``zone_edit``）同一把：同一個最小邊、同一個邊的容差；蓋滿與不重疊照方案自己那一把精確判法
  （``validate_surface_partition``）。
- 深度：一區的上限取「房間預設／這面牆／這一塊」三層裡最小的那一個（一樣小時取範圍最小的那一層），並記是哪一層擋的；
  固定區貼上去的總深（``MountedOnWall`` 加貼的每一層厚度加總，板與布這類層沒有厚度欄、當 0）超過上限加 ``EDGE_MERGE_TOL`` 就紅（跟搜尋的固定材料候選 ``depth_fit.layers_exceed`` 同一把）。
  分區的固定阻抗與實測表若按物理內容認出材料庫那一筆（:func:`picked_entry`），優先用搜尋的同一把尺判深度。
  裸構造要跟原方案同位置比過才知道是不是選上去的；相同就保留原牆，不同才判深度，六個數字代表六面整面固定阻抗，一定與裸構造不同。
  原牆比不出時，改查設計設定記的選料。紀錄失配仍按方案現有構造判上限，量不出深度要重選；能證明未更換的材料照原判。
  六個數字的固定阻抗與分區存法照同一面牆的阻抗值比對。
  放回原牆另外需要取得材料；六個數字要轉成材料時，放進設計擺位過不了檢查也用不上。
  搜尋區的上限交給搜尋（開跑請求的 ``max_depths``）；保留原牆的區不加層、管不到。
- 搜尋區的上限跟材料組自己的限制取較嚴（#1082 段 2-4，判法只在引擎一份 ``fixed_surfaces.target_fit``）：准用的每一組
  （沒寫清單就是全部可調材料）列出生效上限與是哪一條擋住的（個案哪一層／材料組自己）；比一組最淺排法還淺、或固定材料
  總深超過（量不出多深也算）＝那一組在這一區不可用（列在 ``caps`` 的 ``targets``、不擋，搜尋不換過去）；准用的材料組一組都不可用＝紅
  （「不貼」不算救：交給搜尋卻什麼都貼不上）。清單本來只有「不貼」的區不判。沒傳材料組（``depth_targets``）就不判這一項。
- 每一區自己的可調範圍（#1199，設計設定 ``ZoneDesign.ranges``；判法在引擎一份 ``depth_fit``）：對材料庫核（:func:`zone_range_problems`，
  跟准用清單認代號那一支 :func:`zone_targets` 放在一起）——不是可最佳化材料、超出材料頁的範圍（只准收窄，要放寬到材料頁改）、
  棉厚不是產品厚度、空腔超出或材料不調空腔、收窄後放不進材料自己的總深，都紅；有上限時寫了範圍的那一組照收窄後的判，
  放不下就紅（``depth_fit.ranged_misfits``，開跑前檢查同一把），不像沒寫範圍的組只列「不可用」。沒寫範圍時一個字都不多。
- 可信範圍（固定區的材料參數，跟材料頁同一個出處 ``material_curves.range_check``）：物理拒算的標紅；照算但醒目提醒的
  標「警告」、不擋（計畫段 2-2b 寫的是「超出可信範圍警告」，物理那邊也照算）。沒傳檢查函式就不判這一項；
  網頁的限制檢查、開跑預覽、開跑與建替身都傳材料頁那一支（正式入口開機釘中央處理器＋雙精度）。

紅的會擋「拿去搜尋」與開跑（``refusals``，``gradient_plan.plan_search`` 也照它擋）；警告與說明只顯示。

「鋼板直接貼在棉上」的構造不進程式最佳化（#1208，判斷與字句在 ``reference_only``）：准用清單裡有這種固定材料的區，
:func:`reference_only_refusals` 照清單分組、一組一句列出開跑會拒收的那一句（開跑計畫丟的就是第一句）；限制檢查的介面有才多回一格
``reference_only``，沒有這種構造時回覆一個位元都不多。不算進 ``red``／``refusals``（那兩格照舊只是位置、深度、可信範圍）。
這一支只讀不寫；草稿有一格讀不出來就照實列，不丟錯誤碼（畫面在改到一半時也要能問）。
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.config.material_gradient import NONE_TARGET, SCHEME_MATERIAL, MaterialGradientSettings, centimeters, fixed_target, is_fixed_target
from aosr.geometry.shoebox import Room
from aosr.gradient.depth_fit import TargetFit, ZoneRange, fixed_fit, fixed_material_depth_m, narrowed_family, range_problem, ranged_misfits, target_fit
from aosr.gui.labels import WALLS as WALL_WORDS
from aosr.gui.reference_only import plan_refused, reference_only
from aosr.gui.wall_words import draft_wall_words
from aosr.gui.zone_design import DEFAULT_MODE, WHOLE_WALL_KEY, DesignSettings, ZoneDesign, ZoneMode
from aosr.gui.zone_construction import (
    Bounds, ComparisonMaterial, ComparisonWalls, ConstructionPick, OriginalProblem, OriginalReader, construction_pick, mismatch_words,
)
from aosr.gui.zone_edit import EDGE_MERGE_TOL, MIN_SIDE_M, ZoneCutError, draft_room, too_small
from aosr.materials.construction_models import Assembly, ConstantImpedance, MeasuredImpedance, MountedOnWall, PorousLayer, SurfaceMaterial
from aosr.materials.library import MeasuredTable
from aosr.physics.report_surfaces import RectangleZone, validate_surface_partition
from aosr.physics.zone_fill import FILL_PREFIX, Box, wall_size_m

DepthLevel = Literal["zone", "wall", "room"]
Level = Literal["red", "warn", "note"]
LEVEL_WORDS: dict[DepthLevel, str] = {"zone": "這一塊", "wall": "這面牆", "room": "房間預設"}
WALLS = ("floor", "ceiling", "x0", "xL", "y0", "yL")
_MATERIAL: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
RangeCheck = Callable[[SurfaceMaterial], dict[str, object]]


@dataclass(frozen=True)
class DepthTargets:
    """判一區准用材料組放不放得進上限要的三樣：這一場的搜尋設定（家族，含使用者可最佳化材料）、固定材料（代號 → 材料，
    內建構造與使用者固定材料）與顯示名（代號 → 材料庫的名字，沒有就照代號）。"""

    settings: MaterialGradientSettings
    fixed: Mapping[str, object]
    labels: Mapping[str, str]
    standalone_constructions: frozenset[str] = frozenset()
    wall_dependent: frozenset[str] = frozenset()
    read_original: OriginalReader | None = None
    legacy_replacements: frozenset[str] = frozenset()

    def label(self, identifier: str) -> str:
        return self.labels.get(identifier, identifier)


class CandidateSelection(BaseModel):
    """一區原來的勾選、草稿可調範圍、當下生效的總深上限。"""

    allowed: tuple[str, ...]
    ranges: dict[str, ZoneRange] | None = None
    cap: float | None = None


class EffectiveCandidates(BaseModel):
    allowed: tuple[str, ...]
    covered: dict[str, str]


def effective_candidates(selection: CandidateSelection, targets: DepthTargets) -> EffectiveCandidates:
    """畫面與開跑共用；設計檔保留原勾選，以便移除家族時回復固定款。"""
    covered: dict[str, str] = {}
    for family in targets.settings.family:
        if family.id not in selection.allowed:
            continue
        given = (selection.ranges or {}).get(family.id)
        bounds = narrowed_family(family, given) if given else family
        cap = min(family.max_total_depth_m, selection.cap if selection.cap is not None else family.max_total_depth_m)
        for member in family.presets:
            material = targets.fixed.get(member)
            if not isinstance(material, Assembly) or member == family.id:
                continue
            wool = sum(layer.thickness_m for layer in material.layers if layer.kind == "porous")
            air = sum(layer.thickness_m for layer in material.layers if layer.kind == "air")
            cavity = bounds.cavity_range_m if bounds.initial_cavity_m is not None else (0.0, 0.0)
            if wool in bounds.thicknesses_m and cavity[0] <= air <= cavity[1] and wool + air <= cap + 1e-12:
                covered[member] = family.id
    excluded = set(covered)
    if selection.cap is not None:
        _, valid = zone_range_problems(selection.ranges, targets)
        fits = target_fits(selection.allowed, selection.cap, targets, valid)
        excluded.update(fit.target.removeprefix("fixed:") for fit in fits or () if not fit.usable)
    return EffectiveCandidates(allowed=tuple(code for code in selection.allowed if code not in excluded), covered=covered)


@dataclass(frozen=True)
class DepthCap:
    """一區實際的深度上限：值（公尺）與是哪一層擋的。"""

    value_m: float
    level: DepthLevel
    # 背後換成材料庫一筆時那一筆的總深（#1330）：``value_m`` 已經扣掉它，是前面那幾層能用的深度；沒換是 0。
    backing_m: float = 0.0

    def words(self) -> str:
        if self.backing_m > 0:
            return f"{self.value_m:g} 公尺（{LEVEL_WORDS[self.level]}的上限扣掉背後 {self.backing_m:g} 公尺）"
        return f"{self.value_m:g} 公尺（{LEVEL_WORDS[self.level]}的上限）"


class Problem(BaseModel):
    """一條沒過的限制：代號、白話、輕重（紅＝擋、警告與說明只顯示）。"""

    model_config = ConfigDict(frozen=True)
    code: str
    level: Level
    text: str


def depth_cap(design: DesignSettings | None, wall: str, zone: str) -> DepthCap | None:
    """三層取最小；一樣小時取範圍最小的那一層（這一塊 → 這面牆 → 房間預設）。都沒給就是 None。"""
    if design is None:
        return None
    item = design.zones.get(wall, {}).get(zone)
    given: list[tuple[float, DepthLevel]] = []
    if item is not None and item.max_depth_m is not None:
        given.append((item.max_depth_m, "zone"))
    if wall in design.wall_max_depth_m:
        given.append((design.wall_max_depth_m[wall], "wall"))
    if design.room_max_depth_m is not None:
        given.append((design.room_max_depth_m, "room"))
    if not given:
        return None
    value, level = min(given, key=lambda pair: pair[0])
    return DepthCap(value, level)


def backing_depth_m(design: DesignSettings | None, wall: str, zone: str, material: object) -> float | None:
    """這一區背後換成材料庫一筆時（#1330，設計設定有 ``backing_entry``）那一筆的總深；沒換回 None。
    量的是方案那一區現在的材料（換上去的就是它，搜尋也照它當背材），量法同固定材料（``fixed_material_depth_m``，
    沒有厚度欄的層計零；量不出來當 0）。"""
    item = design.zones.get(wall, {}).get(zone) if design is not None else None
    if item is None or item.mode != "search" or item.backing_entry is None:
        return None
    return fixed_material_depth_m(material) or 0.0


def front_cap(design: DesignSettings | None, wall: str, zone: str, material: object) -> DepthCap | None:
    """搜尋那幾層實際能用的上限：「最深」從原牆往房間量、含背後換上去的那一筆，所以扣掉它（#1330）；
    最深比背後那一筆還淺時回 None：前面那幾層不必再判，那一區由 :func:`backing_problems` 那一條紅字擋、只講那一句。
    沒換背後就是 :func:`depth_cap`。"""
    cap = depth_cap(design, wall, zone)
    backing = backing_depth_m(design, wall, zone, material)
    if cap is None or backing is None or backing <= 0:
        return cap
    if backing > cap.value_m + EDGE_MERGE_TOL:
        return None
    return DepthCap(max(cap.value_m - backing, 0.0), cap.level, backing)


def backing_problems(design: DesignSettings | None, wall: str, zone: str, material: object,
                     label: Callable[[str], str]) -> list[Problem]:
    """最深放不下背後那一筆就擋（#1330）：前面一層都還沒放，最深已經比背後那一筆淺。"""
    cap = depth_cap(design, wall, zone)
    backing = backing_depth_m(design, wall, zone, material)
    if cap is None or backing is None or backing <= cap.value_m + EDGE_MERGE_TOL:
        return []
    item = design.zones[wall][zone] if design is not None else None
    name = label(item.backing_entry) if item is not None and item.backing_entry is not None else "背後那一筆"
    return [Problem(code="depth_backing", level="red",
                    text=f"最深 {cap.value_m:g} 公尺（{LEVEL_WORDS[cap.level]}的上限）比背後的「{name}」{backing:g} 公尺還淺，放不下；"
                         "把最深加大，或背後換回原牆")]


def added_depth(material: SurfaceMaterial) -> float | None:
    """貼上去的總深（公尺）：只算得出貼在原牆上的構造；直接取代牆面的材料回 None（看不出貼了多深）。"""
    if not isinstance(material, MountedOnWall):
        return None
    return math.fsum(float(getattr(layer, "thickness_m", 0.0)) for layer in material.layers)


def _number_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    if not all(isinstance(n, int | float) and not isinstance(n, bool) and math.isfinite(n) for n in value):
        return None
    return float(value[0]), float(value[1])


def zone_words(wall: str, key: str, names: Mapping[str, str] = WALL_WORDS) -> str:
    """一區在畫面上叫什麼（擋下的原文給人看，不露內部代號）：牆名照表單（``names``，方案驗得過時是
    ``wall_words`` 排的前後左右，否則 ``labels.WALLS``），區名同逐區表——整面、自動補的寫「自動 代號」、畫的框寫它的代號。"""
    zone = "整面" if key == WHOLE_WALL_KEY else f"自動 {key}" if key.startswith(FILL_PREFIX) else key
    return f"{names.get(wall, wall)}「{zone}」"


def _zone_order(scheme: Mapping[str, object]) -> list[tuple[str, str]]:
    """方案裡現在有的區，照限制檢查自己逐面逐區走的順序（:data:`WALLS` 的面序，同一面照方案裡那一面的區序）。
    草稿是壞的時候哪些區算在方案裡：某一面的區清單不是清單（例如是數字）→ 那一面沒有區；某一區不是物件（例如是字串）→
    那一區不算（矩形沒有字串代號的也不算）；某一面沒寫或是空清單 → 整面一區。"""
    scene = scheme.get("scene")
    by_wall = scene.get("surface_zones_by_wall") if isinstance(scene, Mapping) else None
    found: dict[tuple[str, str], None] = {}
    for wall in WALLS:
        entries = by_wall.get(wall) if isinstance(by_wall, Mapping) else None
        if entries is None or entries == []:
            entries = [{"area": "whole_wall"}]
        for entry in entries if isinstance(entries, list) else []:
            key = _key(entry) if isinstance(entry, Mapping) else None
            if key is not None:
                found[wall, key] = None
    return list(found)


def reference_only_refusals(scheme: Mapping[str, object], design: DesignSettings | None, targets: DepthTargets,
                            names: Mapping[str, str]) -> list[dict[str, object]]:
    """准用清單原文裡的「鋼板直接貼在棉上」的固定材料（#1208）：照（哪一筆, 那一區清單去重排序）分組，一組一句
    （分區頁的清單卡也是照清單去重排序分的，同一張卡的區在同一組）。每一組回 ``id``、``zones``（``{wall, zone}``）與
    ``text``（開跑會拒收的那一句，區名用「、」接）。只收方案裡現在有的區（:func:`_zone_order`；設計設定裡方案沒有的區
    不進任何一組、不出現在句子裡），區照那一支的順序，組照組裡第一區；同一份清單有兩筆命中時，兩組照第一區那份清單裡
    寫的先後。設計設定檔裡區寫的先後不影響結果。清單原文＝設計設定存的那一份，放不放得下深度都算。"""
    zones = design.zones if design is not None else {}
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[str, str]]] = {}
    for wall, zone in _zone_order(scheme):
        item = zones.get(wall, {}).get(zone)
        allowed = item.allowed if item is not None and item.allowed else []
        for identifier in dict.fromkeys(allowed):
            if identifier not in targets.standalone_constructions and reference_only(targets.fixed.get(identifier)):
                groups.setdefault((identifier, tuple(sorted(set(allowed)))), []).append((wall, zone))
    return [{"id": identifier, "zones": [{"wall": wall, "zone": zone} for wall, zone in listed],
             "text": plan_refused("、".join(zone_words(wall, zone, names) for wall, zone in listed), targets.label(identifier))}
            for (identifier, _), listed in groups.items()]


def _key(entry: Mapping[str, object]) -> str | None:
    if entry.get("area") == "rectangle":
        identifier = entry.get("id")
        return identifier if isinstance(identifier, str) else None
    return WHOLE_WALL_KEY


def _overlap(first: Box, second: Box) -> bool:
    du = min(first.u_m[1], second.u_m[1]) - max(first.u_m[0], second.u_m[0])
    dv = min(first.v_m[1], second.v_m[1]) - max(first.v_m[0], second.v_m[0])
    return du > EDGE_MERGE_TOL and dv > EDGE_MERGE_TOL


def _place_problems(room: Room, wall: str, entries: Sequence[Mapping[str, object]],
                    zones: dict[str, list[Problem]], walls: dict[str, list[Problem]]) -> None:
    width, height = wall_size_m(room, wall)
    boxes: list[Box] = []
    for entry in entries:
        if entry.get("area") != "rectangle":
            continue
        key, u, v = _key(entry), _number_pair(entry.get("u_m")), _number_pair(entry.get("v_m"))
        if key is None or u is None or v is None:
            walls[wall].append(Problem(code="unreadable", level="red", text="這一面有一區的代號或座標讀不出來；請重新打開方案或清掉這一面的切法"))
            continue
        box = Box(id=key, u_m=u, v_m=v)
        found = zones.setdefault(key, [])
        if not (-EDGE_MERGE_TOL <= u[0] < u[1] <= width + EDGE_MERGE_TOL and -EDGE_MERGE_TOL <= v[0] < v[1] <= height + EDGE_MERGE_TOL):
            found.append(Problem(code="outside", level="red",
                                 text=f"超出牆面：這一面現在寬 {width:g}、高 {height:g} 公尺（房間尺寸改過）；請改框或清掉這一面的切法"))
        elif too_small(box):
            found.append(Problem(code="too_small", level="red", text=f"太小：寬或高不到最小邊 {MIN_SIDE_M * 100:g} 公分"))
        for other in boxes:
            if _overlap(box, other):
                found.append(Problem(code="overlap", level="red", text=f"跟 {other.id} 重疊"))
                zones.setdefault(other.id, []).append(Problem(code="overlap", level="red", text=f"跟 {box.id} 重疊"))
        boxes.append(box)
    if not boxes or any(problem.level == "red" for key in (box.id for box in boxes) for problem in zones.get(key, [])) \
            or walls[wall]:
        return
    try:
        validate_surface_partition(room, {wall: tuple(RectangleZone.model_construct(
            area="rectangle", id=box.id, u_m=box.u_m, v_m=box.v_m) for box in boxes)})
    except ValidationError:
        walls[wall].append(Problem(code="not_covered", level="red", text="這一面的分區沒有正好蓋滿整面（留了空）；請重切這一面"))


def _zone_bounds(room: Room, wall: str, entry: Mapping[str, object]) -> Bounds | None:
    if entry.get("area") != "rectangle":
        width, height = wall_size_m(room, wall)
        return (0.0, width), (0.0, height)
    u, v = _number_pair(entry.get("u_m")), _number_pair(entry.get("v_m"))
    return None if u is None or v is None else (u, v)


def picked_entry(material: SurfaceMaterial, targets: DepthTargets) -> str | None:
    """固定阻抗與實測表照物理內容認材料庫代號。裸構造另跟原方案同位置比較。"""
    if isinstance(material, (MountedOnWall, Assembly)):
        return None
    for identifier, entry in targets.fixed.items():
        physics = entry.to_physics() if isinstance(entry, MeasuredTable) else entry
        if physics == material:
            return identifier
    return None


def _picked_problems(identifier: str, cap: DepthCap, targets: DepthTargets) -> list[Problem]:
    """固定區選了材料庫直接取代牆面的那一筆：照搜尋的固定材料候選同一把判（``fixed_fit``，實測表照底構造算總深），
    句子同段 2-4 那一行（``target_line``）。"""
    fit = fixed_fit(fixed_target(identifier), targets.fixed[identifier], cap.value_m)
    text = str(target_line(fit, cap, targets)["text"])
    if fit.usable:
        return [Problem(code="depth_fits", level="note", text=text)]
    if fit.reason == "fixed_not_comparable":
        return [Problem(code="depth_unknown", level="red", text=f"{text}；這一區的上限 {cap.words()}")]
    return [Problem(code="depth_over", level="red", text=text)]


def _saved_depth_problems(material: ComparisonMaterial, cap: DepthCap) -> list[Problem]:
    fit = fixed_fit(SCHEME_MATERIAL, material, cap.value_m)
    if fit.material_m is None:
        reason = ("這一區存的是固定阻抗，從阻抗值看不出加貼厚度" if isinstance(material, (ConstantImpedance, float))
                  else "看不出這一區現在的材料有多深")
        return [Problem(code="depth_unknown", level="red",
                        text=f"{reason}，無法確認是否符合上限 {cap.words()}。請回「這間房貼什麼」重選一筆材料")]
    depth = f"這一區現在的構造總深 {fit.material_m:g} 公尺"
    if fit.usable:
        return [Problem(code="depth_fits", level="note", text=f"{depth}，放得進深度上限 {cap.words()}")]
    return [Problem(code="depth_over", level="red", text=f"{depth}，超過深度上限 {cap.words()}")]


def _depth_problems(material: ComparisonMaterial, mode: ZoneMode, cap: DepthCap | None,
                    targets: Callable[[], DepthTargets | None] = lambda: None, *, construction: ConstructionPick | None = None) -> list[Problem]:
    """固定區照已判的裸構造身分給深度與原因；其他直接取代牆面的材料有上限時才查。"""
    if mode != "fixed":
        return []
    # 有上限的裸構造也要能查材料庫；讀取失敗由 required 記進拒收理由。
    if cap is not None and isinstance(material, (Assembly, ConstantImpedance, MeasuredImpedance)):
        loaded = targets()
        if loaded is not None and (picked := picked_entry(material, loaded)) is not None:
            return _picked_problems(picked, cap, loaded)
    if construction is not None and construction.comparison.state == "original":
        return []
    if construction is not None and construction.identifier is not None and not construction.mismatch:
        identifier, comparison = construction.identifier, construction.comparison
        loaded = targets()
        if loaded is None:
            return []
        if comparison.state == "picked":
            return _picked_problems(identifier, cap, loaded) if cap else []
        fit = fixed_fit(fixed_target(identifier), loaded.fixed[identifier], cap.value_m if cap else math.inf)
        relation = "放得進" if fit.usable else "超過"
        limit = f"{relation}深度上限 {cap.words()}" if cap else "這一區沒設深度上限"
        return [Problem(code="depth_origin_unknown", level="note",
                        text=f"看不出：跟材料庫「{loaded.label(identifier)}」一樣，總深 {fit.material_m:g} 公尺，{limit}；{comparison.reason}")]
    if cap is None:
        return []
    depth = added_depth(material) if isinstance(material, MountedOnWall) else None
    if depth is None:
        if construction is not None:
            # 有選料紀錄或已比出跟原牆不同，直接量方案裡保存的實物。
            return _saved_depth_problems(material, cap)
        if isinstance(material, float):
            return [Problem(code="depth_not_comparable", level="note", text=f"這面牆只記了固定阻抗值，沒有加貼厚度可與上限 {cap.words()} 比對")]
        return [Problem(code="depth_not_comparable", level="note",
                        text=f"這一區的材料直接取代牆面、看不出貼了多深，上限 {cap.words()} 比不了")]
    if depth > cap.value_m + EDGE_MERGE_TOL:
        return [Problem(code="depth_over", level="red",
                        text=f"貼上去總深 {depth:g} 公尺，超過深度上限 {cap.words()}")]
    return []


def _default_candidates(targets: DepthTargets) -> tuple[str, ...]:
    return (*(family.id for family in targets.settings.family), NONE_TARGET)


def zone_targets(allowed: Sequence[str] | None, targets: DepthTargets) -> tuple[str, ...] | None:
    """一區可選的目標（引擎的寫法：家族代號、``fixed:<代號>``、none）：沒寫清單是全部家族加不貼；清單有不認識的代號回 None
    （開跑那邊照講拒收，這裡不猜）。"""
    families = {family.id for family in targets.settings.family}
    if allowed is None:
        return _default_candidates(targets)
    found: list[str] = []
    for item in allowed:
        if item == NONE_TARGET or item in families:
            found.append(item)
        elif item in targets.fixed:
            found.append(fixed_target(item))
        else:
            return None
    return tuple(found)


def zone_range_problems(ranges: Mapping[str, ZoneRange] | None, targets: DepthTargets) -> tuple[list[Problem], dict[str, ZoneRange]]:
    """一區的可調範圍對材料庫核（#1199，判法 ``depth_fit.range_problem``）：回紅的那幾條與核過的那幾種（給深度判法用）。
    代號要是這一場的可最佳化材料（家族代號就是材料庫代號）；不合的每一種一條紅，句子點名是哪一種材料。"""
    families = {family.id: family for family in targets.settings.family}
    problems: list[Problem] = []
    valid: dict[str, ZoneRange] = {}
    for code, given in (ranges or {}).items():
        name = targets.label(code)
        family = families.get(code)
        if family is None:
            text = (f"{name}是固定成品，沒有可調範圍（範圍只給可最佳化材料）" if code in targets.fixed
                    else f"可調範圍的材料 {code} 材料庫裡沒有")
            if code in targets.wall_dependent:
                text = f"{name}的背後要接這一區的原牆，不能整塊當候選，也沒有可調範圍"
            problems.append(Problem(code="range_unknown", level="red", text=text))
            continue
        problem = range_problem(family, given, targets.settings.export)
        if problem is not None:
            problems.append(Problem(code="range_invalid", level="red", text=f"{name}的可調範圍：{problem}"))
            continue
        valid[code] = given
    return problems, valid


def target_line(fit: TargetFit, cap: DepthCap, targets: DepthTargets, *, level_words: str | None = None) -> dict[str, object]:
    """一組在一區的判決給畫面：代號、名字、可不可用、生效上限、哪一條擋住（``case`` 個案／``material`` 材料組自己）與一句白話。"""
    identifier = fit.target[len("fixed:"):] if is_fixed_target(fit.target) else fit.target
    name = targets.label(identifier)
    case = f"{cap.value_m:g} 公尺（{level_words or LEVEL_WORDS[cap.level]}的上限）"
    if fit.reason == "shallower_than_family":
        text = f"{name}：不可用——深度上限 {case}比這一組最淺排法 {fit.shallowest_m:g} 公尺還淺（個案上限擋住）"
    elif fit.reason == "fixed_too_deep":
        text = f"{name}：不可用——總深 {fit.material_m:g} 公尺超過深度上限 {case}（個案上限擋住）"
    elif fit.reason == "fixed_not_comparable":
        text = f"{name}：不可用——直接取代牆面、看不出多深，有深度上限的區不能用（限制是硬的）"
    elif is_fixed_target(fit.target):
        text = f"{name}：總深 {fit.material_m:g} 公尺，放得進深度上限 {case}"
    elif fit.blocker == "material":
        text = f"{name}：生效 {fit.effective_m:g} 公尺（材料組自己的上限擋住；{case}比較寬）"
    else:
        own = "" if fit.material_m is None or fit.material_m <= cap.value_m else f"；材料組自己 {fit.material_m:g} 公尺比較寬"
        text = f"{name}：生效 {fit.effective_m:g} 公尺（{level_words or LEVEL_WORDS[cap.level]}的上限擋住{own}）"
    return {"target": fit.target, "label": name, "usable": fit.usable, "effective_m": fit.effective_m,
            "blocker": fit.blocker, "text": text}


def target_fits(allowed: Sequence[str] | None, cap: float, targets: DepthTargets,
                ranges: Mapping[str, ZoneRange] | None = None) -> tuple[TargetFit, ...] | None:
    """一區每一個材料組（不含不貼）在上限 ``cap`` 下的判決；清單有不認識的代號回 None。
    ``ranges`` 是這一區核過的可調範圍（:func:`zone_range_problems` 回的）：有範圍的組照收窄後的判。"""
    listed = zone_targets(allowed, targets)
    if listed is None:
        return None
    fixed = {fixed_target(key): material for key, material in targets.fixed.items()}
    return tuple(target_fit(target, cap, targets.settings, fixed, ranges) for target in listed if target != NONE_TARGET)


def no_group_words(lines: Sequence[Mapping[str, object]]) -> str:
    """准用的材料組一組都放不下時擋下的那一句（頁面紅、開跑擋同一句）。"""
    return "程式能選的材料沒有一種放得進深度上限，這一區什麼都貼不上（「不貼」不算）：" + "；".join(str(line["text"]) for line in lines)


def ranged_misfit_words(fit: TargetFit, cap: DepthCap, targets: DepthTargets, *, level_words: str | None = None) -> str:
    """寫了範圍、收窄後放不下的那一組擋下的那一句（頁面紅、網頁開跑擋同一句）。``fit`` 是 ``depth_fit.ranged_misfits``
    挑出來的那一組（家族、比最淺排法淺）。寫法同 :func:`target_line` 那一句，數字換成公分（:func:`~aosr.config.material_gradient.
    centimeters`，第 2 支的範圍格子是公分）；沒寫範圍的組那一句照舊是公尺。"""
    case = f"{centimeters(cap.value_m)} cm（{level_words or LEVEL_WORDS[cap.level]}的上限）"
    return (f"{targets.label(fit.target)}：不可用——深度上限 {case}比這一組最淺排法 {centimeters(cast(float, fit.shallowest_m))} cm "
            "還淺（個案上限擋住）；這是照這一區的可調範圍收窄後的最淺排法（最薄棉加空腔下界），請調小範圍的下限或放寬上限")


def _target_problems(allowed: Sequence[str] | None, cap: DepthCap, targets: DepthTargets,
                     ranges: Mapping[str, ZoneRange] | None = None) -> tuple[list[Problem], list[dict[str, object]]]:
    fits = target_fits(allowed, cap.value_m, targets, ranges)
    if fits is None or not fits:
        return [], []
    lines = [target_line(fit, cap, targets) for fit in fits]
    problems = [Problem(code="range_depth", level="red", text=ranged_misfit_words(fit, cap, targets))
                for fit in ranged_misfits(fits, ranges)]
    if not any(fit.usable for fit in fits):
        return [*problems, Problem(code="depth_no_group", level="red", text=no_group_words(lines))], lines
    return problems, lines


def _search_problems(item: ZoneDesign | None, cap: DepthCap | None, targets: Callable[[], DepthTargets | None]
                     ) -> tuple[list[Problem], list[dict[str, object]]]:
    """交給搜尋的一區：寫了可調範圍就對材料庫核（#1199），有上限時准用的每一組放不放得下（有範圍的組照收窄後的判）。
    材料組（``targets``）只在寫了範圍或有上限時才讀。"""
    ranges = None if item is None else item.ranges
    problems: list[Problem] = []
    if ranges and (found := targets()) is not None:
        problems, ranges = zone_range_problems(ranges, found)
    if cap is None or (found := targets()) is None:
        return problems, []
    depth, lines = _target_problems(None if item is None else item.allowed, cap, found, ranges)
    return [*problems, *depth], lines


def range_target(material: SurfaceMaterial) -> SurfaceMaterial | None:
    """可信範圍要查的那一份：貼在原牆上的只查加貼的那幾層（當一份多層構造）；沒有小松纖維層的不必查（沒有登記範圍）。"""
    # 參數範圍只看層本身，背後接什麼不影響；跟材料頁的初值排法一樣當硬背。
    target: SurfaceMaterial = Assembly(layers=material.layers, backing="rigid") if isinstance(material, MountedOnWall) else material
    layers = (target.base_assembly.layers if isinstance(target, MeasuredImpedance)
              else target.layers if isinstance(target, Assembly) else ())
    if not any(isinstance(layer, PorousLayer) and layer.model == "komatsu" for layer in layers):
        return None
    return target


def _range_problems(check: RangeCheck, material: SurfaceMaterial) -> list[Problem]:
    target = range_target(material)
    if target is None:
        return []
    result = check(target)
    status = result.get("status")
    if status == "uncomputable":
        return [Problem(code="range_uncomputable", level="red", text=f"材料參數算不出來：{result.get('message')}")]
    if status == "no_precision":
        return [Problem(code="range_unchecked", level="note", text="這個網頁行程沒有開雙精度，這一跑查不了可信範圍")]
    strong = cast(list[dict[str, object]], result.get("warnings", []))
    return [Problem(code=str(line["code"]), level="warn", text=f"警告（不擋）：{line['text']}")
            for line in strong if line.get("level") == "strong"]


def _design_of(value: object, scheme_id: str) -> tuple[DesignSettings | None, list[Problem]]:
    if value is None:
        return None, []
    try:
        return DesignSettings.model_validate(value), []
    except ValidationError as exc:
        return DesignSettings(scheme_id=scheme_id), [Problem(code="design_invalid", level="red",
                                                             text=f"設計設定格式不對：{exc.error_count()} 處（深度上限要填大於 0 的數字）")]


def zone_limits(document: Mapping[str, object], design_value: object = None,
                range_check: RangeCheck | None = None,
                depth_targets: Callable[[], DepthTargets] | None = None, *, include_candidate_rows: bool = False) -> dict[str, object]:
    """一份方案（草稿也可以）逐面逐區的限制檢查；回 ``zones``（牆 → 區 → 幾條）、``walls``（整面的）、``caps``（每區生效上限；
    搜尋區另有 ``targets``：每一組的生效上限與哪一條擋住）、``red``（擋下的條數）與 ``refusals``（擋下的每一條，給開跑與
    「拿去搜尋」照講）。``depth_targets`` 用來查裸構造的身分（即使沒有深度上限），也查固定區和搜尋區的深度、可調範圍。
    這次逐區檢查最多讀一次材料庫。"""
    scheme_id = str(document.get("scheme_id", ""))
    design, top = _design_of(design_value, scheme_id)
    try:
        room = draft_room(dict(document))
    except ZoneCutError as exc:
        return {"status": "room", "message": str(exc), "zones": {}, "walls": {}, "caps": {}, "red": 0, "refusals": []}
    scene = cast(Mapping[str, object], document.get("scene") or {})
    by_wall = scene.get("surface_zones_by_wall")
    impedances = scene.get("impedance_pa_s_per_m_by_wall") if by_wall is None else None
    walls: dict[str, list[Problem]] = {wall: [] for wall in WALLS}
    zones: dict[str, dict[str, list[Problem]]] = {}
    caps: dict[str, dict[str, dict[str, object]]] = {}
    checked: dict[str, list[Problem]] = {}
    catalog = _ZoneCatalog(depth_targets, design.original_scheme_id if design else None, top)

    for wall in WALLS:
        entries = cast(list[Mapping[str, object]], by_wall.get(wall) if isinstance(by_wall, dict) else None) or [{"area": "whole_wall"}]
        if not isinstance(entries, list) or not all(isinstance(entry, Mapping) for entry in entries):
            walls[wall].append(Problem(code="unreadable", level="red", text="這一面的分區讀不出來；請重新打開方案"))
            continue
        number = impedances.get(wall) if isinstance(impedances, Mapping) else None
        folded = float(number) if isinstance(number, (int, float)) and not isinstance(number, bool) and 0 < number <= sys.float_info.max else None
        found: dict[str, list[Problem]] = {}
        _place_problems(room, wall, entries, found, walls)
        for entry in entries:
            key = _key(entry) if isinstance(entry, Mapping) else None
            if key is None:
                continue
            setting = design.zones.get(wall, {}).get(key) if design else None
            mode = setting.mode if setting else DEFAULT_MODE
            material = _material(entry.get("material"))
            cap = front_cap(design, wall, key, material)
            if cap is not None and mode != "keep_original":
                caps.setdefault(wall, {})[key] = {"value_m": cap.value_m, "level": cap.level, "words": cap.words(),
                                                  **({"backing_m": cap.backing_m} if cap.backing_m > 0 else {})}
            current = folded if folded is not None else material
            lines = found.setdefault(key, [])
            construction = catalog.construction(room, wall, key, entry, current, setting, lines)
            lines.extend(_depth_problems(current, mode, cap, catalog.required, construction=construction))
            if mode == "search":
                catalog.record_candidates(wall, key, setting, cap, include_candidate_rows)
                problems, lines_by_group = _search_problems(setting, cap, catalog.required)
                lines.extend([*backing_problems(design, wall, key, material, catalog.label), *problems])
                if lines_by_group:
                    caps[wall][key]["targets"] = lines_by_group
            if range_check is not None and mode == "fixed" and material is not None:
                text = json.dumps(material.model_dump(mode="json"), sort_keys=True)
                if text not in checked:
                    checked[text] = _range_problems(range_check, material)
                lines.extend(checked[text])
        zones[wall] = {key: lines for key, lines in found.items() if lines}
    refusals, names = [problem.text for problem in top], draft_wall_words(document)
    refusals += [f"{names.get(wall, wall)}：{problem.text}" for wall, lines in walls.items() for problem in lines if problem.level == "red"]
    refusals += [f"{zone_words(wall, key, names)}：{problem.text}" for wall, items in zones.items() for key, lines in items.items()
                 for problem in lines if problem.level == "red"]
    return {**({"candidates": catalog.candidates} if catalog.candidates else {}),
            **({"picked_constructions": catalog.picked} if catalog.picked else {}), "status": "checked", "design": [problem.model_dump() for problem in top],
            "zones": {wall: {key: [problem.model_dump() for problem in lines] for key, lines in items.items()}
                      for wall, items in zones.items() if items},
            "walls": {wall: [problem.model_dump() for problem in lines] for wall, lines in walls.items() if lines},
            "caps": caps, "red": len(refusals), "refusals": refusals}


@dataclass
class _ZoneCatalog:
    """一份草稿或開跑計畫共用的查詢。逐區查身分時，材料庫與原方案各只讀一次。"""

    provider: Callable[[], DepthTargets] | None
    original_id: str | None
    problems: list[Problem]
    picked: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    candidates: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    load_error: Problem | None = None

    def record_candidates(self, wall: str, key: str, setting: ZoneDesign | None, cap: DepthCap | None, include_candidate_rows: bool) -> None:
        if setting is None or (setting.allowed is None and not include_candidate_rows) or self.targets is None:
            return
        allowed = setting.allowed if setting.allowed is not None else _default_candidates(self.targets)
        selection = CandidateSelection(allowed=allowed, ranges=setting.ranges, cap=cap.value_m if cap else None)
        effective = effective_candidates(selection, self.targets)
        if include_candidate_rows or effective.allowed != allowed:
            self.candidates.setdefault(wall, {})[key] = effective.model_dump()

    @cached_property
    def targets(self) -> DepthTargets | None:
        try:
            return self.provider() if self.provider else None
        except ValueError as error:
            self.load_error = Problem(code="materials_unreadable", level="red",
                                      text=f"查不了由程式最佳化的區放不放得下清單裡的材料、也認不出指定貼一種的區是不是選了材料庫的材料：{error}")
            return None

    def label(self, identifier: str) -> str:
        """材料庫那一筆的名字；材料庫讀不進來就寫代號。"""
        return self.targets.label(identifier) if self.targets is not None else identifier

    def required(self) -> DepthTargets | None:
        targets = self.targets
        if self.load_error:
            self.problems.append(self.load_error)
            self.load_error = None
        return targets

    @cached_property
    def original(self) -> ComparisonWalls | None:
        if not self.original_id:
            return None
        if self.targets and self.targets.read_original:
            return self.targets.read_original(self.original_id)
        return OriginalProblem(f"指的原方案「{self.original_id}」讀不進來")

    @cached_property
    def standalone(self) -> dict[str, object]:
        targets = self.targets
        return {} if targets is None else {key: value for key, value in targets.fixed.items() if key in targets.standalone_constructions}

    def construction(self, room: Room, wall: str, key: str, entry: Mapping[str, object],
                     material: ComparisonMaterial, setting: ZoneDesign | None,
                     lines: list[Problem]) -> ConstructionPick | None:
        recorded = setting.picked_entry if setting else None
        if not isinstance(material, Assembly) and recorded is None:
            return None
        targets = self.required() if recorded else self.targets
        if targets is None and recorded is None:
            return None
        standalone = self.standalone
        if targets and recorded in targets.legacy_replacements and recorded is not None:
            standalone = {**standalone, recorded: targets.fixed[recorded]}
        pick = construction_pick(material, standalone, recorded, lambda: self.original,
                                 room, wall, key, _zone_bounds(room, wall, entry))
        if pick is None:
            return None
        if pick.mismatch:
            label = targets.label(pick.mismatch) if targets else pick.mismatch
            lines.append(Problem(code="depth_pick_mismatch", level="note", text=mismatch_words(label)))
        if pick.identifier is not None and targets is not None:
            self.picked.setdefault(wall, {})[key] = {"state": pick.comparison.state, "id": pick.identifier, "label": targets.label(pick.identifier)}
        return pick


def _material(value: object) -> SurfaceMaterial | None:
    try:
        return _MATERIAL.validate_python(value)
    except ValidationError:
        return None


def saved_refusals(scheme_document: Mapping[str, object], design: DesignSettings | None,
                   range_check: RangeCheck | None = None, depth_targets: DepthTargets | None = None) -> list[str]:
    """存好的方案開搜尋（或建替身）前要擋的那幾條：跟立面圖標紅同一把；``range_check`` 給了就連可信範圍一起判
    （網頁開跑與建替身傳材料頁那一支，頁面紅的、伺服器就擋）；``depth_targets`` 給了就連准用材料組放不放得下一起判。"""
    value = None if design is None else design.model_dump(mode="json")
    provider = None if depth_targets is None else (lambda: depth_targets)
    return cast(list[str], zone_limits(scheme_document, value, range_check, provider)["refusals"])


def _draft_reference_only(scheme: Mapping[str, object], design_value: object,
                          depth_targets: Callable[[], DepthTargets] | None) -> list[dict[str, object]]:
    """草稿那一份的 :func:`reference_only_refusals`。沒有一區寫准用清單就不讀材料庫；設計設定格式不對、材料庫讀不進來
    都回空的（那兩件事限制檢查自己會講）。"""
    if depth_targets is None or design_value is None:
        return []
    try:
        design = DesignSettings.model_validate(design_value)
    except ValidationError:
        return []
    if not any(item.allowed for zones in design.zones.values() for item in zones.values()):
        return []
    try:
        targets = depth_targets()
    except ValueError:
        return []
    return reference_only_refusals(scheme, design, targets, draft_wall_words(scheme))


def zone_limit_routes(range_check: RangeCheck | None,
                      depth_targets: Callable[[], DepthTargets] | None = None,
                      symmetry: Callable[[object, object], dict[str, object]] | None = None) -> list[Route]:
    """``POST /api/zone-limits``：收 ``{"scheme": 草稿, "design": 設計設定或 null}``，內容有問題一律回 200 照講。
    ``symmetry`` 給了就順便回左右對稱的配對（#1206，鍵 ``symmetry``；算法在 ``zone_symmetry``，網頁只照著畫）：
    分區頁每次改動後本來就會問這一支，配對跟著同一份草稿、同一個世代號回來。
    分區畫面傳 include_candidate_rows 取得每個搜尋區算好的候選；其他呼叫沿用只傳清單有變的區。"""

    def judged(scheme: dict[str, object], design: object, include_candidate_rows: bool) -> dict[str, object]:
        result = zone_limits(scheme, design, range_check, depth_targets, include_candidate_rows=include_candidate_rows)
        refused = _draft_reference_only(scheme, design, depth_targets)
        if refused:
            result["reference_only"] = refused
        if symmetry is not None:
            result["symmetry"] = symmetry(scheme, design)
        return result

    async def limits(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "限制檢查的請求不是 JSON"}, status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("scheme"), dict):
            return JSONResponse({"error": "限制檢查的請求要有 scheme（整份方案）"}, status_code=400)
        return JSONResponse(await run_in_threadpool(judged, body["scheme"], body.get("design"), body.get("include_candidate_rows") is True))

    async def candidates(request: Request) -> Response:
        if depth_targets is None:
            return JSONResponse({"error": "還沒有材料庫，無法核對候選"}, status_code=400)
        try:
            selections = TypeAdapter(list[CandidateSelection]).validate_python(await request.json())
            targets = await run_in_threadpool(depth_targets)
            return JSONResponse([effective_candidates(selection, targets).model_dump() for selection in selections])
        except (ValueError, OSError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    return [Route("/api/zone-limits", limits, methods=["POST"]), Route("/api/zone-candidates", candidates, methods=["POST"])]


