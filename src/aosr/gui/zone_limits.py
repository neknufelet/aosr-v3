"""超過限制的區塊標紅、寫出是哪一條擋住（#1111，#1082 第 1-4 段）。

一份方案（可以是網頁上還沒存的草稿）加上它的設計設定，逐面逐區列出哪幾條限制沒過：

- 位置：超出牆面（房間尺寸改小以後最常見）、太小（寬或高不到最小邊）、跟別區重疊；一面沒蓋滿另列在那一面。
  判法跟切牆面（``zone_edit``）同一把：同一個最小邊、同一個邊的容差；蓋滿與不重疊照方案自己那一把精確判法
  （``validate_surface_partition``）。
- 深度：一區的上限取「房間預設／這面牆／這一塊」三層裡最小的那一個（一樣小時取範圍最小的那一層），並記是哪一層擋的；
  固定區貼上去的總深（``MountedOnWall`` 加貼的每一層厚度加總，板與布這類層沒有厚度欄、當 0）超過就紅。
  直接取代牆面的材料（固定阻抗、實測表、整面多層構造）看不出貼了多深，照實寫「比不了」、不當作過，也不拿原牆的厚度標紅。
  搜尋區的上限交給搜尋（開跑請求的 ``max_depths``）；保留原牆的區不加層、管不到。
- 搜尋區的上限跟材料組自己的限制取較嚴（#1082 段 2-4，判法只在引擎一份 ``fixed_surfaces.target_fit``）：准用的每一組
  （沒寫清單就是全部可調材料）列出生效上限與是哪一條擋住的（個案哪一層／材料組自己）；比一組最淺排法還淺、或固定材料
  總深超過（量不出多深也算）＝那一組在這一區不可用（列在 ``caps`` 的 ``targets``、不擋，搜尋不換過去）；准用的材料組一組都不可用＝紅
  （「不貼」不算救：交給搜尋卻什麼都貼不上）。清單本來只有「不貼」的區不判。沒傳材料組（``depth_targets``）就不判這一項。
- 可信範圍（固定區的材料參數，跟材料頁同一個出處 ``material_curves.range_check``）：物理拒算的標紅；照算但醒目提醒的
  標「警告」、不擋（計畫段 2-2b 寫的是「超出可信範圍警告」，物理那邊也照算）。沒傳檢查函式就不判這一項；
  網頁的限制檢查、開跑預覽、開跑與建替身都傳材料頁那一支（正式入口開機釘中央處理器＋雙精度）。

紅的會擋「拿去搜尋」與開跑（``refusals``，``gradient_plan.plan_search`` 也照它擋）；警告與說明只顯示。
這一支只讀不寫；草稿有一格讀不出來就照實列，不丟錯誤碼（畫面在改到一半時也要能問）。
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aosr.config.material_gradient import NONE_TARGET, MaterialGradientSettings, fixed_target, is_fixed_target
from aosr.geometry.shoebox import Room
from aosr.gradient.depth_fit import TargetFit, target_fit
from aosr.gui.labels import WALLS as WALL_WORDS
from aosr.gui.zone_design import WHOLE_WALL_KEY, DesignSettings, ZoneMode
from aosr.gui.zone_edit import EDGE_MERGE_TOL, MIN_SIDE_M, ZoneCutError, draft_room, too_small
from aosr.materials.construction_models import Assembly, MeasuredImpedance, MountedOnWall, PorousLayer, SurfaceMaterial
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

    def label(self, identifier: str) -> str:
        return self.labels.get(identifier, identifier)


@dataclass(frozen=True)
class DepthCap:
    """一區實際的深度上限：值（公尺）與是哪一層擋的。"""

    value_m: float
    level: DepthLevel

    def words(self) -> str:
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


def zone_words(wall: str, key: str) -> str:
    """一區在畫面上叫什麼（擋下的原文給人看，不露內部代號）：牆名照表單（``labels.WALLS``），
    區名同逐區表——整面、自動補的寫「自動 代號」、畫的框寫它的代號。"""
    zone = "整面" if key == WHOLE_WALL_KEY else f"自動 {key}" if key.startswith(FILL_PREFIX) else key
    return f"{WALL_WORDS.get(wall, wall)}「{zone}」"


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


def _depth_problems(material: SurfaceMaterial | None, mode: ZoneMode, cap: DepthCap | None) -> list[Problem]:
    if cap is None or mode != "fixed":
        return []
    depth = None if material is None else added_depth(material)
    if depth is None:
        return [Problem(code="depth_not_comparable", level="note",
                        text=f"這一區的材料直接取代牆面、看不出貼了多深，上限 {cap.words()} 比不了")]
    if depth > cap.value_m + EDGE_MERGE_TOL:
        return [Problem(code="depth_over", level="red",
                        text=f"貼上去總深 {depth:g} 公尺，超過深度上限 {cap.words()}")]
    return []


def zone_targets(allowed: Sequence[str] | None, targets: DepthTargets) -> tuple[str, ...] | None:
    """一區可選的目標（引擎的寫法：家族代號、``fixed:<代號>``、none）：沒寫清單是全部家族加不貼；清單有不認識的代號回 None
    （開跑那邊照講拒收，這裡不猜）。"""
    families = {family.id for family in targets.settings.family}
    if allowed is None:
        return (*(family.id for family in targets.settings.family), NONE_TARGET)
    found: list[str] = []
    for item in allowed:
        if item == NONE_TARGET or item in families:
            found.append(item)
        elif item in targets.fixed:
            found.append(fixed_target(item))
        else:
            return None
    return tuple(found)


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


def target_fits(allowed: Sequence[str] | None, cap: float, targets: DepthTargets) -> tuple[TargetFit, ...] | None:
    """一區每一個材料組（不含不貼）在上限 ``cap`` 下的判決；清單有不認識的代號回 None。"""
    listed = zone_targets(allowed, targets)
    if listed is None:
        return None
    fixed = {fixed_target(key): material for key, material in targets.fixed.items()}
    return tuple(target_fit(target, cap, targets.settings, fixed) for target in listed if target != NONE_TARGET)


def no_group_words(lines: Sequence[Mapping[str, object]]) -> str:
    """准用的材料組一組都放不下時擋下的那一句（頁面紅、開跑擋同一句）。"""
    return "准用的材料組一組都放不進深度上限，交給搜尋什麼都貼不上（「不貼」不算）：" + "；".join(str(line["text"]) for line in lines)


def _target_problems(allowed: Sequence[str] | None, cap: DepthCap, targets: DepthTargets
                     ) -> tuple[list[Problem], list[dict[str, object]]]:
    fits = target_fits(allowed, cap.value_m, targets)
    if fits is None or not fits:
        return [], []
    lines = [target_line(fit, cap, targets) for fit in fits]
    if not any(fit.usable for fit in fits):
        return [Problem(code="depth_no_group", level="red", text=no_group_words(lines))], lines
    return [], lines


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
                depth_targets: Callable[[], DepthTargets] | None = None) -> dict[str, object]:
    """一份方案（草稿也可以）逐面逐區的限制檢查；回 ``zones``（牆 → 區 → 幾條）、``walls``（整面的）、``caps``（每區生效上限；
    搜尋區另有 ``targets``：每一組的生效上限與哪一條擋住）、``red``（擋下的條數）與 ``refusals``（擋下的每一條，給開跑與
    「拿去搜尋」照講）。``depth_targets`` 只在有一區交給搜尋又有上限時才叫（讀材料庫）；叫失敗照講成一條紅。"""
    scheme_id = str(document.get("scheme_id", ""))
    design, top = _design_of(design_value, scheme_id)
    try:
        room = draft_room(dict(document))
    except ZoneCutError as exc:
        return {"status": "room", "message": str(exc), "zones": {}, "walls": {}, "caps": {}, "red": 0, "refusals": []}
    scene = cast(Mapping[str, object], document.get("scene") or {})
    by_wall = scene.get("surface_zones_by_wall")
    walls: dict[str, list[Problem]] = {wall: [] for wall in WALLS}
    zones: dict[str, dict[str, list[Problem]]] = {}
    caps: dict[str, dict[str, dict[str, object]]] = {}
    modes = _modes(design)
    checked: dict[str, list[Problem]] = {}
    loaded: DepthTargets | bool | None = None
    for wall in WALLS:
        entries = cast(list[Mapping[str, object]], by_wall.get(wall) if isinstance(by_wall, dict) else None) or [{"area": "whole_wall"}]
        if not isinstance(entries, list) or not all(isinstance(entry, Mapping) for entry in entries):
            walls[wall].append(Problem(code="unreadable", level="red", text="這一面的分區讀不出來；請重新打開方案"))
            continue
        found: dict[str, list[Problem]] = {}
        _place_problems(room, wall, entries, found, walls)
        for entry in entries:
            key = _key(entry) if isinstance(entry, Mapping) else None
            if key is None:
                continue
            mode = modes.get(wall, {}).get(key, "fixed")
            cap = depth_cap(design, wall, key)
            if cap is not None and mode != "keep_original":
                caps.setdefault(wall, {})[key] = {"value_m": cap.value_m, "level": cap.level, "words": cap.words()}
            material = _material(entry.get("material"))
            lines = found.setdefault(key, [])
            lines.extend(_depth_problems(material, mode, cap))
            if cap is not None and mode == "search" and depth_targets is not None:
                if loaded is None:
                    loaded = _loaded(depth_targets, top)
                if isinstance(loaded, DepthTargets):
                    item = design.zones.get(wall, {}).get(key) if design is not None else None
                    problems, targets = _target_problems(None if item is None else item.allowed, cap, loaded)
                    lines.extend(problems)
                    if targets:
                        caps[wall][key]["targets"] = targets
            if range_check is not None and mode == "fixed" and material is not None:
                text = json.dumps(material.model_dump(mode="json"), sort_keys=True)
                if text not in checked:
                    checked[text] = _range_problems(range_check, material)
                lines.extend(checked[text])
        zones[wall] = {key: lines for key, lines in found.items() if lines}
    refusals = [problem.text for problem in top]
    refusals += [f"{WALL_WORDS.get(wall, wall)}：{problem.text}" for wall, lines in walls.items() for problem in lines if problem.level == "red"]
    refusals += [f"{zone_words(wall, key)}：{problem.text}" for wall, items in zones.items() for key, lines in items.items()
                 for problem in lines if problem.level == "red"]
    return {"status": "checked", "design": [problem.model_dump() for problem in top],
            "zones": {wall: {key: [problem.model_dump() for problem in lines] for key, lines in items.items()}
                      for wall, items in zones.items() if items},
            "walls": {wall: [problem.model_dump() for problem in lines] for wall, lines in walls.items() if lines},
            "caps": caps, "red": len(refusals), "refusals": refusals}


def _loaded(provider: Callable[[], DepthTargets], top: list[Problem]) -> DepthTargets | bool:
    """讀這一場的材料組（材料庫）；讀不進來照講成一條紅（開跑那邊也一樣開不了），回 False 不再試。"""
    try:
        return provider()
    except ValueError as error:
        top.append(Problem(code="materials_unreadable", level="red",
                           text=f"查不了交給搜尋的區放不放得下材料組：{error}"))
        return False


def _material(value: object) -> SurfaceMaterial | None:
    try:
        return _MATERIAL.validate_python(value)
    except ValidationError:
        return None


def _modes(design: DesignSettings | None) -> dict[str, dict[str, ZoneMode]]:
    """草稿的做法：設計設定有寫照寫，沒寫是固定（同 ``effective_modes``，只是草稿還不是合法方案、不能先驗）。"""
    if design is None:
        return {}
    return {wall: {key: item.mode for key, item in items.items()} for wall, items in design.zones.items()}


def saved_refusals(scheme_document: Mapping[str, object], design: DesignSettings | None,
                   range_check: RangeCheck | None = None, depth_targets: DepthTargets | None = None) -> list[str]:
    """存好的方案開搜尋（或建替身）前要擋的那幾條：跟立面圖標紅同一把；``range_check`` 給了就連可信範圍一起判
    （網頁開跑與建替身傳材料頁那一支，頁面紅的、伺服器就擋）；``depth_targets`` 給了就連准用材料組放不放得下一起判。"""
    value = None if design is None else design.model_dump(mode="json")
    provider = None if depth_targets is None else (lambda: depth_targets)
    return cast(list[str], zone_limits(scheme_document, value, range_check, provider)["refusals"])


def zone_limit_routes(range_check: RangeCheck | None,
                      depth_targets: Callable[[], DepthTargets] | None = None) -> list[Route]:
    """``POST /api/zone-limits``：收 ``{"scheme": 草稿, "design": 設計設定或 null}``，內容有問題一律回 200 照講。"""

    async def limits(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "限制檢查的請求不是 JSON"}, status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("scheme"), dict):
            return JSONResponse({"error": "限制檢查的請求要有 scheme（整份方案）"}, status_code=400)
        return JSONResponse(await run_in_threadpool(zone_limits, body["scheme"], body.get("design"), range_check, depth_targets))

    return [Route("/api/zone-limits", limits, methods=["POST"])]


