"""分區方案的結果頁附表（#1092）：逐區表、原牆基準、改善多少。

「改善多少」只有一個基準：**同一套切法**、每一區換回原牆的那一場，用同一把尺（同一份評分設定、同一版物理）比。
同一套切法是因為牆上一切矩形，低頻網格就不同（#1088 量到小房 1568→1583 節點、低頻逐點差 3.3%～13.9%），
拿整面原牆那一場比，會把網格的差也算成改善。所以基準方案逐區沿用設計的代號與座標，只換材料。

每一區的原牆怎麼來（一條規則，畫面逐區寫出來）：

- 設計設定指了原方案（``original_scheme_id``）：照原方案在那一區的材料。原方案要有一區整個蓋住設計的這一區；
  原方案的切線劃過設計的一區就照實拒收，不猜要取哪一邊。
- 沒指：拆掉這一區加貼的層（貼在原牆上的構造，取它記的原牆）；沒有加貼層的區看不出原牆，兩邊一樣，畫面點名。

基準方案的代號＝設計代號＋「-orig-」＋基準內容雜湊前 8 碼：設計或原方案改了，代號跟著換，舊的基準結果不會被誤認。
原方案六面都是整面固定阻抗時，方案檔存成六個數字；這裡不自己把數字建成材料（固定阻抗只在已驗輸入出生，
見 ``report_io.solver_inputs``），而是把原方案的牆放進設計的擺位、走同一條驗證拿那六面材料，見 ``original_walls``。
這一支不在物理身分範圍裡：只組方案、讀結果、排版，不算物理。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.quality_targets import QualityTargets
from aosr.gui.labels import LABELS, WALLS
from aosr.gui.problem_text import material_summary
from aosr.gui.zone_design import MODE_WORDS, WHOLE_WALL_KEY, DesignSettings, effective_modes, mismatches
from aosr.geometry.shoebox import Room
from aosr.materials.construction_models import MountedOnWall, SurfaceMaterial
from aosr.physics.report_io import solver_inputs
from aosr.physics.report_surfaces import RectangleZone, SurfaceZone, WholeWallZone
from aosr.physics.zone_fill import wall_size_m
from aosr.reporting.compare import compare_results
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError, checked_inputs
from aosr.scoring.contract import QualityCategory
from aosr.scoring.ranking_models import CandidateStatus

BaselineSource = Literal["original_scheme", "unwrapped", "unchanged"]
Verdict = Literal["better", "worse", "same", "missing"]
SUFFIX = "-orig-"
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
# 分區座標兩軸的白話：軸向同 RectangleZone 的說明（地板與天花 u＝x、v＝y；x 牆 u＝y；y 牆 u＝x；牆的 v＝z）。
AXES = {"floor": ("長向", "寬向"), "ceiling": ("長向", "寬向"), "x0": ("寬向", "高度"), "xL": ("寬向", "高度"),
        "y0": ("長向", "高度"), "yL": ("長向", "高度")}
PLACE_NOTE = "位置從房間座標原點那個角量起：長向＝x、寬向＝y、高度＝z（公尺）。"
STATUS_WORDS = {CandidateStatus.RANKABLE: "可排名", CandidateStatus.ELIMINATED: "被淘汰",
                CandidateStatus.NOT_EVALUATED: "有類別還沒評", CandidateStatus.NOT_COMPARABLE: "跟另一份算到的類別或評分身分不同、不能同表",
                CandidateStatus.ILLEGAL: "不合法"}
IMPROVEMENT_NOTE = "代價越低越好；兩邊用同一份評分設定、同一版物理、同一套切法，差只來自每一區換了什麼。"


class BaselineRefused(ValueError):
    """原牆鋪不上這一套切法；訊息是給人看的一句話。"""


@dataclass(frozen=True)
class BaselineZone:
    """一區：位置照設計，設計與基準各一個材料，以及基準材料從哪裡來。"""

    wall: str
    zone_id: str
    u_m: tuple[float, float]
    v_m: tuple[float, float]
    design: SurfaceMaterial
    baseline: SurfaceMaterial
    source: BaselineSource


@dataclass(frozen=True)
class OriginalWalls:
    """原方案鋪原牆要用的三樣：代號、房間尺寸、每一面牆的分區（六個數字的方案已換成整面一區）。"""

    scheme_id: str
    room_m: Room
    zones: Mapping[str, tuple[SurfaceZone, ...]]


def original_walls(original: Scheme, design: Scheme, *, capabilities: CapabilityTable,
                   directivity: DirectivityDefaults) -> OriginalWalls:
    """原方案每一面牆的分區；只拿牆，原方案自己的喇叭、座位、家具不帶進基準，也不拿來擋。

    六面固定阻抗存成六個數字的：把原方案的場景放進設計的擺位（基準就是用設計的擺位算），走計算同一條驗證、
    從唯一出生地拿六面材料；過不了就是原方案的牆本身鋪不上這一場。
    """
    if original.scene.room_m != design.scene.room_m:
        raise BaselineRefused(f"原方案「{original.scheme_id}」的房間尺寸跟設計不同，原牆鋪不上去")
    zones = original.scene.surface_zones_by_wall
    if zones is not None:
        return OriginalWalls(original.scheme_id, original.scene.room_m, dict(zones))
    try:
        placed = Scheme.model_validate(design.model_copy(update={"scene": original.scene}).model_dump())
        _, documents = checked_inputs(placed, capabilities=capabilities, directivity=directivity)
    except (SchemeValidationError, ValueError) as exc:
        raise BaselineRefused(f"原方案「{original.scheme_id}」的牆放進設計的擺位過不了檢查：{exc}") from exc
    _, inputs = next(iter(documents.values()))
    return OriginalWalls(original.scheme_id, original.scene.room_m, {
        wall.wall_name(): (WholeWallZone(area="whole_wall", material=material),)
        for wall, material in solver_inputs(inputs).materials_by_wall.items()})


@dataclass(frozen=True)
class Baseline:
    scheme: Scheme
    zones: tuple[BaselineZone, ...]
    original_scheme_id: str | None

    @property
    def changed(self) -> bool:
        return any(zone.design != zone.baseline for zone in self.zones)


def is_zoned(scheme: Scheme) -> bool:
    """有任何一面牆切了矩形才是分區方案；只有整面一區的方案不出這一節。"""
    return any(isinstance(zone, RectangleZone)
               for zones in (scheme.scene.surface_zones_by_wall or {}).values() for zone in zones)


def _box(zone: SurfaceZone, size: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    return (zone.u_m, zone.v_m) if isinstance(zone, RectangleZone) else ((0.0, size[0]), (0.0, size[1]))


def _zone_name(zone_id: str) -> str:
    return "整面" if zone_id == WHOLE_WALL_KEY else zone_id


def _original_material(original: OriginalWalls, wall: str, zone_id: str,
                       box: tuple[tuple[float, float], tuple[float, float]]) -> SurfaceMaterial:
    """原方案在這一區的材料：要有一區整個蓋住它（邊照原值精確比，不加容差）；劃過兩區就拒收。"""
    zones = original.zones.get(wall)
    if zones is None:
        raise BaselineRefused(f"原方案「{original.scheme_id}」沒有{WALLS[wall]}，原牆鋪不上去")
    size = wall_size_m(original.room_m, wall)
    (u0, u1), (v0, v1) = box
    for zone in zones:
        (a0, a1), (b0, b1) = _box(zone, size)
        if a0 <= u0 and u1 <= a1 and b0 <= v0 and v1 <= b1:
            return zone.material
    raise BaselineRefused(f"原方案「{original.scheme_id}」在{WALLS[wall]}的切線劃過設計的「{_zone_name(zone_id)}」"
                          "這一區，原牆鋪不上去；把設計這一區沿原方案的切線切開，或改設計設定指的原方案")


def _baseline_scheme(design: Scheme, zones: dict[str, tuple[SurfaceZone, ...]]) -> Scheme:
    scene = design.scene.model_copy(update={"surface_zones_by_wall": zones, "impedance_pa_s_per_m_by_wall": None})
    body = Scheme.model_validate(design.model_copy(update={"scene": scene}).model_dump()).model_dump(mode="json")
    body.pop("scheme_id")
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    prefix = design.scheme_id[:180] if SAFE.fullmatch(design.scheme_id) else "scheme"
    return Scheme.model_validate({**body, "scheme_id": f"{prefix}{SUFFIX}{digest}"})


def derive_baseline(design: Scheme, original: OriginalWalls | None = None) -> Baseline:
    """同一套切法、每一區換回原牆的基準方案；鋪不上就丟 BaselineRefused。設計要有分區清單（切了矩形的方案一定有）。"""
    if design.scene.surface_zones_by_wall is None:
        raise BaselineRefused("這一份方案沒有分區，沒有逐區的原牆可推")
    if original is not None and original.room_m != design.scene.room_m:
        raise BaselineRefused(f"原方案「{original.scheme_id}」的房間尺寸跟設計不同，原牆鋪不上去")
    rows: list[BaselineZone] = []
    laid_by_wall: dict[str, tuple[SurfaceZone, ...]] = {}
    for wall, zones in design.scene.surface_zones_by_wall.items():
        size = wall_size_m(design.scene.room_m, wall)
        laid: list[SurfaceZone] = []
        for zone in zones:
            zone_id = zone.id if isinstance(zone, RectangleZone) else WHOLE_WALL_KEY
            box = _box(zone, size)
            material: SurfaceMaterial
            source: BaselineSource
            if original is not None:
                material, source = _original_material(original, wall, zone_id, box), "original_scheme"
            elif isinstance(zone.material, MountedOnWall):
                material, source = zone.material.wall, "unwrapped"
            else:
                material, source = zone.material, "unchanged"
            laid.append(zone.model_copy(update={"material": material}))
            rows.append(BaselineZone(wall, zone_id, box[0], box[1], zone.material, material, source))
        laid_by_wall[wall] = tuple(laid)
    return Baseline(_baseline_scheme(design, laid_by_wall), tuple(rows),
                    original.scheme_id if original is not None else None)


class ZoneRow(BaseModel):
    """逐區表一列；字都排好了，頁面只照印。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    wall: str
    zone: str
    place: str
    size: str
    area: str
    mode: str
    design: str
    baseline: str
    baseline_source: str
    changed: bool


def _source_text(zone: BaselineZone, original: str | None) -> str:
    if zone.design == zone.baseline:
        return ("兩邊一樣：原方案這一區就是這個材料" if zone.source == "original_scheme"
                else "兩邊一樣：這一區沒有加貼的層，當成沒動")
    return {"original_scheme": f"照原方案「{original}」那一區", "unwrapped": "拆掉這一區加貼的層",
            "unchanged": ""}[zone.source]


def zone_rows(baseline: Baseline, settings: DesignSettings | None, scheme: Scheme, *,
              baseline_known: bool = True) -> tuple[ZoneRow, ...]:
    """逐區表：位置、尺寸、面積、做法、設計的材料、原牆基準的材料與它怎麼來。

    ``baseline_known`` 是假（原牆推不出來）時，基準兩欄印「—」，不拿別的規則補上去。
    兩邊一樣的區，基準材料印「同左」，免得同一段長字印兩次。
    """
    modes = effective_modes(settings, scheme) if settings is None or not mismatches(settings, scheme) else {}
    rows = []
    for zone in baseline.zones:
        (u0, u1), (v0, v1) = zone.u_m, zone.v_m
        first, second = AXES[zone.wall]
        mode = modes.get(zone.wall, {}).get(zone.zone_id)
        rows.append(ZoneRow(
            wall=WALLS[zone.wall], zone=_zone_name(zone.zone_id),
            place=f"{first} {u0:.2f}–{u1:.2f}、{second} {v0:.2f}–{v1:.2f}",
            size=f"{u1 - u0:.2f} × {v1 - v0:.2f} m", area=f"{(u1 - u0) * (v1 - v0):.2f} m²",
            mode=MODE_WORDS[mode] if mode is not None else "—",
            design=material_summary(zone.design),
            baseline=("—" if not baseline_known else material_summary(zone.baseline) if zone.design != zone.baseline
                      else "同左"),
            baseline_source=_source_text(zone, baseline.original_scheme_id) if baseline_known else "—",
            changed=baseline_known and zone.design != zone.baseline))
    return tuple(rows)


class ImprovementRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    category: str
    baseline: str
    design: str
    change: str
    verdict: Verdict


class Improvement(BaseModel):
    """改善多少：逐類與總分；總分只在兩邊同一張排名表都可排名時才比。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    rows: tuple[ImprovementRow, ...]
    total: ImprovementRow | None
    total_note: str
    note: str = IMPROVEMENT_NOTE


def _cost_text(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _change(baseline: float | None, design: float | None) -> tuple[str, Verdict]:
    """設計減原牆；代價越低越好，所以負的是改善。百分比以原牆為分母，原牆是 0 就不算百分比。"""
    if baseline is None or design is None:
        return "缺分數，不比", "missing"
    difference = design - baseline
    if difference == 0:
        return "一樣", "same"
    verdict: Verdict = "better" if difference < 0 else "worse"
    word = "改善" if difference < 0 else "變差"
    amount = f"{abs(difference):.4f}" if f"{abs(difference):.4f}" != "0.0000" else f"{abs(difference):.2e}"
    share = (f"（{'少' if difference < 0 else '多'} {abs(difference) / baseline * 100:.1f}%）" if baseline > 0
             else "（原牆是 0，不算百分比）")
    return f"{word} {amount}{share}", verdict


def _side(result: SchemeResult, registry: QualityTargets
          ) -> tuple[dict[QualityCategory, float | None], float | None, CandidateStatus]:
    """跟結果頁同一個取法：可排名就取排名表的逐類代價與總分，不然取評估裡算得出的那幾類、沒有總分。"""
    ranking = compare_results((result,), quality_targets=registry, run_date=result.run_date)
    name = result.scheme.scheme_id
    status = ranking.status_of(name)
    row = next((item for item in ranking.rankable if item.candidate_id == name), None)
    if row is not None:
        return {line.identity.category: line.category_cost for line in row.categories}, row.total_cost, status
    unranked = [item.evaluations for item in ranking.eliminated if item.candidate_id == name]
    unranked += [item.evaluations for item in ranking.not_evaluated if item.candidate_id == name]
    evaluations = unranked[0] if unranked else result.candidate.evaluations
    return ({item.category: item.category_cost.value if item.category_cost else None for item in evaluations},
            None, status)


def improvement(design: SchemeResult, baseline: SchemeResult, registry: QualityTargets) -> Improvement:
    """兩份要先過同一套比較檢查（物理身分、用途、聲道組），不過就丟 ValueError，跟比較頁同一句。

    逐類跟各自結果頁的數字一樣（同一份評分設定各排一次）；總分另外把兩份放進同一張排名表，兩份都可排名才比。
    """
    joint = compare_results((design, baseline), quality_targets=registry, run_date=design.run_date)
    design_costs, design_total, design_status = _side(design, registry)
    baseline_costs, baseline_total, baseline_status = _side(baseline, registry)
    rows = []
    for category in QualityCategory:
        if category not in design_costs and category not in baseline_costs:
            continue
        before, after = baseline_costs.get(category), design_costs.get(category)
        change, verdict = _change(before, after)
        rows.append(ImprovementRow(category=LABELS.get(category.value, category.value), baseline=_cost_text(before),
                                   design=_cost_text(after), change=change, verdict=verdict))
    # 各自排一次就不可排名的，講各自的原因；各自可排名、放進同一張表才不行的，講同表的原因。
    statuses = {side: own if own is not CandidateStatus.RANKABLE else joint.status_of(result.scheme.scheme_id)
                for side, own, result in (("設計", design_status, design), ("原牆基準", baseline_status, baseline))}
    if all(status is CandidateStatus.RANKABLE for status in statuses.values()) and None not in (design_total,
                                                                                                 baseline_total):
        change, verdict = _change(baseline_total, design_total)
        total = ImprovementRow(category="總分", baseline=_cost_text(baseline_total), design=_cost_text(design_total),
                               change=change, verdict=verdict)
        return Improvement(rows=tuple(rows), total=total, total_note="")
    note = "總分不比：" + "；".join(f"{side}{STATUS_WORDS[status]}" for side, status in statuses.items()
                                    if status is not CandidateStatus.RANKABLE)
    return Improvement(rows=tuple(rows), total=None, total_note=note)
