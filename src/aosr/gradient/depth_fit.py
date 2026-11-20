"""個案深度上限跟材料組自己的限制取較嚴（#1082 段 2-4）：一個換樣板目標（家族、固定材料或不貼）在一面的個案上限下
能不能用、生效上限多少、是哪一條定的。判法只有這一份：命令列開跑前、換樣板候選、搜尋報告與網頁（立面圖標紅、開跑預覽）
都叫這裡。純函式、只讀設定與材料資料，不載入 JAX（網頁行程也匯入它）。

每一區自己的可調範圍（#1199，:class:`ZoneRange`）：一區替一種可最佳化材料寫「棉厚從哪個產品厚度到哪個、空腔從幾公尺到幾公尺」，
只准收窄（:func:`range_problem`；家族代號不變，替身訓練過的範圍照舊涵蓋）。收窄後那一族放不放得進上限也是這裡判
（:func:`target_fit` 給了 ``ranges`` 就拿 :func:`narrowed_family` 判）：網頁標紅與命令列開跑同一把。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.material_gradient import NONE_TARGET, FamilySettings, MaterialGradientSettings, is_fixed_target
from aosr.materials.construction_models import Assembly, MeasuredImpedance
from aosr.materials.library import MeasuredTable

DEPTH_SLACK_M = 1e-6
"""總深跟上限比時放過的浮點尾數（公尺）：同網頁切牆面的 ``zone_edit.EDGE_MERGE_TOL``（考卷釘兩邊相等）。搜尋把空腔切到
上限減厚度，存回方案的那幾層照寫下來相加可能多出 1e-17 量級，不放過這一點，搜尋結果帶著上限重開就會自己標紅。"""

FitReason = Literal["fits", "shallower_than_family", "fixed_too_deep", "fixed_not_comparable"]


class ZoneRange(BaseModel):
    """一區一種可最佳化材料自己的可調範圍（#1199）：棉厚的下、上（公尺，要是那一種材料的產品厚度）與空腔的下、上
    （公尺；選填，沒給照材料頁）。這裡只守形狀（有限、不為負、下不大於上）；對材料頁那一份合不合由 :func:`range_problem` 判。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    porous_m: tuple[float, float]
    cavity_m: tuple[float, float] | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        for words, pair in (("棉厚", self.porous_m), ("空腔", self.cavity_m)):
            if pair is None:
                continue
            if pair[0] < 0.0:
                raise ValueError(f"{words}範圍不能是負的，收到 {list(pair)}")
            if pair[0] > pair[1]:
                raise ValueError(f"{words}範圍的下限要不大於上限，收到 {list(pair)}")
        return self


def range_thicknesses(thicknesses: Sequence[float], given: ZoneRange) -> tuple[float, ...]:
    """產品厚度裡落在範圍下～上的那幾個（取原本那幾個數字本身，合法厚度逐位比得到）。"""
    low, high = given.porous_m
    return tuple(value for value in thicknesses if low <= value <= high)


def narrowed_family(family: FamilySettings, given: ZoneRange) -> FamilySettings:
    """收窄後的那一族（判放不放得下、畫面用）：產品厚度切成範圍裡那幾個，空腔範圍換成這一區給的（沒給照原樣）；
    代號、初值與總深度上限不動（初值可能不在範圍裡，所以不重驗：起跑值不在範圍照開跑前那一條拒收）。"""
    update: dict[str, object] = {"thicknesses_m": range_thicknesses(family.thicknesses_m, given)}
    if given.cavity_m is not None:
        update["cavity_range_m"] = given.cavity_m
    return family.model_copy(update=update)


def _meters(values: Sequence[float]) -> str:
    return "、".join(f"{value:g}" for value in values)


def range_problem(family: FamilySettings, given: ZoneRange) -> str | None:
    """一區的範圍對那一種材料（材料頁那一份，也就是這一場的家族）合不合；不合回一句白話（區名、材料名由呼叫端加），合回 None。

    只准收窄：超出材料頁的範圍拒收、叫人到材料頁改；棉厚兩端要是產品厚度之一；空腔要在材料的空腔範圍裡，材料搜尋不調空腔
    （樣板沒有空氣層，同 :func:`family_shallowest_m` 的判法）時不准給；收窄後的最淺排法要放得進材料自己的總深度上限。"""
    thicknesses = family.thicknesses_m
    low, high = given.porous_m
    page = f"材料頁的產品厚度是 {_meters(thicknesses)} 公尺"
    if low < thicknesses[0] or high > thicknesses[-1]:
        return f"棉厚 {low:g}～{high:g} 公尺超出材料頁的範圍（{page}）；這裡只准收窄，要放寬請到材料頁改那份材料"
    off = [value for value in dict.fromkeys((low, high)) if value not in thicknesses]
    if off:
        return f"棉厚的下、上要是產品厚度之一，{_meters(off)} 公尺不是（{page}）"
    if given.cavity_m is not None:
        if family.initial_cavity_m is None:
            return "這種材料搜尋不調空腔（樣板沒有空氣層），不能給空腔範圍"
        (cavity_low, cavity_high), (page_low, page_high) = given.cavity_m, family.cavity_range_m
        if cavity_low < page_low or cavity_high > page_high:
            return (f"空腔 {cavity_low:g}～{cavity_high:g} 公尺超出材料頁的範圍（{page_low:g}～{page_high:g} 公尺）；"
                    "這裡只准收窄，要放寬請到材料頁改那份材料")
    narrowed = narrowed_family(family, given)
    if not _snap_fits(narrowed, family.max_total_depth_m):
        return (f"收窄後的最淺排法 {family_shallowest_m(narrowed):g} 公尺（最薄棉加空腔下界）放不進材料自己的總深度上限 "
                f"{family.max_total_depth_m:g} 公尺；請把下限調小")
    return None


def start_problem(family: FamilySettings, given: ZoneRange, thickness_m: float, cavity_m: float | None) -> str | None:
    """起跑值（樣板的棉厚與空腔；沒有空氣層是 None）在不在這一區收窄後的範圍裡，不在回一句。照材料頁「初值不在範圍內」
    與搜尋起點「θ 超出盒界就拒收」（``material_gradient.baseline_theta``）那一條：拒收、不投影。``given`` 要先過 :func:`range_problem`。"""
    narrowed = narrowed_family(family, given)
    low, high = narrowed.thicknesses_m[0], narrowed.thicknesses_m[-1]
    if not low <= thickness_m <= high:
        return f"起跑值棉厚 {thickness_m:g} 公尺不在這一區的範圍 {low:g}～{high:g} 公尺內"
    cavity_low, cavity_high = narrowed.cavity_range_m
    if cavity_m is not None and not cavity_low <= cavity_m <= cavity_high:
        return f"起跑值空腔 {cavity_m:g} 公尺不在這一區的範圍 {cavity_low:g}～{cavity_high:g} 公尺內"
    return None


START_ADVICE = "起跑值要在範圍內（同材料頁「初值不在範圍內」那一條）：請把範圍放寬到含起跑值，或讓這一區從別種材料起跑"


@dataclass(frozen=True)
class TargetFit:
    """一個目標（家族、固定材料或不貼）在一面的個案上限下能不能用、生效上限多少、是哪一條定的。

    ``blocker``：``case`` 是個案上限（使用者給的那一面的上限）定的或擋的，``material`` 是材料組自己的上限定的；
    ``material_m`` 家族是它自己的總深上限、固定材料是它的總深（量不出來是 None）；``shallowest_m`` 是家族排得出的最淺總深。
    """

    target: str
    usable: bool
    reason: FitReason
    effective_m: float | None
    blocker: Literal["case", "material"] | None
    material_m: float | None = None
    shallowest_m: float | None = None


def family_shallowest_m(family: FamilySettings) -> float:
    """家族排得出的最淺總深（公尺）：最薄棉加空腔下界（樣板沒有空氣層的家族不加）。"""
    return family.thicknesses_m[0] + (family.cavity_range_m[0] if family.initial_cavity_m is not None else 0.0)


def family_fit(family: FamilySettings, cap: float) -> TargetFit:
    """家族對個案上限：最淺排法放不進就不可用（個案擋）；放得下時生效上限取兩邊較嚴的那一個，一樣大算個案定的。

    「放得進」照搜尋取規格的同一條算式判（``material_gradient.snap_to_specs``：生效上限減最薄棉、在精度下往下修到
    不超過，要不小於空腔下界），在 θ 的精度（float64）下算——十進位剛好、浮點放不進的邊界判不可用，
    不讓換樣板在快速評分中途丟錯。"""
    shallowest = family_shallowest_m(family)
    blocker: Literal["case", "material"] = "case" if cap <= family.max_total_depth_m else "material"
    effective = cap if blocker == "case" else family.max_total_depth_m
    if not _snap_fits(family, effective):
        return TargetFit(family.id, False, "shallower_than_family", None, "case", family.max_total_depth_m, shallowest)
    return TargetFit(family.id, True, "fits", effective, blocker, family.max_total_depth_m, shallowest)


def _snap_fits(family: FamilySettings, max_total: float) -> bool:
    """最薄棉放得進 ``max_total``：空腔留得下下界（樣板沒有空氣層的家族下界當 0）。
    算法逐字同 ``material_gradient._depth_room``，在 θ 的精度 float64 下算（``baseline_theta`` 建 float64、
    取規格與換樣板都跟著 θ 的型別；顯卡只在評分 kernel 裡轉單精度）。那支在載入 JAX 的模組裡，網頁行程不能匯入；考卷核兩邊同答案。"""
    total, thickness = np.float64(max_total), np.float64(family.thicknesses_m[0])
    room = total - thickness
    while thickness + room > total:
        room = np.nextafter(room, -np.inf)
    low = np.float64(family.cavity_range_m[0]) if family.initial_cavity_m is not None else np.float64(0.0)
    return bool(room >= low)


def fixed_material_depth_m(material: object) -> float | None:
    """固定材料整塊的總深（公尺）：多層構造是每一層厚度加總（板與布這類層沒有厚度欄、當 0，同網頁 ``zone_limits.added_depth``）；
    實測表（物理用的或材料庫那一份都一樣）一定附底構造（段 2-1），總深照底構造算（#1082 段 2-3d）；
    固定阻抗直接給阻抗、看不出多深，回 None。"""
    total = _written_depth(material)
    return None if total is None else float(total)


def _written_depth(material: object) -> Decimal | None:
    """照寫下來的十進位數字逐層相加（0.05＋0.1 就是 0.15，不帶浮點尾數），只拿來印。"""
    assembly = depth_assembly(material)
    return None if assembly is None else _layers_depth(assembly.layers)


def depth_assembly(material: object) -> Assembly | None:
    """拿來量總深的那一份構造：多層構造是它自己；實測表是它的底構造（量測時樣品就是那一份構造，表外也照它補值）；
    其餘（固定阻抗）回 None。"""
    if isinstance(material, Assembly):
        return material
    if isinstance(material, MeasuredImpedance | MeasuredTable):
        return material.base_assembly
    return None


def _layers_depth(layers: Sequence[object]) -> Decimal:
    return sum((Decimal(repr(float(getattr(layer, "thickness_m", 0.0)))) for layer in layers), Decimal(0))


def layers_exceed(layers: Sequence[object], cap: float) -> bool:
    """這幾層加起來比上限深（``math.fsum``，放過 :data:`DEPTH_SLACK_M`）。逐字同網頁固定區那一把
    （``zone_limits.added_depth`` 加總、``depth > cap + EDGE_MERGE_TOL``），同一份材料兩邊不會判得不一樣。"""
    return math.fsum(float(getattr(layer, "thickness_m", 0.0)) for layer in layers) > cap + DEPTH_SLACK_M


def fixed_fit(target: str, material: object, cap: float) -> TargetFit:
    """固定材料對個案上限：總深超過就不可用；量不出多深也不可用（限制是硬的，判不了放不放得下就不放）。
    實測表用底構造的總深比（:func:`depth_assembly`），比法同多層構造這一把。"""
    written, assembly = _written_depth(material), depth_assembly(material)
    if written is None or assembly is None:
        return TargetFit(target, False, "fixed_not_comparable", None, "case")
    if layers_exceed(assembly.layers, cap):
        return TargetFit(target, False, "fixed_too_deep", None, "case", float(written))
    return TargetFit(target, True, "fits", cap, "case", float(written))


def target_fit(target: str, cap: float, settings: MaterialGradientSettings,
               fixed_materials: Mapping[str, object], ranges: Mapping[str, ZoneRange] | None = None) -> TargetFit:
    """一個換樣板目標在個案上限 ``cap`` 下的判法；不貼不加層、一定可用。``fixed_materials`` 鍵是 ``fixed:<代號>``。
    ``ranges`` 是這一面每一種材料自己的可調範圍（#1199，家族代號 → 範圍）：有範圍的家族照收窄後的那一族判。"""
    if target == NONE_TARGET:
        return TargetFit(target, True, "fits", None, None)
    if is_fixed_target(target):
        material = fixed_materials.get(target)
        if material is None:
            raise ValueError(f"固定材料 {target} 不在這一場的材料清單裡，判不了放不放得進深度上限")
        return fixed_fit(target, material, cap)
    family = settings.family_by_id(target)
    given = None if ranges is None else ranges.get(target)
    return family_fit(family if given is None else narrowed_family(family, given), cap)


def fit_words(fit: TargetFit, cap: float) -> str:
    """命令列與報告用的一句（面名由呼叫端加）。"""
    if fit.reason == "shallower_than_family":
        return (f"深度上限 {cap:g} m 淺於家族 {fit.target} 最淺排法 {fit.shallowest_m:g} m（最薄棉加空腔下界），"
                "這一組在這一面放不下")
    if fit.reason == "fixed_too_deep":
        return f"固定材料 {fit.target} 總深 {fit.material_m:g} m 超過深度上限 {cap:g} m"
    if fit.reason == "fixed_not_comparable":
        return f"固定材料 {fit.target} 直接取代牆面、看不出多深，有深度上限 {cap:g} m 的面不能用（限制是硬的）"
    if fit.effective_m is None:
        return "不貼：不加層"
    if fit.blocker == "material":
        return f"{fit.target} 生效 {fit.effective_m:g} m（材料組自己的上限，比個案上限 {cap:g} m 嚴）"
    return f"{fit.target} 生效 {fit.effective_m:g} m（個案上限" + (
        f"，比材料組自己的 {fit.material_m:g} m 嚴）" if fit.material_m is not None and fit.material_m > cap else "）")


def ranged_misfits(fits: Sequence[TargetFit], ranges: Mapping[str, ZoneRange] | None) -> list[TargetFit]:
    """寫了範圍、收窄後放不進上限的那幾組（#1199）：這一區點名要這種材料、還自己定了範圍，放不下就擋（分區頁標紅、
    開跑拒收同一把），不像沒寫範圍的組只是不換過去。``fits`` 要是照 ``ranges`` 判的（:func:`target_fit`）。"""
    return [fit for fit in fits if ranges and fit.target in ranges and not fit.usable]


def family_fits(family: FamilySettings, cap: float | None) -> bool:
    """沒有個案上限一律放得下；有上限時同 :func:`family_fit`。"""
    return cap is None or family_fit(family, cap).usable
