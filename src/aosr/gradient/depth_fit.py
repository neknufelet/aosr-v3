"""個案深度上限跟材料組自己的限制取較嚴（#1082 段 2-4）：一個換樣板目標（家族、固定材料或不貼）在一面的個案上限下
能不能用、生效上限多少、是哪一條定的。判法只有這一份：命令列開跑前、換樣板候選、搜尋報告與網頁（立面圖標紅、開跑預覽）
都叫這裡。純函式、只讀設定與材料資料，不載入 JAX（網頁行程也匯入它）。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, cast

import numpy as np

from aosr.config.material_gradient import NONE_TARGET, FamilySettings, MaterialGradientSettings, is_fixed_target
from aosr.materials.construction_models import Assembly

DEPTH_SLACK_M = 1e-6
"""總深跟上限比時放過的浮點尾數（公尺）：同網頁切牆面的 ``zone_edit.EDGE_MERGE_TOL``（考卷釘兩邊相等）。搜尋把空腔切到
上限減厚度，存回方案的那幾層照寫下來相加可能多出 1e-17 量級，不放過這一點，搜尋結果帶著上限重開就會自己標紅。"""

FitReason = Literal["fits", "shallower_than_family", "fixed_too_deep", "fixed_not_comparable"]


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
    直接給阻抗的材料（固定阻抗、實測表，物理用的或材料庫那一份都一樣）看不出多深，回 None。"""
    total = _written_depth(material)
    return None if total is None else float(total)


def _written_depth(material: object) -> Decimal | None:
    """照寫下來的十進位數字逐層相加（0.05＋0.1 就是 0.15，不帶浮點尾數），只拿來印。"""
    if not isinstance(material, Assembly):
        return None
    return _layers_depth(material.layers)


def _layers_depth(layers: Sequence[object]) -> Decimal:
    return sum((Decimal(repr(float(getattr(layer, "thickness_m", 0.0)))) for layer in layers), Decimal(0))


def layers_exceed(layers: Sequence[object], cap: float) -> bool:
    """這幾層加起來比上限深（``math.fsum``，放過 :data:`DEPTH_SLACK_M`）。逐字同網頁固定區那一把
    （``zone_limits.added_depth`` 加總、``depth > cap + EDGE_MERGE_TOL``），同一份材料兩邊不會判得不一樣。"""
    return math.fsum(float(getattr(layer, "thickness_m", 0.0)) for layer in layers) > cap + DEPTH_SLACK_M


def fixed_fit(target: str, material: object, cap: float) -> TargetFit:
    """固定材料對個案上限：總深超過就不可用；量不出多深也不可用（限制是硬的，判不了放不放得下就不放）。"""
    written = _written_depth(material)
    if written is None:
        return TargetFit(target, False, "fixed_not_comparable", None, "case")
    if layers_exceed(cast(Assembly, material).layers, cap):
        return TargetFit(target, False, "fixed_too_deep", None, "case", float(written))
    return TargetFit(target, True, "fits", cap, "case", float(written))


def target_fit(target: str, cap: float, settings: MaterialGradientSettings,
               fixed_materials: Mapping[str, object]) -> TargetFit:
    """一個換樣板目標在個案上限 ``cap`` 下的判法；不貼不加層、一定可用。``fixed_materials`` 鍵是 ``fixed:<代號>``。"""
    if target == NONE_TARGET:
        return TargetFit(target, True, "fits", None, None)
    if is_fixed_target(target):
        material = fixed_materials.get(target)
        if material is None:
            raise ValueError(f"固定材料 {target} 不在這一場的材料清單裡，判不了放不放得進深度上限")
        return fixed_fit(target, material, cap)
    return family_fit(settings.family_by_id(target), cap)


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


def family_fits(family: FamilySettings, cap: float | None) -> bool:
    """沒有個案上限一律放得下；有上限時同 :func:`family_fit`。"""
    return cap is None or family_fit(family, cap).usable
