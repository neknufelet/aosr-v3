"""換樣板的候選確認迴圈的一輪（#561 庚三）：單牆換到別的家族或「不貼」、快速評分、少數候選進空腔重調、跟同預算的基準比。

老闆 2026-10-28 回「B 的候選範圍，但把重調次數、輪數和總時間限住」、2026-10-30 再定「保留原牆」要納入正式候選：
起點是庚二交出的整房方案；每一面列出所有可換的目標——別的家族（設定檔 ``[[family]]``，換過去用家族的
``template_preset`` 當構造、``initial_thickness_m``／``initial_cavity_m`` 當初值）加「不貼」（``SurfaceDesign.kept_material``，
方案原本那面的牆，不是剛性）——每個（面, 目標）各求值一次（快速評分、不重調），照值由小到大取前
``[templates].max_per_round`` 個進重調。基準＝目前配置 :func:`retune_cavities`；每個候選＝換過去再 :func:`retune_cavities`
（同步數）。重調是全部面的空腔一起走（決策紙「決定」第 3 條：所有牆的空腔做有限步數的梯度重調、基準同步數），
換過去那面若沒有空腔變數（24K、96K 的樣板沒有空氣層、合法厚度只有一種；「不貼」沒有變數），它本身沒有可重調的變數，
重調照步數走其他面的空腔（整房都沒有空腔時是同一點重複求值）。改善＝基準重調後的值減候選重調後的值，最好的「超過」``[candidates].improvement_threshold``
才接受、交出它重調後的配置與 θ；都沒超過就交出重調後的基準、停在 :data:`TEMPLATE_STOP_NO_IMPROVEMENT`（字面照老闆的話：
只說在本輪已確認的單牆候選與預算內沒找到，不說排除了單牆換樣板，也沒涵蓋兩面一起換）。

不同配置是不同的設計清單、不同的目標函式：每個（面, 目標）用 ``TemplateContext.objective_for`` 重建一份
``chain_objective``，編譯版掛在那份目標函式自己身上（``compiled_evaluator``），配置之間不共用快取、不會撞；
快速評分完就丟；進重調的候選各自重建一份（多幾次編譯），最多 ``max_per_round`` 份同時在手（加上目前配置那一份）。
「不貼」的保留代號 :data:`~aosr.config.material_gradient.NONE_TARGET` 只住在設定層（載入器拒收叫這個名字的家族）。

純函式，沒有命令列、沒有行程、不印東西；給了 ``should_stop`` 就在每次求值之前查一次。精度跟 θ 走。
整段、帳本與接續在 ``template_segment``、帳本格式在 ``template_ledger``。
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal, NamedTuple

import jax.numpy as jnp
import numpy as np

from aosr.config.construction_presets import ConstructionPreset
from aosr.config.material_gradient import MOUNT_RIGID, NONE_TARGET, MaterialGradientSettings, Mount
from aosr.gradient.endpoint import checked_wall_designs
from aosr.gradient.material_gradient import (
    ChainObjective, Components, DescentResult, FloatVector, StopCheck, SurfaceDesign, check_stop, checked_theta, compiled_evaluator, design_names, designs_from_presets, is_feasible, retune_cavities,
)
from aosr.gradient.material_gradient_loop import retune_evaluation_bound
from aosr.materials.construction_models import SurfaceMaterial
from aosr.scoring.gradient_objective_jax import objective_reading


TEMPLATE_STOP_NO_ALTERNATIVES: Final = "沒有可換的樣板"
TEMPLATE_STOP_NO_IMPROVEMENT: Final = "在本輪已確認的單牆候選與計算預算內，未找到超過改善門檻的方案"
TEMPLATE_STOP_MAX_EVALUATIONS: Final = "換樣板求值到上限"
TEMPLATE_STOP_MAX_ROUNDS: Final = "換樣板輪數到上限"
TEMPLATE_STOP_MAX_SECONDS: Final = "換樣板總時間到上限"
TemplateStopReason = Literal[
    "沒有可換的樣板", "在本輪已確認的單牆候選與計算預算內，未找到超過改善門檻的方案", "換樣板求值到上限",
    "換樣板輪數到上限", "換樣板總時間到上限"]
TEMPLATE_STOP_REASONS: Final[tuple[TemplateStopReason, ...]] = (
    TEMPLATE_STOP_NO_ALTERNATIVES, TEMPLATE_STOP_NO_IMPROVEMENT, TEMPLATE_STOP_MAX_EVALUATIONS,
    TEMPLATE_STOP_MAX_ROUNDS, TEMPLATE_STOP_MAX_SECONDS)
CANDIDATE_SEPARATOR: Final = "->"

Surfaces = tuple[tuple[str, str | None], ...]
ObjectiveFactory = Callable[[tuple[SurfaceDesign, ...]], ChainObjective]


class TemplateContext(NamedTuple):
    """換樣板要的固定材料：搜尋設定、構造預設清單、每一面方案原本的牆（面代號 → 材料，六面都要有）、
    設計清單 → 目標函式的工廠（產品是 ``chain_objective``；考卷可給合成目標）、這一次搜尋的安裝方式（#990：換樣板沿用它，
    不列舉安裝方式；貼在原牆上時換過去的樣板背後接的仍是那一面原牆）。"""

    settings: MaterialGradientSettings
    presets: tuple[ConstructionPreset, ...]
    kept: Mapping[str, SurfaceMaterial]
    objective_for: ObjectiveFactory
    mount: Mount = MOUNT_RIGID


class TemplatePoint(NamedTuple):
    """一個整房配置：每面設計、可行 θ、值（新合分）、舊合分與三分項（主位起伏、座位穩健性的聲道平均、大峰代理）。"""

    designs: tuple[SurfaceDesign, ...]
    theta: FloatVector
    value: float
    screening_total: float
    components: Components


class QuickScore(NamedTuple):
    """一個（面, 目標）的快速評分：換過去的配置、變數名、θ、值（新合分）、舊合分與三分項、秒數
    （重建目標、編譯與求值一起量）。"""

    surface_id: str
    target: str
    surfaces: Surfaces
    variable_names: tuple[str, ...]
    theta: FloatVector
    value: float
    screening_total: float
    components: Components
    seconds: float

    @property
    def candidate_id(self) -> str:
        """帳本用的代號，例如 ``x0->glass_wool_24k``、``ceiling->none``。"""
        return candidate_id(self.surface_id, self.target)


class RetunedTemplate(NamedTuple):
    """進重調的候選：它的快速評分、設計清單、目標函式、重調、重調後的值、改善（基準重調後的值減它）。"""

    quick: QuickScore
    designs: tuple[SurfaceDesign, ...]
    objective: ChainObjective
    retune: DescentResult
    value: float
    improvement: float


class TemplateRoundResult(NamedTuple):
    """一輪的結果。``opened`` 為假時這一輪沒開（沒有可換的、或最壞求值數放不進剩下的預算），沒有求值、配置原樣。

    ``point``／``objective`` 是這一輪交出去的：接受了就是那個候選重調後的、沒接受就是重調後的基準、沒開就是原樣。
    ``stop_reason`` 是這一輪自己的結論（沒開的兩種、或沒接受），接受了是 None。``bound`` 是開這一輪前算的最壞求值數。
    """

    opened: bool
    bound: int
    quick: tuple[QuickScore, ...]
    baseline: DescentResult | None
    retuned: tuple[RetunedTemplate, ...]
    accepted: RetunedTemplate | None
    point: TemplatePoint
    objective: ChainObjective
    evaluations: int
    seconds: float
    stop_reason: TemplateStopReason | None


# ---- 候選代號與可換的目標 ----

def candidate_id(surface_id: str, target: str) -> str:
    return f"{surface_id}{CANDIDATE_SEPARATOR}{target}"


def surfaces_of(designs: Sequence[SurfaceDesign]) -> Surfaces:
    """設計清單的配置：每面（面代號, preset id 或 None）。"""
    return tuple((design.surface_id, design.preset_id) for design in designs)


def target_of(preset_id: str | None, settings: MaterialGradientSettings) -> str:
    """一面現在的目標代號：不貼是 :data:`NONE_TARGET`，有 preset 是它所屬的家族；不在任何家族就拒收。"""
    if preset_id is None:
        return NONE_TARGET
    family = settings.family_of(preset_id)
    if family is None:
        raise ValueError(f"preset {preset_id} 不在家族清單裡（設定檔 [[family]]），不知道它是哪一個家族")
    return family.id


def target_order(settings: MaterialGradientSettings) -> tuple[str, ...]:
    """目標的固定順序：設定檔家族的順序，最後是不貼。"""
    return (*(family.id for family in settings.family), NONE_TARGET)


def alternatives(surfaces: Surfaces, settings: MaterialGradientSettings) -> tuple[tuple[str, str], ...]:
    """每一面可換的（面代號, 目標）：所有家族扣掉它現在的、再加不貼（它現在不是不貼時）；照面序、再照目標序。"""
    found: list[tuple[str, str]] = []
    for surface_id, preset_id in surfaces:
        current = target_of(preset_id, settings)
        found.extend((surface_id, target) for target in target_order(settings) if target != current)
    return tuple(found)


def preset_for_target(target: str, settings: MaterialGradientSettings) -> str | None:
    """目標換成哪個 preset：不貼是 None，家族是它的 ``template_preset``。"""
    return None if target == NONE_TARGET else settings.family_by_id(target).template_preset


def designs_for_surfaces(surfaces: Surfaces, context: TemplateContext) -> tuple[SurfaceDesign, ...]:
    """配置 → 每面設計：有 preset 照家族建樣板設計、不貼的面帶方案原本那面的牆；要照 ``Wall.all()`` 給六面。"""
    designs = designs_from_presets(surfaces, context.settings, context.presets, kept=context.kept, mount=context.mount)
    return checked_wall_designs(designs)


# ---- 換一面 ----

def swap(designs: Sequence[SurfaceDesign], theta: FloatVector, surface_id: str, target: str,
         context: TemplateContext) -> tuple[tuple[SurfaceDesign, ...], FloatVector]:
    """一面換成目標：家族就用它的 ``template_preset`` 構造（變數照 ``_surface_design`` 的規則）、初值用家族的
    ``initial_thickness_m``／``initial_cavity_m``；不貼就是方案原本的牆、沒有變數。其他面的變數值逐位照搬、精度不變。

    樣板有空氣層、家族卻沒寫空腔初值（或反過來）拒收；新 θ 要在盒界內、可行（:func:`is_feasible`），不然拒收。
    換成它現在的家族（或現在就不貼還換不貼）不是替換，拒收。
    """
    values = checked_theta(theta, designs)
    surfaces = surfaces_of(designs)
    ids = [item[0] for item in surfaces]
    if surface_id not in ids:
        raise ValueError(f"面 {surface_id} 不在設計清單裡")
    index = ids.index(surface_id)
    if target_of(surfaces[index][1], context.settings) == target:
        raise ValueError(f"面 {surface_id} 現在就是 {target}，換成自己不是替換")
    new_surfaces = (*surfaces[:index], (surface_id, preset_for_target(target, context.settings)), *surfaces[index + 1:])
    new_designs = designs_for_surfaces(new_surfaces, context)
    dtype = values.dtype.type
    pieces: list[np.floating] = []
    offset = 0
    for old, new in zip(designs, new_designs, strict=True):
        size = len(old.variables)
        if old.surface_id != surface_id:
            pieces.extend(values[offset:offset + size])
        elif new.material is not None:
            family = context.settings.family_by_id(target)
            if (new.cavity_variable is None) != (family.initial_cavity_m is None):
                raise ValueError(f"家族 {target} 的樣板 {family.template_preset} "
                                 f"{'沒有' if new.cavity_variable is None else '有'}空氣層，空腔初值要"
                                 f"{'不寫' if new.cavity_variable is None else '寫'}")
            pieces.append(dtype(family.initial_thickness_m))
            if family.initial_cavity_m is not None:
                pieces.append(dtype(family.initial_cavity_m))
        offset += size
    new_theta = checked_theta(np.asarray(pieces, dtype=values.dtype), new_designs)
    if not is_feasible(new_theta, new_designs):
        raise ValueError(f"面 {surface_id} 換成 {target} 的初值不可行（厚度不是合法值或厚度＋空腔超過總深度上限）")
    return new_designs, new_theta


# ---- 快速評分與取前幾個 ----

def quick_scores(point: TemplatePoint, context: TemplateContext, *,
                 should_stop: StopCheck | None = None) -> tuple[QuickScore, ...]:
    """每個（面, 目標）各求值一次（不重調）：換過去、重建目標、編譯、求值；記值、舊合分、三分項與秒數。目標用完就丟。

    非有限的值拒收（帳本不收、也排不出名次）。順序照 :func:`alternatives`。
    """
    found = []
    for surface_id, target in alternatives(surfaces_of(point.designs), context.settings):
        check_stop(should_stop)
        clock = time.perf_counter()
        designs, theta = swap(point.designs, point.theta, surface_id, target, context)
        (value, aux), _ = compiled_evaluator(context.objective_for(designs))(jnp.asarray(theta))
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"快速評分 {candidate_id(surface_id, target)} 的值不是有限數：{number}")
        reading = objective_reading(aux)
        found.append(QuickScore(surface_id, target, surfaces_of(designs), design_names(designs), theta, number,
                                reading.screening_total, reading.components, time.perf_counter() - clock))
    return tuple(found)


def ranked(scores: Sequence[QuickScore], count: int) -> tuple[QuickScore, ...]:
    """照值由小到大取前 ``count`` 個；同值照快速評分的順序（＝面序、再目標序）。"""
    order = sorted(range(len(scores)), key=lambda index: (scores[index].value, index))
    return tuple(scores[index] for index in order[:count])


def retune_count(settings: MaterialGradientSettings, quick_count: int) -> int:
    """一輪進重調的候選數：``max_per_round`` 與快速評分數取小。"""
    return min(settings.templates.max_per_round, quick_count)


def template_round_bound(settings: MaterialGradientSettings, quick_count: int, retune_count: int) -> int:
    """一輪最壞要求幾次目標：快速評分每個一次；基準與每個進重調的候選各一段重調，每段照 :func:`retune_evaluation_bound`
    （起點一次＋最多 K 步 × 每步最多試探數，跟庚二同一個算法）。沒有可換的（快速評分 0 個）是 0。

    ``retune_count`` 要是這一輪真的會進重調的數目（:func:`retune_count`），不在 0～快速評分數之間就拒收。
    預算照這個最壞值判：開得了的輪一定跑得完，帳本記的求值次數不會超過上限。
    """
    if quick_count < 0 or not (0 < retune_count <= quick_count or retune_count == quick_count == 0):
        raise ValueError(f"快速評分數 {quick_count} 與進重調數 {retune_count} 不合（要 1 ≤ 進重調數 ≤ 快速評分數，或兩個都 0）")
    if quick_count == 0:
        return 0
    return quick_count + (retune_count + 1) * retune_evaluation_bound(settings)


def next_round_bound(surfaces: Surfaces, settings: MaterialGradientSettings) -> int:
    """這個配置開下一輪的最壞求值數（候選有沒有只看配置，不求值）。"""
    count = len(alternatives(surfaces, settings))
    return template_round_bound(settings, count, retune_count(settings, count))


# ---- 一輪 ----

def template_round(objective: ChainObjective, point: TemplatePoint, context: TemplateContext, *,
                   budget_left: int, should_stop: StopCheck | None = None) -> TemplateRoundResult:
    """在目前配置做一輪：快速評分全部可換的、取前 N 個；基準與每個候選各自從自己的起點做最多 K 步空腔重調再比。

    ``objective`` 是 ``point`` 那個配置的目標函式。沒有可換的就不開、記 :data:`TEMPLATE_STOP_NO_ALTERNATIVES`；
    最壞求值數（:func:`template_round_bound`）大於 ``budget_left`` 就不開、不求值，記 :data:`TEMPLATE_STOP_MAX_EVALUATIONS`。
    最好的候選改善「超過」門檻才接受；都沒超過交出重調後的基準、記 :data:`TEMPLATE_STOP_NO_IMPROVEMENT`。
    """
    if isinstance(budget_left, bool) or budget_left < 0:
        raise ValueError("剩下的求值次數必須是非負整數")
    settings = context.settings
    designs = checked_wall_designs(point.designs)
    if not is_feasible(point.theta, designs):
        raise ValueError("換樣板要從可行的整房配置開始")
    count = len(alternatives(surfaces_of(designs), settings))
    top = retune_count(settings, count)
    bound = template_round_bound(settings, count, top)
    if count == 0 or bound > budget_left:
        reason: TemplateStopReason = TEMPLATE_STOP_NO_ALTERNATIVES if count == 0 else TEMPLATE_STOP_MAX_EVALUATIONS
        return TemplateRoundResult(False, bound, (), None, (), None, point, objective, 0, 0.0, reason)
    clock = time.perf_counter()
    scores = quick_scores(point, context, should_stop=should_stop)
    baseline = retune_cavities(objective, point.theta, designs, settings, should_stop=should_stop)
    base_value = baseline.steps[-1].value
    retuned = []
    for score in ranked(scores, top):
        swapped, theta = swap(designs, point.theta, score.surface_id, score.target, context)
        candidate_objective = context.objective_for(swapped)
        result = retune_cavities(candidate_objective, theta, swapped, settings, should_stop=should_stop)
        value = result.steps[-1].value
        retuned.append(RetunedTemplate(score, swapped, candidate_objective, result, value, base_value - value))
    best = max(retuned, key=lambda item: item.improvement)
    accepted = best if best.improvement > settings.candidates.improvement_threshold else None
    evaluations = len(scores) + baseline.evaluations + sum(item.retune.evaluations for item in retuned)
    if accepted is None:
        chosen = TemplatePoint(designs, baseline.theta, base_value, baseline.steps[-1].screening_total,
                               baseline.steps[-1].components)
        chosen_objective = objective
    else:
        chosen = TemplatePoint(accepted.designs, accepted.retune.theta, accepted.value,
                               accepted.retune.steps[-1].screening_total, accepted.retune.steps[-1].components)
        chosen_objective = accepted.objective
    return TemplateRoundResult(True, bound, scores, baseline, tuple(retuned), accepted, chosen, chosen_objective,
                               evaluations, time.perf_counter() - clock,
                               None if accepted is not None else TEMPLATE_STOP_NO_IMPROVEMENT)
