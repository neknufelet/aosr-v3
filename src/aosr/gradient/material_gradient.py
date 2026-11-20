"""第三階段牆面材料梯度搜尋的前半段（#561）：整條可微鏈、投影梯度下降、取規格聯合重調空腔。

找法照決策紙 ``stage-three-gradient-objective-low-frequency-lane.md``「決定」第 3 條：全牆聯合梯度為主；
連續下降每一步投影到「盒界 ∩ 每面厚度＋空腔 ≤ 總深度上限」（老闆 2026-10-28 回 A）；連續厚度轉合法規格時，
總深度限制在取規格時一起滿足、之後空腔聯合重調，不准各面各自取最近就交付。
候選確認迴圈與接續在 ``material_gradient_loop``；終點完整模型重算不在這裡（下一支）。

設計以「一串面（surfaces）」寫，長度不寫死：扁平化、下降、取規格、重調與帳本對任何面數都成立。
最底下接替身的 :func:`chain_objective` 核對完整有序面片清單；整面仍照 ``Wall.all()``，分區各有一個設計。

設計變數只有每面棉板厚度（多孔層）與空腔深度（空氣層），樣板有那一層才有那個變數；
流阻與板材幾何固定、每面樣板固定、沒有樣板的面沒有變數。沒有樣板的面有兩種：帶 ``kept_material``
（保留方案原本的牆，#561 辛三：產品路徑 preset 寫 none 的面一律是這種）整條鏈用那份材料的垂直入射阻抗（常數）；
兩個都沒有才是剛性（合成題用）。
安裝方式（#990，``Mount``）每次搜尋一個值：``rigid`` 是樣板照原樣剛背；``on_wall`` 是「貼在原牆上」——貼樣板的面
新增層背後接那面牆在方案裡原本的材料（``SurfaceDesign.backing_material``），整條鏈與終點都用
``MountedOnWall(layers=樣板的層, wall=原牆)``（:func:`design_template_material`）；不貼的面照舊保留原牆。θ 是一維浮點陣列，順序照面序、
每面先厚度後空腔，:func:`design_names` 列出來。精度跟 θ 走：雙精度 θ 給雙精度目標，
單精度 θ 給單精度；本模組不碰 JAX 全域設定。純函式，沒有命令列、沒有行程、不印東西。
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aosr.config.material_gradient import (
    DescentSettings, MaterialGradientSettings,
)
from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.gradient.gpu import SinglePrecisionObjective
from aosr.gradient.surface_design import (
    SurfaceDesign as SurfaceDesign, WallMaterial as WallMaterial, design_template_material as design_template_material,
    designs_from_presets as designs_from_presets, designs_mount as designs_mount,
    fixed_candidate_design as fixed_candidate_design, wall_designs_from_presets as wall_designs_from_presets,
    wall_surface_ids as wall_surface_ids,
)
from aosr.gradient.surface_design import _THICKNESS
from aosr.gradient.surrogate_guard import require_matching_patches
from aosr.gradient.ties import Cells, check_tied_equal, check_ties_hold, tied_cells
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_diff import ConstructionTemplate, build_template, surface_impedance_from_parameters
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_jax import ReducedConstants, TemplateImpedances, reduced_constants, reduced_pressures, wall_impedances_from_templates
from aosr.scoring.gradient_objective_jax import (
    GradientObjectiveArrays, GradientObjectiveConstants, gradient_objective_from_pressures, objective_reading as low_frequency_reading,
)


STOP_UNESTIMABLE: Final = "起點不可估"
STOP_GRADIENT: Final = "梯度夠小"
STOP_STEP: Final = "步長到下限"
STOP_MAX_STEPS: Final = "步數到上限"
StopReason = Literal["梯度夠小", "步長到下限", "步數到上限", "起點不可估"]
STOP_REASONS: Final[tuple[StopReason, ...]] = (STOP_GRADIENT, STOP_STEP, STOP_MAX_STEPS, STOP_UNESTIMABLE)
# 候選確認迴圈（``material_gradient_loop``）整場的停止原因，封閉集合；字面只說做過的事：
# 「已確認」只指重調過、跟同步數重調的基準比過的候選。
LOOP_STOP_NO_IMPROVEMENT: Final = "已確認候選內沒有超過門檻"
LOOP_STOP_NO_CANDIDATES: Final = "沒有候選"
LOOP_STOP_MAX_ROUNDS: Final = "輪數到上限"
LOOP_STOP_MAX_EVALUATIONS: Final = "求值到上限"
LoopStopReason = Literal["已確認候選內沒有超過門檻", "沒有候選", "輪數到上限", "求值到上限"]
LOOP_STOP_REASONS: Final[tuple[LoopStopReason, ...]] = (
    LOOP_STOP_NO_IMPROVEMENT, LOOP_STOP_NO_CANDIDATES, LOOP_STOP_MAX_ROUNDS, LOOP_STOP_MAX_EVALUATIONS)

if TYPE_CHECKING:
    from aosr.gradient.six_category_chain import SixCategoryChainAux

# 舊鏈仍供低頻控制與既有顯卡入口；正式搜尋只建六類工廠。
type ObjectiveAux = GradientObjectiveArrays | SixCategoryChainAux
ChainObjective = Callable[[jax.Array], tuple[jax.Array, ObjectiveAux]]
Evaluator = Callable[[jax.Array], tuple[tuple[jax.Array, ObjectiveAux], jax.Array]]
# 低頻控制回三分項；正式搜尋回低頻三分項、六類與兩個總分。
Components = tuple[float, ...]
# 保留低頻三項的既有位置；之後是六類產品代價、搜尋總分、產品總分。
COMPONENT_NAMES: Final = ("own_line_ripple_rms_db", "seat_robustness_rms_db", "big_peak_proxy_db",
    "timbre_cost", "listening_cost", "reflections_cost", "reverberation_cost", "matching_cost",
    "low_frequency_cost", "objective_total", "product_total")


class ObjectiveReading(NamedTuple):
    screening_total: float
    components: Components
    objective_total: float


def objective_reading(aux: ObjectiveAux) -> ObjectiveReading:
    """同次求值讀出低頻三項與六類；不可估值留在記憶體，有限性閘門擋在寫列之前。"""
    if isinstance(aux, GradientObjectiveArrays):
        old = low_frequency_reading(aux)
        return ObjectiveReading(old.screening_total, old.components, old.objective_total)
    low = low_frequency_reading(aux.low_frequency)
    values = (*low.components, *aux.categories[:6], aux.objective_total, aux.product_total)
    components = tuple(float(x) for x in values)
    return ObjectiveReading(low.screening_total, components, float(aux.objective_total))


def estimable(value: jax.Array, aux: ObjectiveAux) -> bool:
    """總值與每個類別都有限，才可接受或落帳；不把 NaN 類別藏在有限總分後。"""
    reading = objective_reading(aux)
    return math.isfinite(float(value)) and math.isfinite(reading.screening_total) and all(
        x is not None and math.isfinite(x) for x in reading.components)
FloatVector = NDArray[np.floating]
StopCheck = Callable[[], bool]


class Unestimable(ValueError):
    """目前階段不可估，保留已完成的有限列並停止；不寫虛假的候選值。"""


class SearchStopped(Exception):
    """人叫停（停止記號）：在兩次求值之間查到 ``should_stop`` 為真就拋這個，不是停止原因。

    迴圈的停止原因是封閉集合（``LOOP_STOP_REASONS``），「人叫停」不在裡面、不寫進帳本；
    拋出時帳本停在最後一列寫完整的地方，接續照既有規則丟掉沒寫完的半輪重跑。
    """


def check_stop(should_stop: StopCheck | None) -> None:
    """每次求值之前叫一次：給了 ``should_stop`` 而它回真就拋 :class:`SearchStopped`。"""
    if should_stop is not None and should_stop():
        raise SearchStopped("停止記號：在兩次求值之間停下")


# 一個面的設計與「從 preset 組設計」那幾支住 ``surface_design``（不載入 JAX，網頁行程核綁定時也用，#1206 第 2 支）；
# 這裡照原名接出去，搜尋那一層的呼叫端不用改。


class SurfaceSnap(NamedTuple):
    """一個面取了哪個規格：最近的合法厚度、實際取的厚度與空腔、空腔有沒有被切、厚度有沒有退。"""

    surface_id: str
    nearest_thickness_m: float | None
    thickness_m: float | None
    cavity_m: float | None
    cavity_clipped: bool
    thickness_retreated: bool

    @property
    def blocked_by_total_depth(self) -> bool:
        """被總深度上限擋過（空腔被切或厚度退到較薄）。"""
        return self.cavity_clipped or self.thickness_retreated


class SnapResult(NamedTuple):
    """取規格後的可行 θ（與輸入同精度）與每面的取法。"""

    theta: FloatVector
    surfaces: tuple[SurfaceSnap, ...]


class DescentStep(NamedTuple):
    """一步：第幾步（0 是起點）、θ、值（梯度目標的新合分）、舊合分（第一階段兩分項合分）、三分項（主位起伏、
    座位穩健性的聲道平均、大峰代理）、投影梯度範數、步長、秒數。"""

    step: int
    theta: tuple[float, ...]
    value: float
    screening_total: float
    components: Components
    gradient_norm: float
    step_size: float
    seconds: float


class DescentResult(NamedTuple):
    """最後的 θ（與起點同精度）、每一步紀錄、停止原因、最後一次試的步長、這段一共求了幾次目標。

    ``evaluations`` 數的是真的叫目標（值＋梯度一起算）的次數：起點一次、線搜尋每一個試探一次
    （過了的與沒過的都算）。
    """

    theta: FloatVector
    steps: tuple[DescentStep, ...]
    stop_reason: StopReason
    final_trial_step: float
    evaluations: int
    unestimable_trials: int = 0


class DepthLimit(NamedTuple):
    """一面的總深度限制：θ 裡厚度那一格、空腔那一格（沒有空腔層是 None）、「厚度＋空腔」上限（公尺）。"""

    thickness_index: int
    cavity_index: int | None
    max_total_m: float


# ---- 扁平化 ----

def design_names(designs: Sequence[SurfaceDesign]) -> tuple[str, ...]:
    """θ 每一格的名字 ``<面代號>.<樣板變數名>``，照面序、每面先厚度後空腔；沒有樣板的面不占格。"""
    return tuple(f"{design.surface_id}.{name}" for design in designs for name in design.variables)


def design_bounds(designs: Sequence[SurfaceDesign]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """θ 的盒界（下界, 上界），雙精度；呼叫端照 θ 的精度轉。"""
    lower = [value for design in designs for value in design.lower]
    upper = [value for design in designs for value in design.upper]
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def depth_limits(designs: Sequence[SurfaceDesign]) -> tuple[DepthLimit, ...]:
    """每個有樣板的面一條總深度限制（θ 的格號照 :func:`design_names`）；剛性面沒有。"""
    limits = []
    offset = 0
    for design in designs:
        if design.material is None or design.max_total_depth_m is None:
            continue
        cavity = offset + 1 if design.cavity_variable is not None else None
        limits.append(DepthLimit(offset, cavity, design.max_total_depth_m))
        offset += len(design.variables)
    return tuple(limits)


def baseline_theta(designs: Sequence[SurfaceDesign], ranged: Collection[str] = ()) -> FloatVector:
    """每面樣板本身的厚度與空腔當起點；超出盒界就拒收（例如 preset 的空腔比家族範圍深）。

    ``ranged`` 是起跑那一族有可調範圍的面片（#1199，``fixed_surfaces.ranged_starts``）：這幾面的樣板值先用換樣板那一把
    （:func:`project_initial`：盒界 ∩ 總深，再取規格）投影進收窄後的範圍，不拒收；樣板本來就在範圍裡、總深也放得下時
    投影不動它（逐位等於樣板）。沒給（空）整份逐位照舊。"""
    values: list[float] = []
    for design in designs:
        if design.material is None:
            continue
        own = []
        for name in design.variables:
            layer_index = int(name.split(".")[0].removeprefix("layer"))
            own.append(float(getattr(design.material.layers[layer_index - 1], _THICKNESS)))
        values.extend(project_initial(np.asarray(own, dtype=np.float64), design) if design.surface_id in ranged else own)
    theta = np.asarray(values, dtype=np.float64)
    return checked_theta(theta, designs)


def project_initial(values: FloatVector, design: SurfaceDesign) -> FloatVector:
    """一面的初值（照 ``design.variables`` 的格序）投影進這一面的可行域：盒界 ∩ 總深（:func:`project_to_limits`，精確最近點），
    再取規格（:func:`snap_to_specs`：棉取合法厚度、空腔照總深切）；在 ``values`` 的精度下算、回同精度。

    換樣板（有深度上限或範圍時把家族初值投進新那一面）與有範圍的起跑（:func:`baseline_theta` 的 ``ranged``）共用這一把（#1199）。"""
    values = np.asarray(values)
    selected = (design,)
    lower, upper = design_bounds(selected)
    return snap_to_specs(project_to_limits(values, lower.astype(values.dtype), upper.astype(values.dtype),
                                           depth_limits(selected)), selected).theta


def checked_theta(theta: jax.Array | FloatVector, designs: Sequence[SurfaceDesign]) -> FloatVector:
    """具體值的 θ：一維、長度等於設計變數數、浮點、有限、在盒界內；回 numpy 陣列（精度不變）。"""
    values = np.asarray(theta)
    names = design_names(designs)
    if values.ndim != 1 or values.shape[0] != len(names):
        raise ValueError(f"θ 必須是一維、長度 {len(names)}（{list(names)}），收到形狀 {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"θ 必須是浮點陣列，收到 {values.dtype}")
    if not np.all(np.isfinite(values)):
        raise ValueError("θ 必須全是有限數")
    lower, upper = design_bounds(designs)
    outside = [name for name, value, low, high in zip(names, values, lower.astype(values.dtype),
                                                      upper.astype(values.dtype), strict=True)
               if not low <= value <= high]
    if outside:
        raise ValueError(f"θ 超出盒界：{outside}")
    return values


def surface_parameters(designs: Sequence[SurfaceDesign], theta: jax.Array) -> tuple[dict[str, jax.Array], ...]:
    """θ 切回每面的參數字典（零維陣列、與 θ 同精度）；剛性面是空字典。可在追蹤中用。"""
    values = jnp.asarray(theta)
    names = design_names(designs)
    if values.ndim != 1 or values.shape[0] != len(names):
        raise ValueError(f"θ 必須是一維、長度 {len(names)}，收到形狀 {values.shape}")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(f"θ 必須是浮點陣列，收到 {values.dtype}")
    parameters = []
    offset = 0
    for design in designs:
        parameters.append({name: values[offset + index] for index, name in enumerate(design.variables)})
        offset += len(design.variables)
    return tuple(parameters)


# ---- 整條可微鏈（替身那一層：今天只有六面） ----

def kept_impedance_rows(designs: Sequence[SurfaceDesign], frequencies: Sequence[float] | NDArray[np.float64], *,
                        air: PhysicsConstants, ranges: PorousModelRanges) -> tuple[NDArray[np.complex128] | None, ...]:
    """每面「保留原牆」那一列垂直入射阻抗（雙精度複數，``(F,)``）；其餘面 None。

    走產品 ``surface_impedance``（非可微入口），頻率軸照終點 ``material_impedances`` 同一個寫法轉成雙精度 JAX 陣列：
    替身鏈建目標時用的常數列與終點完整模型用的阻抗是同一支函式、同一個輸入算的，雙精度行程下逐位相同（θ 是單精度時，
    目標函式裡換進去的是這一列轉成 complex64 的版本）。要在雙精度已開的行程裡呼叫。
    """
    axis = jnp.asarray(np.asarray(frequencies, dtype=np.float64))
    rows: list[NDArray[np.complex128] | None] = []
    for design in designs:
        if design.kept_material is None:
            rows.append(None)
            continue
        values, _ = surface_impedance(design.kept_material, axis, air=air, ranges=ranges)
        rows.append(np.asarray(values, dtype=np.complex128))
    return tuple(rows)


def design_templates(designs: Sequence[SurfaceDesign], frequencies_hz: jax.Array, *, air: PhysicsConstants,
                     ranges: PorousModelRanges) -> tuple[ConstructionTemplate | None, ...]:
    """每面的可微樣板（:func:`design_template_material` 那份構造、設計變數照 ``design.variables``）；沒有樣板的面 None。
    用具體值建，要在雙精度已開的行程裡呼叫。"""
    return tuple(
        None if (material := design_template_material(design)) is None
        else build_template(material, frequencies_hz, air=air, ranges=ranges, variables=design.variables)
        for design in designs)


def _patch_impedances(templates: Sequence[ConstructionTemplate | None], parameters: Sequence[Mapping[str, jax.Array]],
                      axis: jax.Array, *, air: PhysicsConstants) -> TemplateImpedances:
    """面片樣板的阻抗；不貼片的零只作佔位，呼叫端隨後換入已驗過的原材料常數列。"""
    rows = tuple(jnp.zeros(axis.shape, dtype=jnp.result_type(axis.dtype, jnp.complex64)) if template is None else
                 surface_impedance_from_parameters(template, values, axis, air=air)
                 for template, values in zip(templates, parameters, strict=True))
    return TemplateImpedances(jnp.stack(rows), jnp.asarray([template is None for template in templates], dtype=bool))


# 面片阻抗的公開別名；原搜尋仍使用同一份既有實作與私名。
patch_impedances = _patch_impedances


def chain_objective(model: ReducedModel, designs: Sequence[SurfaceDesign],
                    objective_constants: GradientObjectiveConstants, *, air: PhysicsConstants,
                    ranges: PorousModelRanges, frequencies: Sequence[float] | NDArray[np.float64],
                    frequency_batch: int | None = None, single_precision: bool = False,
                    device_constants: ReducedConstants | None = None) -> Callable[[jax.Array], tuple[jax.Array, GradientObjectiveArrays]]:
    """回 ``θ → (objective_total, 各分項)`` 的純函式，可 ``jax.jit``／``jax.grad``、預設精度跟 θ 走。

    顯卡模式的材料與替身求解維持單精度，聲壓轉成 complex128 後才進既有評分；帳本恆等式仍用雙精度守。
    ``device_constants`` 是同一替身已轉單精度的裝置常數；搜尋工廠明確傳入，所有配置共用。

    θ → 每面參數 → 垂直入射阻抗（``wall_impedances_from_templates``）→ 替身聲壓（``reduced_pressures``）
    → 梯度目標（``gradient_objective_from_pressures``：第一階段兩分項合分原樣，加權重 × 大峰代理；#561 壬一）。
    梯度沿新合分走；舊合分（``screening_total``）同一次求值一起交回。樣板用具體值建，要在雙精度已開的行程裡呼叫；
    設計清單照替身的有序面片清單、只有一種安裝方式（:func:`designs_mount`，混了拒收）。
    樣板照 :func:`design_templates`（貼在原牆上的面背後接原牆）。保留原牆的面：建目標時用 :func:`kept_impedance_rows` 算一次當常數列，
    目標函式裡轉成 θ 對應的複數精度、換進那一列，剛性遮罩那一格是 False；求值入口的語意（None 樣板＝剛性）不動。
    """
    require_matching_patches(model, tuple(d.surface_id for d in designs))
    for design, (_, zone) in zip(designs, model.patch_keys, strict=True):
        if zone and design.material is None and design.kept_material is None:
            raise ValueError(f"分區面片 {design.surface_id} 一定要有材料")
    designs_mount(designs)
    axis = np.asarray(frequencies, dtype=np.float64)
    if axis.shape != (len(model.frequencies_hz),) or not np.array_equal(axis, np.asarray(model.frequencies_hz)):
        raise ValueError("頻率軸必須等於替身的頻率軸")
    if objective_constants.log2_frequencies.shape != axis.shape:
        raise ValueError("目標常數的頻率軸長度跟替身不同")
    constants = reduced_constants(model)
    double_axis = jnp.asarray(axis)
    templates = design_templates(designs, double_axis, air=air, ranges=ranges)
    frozen_designs = tuple(designs)
    kept_rows = tuple((index, row) for index, row in enumerate(kept_impedance_rows(designs, axis, air=air, ranges=ranges))
                      if row is not None)
    not_kept = jnp.asarray([design.kept_material is None for design in designs], dtype=bool)

    def objective(theta: jax.Array, reduced: ReducedConstants = constants) -> tuple[jax.Array, GradientObjectiveArrays]:
        values = jnp.asarray(theta)
        parameters = surface_parameters(frozen_designs, values)
        frequency_axis = jnp.asarray(reduced.frequencies_hz, dtype=values.dtype)
        impedances = (_patch_impedances(templates, parameters, frequency_axis, air=air)
                      if any(zone for _, zone in model.patch_keys) else
                      wall_impedances_from_templates(templates, parameters, frequency_axis, air=air))
        walls = impedances.values
        for index, row in kept_rows:
            walls = walls.at[index].set(jnp.asarray(row, dtype=walls.dtype))
        rigid = jnp.logical_and(impedances.rigid_mask, not_kept)
        pressures = reduced_pressures(reduced, walls, rigid_mask=rigid, frequency_batch=frequency_batch)
        if single_precision:
            pressures = pressures.astype(jnp.complex128)
        result = gradient_objective_from_pressures(objective_constants, pressures)
        return result.objective_total, result

    return SinglePrecisionObjective(objective, constants, device_constants=device_constants) if single_precision else objective


# ---- 投影梯度下降 ----

# 編譯好的評估函式掛在目標函式物件自己身上的屬性名。
_EVALUATOR_ATTRIBUTE: Final = "aosr_compiled_evaluator"


def compiled_evaluator(objective: ChainObjective) -> Evaluator:
    """``θ → ((值, 分項), 梯度)`` 的編譯版；同一個目標函式物件只包一次 ``jax.jit``，之後每段下降共用同一份編譯。

    每段下降都重新包一次會每段重新追蹤、重新編譯（候選確認一輪最多四段）；共用不改數值：同一個函式、
    同一個形狀與精度，編譯出同一份程式。編譯版掛在目標函式自己的屬性上，不放模組層的表：編譯版強抓著目標，
    放在以目標為鍵的弱參照表裡鍵永遠不會死（#561 審查）；掛在自己身上是「目標 → 編譯版 → 目標」的循環，
    外面沒人抓著目標時垃圾回收連編譯版一起收。目標必須是能掛屬性的函式物件（``def`` 或 lambda）。
    """
    if isinstance(objective, SinglePrecisionObjective):
        return objective.evaluate
    cached: Evaluator | None = getattr(objective, _EVALUATOR_ATTRIBUTE, None)
    if cached is not None:
        return cached
    evaluator: Evaluator = jax.jit(jax.value_and_grad(objective, has_aux=True))
    setattr(objective, _EVALUATOR_ATTRIBUTE, evaluator)
    return evaluator


DEPTH_EDGE_ULPS: Final = 4


def on_depth_edge(values: FloatVector, limit: DepthLimit) -> bool:
    """這一面貼在總深度那條邊上：「厚度＋空腔」（沒有空腔層就是厚度）不小於上限減 ``DEPTH_EDGE_ULPS`` 個
    上限的浮點間距（在 θ 的精度下算）。

    投影與投影梯度共用這一個契約：:func:`project_to_limits` 投到斜邊上的點（含空腔切到下界的那個角）
    一定滿足它，:func:`_projected_gradient_norm` 也只用它判哪一面貼邊。容差是浮點捨入的量級：斜邊上的點
    「上限減厚度」往下退成可行時離上限一兩個間距；離邊超過容差的內點不算貼邊（#561 審查 B1）。
    """
    dtype = values.dtype.type
    top = dtype(limit.max_total_m)
    total = values[limit.thickness_index] + (values[limit.cavity_index] if limit.cavity_index is not None
                                             else dtype(0.0))
    return bool(total >= top - dtype(DEPTH_EDGE_ULPS) * np.spacing(top))


def _within_limits(values: FloatVector, limits: Sequence[DepthLimit]) -> bool:
    """每面「厚度＋空腔 ≤ 上限」（在 θ 的精度下算，跟 :func:`is_feasible` 同一個判法）。"""
    dtype = values.dtype.type
    return all(values[limit.thickness_index] + (values[limit.cavity_index] if limit.cavity_index is not None
                                                else dtype(0.0)) <= dtype(limit.max_total_m) for limit in limits)


def _checked_limits(limits: Sequence[DepthLimit], size: int) -> tuple[DepthLimit, ...]:
    used: list[int] = []
    for limit in limits:
        indices = [limit.thickness_index] + ([] if limit.cavity_index is None else [limit.cavity_index])
        if any(not 0 <= index < size for index in indices) or not limit.max_total_m > 0.0:
            raise ValueError(f"總深度限制的格號要在 θ 長度 {size} 內、上限要是正數：{limit}")
        used.extend(indices)
    if len(set(used)) != len(used):
        raise ValueError("一格 θ 最多屬於一條總深度限制")
    return tuple(limits)


def project_to_limits(moved: FloatVector, lower: FloatVector, upper: FloatVector,
                      limits: Sequence[DepthLimit]) -> FloatVector:
    """盒界 ∩ 每面「厚度＋空腔 ≤ 上限」的精確最近點（在 ``moved`` 的精度下算，回同精度）。

    每面是一個二維凸多邊形，面與面之間沒有共用變數，所以逐面各自投。先投到盒界；盒界上的點已在斜邊內，
    它就是答案（多邊形是盒的子集）。否則最近點一定在斜邊上（盒的最近點不在半平面裡，最近點就落在半平面的邊界），
    在「斜邊 ∩ 盒界」那一段線段上找：原點投到直線 t＋c＝上限、再夾到線段的兩個端點之間（端點也收）。
    空腔取「上限減厚度」在這個精度下不超過上限的最大值；沒有空腔層的面只把厚度夾到上限。
    """
    values = np.asarray(moved)
    dtype = values.dtype.type
    low = np.asarray(lower, dtype=values.dtype)
    high = np.asarray(upper, dtype=values.dtype)
    result = np.clip(values, low, high)
    for limit in limits:
        thickness_index, cavity_index, top = limit.thickness_index, limit.cavity_index, dtype(limit.max_total_m)
        if cavity_index is None:
            result[thickness_index] = min(result[thickness_index], top)
            continue
        if result[thickness_index] + result[cavity_index] <= top:
            continue
        thickness_low = max(low[thickness_index], top - high[cavity_index])
        thickness_high = min(high[thickness_index], top - low[cavity_index])
        if thickness_low > thickness_high:
            raise ValueError(f"總深度限制 {limit} 跟盒界沒有交集")
        along = (values[thickness_index] + top - values[cavity_index]) / dtype(2.0)
        thickness = min(max(along, thickness_low), thickness_high)
        cavity = min(_depth_room(top, thickness), high[cavity_index])
        if cavity < low[cavity_index]:
            # 空腔切到下界：空腔就放在下界（角上，下界那條限制也要判得到作用），厚度回算成「上限減下界」的
            # 最大可行值；這個角離上限在捨入量級內，on_depth_edge 判得到貼斜邊（#561 審查 B1）。
            cavity = low[cavity_index]
            thickness = min(_depth_room(top, cavity), high[thickness_index])
        result[thickness_index], result[cavity_index] = thickness, cavity
    return result


def _cone_projection(direction: NDArray[np.float64], normals: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    """把一個方向投到「每條作用中限制的法向 · 方向 ≤ 0」的錐上（二維，逐一試每條邊與頂點、取最近的可行點）。"""
    if all(float(normal @ direction) <= 0.0 for normal in normals):
        return direction
    best, best_distance = np.zeros_like(direction), float(direction @ direction)
    for index, normal in enumerate(normals):
        point = direction - float(normal @ direction) / float(normal @ normal) * normal
        feasible = all(float(other @ point) <= 0.0 for position, other in enumerate(normals) if position != index)
        distance = float((direction - point) @ (direction - point))
        if feasible and distance < best_distance:
            best, best_distance = point, distance
    return best


def _projected_gradient_norm(theta: jax.Array, gradient: jax.Array, lower: jax.Array, upper: jax.Array,
                             limits: Sequence[DepthLimit] = ()) -> float:
    """可下降方向的梯度範數：貼著界還往外推的分量去掉。

    盒界逐格：貼著下界還往外推（梯度為正）、貼著上界還往外推（梯度為負）的分量歸零。貼在斜邊上的面
    （空腔已是「上限減厚度」）改成整面一起投：負梯度投到那一面作用中限制（盒界與斜邊，法向 (1,1)/√2）
    圍成的錐上，往外推的法向分量去掉——不然在斜邊上永遠停不下來。沒有任何一面貼斜邊時跟只守盒界逐位相同。
    """
    blocked = ((theta <= lower) & (gradient > 0)) | ((theta >= upper) & (gradient < 0))
    masked = jnp.where(blocked, jnp.zeros_like(gradient), gradient)
    values, raw = np.asarray(theta), np.asarray(gradient)
    on_edge = [limit for limit in limits if on_depth_edge(values, limit)]
    if not on_edge:
        return float(jnp.linalg.norm(masked))
    adjusted = np.asarray(masked).copy()
    low, high = np.asarray(lower), np.asarray(upper)
    for limit in on_edge:
        indices = [limit.thickness_index] + ([] if limit.cavity_index is None else [limit.cavity_index])
        normals = [np.ones(len(indices))]
        for axis, index in enumerate(indices):
            unit = np.zeros(len(indices))
            unit[axis] = 1.0
            if values[index] <= low[index]:
                normals.append(-unit)
            if values[index] >= high[index]:
                normals.append(unit)
        descent = -np.asarray(raw[indices], dtype=np.float64)
        adjusted[indices] = -_cone_projection(descent, normals)
    return float(jnp.linalg.norm(jnp.asarray(adjusted)))


def tied_evaluator(evaluate: Evaluator, cells: Cells) -> Evaluator:
    """把求值器包一層（#1206）：成對兩格的梯度換成兩格的**平均**（把梯度投到「兩格相等」的子空間；不是相加——相加是
    共用一格的鏈鎖律，在各片保留自己那一格的這裡會讓步長變兩倍）。沒有成對的格就原樣交回，不多一層。"""
    if not cells:
        return evaluate
    firsts, seconds = np.asarray([cell[0] for cell in cells]), np.asarray([cell[1] for cell in cells])

    def averaged(theta: jax.Array) -> tuple[tuple[jax.Array, ObjectiveAux], jax.Array]:
        output, raw = evaluate(theta)
        gradient = jnp.asarray(raw)
        mean = (gradient[firsts] + gradient[seconds]) / 2
        return output, gradient.at[firsts].set(mean).at[seconds].set(mean)

    return averaged


def _descent_step(index: int, theta: jax.Array, value: jax.Array, aux: ObjectiveAux, gradient_norm: float,
                  step: float, seconds: float) -> DescentStep:
    """一步的紀錄：值（新合分）照目標函式交回的那一個，舊合分與三分項從同一次求值的分項讀。"""
    reading = objective_reading(aux)
    return DescentStep(index, tuple(float(x) for x in np.asarray(theta)), float(value), reading.screening_total,
                       reading.components, gradient_norm, step, seconds)


def descent_start(theta0: jax.Array | FloatVector, bounds: tuple[FloatVector, FloatVector],
                  limits: Sequence[DepthLimit] = ()) -> jax.Array:
    """實際第一次求值的 θ：保留輸入精度、檢查盒界，超過總深度才投回；把關與下降共用。"""
    start = jnp.asarray(theta0)
    if start.ndim != 1 or not jnp.issubdtype(start.dtype, jnp.floating):
        raise ValueError("θ 起點必須是一維浮點陣列")
    dtype = start.dtype
    lower = jnp.asarray(np.asarray(bounds[0]), dtype=dtype)
    upper = jnp.asarray(np.asarray(bounds[1]), dtype=dtype)
    if lower.shape != start.shape or upper.shape != start.shape:
        raise ValueError(f"盒界長度必須等於 θ 長度 {start.shape[0]}")
    if not bool(jnp.all(jnp.isfinite(start))) or not bool(jnp.all((lower <= start) & (start <= upper))):
        raise ValueError("θ 起點必須有限且在盒界內")
    limits = _checked_limits(limits, start.shape[0])
    lower_values, upper_values = np.asarray(lower), np.asarray(upper)
    if not _within_limits(np.asarray(start), limits):
        # 起點超過總深度上限就先投回可行域（精確最近點）：預設樣板「50 mm 棉板＋100 mm 空腔」在雙精度是
        # 0.15000000000000002，比上限多一個末位；盒界外仍照舊拒收。
        start = jnp.asarray(project_to_limits(np.asarray(start), lower_values, upper_values, limits))
    return start


def _descend(objective: ChainObjective, theta0: jax.Array | FloatVector,
             bounds: tuple[FloatVector, FloatVector], settings: DescentSettings, *,
             max_steps: int, stop_on_gradient: bool, limits: Sequence[DepthLimit] = (),
             should_stop: StopCheck | None = None, tied_cells: Cells = ()) -> DescentResult:
    start = descent_start(theta0, bounds, limits)
    check_tied_equal(np.asarray(start), tied_cells, "下降起點")
    dtype = start.dtype
    lower = jnp.asarray(np.asarray(bounds[0]), dtype=dtype)
    upper = jnp.asarray(np.asarray(bounds[1]), dtype=dtype)
    limits = tuple(limits)
    lower_values, upper_values = np.asarray(lower), np.asarray(upper)
    evaluate = tied_evaluator(compiled_evaluator(objective), tied_cells)

    def record(index: int, theta: jax.Array, value: jax.Array, aux: ObjectiveAux,
               gradient: jax.Array, step: float, seconds: float) -> DescentStep:
        check_tied_equal(np.asarray(theta), tied_cells, f"下降第 {index} 步")
        return _descent_step(index, theta, value, aux, _projected_gradient_norm(theta, gradient, lower, upper, limits),
                             step, seconds)

    check_stop(should_stop)
    clock = time.perf_counter()
    (value, aux), gradient = evaluate(start)
    evaluations = 1
    if not estimable(value, aux) or not np.all(np.isfinite(np.asarray(gradient))):
        return DescentResult(np.asarray(start), (), STOP_UNESTIMABLE, 0.0, evaluations)
    unestimable_trials = 0
    theta = start
    steps = [record(0, theta, value, aux, gradient, 0.0, time.perf_counter() - clock)]
    trial = 0.0
    reason: StopReason
    while True:
        if stop_on_gradient and steps[-1].gradient_norm < settings.gradient_norm_stop:
            reason = STOP_GRADIENT
            break
        if len(steps) - 1 >= max_steps:
            reason = STOP_MAX_STEPS
            break
        clock = time.perf_counter()
        trial = settings.initial_step
        accepted = False
        while trial >= settings.min_step:
            moved = theta - jnp.asarray(trial, dtype=dtype) * gradient
            candidate = (jnp.clip(moved, lower, upper) if not limits else
                         jnp.asarray(project_to_limits(np.asarray(moved), lower_values, upper_values, limits)))
            check_stop(should_stop)
            (new_value, new_aux), new_gradient = evaluate(candidate)
            evaluations += 1
            # 投影弧上的 Armijo：梯度乘實際位移。投到凸集合上的位移跟梯度的內積不大於零（投影的性質），
            # 所以右邊不大於舊值；只守盒界時每一格位移與梯度異號或為零。
            predicted = float(jnp.sum(gradient * (candidate - theta)))
            valid = estimable(new_value, new_aux)
            if not valid:
                unestimable_trials += 1
            if valid and float(new_value) <= float(value) + settings.armijo_constant * predicted:
                accepted = True
                break
            trial *= settings.backtrack_ratio
        if not accepted:
            reason = STOP_STEP
            break
        theta, value, aux, gradient = candidate, new_value, new_aux, new_gradient
        steps.append(record(len(steps), theta, value, aux, gradient, trial, time.perf_counter() - clock))
    return DescentResult(np.asarray(theta), tuple(steps), reason, trial, evaluations, unestimable_trials)


def projected_descent(objective: ChainObjective, theta0: jax.Array | FloatVector,
                      bounds: tuple[FloatVector, FloatVector], settings: DescentSettings, *,
                      limits: Sequence[DepthLimit], should_stop: StopCheck | None = None,
                      tied_cells: Cells = ()) -> DescentResult:
    """投影到「盒界 ∩ 每面厚度＋空腔 ≤ 總深度上限」的梯度下降＋回溯線搜尋（Armijo）。

    ``limits`` 一般是 :func:`depth_limits` 由設計清單算出來的；空的就只守盒界（合成目標的考卷用）。
    每一步的試探點用 :func:`project_to_limits` 投回可行域（精確最近點）；起點要在盒界內，超過總深度上限的先投回。
    每一步從 ``initial_step`` 起試、沒過就乘 ``backtrack_ratio``；步長小於 ``min_step`` 仍沒過就不走這一步、
    停在「步長到下限」。投影梯度範數（貼界、貼斜邊還往外推的分量去掉）小於 ``gradient_norm_stop`` 停在
    「梯度夠小」；走滿 ``max_steps`` 停在「步數到上限」。值單調不增；同一起點重跑逐位相同（秒數除外）。
    取規格那一步照舊保留：厚度仍要取合法值，連續段只是不再跑到總深度外面。
    給了 ``should_stop`` 就在每次求值之前查一次，為真拋 :class:`SearchStopped`（這一段不交出結果）。
    ``tied_cells``（#1206，:func:`aosr.gradient.ties.tied_cells`）是成對的 θ 格：那兩格的梯度換成兩格的平均，試探點、
    預測降幅、投影梯度範數都用它；起點與每一步都檢查成對的格逐位相等，不等拋 ``TieBroken``。空的就跟沒有這一格一樣。
    """
    return _descend(objective, theta0, bounds, settings, max_steps=settings.max_steps, stop_on_gradient=True,
                    limits=limits, should_stop=should_stop, tied_cells=tied_cells)


# ---- 取規格與空腔重調 ----

def _depth_room(max_total: np.floating, thickness: np.floating) -> np.floating:
    """在 θ 的精度下，厚度加上它不超過總深度上限的最大空腔（可能是負的）。"""
    room = max_total - thickness
    while thickness + room > max_total:
        room = np.nextafter(room, -np.inf)
    return room


def depth_room(max_total: np.floating, thickness: np.floating) -> np.floating:
    """公開版的 :func:`_depth_room`：給候選產生用同一條「空腔切到上限減厚度」的規則（在 θ 的精度下算）。"""
    return _depth_room(max_total, thickness)


def snap_to_specs(theta: jax.Array | FloatVector, designs: Sequence[SurfaceDesign],
                  ties: Sequence[tuple[str, str]] = ()) -> SnapResult:
    """棉板厚度取可行的最近合法值，空腔維持連續、總深度上限在取規格時一起滿足。

    先取最近（一樣近取較薄）；「厚度＋空腔」超過那面的總深度上限就把空腔切到上限減厚度；
    空腔切到下界仍超過（或沒有空腔層、厚度本身就超過）就退到下一個較薄的合法厚度。
    不是各面各自取最近就交付：取完之後由 :func:`retune_cavities` 聯合重調空腔。
    ``ties`` 是綁定表（#1206）：取完之後成對兩片的格要逐位相等，不等拋 ``TieBroken``。
    """
    values = checked_theta(theta, designs).copy()
    dtype = values.dtype.type
    choices = []
    offset = 0
    for design in designs:
        if design.material is None or design.max_total_depth_m is None:
            choices.append(SurfaceSnap(design.surface_id, None, None, None, False, False))
            continue
        has_cavity = design.cavity_variable is not None
        thickness = values[offset]
        legal = [dtype(value) for value in design.legal_thicknesses_m]
        nearest = min(range(len(legal)), key=lambda index: (abs(legal[index] - thickness), index))
        max_total = dtype(design.max_total_depth_m)
        cavity = values[offset + 1] if has_cavity else dtype(0.0)
        cavity_low = dtype(design.lower[1]) if has_cavity else dtype(0.0)
        chosen: int | None = None
        new_cavity = cavity
        for index in range(nearest, -1, -1):
            room = _depth_room(max_total, legal[index])
            if room >= cavity_low:
                chosen, new_cavity = index, min(cavity, room)
                break
        if chosen is None:
            raise ValueError(f"面 {design.surface_id} 沒有一個合法厚度放得進總深度上限")
        values[offset] = legal[chosen]
        if has_cavity:
            values[offset + 1] = new_cavity
        choices.append(SurfaceSnap(
            design.surface_id, float(legal[nearest]), float(legal[chosen]),
            float(new_cavity) if has_cavity else None, bool(has_cavity and new_cavity < cavity), chosen != nearest))
        offset += len(design.variables)
    check_ties_hold(values, designs, ties, "取規格之後")
    return SnapResult(values, tuple(choices))


def is_feasible(theta: jax.Array | FloatVector, designs: Sequence[SurfaceDesign]) -> bool:
    """θ 在盒界內、每面厚度恰是合法值、厚度＋空腔不超過總深度上限（在 θ 的精度下算）。"""
    values = checked_theta(theta, designs)
    dtype = values.dtype.type
    offset = 0
    for design in designs:
        if design.material is None or design.max_total_depth_m is None:
            continue
        thickness = values[offset]
        cavity = values[offset + 1] if design.cavity_variable is not None else dtype(0.0)
        if thickness not in [dtype(value) for value in design.legal_thicknesses_m]:
            return False
        if thickness + cavity > dtype(design.max_total_depth_m):
            return False
        offset += len(design.variables)
    return True


def retune_bounds(theta_feasible: jax.Array | FloatVector,
                  designs: Sequence[SurfaceDesign]) -> tuple[FloatVector, FloatVector]:
    """重調用的盒界（與 θ 同精度）：厚度上下界都釘在取到的規格；空腔上界再收到「總深度上限減厚度」。"""
    values = checked_theta(theta_feasible, designs)
    if not is_feasible(values, designs):
        raise ValueError("重調的起點必須是取過規格的可行 θ")
    dtype = values.dtype.type
    lower, upper = (bound.astype(values.dtype) for bound in design_bounds(designs))
    offset = 0
    for design in designs:
        if design.material is None or design.max_total_depth_m is None:
            continue
        lower[offset] = upper[offset] = values[offset]
        if design.cavity_variable is not None:
            upper[offset + 1] = min(upper[offset + 1], _depth_room(dtype(design.max_total_depth_m), values[offset]))
        offset += len(design.variables)
    return lower, upper


def retune_cavities(objective: ChainObjective, theta_feasible: jax.Array | FloatVector,
                    designs: Sequence[SurfaceDesign], settings: MaterialGradientSettings, *,
                    should_stop: StopCheck | None = None, ties: Sequence[tuple[str, str]] = ()) -> DescentResult:
    """厚度固定在規格、所有面的空腔一起走同一個投影下降，**最多** ``[retune].steps`` 步。

    不因梯度小提早停；線搜尋退到步長下限仍沒過就不走這一步、停在「步長到下限」（起點就不可下降時是 0 步、
    θ 原樣），停止原因與最後一次試的步長都交回、進帳本；走滿上限停在「步數到上限」。不硬湊步數。
    厚度的上下界相等，投影把它釘回原值、逐位不變；空腔上界收在總深度內，所以每一步都可行。
    ``should_stop`` 同 :func:`projected_descent`：每次求值之前查一次。``ties`` 是綁定表（#1206）：成對兩片的空腔用平均梯度
    一起走，進來時與每一步都檢查逐位相等。
    """
    bounds = retune_bounds(theta_feasible, designs)
    check_ties_hold(np.asarray(theta_feasible), designs, ties, "空腔重調起點")
    return _descend(objective, np.asarray(theta_feasible), bounds, settings.descent, max_steps=settings.retune.steps,
                    stop_on_gradient=False, should_stop=should_stop, tied_cells=tied_cells(designs, ties))
