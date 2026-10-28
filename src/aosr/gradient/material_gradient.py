"""第三階段牆面材料梯度搜尋的前半段（#561）：整條可微鏈、投影梯度下降、取規格聯合重調空腔。

找法照決策紙 ``stage-three-gradient-objective-low-frequency-lane.md``「決定」第 3 條：全牆聯合梯度為主；
連續下降每一步投影到「盒界 ∩ 每面厚度＋空腔 ≤ 總深度上限」（老闆 2026-10-28 回 A）；連續厚度轉合法規格時，
總深度限制在取規格時一起滿足、之後空腔聯合重調，不准各面各自取最近就交付。
候選確認迴圈與接續在 ``material_gradient_loop``；終點完整模型重算不在這裡（下一支）。

設計以「一串面（surfaces）」寫，長度不寫死：扁平化、下降、取規格、重調與帳本對任何面數都成立。
只有最底下接替身那一層（:func:`chain_objective`）因為替身現在只有六面，才要求剛好照 ``Wall.all()``
六面給、長度不對就拒收——那是替身的限制，不是設計的限制。

設計變數只有每面棉板厚度（多孔層）與空腔深度（空氣層），樣板有那一層才有那個變數；
流阻與板材幾何固定、每面樣板固定、剛性面（None）沒有變數。θ 是一維浮點陣列，順序照面序、
每面先厚度後空腔，:func:`design_names` 列出來。精度跟 θ 走：雙精度 θ 給雙精度目標，
單精度 θ 給單精度；本模組不碰 JAX 全域設定。純函式，沒有命令列、沒有行程、不印東西。
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, NamedTuple
from weakref import WeakKeyDictionary

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter

from aosr.config.construction_presets import ConstructionPreset
from aosr.config.material_gradient import DescentSettings, MaterialGradientSettings
from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.geometry.shoebox import Wall
from aosr.materials.construction_diff import ConstructionTemplate, build_template, variable_name
from aosr.materials.construction_models import AirLayer, Assembly, PorousLayer, SurfaceMaterial
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_jax import reduced_constants, reduced_pressures, wall_impedances_from_templates
from aosr.scoring.stage_one_objective_jax import (
    StageOneObjectiveArrays, StageOneObjectiveConstants, objective_from_pressures,
)


STOP_GRADIENT: Final = "梯度夠小"
STOP_STEP: Final = "步長到下限"
STOP_MAX_STEPS: Final = "步數到上限"
StopReason = Literal["梯度夠小", "步長到下限", "步數到上限"]
STOP_REASONS: Final[tuple[StopReason, ...]] = (STOP_GRADIENT, STOP_STEP, STOP_MAX_STEPS)
# 候選確認迴圈（``material_gradient_loop``）整場的停止原因，封閉集合；字面只說做過的事：
# 「已確認」只指重調過、跟同步數重調的基準比過的候選。
LOOP_STOP_NO_IMPROVEMENT: Final = "已確認候選內沒有超過門檻"
LOOP_STOP_NO_CANDIDATES: Final = "沒有候選"
LOOP_STOP_MAX_ROUNDS: Final = "輪數到上限"
LOOP_STOP_MAX_EVALUATIONS: Final = "求值到上限"
LoopStopReason = Literal["已確認候選內沒有超過門檻", "沒有候選", "輪數到上限", "求值到上限"]
LOOP_STOP_REASONS: Final[tuple[LoopStopReason, ...]] = (
    LOOP_STOP_NO_IMPROVEMENT, LOOP_STOP_NO_CANDIDATES, LOOP_STOP_MAX_ROUNDS, LOOP_STOP_MAX_EVALUATIONS)
_THICKNESS = "thickness_m"

ChainObjective = Callable[[jax.Array], tuple[jax.Array, StageOneObjectiveArrays]]
Evaluator = Callable[[jax.Array], tuple[tuple[jax.Array, StageOneObjectiveArrays], jax.Array]]
FloatVector = NDArray[np.floating]


@dataclass(frozen=True)
class SurfaceDesign:
    """一個面的設計：樣板（剛性面是 None）、哪些樣板變數是設計變數、各自上下界、合法厚度與總深度上限。

    ``variables`` 照「厚度、空腔」的順序，名字是樣板的 ``layer<層號>.<欄位>``；
    厚度界是合法厚度的最薄到最厚，空腔界是家族的空腔範圍（公尺）。
    """

    surface_id: str
    preset_id: str | None
    family_id: str | None
    material: Assembly | None
    variables: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    thickness_variable: str | None
    cavity_variable: str | None
    legal_thicknesses_m: tuple[float, ...]
    max_total_depth_m: float | None

    def __post_init__(self) -> None:
        if not self.surface_id.strip():
            raise ValueError("面代號不可空白")
        if self.material is None:
            if (self.preset_id is not None or self.family_id is not None or self.variables or self.lower
                    or self.upper or self.thickness_variable is not None or self.cavity_variable is not None
                    or self.legal_thicknesses_m or self.max_total_depth_m is not None):
                raise ValueError(f"剛性面 {self.surface_id} 不收樣板、變數、界或規格")
            return
        kinds = _variable_kinds(self.material)
        unknown = [name for name in self.variables if name not in kinds]
        if unknown:
            raise ValueError(f"設計變數名樣板沒有：{unknown}；這個樣板可當設計變數的是 {sorted(kinds)}")
        if self.thickness_variable is None or kinds.get(self.thickness_variable) is not PorousLayer:
            raise ValueError(f"面 {self.surface_id} 的厚度變數必須是樣板裡多孔層的厚度")
        if self.cavity_variable is not None and kinds.get(self.cavity_variable) is not AirLayer:
            raise ValueError(f"面 {self.surface_id} 的空腔變數必須是樣板裡空氣層的厚度")
        expected = tuple(name for name in (self.thickness_variable, self.cavity_variable) if name is not None)
        if self.variables != expected:
            raise ValueError(f"面 {self.surface_id} 的設計變數必須照厚度、空腔排：{list(expected)}")
        if len(self.lower) != len(self.variables) or len(self.upper) != len(self.variables):
            raise ValueError(f"面 {self.surface_id} 的上下界數目要等於設計變數數")
        if any(not np.isfinite(low) or not np.isfinite(high) or low > high
               for low, high in zip(self.lower, self.upper, strict=True)):
            raise ValueError(f"面 {self.surface_id} 的上下界必須有限且下界不大於上界")
        legal = self.legal_thicknesses_m
        if not legal or list(legal) != sorted(set(legal)) or self.max_total_depth_m is None:
            raise ValueError(f"面 {self.surface_id} 要有由薄到厚的合法厚度與總深度上限")
        if self.lower[0] != legal[0] or self.upper[0] != legal[-1]:
            raise ValueError(f"面 {self.surface_id} 的厚度界必須是合法厚度的最薄到最厚")


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
    """一步：第幾步（0 是起點）、θ、值、兩分項（主位起伏、座位穩健性的聲道平均）、投影梯度範數、步長、秒數。"""

    step: int
    theta: tuple[float, ...]
    value: float
    components: tuple[float, float]
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


class DepthLimit(NamedTuple):
    """一面的總深度限制：θ 裡厚度那一格、空腔那一格（沒有空腔層是 None）、「厚度＋空腔」上限（公尺）。"""

    thickness_index: int
    cavity_index: int | None
    max_total_m: float


# ---- 一串面：從 preset 組設計 ----

def _variable_kinds(material: Assembly) -> dict[str, type]:
    """這個樣板可當設計變數的名字 → 那一層的型別（只有多孔層與空氣層的厚度）。"""
    return {variable_name(index, _THICKNESS): type(layer)
            for index, layer in enumerate(material.layers, start=1) if isinstance(layer, PorousLayer | AirLayer)}


def wall_surface_ids() -> tuple[str, ...]:
    """替身那六面的面代號，照 ``Wall.all()`` 的順序；給今天只有六面的呼叫端用。"""
    return tuple(wall.wall_name() for wall in Wall.all())


def _surface_design(surface_id: str, preset_id: str, settings: MaterialGradientSettings,
                    presets: Mapping[str, ConstructionPreset]) -> SurfaceDesign:
    preset = presets.get(preset_id)
    if preset is None:
        raise ValueError(f"preset id 不認識：{preset_id}")
    family = settings.family_of(preset_id)
    if family is None:
        raise ValueError(f"preset {preset_id} 不在家族清單裡（設定檔 [[family]]）")
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    material = adapter.validate_python(preset.material)
    if not isinstance(material, Assembly):
        raise ValueError(f"preset {preset_id} 不是多層構造，沒有厚度可當設計變數")
    porous = [index for index, layer in enumerate(material.layers, start=1) if isinstance(layer, PorousLayer)]
    air = [index for index, layer in enumerate(material.layers, start=1) if isinstance(layer, AirLayer)]
    if len(porous) != 1 or len(air) > 1 or len(porous) + len(air) != len(material.layers):
        raise ValueError(f"preset {preset_id} 必須是一層棉板、最多一層空腔，沒有別的層")
    thickness = variable_name(porous[0], _THICKNESS)
    cavity = variable_name(air[0], _THICKNESS) if air else None
    lower = [family.thicknesses_m[0]]
    upper = [family.thicknesses_m[-1]]
    if cavity is not None:
        lower.append(family.cavity_range_m[0])
        upper.append(family.cavity_range_m[1])
    return SurfaceDesign(
        surface_id=surface_id, preset_id=preset_id, family_id=family.id, material=material,
        variables=tuple(name for name in (thickness, cavity) if name is not None),
        lower=tuple(lower), upper=tuple(upper), thickness_variable=thickness, cavity_variable=cavity,
        legal_thicknesses_m=family.thicknesses_m, max_total_depth_m=family.max_total_depth_m)


def designs_from_presets(surfaces: Sequence[tuple[str, str | None]], settings: MaterialGradientSettings,
                         presets: Sequence[ConstructionPreset]) -> tuple[SurfaceDesign, ...]:
    """一串（面代號, preset id 或 None）→ 每面的設計；None 是剛性面。面數不限，面代號不可重複。"""
    if not surfaces:
        raise ValueError("至少要有一個面")
    ids = [surface_id for surface_id, _ in surfaces]
    if len(set(ids)) != len(ids):
        raise ValueError(f"面代號重複：{ids}")
    by_id = {preset.id: preset for preset in presets}
    return tuple(
        SurfaceDesign(surface_id, None, None, None, (), (), (), None, None, (), None) if preset_id is None
        else _surface_design(surface_id, preset_id, settings, by_id)
        for surface_id, preset_id in surfaces)


def wall_designs_from_presets(preset_ids: Sequence[str | None], settings: MaterialGradientSettings,
                              presets: Sequence[ConstructionPreset]) -> tuple[SurfaceDesign, ...]:
    """今天的替身只有六面：照 ``Wall.all()`` 給六個 preset id（或 None），面代號是牆名。"""
    names = wall_surface_ids()
    if len(preset_ids) != len(names):
        raise ValueError(f"六面牆要給 {len(names)} 個 preset id（照 Wall.all()），收到 {len(preset_ids)} 個")
    return designs_from_presets(tuple(zip(names, preset_ids, strict=True)), settings, presets)


# ---- 扁平化 ----

def design_names(designs: Sequence[SurfaceDesign]) -> tuple[str, ...]:
    """θ 每一格的名字 ``<面代號>.<樣板變數名>``，照面序、每面先厚度後空腔；剛性面不占格。"""
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


def baseline_theta(designs: Sequence[SurfaceDesign]) -> FloatVector:
    """每面樣板本身的厚度與空腔當起點；超出盒界就拒收（例如 preset 的空腔比家族範圍深）。"""
    values = []
    for design in designs:
        if design.material is None:
            continue
        for name in design.variables:
            layer_index = int(name.split(".")[0].removeprefix("layer"))
            values.append(float(getattr(design.material.layers[layer_index - 1], _THICKNESS)))
    theta = np.asarray(values, dtype=np.float64)
    return checked_theta(theta, designs)


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

def chain_objective(model: ReducedModel, designs: Sequence[SurfaceDesign],
                    objective_constants: StageOneObjectiveConstants, *, air: PhysicsConstants,
                    ranges: PorousModelRanges, frequencies: Sequence[float] | NDArray[np.float64]) -> ChainObjective:
    """回 ``θ → (screening_total, 各分項)`` 的純函式，可 ``jax.jit``／``jax.grad``、精度跟 θ 走。

    θ → 每面參數 → 垂直入射阻抗（``wall_impedances_from_templates``）→ 替身聲壓（``reduced_pressures``）
    → 第一階段低頻目標（``objective_from_pressures``）。樣板用具體值建，要在雙精度已開的行程裡呼叫；
    設計清單照 ``Wall.all()`` 對到替身的六面。
    """
    walls = Wall.all()
    if len(designs) != len(walls):
        raise ValueError(f"替身目前只有六面牆（Wall.all()），設計清單有 {len(designs)} 個面；"
                         "一面切成幾區要等替身支援更多面，這是替身的限制")
    axis = np.asarray(frequencies, dtype=np.float64)
    if axis.shape != (len(model.frequencies_hz),) or not np.array_equal(axis, np.asarray(model.frequencies_hz)):
        raise ValueError("頻率軸必須等於替身的頻率軸")
    if objective_constants.log2_frequencies.shape != axis.shape:
        raise ValueError("目標常數的頻率軸長度跟替身不同")
    constants = reduced_constants(model)
    double_axis = jnp.asarray(axis)
    templates: tuple[ConstructionTemplate | None, ...] = tuple(
        None if design.material is None
        else build_template(design.material, double_axis, air=air, ranges=ranges, variables=design.variables)
        for design in designs)
    frozen_designs = tuple(designs)

    def objective(theta: jax.Array) -> tuple[jax.Array, StageOneObjectiveArrays]:
        values = jnp.asarray(theta)
        parameters = surface_parameters(frozen_designs, values)
        frequency_axis = jnp.asarray(constants.frequencies_hz, dtype=values.dtype)
        impedances = wall_impedances_from_templates(templates, parameters, frequency_axis, air=air)
        pressures = reduced_pressures(constants, impedances.values, rigid_mask=impedances.rigid_mask)
        result = objective_from_pressures(objective_constants, pressures)
        return result.screening_total, result

    return objective


# ---- 投影梯度下降 ----

_EVALUATORS: WeakKeyDictionary[ChainObjective, Evaluator] = WeakKeyDictionary()


def compiled_evaluator(objective: ChainObjective) -> Evaluator:
    """``θ → ((值, 分項), 梯度)`` 的編譯版；同一個目標函式物件只包一次 ``jax.jit``，之後每段下降共用同一份編譯。

    每段下降都重新包一次會每段重新追蹤、重新編譯（候選確認一輪最多四段）；共用不改數值：同一個函式、
    同一個形狀與精度，編譯出同一份程式。目標函式被丟掉時快取跟著消失。
    """
    evaluator = _EVALUATORS.get(objective)
    if evaluator is None:
        evaluator = jax.jit(jax.value_and_grad(objective, has_aux=True))
        _EVALUATORS[objective] = evaluator
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


def _descend(objective: ChainObjective, theta0: jax.Array | FloatVector,
             bounds: tuple[FloatVector, FloatVector], settings: DescentSettings, *,
             max_steps: int, stop_on_gradient: bool, limits: Sequence[DepthLimit] = ()) -> DescentResult:
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
    evaluate = compiled_evaluator(objective)

    def record(index: int, theta: jax.Array, value: jax.Array, aux: StageOneObjectiveArrays,
               gradient: jax.Array, step: float, seconds: float) -> DescentStep:
        return DescentStep(index, tuple(float(x) for x in np.asarray(theta)), float(value),
                           (float(aux.mean_own_line_ripple_rms_db), float(aux.mean_seat_robustness_rms_db)),
                           _projected_gradient_norm(theta, gradient, lower, upper, limits), step, seconds)

    clock = time.perf_counter()
    (value, aux), gradient = evaluate(start)
    evaluations = 1
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
            (new_value, new_aux), new_gradient = evaluate(candidate)
            evaluations += 1
            # 投影弧上的 Armijo：梯度乘實際位移。投到凸集合上的位移跟梯度的內積不大於零（投影的性質），
            # 所以右邊不大於舊值；只守盒界時每一格位移與梯度異號或為零。
            predicted = float(jnp.sum(gradient * (candidate - theta)))
            if bool(jnp.isfinite(new_value)) and float(new_value) <= float(value) + settings.armijo_constant * predicted:
                accepted = True
                break
            trial *= settings.backtrack_ratio
        if not accepted:
            reason = STOP_STEP
            break
        theta, value, aux, gradient = candidate, new_value, new_aux, new_gradient
        steps.append(record(len(steps), theta, value, aux, gradient, trial, time.perf_counter() - clock))
    return DescentResult(np.asarray(theta), tuple(steps), reason, trial, evaluations)


def projected_descent(objective: ChainObjective, theta0: jax.Array | FloatVector,
                      bounds: tuple[FloatVector, FloatVector], settings: DescentSettings, *,
                      limits: Sequence[DepthLimit]) -> DescentResult:
    """投影到「盒界 ∩ 每面厚度＋空腔 ≤ 總深度上限」的梯度下降＋回溯線搜尋（Armijo）。

    ``limits`` 一般是 :func:`depth_limits` 由設計清單算出來的；空的就只守盒界（合成目標的考卷用）。
    每一步的試探點用 :func:`project_to_limits` 投回可行域（精確最近點）；起點要在盒界內，超過總深度上限的先投回。
    每一步從 ``initial_step`` 起試、沒過就乘 ``backtrack_ratio``；步長小於 ``min_step`` 仍沒過就不走這一步、
    停在「步長到下限」。投影梯度範數（貼界、貼斜邊還往外推的分量去掉）小於 ``gradient_norm_stop`` 停在
    「梯度夠小」；走滿 ``max_steps`` 停在「步數到上限」。值單調不增；同一起點重跑逐位相同（秒數除外）。
    取規格那一步照舊保留：厚度仍要取合法值，連續段只是不再跑到總深度外面。
    """
    return _descend(objective, theta0, bounds, settings, max_steps=settings.max_steps, stop_on_gradient=True,
                    limits=limits)


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


def snap_to_specs(theta: jax.Array | FloatVector, designs: Sequence[SurfaceDesign]) -> SnapResult:
    """棉板厚度取可行的最近合法值，空腔維持連續、總深度上限在取規格時一起滿足。

    先取最近（一樣近取較薄）；「厚度＋空腔」超過那面的總深度上限就把空腔切到上限減厚度；
    空腔切到下界仍超過（或沒有空腔層、厚度本身就超過）就退到下一個較薄的合法厚度。
    不是各面各自取最近就交付：取完之後由 :func:`retune_cavities` 聯合重調空腔。
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
                    designs: Sequence[SurfaceDesign], settings: MaterialGradientSettings) -> DescentResult:
    """厚度固定在規格、所有面的空腔一起走同一個投影下降，**最多** ``[retune].steps`` 步。

    不因梯度小提早停；線搜尋退到步長下限仍沒過就不走這一步、停在「步長到下限」（起點就不可下降時是 0 步、
    θ 原樣），停止原因與最後一次試的步長都交回、進帳本；走滿上限停在「步數到上限」。不硬湊步數。
    厚度的上下界相等，投影把它釘回原值、逐位不變；空腔上界收在總深度內，所以每一步都可行。
    """
    bounds = retune_bounds(theta_feasible, designs)
    return _descend(objective, np.asarray(theta_feasible), bounds, settings.descent,
                    max_steps=settings.retune.steps, stop_on_gradient=False)
