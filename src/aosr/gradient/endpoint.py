"""第三階段梯度搜尋的終點（#561 辛二）：六面規格匯出成牆面材料、完整模型重算同一套分數、量大峰。

梯度在替身上找出一組六面規格之後，這一支把它變成方案檔認得的構造層清單（:func:`materials_from_surfaces`、
:func:`scene_with_materials`），用完整模型（不是替身）逐頻重解（:func:`full_model_pressures`），
並把替身值與完整模型值、兩者的差、每支喇叭的大峰並列成 :class:`EndpointReport`（:func:`endpoint_check`）。

只報數字、不下判決：替身準不準由決策紙 ``fem-reduced-model-jax-evaluation-for-stage-three.md`` 的兩把尺管，
大峰門檻 #451 說先不訂。分數一律走第一階段低頻目標同一把尺（``objective_from_pressures``），大峰走第一階段
評分程式的低頻量測零件（``low_frequency_input`` → ``evaluate_stage_one_low_frequency`` 的最強峰），不另寫算式。
只用那幾個零件：第一階段整條入口（``reporting.stage_one_result`` 的前提檢查）今天不收構造牆，這是既有限制，
這一支不宣稱匯出的方案能直接走第一階段整條。

材料只有一個來源：每面的 ``SurfaceDesign.material``（替身鏈用的那一份），匯出時只換棉板厚度、照設定檔
``[export] min_cavity_m`` 決定空氣層放不放；設計清單必須照 ``Wall.all()`` 的順序給六面。終點入口另外核身分：
每面的樣板構造必須等於它 preset id 在構造預設清單裡的那一份（:func:`checked_design_identity`），報告每面再記
樣板構造的摘要（``material_digest``），preset id 跟實際用的材料不會脫鉤。
材料求阻抗用的空氣（``air``）必須跟完整模型算子、替身同一組密度與聲速，不等就拒收。

方案格式沒有「剛性」的寫法（表面分區每面一種材料、固定阻抗欄只收正數），所以 :func:`scene_with_materials`
遇到剛性面（None）刻意拒收；「沒設計的面保留方案原本的牆面」要從計算起就一致，由接線那支定，這一支不做。

完整模型的算子由呼叫端給（方案走 ``reporting.fem_reduced_scheme.export_operators``，跟建替身同一份或同網格重匯出）；
替身與算子的網格、喇叭、座位、空氣一致性由 :func:`aosr.physics.fem_reduced_check.validate_reduced` 核。
材料求阻抗要雙精度 JAX：在雙精度已開的行程裡呼叫。本模組不碰 JAX 全域設定、不印東西、沒有行程與檔案。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, TypeAdapter

from aosr.config.construction_presets import ConstructionPreset
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.material_gradient import MaterialGradientSettings
from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.geometry.shoebox import Wall
from aosr.gradient.material_gradient import (
    FloatVector, SurfaceDesign, chain_objective, checked_theta, is_feasible, wall_surface_ids,
)
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_models import AirLayer, Assembly, PorousLayer, SurfaceMaterial
from aosr.physics.fem_helmholtz import FrequencyWallImpedances
from aosr.physics.fem_reduced import FullOperators, FullSolver, ReducedModel, impedance_rows, walls_at
from aosr.physics.fem_reduced_check import validate_reduced
from aosr.physics.report_surfaces import OLD_WALL_FIELD, ZONE_WALL_FIELD, WholeWallZone, validation_content
from aosr.reporting.scheme import Scene
from aosr.scoring.low_frequency_input import low_frequency_input
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_objective_jax import (
    StageOneObjectiveArrays, StageOneObjectiveConstants, objective_constants, objective_from_pressures,
    rms_floor_factor,
)
from aosr.scoring.stage_one_score import StageOneScoreSettings

_CANDIDATE_ID: Final = "gradient-endpoint"


class SurfaceSpec(NamedTuple):
    """一面的規格：面代號（牆名）、preset id、棉板厚度與空腔（公尺）、樣板構造摘要；剛性面後四格都是 None。

    ``cavity_m`` 是搜尋給的原值（可能是 0 或小於設定的最小空腔）；放不放空氣層由 :func:`kept_cavity` 決定。
    樣板沒有空氣層的面 ``cavity_m`` 是 None。``material_digest`` 是換 θ 之前的樣板構造（``design.material``）
    的 :func:`material_digest`。
    """

    surface_id: str
    preset_id: str | None
    thickness_m: float | None
    cavity_m: float | None
    material_digest: str | None


def material_digest(material: SurfaceMaterial) -> str:
    """構造的身分摘要：驗證後內容的正規化 JSON（鍵排序、緊湊分隔、不轉義非 ASCII）做 SHA-256 十六進位。"""
    content = json.dumps(material.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def kept_cavity(cavity_m: float, min_cavity_m: float) -> float | None:
    """空腔匯出規則：小於 ``min_cavity_m``（設定檔 ``[export]``，含 0）就不放空氣層、回 None；否則原值照放。

    非有限數或負數拒收。
    """
    if not math.isfinite(cavity_m) or cavity_m < 0.0:
        raise ValueError(f"空腔必須是非負有限數，收到 {cavity_m}")
    return None if cavity_m < min_cavity_m else float(cavity_m)


def checked_wall_designs(designs: Sequence[SurfaceDesign]) -> tuple[SurfaceDesign, ...]:
    """設計清單的面代號必須照 ``Wall.all()`` 的順序剛好是六面牆名：替身鏈照位置配牆，匯出照名字配牆，兩邊要同一個序。"""
    ids = tuple(design.surface_id for design in designs)
    if ids != wall_surface_ids():
        raise ValueError(f"設計清單必須照 Wall.all() 的順序給六面 {list(wall_surface_ids())}，收到 {list(ids)}")
    return tuple(designs)


def checked_design_identity(designs: Sequence[SurfaceDesign], presets: Sequence[ConstructionPreset]) -> None:
    """每面的樣板構造（``design.material``，換 θ 之前）必須等於它 preset id 在 ``presets`` 裡的構造（驗證後逐欄相等）；
    preset id 找不到或構造不同就拒收，訊息指名面與 preset id。只核身分，匯出材料仍只從 ``design.material`` 來。"""
    by_id = {preset.id: preset for preset in presets}
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    for design in designs:
        if design.material is None or design.preset_id is None:
            continue
        preset = by_id.get(design.preset_id)
        if preset is None:
            raise ValueError(f"面 {design.surface_id} 的 preset {design.preset_id} 不在構造預設清單裡")
        if adapter.validate_python(preset.material) != design.material:
            raise ValueError(f"面 {design.surface_id} 的樣板構造跟 preset {design.preset_id} 的構造不同，"
                             "報告的 preset id 會認不出實際用的材料")


def surface_specs(designs: Sequence[SurfaceDesign], theta: jax.Array | FloatVector) -> tuple[SurfaceSpec, ...]:
    """θ（照 :func:`~aosr.gradient.material_gradient.design_names` 的格序）切回每面規格；剛性面只有面代號。"""
    values = checked_theta(theta, designs)
    specs = []
    offset = 0
    for design in designs:
        if design.material is None:
            specs.append(SurfaceSpec(design.surface_id, None, None, None, None))
            continue
        cavity = float(values[offset + 1]) if design.cavity_variable is not None else None
        specs.append(SurfaceSpec(design.surface_id, design.preset_id, float(values[offset]), cavity,
                                 material_digest(design.material)))
        offset += len(design.variables)
    return tuple(specs)


def exported_theta(designs: Sequence[SurfaceDesign], theta: jax.Array | FloatVector,
                   settings: MaterialGradientSettings) -> FloatVector:
    """匯出後的 θ（與輸入同精度）：照 :func:`kept_cavity` 被拿掉空氣層的那幾格空腔設成 0，其餘逐位不變。

    替身鏈在空腔 0 跟「沒有空氣層」是同一個阻抗（傳遞矩陣厚度 0 是單位矩陣，終點考卷驗），
    所以拿這個 θ 算替身值，就是對匯出的材料算。
    """
    values = checked_theta(theta, designs).copy()
    offset = 0
    for design in designs:
        if design.material is None:
            continue
        if (design.cavity_variable is not None
                and kept_cavity(float(values[offset + 1]), settings.export.min_cavity_m) is None):
            values[offset + 1] = values.dtype.type(0.0)
        offset += len(design.variables)
    return values


def _design_material(design: SurfaceDesign, spec: SurfaceSpec, min_cavity_m: float) -> Assembly | None:
    """這一面的構造：從 ``design.material``（替身鏈同一份）換棉板厚度，空氣層照規則換厚度或拿掉。"""
    material = design.material
    if material is None:
        return None
    kinds = [type(layer) for layer in material.layers]
    if kinds.count(PorousLayer) != 1 or kinds.count(AirLayer) > 1 or set(kinds) - {PorousLayer, AirLayer}:
        raise ValueError(f"面 {design.surface_id} 的構造必須是一層棉板、最多一層空腔，沒有別的層")
    if spec.thickness_m is None or (AirLayer in kinds) != (spec.cavity_m is not None):
        raise ValueError(f"面 {design.surface_id} 的規格跟樣板的層對不上")
    cavity = None if spec.cavity_m is None else kept_cavity(spec.cavity_m, min_cavity_m)
    layers: list[PorousLayer | AirLayer] = []
    for layer in material.layers:
        if isinstance(layer, PorousLayer):
            layers.append(PorousLayer.model_validate({**layer.model_dump(), "thickness_m": float(spec.thickness_m)}))
        elif cavity is not None:
            layers.append(AirLayer(thickness_m=cavity))
    return Assembly(layers=tuple(layers), backing=material.backing)


def materials_from_surfaces(designs: Sequence[SurfaceDesign], theta: jax.Array | FloatVector,
                            settings: MaterialGradientSettings) -> dict[Wall, Assembly | None]:
    """六面設計＋θ → 每面牆的構造層清單（照 ``Wall.all()`` 排）；剛性面是 None。

    材料只從 ``design.material`` 來（替身鏈建樣板的同一份），不再拿 preset id 去另一份清單重查：棉板層換成
    θ 的厚度；空氣層照**空腔匯出規則**——空腔小於設定檔 ``[export] min_cavity_m``（含 0）就整層拿掉、
    不放 ``AirLayer``（構造格式的空氣層厚度必須大於 0），否則放 ``AirLayer(thickness_m=空腔)``。
    設計清單必須照 ``Wall.all()`` 的順序給六面（:func:`checked_wall_designs`）。
    """
    checked = checked_wall_designs(designs)
    specs = surface_specs(checked, theta)
    minimum = settings.export.min_cavity_m
    return {wall: _design_material(design, spec, minimum)
            for wall, design, spec in zip(Wall.all(), checked, specs, strict=True)}


def checked_air(air: PhysicsConstants, *, density_kg_m3: float, sound_speed_m_s: float, source: str) -> None:
    """材料求阻抗用的空氣必須跟有限元素那一份同密度、同聲速（逐位相等）；不等就拒收、訊息列出兩邊的值。"""
    if air.air_density_kg_m3 != density_kg_m3 or air.sound_speed_m_s != sound_speed_m_s:
        raise ValueError(
            f"材料的空氣（密度 {air.air_density_kg_m3}、聲速 {air.sound_speed_m_s}）跟{source}"
            f"（密度 {density_kg_m3}、聲速 {sound_speed_m_s}）不同，阻抗與有限元素會用兩組空氣")


def material_impedances(materials: Mapping[Wall, Assembly | None], frequencies_hz: Sequence[float], *,
                        air: PhysicsConstants, ranges: PorousModelRanges) -> FrequencyWallImpedances:
    """每面構造走產品 ``surface_impedance``（非可微入口）求垂直入射阻抗，攤成逐頻 Python 複數；剛性面 None。"""
    axis = jnp.asarray(np.asarray(frequencies_hz, dtype=np.float64))
    impedances: dict[Wall, tuple[complex, ...] | None] = {}
    for wall in Wall.all():
        material = materials[wall]
        if material is None:
            impedances[wall] = None
            continue
        values, _ = surface_impedance(material, axis, air=air, ranges=ranges)
        impedances[wall] = tuple(complex(value) for value in np.asarray(values))
    return impedances


def full_model_pressures(operators: FullOperators, materials: Mapping[Wall, Assembly | None],
                         frequencies_hz: Sequence[float], *, air: PhysicsConstants, ranges: PorousModelRanges,
                         max_relative_residual: float) -> NDArray[np.complex128]:
    """完整模型逐頻解（每頻一次 pydiso 分解），回座位聲壓 ``(F, 喇叭, 座位)``。

    跟 :func:`~aosr.physics.fem_reduced_check.validate_reduced` 的完整解同一條路：阻抗過產品閘門
    （``impedance_rows``）、``FullSolver.solve`` 後乘座位取值列。頻率軸由呼叫端給（用替身的軸），不自己造。
    ``air`` 必須跟算子同密度、同聲速（:func:`checked_air`）。
    """
    checked_air(air, density_kg_m3=operators.density_kg_m3, sound_speed_m_s=operators.sound_speed_m_s,
                source="完整模型算子")
    axis = tuple(float(value) for value in frequencies_hz)
    rows = impedance_rows([material_impedances(materials, axis, air=air, ranges=ranges)], axis)[0]
    solver = FullSolver(operators, max_relative_residual=max_relative_residual)
    receivers = np.asarray(list(operators.receivers.values()), dtype=np.float64)
    pressures = np.empty((len(axis), len(operators.sources), len(operators.receivers)), dtype=np.complex128)
    for index, frequency in enumerate(axis):
        pressures[index] = solver.solve(frequency, walls_at(rows, index)) @ receivers.T
    return pressures


def big_peaks(pressures: NDArray[np.complex128], *, speaker_ids: Sequence[str], receiver_set: ReceiverSet,
              frequencies_hz: Sequence[float], low_frequency_axis: LowFrequencyAxis,
              scoring_range_hz: tuple[float, float]) -> dict[str, float | None]:
    """每支喇叭的大峰（主位扣自己直線後最強峰的深度，dB）；量不到（不可估或沒有峰）是 None。

    走第一階段評分程式本身：``low_frequency_input``（頻率軸逐位核對宣告軸）→
    ``evaluate_stage_one_low_frequency`` → ``features[strongest_peak_index].depth_db``，跟搜尋層
    ``big_peak`` 讀的是同一格；峰值演算法不在這裡重寫。``pressures`` 是 ``(F, 喇叭, 座位)``，座位照座位組原序。
    """
    values = np.asarray(pressures)
    points = receiver_set.points
    if values.shape != (len(frequencies_hz), len(speaker_ids), len(points)):
        raise ValueError(f"聲壓形狀必須是 (F, 喇叭, 座位)＝{(len(frequencies_hz), len(speaker_ids), len(points))}，"
                         f"收到 {values.shape}")
    energies = np.abs(values) ** 2
    data = low_frequency_input(
        candidate_id=_CANDIDATE_ID, speaker_ids=tuple(speaker_ids), receiver_set=receiver_set,
        axis=low_frequency_axis,
        energies={(speaker, point.receiver_id): tuple(float(value) for value in energies[:, s, r])
                  for s, speaker in enumerate(speaker_ids) for r, point in enumerate(points)},
        frequency_hex=tuple(float(value).hex() for value in frequencies_hz))
    evaluation = evaluate_stage_one_low_frequency(data, scoring_range_hz=scoring_range_hz)
    peaks: dict[str, float | None] = {}
    for speaker in evaluation.speakers:
        payload = speaker.payload
        index = None if payload is None else payload.strongest_peak_index
        peaks[speaker.speaker_id] = None if payload is None or index is None else payload.features[index].depth_db
    return peaks


def worst_big_peak(peaks: Mapping[str, float | None]) -> float | None:
    """各支取較大（較差）；任一支量不到就是 None——跟搜尋層 ``big_peak`` 同一個取法。"""
    values = list(peaks.values())
    if not values or any(value is None for value in values):
        return None
    return max(value for value in values if value is not None)


def objective_floor_db(*pressures: NDArray[np.complex128]) -> float:
    """這幾組聲壓交目標時，均方根零判地板的上限（dB）：目標同一個倍率（``rms_floor_factor``，雙精度）
    × max(各組全軸、全喇叭、全座位 |10·log10|p|²| 的最大值, 1)。

    目標的地板逐喇叭、只看計分範圍內的頻點，這裡取所有組、全軸的最大值，不會比任何一支喇叭的地板小。
    """
    scale = max(1.0, *(float(np.max(np.abs(10.0 * np.log10(np.abs(np.asarray(values)) ** 2))))
                       for values in pressures))
    return rms_floor_factor(np.float64) * scale


def score_difference_bound(constants: StageOneObjectiveConstants, level_db: float, floor_db: float) -> float:
    """座位聲壓逐頻 dB 差都不超過 ``level_db``（ε）、兩邊的均方根零判地板都不超過 ``floor_db``（F）時，
    兩邊合分之差的上界 (ε+F)·w_主/ref_主 + (2ε+F)·w_穩/ref_穩。

    推導：dB 差就是能量 dB（10·log10|p|²＝20·log10|p|）的差，每格不超過 ε。主項是主位曲線扣加權最小平方直線後的
    加權均方根；扣直線是同一個加權內積下的正交投影（範數不增），加權均方根是那個內積的範數，三角不等式給
    「未歸零的主項之差 ≤ ε」。座位穩健性是周圍點減主位的加權均方根，每格差不超過 2ε，所以「未歸零之差 ≤ 2ε」。
    目標把不高於地板的均方根當 0（#959）：兩邊都歸零差 0；都沒歸零差不變；一邊歸零時另一邊的值不超過
    「它的未歸零值 ≤ 歸零那邊的未歸零值＋差 ≤ F＋差」，所以每一項之差再加最多 F。聲道平均、等權平均都不放大。
    合分是兩項聲道平均各自乘權重除較差參考相加，權重與參考取 ``constants`` 那一份（主項、穩健性的固定順序）。
    """
    weights = np.asarray(constants.component_weights, dtype=np.float64)
    references = np.asarray(constants.worse_references, dtype=np.float64)
    return float((level_db + floor_db) * weights[0] / references[0]
                 + (2.0 * level_db + floor_db) * weights[1] / references[1])


def _components(result: StageOneObjectiveArrays) -> tuple[float, float]:
    return float(result.mean_own_line_ripple_rms_db), float(result.mean_seat_robustness_rms_db)


class EndpointReport(BaseModel):
    """終點重算的數字（只報不判）：替身值、完整模型值、差與差的上界、每座位逐頻最大 dB 差、帶能量差、大峰。

    ``reduced_*`` 是替身鏈（``chain_objective``）在 ``exported_theta`` 的值；``full_*`` 是完整模型解同一組匯出材料、
    交同一把尺（``objective_from_pressures``）的值；``total_difference`` ＝完整減替身。``max_frequency_db`` 是
    每座位、喇叭、頻點 |20·log10(|p_替身|/|p_完整|)| 的最大值（全軸），``rms_floor_db`` 是兩邊聲壓的均方根零判
    地板上限，``total_difference_bound`` 由兩者照 :func:`score_difference_bound` 推得。大峰是完整模型的。
    ``surfaces`` 每面帶 preset id 與樣板構造摘要（``material_digest``），入口已核兩者是同一份構造。
    非有限數不收（驗證時拒收，不靜默報一個壞數）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    low_frequency_axis: LowFrequencyAxis
    speaker_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    search_theta: tuple[float, ...]
    exported_theta: tuple[float, ...]
    surfaces: tuple[SurfaceSpec, ...]
    dropped_cavities: tuple[str, ...]
    reduced_total: float
    reduced_components: tuple[float, float]
    full_total: float
    full_components: tuple[float, float]
    total_difference: float
    total_difference_bound: float
    max_frequency_db: float
    rms_floor_db: float
    frequency_db_by_seat: dict[str, float]
    band_db_by_seat: dict[str, float]
    partial_band_db_by_seat: dict[str, float]
    big_peak_db_by_speaker: dict[str, float | None]
    big_peak_db: float | None
    full_solves: int
    model_digest: str

    def to_json(self) -> str:
        return self.model_dump_json()


class EndpointResult(NamedTuple):
    """終點報告與兩邊的座位聲壓 ``(F, 喇叭, 座位)``（替身是 numpy ``evaluate_reduced``，同一組匯出材料）。"""

    report: EndpointReport
    full_pressure: NDArray[np.complex128]
    reduced_pressure: NDArray[np.complex128]


def _checked_endpoint_inputs(operators: FullOperators, model: ReducedModel, designs: Sequence[SurfaceDesign],
                             theta: jax.Array | FloatVector, receiver_set: ReceiverSet,
                             air: PhysicsConstants) -> FloatVector:
    checked_air(air, density_kg_m3=operators.density_kg_m3, sound_speed_m_s=operators.sound_speed_m_s,
                source="完整模型算子")
    checked_air(air, density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s, source="替身")
    checked_wall_designs(designs)
    values = checked_theta(theta, designs)
    if not is_feasible(values, designs):
        raise ValueError("終點的 θ 必須是取過規格的可行解（厚度是合法值、厚度＋空腔不超過總深度上限）")
    if model.receiver_names != tuple(point.receiver_id for point in receiver_set.points):
        raise ValueError("替身的座位順序必須等於座位組的原序")
    return values


def endpoint_check(operators: FullOperators, designs: Sequence[SurfaceDesign], theta: jax.Array | FloatVector, *,
                   model: ReducedModel, score_settings: StageOneScoreSettings, receiver_set: ReceiverSet,
                   low_frequency_axis: LowFrequencyAxis, presets: Sequence[ConstructionPreset],
                   settings: MaterialGradientSettings, air: PhysicsConstants,
                   ranges: PorousModelRanges) -> EndpointResult:
    """同一組可行規格：替身值與完整模型值並列、兩者的差與差的上界、每座位逐頻最大 dB 差、每支喇叭的大峰。

    入口先核：``air`` 跟算子、替身同密度同聲速；設計清單照 ``Wall.all()`` 六面；每面樣板構造等於它的 preset
    （``presets`` 只用來核身分，:func:`checked_design_identity`）；θ 是可行解；座位順序一致。
    θ 照空腔匯出規則變成材料（:func:`materials_from_surfaces`，材料只從 ``design.material`` 來）；完整模型與替身都對
    **匯出的材料**算（替身用 :func:`exported_theta`），差裡只有替身誤差、沒有匯出的捨入。目標常數由
    ``objective_constants`` 以替身的頻率軸、``score_settings``、``receiver_set`` 現算，跟搜尋那邊同一份；
    完整解殘差上限沿用建替身當次的設定。完整解用 :func:`~aosr.physics.fem_reduced_check.validate_reduced`
    （同時給替身聲壓、逐頻與帶能量 dB 差）。
    """
    values = _checked_endpoint_inputs(operators, model, designs, theta, receiver_set, air)
    checked_design_identity(designs, presets)
    axis = tuple(float(value) for value in model.frequencies_hz)
    specs = surface_specs(designs, values)
    materials = materials_from_surfaces(designs, values, settings)
    validation = validate_reduced(model, operators, [material_impedances(materials, axis, air=air, ranges=ranges)])
    full, reduced = validation.full_pressure[0], validation.reduced_pressure[0]
    constants = objective_constants(axis, score_settings, receiver_set, len(model.source_names))
    exported = exported_theta(designs, values, settings)
    objective = chain_objective(model, designs, constants, air=air, ranges=ranges, frequencies=axis)
    reduced_total, reduced_result = objective(jnp.asarray(exported))
    full_result = objective_from_pressures(constants, jnp.asarray(full))
    level_db = max(validation.frequency_db_by_seat.values())
    floor_db = objective_floor_db(full, reduced)
    peaks = big_peaks(full, speaker_ids=model.source_names, receiver_set=receiver_set, frequencies_hz=axis,
                      low_frequency_axis=low_frequency_axis, scoring_range_hz=score_settings.scored_range_hz)
    minimum = settings.export.min_cavity_m
    report = EndpointReport(
        low_frequency_axis=low_frequency_axis, speaker_ids=model.source_names, receiver_ids=model.receiver_names,
        search_theta=tuple(float(value) for value in values),
        exported_theta=tuple(float(value) for value in exported), surfaces=specs,
        dropped_cavities=tuple(spec.surface_id for spec in specs
                               if spec.cavity_m is not None and kept_cavity(spec.cavity_m, minimum) is None),
        reduced_total=float(reduced_total), reduced_components=_components(reduced_result),
        full_total=float(full_result.screening_total), full_components=_components(full_result),
        total_difference=float(full_result.screening_total) - float(reduced_total),
        total_difference_bound=score_difference_bound(constants, level_db, floor_db), max_frequency_db=level_db,
        rms_floor_db=floor_db,
        frequency_db_by_seat=dict(validation.frequency_db_by_seat), band_db_by_seat=dict(validation.band_db_by_seat),
        partial_band_db_by_seat=dict(validation.partial_band_db_by_seat),
        big_peak_db_by_speaker=peaks, big_peak_db=worst_big_peak(peaks),
        full_solves=validation.full_solves, model_digest=validation.model_digest)
    return EndpointResult(report, full, reduced)


def scene_with_materials(scene: Scene, materials: Mapping[Wall, Assembly | None]) -> Scene:
    """把六面構造寫進方案場景的牆面欄位：每面一區（``whole_wall``，#309 構造牆進方案的格式），舊的固定阻抗欄拿掉。

    方案格式沒有「剛性」的寫法（表面分區每面必須是一種材料，固定阻抗欄只收正數），所以剛性面（None）拒收：
    替身與完整模型都把那一面當剛性算過，寫成任何別的材料，匯出的方案就不再是剛才驗過的那一組。其餘場景欄位不動。
    """
    given = {wall: material for wall in Wall.all() if (material := materials.get(wall)) is not None}
    absent = [wall.wall_name() for wall in Wall.all() if wall not in given]
    if absent:
        raise ValueError(f"方案牆面要六面都有構造；缺或剛性：{absent}（方案格式沒有剛性的寫法）")
    zones = {wall.wall_name(): (WholeWallZone(area="whole_wall", material=given[wall]),) for wall in Wall.all()}
    content = validation_content(scene)
    if not isinstance(content, dict):
        raise TypeError("場景內容必須是欄位字典")
    document = {name: value for name, value in content.items() if name not in (OLD_WALL_FIELD, ZONE_WALL_FIELD)}
    return Scene.model_validate({**document, ZONE_WALL_FIELD: zones})
