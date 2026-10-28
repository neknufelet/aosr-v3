"""第三階段梯度搜尋的終點（#561 辛二）：六面規格匯出成牆面材料、完整模型重算同一套分數、量大峰。

梯度在替身上找出一組六面規格之後，這一支把它變成方案檔認得的構造層清單（:func:`materials_from_surfaces`、
:func:`scene_with_materials`），用完整模型（不是替身）逐頻重解（:func:`full_model_pressures`），
並把替身值與完整模型值、兩者的差、每支喇叭的大峰並列成 :class:`EndpointReport`（:func:`endpoint_check`）。

只報數字、不下判決：替身準不準由決策紙 ``fem-reduced-model-jax-evaluation-for-stage-three.md`` 的兩把尺管，
大峰門檻 #451 說先不訂。分數一律走第一階段低頻目標同一把尺（``objective_from_pressures``），大峰走第一階段
評分程式（``low_frequency_input`` → ``evaluate_stage_one_low_frequency`` 的最強峰），不另寫算式。

完整模型的算子由呼叫端給（方案走 ``reporting.fem_reduced_scheme.export_operators``，跟建替身同一份或同網格重匯出）；
替身與算子的網格、喇叭、座位、空氣一致性由 :func:`aosr.physics.fem_reduced_check.validate_reduced` 核。
材料求阻抗要雙精度 JAX：在雙精度已開的行程裡呼叫。本模組不碰 JAX 全域設定、不印東西、沒有行程與檔案。
"""
from __future__ import annotations

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
    FloatVector, SurfaceDesign, chain_objective, checked_theta, is_feasible,
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
)
from aosr.scoring.stage_one_score import StageOneScoreSettings

# 空腔匯出規則（工程定法，#561 辛二派工單，不問老闆）：家族的空腔是連續量（``FamilySettings.cavity_range_m`` 只有上下限、
# 沒有分級刻度），所以用「小於 1 mm 就不放空氣層」；家族以後有了空腔刻度，改成「小於最小一格」。
MIN_CAVITY_M: Final = 0.001
_CANDIDATE_ID: Final = "gradient-endpoint"


class SurfaceSpec(NamedTuple):
    """一面的規格：面代號（牆名）、preset id、棉板厚度與空腔（公尺）；剛性面 preset 與兩個數都是 None。

    ``cavity_m`` 是搜尋給的原值（可能是 0 或小於 :data:`MIN_CAVITY_M`）；放不放空氣層由
    :func:`materials_from_surfaces` 照 :func:`kept_cavity` 決定。樣板沒有空氣層的面 ``cavity_m`` 是 None。
    """

    surface_id: str
    preset_id: str | None
    thickness_m: float | None
    cavity_m: float | None


def kept_cavity(cavity_m: float) -> float | None:
    """空腔匯出規則：小於 :data:`MIN_CAVITY_M`（含 0）就不放空氣層、回 None；否則原值照放。負數拒收。"""
    if not cavity_m >= 0.0:
        raise ValueError(f"空腔必須是非負有限數，收到 {cavity_m}")
    return None if cavity_m < MIN_CAVITY_M else float(cavity_m)


def surface_specs(designs: Sequence[SurfaceDesign], theta: jax.Array | FloatVector) -> tuple[SurfaceSpec, ...]:
    """θ（照 :func:`~aosr.gradient.material_gradient.design_names` 的格序）切回每面規格；剛性面四格只有面代號。"""
    values = checked_theta(theta, designs)
    specs = []
    offset = 0
    for design in designs:
        if design.material is None:
            specs.append(SurfaceSpec(design.surface_id, None, None, None))
            continue
        cavity = float(values[offset + 1]) if design.cavity_variable is not None else None
        specs.append(SurfaceSpec(design.surface_id, design.preset_id, float(values[offset]), cavity))
        offset += len(design.variables)
    return tuple(specs)


def exported_theta(designs: Sequence[SurfaceDesign], theta: jax.Array | FloatVector) -> FloatVector:
    """匯出後的 θ（與輸入同精度）：照 :func:`kept_cavity` 被拿掉空氣層的那幾格空腔設成 0，其餘逐位不變。

    替身鏈在空腔 0 跟「沒有空氣層」是同一個阻抗（傳遞矩陣厚度 0 是單位矩陣，終點考卷驗），
    所以拿這個 θ 算替身值，就是對匯出的材料算。
    """
    values = checked_theta(theta, designs).copy()
    offset = 0
    for design in designs:
        if design.material is None:
            continue
        if design.cavity_variable is not None and kept_cavity(float(values[offset + 1])) is None:
            values[offset + 1] = values.dtype.type(0.0)
        offset += len(design.variables)
    return values


def _preset_assembly(preset_id: str, presets: Mapping[str, ConstructionPreset],
                     settings: MaterialGradientSettings) -> Assembly:
    preset = presets.get(preset_id)
    if preset is None:
        raise ValueError(f"preset id 不認識：{preset_id}")
    if settings.family_of(preset_id) is None:
        raise ValueError(f"preset {preset_id} 不在家族清單裡（設定檔 [[family]]）")
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    material = adapter.validate_python(preset.material)
    if not isinstance(material, Assembly):
        raise ValueError(f"preset {preset_id} 不是多層構造")
    kinds = [type(layer) for layer in material.layers]
    if kinds.count(PorousLayer) != 1 or kinds.count(AirLayer) > 1 or set(kinds) - {PorousLayer, AirLayer}:
        raise ValueError(f"preset {preset_id} 必須是一層棉板、最多一層空腔，沒有別的層")
    return material


def _spec_material(spec: SurfaceSpec, presets: Mapping[str, ConstructionPreset],
                   settings: MaterialGradientSettings) -> Assembly | None:
    if spec.preset_id is None:
        if spec.thickness_m is not None or spec.cavity_m is not None:
            raise ValueError(f"剛性面 {spec.surface_id} 不收厚度或空腔")
        return None
    template = _preset_assembly(spec.preset_id, presets, settings)
    has_air = any(isinstance(layer, AirLayer) for layer in template.layers)
    if spec.thickness_m is None or not spec.thickness_m > 0.0:
        raise ValueError(f"面 {spec.surface_id} 的棉板厚度必須是正數")
    if has_air != (spec.cavity_m is not None):
        raise ValueError(f"面 {spec.surface_id}：樣板{'有' if has_air else '沒有'}空氣層，空腔就要{'給' if has_air else '是 None'}")
    cavity = None if spec.cavity_m is None else kept_cavity(spec.cavity_m)
    layers: list[PorousLayer | AirLayer] = []
    for layer in template.layers:
        if isinstance(layer, PorousLayer):
            layers.append(PorousLayer.model_validate({**layer.model_dump(), "thickness_m": float(spec.thickness_m)}))
        elif cavity is not None:
            layers.append(AirLayer(thickness_m=cavity))
    return Assembly(layers=tuple(layers), backing=template.backing)


def materials_from_surfaces(surfaces: Sequence[SurfaceSpec], presets: Sequence[ConstructionPreset],
                            settings: MaterialGradientSettings) -> dict[Wall, Assembly | None]:
    """六面規格 → 每面牆的構造層清單（照 ``Wall.all()`` 排）；剛性面是 None。

    每面拿那個 preset 的構造當樣板：棉板層換成規格厚度；空氣層照**空腔匯出規則**——空腔小於
    :data:`MIN_CAVITY_M`（1 mm，含 0）就整層拿掉、不放 ``AirLayer``（構造格式的空氣層厚度必須大於 0），
    否則放 ``AirLayer(thickness_m=空腔)``。家族的空腔是連續量、沒有分級刻度，所以門檻是 1 mm 而不是「最小一格」。
    面代號必須剛好是六面牆名，各一次；preset 必須在家族清單裡、是一層棉板加最多一層空腔。
    """
    walls = {wall.wall_name(): wall for wall in Wall.all()}
    ids = [spec.surface_id for spec in surfaces]
    if sorted(ids) != sorted(walls):
        raise ValueError(f"面代號必須剛好是六面牆名各一次 {sorted(walls)}，收到 {ids}")
    by_id = {preset.id: preset for preset in presets}
    by_wall = {walls[spec.surface_id]: _spec_material(spec, by_id, settings) for spec in surfaces}
    return {wall: by_wall[wall] for wall in Wall.all()}


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
    """
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


def score_difference_bound(constants: StageOneObjectiveConstants, level_db: float) -> float:
    """座位聲壓逐頻 dB 差都不超過 ``level_db``（ε）時，兩邊合分之差的上界 ε·(w_主/ref_主 + 2·w_穩/ref_穩)。

    推導：dB 差就是能量 dB（10·log10|p|²＝20·log10|p|）的差，每格不超過 ε。主項是主位曲線扣加權最小平方直線後的
    加權均方根；扣直線是同一個加權內積下的正交投影（範數不增），加權均方根是那個內積的範數，三角不等式給
    「主項之差 ≤ ε」。座位穩健性是周圍點減主位的加權均方根，每格差不超過 2ε，所以「穩健性之差 ≤ 2ε」；
    聲道平均、等權平均都不放大。合分是兩項聲道平均各自乘權重除較差參考相加，權重與參考取 ``constants`` 那一份。
    沒算均方根零判的地板（雙精度約 1e-12 dB 量級）。
    """
    weights = np.asarray(constants.component_weights, dtype=np.float64)
    references = np.asarray(constants.worse_references, dtype=np.float64)
    return float(level_db * (weights[0] / references[0] + 2.0 * weights[1] / references[1]))


def _components(result: StageOneObjectiveArrays) -> tuple[float, float]:
    return float(result.mean_own_line_ripple_rms_db), float(result.mean_seat_robustness_rms_db)


class EndpointReport(BaseModel):
    """終點重算的數字（只報不判）：替身值、完整模型值、差與差的上界、每座位逐頻最大 dB 差、帶能量差、大峰。

    ``reduced_*`` 是替身鏈（``chain_objective``）在 ``exported_theta`` 的值；``full_*`` 是完整模型解同一組匯出材料、
    交同一把尺（``objective_from_pressures``）的值；``total_difference`` ＝完整減替身。``max_frequency_db`` 是
    每座位、喇叭、頻點 |20·log10(|p_替身|/|p_完整|)| 的最大值（全軸），``total_difference_bound`` 由它照
    :func:`score_difference_bound` 推得。大峰是完整模型的。非有限數不收（驗證時拒收，不靜默報一個壞數）。
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


def _checked_endpoint_inputs(model: ReducedModel, designs: Sequence[SurfaceDesign],
                             theta: jax.Array | FloatVector, receiver_set: ReceiverSet) -> FloatVector:
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

    θ 先照空腔匯出規則變成材料（:func:`materials_from_surfaces`）；完整模型與替身都對**匯出的材料**算
    （替身用 :func:`exported_theta`），差裡只有替身誤差、沒有匯出的捨入。目標常數由 ``objective_constants``
    以替身的頻率軸、``score_settings``、``receiver_set`` 現算，跟搜尋那邊同一份；完整解殘差上限沿用建替身當次的設定。
    完整解用 :func:`~aosr.physics.fem_reduced_check.validate_reduced`（同時給替身聲壓、逐頻與帶能量 dB 差）。
    """
    values = _checked_endpoint_inputs(model, designs, theta, receiver_set)
    axis = tuple(float(value) for value in model.frequencies_hz)
    specs = surface_specs(designs, values)
    materials = materials_from_surfaces(specs, presets, settings)
    validation = validate_reduced(model, operators, [material_impedances(materials, axis, air=air, ranges=ranges)])
    full, reduced = validation.full_pressure[0], validation.reduced_pressure[0]
    constants = objective_constants(axis, score_settings, receiver_set, len(model.source_names))
    exported = exported_theta(designs, values)
    objective = chain_objective(model, designs, constants, air=air, ranges=ranges, frequencies=axis)
    reduced_total, reduced_result = objective(jnp.asarray(exported))
    full_result = objective_from_pressures(constants, jnp.asarray(full))
    level_db = max(validation.frequency_db_by_seat.values())
    peaks = big_peaks(full, speaker_ids=model.source_names, receiver_set=receiver_set, frequencies_hz=axis,
                      low_frequency_axis=low_frequency_axis, scoring_range_hz=score_settings.scored_range_hz)
    report = EndpointReport(
        low_frequency_axis=low_frequency_axis, speaker_ids=model.source_names, receiver_ids=model.receiver_names,
        search_theta=tuple(float(value) for value in values),
        exported_theta=tuple(float(value) for value in exported), surfaces=specs,
        dropped_cavities=tuple(spec.surface_id for spec in specs
                               if spec.cavity_m is not None and kept_cavity(spec.cavity_m) is None),
        reduced_total=float(reduced_total), reduced_components=_components(reduced_result),
        full_total=float(full_result.screening_total), full_components=_components(full_result),
        total_difference=float(full_result.screening_total) - float(reduced_total),
        total_difference_bound=score_difference_bound(constants, level_db), max_frequency_db=level_db,
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
