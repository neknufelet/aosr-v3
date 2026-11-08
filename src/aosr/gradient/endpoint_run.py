"""牆面材料梯度搜尋的終點接線（#561 辛三）：命令列 ``endpoint <資料夾>`` 的子行程那一段。

搜尋跑完（有 ``result.json``）之後，在雙精度子行程裡：輸入走同一道閘門（:func:`aosr.gradient.run.check_inputs`）；
``result.json`` 不只信它自己說的——從兩本帳（庚二帳本與換樣板帳本，標頭都等於這一次重建的）重建整份結果、跟檔裡整個相等才收，
用重建那份的**最後配置**與 θ（庚三：設計從結果的 ``surfaces`` 建，換過樣板的面用新家族的構造、不貼的面用方案原本的牆，
不再從 ``project.presets`` 建）；方案匯出完整模型算子，網格雜湊與物理身分要等於 ``project.json`` 記的；θ 交 ``endpoint.endpoint_check``
（替身值對完整模型值、差與上界、大峰）；基準是方案原本的六面（全部保留原牆）用完整模型解同一條軸、交同一份目標常數；
最後把終點那一套寫成完整方案檔 ``scheme-endpoint.json``，數字寫成 ``endpoint.json``（綁帳本檔、換樣板帳本檔與 ``result.json``
檔的摘要，任一變了它就過期）。兩個檔都整檔寫完再換名。不寫 ``status.json``：終點不是搜尋狀態機的一段。

六類三組數走 ``endpoint_six``：替身／完整聲壓各交同一份四路阻抗與六類 JAX 常數；完整聲壓的能量另注入正式 numpy 管線。
歷史低頻對照仍用 ``gradient_objective_from_pressures``，大峰仍用 ``endpoint.big_peaks``。要在雙精度已開的行程裡叫。

v3（#561 壬一）：梯度目標加了大峰代理項；終點報告、原牆基準與改善各自並列舊合分與新合分、代理（老闆 2026-10-30：
「第三場一定要同時列舊分數與新分數」）。v2 只准報告讀（``legacy``），終點不收 v3 之前的結果檔。
v4（#1032）：終點與原牆各加低於第三階段計分下界的觀察，改善另列；v3、v2 僅准報告讀。
#990：設計照 ``project.json`` 的安裝方式建（貼在原牆上時匯出的面是 ``MountedOnWall``，背後是方案原本那面牆）；
結果檔剛背是 v4（照舊，終點照收、讀成剛背）、貼在原牆上是 v5（每面帶安裝方式），兩種都要跟帳本重建的整份相等。
v5（#562 C）：接六類搜尋結果 v6／v7；三組數與原牆改善必填，交接頻率與截斷旗標各記一次；舊終點 v4／v3／v2 僅准報告讀。
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, Self

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.construction_presets import load_construction_presets
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.stage_one_score import stage_one_score_settings
from aosr.gradient.cli import keep_signal_handlers
from aosr.gradient.gpu import Device
from aosr.physics.fem_full_batch import checked_device_workers
from aosr.gradient.fixed_surfaces import FixedSurfaces, MaxDepths
from aosr.gradient.endpoint_six import EndpointSix, score_endpoint, scoring_context
from aosr.gradient.endpoint import (
    EndpointReport, EndpointResult, big_peaks, endpoint_check, full_model_pressures, materials_from_surfaces, scene_with_materials,
    UnscoredBand, UnscoredImprovement, below_scoring_improvement, below_scoring_notes, worst_big_peak,
)
from aosr.config.material_gradient import MaterialGradientSettings
from aosr.gradient.boundary import BoundaryKey
from aosr.gradient.material_gradient_ledger import MaterialGradientHeader, header_for
from aosr.gradient.material_gradient import SurfaceDesign
from aosr.gradient.run import (
    LEDGER_FILE, RESULT_FILE, RESULT_VERSION, MOUNTED_RESULT_VERSION, LOW_FREQUENCY_RESULT_VERSION,
    LOW_FREQUENCY_MOUNTED_RESULT_VERSION, GradientProject, GradientResult, check_inputs, designs_for, file_sha256, final_designs, now_text,
    parse_result, read_project, result_from_ledger, scene_wall_materials, search_objective_constants,
    search_objective_settings, template_header_for, write_json_atomic,
)
from aosr.materials.construction_models import SurfaceMaterial
from aosr.physics.report_surfaces import validation_content
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.scoring.gradient_objective_settings import GradientObjectiveSettings
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.reporting.scheme import Scene, Scheme
from aosr.scoring.gradient_objective_jax import (
    GradientObjectiveConstants, ObjectiveReading, gradient_objective_constants, gradient_objective_from_pressures,
    objective_reading,
)

ENDPOINT_FILE: Final = "endpoint.json"
EXPORTED_SCHEME_FILE: Final = "scheme-endpoint.json"
ENDPOINT_VERSION: Final = "aosr.material_gradient_endpoint.v5"
PRE_SIX_ENDPOINT_VERSION: Final = "aosr.material_gradient_endpoint.v4"
PRE_RANGE_ENDPOINT_VERSION: Final = "aosr.material_gradient_endpoint.v3"
# 上一版（梯度目標還沒有大峰代理項）：報告准讀。
LEGACY_ENDPOINT_VERSION: Final = "aosr.material_gradient_endpoint.v2"
_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_SHA256 = r"^[0-9a-f]{64}$"


class EndpointBaseline(BaseModel):
    """基準：方案原本六面（全部保留原牆）用完整模型解終點同一條軸、交同一份目標常數的值與分項、每支喇叭的大峰。

    ``full_total``／``full_components`` 是舊合分與兩分項；``full_objective_total``／``full_proxy_db`` 是新合分與大峰代理，
    ``full_interior_max_db`` 是代理夾擠的對象（內部最大殘差）（v3 起，v2 沒有、是 None）。"""

    model_config = _FROZEN
    full_total: float
    full_components: tuple[float, float]
    big_peak_db_by_speaker: dict[str, float | None]
    big_peak_db: float | None
    full_solves: int = Field(ge=1, strict=True)
    full_objective_total: float | None = None
    full_proxy_db: float | None = None
    full_interior_max_db: float | None = None
    below_scoring: UnscoredBand | None = None


class EndpointImprovement(BaseModel):
    """改善＝基準減終點（兩邊都是完整模型、同一把尺）；正數是終點比原本好（目標越小越好）。

    ``total``／``components`` 是舊合分與兩分項；``objective_total``／``proxy_db`` 是新合分與代理（v3 起）。"""

    model_config = _FROZEN
    total: float
    components: tuple[float, float]
    objective_total: float | None = None
    proxy_db: float | None = None
    below_scoring: dict[str, UnscoredImprovement] | None = None


class EndpointSeconds(BaseModel):
    """各段牆鐘秒數：匯出完整模型算子、終點重算（``endpoint_check``）、基準完整解。"""

    model_config = _FROZEN
    export_operators: float = Field(ge=0.0)
    endpoint: float = Field(ge=0.0)
    baseline: float = Field(ge=0.0)


class EndpointRecord(BaseModel):
    """``endpoint.json``：終點報告、原牆基準、改善、匯出方案檔與它的 SHA-256、各段秒數。

    ``result_ledger_sha256``、``result_template_ledger_sha256`` 抄 ``result.json`` 的 ``ledger_sha256``、``template_ledger_sha256``
    （終點已核兩者等於當時兩本帳的摘要），``result_sha256`` 是終點讀的那份 ``result.json`` 檔的摘要：報告拿現在的帳本檔、
    結果檔與搜尋身分逐件比，任一件變了就認得出這份過期。``engine_commit`` 是 ``project.json`` 記的（開始時給的那一個），終點不另量。
    v2（庚三）多了換樣板帳本摘要；v1 的檔報告當過期（要重跑 endpoint）。
    v3（#561 壬一）：報告、基準、改善都帶新合分與大峰代理（報告另帶上界 T·ln N）；v2 這幾格都沒有。
    v5 的 ``six_category`` 必須完整帶終點、原牆、改善；舊版不准含此欄（包括空值）。
    """

    model_config = _FROZEN
    format_version: Literal["aosr.material_gradient_endpoint.v5", "aosr.material_gradient_endpoint.v4", "aosr.material_gradient_endpoint.v3",
                            "aosr.material_gradient_endpoint.v2"]
    gradient_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    result_ledger_sha256: str = Field(pattern=_SHA256)
    result_template_ledger_sha256: str = Field(pattern=_SHA256)
    result_sha256: str = Field(pattern=_SHA256)
    report: EndpointReport
    baseline: EndpointBaseline
    improvement: EndpointImprovement
    exported_scheme: str = Field(min_length=1)
    exported_scheme_sha256: str = Field(pattern=_SHA256)
    seconds: EndpointSeconds
    finished_at: str
    engine_commit: str | None
    six_category: EndpointSix | None = Field(default=None, exclude_if=lambda value: value is None)
    fixed_surfaces: FixedSurfaces = Field(default_factory=dict, exclude_if=lambda value: not value)
    max_depths: MaxDepths = Field(default_factory=dict, exclude_if=lambda value: not value)

    device: Device = Field(default="cpu", exclude_if=lambda value: value == "cpu")
    max_refinement_steps: int | None = Field(default=None, ge=0, strict=True, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _version_matches_content(self) -> Self:
        proxy = self.format_version != LEGACY_ENDPOINT_VERSION
        fields = (*self.report.proxy_fields(), self.baseline.full_objective_total, self.baseline.full_proxy_db,
                  self.baseline.full_interior_max_db, self.improvement.objective_total, self.improvement.proxy_db)
        if any((value is not None) != proxy for value in fields):
            raise ValueError(f"{ENDPOINT_VERSION} 的報告、基準與改善都要帶新合分與大峰代理；"
                             f"{LEGACY_ENDPOINT_VERSION} 都沒有")
        notes = (self.report.below_scoring, self.baseline.below_scoring, self.improvement.below_scoring)
        if any((value is not None) != (self.format_version in (ENDPOINT_VERSION, PRE_SIX_ENDPOINT_VERSION)) for value in notes):
            raise ValueError("v4 的報告、原牆與改善都要帶不計分低頻加註；v3、v2 都沒有")
        if ((self.six_category is not None) != (self.format_version == ENDPOINT_VERSION)
                or self.format_version != ENDPOINT_VERSION and 'six_category' in self.model_fields_set):
            raise ValueError("v5 必須帶六類終點三組數；v4、v3、v2 不准有六類終點欄位")
        if self.report.below_scoring is not None and self.baseline.below_scoring is not None:
            expected = below_scoring_improvement(self.baseline.below_scoring, self.report.below_scoring)
            if self.improvement.below_scoring != expected:
                raise ValueError("不計分低頻改善必須是原牆減終點")
            if set(self.report.below_scoring.by_speaker) != set(self.report.speaker_ids):
                raise ValueError("不計分低頻加註必須涵蓋報告每支喇叭")
        return self


def read_endpoint(folder: Path, *, legacy: bool = False) -> EndpointRecord:
    """讀 ``endpoint.json``：先看版本，不是這一版就拒收、訊息講明版本（要重跑 endpoint）；
    ``legacy=True`` 只給報告用，准讀 v4／v3／v2，舊檔不能當新版終點接續或重算的輸入。"""
    content = (folder / ENDPOINT_FILE).read_bytes()
    document = json.loads(content)
    version = document.get("format_version") if isinstance(document, dict) else None
    if version != ENDPOINT_VERSION and not (legacy and version in (PRE_SIX_ENDPOINT_VERSION, PRE_RANGE_ENDPOINT_VERSION, LEGACY_ENDPOINT_VERSION)):
        raise ValueError(f"{ENDPOINT_FILE} 版本是 {version!r}，這一版只收 {ENDPOINT_VERSION}；舊版只准報告讀，不准接續或重算")
    return EndpointRecord.model_validate_json(content)


def read_result(folder: Path, *, legacy: bool = False) -> GradientResult:
    """讀六類結果 v6（剛背）／v7（貼牆）；歷史 v4／v5 仍核帳本，``legacy`` 另准報告讀 v3。"""
    content = (folder / RESULT_FILE).read_bytes()
    document = json.loads(content)
    version = document.get("format_version") if isinstance(document, dict) else None
    return parse_result(content, legacy=legacy or version in (
        LOW_FREQUENCY_RESULT_VERSION, LOW_FREQUENCY_MOUNTED_RESULT_VERSION))


def exported_scheme(scheme: Scheme, scene: Scene, gradient_id: str) -> Scheme:
    """原方案換場景與方案代號（原代號接 ``-gradient-`` 與梯度 id 前八位，免得跟原方案混淆），其餘欄位不動。"""
    content = validation_content(scheme)
    if not isinstance(content, dict):
        raise TypeError("方案內容必須是欄位字典")
    return Scheme.model_validate({**content, "scene": scene, "scheme_id": f"{scheme.scheme_id}-gradient-{gradient_id[:8]}"})


def checked_result[K: BoundaryKey](folder: Path, given: GradientResult, header: MaterialGradientHeader,
                   settings: MaterialGradientSettings, kept: Mapping[K, SurfaceMaterial]) -> GradientResult:
    """結果檔不能只信它自己說的：從兩本帳（庚二帳本標頭要等於 ``header``；換樣板帳本標頭要等於照它與現在的庚二帳本摘要
    重建的）重建整份結果，跟檔裡那份整個模型相等才收，回重建那一份；不等就拒收、列出哪幾格不同。"""
    template_header = template_header_for(header, settings, kept, base_ledger_sha256=file_sha256(folder / LEDGER_FILE))
    rebuilt = result_from_ledger(folder, settings, kept, expected_header=header, expected_template_header=template_header,
                                 legacy=given.format_version not in (RESULT_VERSION, MOUNTED_RESULT_VERSION))
    if rebuilt != given:
        actual, expected = given.model_dump(), rebuilt.model_dump()
        differing = sorted(name for name in expected if actual.get(name) != expected[name])
        raise ValueError(f"{RESULT_FILE} 跟帳本對不上（{differing} 不同）：結果檔要由帳本重建，先把搜尋跑完再跑 endpoint")
    return rebuilt


def _full_baseline(constants: GradientObjectiveConstants, pressures: NDArray[np.complex128],
                   peaks: dict[str, float | None]) -> tuple[EndpointBaseline, ObjectiveReading]:
    """原牆基準：完整模型聲壓 ``(F, 喇叭, 座位)`` 交同一份目標常數，記舊合分、兩分項、新合分、代理、內部最大殘差
    與每支喇叭的大峰；一頻一解，解的次數就是 F。"""
    arrays = gradient_objective_from_pressures(constants, jnp.asarray(pressures))
    base = objective_reading(arrays)
    baseline = EndpointBaseline(
        full_total=base.screening_total, full_components=(base.components[0], base.components[1]),
        big_peak_db_by_speaker=peaks, big_peak_db=worst_big_peak(peaks), full_solves=int(pressures.shape[0]),
        full_objective_total=base.objective_total, full_proxy_db=base.components[2],
        full_interior_max_db=float(arrays.interior_max_residual_db))
    return baseline, base


def _improvement(baseline: EndpointBaseline, report: EndpointReport, base: ObjectiveReading) -> EndpointImprovement:
    """同一把尺下基準減終點；數值與 CPU 既有順序相同。"""
    if report.full_objective_total is None or report.full_proxy_db is None:
        raise ValueError("終點報告沒有新合分或大峰代理")
    return EndpointImprovement(
        total=baseline.full_total - report.full_total,
        components=(baseline.full_components[0] - report.full_components[0],
                    baseline.full_components[1] - report.full_components[1]),
        objective_total=base.objective_total - report.full_objective_total,
        proxy_db=base.components[2] - report.full_proxy_db,
        below_scoring=(None if baseline.below_scoring is None or report.below_scoring is None
                       else below_scoring_improvement(baseline.below_scoring, report.below_scoring)))


def _annotate_endpoint(checked: EndpointResult, baseline: EndpointBaseline, pressures: NDArray[np.complex128],
                       model: ReducedModel, scheme: Scheme, objective: GradientObjectiveSettings,
                       ) -> tuple[EndpointReport, EndpointBaseline]:
    """把兩份已算好的完整聲壓各交純加註函式一次，不再解模型。"""
    axis_id = scheme.scene.low_frequency_axis
    if axis_id is None:
        raise ValueError("終點加註需要方案宣告的低頻軸")
    axis = tuple(float(value) for value in model.frequencies_hz)
    original = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), objective.stage_one.purpose)
    def notes(values: NDArray[np.complex128]) -> UnscoredBand:
        return below_scoring_notes(values, speaker_ids=model.source_names, receiver_set=scheme.receiver_set,
                                   frequencies_hz=axis, low_frequency_axis=axis_id,
                                   stage_one_range_hz=original.scored_range_hz,
                                   stage_three_lower_hz=objective.stage_one.scored_range_hz[0])
    report = checked.report.model_copy(update={"below_scoring": notes(checked.full_pressure)})
    baseline = baseline.model_copy(update={"below_scoring": notes(pressures)})
    return report, baseline


def check_operators_digest(project: GradientProject, operators: FullOperators) -> None:
    """沿用搜尋到終點時，用已匯出的完整算子再核開始時的指紋。"""
    from aosr.physics.fem_reduced import operators_digest

    if project.operators_digest is not None and operators_digest(operators) != project.operators_digest:
        raise ValueError("完整模型指紋跟開始時不同，拒收")


def _baseline_for[K: BoundaryKey](scheme: Scheme, model: ReducedModel, objective: GradientObjectiveSettings,
        operators: FullOperators, kept: Mapping[K, SurfaceMaterial], *, workers: int, device: Device,
        refinement_steps: list[int]) -> tuple[NDArray[np.complex128], EndpointBaseline, ObjectiveReading]:
    """原牆完整解一次，同一份聲壓供歷史低頻對照與六類三組數。"""
    assert scheme.scene.low_frequency_axis is not None
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    axis = tuple(float(value) for value in model.frequencies_hz)
    pressures = full_model_pressures(operators, kept, axis, air=air, ranges=ranges,
        max_relative_residual=model.ledger.settings.full_solve_max_relative_residual,
        workers=workers, device=device, refinement_steps=refinement_steps)
    constants = gradient_objective_constants(axis, objective, scheme.receiver_set, len(model.source_names))
    peaks = big_peaks(pressures, speaker_ids=model.source_names, receiver_set=scheme.receiver_set,
        frequencies_hz=axis, low_frequency_axis=scheme.scene.low_frequency_axis, scoring_range_hz=objective.stage_one.scored_range_hz)
    baseline, reading = _full_baseline(constants, pressures, peaks)
    return pressures, baseline, reading


def _endpoint_for(project: GradientProject, scheme: Scheme, model: ReducedModel, settings: MaterialGradientSettings,
        designs: tuple[SurfaceDesign, ...], theta: NDArray[np.float64], operators: FullOperators, *,
        workers: int, device: Device, refinement_steps: list[int]) -> EndpointResult:
    """沿用既有終點聲壓與低頻診斷入口；這一套完整解同時供六類評分。"""
    assert scheme.scene.low_frequency_axis is not None
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    return endpoint_check(operators, designs, theta, model=model, objective_settings=search_objective_settings(project.purpose),
        receiver_set=scheme.receiver_set, low_frequency_axis=scheme.scene.low_frequency_axis,
        presets=load_construction_presets(config_path('construction_presets.toml')), settings=settings, air=air, ranges=ranges,
        workers=workers, frequency_batch=project.frequency_batch, device=device, refinement_steps=refinement_steps)


def _ordered_operators(operators: FullOperators, model: ReducedModel) -> FullOperators:
    """指紋核對後才按替身聲源／座位序重排，不改任何矩陣或向量值。"""
    if set(operators.sources) != set(model.source_names) or set(operators.receivers) != set(model.receiver_names):
        raise ValueError('完整算子的聲源或座位名稱與替身不同')
    return replace(operators, sources={name: operators.sources[name] for name in model.source_names},
                   receivers={name: operators.receivers[name] for name in model.receiver_names})


def _require_low_frequency_axis(scheme: Scheme) -> None:
    if scheme.scene.low_frequency_axis is None:
        raise ValueError('方案場景沒有 low_frequency_axis（低頻軸），終點量大峰要它，拒收')


def run_endpoint(folder: Path, *, workers: int = 1, device: Device = "cpu") -> EndpointRecord:
    """整段終點：核輸入、匯出算子、終點重算、原牆基準、匯出方案檔、寫 ``endpoint.json``；回寫進去的那一份。"""
    checked_device_workers(device, workers)
    if (folder / ENDPOINT_FILE).exists():
        read_endpoint(folder)
    refinement_steps: list[int] = []
    project = read_project(folder)
    if (folder / RESULT_FILE).exists():
        read_result(folder)
    scheme, model, settings = check_inputs(project, allow_legacy_objective=True)
    if not (folder / RESULT_FILE).exists():
        raise ValueError(f"資料夾 {folder} 沒有 {RESULT_FILE}：先把搜尋跑完（start 或 resume）再跑 endpoint")
    result_sha256 = file_sha256(folder / RESULT_FILE)
    _require_low_frequency_axis(scheme)
    given = read_result(folder)
    current = given.format_version in (RESULT_VERSION, MOUNTED_RESULT_VERSION)
    if current and device != 'cpu':
        raise ValueError("六類終點的完整模型與評分使用中央處理器雙精度，請用 cpu（中央處理器）")
    capabilities = load_capabilities(project.capabilities_path)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    clock = time.perf_counter()
    with keep_signal_handlers():
        operators = export_operators(scheme, capabilities=capabilities, directivity=directivity)
    export_seconds = time.perf_counter() - clock
    if operators.mesh_hash != project.mesh_hash or operators.physics_identity != project.physics_identity:
        raise ValueError("方案匯出的完整模型跟開始時記的網格雜湊或物理身分不同，拒收")
    check_operators_digest(project, operators)
    operators = _ordered_operators(operators, model)
    context = scoring_context(scheme, model, project.purpose, capabilities=capabilities, directivity=directivity,
                               capabilities_path=Path(project.capabilities_path), engine_commit=project.engine_commit)
    kept = scene_wall_materials(scheme, capabilities=capabilities, directivity=directivity)
    initial = designs_for(project.presets, settings, kept=kept, mount=project.mount,
                          fixed_surfaces=project.fixed_surfaces, max_depths=project.max_depths)
    result = checked_result(folder, given, header_for(model, initial, settings,
        objective_constants=context.constants if current else search_objective_constants(project, scheme, model)), settings, kept)
    designs = final_designs([(spec.surface_id, spec.preset_id) for spec in result.surfaces], settings, kept,
                            mount=project.mount, fixed_surfaces=project.fixed_surfaces, max_depths=project.max_depths)
    theta = np.asarray([float.fromhex(value) for value in result.theta_hex], dtype=np.float64)
    objective = search_objective_settings(project.purpose)
    clock = time.perf_counter()
    checked = _endpoint_for(project, scheme, model, settings, designs, theta, operators,
                             workers=workers, device=device, refinement_steps=refinement_steps)
    endpoint_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    pressures, baseline, base = _baseline_for(scheme, model, objective, operators, kept,
        workers=workers, device=device, refinement_steps=refinement_steps)
    baseline_seconds = time.perf_counter() - clock
    report, baseline = _annotate_endpoint(checked, baseline, pressures, model, scheme, objective)
    improvement = _improvement(baseline, report, base)
    materials = materials_from_surfaces(designs, theta, settings)
    exported = exported_scheme(scheme, scene_with_materials(scheme.scene, materials), project.gradient_id)
    original_designs = designs_for((None,) * len(initial), settings, kept=kept)
    six = score_endpoint(context, model, designs, np.asarray(report.exported_theta), checked.full_pressure,
                         checked.reduced_pressure, original_designs, pressures, exported)
    write_json_atomic(folder / EXPORTED_SCHEME_FILE, exported)
    record = EndpointRecord(
        device=device, max_refinement_steps=max(refinement_steps) if device == "gpu" else None,
        format_version=ENDPOINT_VERSION, gradient_id=project.gradient_id, result_ledger_sha256=result.ledger_sha256,
        result_template_ledger_sha256=result.template_ledger_sha256, result_sha256=result_sha256,
        report=report, baseline=baseline, six_category=six,
        improvement=improvement,
        exported_scheme=EXPORTED_SCHEME_FILE,
        exported_scheme_sha256=file_sha256(folder / EXPORTED_SCHEME_FILE),
        seconds=EndpointSeconds(export_operators=export_seconds, endpoint=endpoint_seconds, baseline=baseline_seconds),
        finished_at=now_text(), engine_commit=project.engine_commit, fixed_surfaces=project.fixed_surfaces, max_depths=project.max_depths)
    write_json_atomic(folder / ENDPOINT_FILE, record)
    return record
