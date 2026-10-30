"""牆面材料梯度搜尋的終點接線（#561 辛三）：命令列 ``endpoint <資料夾>`` 的子行程那一段。

搜尋跑完（有 ``result.json``）之後，在雙精度子行程裡：輸入走同一道閘門（:func:`aosr.gradient.run.check_inputs`）；
``result.json`` 不只信它自己說的——從兩本帳（庚二帳本與換樣板帳本，標頭都等於這一次重建的）重建整份結果、跟檔裡整個相等才收，
用重建那份的**最後配置**與 θ（庚三：設計從結果的 ``surfaces`` 建，換過樣板的面用新家族的構造、不貼的面用方案原本的牆，
不再從 ``project.presets`` 建）；方案匯出完整模型算子，網格雜湊與物理身分要等於 ``project.json`` 記的；θ 交 ``endpoint.endpoint_check``
（替身值對完整模型值、差與上界、大峰）；基準是方案原本的六面（全部保留原牆）用完整模型解同一條軸、交同一份目標常數；
最後把終點那一套寫成完整方案檔 ``scheme-endpoint.json``，數字寫成 ``endpoint.json``（綁帳本檔、換樣板帳本檔與 ``result.json``
檔的摘要，任一變了它就過期）。兩個檔都整檔寫完再換名。不寫 ``status.json``：終點不是搜尋狀態機的一段。

分數一律 ``gradient_objective_from_pressures``（舊合分就是裡面原樣的 ``objective_from_pressures``）、大峰一律
``endpoint.big_peaks``，不另寫算式。要在雙精度已開的行程裡叫。

v3（#561 壬一）：梯度目標加了大峰代理項；終點報告、原牆基準與改善各自並列舊合分與新合分、代理（老闆 2026-10-30：
「第三場一定要同時列舊分數與新分數」）。v2 只准報告讀（``legacy``），終點不收 v3 之前的結果檔。
"""
from __future__ import annotations

import json
import time
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
from aosr.gradient.cli import keep_signal_handlers
from aosr.gradient.endpoint import (
    EndpointReport, big_peaks, endpoint_check, full_model_pressures, materials_from_surfaces, scene_with_materials,
    worst_big_peak,
)
from aosr.config.material_gradient import MaterialGradientSettings
from aosr.geometry.shoebox import Wall
from aosr.gradient.material_gradient_ledger import MaterialGradientHeader, header_for
from aosr.gradient.run import (
    LEDGER_FILE, RESULT_FILE, GradientResult, check_inputs, designs_for, file_sha256, final_designs, now_text,
    parse_result, read_project, result_from_ledger, scene_wall_materials, search_objective_constants,
    search_objective_settings, template_header_for, write_json_atomic,
)
from aosr.materials.construction_models import SurfaceMaterial
from aosr.physics.report_surfaces import validation_content
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.reporting.scheme import Scene, Scheme
from aosr.scoring.gradient_objective_jax import (
    GradientObjectiveConstants, ObjectiveReading, gradient_objective_constants, gradient_objective_from_pressures,
    objective_reading,
)

ENDPOINT_FILE: Final = "endpoint.json"
EXPORTED_SCHEME_FILE: Final = "scheme-endpoint.json"
ENDPOINT_VERSION: Final = "aosr.material_gradient_endpoint.v3"
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


class EndpointImprovement(BaseModel):
    """改善＝基準減終點（兩邊都是完整模型、同一把尺）；正數是終點比原本好（目標越小越好）。

    ``total``／``components`` 是舊合分與兩分項；``objective_total``／``proxy_db`` 是新合分與代理（v3 起）。"""

    model_config = _FROZEN
    total: float
    components: tuple[float, float]
    objective_total: float | None = None
    proxy_db: float | None = None


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
    """

    model_config = _FROZEN
    format_version: Literal["aosr.material_gradient_endpoint.v3", "aosr.material_gradient_endpoint.v2"]
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

    @model_validator(mode="after")
    def _version_matches_content(self) -> Self:
        current = self.format_version == ENDPOINT_VERSION
        fields = (*self.report.proxy_fields(), self.baseline.full_objective_total, self.baseline.full_proxy_db,
                  self.baseline.full_interior_max_db, self.improvement.objective_total, self.improvement.proxy_db)
        if any((value is not None) != current for value in fields):
            raise ValueError(f"{ENDPOINT_VERSION} 的報告、基準與改善都要帶新合分與大峰代理；"
                             f"{LEGACY_ENDPOINT_VERSION} 都沒有")
        return self


def read_endpoint(folder: Path, *, legacy: bool = False) -> EndpointRecord:
    """讀 ``endpoint.json``：先看版本，不是這一版就拒收、訊息講明版本（要重跑 endpoint）；
    ``legacy=True`` 只給報告用，准讀 v2（加大峰代理之前）。"""
    content = (folder / ENDPOINT_FILE).read_bytes()
    document = json.loads(content)
    version = document.get("format_version") if isinstance(document, dict) else None
    if version != ENDPOINT_VERSION and not (legacy and version == LEGACY_ENDPOINT_VERSION):
        raise ValueError(f"{ENDPOINT_FILE} 版本是 {version!r}，這一版只收 {ENDPOINT_VERSION}；要重跑 endpoint")
    return EndpointRecord.model_validate_json(content)


def read_result(folder: Path, *, legacy: bool = False) -> GradientResult:
    """讀 ``result.json``（:func:`aosr.gradient.run.parse_result`：v2 與更舊的講明版本拒收；``legacy`` 准讀 v3）。"""
    return parse_result((folder / RESULT_FILE).read_bytes(), legacy=legacy)


def exported_scheme(scheme: Scheme, scene: Scene, gradient_id: str) -> Scheme:
    """原方案換場景與方案代號（原代號接 ``-gradient-`` 與梯度 id 前八位，免得跟原方案混淆），其餘欄位不動。"""
    content = validation_content(scheme)
    if not isinstance(content, dict):
        raise TypeError("方案內容必須是欄位字典")
    return Scheme.model_validate({**content, "scene": scene, "scheme_id": f"{scheme.scheme_id}-gradient-{gradient_id[:8]}"})


def checked_result(folder: Path, given: GradientResult, header: MaterialGradientHeader,
                   settings: MaterialGradientSettings, kept: Mapping[Wall, SurfaceMaterial]) -> GradientResult:
    """結果檔不能只信它自己說的：從兩本帳（庚二帳本標頭要等於 ``header``；換樣板帳本標頭要等於照它與現在的庚二帳本摘要
    重建的）重建整份結果，跟檔裡那份整個模型相等才收，回重建那一份；不等就拒收、列出哪幾格不同。"""
    template_header = template_header_for(header, settings, kept, base_ledger_sha256=file_sha256(folder / LEDGER_FILE))
    rebuilt = result_from_ledger(folder, settings, kept, expected_header=header, expected_template_header=template_header)
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


def run_endpoint(folder: Path) -> EndpointRecord:
    """整段終點：核輸入、匯出算子、終點重算、原牆基準、匯出方案檔、寫 ``endpoint.json``；回寫進去的那一份。"""
    project = read_project(folder)
    scheme, model, settings = check_inputs(project)
    if not (folder / RESULT_FILE).exists():
        raise ValueError(f"資料夾 {folder} 沒有 {RESULT_FILE}：先把搜尋跑完（start 或 resume）再跑 endpoint")
    result_sha256 = file_sha256(folder / RESULT_FILE)
    axis_id = scheme.scene.low_frequency_axis
    if axis_id is None:
        raise ValueError("方案場景沒有 low_frequency_axis（低頻軸），終點量大峰要它，拒收")
    capabilities = load_capabilities(project.capabilities_path)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    kept = scene_wall_materials(scheme, capabilities=capabilities, directivity=directivity)
    initial = designs_for(project.presets, settings, kept=kept)
    result = checked_result(folder, read_result(folder), header_for(
        model, initial, settings, objective_constants=search_objective_constants(project, scheme, model)), settings, kept)
    designs = final_designs([(spec.surface_id, spec.preset_id) for spec in result.surfaces], settings, kept)
    clock = time.perf_counter()
    with keep_signal_handlers():
        operators = export_operators(scheme, capabilities=capabilities, directivity=directivity)
    export_seconds = time.perf_counter() - clock
    if operators.mesh_hash != project.mesh_hash or operators.physics_identity != project.physics_identity:
        raise ValueError("方案匯出的完整模型跟開始時記的網格雜湊或物理身分不同，拒收")
    theta = np.asarray([float.fromhex(value) for value in result.theta_hex], dtype=np.float64)
    objective = search_objective_settings(project.purpose)
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    clock = time.perf_counter()
    checked = endpoint_check(operators, designs, theta, model=model, objective_settings=objective,
                             receiver_set=scheme.receiver_set, low_frequency_axis=axis_id,
                             presets=load_construction_presets(config_path("construction_presets.toml")),
                             settings=settings, air=air, ranges=ranges)
    endpoint_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    axis = tuple(float(value) for value in model.frequencies_hz)
    pressures = full_model_pressures(operators, kept, axis, air=air, ranges=ranges,
                                     max_relative_residual=model.ledger.settings.full_solve_max_relative_residual)
    constants = gradient_objective_constants(axis, objective, scheme.receiver_set, len(model.source_names))
    peaks = big_peaks(pressures, speaker_ids=model.source_names, receiver_set=scheme.receiver_set,
                      frequencies_hz=axis, low_frequency_axis=axis_id,
                      scoring_range_hz=objective.stage_one.scored_range_hz)
    baseline, base = _full_baseline(constants, pressures, peaks)
    baseline_seconds = time.perf_counter() - clock
    report = checked.report
    if report.full_objective_total is None or report.full_proxy_db is None:
        raise ValueError("終點報告沒有新合分或大峰代理")
    materials = materials_from_surfaces(designs, theta, settings)
    exported = exported_scheme(scheme, scene_with_materials(scheme.scene, materials), project.gradient_id)
    write_json_atomic(folder / EXPORTED_SCHEME_FILE, exported)
    record = EndpointRecord(
        format_version=ENDPOINT_VERSION, gradient_id=project.gradient_id, result_ledger_sha256=result.ledger_sha256,
        result_template_ledger_sha256=result.template_ledger_sha256, result_sha256=result_sha256,
        report=report, baseline=baseline,
        improvement=EndpointImprovement(
            total=baseline.full_total - report.full_total,
            components=(baseline.full_components[0] - report.full_components[0],
                        baseline.full_components[1] - report.full_components[1]),
            objective_total=base.objective_total - report.full_objective_total,
            proxy_db=base.components[2] - report.full_proxy_db),
        exported_scheme=EXPORTED_SCHEME_FILE,
        exported_scheme_sha256=file_sha256(folder / EXPORTED_SCHEME_FILE),
        seconds=EndpointSeconds(export_operators=export_seconds, endpoint=endpoint_seconds, baseline=baseline_seconds),
        finished_at=now_text(), engine_commit=project.engine_commit)
    write_json_atomic(folder / ENDPOINT_FILE, record)
    return record
