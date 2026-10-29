"""牆面材料梯度搜尋的終點接線（#561 辛三）：命令列 ``endpoint <資料夾>`` 的子行程那一段。

搜尋跑完（有 ``result.json``）之後，在雙精度子行程裡：輸入走同一道閘門（:func:`aosr.gradient.run.check_inputs`）；
方案匯出完整模型算子，網格雜湊與物理身分要等於 ``project.json`` 記的；結果檔的 θ 交 ``endpoint.endpoint_check``
（替身值對完整模型值、差與上界、大峰）；基準是方案原本的六面（全部保留原牆）用完整模型解同一條軸、交同一份目標常數；
最後把終點那一套寫成完整方案檔 ``scheme-endpoint.json``，數字寫成 ``endpoint.json``（綁 ``result.json`` 的帳本摘要，
帳本變了它就過期）。兩個檔都整檔寫完再換名。不寫 ``status.json``：終點不是搜尋狀態機的一段。

分數一律 ``objective_from_pressures``、大峰一律 ``endpoint.big_peaks``，不另寫算式。要在雙精度已開的行程裡叫。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Final, Literal

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.construction_presets import load_construction_presets
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient.cli import keep_signal_handlers
from aosr.gradient.endpoint import (
    EndpointReport, big_peaks, endpoint_check, full_model_pressures, materials_from_surfaces, scene_with_materials,
    worst_big_peak,
)
from aosr.gradient.run import (
    LEDGER_FILE, RESULT_FILE, GradientResult, check_inputs, designs_for, file_sha256, now_text, read_project,
    write_json_atomic,
)
from aosr.physics.report_surfaces import validation_content
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.reporting.scheme import Scene, Scheme, scene_wall_materials
from aosr.scoring.stage_one_objective_jax import objective_constants, objective_from_pressures
from aosr.scoring.stage_one_score import stage_one_score_settings

ENDPOINT_FILE: Final = "endpoint.json"
EXPORTED_SCHEME_FILE: Final = "scheme-endpoint.json"
ENDPOINT_VERSION: Final = "aosr.material_gradient_endpoint.v1"
_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_SHA256 = r"^[0-9a-f]{64}$"


class EndpointBaseline(BaseModel):
    """基準：方案原本六面（全部保留原牆）用完整模型解終點同一條軸、交同一份目標常數的值與分項、每支喇叭的大峰。"""

    model_config = _FROZEN
    full_total: float
    full_components: tuple[float, float]
    big_peak_db_by_speaker: dict[str, float | None]
    big_peak_db: float | None
    full_solves: int = Field(ge=1, strict=True)


class EndpointImprovement(BaseModel):
    """改善＝基準減終點（兩邊都是完整模型、同一把尺）；正數是終點比原本好（目標越小越好）。"""

    model_config = _FROZEN
    total: float
    components: tuple[float, float]


class EndpointSeconds(BaseModel):
    """各段牆鐘秒數：匯出完整模型算子、終點重算（``endpoint_check``）、基準完整解。"""

    model_config = _FROZEN
    export_operators: float = Field(ge=0.0)
    endpoint: float = Field(ge=0.0)
    baseline: float = Field(ge=0.0)


class EndpointRecord(BaseModel):
    """``endpoint.json``：終點報告、原牆基準、改善、匯出方案檔與它的 SHA-256、各段秒數。

    ``result_ledger_sha256`` 抄 ``result.json`` 的 ``ledger_sha256``：之後帳本變了（再接續過），報告就認得出這份過期。
    ``engine_commit`` 是 ``project.json`` 記的（開始時給的那一個），終點不另量。
    """

    model_config = _FROZEN
    format_version: Literal["aosr.material_gradient_endpoint.v1"]
    gradient_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    result_ledger_sha256: str = Field(pattern=_SHA256)
    report: EndpointReport
    baseline: EndpointBaseline
    improvement: EndpointImprovement
    exported_scheme: str = Field(min_length=1)
    exported_scheme_sha256: str = Field(pattern=_SHA256)
    seconds: EndpointSeconds
    finished_at: str
    engine_commit: str | None


def read_endpoint(folder: Path) -> EndpointRecord:
    return EndpointRecord.model_validate_json((folder / ENDPOINT_FILE).read_bytes())


def read_result(folder: Path) -> GradientResult:
    return GradientResult.model_validate_json((folder / RESULT_FILE).read_bytes())


def exported_scheme(scheme: Scheme, scene: Scene, gradient_id: str) -> Scheme:
    """原方案換場景與方案代號（原代號接 ``-gradient-`` 與梯度 id 前八位，免得跟原方案混淆），其餘欄位不動。"""
    content = validation_content(scheme)
    if not isinstance(content, dict):
        raise TypeError("方案內容必須是欄位字典")
    return Scheme.model_validate({**content, "scene": scene, "scheme_id": f"{scheme.scheme_id}-gradient-{gradient_id[:8]}"})


def run_endpoint(folder: Path) -> EndpointRecord:
    """整段終點：核輸入、匯出算子、終點重算、原牆基準、匯出方案檔、寫 ``endpoint.json``；回寫進去的那一份。"""
    project = read_project(folder)
    scheme, model, settings = check_inputs(project)
    if not (folder / RESULT_FILE).exists():
        raise ValueError(f"資料夾 {folder} 沒有 {RESULT_FILE}：先把搜尋跑完（start 或 resume）再跑 endpoint")
    result = read_result(folder)
    if result.gradient_id != project.gradient_id:
        raise ValueError(f"{RESULT_FILE} 的 gradient_id 跟 project.json 不同，拒收")
    if file_sha256(folder / LEDGER_FILE) != result.ledger_sha256:
        raise ValueError(f"{RESULT_FILE} 跟現在的帳本對不上（ledger_sha256 不同），先把搜尋跑完再跑 endpoint")
    axis_id = scheme.scene.low_frequency_axis
    if axis_id is None:
        raise ValueError("方案場景沒有 low_frequency_axis（低頻軸），終點量大峰要它，拒收")
    clock = time.perf_counter()
    with keep_signal_handlers():
        operators = export_operators(scheme, capabilities=load_capabilities(project.capabilities_path),
                                     directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    export_seconds = time.perf_counter() - clock
    if operators.mesh_hash != project.mesh_hash or operators.physics_identity != project.physics_identity:
        raise ValueError("方案匯出的完整模型跟開始時記的網格雜湊或物理身分不同，拒收")
    kept = scene_wall_materials(scheme.scene)
    designs = designs_for(project.presets, settings, kept=kept)
    theta = np.asarray([float.fromhex(value) for value in result.theta_hex], dtype=np.float64)
    score = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), project.purpose)
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    clock = time.perf_counter()
    checked = endpoint_check(operators, designs, theta, model=model, score_settings=score,
                             receiver_set=scheme.receiver_set, low_frequency_axis=axis_id,
                             presets=load_construction_presets(config_path("construction_presets.toml")),
                             settings=settings, air=air, ranges=ranges)
    endpoint_seconds = time.perf_counter() - clock
    clock = time.perf_counter()
    axis = tuple(float(value) for value in model.frequencies_hz)
    pressures = full_model_pressures(operators, kept, axis, air=air, ranges=ranges,
                                     max_relative_residual=model.ledger.settings.full_solve_max_relative_residual)
    constants = objective_constants(axis, score, scheme.receiver_set, len(model.source_names))
    base = objective_from_pressures(constants, jnp.asarray(pressures))
    peaks = big_peaks(pressures, speaker_ids=model.source_names, receiver_set=scheme.receiver_set,
                      frequencies_hz=axis, low_frequency_axis=axis_id, scoring_range_hz=score.scored_range_hz)
    baseline_seconds = time.perf_counter() - clock
    baseline = EndpointBaseline(
        full_total=float(base.screening_total),
        full_components=(float(base.mean_own_line_ripple_rms_db), float(base.mean_seat_robustness_rms_db)),
        big_peak_db_by_speaker=peaks, big_peak_db=worst_big_peak(peaks), full_solves=len(axis))
    report = checked.report
    materials = materials_from_surfaces(designs, theta, settings)
    exported = exported_scheme(scheme, scene_with_materials(scheme.scene, materials), project.gradient_id)
    write_json_atomic(folder / EXPORTED_SCHEME_FILE, exported)
    record = EndpointRecord(
        format_version=ENDPOINT_VERSION, gradient_id=project.gradient_id, result_ledger_sha256=result.ledger_sha256,
        report=report, baseline=baseline,
        improvement=EndpointImprovement(
            total=baseline.full_total - report.full_total,
            components=(baseline.full_components[0] - report.full_components[0],
                        baseline.full_components[1] - report.full_components[1])),
        exported_scheme=EXPORTED_SCHEME_FILE,
        exported_scheme_sha256=file_sha256(folder / EXPORTED_SCHEME_FILE),
        seconds=EndpointSeconds(export_operators=export_seconds, endpoint=endpoint_seconds, baseline=baseline_seconds),
        finished_at=now_text(), engine_commit=project.engine_commit)
    write_json_atomic(folder / ENDPOINT_FILE, record)
    return record
