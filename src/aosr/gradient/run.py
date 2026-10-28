"""牆面材料梯度搜尋的產品接線（#561 辛一）：從資料夾裡的 ``project.json`` 重建目標、把整條鏈跑完、寫帳本與結果。

整條：每面樣板當起點 → 連續下降（投影到盒界 ∩ 總深度）→ 取規格 → 空腔聯合重調 → 候選確認迴圈。
前段三段照帳本的 ``_PRE_LOOP_ORDER``（descent、snap、retune）分三批寫，迴圈交給 ``material_gradient_loop``；
帳本是唯一的接續來源。前段是一個單位：帳本前段沒有以「收好的重調段」結尾（沒有帳本、只有標頭、
只到下降或取規格、重調沒寫完），接續就截到標頭、從頭重跑前段；前段完整或已進迴圈就交給 ``resume_rounds``。

重建目標要在雙精度已開的行程裡（子行程入口 ``aosr.gradient.worker`` 核過才叫這裡）。替身一律經准用閘門
（``load_validated_reduced``，兩把尺從精度契約登記簿讀），而且要對得上 ``project.json`` 記的網格雜湊、物理身分、
manifest 摘要與驗證收據摘要；方案檔與搜尋設定也要跟記的雜湊、指紋相同，任何一格不同就拒收、不跑。
替身的喇叭與座位要跟方案逐一對上（座位照方案座位組的順序）：目標常數照座位組的索引取替身的列，
順序不同會默默算錯座位。

``stop`` 記號（:func:`stop_requested`）在每兩次求值之間查一次，為真拋 ``SearchStopped``；帳本停在最後一列寫完整的地方，
不寫停止原因（迴圈停止原因是封閉集合，不含「人叫停」）。狀態檔與結果檔只由子行程寫，寫完整檔再換名（原子）。
``result.json`` 是交給辛二的接口：最後可行點的每面 preset、厚度與空腔（公尺；空腔 0 照記 0、沒有空腔層記 None）、
θ 的 hex、值與分項、停止原因、求值次數；終點完整模型重算、量大峰、材料匯出不在這裡做。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, NamedTuple

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.precision_contracts import default_precision_contracts_path, load_precision_contracts
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient.material_gradient import (
    ChainObjective, FloatVector, StopCheck, SurfaceDesign, baseline_theta, chain_objective, check_stop,
    compiled_evaluator, depth_limits, design_bounds, projected_descent, retune_cavities, snap_to_specs,
    wall_designs_from_presets,
)
from aosr.gradient.material_gradient_ledger import (
    MaterialGradientHeader, MaterialGradientRow, append_rows, descent_rows, header_for, read_ledger, snap_row,
    truncate_rows, write_ledger,
)
from aosr.gradient.material_gradient_loop import LoopResult, resume_rounds, run_rounds
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, load_validated_reduced, receipt_path
from aosr.physics.fem_reduced_store import manifest_path
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.scoring.stage_one_objective_jax import objective_constants
from aosr.scoring.stage_one_score import stage_one_score_settings


PROJECT_FILE: Final = "project.json"
LEDGER_FILE: Final = "ledger.jsonl"
STATUS_FILE: Final = "status.json"
RESULT_FILE: Final = "result.json"
STOP_FILE: Final = "stop"
PROJECT_VERSION: Final = "aosr.material_gradient_project.v1"
RESULT_VERSION: Final = "aosr.material_gradient_result.v1"
FREQUENCY_RULER: Final = "fem_reduced_holdout_frequency_db"
BAND_RULER: Final = "fem_reduced_holdout_band_db"
_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_SHA256 = r"^[0-9a-f]{64}$"
State = Literal["running", "finished", "stopped", "failed", "interrupted"]


class GradientProject(BaseModel):
    """``project.json``：這一次搜尋的輸入與身分。路徑一律絕對路徑；雜湊是開始時量的，接續時逐格再比。"""

    model_config = _FROZEN
    format_version: Literal["aosr.material_gradient_project.v1"]
    gradient_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    scheme_path: str
    scheme_sha256: str = Field(pattern=_SHA256)
    surrogate_path: str
    surrogate_manifest_sha256: str = Field(pattern=_SHA256)
    receipt_path: str
    receipt_sha256: str = Field(pattern=_SHA256)
    presets: tuple[str | None, ...]
    purpose: str = Field(min_length=1)
    settings_path: str
    settings_fingerprint: str = Field(pattern=_SHA256)
    capabilities_path: str
    physics_identity: str = Field(min_length=1)
    mesh_hash: str = Field(min_length=1)
    engine_commit: str | None
    started_at: str


class GradientStatus(BaseModel):
    """``status.json``：狀態、這一次子行程開始與最後寫的時間、帳本記的累計求值次數、做完幾輪、帳本最後一列的值、
    迴圈收尾的停止原因（收尾才有）、子行程編號、失敗時的錯誤原文。"""

    model_config = _FROZEN
    state: State
    started_at: str
    updated_at: str
    evaluations: int = Field(ge=0, strict=True)
    rounds_done: int = Field(ge=0, strict=True)
    last_value: float | None
    stop_reason: str | None
    pid: int = Field(ge=1, strict=True)
    message: str | None = None


class SurfaceSpec(BaseModel):
    """一面的規格：面代號、preset（剛性面 None）、棉板厚度與空腔（公尺；剛性面都是 None、沒有空腔層的空腔是 None）。"""

    model_config = _FROZEN
    surface_id: str
    preset_id: str | None
    thickness_m: float | None
    cavity_m: float | None


class Components(BaseModel):
    """兩分項（聲道平均）：主位自己那條線的起伏、座位穩健性。"""

    model_config = _FROZEN
    own_line_ripple_rms_db: float
    seat_robustness_rms_db: float


class GradientResult(BaseModel):
    """``result.json``：交給辛二的最後可行點（帳本最後一列決定），與它出自哪一本帳本（整份 SHA-256）。"""

    model_config = _FROZEN
    format_version: Literal["aosr.material_gradient_result.v1"]
    gradient_id: str
    surfaces: tuple[SurfaceSpec, ...]
    variable_names: tuple[str, ...]
    theta_hex: tuple[str, ...]
    value: float
    components: Components
    stop_reason: str
    evaluations: int = Field(ge=0, strict=True)
    rounds: int = Field(ge=1, strict=True)
    ledger_sha256: str = Field(pattern=_SHA256)


class GradientProblem(NamedTuple):
    """重建好的一次搜尋：每面設計、搜尋設定、目標函式、帳本標頭。"""

    designs: tuple[SurfaceDesign, ...]
    settings: MaterialGradientSettings
    objective: ChainObjective
    header: MaterialGradientHeader


# ---- 小工具 ----

def now_text() -> str:
    """UTC 現在時刻的 ISO 8601 文字（秒）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_sha256(npz_path: Path) -> str:
    """替身 manifest 記的整份摘要（manifest 自己核過才可信；呼叫端先過 :func:`load_validated_reduced`）。"""
    document = json.loads(manifest_path(npz_path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("manifest_sha256"), str):
        raise ValueError("替身 manifest 沒有整份摘要")
    return str(document["manifest_sha256"])


def holdout_rulers() -> HoldoutRulers:
    """准用閘門的兩把尺，數字從精度契約登記簿讀，程式裡不寫。"""
    contracts = load_precision_contracts(default_precision_contracts_path())
    return HoldoutRulers(frequency=Ruler(FREQUENCY_RULER, contracts[FREQUENCY_RULER].value),
                         band=Ruler(BAND_RULER, contracts[BAND_RULER].value))


def gradient_id(*, scheme_sha256: str, surrogate_manifest_sha256: str, presets: Sequence[str | None], purpose: str,
                settings_fingerprint: str, physics_identity: str) -> str:
    """資料夾名：輸入（方案、替身、六面 preset、用途、設定、物理身分）正規化 JSON 的 SHA-256 前 32 位。

    同一組輸入一定落在同一個資料夾，兩個 ``start`` 同時跑同一組輸入會搶同一把資料夾鎖。
    """
    content = json.dumps({"scheme": scheme_sha256, "surrogate": surrogate_manifest_sha256, "presets": list(presets),
                          "purpose": purpose, "settings": settings_fingerprint, "physics": physics_identity},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def stop_requested(folder: Path) -> StopCheck:
    """產品的停止判斷：資料夾裡有 ``stop`` 記號就是要停（每次求值之前查一次，只看檔在不在）。"""
    marker = folder / STOP_FILE
    return marker.exists


def read_project(folder: Path) -> GradientProject:
    return GradientProject.model_validate_json((folder / PROJECT_FILE).read_bytes())


def write_json_atomic(path: Path, value: BaseModel) -> None:
    """同資料夾暫存檔寫完、flush 與 fsync 後換名：讀的人只會看到完整的舊檔或完整的新檔。"""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value.model_dump(mode="json"), ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# ---- 替身與方案 ----

def check_surrogate_matches_scheme(model: ReducedModel, scheme: Scheme) -> None:
    """替身的座位要照方案座位組的順序逐一相同、喇叭要是方案的那幾支；不然目標常數會取錯列。"""
    seats = tuple(point.receiver_id for point in scheme.receiver_set.points)
    if model.receiver_names != seats:
        raise ValueError(f"替身的座位 {list(model.receiver_names)} 跟方案座位組 {list(seats)} 不同（名字或順序），"
                         "目標常數會取錯座位，拒收")
    if sorted(model.source_names) != sorted(scheme.speakers):
        raise ValueError(f"替身的喇叭 {list(model.source_names)} 跟方案的 {sorted(scheme.speakers)} 不同，拒收")


def designs_for(presets: Sequence[str | None], settings: MaterialGradientSettings) -> tuple[SurfaceDesign, ...]:
    """六面（照 ``Wall.all()``）的設計；preset 不認識或不在家族清單裡就拒收。"""
    return wall_designs_from_presets(presets, settings, load_construction_presets(config_path("construction_presets.toml")))


def load_problem(project: GradientProject) -> GradientProblem:
    """照 ``project.json`` 重建目標；方案、替身、收據、設定跟記的不同就拒收。要在雙精度已開的行程裡叫。"""
    scheme_path, surrogate = Path(project.scheme_path), Path(project.surrogate_path)
    if file_sha256(scheme_path) != project.scheme_sha256:
        raise ValueError(f"方案檔 {scheme_path} 跟開始時不同（SHA-256 變了），不接續")
    model = load_validated_reduced(surrogate, rulers=holdout_rulers(), expected_mesh_hash=project.mesh_hash,
                                   expected_physics_identity=project.physics_identity)
    if manifest_sha256(surrogate) != project.surrogate_manifest_sha256:
        raise ValueError("替身跟開始時不同（manifest 摘要變了），不接續")
    if file_sha256(receipt_path(surrogate)) != project.receipt_sha256:
        raise ValueError("替身的驗證收據跟開始時不同（SHA-256 變了），不接續")
    settings = load_material_gradient_settings(project.settings_path)
    if settings.fingerprint != project.settings_fingerprint:
        raise ValueError("搜尋設定跟開始時不同（設定指紋變了），不接續")
    scheme = load_scheme(scheme_path)
    check_surrogate_matches_scheme(model, scheme)
    score = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), project.purpose)
    constants = objective_constants(np.asarray(model.frequencies_hz), score, scheme.receiver_set,
                                    len(model.source_names))
    designs = designs_for(project.presets, settings)
    objective = chain_objective(
        model, designs, constants, air=load_physics_constants(config_path("physics_constants.toml")),
        ranges=load_porous_model_ranges(config_path("porous_model_ranges.toml")), frequencies=model.frequencies_hz)
    return GradientProblem(designs, settings, objective, header_for(model, designs, settings,
                                                                    objective_constants=constants))


# ---- 前段與整場 ----

def _pre_loop(problem: GradientProblem, path: Path, should_stop: StopCheck | None) -> FloatVector:
    """連續下降、取規格、重調三段，各自整段算完才寫成一批；回重調完的可行 θ（逐位等於帳本前段最後一列）。"""
    designs, settings, objective, header = problem
    descent = projected_descent(objective, baseline_theta(designs), design_bounds(designs), settings.descent,
                                limits=depth_limits(designs), should_stop=should_stop)
    append_rows(path, header, descent_rows(descent, "descent", 0))
    snapped = snap_to_specs(descent.theta, designs)
    check_stop(should_stop)
    clock = time.perf_counter()
    (value, aux), _ = compiled_evaluator(objective)(jnp.asarray(snapped.theta))
    components = (float(aux.mean_own_line_ripple_rms_db), float(aux.mean_seat_robustness_rms_db))
    append_rows(path, header, (snap_row(snapped, float(value), components, time.perf_counter() - clock,
                                        len(descent.steps)),))
    retune = retune_cavities(objective, snapped.theta, designs, settings, should_stop=should_stop)
    append_rows(path, header, descent_rows(retune, "retune", len(descent.steps) + 1))
    return retune.theta


def _fresh(problem: GradientProblem, path: Path, should_stop: StopCheck | None) -> LoopResult:
    theta = _pre_loop(problem, path, should_stop)
    return run_rounds(problem.objective, theta, problem.designs, problem.settings, path, problem.header,
                      should_stop=should_stop)


def run_search(folder: Path, *, should_stop: StopCheck | None) -> LoopResult:
    """開始：帳本獨佔建檔（已有帳本就丟錯）、先寫標頭，再跑前段與迴圈。"""
    problem = load_problem(read_project(folder))
    path = folder / LEDGER_FILE
    write_ledger(path, problem.header, ())
    return _fresh(problem, path, should_stop)


def pre_loop_complete(rows: Sequence[MaterialGradientRow]) -> bool:
    """前段完整：已經有迴圈的列，或最後一列是帶停止原因的重調列（重調段收好了）。"""
    if any(row.round is not None for row in rows):
        return True
    return bool(rows) and rows[-1].phase == "retune" and rows[-1].stop_reason is not None


def resume_search(folder: Path, *, should_stop: StopCheck | None) -> LoopResult:
    """接續：沒有帳本就從頭；標頭跟這一次重建的不同就拒收；前段不完整就截到標頭重跑前段；否則 ``resume_rounds``。"""
    problem = load_problem(read_project(folder))
    path = folder / LEDGER_FILE
    if not path.exists():
        write_ledger(path, problem.header, ())
        return _fresh(problem, path, should_stop)
    _, rows = read_ledger(path, expected_header=problem.header)
    if pre_loop_complete(rows):
        return resume_rounds(problem.objective, problem.designs, problem.settings, path, problem.header,
                             should_stop=should_stop)
    truncate_rows(path, problem.header, 0)
    return _fresh(problem, path, should_stop)


# ---- 讀帳本：每面規格、進度、結果 ----

def surface_specs(header: MaterialGradientHeader, theta: Sequence[float]) -> tuple[SurfaceSpec, ...]:
    """θ 切回每面規格：每面的變數照「厚度、空腔」排（變數名 ``<面代號>.<樣板變數名>``），剛性面沒有變數。"""
    if len(theta) != len(header.variable_names):
        raise ValueError("θ 長度跟標頭的變數數不同")
    specs = []
    for surface_id, preset_id in header.surfaces:
        values = [value for name, value in zip(header.variable_names, theta, strict=True)
                  if name.split(".", 1)[0] == surface_id]
        if preset_id is None and values:
            raise ValueError(f"剛性面 {surface_id} 不該有變數")
        if preset_id is not None and not 1 <= len(values) <= 2:
            raise ValueError(f"面 {surface_id} 的變數要是厚度或厚度加空腔")
        specs.append(SurfaceSpec(surface_id=surface_id, preset_id=preset_id,
                                 thickness_m=values[0] if values else None,
                                 cavity_m=values[1] if len(values) > 1 else None))
    return tuple(specs)


def ledger_progress(rows: Sequence[MaterialGradientRow]) -> tuple[int, int, float | None, str | None]:
    """帳本記的累計求值次數、做完幾輪、最後一列的值、迴圈收尾的停止原因（沒收尾是 None）。"""
    decisions = [row for row in rows if row.decision is not None]
    evaluations = decisions[-1].decision.total_evaluations if decisions and decisions[-1].decision else 0
    stop_reason = rows[-1].stop_reason if rows and rows[-1].phase == "round_decision" else None
    return evaluations, len(decisions), rows[-1].value if rows else None, stop_reason


def write_result(folder: Path) -> GradientResult:
    """從帳本最後一列決定（要帶停止原因）組 ``result.json``；帳本沒收尾就丟錯、不寫。"""
    path = folder / LEDGER_FILE
    header, rows = read_ledger(path)
    last = rows[-1] if rows else None
    if last is None or last.phase != "round_decision" or last.stop_reason is None or last.decision is None:
        raise ValueError("帳本沒有以帶停止原因的決定列結尾，沒有結果可寫")
    result = GradientResult(
        format_version=RESULT_VERSION, gradient_id=read_project(folder).gradient_id,
        surfaces=surface_specs(header, last.theta()), variable_names=header.variable_names,
        theta_hex=last.theta_hex, value=last.value,
        components=Components(own_line_ripple_rms_db=last.components[0], seat_robustness_rms_db=last.components[1]),
        stop_reason=last.stop_reason, evaluations=last.decision.total_evaluations,
        rounds=sum(1 for row in rows if row.decision is not None), ledger_sha256=file_sha256(path))
    write_json_atomic(folder / RESULT_FILE, result)
    return result
