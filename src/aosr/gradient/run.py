"""牆面材料梯度搜尋的產品接線（#561 辛一）：從資料夾裡的 ``project.json`` 重建目標、把整條鏈跑完、寫帳本與結果。

整條：每面樣板當起點 → 連續下降（投影到盒界 ∩ 總深度）→ 取規格 → 空腔聯合重調 → 候選確認迴圈。
前段三段照帳本的 ``_PRE_LOOP_ORDER``（descent、snap、retune）分三批寫，迴圈交給 ``material_gradient_loop``；
帳本是唯一的接續來源。前段是一個單位：帳本前段沒有以「收好的重調段」結尾（沒有帳本、只有標頭、
只到下降或取規格、重調沒寫完），接續就截到標頭、從頭重跑前段；前段完整或已進迴圈就交給 ``resume_rounds``。

重建目標要在雙精度已開的行程裡（子行程入口 ``aosr.gradient.worker`` 核過才叫這裡）。替身一律經准用閘門
（``load_validated_reduced``，兩把尺從精度契約登記簿讀），而且要對得上 ``project.json`` 記的網格雜湊、物理身分、
manifest 摘要與驗證收據摘要；方案檔與搜尋設定也要跟記的雜湊、指紋相同，任何一格不同就拒收、不跑。
替身的喇叭與座位要跟方案逐一對上（座位照方案座位組的順序）：目標常數照座位組的索引取替身的列，
順序不同會默默算錯座位。位置也要對上（審查 B1）：網格雜湊與物理身分都不含喇叭與座位座標，同一間房把喇叭移 0.1 m
拿舊替身會照樣過閘門、目標靜默算錯；所以開始、接續（父行程與子行程）都把方案的座標（``float.hex()``）交給閘門，
跟驗證收據記的逐名逐座標比，``project.json`` 也記一份。

``stop`` 記號（:func:`stop_requested`）在每兩次求值之間查一次，為真拋 ``SearchStopped``；帳本停在最後一列寫完整的地方，
不寫停止原因（迴圈停止原因是封閉集合，不含「人叫停」）。狀態檔與結果檔只由子行程寫，寫完整檔再換名（原子）。
``result.json`` 是交給終點的接口：最後可行點的每面 preset、厚度與空腔（公尺；空腔 0 照記 0、沒有空腔層記 None）、
材料摘要、θ 的 hex、值與分項、停止原因、求值次數；終點完整模型重算、量大峰、材料匯出在 ``endpoint_run``。
preset 寫 none 的面是「不貼、保留方案原本的牆」（#561 辛三）：每面設計帶方案那一面原本的材料（:func:`scene_wall_materials`），
從替身鏈起就用它的阻抗，不是剛性。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, NamedTuple

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.construction_presets import load_construction_presets
from aosr.config.directivity_defaults import DirectivityDefaults, load_directivity_defaults
from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.construction_environment import construction_environment
from aosr.config.precision_contracts import default_precision_contracts_path, load_precision_contracts
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.gradient.endpoint import design_digest
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
from aosr.materials.construction_models import SurfaceMaterial
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_check import PointHex, points_hex
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, load_validated_reduced, receipt_path
from aosr.physics.fem_reduced_store import manifest_path
from aosr.physics.report_io import solver_inputs
from aosr.reporting.fem_reduced_scheme import scheme_points
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.reporting.validation import checked_inputs
from aosr.scoring.stage_one_low_frequency import axis_reason_not_measurable
from aosr.scoring.stage_one_objective_jax import StageOneObjectiveConstants, objective_constants
from aosr.scoring.stage_one_score import StageOneScoreSettings, stage_one_score_settings


PROJECT_FILE: Final = "project.json"
LEDGER_FILE: Final = "ledger.jsonl"
STATUS_FILE: Final = "status.json"
RESULT_FILE: Final = "result.json"
STOP_FILE: Final = "stop"
PROJECT_VERSION: Final = "aosr.material_gradient_project.v1"
RESULT_VERSION: Final = "aosr.material_gradient_result.v2"
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
    source_points: dict[str, PointHex]
    receiver_points: dict[str, PointHex]
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
    """一面的規格：面代號、preset（不貼的面 None）、棉板厚度與空腔（公尺；不貼的面都是 None、沒有空腔層的空腔是 None）、
    材料摘要（``endpoint.design_digest``：有樣板的面是換 θ 之前的樣板構造的、不貼的面是保留的原牆材料的；
    報告只要厚度與空腔時是 None）。"""

    model_config = _FROZEN
    surface_id: str
    preset_id: str | None
    thickness_m: float | None
    cavity_m: float | None
    material_digest: str | None


class Components(BaseModel):
    """兩分項（聲道平均）：主位自己那條線的起伏、座位穩健性。"""

    model_config = _FROZEN
    own_line_ripple_rms_db: float
    seat_robustness_rms_db: float


class GradientResult(BaseModel):
    """``result.json``：交給終點的最後可行點（帳本最後一列決定），與它出自哪一本帳本（整份 SHA-256）。

    v2（#561 辛三）每面多了材料摘要；v1 的檔沒有在用、不收。
    """

    model_config = _FROZEN
    format_version: Literal["aosr.material_gradient_result.v2"]
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


class SchemePoints(NamedTuple):
    """方案的喇叭與座位座標：名字 → 三個 ``float.hex()``。"""

    sources: dict[str, PointHex]
    receivers: dict[str, PointHex]


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

def scheme_points_hex(scheme: Scheme) -> SchemePoints:
    """方案的喇叭與座位座標（跟匯出完整模型算子同一個取法）。"""
    sources, receivers = scheme_points(scheme)
    return SchemePoints(points_hex(sources), points_hex(receivers))


def gated_surrogate(surrogate: Path, *, mesh_hash: str, physics_identity: str, points: SchemePoints) -> ReducedModel:
    """准用閘門：兩把尺從登記簿讀，網格雜湊、物理身分、喇叭與座位座標都要跟驗證收據對上。"""
    return load_validated_reduced(surrogate, rulers=holdout_rulers(), expected_mesh_hash=mesh_hash,
                                  expected_physics_identity=physics_identity, expected_source_points=points.sources,
                                  expected_receiver_points=points.receivers)


def check_surrogate_matches_scheme(model: ReducedModel, scheme: Scheme, points: SchemePoints) -> None:
    """替身的座位要照方案座位組的順序逐一相同、喇叭要是方案的那幾支（不然目標常數會取錯列）；
    替身建模用的空氣（密度、聲速）要逐位等於方案場景的（不然材料阻抗用方案的空氣、縮減方程係數用替身的空氣，
    一個目標混用兩組條件；網格不取密度，同網格、同座標、另一組空氣建的合法替身真的存在）；
    交給閘門比的座標要就是這份方案的座標（不然閘門比的是別的位置）。"""
    seats = tuple(point.receiver_id for point in scheme.receiver_set.points)
    if model.receiver_names != seats:
        raise ValueError(f"替身的座位 {list(model.receiver_names)} 跟方案座位組 {list(seats)} 不同（名字或順序），"
                         "目標常數會取錯座位，拒收")
    if sorted(model.source_names) != sorted(scheme.speakers):
        raise ValueError(f"替身的喇叭 {list(model.source_names)} 跟方案的 {sorted(scheme.speakers)} 不同，拒收")
    built = (model.density_kg_m3, model.sound_speed_m_s)
    wanted = (scheme.scene.density_kg_m3, scheme.scene.sound_speed_m_s)
    if built != wanted:
        raise ValueError(f"替身建模用的空氣跟方案的不同（替身 密度 {built[0]!r}、聲速 {built[1]!r}；"
                         f"方案 密度 {wanted[0]!r}、聲速 {wanted[1]!r}），拒收")
    _check_same_points(points, scheme_points_hex(scheme), "交給閘門的")


def check_surrogate_axis_covers_scoring(model: ReducedModel, scheme: Scheme, score: StageOneScoreSettings) -> None:
    """替身的頻率軸要算得了這一次的計分（#984，顧問 #971）：照方案場景宣告的低頻軸、計分範圍，走低頻評估器的
    宣告點與缺段判法（:func:`axis_reason_not_measurable`）；不可估就拒收，不拿剩下的頻點代替整段。"""
    axis = scheme.scene.low_frequency_axis
    if axis is None:
        raise ValueError("方案場景沒有 low_frequency_axis（低頻軸）：搜尋跑完之後終點量大峰要它，開始前就拒收")
    bounds = score.scored_range_hz
    reason = axis_reason_not_measurable(model.frequencies_hz, axis, bounds)
    if reason is not None:
        inside = sum(1 for frequency in model.frequencies_hz if bounds[0] <= frequency <= bounds[1])
        raise ValueError(f"替身的頻率軸算不了這一次的計分（{reason.value}）：宣告軸 {axis.value}、"
                         f"計分範圍 {bounds[0]!r} 到 {bounds[1]!r} Hz；替身共 {len(model.frequencies_hz)} 點、"
                         f"範圍內 {inside} 點，拒收")


def search_score_settings(purpose: str) -> StageOneScoreSettings:
    """這一次搜尋用途的第一階段評分設定（品質登記簿那一段）。"""
    return stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), purpose)


def _check_same_points(given: SchemePoints, scheme: SchemePoints, what: str) -> None:
    """逐名逐座標比，第一個不同的就點名是哪一支喇叭或哪一個座位。"""
    for label, recorded, wanted in (("喇叭", given.sources, scheme.sources), ("座位", given.receivers, scheme.receivers)):
        for name in sorted(set(recorded) | set(wanted)):
            if recorded.get(name) != wanted.get(name):
                raise ValueError(f"{label} {name} 的座標：{what}是 {recorded.get(name)}，方案是 {wanted.get(name)}，拒收")


def check_inputs(project: GradientProject) -> tuple[Scheme, ReducedModel, MaterialGradientSettings]:
    """接續前逐格核輸入（父行程起子行程前、子行程重建目標前各核一次）：方案檔雜湊、方案座標跟 ``project.json`` 記的、
    替身過閘門（網格雜湊、物理身分、座標）、manifest 與收據摘要、設定指紋、替身對得上方案、替身頻率軸算得了
    這一次的計分。任何一格不同就拒收。"""
    scheme_path, surrogate = Path(project.scheme_path), Path(project.surrogate_path)
    if file_sha256(scheme_path) != project.scheme_sha256:
        raise ValueError(f"方案檔 {scheme_path} 跟開始時不同（scheme_sha256 變了），不接續")
    scheme = load_scheme(scheme_path)
    points = scheme_points_hex(scheme)
    _check_same_points(SchemePoints(project.source_points, project.receiver_points), points, "project.json 記的")
    model = gated_surrogate(surrogate, mesh_hash=project.mesh_hash, physics_identity=project.physics_identity,
                            points=points)
    if manifest_sha256(surrogate) != project.surrogate_manifest_sha256:
        raise ValueError("替身跟開始時不同（manifest 摘要變了），不接續")
    if file_sha256(receipt_path(surrogate)) != project.receipt_sha256:
        raise ValueError("替身的驗證收據跟開始時不同（SHA-256 變了），不接續")
    settings = load_material_gradient_settings(project.settings_path)
    if settings.fingerprint != project.settings_fingerprint:
        raise ValueError("搜尋設定跟開始時不同（設定指紋變了），不接續")
    check_surrogate_matches_scheme(model, scheme, points)
    check_surrogate_axis_covers_scoring(model, scheme, search_score_settings(project.purpose))
    return scheme, model, settings


def scene_wall_materials(scheme: Scheme, *, capabilities: CapabilityTable,
                         directivity: DirectivityDefaults) -> dict[Wall, SurfaceMaterial]:
    """方案原本六面的牆面材料（照 ``Wall.all()``）：走報表求解入口那一條——``checked_inputs`` 驗方案、取第一對輸入
    （跟 ``export_operators`` 同一個取法）交 ``report_io.solver_inputs``，用它的 ``materials_by_wall``；
    固定阻抗只在那裡建，這裡不另轉。"""
    _, documents = checked_inputs(scheme, capabilities=capabilities, directivity=directivity)
    return dict(solver_inputs(next(iter(documents.values()))[1]).materials_by_wall)


def project_wall_materials(project: GradientProject, scheme: Scheme) -> dict[Wall, SurfaceMaterial]:
    """:func:`scene_wall_materials`，能力表取 ``project.json`` 記的、指向性取產品預設（跟 ``resume`` 核物理身分同一組）。"""
    return scene_wall_materials(scheme, capabilities=load_capabilities(project.capabilities_path),
                                directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))


def search_objective_constants(project: GradientProject, scheme: Scheme,
                               model: ReducedModel) -> StageOneObjectiveConstants:
    """這一次搜尋的目標常數：替身的頻率軸、``project.purpose`` 的評分設定、方案座位組、替身的喇叭數。"""
    score = search_score_settings(project.purpose)
    return objective_constants(np.asarray(model.frequencies_hz), score, scheme.receiver_set, len(model.source_names))


def designs_for(presets: Sequence[str | None], settings: MaterialGradientSettings, *,
                kept: Mapping[Wall, SurfaceMaterial]) -> tuple[SurfaceDesign, ...]:
    """六面（照 ``Wall.all()``）的設計；preset 不認識或不在家族清單裡就拒收。preset 是 None 的面一律帶 ``kept``
    那一面的材料（方案原本的牆，``scene_wall_materials``），是「不貼、保留原牆」，不是剛性。"""
    return wall_designs_from_presets(presets, settings, load_construction_presets(config_path("construction_presets.toml")),
                                     kept={wall.wall_name(): material for wall, material in kept.items()})


def load_problem(project: GradientProject) -> GradientProblem:
    """照 ``project.json`` 重建目標；輸入先過 :func:`check_inputs`。要在雙精度已開的行程裡叫。

    空氣的密度與聲速取方案場景那一格（``export_operators`` 建完整模型、替身建模用的同一組），熱黏常數與多孔範圍
    走設定層的 ``construction_environment``；這裡不自己拿產品預設的空氣。
    """
    scheme, model, settings = check_inputs(project)
    constants = search_objective_constants(project, scheme, model)
    designs = designs_for(project.presets, settings, kept=project_wall_materials(project, scheme))
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    objective = chain_objective(model, designs, constants, air=air, ranges=ranges, frequencies=model.frequencies_hz)
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
    """開始：帳本獨佔建檔（已有帳本就丟錯）、先寫標頭，再跑前段與迴圈；跑完寫 ``result.json``。"""
    problem = load_problem(read_project(folder))
    path = folder / LEDGER_FILE
    write_ledger(path, problem.header, ())
    result = _fresh(problem, path, should_stop)
    write_result(folder, problem.designs)
    return result


def pre_loop_complete(rows: Sequence[MaterialGradientRow]) -> bool:
    """前段完整：已經有迴圈的列，或最後一列是帶停止原因的重調列（重調段收好了）。"""
    if any(row.round is not None for row in rows):
        return True
    return bool(rows) and rows[-1].phase == "retune" and rows[-1].stop_reason is not None


def resume_search(folder: Path, *, should_stop: StopCheck | None) -> LoopResult:
    """接續：沒有帳本就從頭；標頭跟這一次重建的不同就拒收；前段不完整就截到標頭重跑前段；否則 ``resume_rounds``。
    跑完寫 ``result.json``。"""
    problem = load_problem(read_project(folder))
    result = _resume(problem, folder / LEDGER_FILE, should_stop)
    write_result(folder, problem.designs)
    return result


def _resume(problem: GradientProblem, path: Path, should_stop: StopCheck | None) -> LoopResult:
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

def surface_specs(header: MaterialGradientHeader, theta: Sequence[float], *,
                  digests: Mapping[str, str | None] | None = None) -> tuple[SurfaceSpec, ...]:
    """θ 切回每面規格：每面的變數照「厚度、空腔」排（變數名 ``<面代號>.<樣板變數名>``），不貼的面沒有變數。
    ``digests``（面代號 → 材料摘要）給了就照填、缺一面拒收；沒給（報告只要厚度與空腔）摘要是 None。"""
    if len(theta) != len(header.variable_names):
        raise ValueError("θ 長度跟標頭的變數數不同")
    specs = []
    for surface_id, preset_id in header.surfaces:
        values = [value for name, value in zip(header.variable_names, theta, strict=True)
                  if name.split(".", 1)[0] == surface_id]
        if preset_id is None and values:
            raise ValueError(f"不貼的面 {surface_id} 不該有變數")
        if digests is not None and surface_id not in digests:
            raise ValueError(f"材料摘要缺面 {surface_id}")
        if preset_id is not None and not 1 <= len(values) <= 2:
            raise ValueError(f"面 {surface_id} 的變數要是厚度或厚度加空腔")
        specs.append(SurfaceSpec(surface_id=surface_id, preset_id=preset_id,
                                 thickness_m=values[0] if values else None,
                                 cavity_m=values[1] if len(values) > 1 else None,
                                 material_digest=None if digests is None else digests[surface_id]))
    return tuple(specs)


def ledger_progress(rows: Sequence[MaterialGradientRow]) -> tuple[int, int, float | None, str | None]:
    """帳本記的累計求值次數、做完幾輪、最後一列的值、迴圈收尾的停止原因（沒收尾是 None）。"""
    decisions = [row for row in rows if row.decision is not None]
    evaluations = decisions[-1].decision.total_evaluations if decisions and decisions[-1].decision else 0
    stop_reason = rows[-1].stop_reason if rows and rows[-1].phase == "round_decision" else None
    return evaluations, len(decisions), rows[-1].value if rows else None, stop_reason


def result_from_ledger(folder: Path, designs: Sequence[SurfaceDesign], *,
                       expected_header: MaterialGradientHeader | None = None) -> GradientResult:
    """從帳本最後一列決定（要帶停止原因）重建整份結果，不寫檔；帳本沒收尾就丟錯。給了 ``expected_header`` 帳本標頭要
    逐格相等。每面材料摘要從設計清單（``endpoint.design_digest``）來，設計清單的面要跟帳本標頭同一串。
    寫 ``result.json`` 與終點核結果檔都走這一支（終點拿它跟檔裡那份整個比）。"""
    path = folder / LEDGER_FILE
    header, rows = read_ledger(path, expected_header=expected_header)
    if tuple((design.surface_id, design.preset_id) for design in designs) != header.surfaces:
        raise ValueError("設計清單的面跟帳本標頭不同，不寫結果")
    last = rows[-1] if rows else None
    if last is None or last.phase != "round_decision" or last.stop_reason is None or last.decision is None:
        raise ValueError("帳本沒有以帶停止原因的決定列結尾，沒有結果可寫")
    return GradientResult(
        format_version=RESULT_VERSION, gradient_id=read_project(folder).gradient_id,
        surfaces=surface_specs(header, last.theta(),
                               digests={design.surface_id: design_digest(design) for design in designs}),
        variable_names=header.variable_names,
        theta_hex=last.theta_hex, value=last.value,
        components=Components(own_line_ripple_rms_db=last.components[0], seat_robustness_rms_db=last.components[1]),
        stop_reason=last.stop_reason, evaluations=last.decision.total_evaluations,
        rounds=sum(1 for row in rows if row.decision is not None), ledger_sha256=file_sha256(path))


def write_result(folder: Path, designs: Sequence[SurfaceDesign]) -> GradientResult:
    """:func:`result_from_ledger` 重建、原子寫成 ``result.json``。"""
    result = result_from_ledger(folder, designs)
    write_json_atomic(folder / RESULT_FILE, result)
    return result
