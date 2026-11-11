"""牆面材料梯度搜尋的產品接線（#561 辛一）：從資料夾裡的 ``project.json`` 重建目標、把整條鏈跑完、寫帳本與結果。

整條：每面樣板當起點 → 連續下降（投影到盒界 ∩ 總深度）→ 取規格 → 空腔聯合重調 → 候選確認迴圈（庚二）
→ 換樣板（庚三，``template_segment``，寫 ``templates.jsonl``）→ 寫結果。
前段三段照帳本的 ``_PRE_LOOP_ORDER``（descent、snap、retune）分三批寫，迴圈交給 ``material_gradient_loop``；
帳本是唯一的接續來源。前段是一個單位：帳本前段沒有以「收好的重調段」結尾（沒有帳本、只有標頭、
只到下降或取規格、重調沒寫完），接續就截到標頭、從頭重跑前段；前段完整或已進迴圈就交給 ``resume_rounds``。
接續看資料夾在哪一段：``ledger.jsonl`` 沒收尾就接庚二（這時 ``templates.jsonl`` 在是矛盾、拒收）；收尾了、``templates.jsonl``
不在就開換樣板段（以庚二收尾那一列的 θ 與 ``project.json`` 的六面當起點）；在就接它。

重建目標要在雙精度已開的行程裡（子行程入口 ``aosr.gradient.worker`` 核過才叫這裡）。替身一律經准用閘門
（``load_validated_reduced``，兩把尺從精度契約登記簿讀），而且要對得上 ``project.json`` 記的網格雜湊、物理身分、
manifest 摘要與驗證收據摘要；方案檔與搜尋設定也要跟記的雜湊、指紋相同，任何一格不同就拒收、不跑。
替身的喇叭與座位要跟方案逐一對上（座位照方案座位組的順序）：目標常數照座位組的索引取替身的列，
順序不同會默默算錯座位。位置也要對上（審查 B1）：網格雜湊與物理身分都不含喇叭與座位座標，同一間房把喇叭移 0.1 m
拿舊替身會照樣過閘門、目標靜默算錯；所以開始、接續（父行程與子行程）都把方案的座標（``float.hex()``）交給閘門，
跟驗證收據記的逐名逐座標比，``project.json`` 也記一份。

``stop`` 記號（:func:`stop_requested`）在每兩次求值之間查一次，為真拋 ``SearchStopped``；帳本停在最後一列寫完整的地方，
不寫停止原因（迴圈停止原因是封閉集合，不含「人叫停」）。狀態檔與結果檔只由子行程寫，寫完整檔再換名（原子）。
``result.json``（v3）是交給終點的接口：**最後的**整房配置（換樣板帳本最後一列決定）的每面 preset、厚度與空腔
（公尺；空腔 0 照記 0、沒有空腔層記 None）、材料摘要、變數名、θ 的 hex、值與分項；庚二與換樣板兩段各自的停止原因、輪數、
求值次數與帳本摘要；終點完整模型重算、量大峰、材料匯出在 ``endpoint_run``。
preset 寫 none 的面是「不貼、保留方案原本的牆」（#561 辛三）：每面設計帶方案那一面原本的材料（:func:`scene_wall_materials`），
從替身鏈起就用它的阻抗，不是剛性。

#561 壬一（老闆 2026-10-30 拍 B）：梯度目標＝第一階段兩分項合分＋權重 × 大峰代理（``scoring.gradient_objective_jax``）。
``result.json`` 升 v4：``value`` 是新合分，另帶 ``screening_total``（舊合分）與三分項（含代理）；報告讀得回 v3（沒有代理）。
``project.json`` 多記梯度目標設定指紋（``objective_settings_fingerprint``：登記簿那三格加第一階段設定），資料夾名也算進去；
接續時重算比對，沒記（加代理之前開始的）或不同就拒收、講明梯度目標加了大峰代理項要重跑。

#990：安裝方式（``mount``，每次搜尋一個值）。``on_wall`` 是「貼在原牆上」：貼樣板的面新增層背後接那一面原本的牆
（:func:`scene_wall_materials`），換樣板沿用；``rigid`` 是原本的行為。新方式的分數不准沿用舊的：資料夾名只在非剛背時
多雜湊 ``"mount"``（剛背逐位不變）；``project.json`` 記 ``mount``（剛背不寫、缺這一格讀成剛背，舊資料夾照舊接續），
接續時跟帳本標頭逐格比（標頭也記安裝方式）。``result.json``：剛背照舊寫 v4（逐位等於加這一格之前，v4 就是剛背的
正式結果，終點照收）；貼在原牆上寫 v5，每面記安裝方式、摘要是貼牆構造的。v3 只准報告讀。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, NamedTuple, Protocol, Self, cast

import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter, BaseModel, ConfigDict, Field, model_validator

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.directivity_defaults import DirectivityDefaults, load_directivity_defaults
from aosr.config.material_gradient import (
    MOUNT_ON_WALL, MOUNT_RIGID, MaterialGradientSettings, Mount, is_fixed_target, load_material_gradient_settings, with_max_steps,
)
from aosr.config.paths import config_path
from aosr.config.construction_environment import construction_environment
from aosr.config.precision_contracts import default_precision_contracts_path, load_precision_contracts
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.gradient.gpu import Device, LegacyPrecisionComparison, PrecisionComparison, single_precision_constants
from aosr.gradient.gpu_preflight import ensure_start_precision
from aosr.gradient.surrogate_guard import require_matching_patches, require_supported_original_walls as require_supported_original_walls
from aosr.gradient.fixed_candidates import FixedCandidateCheck
from aosr.gradient.fixed_surfaces import (
    AllowedTargets, FixedSurfaces, MaxDepths, check_fixed_depth_conflicts, designs_with_fixed, designs_with_max_depths,
    expanded_allowed_targets, fixed_targets_of,
)
from aosr.gradient.endpoint import design_digest, material_digest
from aosr.gradient.material_gradient import (
    COMPONENT_NAMES, Unestimable, ChainObjective, FloatVector, StopCheck, SurfaceDesign, baseline_theta, check_stop, objective_reading,
    designs_from_presets, is_feasible,
    compiled_evaluator, depth_limits, design_bounds, design_names, projected_descent, retune_cavities, snap_to_specs,
    wall_designs_from_presets, estimable,
)
from aosr.gradient.material_gradient_ledger import (
    LEDGER_VERSION, LEGACY_LEDGER_VERSION, PROXY_RERUN_TEXT, SCORED_RANGE_RERUN_TEXT, MaterialGradientHeader, MaterialGradientRow, append_rows, descent_rows, header_for, read_ledger, snap_row,
    truncate_rows, write_ledger,
)
from aosr.gradient.material_gradient_loop import LoopResult, resume_rounds, run_rounds
from aosr.gradient.template_ledger import (
    TEMPLATE_LEDGER_VERSION, LEGACY_TEMPLATE_LEDGER_VERSION, PRE_PROXY_TEMPLATE_LEDGER_VERSION, TemplateHeader, TemplateStart, read_template_ledger, template_progress,
)
from aosr.gradient.template_rounds import TemplateContext
from aosr.gradient.template_segment import (
    TemplateResult, finished_result, resume_template_rounds, run_template_rounds,
)
from aosr.materials.construction_models import SurfaceMaterial
from aosr.materials.library import FixedEntry, OptimizableEntry
from aosr.materials.search_library import (
    PresetMaterial, SearchMaterials, fixed_candidates, read_snapshot_entries, with_user_materials, with_user_presets,
)
from aosr.gradient.boundary import (BoundaryKey, material_keys, named_materials, partition_identity, scheme_patch_keys, surface_ids)
from aosr.physics.fem_reduced_boundary import boundary_patches, PatchKeys
from aosr.physics.report_surfaces import RectangleZone
from aosr.physics.fem_reduced import ReducedModel, checked_axis
from aosr.physics.fem_reduced_check import PointHex, SurrogateSpotCheck, points_hex
from aosr.physics.fem_reduced_receipt import HoldoutRulers, Ruler, load_validated_reduced, receipt_path
from aosr.physics.fem_reduced_store import manifest_path
from aosr.physics.report_io import ReportInput, solver_inputs
from aosr.reporting.fem_reduced_scheme import scheme_points
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.reporting.validation import checked_inputs
from aosr.scoring.gradient_objective_jax import (
    GradientObjectiveConstants, gradient_objective_constants,
)
from aosr.scoring.gradient_objective_settings import GradientObjectiveSettings, gradient_objective_settings, legacy_gradient_objective_settings
from aosr.scoring.stage_one_low_frequency import axis_reason_not_measurable
from aosr.scoring.stage_one_score import StageOneScoreSettings
from aosr.scoring.six_category_jax import build_constants
from aosr.gradient.six_category_chain import six_category_chain_objective
from aosr.gradient.six_category_gpu import double_precision_score_arrays
from aosr.physics.fem_reduced_jax import reduced_constants
from aosr.gradient.folder_files import (
    LEDGER_FILE as LEDGER_FILE, PROJECT_FILE as PROJECT_FILE, RESULT_FILE as RESULT_FILE, STATUS_FILE as STATUS_FILE,
    STOP_FILE as STOP_FILE, TEMPLATES_FILE as TEMPLATES_FILE, GradientStatus as GradientStatus, State as State,
)


PROJECT_VERSION: Final = "aosr.material_gradient_project.v1"
# 專案外殼仍向後相容、CPU 位元組不變；六類由目標指紋換版，舊顯卡比較另在接續閘門明拒。
RESULT_VERSION: Final = "aosr.material_gradient_result.v6"
LOW_FREQUENCY_RESULT_VERSION: Final = "aosr.material_gradient_result.v4"
# 貼在原牆上（#990）的結果：每面帶安裝方式。剛背仍寫 RESULT_VERSION（安裝方式不改剛背的分數，位元組同加這一格之前）。
MOUNTED_RESULT_VERSION: Final = "aosr.material_gradient_result.v7"
LOW_FREQUENCY_MOUNTED_RESULT_VERSION: Final = "aosr.material_gradient_result.v5"
LEGACY_RESULT_VERSIONS: Final = (LOW_FREQUENCY_RESULT_VERSION, LOW_FREQUENCY_MOUNTED_RESULT_VERSION, "aosr.material_gradient_result.v3")
# 上一版（梯度目標還沒有大峰代理項）：報告准讀，終點與接續不收。
LEGACY_RESULT_VERSION: Final = "aosr.material_gradient_result.v3"
FREQUENCY_RULER: Final = "fem_reduced_holdout_frequency_db"
BAND_RULER: Final = "fem_reduced_holdout_band_db"
_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_SHA256 = r"^[0-9a-f]{64}$"


class GradientProject(BaseModel):
    """``project.json``：這一次搜尋的輸入與身分。路徑一律絕對路徑；雜湊是開始時量的，接續時逐格再比。

    ``objective_settings_fingerprint`` 是梯度目標設定的指紋（#561 壬一）；加代理之前開始的資料夾沒有這一格（讀成 None，
    報告照讀），接續時一律拒收。``mount`` 是安裝方式（#990）：剛背不寫（寫出的檔逐位等於加這一格之前）、缺這一格讀成剛背。
    ``max_steps`` 是開跑時另外指定的往下走步數上限（#1073）：沒指定不寫、照設定檔；指定了，``settings_fingerprint``
    記的是套上之後的指紋，接續、終點、報告一律經 :func:`project_settings` 套回去再比。
    ``materials_path``／``materials_sha256`` 是材料快照（#1082 段 2-3a）：使用者的可最佳化材料接成家族的搜尋才寫
    （資料夾裡的 ``materials.json`` 與它整份的 SHA-256）；沒接的不寫、寫出的檔逐位等於加這兩格之前。
    ``allowed_targets`` 是每區准用清單（#1082 段 2-3b，面 → 目標，照開跑時給的面名，沒展開）；清單裡的固定材料
    每一個都有一筆 ``fixed_candidate_checks``（替身留出驗證，通過才開得了）。沒有清單兩格都不寫（逐位等於加這兩格之前）。
    使用者固定材料存在材料快照 v2 裡。
    """

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
    objective_settings_fingerprint: str | None = Field(default=None, pattern=_SHA256)
    mount: Mount = Field(default=MOUNT_RIGID, exclude_if=lambda value: value == MOUNT_RIGID)
    fixed_surfaces: FixedSurfaces = Field(default_factory=dict, exclude_if=lambda value: not value)
    max_depths: MaxDepths = Field(default_factory=dict, exclude_if=lambda value: not value)
    device: Device = Field(default="cpu", exclude_if=lambda value: value == "cpu")
    precision_check: PrecisionComparison | LegacyPrecisionComparison | None = Field(default=None, exclude_if=lambda value: value is None)
    frequency_batch: int | None = Field(default=None, gt=0, strict=True, exclude_if=lambda value: value is None)
    max_steps: int | None = Field(default=None, gt=0, strict=True, exclude_if=lambda value: value is None)
    surrogate_physics_identity: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)
    operators_digest: str | None = Field(default=None, pattern=_SHA256, exclude_if=lambda value: value is None)
    surrogate_spot_check: SurrogateSpotCheck | None = Field(default=None, exclude_if=lambda value: value is None)
    materials_path: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)
    materials_sha256: str | None = Field(default=None, pattern=_SHA256, exclude_if=lambda value: value is None)
    allowed_targets: AllowedTargets = Field(default_factory=dict, exclude_if=lambda value: not value)
    fixed_candidate_checks: tuple[FixedCandidateCheck, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def _depth_not_fixed(self) -> Self:
        check_fixed_depth_conflicts(self.fixed_surfaces, self.max_depths)
        if [check.target for check in self.fixed_candidate_checks] != list(fixed_targets_of(self.allowed_targets)):
            raise ValueError("准用清單裡每一個固定材料都要有一筆通過的替身留出驗證（照目標字串排），不多不少")
        if (self.materials_path is None) != (self.materials_sha256 is None):
            raise ValueError("材料快照的路徑與 SHA-256 要一起記（使用者材料接成家族的搜尋），或都不記")
        fields = (self.surrogate_physics_identity, self.operators_digest, self.surrogate_spot_check)
        if any(value is not None for value in fields):
            spot = self.surrogate_spot_check
            if any(value is None for value in fields) or self.surrogate_physics_identity == self.physics_identity:
                raise ValueError("沿用記錄必須完整包含替身原身分、完整模型指紋與抽查結果，且原身分不同")
            if (spot is None or not spot.passed or spot.ruler_name != FREQUENCY_RULER
                    or not all(math.isfinite(value) for value in (spot.max_db, spot.ruler_value, spot.seconds))
                    or not 0 <= spot.max_db <= spot.ruler_value or spot.seconds <= 0):
                raise ValueError("沿用抽查記錄未通過或數字、尺不合法")
            try:
                checked_axis(spot.frequencies_hz)
            except ValueError as exc:
                raise ValueError("沿用抽查頻點不合法") from exc
        return self


class SurfaceSpec(BaseModel):
    """一面的規格：面代號、preset（不貼的面 None）、棉板厚度與空腔（公尺；不貼的面都是 None、沒有空腔層的空腔是 None）、
    材料摘要（``endpoint.design_digest``：有樣板的面是換 θ 之前的樣板構造的——貼在原牆上是貼牆構造的、不貼的面是保留的
    原牆材料的；報告只要厚度與空腔時是 None）、安裝方式（只有貼在原牆上的 v5 每面記 ``on_wall``；剛背的 v4 與更舊的沒有
這一格，是 None、寫出時也不寫）。"""

    model_config = _FROZEN
    surface_id: str
    preset_id: str | None
    thickness_m: float | None
    cavity_m: float | None
    material_digest: str | None
    mount: Mount | None = Field(default=None, exclude_if=lambda value: value is None)
    fixed_material_id: str | None = Field(default=None, exclude_if=lambda value: value is None)


class Components(BaseModel):
    """分項（聲道平均）：主位自己那條線的起伏、座位穩健性、大峰代理（v4 起；v3 的結果檔沒有，是 None）。"""

    model_config = _FROZEN
    own_line_ripple_rms_db: float
    seat_robustness_rms_db: float
    big_peak_proxy_db: float | None = None
    timbre_cost: float | None = Field(default=None, exclude_if=lambda value: value is None)
    listening_cost: float | None = Field(default=None, exclude_if=lambda value: value is None)
    reflections_cost: float | None = Field(default=None, exclude_if=lambda value: value is None)
    reverberation_cost: float | None = Field(default=None, exclude_if=lambda value: value is None)
    matching_cost: float | None = Field(default=None, exclude_if=lambda value: value is None)
    low_frequency_cost: float | None = Field(default=None, exclude_if=lambda value: value is None)
    objective_total: float | None = Field(default=None, exclude_if=lambda value: value is None)
    product_total: float | None = Field(default=None, exclude_if=lambda value: value is None)


class GradientResult(BaseModel):
    """``result.json``：交給終點的最後整房配置（換樣板帳本最後一列決定），與它出自哪兩本帳本（各自整份 SHA-256）。

    v3（#561 庚三）：``surfaces``／``variable_names``／``theta_hex``／``value``／``components`` 都是**最後配置**的；
    ``round_stop_reason``、``evaluations``、``rounds`` 是庚二候選確認迴圈的（求值次數照庚二帳本記的）；``stop_reason`` 是
    換樣板段的（整場的停止原因），``template_rounds``、``template_evaluations`` 是換樣板帳本記的。
    v4（#561 壬一）：``value`` 是新合分（第一階段兩分項合分＋權重 × 大峰代理），``screening_total`` 是舊合分，
    ``components`` 多了代理。v3 只准報告讀（:func:`parse_result` 的 ``legacy``），那時沒有舊合分與代理（v3 的值就是舊合分）。
    v5（#990）：貼在原牆上的搜尋寫這一版，每面多了安裝方式（``surfaces[].mount`` 都是 ``on_wall``），貼樣板的面摘要是
    貼牆構造的；剛背的搜尋照舊寫 v4（沒有安裝方式那一格就是剛背，終點照收）。v2 與更舊的檔不收。
    """

    model_config = _FROZEN
    format_version: Literal["aosr.material_gradient_result.v7", "aosr.material_gradient_result.v6", "aosr.material_gradient_result.v5", "aosr.material_gradient_result.v4",
                            "aosr.material_gradient_result.v3"]
    gradient_id: str
    surfaces: tuple[SurfaceSpec, ...]
    variable_names: tuple[str, ...]
    theta_hex: tuple[str, ...]
    value: float
    screening_total: float | None = None
    components: Components
    round_stop_reason: str
    stop_reason: str
    evaluations: int = Field(ge=0, strict=True)
    rounds: int = Field(ge=1, strict=True)
    template_rounds: int = Field(ge=1, strict=True)
    template_evaluations: int = Field(ge=0, strict=True)
    ledger_sha256: str = Field(pattern=_SHA256)
    template_ledger_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _version_matches_content(self) -> Self:
        proxied = self.format_version != LEGACY_RESULT_VERSION
        filled = (self.screening_total is not None, self.components.big_peak_proxy_db is not None)
        if any(item != proxied for item in filled):
            raise ValueError(f"{MOUNTED_RESULT_VERSION} 與 {RESULT_VERSION} 要帶舊合分與大峰代理；"
                             f"{LEGACY_RESULT_VERSION} 兩個都沒有")
        expected = MOUNT_ON_WALL if self.format_version in (MOUNTED_RESULT_VERSION, LOW_FREQUENCY_MOUNTED_RESULT_VERSION) else None
        if any(spec.mount != expected for spec in self.surfaces):
            raise ValueError(f"{MOUNTED_RESULT_VERSION} 每面要帶安裝方式 {MOUNT_ON_WALL}；"
                             f"{RESULT_VERSION}（剛背）與更舊的沒有這一格")
        if self.format_version in (RESULT_VERSION, MOUNTED_RESULT_VERSION):
            values = tuple(getattr(self.components, name) for name in (
                "timbre_cost", "listening_cost", "reflections_cost", "reverberation_cost", "matching_cost",
                "low_frequency_cost", "objective_total", "product_total"))
            if any(x is None for x in values) or self.components.objective_total != self.value:
                raise ValueError("六類結果缺類別或總分，或 objective_total 與 value 不同")
        return self

    @property
    def mount(self) -> Mount:
        """這份結果的安裝方式：v5 是貼在原牆上；v4（與報告准讀的 v3）沒有這一格，是剛背。"""
        return MOUNT_ON_WALL if self.format_version in (MOUNTED_RESULT_VERSION, LOW_FREQUENCY_MOUNTED_RESULT_VERSION) else MOUNT_RIGID


class SchemePoints(NamedTuple):
    """方案的喇叭與座位座標：名字 → 三個 ``float.hex()``。"""

    sources: dict[str, PointHex]
    receivers: dict[str, PointHex]


class GradientProblem(NamedTuple):
    """重建好的一次搜尋：每面設計（``project.presets`` 的起點配置）、搜尋設定、目標函式、庚二帳本標頭、
    方案原本六面的牆、換樣板要的固定材料（設計清單 → 目標函式的工廠）。"""

    designs: tuple[SurfaceDesign, ...]
    settings: MaterialGradientSettings
    objective: ChainObjective
    header: MaterialGradientHeader
    kept: dict[BoundaryKey, SurfaceMaterial]
    templates: TemplateContext


class Layout(Protocol):
    """帶配置與變數名的東西（庚二帳本標頭、換樣板帳本的列）。"""

    @property
    def surfaces(self) -> tuple[tuple[str, str | None], ...]: ...

    @property
    def variable_names(self) -> tuple[str, ...]: ...


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
                settings_fingerprint: str, physics_identity: str, objective_settings_fingerprint: str,
                mount: Mount = MOUNT_RIGID, fixed_surfaces: Mapping[str, str] | None = None,
                patch_keys: PatchKeys = boundary_patches(), max_depths: Mapping[str, float] | None = None,
                frequency_batch: int | None = None, device: Device = "cpu", materials_sha256: str | None = None,
                allowed_targets: Mapping[str, Sequence[str]] | None = None) -> str:
    """資料夾名：輸入（方案、替身、六面 preset、用途、設定、物理身分、梯度目標設定）正規化 JSON 的 SHA-256 前 32 位。

    同一組輸入一定落在同一個資料夾，兩個 ``start`` 同時跑同一組輸入會搶同一把資料夾鎖。梯度目標設定（#561 壬一：
    登記簿的代理三格與第一階段設定）算進來：只改那三格就是另一個資料夾，不會撞到加代理之前的那一個。
    安裝方式（#990）只在不是剛背時多一個 ``"mount"`` 鍵：剛背的資料夾名逐位等於加這一格之前，貼在原牆上落在另一個資料夾。
    材料快照（#1082 段 2-3a）只在使用者材料接成家族時多一個 ``"materials"`` 鍵（快照整份的 SHA-256）。
    """
    identity: dict[str, object] = {
        "scheme": scheme_sha256, "surrogate": surrogate_manifest_sha256, "presets": list(presets), "purpose": purpose,
        "settings": settings_fingerprint, "physics": physics_identity, "objective": objective_settings_fingerprint}
    if mount != MOUNT_RIGID:
        identity["mount"] = mount
    if fixed_surfaces:
        identity["fixed_surfaces"] = dict(sorted(fixed_surfaces.items()))
    if max_depths:
        identity["max_depths"] = dict(sorted(max_depths.items()))
    if device == "gpu":
        identity["device"] = device
    if frequency_batch is not None:
        identity["frequency_batch"] = frequency_batch
    if materials_sha256 is not None:
        identity["materials"] = materials_sha256
    if allowed_targets:
        identity["allowed_targets"] = {face: list(targets) for face, targets in sorted(allowed_targets.items())}
    if patches := partition_identity(patch_keys):
        identity["boundary_patches"] = patches
    content = json.dumps(identity,
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def stop_requested(folder: Path) -> StopCheck:
    """產品的停止判斷：資料夾裡有 ``stop`` 記號就是要停（每次求值之前查一次，只看檔在不在）。"""
    marker = folder / STOP_FILE
    return marker.exists


def read_project(folder: Path) -> GradientProject:
    return GradientProject.model_validate_json((folder / PROJECT_FILE).read_bytes())


def parse_result(content: bytes, *, legacy: bool = False) -> GradientResult:
    """讀 ``result.json`` 的內容：先看版本，不是這一版就拒收、訊息講明版本。v4（剛背）與 v5（貼在原牆上）都是這一版
    的正式結果。``legacy=True`` 只給報告用，准讀 v3（加大峰代理之前）；v3 交給終點或接續一律拒收（梯度目標加了大峰
    代理項，要重跑）。"""
    document = json.loads(content)
    version = document.get("format_version") if isinstance(document, dict) else None
    if version in LEGACY_RESULT_VERSIONS and not legacy:
        raise ValueError(f"{RESULT_FILE} 版本是 {version!r}（六類目標之前），{PROXY_RERUN_TEXT}；"
                         f"這一版只收 {RESULT_VERSION}（剛背）與 {MOUNTED_RESULT_VERSION}（貼在原牆上）")
    if version not in (RESULT_VERSION, MOUNTED_RESULT_VERSION, *LEGACY_RESULT_VERSIONS):
        raise ValueError(f"{RESULT_FILE} 版本是 {version!r}，這一版只收 {RESULT_VERSION}（剛背）與 "
                         f"{MOUNTED_RESULT_VERSION}（貼在原牆上）；跑 resume 會從帳本重建、覆寫成這一版")
    return GradientResult.model_validate_json(content)


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


def gated_surrogate(surrogate: Path, *, mesh_hash: str, physics_identity: str, points: SchemePoints,
                    current_operators_digest: str | None = None) -> ReducedModel:
    """准用閘門：兩把尺從登記簿讀，網格雜湊、物理身分、喇叭與座位座標都要跟驗證收據對上。"""
    model = load_validated_reduced(surrogate, rulers=holdout_rulers(), expected_mesh_hash=mesh_hash,
                                  expected_physics_identity=physics_identity, expected_source_points=points.sources,
                                  expected_receiver_points=points.receivers, current_operators_digest=current_operators_digest)
    require_matching_patches(model)
    return model


def check_surrogate_matches_scheme(model: ReducedModel, scheme: Scheme, points: SchemePoints) -> None:
    """替身的座位要照方案座位組的順序逐一相同、喇叭要是方案的那幾支（不然目標常數會取錯列）；
    替身建模用的空氣（密度、聲速）要逐位等於方案場景的（不然材料阻抗用方案的空氣、縮減方程係數用替身的空氣，
    一個目標混用兩組條件；網格不取密度，同網格、同座標、另一組空氣建的合法替身真的存在）；
    交給閘門比的座標要就是這份方案的座標（不然閘門比的是別的位置）。"""
    require_matching_patches(model, keys=scheme_patch_keys(scheme))
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
    require_matching_patches(model)
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


def search_objective_settings(purpose: str) -> GradientObjectiveSettings:
    """這一次搜尋用途的梯度目標設定（品質登記簿第一階段那一段：兩分項加代理三格）。"""
    return gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")), purpose)


def _check_same_points(given: SchemePoints, scheme: SchemePoints, what: str) -> None:
    """逐名逐座標比，第一個不同的就點名是哪一支喇叭或哪一個座位。"""
    for label, recorded, wanted in (("喇叭", given.sources, scheme.sources), ("座位", given.receivers, scheme.receivers)):
        for name in sorted(set(recorded) | set(wanted)):
            if recorded.get(name) != wanted.get(name):
                raise ValueError(f"{label} {name} 的座標：{what}是 {recorded.get(name)}，方案是 {wanted.get(name)}，拒收")


def project_materials(project: GradientProject) -> SearchMaterials:
    """這一跑的材料：讀 ``settings_path`` 與產品構造預設清單，有材料快照就把快照裡的使用者材料接成家族
    （讀資料夾裡那份、SHA-256 要等於記的，不讀網頁資料夾裡現在的材料庫）；步數上限還沒套。"""
    settings = load_material_gradient_settings(project.settings_path)
    entries: tuple[OptimizableEntry, ...] = ()
    if project.materials_path is not None and project.materials_sha256 is not None:
        entries = read_snapshot_entries(Path(project.materials_path), project.materials_sha256).materials
    return with_user_materials(settings, presets_table(), entries)


def project_allowed(project: GradientProject, names: Sequence[str]) -> dict[str, tuple[str, ...]] | None:
    """這一跑的准用清單展開到面片（``names`` 是設計清單的面代號）；沒有清單是 None（換樣板照舊）。"""
    return expanded_allowed_targets(project.allowed_targets, names) if project.allowed_targets else None


def project_settings(project: GradientProject) -> MaterialGradientSettings:
    """這一跑生效的搜尋設定：:func:`project_materials` 的設定，開跑時另外指定了步數上限就套上（#1073）；指紋由呼叫端比。"""
    settings = project_materials(project).settings
    return settings if project.max_steps is None else with_max_steps(settings, project.max_steps)


def project_presets(project: GradientProject) -> tuple[PresetMaterial, ...]:
    """這一跑的構造清單：產品構造預設，有材料快照就接上快照裡使用者材料生成的構造（不讀搜尋設定；快照 SHA-256 照核）；
    准用清單有固定材料就再接上每一個 ``fixed:<代號>`` 的那份材料（使用者的從快照、內建的從產品構造預設），
    而且每一份要跟開跑時做留出驗證的那份逐位同一份（``fixed_candidate_checks`` 記的摘要）：內建構造之後改了內容，
    舊的驗證不算數，拒收。"""
    from aosr.gradient.endpoint import material_digest

    table: tuple[PresetMaterial, ...]
    user_fixed: tuple[FixedEntry, ...]
    if project.materials_path is None or project.materials_sha256 is None:
        table, user_fixed = presets_table(), ()
    else:
        snapshot = read_snapshot_entries(Path(project.materials_path), project.materials_sha256)
        table, user_fixed = with_user_presets(presets_table(), snapshot.materials), snapshot.fixed
    targets = fixed_targets_of(project.allowed_targets)
    if not targets:
        return tuple(table)
    candidates = fixed_candidates(targets, table, user_fixed)
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    by_target = {candidate.id: candidate for candidate in candidates}
    for check in project.fixed_candidate_checks:
        if material_digest(adapter.validate_python(by_target[check.target].material)) != check.material_digest:
            raise ValueError(f"固定材料 {check.target} 的內容跟開跑時做留出驗證的那份不同（材料改過了），舊的驗證不算數，拒收")
    return (*table, *candidates)


def check_inputs(project: GradientProject, *, allow_legacy_objective: bool = False) -> tuple[Scheme, ReducedModel, MaterialGradientSettings]:
    """接續前逐格核輸入（父行程起子行程前、子行程重建目標前各核一次）：方案檔雜湊、方案座標跟 ``project.json`` 記的、
    替身過閘門（網格雜湊、物理身分、座標）、manifest 與收據摘要、設定指紋、替身對得上方案、替身頻率軸算得了
    這一次的計分。任何一格不同就拒收。"""
    if not allow_legacy_objective:
        from aosr.gradient.gpu_preflight import require_search_contract
        require_search_contract(project)
    scheme_path, surrogate = Path(project.scheme_path), Path(project.surrogate_path)
    if file_sha256(scheme_path) != project.scheme_sha256:
        raise ValueError(f"方案檔 {scheme_path} 跟開始時不同（scheme_sha256 變了），不接續")
    scheme = load_scheme(scheme_path)
    points = scheme_points_hex(scheme)
    _check_same_points(SchemePoints(project.source_points, project.receiver_points), points, "project.json 記的")
    model = gated_surrogate(surrogate, mesh_hash=project.mesh_hash,
                            physics_identity=project.surrogate_physics_identity or project.physics_identity,
                            points=points, current_operators_digest=project.operators_digest)
    if manifest_sha256(surrogate) != project.surrogate_manifest_sha256:
        raise ValueError("替身跟開始時不同（manifest 摘要變了），不接續")
    if file_sha256(receipt_path(surrogate)) != project.receipt_sha256:
        raise ValueError("替身的驗證收據跟開始時不同（SHA-256 變了），不接續")
    settings = project_settings(project)
    if settings.fingerprint != project.settings_fingerprint:
        raise ValueError("搜尋設定跟開始時不同（設定指紋變了），不接續")
    objective = search_objective_settings(project.purpose)
    previous = legacy_gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")), project.purpose)
    matches_legacy = allow_legacy_objective and project.objective_settings_fingerprint == previous.fingerprint
    if project.objective_settings_fingerprint != objective.fingerprint and not matches_legacy:
        recorded = "沒記（加大峰代理之前開始的）" if project.objective_settings_fingerprint is None else "跟現在不同"
        raise ValueError(f"project.json 的梯度目標設定指紋{recorded}：六類目標換版（梯度目標 v2／帳本 v6），"
                         f"舊資料夾只讀、要重跑；{PROXY_RERUN_TEXT}（或代理三格、第一階段設定改了）；"
                         f"{SCORED_RANGE_RERUN_TEXT}，不接續")
    check_surrogate_matches_scheme(model, scheme, points)
    check_surrogate_axis_covers_scoring(model, scheme, objective.stage_one)
    return scheme, model, settings


def scene_wall_materials(scheme: Scheme, *, capabilities: CapabilityTable,
                         directivity: DirectivityDefaults) -> dict[BoundaryKey, SurfaceMaterial]:
    """沿 ``export_operators`` 的 checked_inputs 第一對輸入取材料；分區直接取每片的 material。

    整面仍交 solver_inputs 取 materials_by_wall，不另轉固定阻抗；分區鍵與序由 boundary_patches 共用。
    """
    _, documents = checked_inputs(scheme, capabilities=capabilities, directivity=directivity)
    return _wall_materials_from_inputs(scheme, next(iter(documents.values()))[1])


def _wall_materials_from_inputs(scheme: Scheme, inputs: ReportInput) -> dict[BoundaryKey, SurfaceMaterial]:
    """共用已驗過的逐對輸入；材料與面片序維持 scene_wall_materials 的讀法。"""
    keys = scheme_patch_keys(scheme)
    if keys != boundary_patches():
        zones = inputs.surface_zones_by_wall or {}
        return {(wall, zone): next(z.material for z in zones[wall.wall_name()]
                                  if (isinstance(z, RectangleZone) and z.id == zone) or
                                  (not isinstance(z, RectangleZone) and not zone)) for wall, zone in keys}
    return cast(dict[BoundaryKey, SurfaceMaterial], dict(solver_inputs(inputs).materials_by_wall))


def project_wall_materials(project: GradientProject, scheme: Scheme) -> dict[BoundaryKey, SurfaceMaterial]:
    """:func:`scene_wall_materials`，能力表取 ``project.json`` 記的、指向性取產品預設（跟 ``resume`` 核物理身分同一組）。"""
    return scene_wall_materials(scheme, capabilities=load_capabilities(project.capabilities_path),
                                directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))


def search_objective_constants(project: GradientProject, scheme: Scheme,
                               model: ReducedModel) -> GradientObjectiveConstants:
    """這一次搜尋的目標常數：替身的頻率軸、``project.purpose`` 的梯度目標設定、方案座位組、替身的喇叭數。"""
    return gradient_objective_constants(np.asarray(model.frequencies_hz), search_objective_settings(project.purpose),
                                        scheme.receiver_set, len(model.source_names))


def _uncapped_designs_for[K: BoundaryKey](presets: Sequence[str | None], settings: MaterialGradientSettings, *,
                kept: Mapping[K, SurfaceMaterial], mount: Mount = MOUNT_RIGID,
                fixed_surfaces: Mapping[str, str] | None = None,
                table: Sequence[PresetMaterial] | None = None) -> tuple[SurfaceDesign, ...]:
    """有分區照共用面片序、整面照 ``Wall.all()`` 的設計；preset 不認識或不在家族清單裡就拒收。preset 是 None 的面一律帶 ``kept``
    那一面的材料（方案原本的牆，``scene_wall_materials``），是「不貼、保留原牆」，不是剛性；``mount`` 是貼在原牆上時，
    貼樣板的面背後接 ``kept`` 那一面。``table`` 是構造清單（不給是產品構造預設；有材料快照的搜尋給 :func:`project_presets`）。"""
    table = presets_table() if table is None else tuple(table)
    keys = material_keys(kept)
    if keys != boundary_patches():
        names = surface_ids(keys)
        if len(presets) != len(names):
            raise ValueError(f"面片清單要給 {len(names)} 個 preset id")
        return designs_with_fixed(tuple(zip(names, presets, strict=True)), settings, table,
                                  kept=named_materials(kept), mount=mount, fixed_surfaces=fixed_surfaces or {})
    whole_kept = cast(Mapping[Wall, SurfaceMaterial], kept)
    if fixed_surfaces:
        if len(presets) != len(Wall.all()):
            raise ValueError("六面牆要給六個 preset id")
        return designs_with_fixed(tuple(zip((wall.wall_name() for wall in Wall.all()), presets, strict=True)),
                                  settings, table, kept={w.wall_name(): m for w, m in whole_kept.items()},
                                  mount=mount, fixed_surfaces=fixed_surfaces)
    return wall_designs_from_presets(presets, settings, table,
                                     kept={wall.wall_name(): material for wall, material in whole_kept.items()}, mount=mount)


def designs_for[K: BoundaryKey](presets: Sequence[str | None], settings: MaterialGradientSettings, *,
                kept: Mapping[K, SurfaceMaterial], mount: Mount = MOUNT_RIGID,
                fixed_surfaces: Mapping[str, str] | None = None,
                max_depths: Mapping[str, float] | None = None,
                table: Sequence[PresetMaterial] | None = None) -> tuple[SurfaceDesign, ...]:
    """建立起跑設計，驗證每面上限不超過起跑家族上限，再套入；空設定沿用舊設計。``table`` 同 :func:`_uncapped_designs_for`。"""
    designs = _uncapped_designs_for(presets, settings, kept=kept, mount=mount, fixed_surfaces=fixed_surfaces, table=table)
    return designs_with_max_depths(designs, settings, max_depths or {})


def load_problem(project: GradientProject) -> GradientProblem:
    """照 ``project.json`` 重建目標；輸入先過 :func:`check_inputs`。要在雙精度已開的行程裡叫。

    空氣的密度與聲速取方案場景那一格（``export_operators`` 建完整模型、替身建模用的同一組），熱黏常數與多孔範圍
    走設定層的 ``construction_environment``；這裡不自己拿產品預設的空氣。換樣板的目標工廠用同一份替身、常數、空氣與範圍
    對任何六面設計重建 ``six_category_chain_objective``（起點配置直接交回這一份目標，不重編譯）。
    """
    scheme, model, settings = check_inputs(project)
    constants = search_objective_constants(project, scheme, model)
    channels = {channel.speaker_id: channel for channel in scheme.channel_group.channels}
    # 替身聲壓的第一軸照 model.source_names；幾何與評分必須同序，不能只核對喇叭數。
    group = scheme.channel_group.model_copy(update={"channels": tuple(channels[name] for name in model.source_names)})
    score_scheme = scheme.model_copy(update={"purpose": project.purpose, "channel_group": group})
    _, pairs = checked_inputs(score_scheme, capabilities=load_capabilities(project.capabilities_path),
                              directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    kept = _wall_materials_from_inputs(score_scheme, next(iter(pairs.values()))[1])
    require_supported_original_walls(kept.values())
    table = project_presets(project)
    designs = designs_for(project.presets, settings, kept=kept, mount=project.mount,
                          fixed_surfaces=project.fixed_surfaces, max_depths=project.max_depths, table=table)
    air, ranges = construction_environment(density_kg_m3=scheme.scene.density_kg_m3,
                                           sound_speed_m_s=scheme.scene.sound_speed_m_s)
    six = build_constants(score_scheme, pair_inputs={key: item[1] for key, item in pairs.items()},
                          quality_targets_path=config_path("quality_targets.toml"))
    single = project.device == 'gpu'
    device_constants = single_precision_constants(reduced_constants(model)) if single else None
    score_arrays = double_precision_score_arrays(six) if single else None
    objective = six_category_chain_objective(model, designs, constants, six, air=air, ranges=ranges,
        frequencies=model.frequencies_hz, frequency_batch=project.frequency_batch, single_precision=single,
        device_constants=device_constants, score_arrays=score_arrays)

    def objective_for(candidate: tuple[SurfaceDesign, ...]) -> ChainObjective:
        if candidate == designs:
            return objective
        return six_category_chain_objective(model, candidate, constants, six, air=air, ranges=ranges,
            frequencies=model.frequencies_hz, frequency_batch=project.frequency_batch, single_precision=single,
            device_constants=device_constants, score_arrays=score_arrays)

    from aosr.gradient.fixed_surfaces import expanded_fixed_surfaces

    names = tuple(d.surface_id for d in designs)
    context = TemplateContext(settings, table, named_materials(kept),
                              objective_for, project.mount, expanded_fixed_surfaces(project.fixed_surfaces, names),
                              max_depths=project.max_depths, allowed=project_allowed(project, names))
    return GradientProblem(designs, settings, objective, header_for(model, designs, settings,
                                                                    objective_constants=six), kept, context)


def check_user_families(materials: SearchMaterials, frequencies_hz: Sequence[float], *, density_kg_m3: float,
                        sound_speed_m_s: float) -> None:
    """開跑前把每個接進來的使用者家族都真的建一次（#1082 段 2-3a）：換樣板會換到每一個家族，不先擋會在搜尋半路才壞。

    每一筆：用它的樣板建一面設計（一層多孔、最多一層空腔、厚度界照合法厚度）、初值要在盒界內而且可行、
    樣板在這一場搜尋的頻點算得出垂直入射阻抗（空氣照方案場景、多孔範圍照產品設定，跟 :func:`load_problem` 同一組）。
    物理拒算（例如小松模型流阻出了手動試算範圍）原文照轉、前面點名是哪一筆材料的多孔層；範圍數字只住設定檔，這裡不重寫。
    """
    if not materials.entries:
        return
    from aosr.materials.construction import surface_impedance
    from aosr.materials.search_library import MaterialsRefused

    if jnp.asarray(0.0).dtype != jnp.float64:
        # 正式入口（命令列 main）開機就開雙精度；沒開時物理會拒算，那是行程設定的問題，不准講成材料參數的錯。
        raise RuntimeError("這個行程沒有開雙精度，算不了材料物理檢查；這是開機設定的問題，不是材料參數的問題")
    air, ranges = construction_environment(density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
    axis = jnp.asarray(tuple(frequencies_hz), dtype=jnp.float64)
    for entry in materials.entries:
        name = f"材料「{entry.label}」（{entry.id}）"
        family = materials.settings.family_by_id(entry.id)
        try:
            designs = designs_from_presets((("check", family.template_preset),), materials.settings, materials.presets)
            if not is_feasible(baseline_theta(designs), designs):
                raise ValueError("初值的厚度加空腔超過總深度上限")
        except ValueError as error:
            raise MaterialsRefused(f"{name}建不成搜尋樣板：{error}") from error
        template = designs[0].material
        if template is None:
            raise MaterialsRefused(f"{name}建不成搜尋樣板：樣板是空的")
        try:
            surface_impedance(template, axis, air=air, ranges=ranges)
        except ValueError as error:
            raise MaterialsRefused(f"{name}的多孔層（porous）參數物理拒算，搜尋不開：{error}") from error


def presets_table() -> tuple[ConstructionPreset, ...]:
    """產品的構造預設清單（``construction_presets.toml``）。"""
    return load_construction_presets(config_path("construction_presets.toml"))


# ---- 前段與整場 ----

def _pre_loop(problem: GradientProblem, path: Path, should_stop: StopCheck | None) -> FloatVector:
    """連續下降、取規格、重調三段，各自整段算完才寫成一批；回重調完的可行 θ（逐位等於帳本前段最後一列）。"""
    designs, settings, objective, header = problem.designs, problem.settings, problem.objective, problem.header
    descent = projected_descent(objective, baseline_theta(designs), design_bounds(designs), settings.descent,
                                limits=depth_limits(designs), should_stop=should_stop)
    if not descent.steps:
        raise Unestimable(descent.stop_reason)
    append_rows(path, header, descent_rows(descent, "descent", 0))
    snapped = snap_to_specs(descent.theta, designs)
    check_stop(should_stop)
    clock = time.perf_counter()
    (value, aux), _ = compiled_evaluator(objective)(jnp.asarray(snapped.theta))
    if not estimable(value, aux):
        raise Unestimable("取規格不可估：取規格點當不可行，不寫列")
    reading = objective_reading(aux)
    append_rows(path, header, (snap_row(snapped, float(value), reading.screening_total, reading.components,
                                        time.perf_counter() - clock, len(descent.steps)),))
    retune = retune_cavities(objective, snapped.theta, designs, settings, should_stop=should_stop)
    if not retune.steps:
        raise Unestimable("重調不可估：起點當不可行，不寫列")
    append_rows(path, header, descent_rows(retune, "retune", len(descent.steps) + 1))
    return retune.theta


def _fresh(problem: GradientProblem, path: Path, should_stop: StopCheck | None) -> LoopResult:
    theta = _pre_loop(problem, path, should_stop)
    return run_rounds(problem.objective, theta, problem.designs, problem.settings, path, problem.header,
                      should_stop=should_stop)


def run_search(folder: Path, *, should_stop: StopCheck | None) -> GradientResult:
    """開始：帳本獨佔建檔（已有帳本就丟錯）、先寫標頭，再跑前段與迴圈、換樣板段；跑完寫 ``result.json``。"""
    project = read_project(folder)
    problem = load_problem(project)
    ensure_start_precision(folder, project, problem)
    path = folder / LEDGER_FILE
    write_ledger(path, problem.header, ())
    _fresh(problem, path, should_stop)
    _template_segment(problem, folder, should_stop)
    return write_result(folder, problem)


def pre_loop_complete(rows: Sequence[MaterialGradientRow]) -> bool:
    """前段完整：已經有迴圈的列，或最後一列是帶停止原因的重調列（重調段收好了）。"""
    if any(row.round is not None for row in rows):
        return True
    return bool(rows) and rows[-1].phase == "retune" and rows[-1].stop_reason is not None


def ledger_closed(rows: Sequence[MaterialGradientRow]) -> bool:
    """庚二帳本收尾了：最後一列是帶停止原因與決定的決定列。"""
    return bool(rows) and rows[-1].phase == "round_decision" and rows[-1].stop_reason is not None and (
        rows[-1].decision is not None)


def resume_search(folder: Path, *, should_stop: StopCheck | None) -> GradientResult:
    """接續：看資料夾在哪一段。沒有帳本就從頭；標頭跟這一次重建的不同就拒收；庚二沒收尾就接庚二（前段不完整截到標頭
    重跑前段，否則 ``resume_rounds``），這時換樣板帳本在是矛盾、拒收；接著開或接換樣板段。跑完寫 ``result.json``（覆寫）。"""
    project = read_project(folder)
    problem = load_problem(project)
    ensure_start_precision(folder, project, problem)
    path, templates = folder / LEDGER_FILE, folder / TEMPLATES_FILE
    if templates.exists() and not (path.exists() and ledger_closed(read_ledger(path, expected_header=problem.header)[1])):
        raise ValueError(f"{TEMPLATES_FILE} 在，{LEDGER_FILE} 卻沒收尾：換樣板段只能接在庚二收尾之後，帳本與資料夾矛盾，不接續")
    _resume(problem, path, should_stop)
    _template_segment(problem, folder, should_stop)
    return write_result(folder, problem)


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


# ---- 換樣板段 ----

def template_start(header: MaterialGradientHeader, rows: Sequence[MaterialGradientRow]) -> TemplateStart:
    """換樣板的起點：庚二帳本收尾那一列的 θ、值、舊合分與分項，配置是庚二標頭的（＝``project.presets``）；沒收尾就拒收。"""
    if not ledger_closed(rows):
        raise ValueError(f"{LEDGER_FILE} 沒有以帶停止原因的決定列結尾，換樣板段不知道從哪個點起跑")
    last = rows[-1]
    return TemplateStart(header.surfaces, last.theta_hex, *last.scores())


def template_header_for[K: BoundaryKey](header: MaterialGradientHeader, settings: MaterialGradientSettings,
                        kept: Mapping[K, SurfaceMaterial], *, base_ledger_sha256: str,
                        allowed: Mapping[str, tuple[str, ...]] | None = None,
                        table: Sequence[PresetMaterial] | None = None) -> TemplateHeader:
    """換樣板帳本的標頭：替身身分與評分常數指紋抄庚二標頭、完整設定指紋（含 ``[templates]``）、庚二帳本整份摘要、
    起點配置（庚二標頭的六面）、六面方案原本那面牆的材料摘要；有准用清單（已展開到面片）就再記清單與清單裡每個
    固定材料的材料摘要（材料從 ``table`` 裡同名那一筆來）。"""
    targets = fixed_targets_of(allowed or {})
    by_id = {preset.id: preset for preset in (table or ())}
    missing = [target for target in targets if target not in by_id]
    if missing:
        raise ValueError(f"准用清單的固定材料 {missing} 不在這一場的構造清單裡")
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    candidate_digests = {target: material_digest(adapter.validate_python(by_id[target].material)) for target in targets}
    whole_kept = cast(Mapping[Wall, SurfaceMaterial], kept)
    digests = ({name: material_digest(material) for name, material in named_materials(kept).items()}
               if header.boundary_patches else
               {wall.wall_name(): material_digest(whole_kept[wall]) for wall in Wall.all()})
    return TemplateHeader(
        ledger_version=(TEMPLATE_LEDGER_VERSION if header.ledger_version == LEDGER_VERSION else
                        LEGACY_TEMPLATE_LEDGER_VERSION if header.ledger_version == LEGACY_LEDGER_VERSION else
                        PRE_PROXY_TEMPLATE_LEDGER_VERSION), input_hash=header.input_hash, mesh_hash=header.mesh_hash,
        physics_identity=header.physics_identity, settings_fingerprint=settings.full_fingerprint,
        objective_fingerprint=header.objective_fingerprint, base_ledger_sha256=base_ledger_sha256,
        initial_surfaces=header.surfaces, score_constants=header.score_constants,
        kept_digests=digests, mount=header.mount,
        fixed_surfaces=header.fixed_surfaces, max_depths=header.max_depths, boundary_patches=header.boundary_patches,
        allowed_targets=dict(allowed or {}), fixed_candidate_digests=candidate_digests)


def _template_segment(problem: GradientProblem, folder: Path, should_stop: StopCheck | None) -> TemplateResult:
    """庚二收尾之後：沒有 ``templates.jsonl`` 就開、有就接；起點與標頭由現在的庚二帳本算。"""
    path = folder / LEDGER_FILE
    _, rows = read_ledger(path, expected_header=problem.header)
    origin = template_start(problem.header, rows)
    header = template_header_for(problem.header, problem.settings, problem.kept, base_ledger_sha256=file_sha256(path),
                                 allowed=problem.templates.allowed, table=problem.templates.presets)
    templates = folder / TEMPLATES_FILE
    if templates.exists():
        return resume_template_rounds(problem.templates, templates, header, origin, should_stop=should_stop)
    return run_template_rounds(problem.templates, templates, header, origin, should_stop=should_stop)


# ---- 讀帳本：每面規格、進度、結果 ----

def surface_specs(layout: Layout, theta: Sequence[float], *, digests: Mapping[str, str | None] | None = None,
                  mount: Mount | None = None,
                  fixed_surfaces: Mapping[str, str] | None = None) -> tuple[SurfaceSpec, ...]:
    """θ 切回每面規格（版面照 ``layout`` 的配置與變數名：庚二帳本標頭，或換樣板帳本的一列）：每面的變數照「厚度、空腔」排
    （變數名 ``<面代號>.<樣板變數名>``），不貼的面沒有變數。``digests``（面代號 → 材料摘要）給了就照填、缺一面拒收；
    沒給（報告只要厚度與空腔）摘要是 None。``mount`` 每面照抄（貼在原牆上的 v5 結果檔要它；剛背與報告不給是 None）。"""
    if len(theta) != len(layout.variable_names):
        raise ValueError("θ 長度跟版面的變數數不同")
    specs = []
    for surface_id, preset_id in layout.surfaces:
        values = [value for name, value in zip(layout.variable_names, theta, strict=True)
                  if name.split(".", 1)[0] == surface_id]
        if (preset_id is None or is_fixed_target(preset_id)) and values:
            raise ValueError(f"不貼或整塊換成固定材料的面 {surface_id} 不該有變數")
        if digests is not None and surface_id not in digests:
            raise ValueError(f"材料摘要缺面 {surface_id}")
        if preset_id is not None and not is_fixed_target(preset_id) and not 1 <= len(values) <= 2:
            raise ValueError(f"面 {surface_id} 的變數要是厚度或厚度加空腔")
        specs.append(SurfaceSpec(surface_id=surface_id, preset_id=preset_id,
                                 thickness_m=values[0] if values else None,
                                 cavity_m=values[1] if len(values) > 1 else None,
                                 material_digest=None if digests is None else digests[surface_id], mount=mount,
                                 fixed_material_id=(fixed_surfaces or {}).get(surface_id)))
    return tuple(specs)


def ledger_progress(rows: Sequence[MaterialGradientRow]) -> tuple[int, int, float | None, str | None]:
    """帳本記的累計求值次數、做完幾輪、最後一列的值、迴圈收尾的停止原因（沒收尾是 None）。"""
    decisions = [row for row in rows if row.decision is not None]
    evaluations = decisions[-1].decision.total_evaluations if decisions and decisions[-1].decision else 0
    stop_reason = rows[-1].stop_reason if rows and rows[-1].phase == "round_decision" else None
    return evaluations, len(decisions), rows[-1].value if rows else None, stop_reason


def final_designs[K: BoundaryKey](surfaces: Sequence[tuple[str, str | None]], settings: MaterialGradientSettings,
                  kept: Mapping[K, SurfaceMaterial], *, mount: Mount = MOUNT_RIGID,
                  fixed_surfaces: Mapping[str, str] | None = None,
                  max_depths: Mapping[str, float] | None = None,
                  table: Sequence[PresetMaterial] | None = None) -> tuple[SurfaceDesign, ...]:
    """最後配置（每面 preset 或 None，照 ``Wall.all()``）→ 每面設計；不貼的面帶方案原本那面的牆；安裝方式照 ``mount``。
    ``table`` 同 :func:`_uncapped_designs_for`（有材料快照的搜尋要給 :func:`project_presets`）。

    每面上限已在起跑驗證，換家族後沿用換樣板的 min 規則，不再用新家族上限拒收原選項。
    """
    if tuple(surface_id for surface_id, _ in surfaces) != surface_ids(material_keys(kept)):
        raise ValueError(f"配置的面要照 Wall.all() 給六面，收到 {[surface_id for surface_id, _ in surfaces]}")
    designs = _uncapped_designs_for(tuple(preset for _, preset in surfaces), settings, kept=kept, mount=mount,
                                    fixed_surfaces=fixed_surfaces, table=table)
    return designs_with_max_depths(designs, settings, max_depths or {}, check_family_upper=False)


def result_from_ledger[K: BoundaryKey](folder: Path, settings: MaterialGradientSettings, kept: Mapping[K, SurfaceMaterial], *,
                       expected_header: MaterialGradientHeader | None = None,
                       expected_template_header: TemplateHeader | None = None, legacy: bool = False,
                       table: Sequence[PresetMaterial] | None = None) -> GradientResult:
    """從兩本帳重建整份結果，不寫檔：庚二帳本要收尾（給了 ``expected_header`` 標頭要逐格相等）；換樣板帳本要收尾、
    標頭記的庚二帳本摘要等於現在那本、起點是庚二收尾那一點（給了 ``expected_template_header`` 要逐格相等），
    最後配置與 θ 取它最後一列決定。每面材料摘要從最後配置重建的設計（``endpoint.design_digest``）來，安裝方式照庚二帳本標頭：
    六類剛背寫 v6、貼在原牆上寫 v7；legacy 唯讀重建仍回 v4／v5。
    寫 ``result.json`` 與終點核結果檔都走這一支（終點拿它跟檔裡那份整個比）。``table`` 同 :func:`final_designs`。"""
    path, templates = folder / LEDGER_FILE, folder / TEMPLATES_FILE
    header, rows = read_ledger(path, expected_header=expected_header, legacy=legacy)
    origin = template_start(header, rows)
    last = rows[-1]
    if not templates.exists():
        raise ValueError(f"沒有 {TEMPLATES_FILE}：換樣板段還沒開，沒有結果可寫")
    template_header, template_rows = read_template_ledger(templates, expected_header=expected_template_header,
                                                          settings=settings, start=origin, legacy=legacy)
    if template_header.base_ledger_sha256 != file_sha256(path):
        raise ValueError(f"{TEMPLATES_FILE} 標頭記的庚二帳本摘要跟現在的 {LEDGER_FILE} 不同，不寫結果")
    final = finished_result(template_rows)
    if final is None or last.decision is None or last.stop_reason is None:
        raise ValueError(f"{TEMPLATES_FILE} 沒有以帶停止原因的決定列結尾，沒有結果可寫")
    closing = template_rows[-1]
    value, screening, components = closing.scores()
    designs = final_designs(final.surfaces, settings, kept, mount=header.mount, fixed_surfaces=header.fixed_surfaces,
                            max_depths=header.max_depths, table=table)
    if design_names(designs) != final.variable_names:
        raise ValueError("換樣板帳本最後配置的變數名跟重建的設計不同")
    mounted = header.mount == MOUNT_ON_WALL
    version = (LOW_FREQUENCY_MOUNTED_RESULT_VERSION if mounted else LOW_FREQUENCY_RESULT_VERSION) if legacy else (
        MOUNTED_RESULT_VERSION if mounted else RESULT_VERSION)
    result_components = (Components(own_line_ripple_rms_db=components[0], seat_robustness_rms_db=components[1],
        big_peak_proxy_db=components[2]) if legacy else Components.model_validate(dict(zip(COMPONENT_NAMES, components, strict=True))))
    return GradientResult(
        format_version=cast(Literal["aosr.material_gradient_result.v7", "aosr.material_gradient_result.v6", "aosr.material_gradient_result.v5", "aosr.material_gradient_result.v4", "aosr.material_gradient_result.v3"], version), gradient_id=read_project(folder).gradient_id,
        surfaces=surface_specs(closing, closing.theta(),
                               digests={design.surface_id: design_digest(design) for design in designs},
                               mount=header.mount if mounted else None, fixed_surfaces=header.fixed_surfaces),
        variable_names=closing.variable_names, theta_hex=closing.theta_hex, value=value, screening_total=screening,
        components=result_components,
        round_stop_reason=last.stop_reason, stop_reason=final.stop_reason, evaluations=last.decision.total_evaluations,
        rounds=sum(1 for row in rows if row.decision is not None), template_rounds=final.rounds,
        template_evaluations=final.evaluations, ledger_sha256=file_sha256(path),
        template_ledger_sha256=file_sha256(templates))


def write_result(folder: Path, problem: GradientProblem) -> GradientResult:
    """:func:`result_from_ledger`（兩本帳的標頭都要等於這一次重建的）重建、原子寫成 ``result.json``。"""
    template_header = template_header_for(problem.header, problem.settings, problem.kept,
                                          base_ledger_sha256=file_sha256(folder / LEDGER_FILE),
                                          allowed=problem.templates.allowed, table=problem.templates.presets)
    result = result_from_ledger(folder, problem.settings, problem.kept, expected_header=problem.header,
                                expected_template_header=template_header, table=problem.templates.presets)
    write_json_atomic(folder / RESULT_FILE, result)
    return result


def template_ledger_progress(folder: Path) -> tuple[int, int, str | None]:
    """換樣板帳本記的累計求值次數、做完幾輪、收尾的停止原因；沒有帳本是 (0, 0, None)。"""
    path = folder / TEMPLATES_FILE
    if not path.exists():
        return 0, 0, None
    return template_progress(read_template_ledger(path)[1])
