"""一次搜尋的凍結總表；用途與原方案在建立搜尋資料夾時核對。"""

import hashlib
import json
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_serializer, model_serializer, model_validator

from aosr.search.layout_settings import LayoutSettings
from aosr.search.sampler import SamplerSettings

# 老闆 2026-10-22 先定每個候選 10 分鐘搜尋成本上限（不套細算與模態診斷）；設定沒寫這一格時照這個值，#902 之前開的搜尋也是。
DEFAULT_STAGE_ONE_CANDIDATE_CAP_S: Final = 600.0


class RefineSettings(BaseModel):
    """第五節第 5、6 條待定，要實跑後問老闆；必填，不自編預設。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    convergence_run: int = Field(ge=1, strict=True, description="第五節第 5、6 條待定，要實跑後問老闆；必填，不自編預設")
    budget: int = Field(ge=1, strict=True, description="第五節第 5、6 條待定，要實跑後問老闆；必填，不自編預設")


class StageOneRefineSettings(BaseModel):
    """第一階段保留名額與較遠對照的工程分界；開搜尋時固定。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    top_n: int = Field(ge=1, strict=True)
    far_control_min_front_distance_m: float = Field(gt=0,
        description="較遠對照的選席分界（聲學中心離前牆，公尺）；工程暫定，老闆 2026-10-21")
    # 面積對照（老闆 2026-10-23）：兩個比例一起設或一起不設；舊設定沒有這兩格，指紋與挑席照舊。
    area_control_max_spread: float | None = Field(default=None, gt=0, exclude_if=lambda value: value is None,
        description="前 N 面積跨度（最大減最小，再除以最小）不超過這個比例才開面積對照；工程暫定，老闆 2026-10-23")
    area_control_min_difference: float | None = Field(default=None, gt=0, lt=1, exclude_if=lambda value: value is None,
        description="比前 N 最小再小、或比最大再大至少這個比例才算面積明顯不同；工程暫定，老闆 2026-10-23")

    @model_validator(mode="after")
    def _area_pair(self) -> Self:
        if (self.area_control_max_spread is None) != (self.area_control_min_difference is None):
            raise ValueError("面積對照的兩個比例要一起設定")
        return self


class FeedbackSettings(BaseModel):
    """單位空間的回饋偏移量；暫行，要實跑後問老闆。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    offset: float = Field(gt=0, le=0.5, description="暫行，要實跑後問老闆；單位空間的偏移量，必填")


class SearchSettings(BaseModel):
    """批大小 K 跟工作行程數脫鉤；同一個 K 才保證單行程與多行程逐位相同。

    budget（預算）與 convergence_run（第一名連續未被超過的候選數）在設計紙
    第五節第 6 條標「待定」，要實跑後問老闆；這裡必填是為了不讓程式自己編一個值。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    purpose: str = Field(min_length=1)
    layout: LayoutSettings
    seed: int = Field(strict=True)
    n_startup_trials: int = Field(ge=1, strict=True)
    constant_liar: bool = Field(default=True, strict=True)
    batch_size: int = Field(ge=1, strict=True, description="同一個 K 才保證單行程與多行程逐位相同")
    max_workers: int = Field(ge=1, strict=True)
    budget: int = Field(ge=1, strict=True, description="第五節第 6 條待定，要實跑後問老闆；必填，不自編預設")
    convergence_run: int = Field(ge=1, strict=True, description="第五節第 6 條待定，要實跑後問老闆；必填，不自編預設")
    stage: Literal["stage_one_fem_only"] | None = Field(default=None, exclude_if=lambda value: value is None)
    stage_one_refine: StageOneRefineSettings | None = Field(default=None, exclude_if=lambda value: value is None)
    refine: RefineSettings | None = None
    feedback: FeedbackSettings | None = Field(default=None, exclude_if=lambda value: value is None)
    # 第一階段（老闆 2026-10-23）：搜尋階段的時間預算與單題搜尋成本上限住設定檔；舊設定沒有這兩格，指紋照舊。
    # 時間預算第二階段也收（老闆 2026-10-24，#558 房 109 受控試跑：搜尋三小時，用完要能接細算）；單題上限仍只准第一階段。
    search_time_budget_s: float | None = Field(default=None, gt=0, exclude_if=lambda value: value is None,
        description="搜尋階段的牆鐘預算（秒，接續也累計）；用完就在批界停、標因預算停止，不代表已收斂；不含細算、入選後檢查與附件")
    stage_one_candidate_cap_s: float | None = Field(default=None, gt=0, exclude_if=lambda value: value is None,
        description="每個候選的搜尋成本上限（秒）；沒寫照 600 秒（老闆 2026-10-22 的 10 分鐘），不套細算與入選後檢查")

    @model_validator(mode="after")
    def _nonblank_purpose(self) -> Self:
        if not self.purpose.strip():
            raise ValueError("purpose must not be blank")
        if self.stage_one_refine is not None and self.stage is None:
            raise ValueError("stage_one_refine 必須設定 stage（第一階段）")
        if self.layout.room_range is not None and self.stage != "stage_one_fem_only":
            raise ValueError("房間範圍只准第一階段（stage_one_fem_only）")
        if self.stage is None and self.stage_one_candidate_cap_s is not None:
            raise ValueError("單題搜尋成本上限只准第一階段（stage_one_fem_only）")
        return self

    @property
    def stage_one_search_cap_s(self) -> float:
        """第一階段搜尋每個候選的成本上限；設定沒寫照老闆 10-22 的 600 秒。"""
        return DEFAULT_STAGE_ONE_CANDIDATE_CAP_S if self.stage_one_candidate_cap_s is None else self.stage_one_candidate_cap_s

    def sampler_settings(self) -> SamplerSettings:
        """把三個取樣欄位交給既有取樣器轉接設定。"""
        return SamplerSettings(self.seed, self.n_startup_trials, self.constant_liar)

    @field_serializer("layout", mode="wrap", when_used="json")
    def _snapshot_layout(self, layout: LayoutSettings, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """JSON 快照省略未設夾角與房間外框，與 canonical 的指紋輸入一致。"""
        document: dict[str, object] = handler(layout)
        if layout.base_angle_deg is None:
            document.pop("base_angle_deg", None)
        if layout.room_range is None:
            document.pop("room_range", None)
        return document

    @model_serializer(mode="wrap", when_used="json")
    def _snapshot_refine(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """頂層未設細算就省略，JSON 快照不增加 null，保留舊設定形狀。"""
        document: dict[str, object] = handler(self)
        if self.stage is None:
            document.pop("stage", None)
        if self.refine is None:
            document.pop("refine", None)
        if self.feedback is None:
            document.pop("feedback", None)
        return document

    def canonical(self) -> dict[str, object]:
        """所有欄位的 JSON（交換資料格式）形狀，不混入摘要本身。

        未設夾角與房間外框時省略它們：舊搜尋資料夾存的是沒有這些格時算的指紋，
        多一格 null 會讓 SearchStore.open 判設定被改過；其餘未設欄位仍保留。
        """
        # 細算也是選填：未設時不進指紋，設了才釘住兩個必填停止設定。
        excluded: dict[str, bool | set[str]] = {}
        layout_excluded = {name for name in ("base_angle_deg", "room_range") if getattr(self.layout, name) is None}
        if layout_excluded:
            excluded["layout"] = layout_excluded
        if self.refine is None:
            excluded["refine"] = True
        if self.feedback is None:
            excluded["feedback"] = True
        return self.model_dump(mode="json", exclude=excluded)

    @property
    def fingerprint(self) -> str:
        """排序鍵與緊湊分隔的 SHA-256（內容摘要）；禁止非有限 JSON 數字。"""
        canonical = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
