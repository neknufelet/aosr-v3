"""一次搜尋的凍結總表；用途與原方案在建立搜尋資料夾時核對。"""

import hashlib
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_serializer, model_serializer, model_validator

from aosr.search.layout_settings import LayoutSettings
from aosr.search.sampler import SamplerSettings


class RefineSettings(BaseModel):
    """第五節第 5、6 條待定，要實跑後問老闆；必填，不自編預設。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    convergence_run: int = Field(ge=1, strict=True, description="第五節第 5、6 條待定，要實跑後問老闆；必填，不自編預設")
    budget: int = Field(ge=1, strict=True, description="第五節第 5、6 條待定，要實跑後問老闆；必填，不自編預設")


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
    refine: RefineSettings | None = None
    feedback: FeedbackSettings | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _nonblank_purpose(self) -> Self:
        if not self.purpose.strip():
            raise ValueError("purpose must not be blank")
        return self

    def sampler_settings(self) -> SamplerSettings:
        """把三個取樣欄位交給既有取樣器轉接設定。"""
        return SamplerSettings(self.seed, self.n_startup_trials, self.constant_liar)

    @field_serializer("layout", mode="wrap", when_used="json")
    def _snapshot_layout(self, layout: LayoutSettings, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """存下的 JSON 設定也只省略未設夾角，使快照形狀與 canonical 的指紋輸入一致。"""
        document: dict[str, object] = handler(layout)
        if layout.base_angle_deg is None:
            document.pop("base_angle_deg", None)
        return document

    @model_serializer(mode="wrap", when_used="json")
    def _snapshot_refine(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """頂層未設細算就省略，JSON 快照不增加 null，保留舊設定形狀。"""
        document: dict[str, object] = handler(self)
        if self.refine is None:
            document.pop("refine", None)
        if self.feedback is None:
            document.pop("feedback", None)
        return document

    def canonical(self) -> dict[str, object]:
        """所有欄位的 JSON（交換資料格式）形狀，不混入摘要本身。

        未設夾角時只省略這一格：舊搜尋資料夾存的是沒有這一格時算的指紋，
        多一格 null 會讓 SearchStore.open 判設定被改過；其餘未設欄位仍保留。
        """
        # 細算也是選填：未設時不進指紋，設了才釘住兩個必填停止設定。
        excluded: dict[str, bool | set[str]] = {}
        if self.layout.base_angle_deg is None:
            excluded["layout"] = {"base_angle_deg"}
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
