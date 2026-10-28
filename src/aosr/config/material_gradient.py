"""第三階段牆面材料梯度搜尋設定的唯讀載入（#561）：連續下降、重調、候選確認迴圈、樣板家族。

路徑必給（``docs/decisions/config-loaders-keep-path-required.md``）；正式檔是
``config_path("material_gradient.toml")``。數字全是工程暫定，依據紙
``stage-three-gradient-objective-low-frequency-lane.md`` 代價段；這裡只守形狀與彼此一致，不給預設值；
數字欄位一律嚴格：整數欄只收整數，浮點欄只收寫成浮點的數（字串、整數都拒收）。
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictInt, StrictStr, model_validator


_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def _exact_float(value: object) -> object:
    """浮點欄只收浮點數：字串 "0.05"、整數 1、布林都拒收（pydantic 的嚴格浮點仍會收整數）。"""
    if type(value) is not float:
        raise ValueError(f"必須寫成浮點數（帶小數點），收到 {type(value).__name__} {value!r}")
    return value


ExactFloat = Annotated[float, BeforeValidator(_exact_float)]


class DescentSettings(BaseModel):
    """投影梯度下降＋回溯線搜尋（Armijo）的步數、步長與兩個停止值。"""

    model_config = _FROZEN

    max_steps: StrictInt = Field(ge=1)
    initial_step: ExactFloat = Field(gt=0.0)
    backtrack_ratio: ExactFloat = Field(gt=0.0, lt=1.0)
    armijo_constant: ExactFloat = Field(gt=0.0, lt=1.0)
    gradient_norm_stop: ExactFloat = Field(gt=0.0)
    min_step: ExactFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def _min_below_initial(self) -> Self:
        if self.min_step >= self.initial_step:
            raise ValueError("min_step 必須小於 initial_step")
        return self


class RetuneSettings(BaseModel):
    """取規格後空腔聯合重調的步數上限（線搜尋退到步長下限就提早停）。"""

    model_config = _FROZEN

    steps: StrictInt = Field(ge=1)


class CandidateSettings(BaseModel):
    """候選確認迴圈：每輪最多確認幾個候選、改善門檻（合分絕對差）、最多幾輪、整場最多求幾次目標。

    求值次數算次數不算秒數：值＋梯度算一次，線搜尋的每一個試探都算一次。
    """

    model_config = _FROZEN

    max_per_round: StrictInt = Field(ge=1)
    improvement_threshold: ExactFloat = Field(gt=0.0)
    max_rounds: StrictInt = Field(ge=1)
    max_evaluations: StrictInt = Field(ge=1)


class FamilySettings(BaseModel):
    """一個樣板家族：哪些 preset、合法棉板厚度、空腔範圍、每面總深度上限（公尺）。"""

    model_config = _FROZEN

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    presets: tuple[StrictStr, ...] = Field(min_length=1)
    thicknesses_m: tuple[ExactFloat, ...] = Field(min_length=1)
    cavity_range_m: tuple[ExactFloat, ExactFloat]
    max_total_depth_m: ExactFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.presets)) != len(self.presets):
            raise ValueError(f"家族 {self.id} 的 preset 重複")
        thicknesses = self.thicknesses_m
        if any(value <= 0.0 for value in thicknesses) or list(thicknesses) != sorted(set(thicknesses)):
            raise ValueError(f"家族 {self.id} 的合法厚度必須是正數、由薄到厚、不重複")
        low, high = self.cavity_range_m
        if not 0.0 <= low <= high:
            raise ValueError(f"家族 {self.id} 的空腔範圍必須 0 ≤ 下限 ≤ 上限")
        if thicknesses[0] > self.max_total_depth_m:
            raise ValueError(f"家族 {self.id} 最薄的合法厚度已超過總深度上限，取規格無處可退")
        return self


class MaterialGradientSettings(BaseModel):
    """四節：下降、重調、候選確認、家族；一個 preset 最多屬於一個家族。"""

    model_config = _FROZEN

    descent: DescentSettings
    retune: RetuneSettings
    candidates: CandidateSettings
    family: tuple[FamilySettings, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_families(self) -> Self:
        ids = [item.id for item in self.family]
        presets = [preset for item in self.family for preset in item.presets]
        if len(set(ids)) != len(ids) or len(set(presets)) != len(presets):
            raise ValueError("家族代號不可重複，一個 preset 最多屬於一個家族")
        return self

    def family_of(self, preset_id: str) -> FamilySettings | None:
        """這個 preset 屬於哪個家族；不在任何家族回 None。"""
        return next((item for item in self.family if preset_id in item.presets), None)

    @property
    def fingerprint(self) -> str:
        """驗證後內容的正規化 JSON SHA-256 十六進位指紋，寫進步驟帳本標頭。"""
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_material_gradient_settings(path: str | Path) -> MaterialGradientSettings:
    """讀 ``[descent]``、``[retune]``、``[candidates]``、``[[family]]``；缺節、多欄或數字不合就報錯。"""
    with Path(path).open("rb") as config_file:
        data = tomllib.load(config_file)
    return MaterialGradientSettings.model_validate(data)
