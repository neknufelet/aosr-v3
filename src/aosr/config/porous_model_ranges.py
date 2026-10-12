"""多孔模型原研究範圍的凍結設定；不把樣品範圍說成物理限制。

Komatsu（纖維多孔模型）的 f/σ 涵蓋界線只在此由原文四個數字推導，單位 m³/kg。
設定由呼叫端明給路徑載入，再注入材料計算；不在載入期搜尋環境或讀取設定。
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator


_PositiveNumber = Annotated[StrictFloat, Field(gt=0)]


class _RangeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class RangeProvenance(_RangeRecord):
    """原研究出處：完整書目、DOI（文獻識別碼）與各範圍所在段落。"""

    reference: str = Field(min_length=1, description="reference（完整書目）")
    doi: str = Field(min_length=1, description="doi（文獻識別碼）")
    flow_resistivity_source: str = Field(min_length=1, description="flow_resistivity_source（樣品流阻的表格出處）")
    frequency_source: str = Field(min_length=1, description="frequency_source（量測頻率的原文與段落）")
    material_class_source: str = Field(min_length=1, description="material_class_source（擬合材料類別的出處）")


class KomatsuRanges(_RangeRecord):
    """Komatsu（纖維多孔模型）原研究的樣品流阻、量測頻率與材料類別。"""

    sample_flow_resistivity_min_pa_s_m2: _PositiveNumber = Field(description="sample_flow_resistivity_min_pa_s_m2（樣品流阻下限，Pa·s/m²）")
    sample_flow_resistivity_max_pa_s_m2: _PositiveNumber = Field(description="sample_flow_resistivity_max_pa_s_m2（樣品流阻上限，Pa·s/m²）")
    measured_frequency_min_hz: _PositiveNumber = Field(description="measured_frequency_min_hz（量測頻率下限，Hz）")
    measured_frequency_max_hz: _PositiveNumber = Field(description="measured_frequency_max_hz（量測頻率上限，Hz）")
    fitted_material_class: Literal["fibrous"] = Field(description="fitted_material_class（擬合材料類別，fibrous 表示纖維類）")
    fitted_materials: tuple[Literal["glass wool", "rock wool"], ...] = Field(min_length=1, description="fitted_materials（擬合材料清單：glass wool 是玻璃棉，rock wool 是岩棉）")
    provenance: RangeProvenance = Field(description="provenance（原研究出處）")

    @model_validator(mode="after")
    def ordered_ranges(self) -> Self:
        """樣品與量測兩個區間都須下界小於上界。"""
        if self.sample_flow_resistivity_min_pa_s_m2 >= self.sample_flow_resistivity_max_pa_s_m2:
            raise ValueError("樣品流阻下界必須小於上界，請核對原研究表格")
        if self.measured_frequency_min_hz >= self.measured_frequency_max_hz:
            raise ValueError("量測頻率下界必須小於上界，請核對原研究段落")
        return self

    @property
    def f_over_sigma_bounds_m3_kg(self) -> tuple[float, float]:
        """f/σ（頻率除流阻）的量測涵蓋區間，m³/kg；只在此推導。"""
        return (self.measured_frequency_min_hz / self.sample_flow_resistivity_max_pa_s_m2,
                self.measured_frequency_max_hz / self.sample_flow_resistivity_min_pa_s_m2)


class PorousModelRanges(_RangeRecord):
    """多孔模型範圍登記簿；目前僅有 Komatsu（纖維多孔模型）資料。"""

    komatsu: KomatsuRanges = Field(description="komatsu（纖維多孔模型的原研究範圍）")


def load_porous_model_ranges(path: str | Path) -> PorousModelRanges:
    """載入呼叫端明給的 TOML（設定資料檔）並拒收壞欄位與顛倒上下界。"""
    with Path(path).open("rb") as file:
        return PorousModelRanges.model_validate(tomllib.load(file))
