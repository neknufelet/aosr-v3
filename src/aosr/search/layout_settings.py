"""第二階段子集的專案擺位設定；出處為搜尋設計紙第二節。"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Span(BaseModel):
    """有限、非退化的閉區間；第二節第 4 條的搜尋範圍與第 1 條的型號規格。"""

    model_config = FROZEN
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low >= self.high:
            raise ValueError("span requires low < high")
        return self


class Box(BaseModel):
    """房間世界座標的軸對齊盒；第二節第 1 條的禁區與桌面／支架可用區。"""

    model_config = FROZEN
    x: Span
    y: Span
    z: Span


class Cabinet(BaseModel):
    """第二節第 5 條：只供施工檢查，求解的聲源仍是聲學中心的點。

    width_m／depth_m／height_m 是專案箱子的寬、深、高。
    acoustic_center_behind_front_m 預設 0＝前面板，是第二節第 5 條主對話的預設。
    acoustic_center_above_bottom_m 的 None＝箱高的一半，是施工單上主對話定的預設（設計紙沒寫）。
    聲學中心的左右位置固定在箱子正中，箱底到聲學中心可以包含底／頂面。
    """

    model_config = FROZEN
    width_m: float = Field(gt=0.0, description="第二節第 5 條：專案輸入箱寬")
    depth_m: float = Field(gt=0.0, description="第二節第 5 條：專案輸入箱深")
    height_m: float = Field(gt=0.0, description="第二節第 5 條：專案輸入箱高")
    acoustic_center_behind_front_m: float = Field(
        default=0.0, ge=0.0, description="第二節第 5 條主對話預設：0＝前面板，向箱背為正",
    )
    acoustic_center_above_bottom_m: float | None = Field(
        default=None, ge=0.0, description="主對話預設（施工單定、設計紙沒寫）：None＝箱高的一半",
    )

    @model_validator(mode="after")
    def _center_in_cabinet(self) -> Self:
        if self.acoustic_center_behind_front_m >= self.depth_m:
            raise ValueError("acoustic center must be in front of the back panel")
        height = self.acoustic_center_above_bottom_m
        if height is not None and height > self.height_m:
            raise ValueError("acoustic center cannot be above the cabinet")
        return self


class LayoutSettings(BaseModel):
    """專案指定牆與中軸；未鎖定搜三個距離，鎖定只搜離前牆與間距。

    axis_offset_m 的正號是沿前牆往世界座標增加的方向：x 牆沿 +y、y 牆沿 +x。
    0 是第二節第 3 條的牆面正中預設；電腦不自行換牆或偏軸。
    高度沒有產品預設；wall_gap_m 的 0、keep_out 的空清單，以及兩個選填限制的
    None 代表專案未宣告該限制，不把標準建議距離冒充硬限制（第 1、6、7 條）。
    另有選填的 base_angle_deg，是每個專案自己的水平夾角規格，不代表空間感已評估。
    """

    model_config = FROZEN
    front_wall: Literal["x0", "xL", "y0", "yL"] = Field(description="第二節第 3 條：專案指定前牆")
    axis_offset_m: float = Field(default=0.0, description="第二節第 3 條：沿牆座標增加為正，預設正中")
    speaker_height_m: float = Field(gt=0.0, description="第二節第 4 條：專案輸入聲學中心高度，不搜尋")
    ear_height_m: float = Field(gt=0.0, description="第二節第 4 條：專案輸入耳高，不搜尋")
    front_distance_m: Span = Field(description="第二節第 4 條：聲學中心離前牆距離的搜尋範圍")
    spacing_m: Span = Field(description="第二節第 4 條：兩聲學中心間距的搜尋範圍")
    listening_distance_m: Span | None = Field(default=None, exclude_if=lambda value: value is None,
        description="喇叭連線到主位水平距離；未鎖定必填，鎖定時省略、由座位推出")
    seat_locked: bool = Field(default=False, strict=True, exclude_if=lambda value: value is False,
        description="鎖定原方案座位，只搜尋離前牆與間距")
    cabinet: Cabinet = Field(description="第二節第 5 條：只供幾何檢查的箱體")
    wall_gap_m: float = Field(
        default=0.0, ge=0.0, description="第二節第 1 條：箱體到四面牆（不含地板、天花板）的必要間隙",
    )
    keep_out: tuple[Box, ...] = Field(default=(), description="第二節第 1 條：箱體與全部座位的門／走道禁區")
    speaker_areas: tuple[Box, ...] | None = Field(
        default=None, description="第二節第 1 條：桌面／支架聲學中心可用區，None＝未限制",
    )
    listening_range_m: Span | None = Field(
        default=None, description="第二節第 1 條：型號適用的三維聆聽距離，None＝未限制",
    )
    base_angle_deg: Span | None = Field(
        default=None, description="主位看左右兩支喇叭聲學中心的水平夾角範圍（度）；"
        "專案規格、每個專案自己設，None＝未限制；不代表空間感已評估",
    )

    @model_validator(mode="after")
    def _base_angle_limits(self) -> Self:
        limits = self.base_angle_deg
        if limits is not None and not 0.0 < limits.low < limits.high < 180.0:
            raise ValueError("base angle requires 0 < low < high < 180 degrees")
        return self

    @model_validator(mode="after")
    def _search_ranges_positive(self) -> Self:
        if not self.seat_locked and self.listening_distance_m is None:
            raise ValueError("未鎖定座位時 listening_distance_m 必填")
        for span in (self.front_distance_m, self.spacing_m, self.listening_distance_m):
            if span is not None and span.low <= 0.0:
                raise ValueError("search distances require low > 0")
        if self.speaker_areas == ():
            raise ValueError("declared speaker areas must contain an available box")
        return self
