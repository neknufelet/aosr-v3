"""選席可序列化契約與搜尋軸大峰量法；不讀檔、不依賴報告。"""
from __future__ import annotations
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
from aosr.reporting.stage_one_result import StageOneResult

FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
Role = Literal["top_n", "big_peak_control", "far_control", "area_control"]
CONTROLS = ("big_peak_control", "far_control", "area_control")
CONTROL_TEXT = {"big_peak_control": "大峰對照", "far_control": "較遠對照", "area_control": "面積對照"}


def role_text(role: Role, search_rank: int) -> str:
    """細算報告、入選清單與入選後檢查共用的角色字樣。"""
    return f"前 N（搜尋第 {search_rank} 名）" if role == "top_n" else CONTROL_TEXT[role]


def percent(value: float) -> str:
    return f"{value * 100:g}%"


class SelectionSeat(BaseModel):
    """同一試算共用所有角色；讀不回的前段名次保留位置與原因。"""

    model_config = FROZEN
    trial_number: int
    roles: tuple[Role, ...]
    search_rank: int
    search_score: float
    big_peak_db: float | None
    peak_reason: str | None
    params_m: dict[str, float]
    error: str | None


class StageOneSelection(BaseModel):
    """挑席的完整可重播輸入、門檻與空缺。"""

    model_config = FROZEN
    top_n: int
    actual_top_n: int
    far_control_min_front_distance_m: float
    big_peak_control_margin_db: float
    big_peak_threshold_db: float | None
    candidates: tuple[SelectionSeat, ...]
    missing: dict[str, str]
    unreadable: dict[int, str]
    # 舊設定沒有面積門檻：三格都不寫，舊挑席讀回、重挑都跟原本逐位相同。
    area_control_max_spread: float | None = Field(default=None, exclude_if=lambda value: value is None)
    area_control_min_difference: float | None = Field(default=None, exclude_if=lambda value: value is None)
    top_n_area_spread: float | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _each_control_is_chosen_or_explained(self) -> Self:
        # 摘要是要給報告與接續讀回的：一席沒取到卻沒寫原因，報告就只能當機或亂印，讀回時就拒收。
        if (self.area_control_max_spread is None) != (self.area_control_min_difference is None):
            raise ValueError("面積對照的兩個比例要一起記")
        for role in CONTROLS:
            chosen = sum(role in seat.roles for seat in self.candidates)
            if role == "area_control" and self.area_control_max_spread is None:
                if chosen or role in self.missing:
                    raise ValueError("沒有面積門檻就不開面積對照，也不寫原因")
            elif chosen > 1 or (chosen == 1) == (role in self.missing):
                raise ValueError(f"{role} 必須剛好取到一席、或寫明沒取到的原因")
        if set(self.missing) - set(CONTROLS):
            raise ValueError("missing 只收三席對照的原因")
        return self


def big_peak(result: StageOneResult) -> float | None:
    """每支自己的最強峰，兩支取較差；任一支量不到就不產生數值。

    先核完每一支的索引才判量不到：前一支量不到就提早回，後一支壞掉的索引會被放過。
    """
    payloads = [speaker.payload for speaker in result.evaluation.speakers]
    for speaker, payload in zip(result.evaluation.speakers, payloads, strict=True):
        if payload is not None and payload.strongest_peak_index is not None \
                and payload.strongest_peak_index >= len(payload.features):
            raise ValueError(f"{speaker.speaker_id} 的最強峰索引超出峰谷清單")
    values = []
    for payload in payloads:
        if payload is None or payload.strongest_peak_index is None:
            return None
        values.append(payload.features[payload.strongest_peak_index].depth_db)
    return max(values)
