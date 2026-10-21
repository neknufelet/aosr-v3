"""選席可序列化契約與搜尋軸大峰量法；不讀檔、不依賴報告。"""
from __future__ import annotations
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, model_validator
from aosr.reporting.stage_one_result import StageOneResult

FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
Role = Literal["top_n", "big_peak_control", "far_control"]


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

    @model_validator(mode="after")
    def _each_control_is_chosen_or_explained(self) -> Self:
        # 摘要是要給報告與接續讀回的：一席沒取到卻沒寫原因，報告就只能當機或亂印，讀回時就拒收。
        for role in ("big_peak_control", "far_control"):
            chosen = sum(role in seat.roles for seat in self.candidates)
            if chosen > 1 or (chosen == 1) == (role in self.missing):
                raise ValueError(f"{role} 必須剛好取到一席、或寫明沒取到的原因")
        if set(self.missing) - {"big_peak_control", "far_control"}:
            raise ValueError("missing 只收兩席對照的原因")
        return self


def big_peak(result: StageOneResult) -> float | None:
    """每支自己的最強峰，兩支取較差；任一支量不到就不產生數值。"""
    values = []
    for speaker in result.evaluation.speakers:
        payload = speaker.payload
        if payload is None or payload.strongest_peak_index is None:
            return None
        if payload.strongest_peak_index >= len(payload.features):
            raise ValueError(f"{speaker.speaker_id} 的最強峰索引超出峰谷清單")
        values.append(payload.features[payload.strongest_peak_index].depth_db)
    return max(values)
