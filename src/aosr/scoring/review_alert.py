"""排名列上的複核警戒（review alert）：提醒人工確認，不改候選資格。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from aosr.scoring.contract import QualityCategory


class ReviewAlert(BaseModel):
    """一個峰谷複核警戒（review alert）的來源身分、原始量與人話說明。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    category: QualityCategory
    speaker_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)
    kind: Literal["peak", "dip"]
    center_frequency_hz: Annotated[float, Field(gt=0.0)]
    depth_db: float
    width_octave: Annotated[float, Field(gt=0.0)] | None
    limit_db: Annotated[float, Field(gt=0.0)]
    narrower_than_axis: bool
    note: str = Field(min_length=1)
