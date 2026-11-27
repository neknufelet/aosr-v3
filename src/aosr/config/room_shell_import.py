"""房間殼匯入檢查的界線表（票 #1299）；十進位字串精確載入成浮點，不選預設檔。"""
from __future__ import annotations

from pathlib import Path
import re
import tomllib

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RoomShellImportSettings(BaseModel):
    """匯入檢查用的幾個界線；全部必填，缺一格就炸，不補預設值。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weld_m: float
    coplanar_angle_deg: float
    coplanar_offset_m: float
    min_triangle_area_m2: float
    parallel_angle_deg: float
    parallel_min_area_m2: float
    max_planes: int = Field(gt=0)
    max_file_bytes: int = Field(gt=0)

    @field_validator(
        "weld_m", "coplanar_angle_deg", "coplanar_offset_m", "min_triangle_area_m2",
        "parallel_angle_deg", "parallel_min_area_m2", mode="before",
    )
    @classmethod
    def _positive_decimal(cls, value: object) -> float:
        if not isinstance(value, str) or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None:
            raise ValueError("界線必須是十進位字串")
        number = float(value)
        if number <= 0:
            raise ValueError("界線必須大於零")
        return number


def load_room_shell_import_settings(path: str | Path) -> RoomShellImportSettings:
    """載入呼叫端指定的 TOML（設定表）。"""
    with Path(path).open("rb") as config_file:
        loaded = tomllib.load(config_file)
    return RoomShellImportSettings.model_validate(loaded)
