"""早期反射門檻的型別（``[early_reflection]`` 區塊）。

**為什麼它住在第 2 層。** 上一代這個型別住在 zoning 那一層（第 8 層），
而第 2 層的 perceptual 那一支反過來去拿它——一條由下往上的線，分層卡一立就紅。
新家把它沉到第 2 層（它的家），perceptual 那一支改成從本層拿，那條線就消失。
決策紙 ``docs/decisions/engine-first-block-config-shape.md`` 的第二題選了這個做法。

**三個欄位、零預設。** 上一代的註解寫得很清楚：給預設就等於 15／10 有兩個家，
G8「同一個量只有一個主人」那條掃描就變成謊。所以三個欄位都是必填，也沒有 ``Field(default=…)``。

這一支只碰 pydantic，不開檔、不 import 上面的層。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EarlyReflectionConfig(BaseModel):
    """perceptual 那份設定檔的 ``[early_reflection]`` 區塊。

    三個欄位都是必填、沒有預設（上一代 spec D6）：這裡放一個預設，就等於
    15／10 那兩個數字多了一個家，SSOT 掃描從此說謊。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_ms: float
    required_attenuation_db: float
    check_band_hz: tuple[float, float]
