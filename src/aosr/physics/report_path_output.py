"""報表輸出契約裡的逐路徑表三個模型：方向角、一列路徑、整張表的表頭（先從 report_io 搬出騰行數）。

:mod:`aosr.physics.report_io` 用同名轉出，外面照舊 ``report_io.PathRow`` 取用；欄位與說明一字未改，
騰行數那顆提交時匯出的兩份 schema 逐位元組不變；第二刀在此加聲源模型種類與離軸角。
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from aosr.geometry.shoebox import WALL_SEQUENCE_ORDER
from aosr.physics.report_facts import (
    NO_BASIS_NAMES,
    NOT_MEASURED,
    ORDER_K,
    POSITIVE_EXCLUSIVE_MINIMUM,
    FactsModel,
    facts,
)
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER, SUPPORTED_MIN_ORDER
from aosr.physics.report_source import SourceModelKind


class PathDirectionAngles(FactsModel):
    """未折算聆聽軸的原始幾何水平角與仰角。"""

    azimuth_deg: float = Field(
        json_schema_extra=facts("角度", "deg", "房間座標的 +x 軸起算")
    )
    elevation_deg: float = Field(
        json_schema_extra=facts("角度", "deg", "房間座標的水平面起算")
    )


class PathRow(FactsModel):
    """路徑表一列；方向保留房間座標原值，能量逐細軸點存放。

    能量那一欄的定義與「為什麼不能拿去跟報表的逐階能量互相驗證」寫在
    :func:`aosr.physics.report_path_table.build_path_table` 的說明裡。
    """

    order: int = Field(
        ge=0,
        json_schema_extra=facts(
            "反射階數",
            "1",
            "沒有基準（只是這條路徑自己的反射階數，直達路徑為 0，不是這一跑算到第幾階的設定）",
            NOT_MEASURED,
        ),
    )
    wall_sequence: tuple[str, ...] = Field(
        description=WALL_SEQUENCE_ORDER,
        json_schema_extra=facts("牆名", "1", NO_BASIS_NAMES, NOT_MEASURED),
    )
    delay_s: float = Field(json_schema_extra=facts("時間", "s", "相對於聲源發聲時刻"))
    distance_m: float = Field(
        gt=POSITIVE_EXCLUSIVE_MINIMUM,
        json_schema_extra=facts("距離", "m", "鏡像聲源到接收點的直線距離，就是這條路徑走的總長；延遲＝距離÷聲速"),
    )
    direction_vector: tuple[float, float, float] = Field(
        json_schema_extra=facts("方向", "1", "房間座標中的接收點指向鏡像源單位向量")
    )
    direction_angles: PathDirectionAngles = Field(
        json_schema_extra=facts("方向角", "deg", "未折算聆聽軸的房間座標原始角度", NOT_MEASURED)
    )
    departure_off_axis_deg: float | None = Field(
        json_schema_extra=facts("離軸角", "deg", "聲源軸線與路徑出發方向的三維夾角；全向時無軸線", NOT_MEASURED)
    )
    relative_direct_energy: tuple[float, ...] = Field(
        json_schema_extra=facts(
            "逐路徑能量",
            "1",
            "每條路徑各自的 (1−散射)^階數 × |路徑壓力|² ÷ |同頻點直達壓力|²；已乘散射留存，"
            "直達列為 1（0 dB 基準），不含同階內干涉。報表逐階能量是同階複數壓力相加後再"
            "取模平方；兩者刻意不同，不可拿來互相驗證",
        )
    )


class PathTableSection(FactsModel):
    """只在要求時出現的逐路徑表與當次計算表頭。"""

    reflection_order_k: int = Field(
        ge=SUPPORTED_MIN_ORDER,
        le=SUPPORTED_MAX_ORDER,
        json_schema_extra=facts("反射階數", "1", ORDER_K, NOT_MEASURED),
    )
    frequencies_hz: tuple[float, ...] = Field(
        json_schema_extra=facts("頻率", "Hz", "逐細軸點的絕對頻率")
    )
    scattering_coefficient: tuple[float, ...] = Field(
        json_schema_extra=facts("散射係數", "1", "當次逐細軸點的房間合成散射係數")
    )
    source_model_kind: SourceModelKind = Field(
        json_schema_extra=facts(
            "聲源模型種類",
            "1",
            "能量含不含指向性看 source_model_kind",
            NOT_MEASURED,
        )
    )
    rows: tuple[PathRow, ...] = Field(
        json_schema_extra=facts("路徑列", "1", "見底下每一欄自己的基準", NOT_MEASURED)
    )

    @model_validator(mode="after")
    def _departure_angles_match_model(self) -> Self:
        omnidirectional = self.source_model_kind == SourceModelKind.OMNIDIRECTIONAL
        if any((row.departure_off_axis_deg is None) != omnidirectional for row in self.rows):
            raise ValueError("path_table.rows.departure_off_axis_deg 與 source_model_kind 不符")
        return self
