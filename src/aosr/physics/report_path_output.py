"""報表輸出契約裡的逐路徑表三個模型：方向角、一列路徑、整張表的表頭（先從 report_io 搬出騰行數）。

:mod:`aosr.physics.report_io` 用同名轉出，外面照舊 ``report_io.PathRow`` 取用。搬出那一步欄位與說明一字未改、
匯出的兩份 schema 逐位元組不變；之後第二刀把 ``PathTableSection.includes_speaker_directivity`` 換成
``source_model_kind``，並在 ``PathRow`` 加 ``departure_off_axis_deg``。
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from aosr.geometry.shoebox import WALL_SEQUENCE_ORDER
from aosr.geometry.furniture import FaceDirection
from aosr.config.furniture_materials import FurnitureMaterialName
from aosr.physics.report_furniture import FiniteCoordinate
from aosr.physics.report_facts import (
    EMPTY_FOR_OMNIDIRECTIONAL,
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
        description=WALL_SEQUENCE_ORDER + "；家具一次反射固定為單格 furniture，代號、面與反射點由結構化欄位表示。",
        json_schema_extra=facts("牆名", "1", NO_BASIS_NAMES, NOT_MEASURED),
    )
    delay_s: float = Field(json_schema_extra=facts("時間", "s", "相對於聲源發聲時刻"))
    distance_m: float = Field(
        gt=POSITIVE_EXCLUSIVE_MINIMUM,
        json_schema_extra=facts("距離", "m", "鏡像聲源到接收點的直線距離，就是這條路徑走的總長；延遲＝距離÷聲速"),
    )
    direction_vector: tuple[float, float, float] = Field(
        json_schema_extra=facts("方向", "1", "房間座標中的到達方向：牆面列接收點指向鏡像源，家具列接收點指向反射點的單位向量")
    )
    direction_angles: PathDirectionAngles = Field(
        json_schema_extra=facts("方向角", "deg", "未折算聆聽軸的房間座標原始角度", NOT_MEASURED)
    )
    departure_off_axis_deg: float | None = Field(
        ge=0.0, le=180.0,
        json_schema_extra=facts(
            "離軸角", "deg", "聲源軸線與路徑出發方向的三維夾角；全向時無軸線", EMPTY_FOR_OMNIDIRECTIONAL
        )
    )
    relative_direct_energy: tuple[float, ...] = Field(
        json_schema_extra=facts(
            "逐路徑能量",
            "1",
            "牆面列為 (1−散射)^階數 × |路徑壓力|² ÷ |同頻點直達壓力|²；家具列為 "
            "|家具路徑壓力|² ÷ |同頻點直達壓力|²，不乘整房散射，聲壓已含有限尺寸修正 √(K₁K₂)；"
            "解析近似時分子分母都乘了 D；直達列為 1（0 dB 基準），不含同階內干涉。"
            "報表逐階能量是同階複數壓力相加後再"
            "取模平方；兩者刻意不同，不可拿來互相驗證",
        )
    )
    furniture_id: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None,
        json_schema_extra=facts("家具代號", "1", "輸入家具的原始代號；牆面列省略", NOT_MEASURED))
    furniture_face: FaceDirection | None = Field(default=None, exclude_if=lambda value: value is None,
        json_schema_extra=facts("家具面方向", "1", "房間座標方向 top、bottom、+x、-x、+y、-y；牆面列省略", NOT_MEASURED))
    reflection_point_m: tuple[FiniteCoordinate, FiniteCoordinate, FiniteCoordinate] | None = Field(
        default=None, exclude_if=lambda value: value is None,
        json_schema_extra=facts("反射點座標", "m", "房間角落為原點的 x、y、z 座標；牆面列省略", NOT_MEASURED))

    @model_validator(mode="after")
    def _furniture_row_shape(self) -> Self:
        values = (self.furniture_id, self.furniture_face, self.reflection_point_m)
        if "furniture" in self.wall_sequence:
            if self.order != 1 or self.wall_sequence != ("furniture",) or any(value is None for value in values):
                raise ValueError("家具列必須 order=1、wall_sequence=(furniture,)，且家具代號、面與反射點都有值")
            if self.furniture_id is not None and not self.furniture_id.strip():
                raise ValueError("家具列的 furniture_id 不可為空白")
        elif any(value is not None for value in values):
            raise ValueError("牆面列的家具代號、面與反射點必須是 None")
        return self


class PathFurnitureMaterial(FactsModel):
    """表頭每件輸入家具的材質估計與登記簿未知頻帶，不冒充本件實測。"""

    furniture_id: str = Field(min_length=1,
        json_schema_extra=facts("家具代號", "1", "輸入家具的原始代號", NOT_MEASURED))
    material: FurnitureMaterialName = Field(
        json_schema_extra=facts("材質類型", "1", "材質登記簿的類型代號；估計，非本件實測", NOT_MEASURED))
    unknown_bands_hz: tuple[float, ...] = Field(
        json_schema_extra=facts("未知頻帶中心頻率", "Hz",
            "材質登記簿 unknown_bands_hz；未知（計算時用相鄰頻帶延伸代算）", NOT_MEASURED))


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
    furniture_ids: tuple[str, ...] | None = Field(default=None, exclude_if=lambda value: value is None,
        json_schema_extra=facts("家具代號清單", "1", "本表使用的全部輸入家具，按代號排序；沒有家具時省略", NOT_MEASURED))
    furniture_model: Literal["single_bounce_finite_size_v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
        json_schema_extra=facts("家具模型", "1", "家具一次反射、有限尺寸鏡面修正；沒有家具時省略", NOT_MEASURED))
    blocked_wall_paths: tuple[tuple[str, ...], ...] | None = Field(default=None, exclude_if=lambda value: value is None,
        json_schema_extra=facts("被家具擋住的牆面路徑", "1", "移除路徑的 wall_sequence，保留原路徑列舉順序；沒有家具時省略", NOT_MEASURED))
    furniture_materials: tuple[PathFurnitureMaterial, ...] | None = Field(default=None, exclude_if=lambda value: value is None,
        json_schema_extra=facts("家具材質清單", "1", "每件輸入家具一筆，按代號排序；估計，非本件實測；沒有家具時省略", NOT_MEASURED))

    @model_validator(mode="after")
    def _furniture_matches_rows(self) -> Self:
        row_ids = {row.furniture_id for row in self.rows if row.furniture_id is not None}
        metadata = (self.furniture_model, self.blocked_wall_paths, self.furniture_materials)
        if self.furniture_ids is None:
            if row_ids or any(value is not None for value in metadata):
                raise ValueError("表頭沒有家具時，路徑列與表頭不准帶家具資料")
            return self
        ids = self.furniture_ids
        if not ids or tuple(sorted(set(ids))) != ids or any(not value.strip() for value in ids):
            raise ValueError("家具代號清單必須非空、唯一且按代號排序")
        if any(value is None for value in metadata):
            raise ValueError("有家具時，家具模型、被擋牆面路徑與家具材質必須齊全")
        if tuple(item.furniture_id for item in self.furniture_materials or ()) != ids or not row_ids <= set(ids):
            raise ValueError("表頭家具代號必須等於輸入材質清單，且涵蓋每條家具反射列")
        return self

    @model_validator(mode="after")
    def _departure_angles_match_model(self) -> Self:
        omnidirectional = self.source_model_kind == SourceModelKind.OMNIDIRECTIONAL
        if any((row.departure_off_axis_deg is None) != omnidirectional for row in self.rows):
            raise ValueError("path_table.rows.departure_off_axis_deg 與 source_model_kind 不符")
        return self
