"""三路接合報表的最小輸入／輸出契約（票 #316）：欄位形狀與每一欄的四件事。

**為什麼住物理層。** 輸入那一半要用 :mod:`aosr.geometry.shoebox` 的 ``Room``／``Point``／``Wall``
把 JSON 收成求解層吃得下的形狀，而 ``config`` 那一層在 ``geometry`` 下面、拿不到它們
（規矩卡 ``layers-import-downward-only``）。

**每一欄記四件事。** 報表今天是一堆沒有單位標記的浮點數：能量是相對量還是絕對量、時間的
單位、這個指標這一跑算不算得出來，只有讀過程式與決策紙的人知道。這一份契約把四件事寫進
``Field(json_schema_extra=...)``，於是 ``model_json_schema()`` 匯出的正式 schema 檔也帶著它們，
而 :func:`quantity_table` 另外回一份「欄名 → 四件事」的對照表給人與考卷用：

* ``quantity`` 物理量（能量／時間／頻率／權重／計數／狀態／欄名）
* ``unit`` 單位（``1``＝無因次、``Hz``、``s``、``m``、``m/s``、``kg/m^3``、``Pa*s/m``、``m^3``）
* ``reference`` 參考基準（相對於什麼，或「絕對值」；**不是**單位。字串欄、計數欄、欄名欄
  與能力表上宣告的頻率範圍走同一族寫法「沒有基準（只是⋯⋯，不是量測值）」，中間那一段各自
  說自己是哪一種——四個字面在 ``NO_BASIS_TEXT``／``NO_BASIS_COUNT``／``NO_BASIS_NAMES``／
  ``NO_BASIS_RANGE``，這裡不抄全文。每一條回得到程式或決策紙）
* ``validity`` 有效狀態。**只有真的「估出來的數值欄」才寫「可估」**；不是數值的那些格子
  （文字、狀態旗標、計數、收據、欄名、以及 ``scene``／``capability``／``top``／``bands``／``points``
  這種本身不是量測值的容器欄）各自寫實話「不是估出來的量測值」。三族「空」另外分開寫：
  ``fem_energy`` 空＝這一帶沒有有限元素頻點，不必帶原因；``t20_s``／``t30_s`` 空＝值算不出來，
  必須帶原因；聲源軸線與離軸角空＝全向聲源沒有軸線，不必帶原因。說明與 validity 寫在同一句裡，不讓兩處各說各話）

**四件事的詞彙與界限住 :mod:`aosr.physics.report_facts`**（搬出去的直接原因是這一支頂到
寫法警衛的行數上限；分工是那一支放詞彙與界限、這一支放欄位形狀與驗證規則）（每一條的出處、界限常數
與寫進 schema 的小工具都在那一支；這裡只放欄位形狀與驗證規則）。
"""

from __future__ import annotations


import hashlib
import json
from pathlib import Path
from typing import Annotated, Final, NamedTuple, Self

from aosr.materials.construction_models import ConstructionNote

from pydantic import (
    WithJsonSchema,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    ModelWrapValidatorHandler,
    field_validator,
    model_validator,
)

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults, curve_within_allowed_range
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial
from aosr.physics.report_measured import MeasuredSeam
from aosr.physics.report_furniture import AbsoluteFurniture, normalized_furniture
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER, SUPPORTED_MIN_ORDER
from aosr.physics.report_path_output import (
    PathFurnitureMaterial as PathFurnitureMaterial,
    PathDirectionAngles as PathDirectionAngles,
    PathRow as PathRow,
    PathTableSection as PathTableSection,
)
from aosr.physics.report_facts import (
    FactsModel as _FactsModel,
    EMPTY_UNLESS,
    coordinate_object_facts,
    EMPTY_WHEN,
    ESTIMABLE,
    F_S_REFERENCE,
    FEM,
    FRACTION,
    INTERFERENCE,
    LATE,
    MIXED,
    MIXED_TOTAL,
    ORDER_K,
    REFLECTED,
    NO_BASIS_COUNT,
    NO_BASIS_NAMES,
    NO_BASIS_RANGE,
    NO_BASIS_TEXT,
    NOT_MEASURED,
    POSITIVE_EXCLUSIVE_MINIMUM,
    RELATIVE,
    SCATTERING_MAXIMUM,
    SCATTERING_MINIMUM,
    FieldFacts,
    facts,
    positive_facts,
    surface_zone_facts,
    wall_object_facts,
)
from aosr.physics.report_source import (
    AnalyticAxisymmetricInput,
    SourceCurveParameters,
    SourceModelInput,
    SourceModelSection,
    SourceModelSpec,
    source_model_spec,
)
from aosr.physics.source_directivity import speaker_axis
from aosr.physics.report_surfaces import (
    ZonesByWall, exclusive_wall_schema, normalized_surfaces,
)
from aosr.physics.report_validation import (
    FieldValueError, _checked_number, _checked_mapping, _number_is_finite,
    _table_of, _directivity_of, _wall_numbers, _reason_or_value, _rejected,
)


# 房與座標那幾格的鍵名：驗證器與匯出的格式檔都讀這兩份，不各寫一次。
ROOM_LENGTHS: Final[tuple[str, ...]] = ("Lx", "Ly", "Lz")
POINT_COORDINATES: Final[tuple[str, ...]] = ("x", "y", "z")



class ReportInput(_FactsModel):
    """三路接合報表輸入 JSON 的迷你契約（房、聲源模型與座標、接收點、介質、六面材料）。

    驗證規則照命令列原本那一套：六個牆名固定、數字要有限、阻抗為正、散射落在 ``[0,1]``；
    新舊欄位恰好給一個：舊六個固定阻抗永遠照收，新表面分區每面只收整面一區、
    材料為構造或固定阻抗；實測阻抗表尚未拍板，全固定新寫法正規化成舊寫法。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=exclusive_wall_schema)

    @model_validator(mode="wrap")
    @classmethod
    def _normalized_surfaces(cls, value: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        return handler(normalized_surfaces(value, json_shape=True))

    room_m: Annotated[
        Room,
        WithJsonSchema(
            coordinate_object_facts(
                "長度", "m", "房間角落為原點", names=ROOM_LENGTHS, positive=True
            )
        ),
    ] = Field(description="鞋盒房間三軸長度（公尺），三軸都必須大於零")
    source_m: Annotated[
        Point,
        WithJsonSchema(
            coordinate_object_facts(
                "長度", "m", "房間角落為原點", names=POINT_COORDINATES, positive=False
            )
        ),
    ] = Field(description="點聲源座標（公尺）")
    source_model: SourceModelInput = Field(
        description="必填聲源模型；解析近似的可調界限由呼叫端給的指向性登記簿驗證",
        json_schema_extra=facts("聲源模型", "1", NO_BASIS_TEXT, NOT_MEASURED),
    )
    receiver_m: Annotated[
        Point,
        WithJsonSchema(
            coordinate_object_facts(
                "長度", "m", "房間角落為原點", names=POINT_COORDINATES, positive=False
            )
        ),
    ] = Field(description="接收點座標（公尺）")
    sound_speed_m_s: float = Field(
        description="聲速（公尺／秒），必須大於零",
        json_schema_extra=positive_facts(
            "聲速", "m/s", "絕對值（帕·秒／公尺那條阻抗換算用）"
        ),
    )
    density_kg_m3: float = Field(
        description="空氣密度（公斤／立方公尺），必須大於零",
        json_schema_extra=positive_facts("密度", "kg/m^3", "絕對值（與聲速相乘得 ρc）"),
    )
    impedance_pa_s_per_m_by_wall: dict[str, float] | None = Field(
        default=None, exclude_if=lambda v: v is None,
        description="六面牆各一個與頻率無關的正實數阻抗（帕·秒／公尺）；六個牆名都必填",
        json_schema_extra=wall_object_facts(
            "表面阻抗", "Pa*s/m", "絕對值（不是無因次）", allow_zero=False
        ),
    )
    surface_zones_by_wall: ZonesByWall | None = Field(
        default=None, exclude_if=lambda v: v is None,
        description="六面表面分區清單；新舊恰好一個，每面整面一區，構造或固定阻抗；實測阻抗表不收",
        json_schema_extra=surface_zone_facts(),
    )
    scattering_by_wall: dict[str, float] | None = Field(
        default=None,
        description="六面牆各一個落在 [0,1] 的散射係數；整格可省略，給了就六個牆名都必填",
        json_schema_extra=wall_object_facts(
            "散射係數", "1", "相對於入射功率的比例（0 到 1）", allow_zero=True
        ),
    )
    reflection_order_k: int = Field(
        default=REFLECTION_ORDER_K,
        ge=SUPPORTED_MIN_ORDER,
        le=SUPPORTED_MAX_ORDER,
        description=(
            "幾何路與晚期混響的交接階數 K：K 階以內的鏡面留在鏡像法，散射掉的那一份與 K "
            "階以上交給晚期混響。整格可省略，省略時用產品設定 REFLECTION_ORDER_K"
        ),
        json_schema_extra=facts("反射階數", "1", ORDER_K, NOT_MEASURED),
    )
    low_frequency_axis: LowFrequencyAxis = Field(
        default=LowFrequencyAxis.SEARCH,
        description="低頻報表軸；search_octave_24 為舊搜尋軸（每八度 24 點，單方案省略時的預設），search_octave_48 為正式擺位搜尋軸（每八度 48 點），verification_linear_1hz 為驗證軸（20–300 Hz 每 1 Hz）",
        json_schema_extra=facts("頻率軸身分", "1", NO_BASIS_TEXT, NOT_MEASURED),
    )
    furniture: tuple[AbsoluteFurniture, ...] | None = Field(
        default=None, description="共用場景的絕對家具清單；不含主位從屬關係，空清單等同省略",
        json_schema_extra=facts("家具清單", "1", NO_BASIS_TEXT, NOT_MEASURED),
    )

    @field_validator("furniture")
    @classmethod
    def _normalized_furniture(cls, value: tuple[AbsoluteFurniture, ...] | None) -> tuple[AbsoluteFurniture, ...] | None:
        return normalized_furniture(value)

    @field_validator("source_model")
    @classmethod
    def _source_model_supported(cls, value: object, info: ValidationInfo) -> object:
        # 收不收只看能力表 source_directivity 那一節裡這個種類自己那一列：標 unsupported 就拒收、
        # 訊息帶那一列的 note；表上沒有這一列也拒收。能力表是唯一的開關，
        # 可調參數界線則由同一次輸入脈絡內的指向性登記簿驗。
        if isinstance(value, AnalyticAxisymmetricInput):
            rows = [item for item in _table_of(info).for_entry("source_directivity").capability
                    if item.materials == value.kind.value]
            if not rows:
                raise ValueError(f"source_model {value.kind.value} 不在能力表 source_directivity 入口裡")
            for item in rows:
                if item.status == "unsupported":
                    raise ValueError(f"source_model {value.kind.value} 不支援：{item.note}")
            try:
                curve_within_allowed_range(value.parameters.to_curve(), _directivity_of(info).allowed_range)
            except ValueError as exc:
                raise ValueError(f"source_model.parameters.{exc}") from exc
        return value

    @field_validator("room_m", "source_m", "receiver_m", mode="before")
    @classmethod
    def _objects_need_a_json_object(cls, value: object, info: object) -> object:
        """點與房只收 JSON 物件；訊息跟原本一樣指名是哪一格。

        ``Room``／``Point`` 是普通凍結 dataclass，Pydantic 會把 ``True`` 靜靜轉成 1.0，
        所以座標那一層要在這裡自己驗：布林不算數字、非有限值不收。
        """
        where = str(getattr(info, "field_name", "?"))
        if not isinstance(value, dict):
            raise ValueError(f"{where} 必須是 JSON 物件")
        coordinates = ROOM_LENGTHS if where == "room_m" else POINT_COORDINATES
        unknown = sorted(str(key) for key in value if str(key) not in coordinates)
        if unknown:
            raise ValueError(
                f"{where} 多了不認識的鍵 {unknown}；這一格只收 {list(coordinates)}。"
                "多打的那一個會被靜靜忽略，改的人會以為改生效了，所以這裡直接擋"
            )
        for coordinate in coordinates:
            if coordinate in value:
                _checked_number(value[coordinate], f"{where}.{coordinate}")
        return value

    @field_validator("sound_speed_m_s", "density_kg_m3", mode="before")
    @classmethod
    def _scalars_need_a_finite_number(cls, value: object, info: object) -> object:
        """聲速與密度要有限數字；布林不算數字。"""
        return _checked_number(value, str(getattr(info, "field_name", "?")))

    @field_validator("impedance_pa_s_per_m_by_wall", mode="before")
    @classmethod
    def _impedance_cells(cls, value: object, info: ValidationInfo) -> object:
        """阻抗六面：物件、六個固定牆名、複數或逐頻形式當場拒收。"""
        if value is None:
            return None
        return _wall_numbers(
            value,
            where="impedance_pa_s_per_m_by_wall",
            allow_zero=False,
            table=_table_of(info),
        )

    @field_validator("scattering_by_wall", mode="before")
    @classmethod
    def _scattering_cells(cls, value: object, info: ValidationInfo) -> object:
        """散射係數是另一種材料形式：逐面一個落在 [0,1] 的係數，可以等於 0。"""
        if value is None:
            return None
        return _wall_numbers(
            value,
            where="scattering_by_wall",
            allow_zero=True,
            table=_table_of(info),
        )

    @model_validator(mode="after")
    def _medium_must_be_physical(self) -> Self:
        """聲速與密度都必須是有限正數。"""
        for where, value in (
            ("sound_speed_m_s", self.sound_speed_m_s),
            ("density_kg_m3", self.density_kg_m3),
        ):
            if value <= POSITIVE_EXCLUSIVE_MINIMUM:
                raise FieldValueError(where, f"{where} 必須是有限正數")
        return self

    @model_validator(mode="after")
    def _room_must_be_a_box(self) -> Self:
        """房的三軸長都必須是有限正數。"""
        for name, length in (
            ("Lx", self.room_m.Lx),
            ("Ly", self.room_m.Ly),
            ("Lz", self.room_m.Lz),
        ):
            if not _number_is_finite(length) or length <= POSITIVE_EXCLUSIVE_MINIMUM:
                raise FieldValueError(f"room_m.{name}", f"room_m.{name} 必須是有限正數")
        return self

    @model_validator(mode="after")
    def _analytic_aim_is_inside_room_and_not_source(self) -> Self:
        if not isinstance(self.source_model, AnalyticAxisymmetricInput):
            return self
        aim = self.source_model.aim_m.as_tuple()
        lengths = (self.room_m.Lx, self.room_m.Ly, self.room_m.Lz)
        if any(not 0.0 <= coordinate <= length for coordinate, length in zip(aim, lengths, strict=True)):
            raise FieldValueError("source_model.aim_m", "source_model.aim_m 必須在房間的閉區間內")
        if self.source_model.aim_m == self.source_m:
            raise FieldValueError("source_model.aim_m", "source_model.aim_m 不可與 source_m 重合",
                                  pair_specific=True)
        return self


# 報表輸入的每一格只准屬於下面兩張清單之一；考卷守「兩張加起來等於全部欄位」，
# 所以替 ``ReportInput`` 新增一格的人一定得回答：它是整個場景共用的，還是每一份報表自己的。
SCENE_FINGERPRINT_FIELDS: Final[tuple[str, ...]] = (
    "room_m",
    "source_model",
    "sound_speed_m_s",
    "density_kg_m3",
    "impedance_pa_s_per_m_by_wall",
    "surface_zones_by_wall",
    "scattering_by_wall",
    "reflection_order_k",
    "low_frequency_axis",
    "furniture",
)


def scene_fingerprint(inputs: ReportInput) -> str:
    """回傳跨報表共用場景輸入的 SHA-256 十六進位指紋。

    空的牆面欄位在模型輸出時省略；全固定新寫法正規化為舊阻抗，既有指紋不變。
    納入 ``room_m``、``sound_speed_m_s``、``density_kg_m3``、
    ``impedance_pa_s_per_m_by_wall`` 或 ``surface_zones_by_wall``、``scattering_by_wall`` 與
    ``reflection_order_k``；也就是 :class:`ReportInput` 除座標外的每一格。
    ``low_frequency_axis`` 也納入；省略與明寫搜尋軸經模型預設值正規化後同指紋。
    ``source_model`` 整格也納入：模型種類、解析參數與共同對準點都是場景身分。
    不納入 ``source_m`` 與 ``receiver_m``：同一候選的各份報表可有不同聲源／接收點，
    兩座標由 :class:`SceneSection` 逐份另帶，不能拆散共享場景的身分。
    ``furniture``（換算後的絕對家具清單）也納入；沒有家具時只拿掉這一個鍵、其他欄位的
    ``null`` 照舊，所以沒有家具的場景指紋跟加家具以前逐位相同（#559 第四支）。
    """
    shared = inputs.model_dump(mode="json", include=set(SCENE_FINGERPRINT_FIELDS))
    if inputs.furniture is None:
        shared.pop("furniture", None)
    canonical = json.dumps(
        shared, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SceneSection(_FactsModel):
    """共享場景指紋、聲源模型，以及這一份報表自己的聲源與接收點座標。"""

    scene_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="排除聲源與接收點座標後，共享場景輸入的 SHA-256 十六進位指紋",
        json_schema_extra=facts("場景指紋", "1", NO_BASIS_TEXT, NOT_MEASURED),
    )
    source_m: Annotated[
        Point,
        WithJsonSchema(
            coordinate_object_facts(
                "長度", "m", "房間角落為原點", names=POINT_COORDINATES, positive=False
            )
        ),
    ] = Field(description="這一份報表的點聲源座標（公尺）")
    source_model: SourceModelSection = Field(
        json_schema_extra=facts("聲源模型", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )
    receiver_m: Annotated[
        Point,
        WithJsonSchema(
            coordinate_object_facts(
                "長度", "m", "房間角落為原點", names=POINT_COORDINATES, positive=False
            )
        ),
    ] = Field(description="這一份報表的接收點座標（公尺）")

    @model_validator(mode="after")
    def _axis_matches_source_and_aim(self) -> Self:
        model = self.source_model
        if isinstance(model, SourceModelSection) and model.aim_m is not None:
            if model.axis_unit_vector != speaker_axis(self.source_m, model.aim_m):
                raise ValueError("scene.source_model.axis_unit_vector 與 source_m、aim_m 不符")
        return self


class CapabilitySection(_FactsModel):
    """能力表那一條本人：範圍、輸出欄、狀態與收據（``capability_line`` 的四格）。"""

    frequency_hz: tuple[float, float] | tuple[()] = Field(
        description="這一條宣告的頻率範圍：空（沒查表）或剛好兩個端點",
        json_schema_extra=facts("頻率", "Hz", NO_BASIS_RANGE, NOT_MEASURED),
    )
    outputs: tuple[str, ...] = Field(
        description="這一條宣告的輸出欄名",
        json_schema_extra=facts("欄名", "1", NO_BASIS_NAMES, NOT_MEASURED),
    )
    status: str = Field(
        min_length=1,
        description="能力表三個狀態之一；unchecked 代表這一跑沒查表",
        json_schema_extra=facts("能力狀態", "1", NO_BASIS_TEXT, NOT_MEASURED),
    )
    evidence: tuple[str, ...] = Field(
        description="這一條指名的收據；沒查表時是空的",
        json_schema_extra=facts("收據", "1", NO_BASIS_TEXT, NOT_MEASURED),
    )


class TopFields(_FactsModel):
    """頂層那幾格；宣告順序不是印出來的順序（JSON 照字母排，文字表由 ``_top_table`` 自己排）。

    這三個列類別跟其他模型一樣是凍結的 Pydantic 模型，**不是** NamedTuple：JSON 出去
    的是有欄名的物件（``{"f_s_hz": …}``），不是位置陣列——前端不必數到第幾格才知道
    那一格是什麼，加一欄也不會靜靜改掉既有消費者讀到的意思。
    """

    f_s_hz: float = Field(json_schema_extra=facts("頻率", "Hz", F_S_REFERENCE))
    crossover_lower_hz: float = Field(
        json_schema_extra=facts("頻率", "Hz", "絕對值：交接區間下端的頻率")
    )
    crossover_upper_hz: float = Field(
        json_schema_extra=facts("頻率", "Hz", "絕對值：交接區間上端，等於有限元素上限")
    )
    capped_by_upper_limit: bool = Field(
        json_schema_extra=facts("狀態", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )
    reflection_order_k: int = Field(
        # 界線跟輸入那一格同一份（``physics.room_paths`` 那兩格）：抄成字面量的話，上限哪天
        # 動了輸入會跟、輸出不會，輸出契約就會收下一個輸入契約已經拒收的 K。
        ge=SUPPORTED_MIN_ORDER,
        le=SUPPORTED_MAX_ORDER,
        description="這一跑幾何路與晚期混響的交接階數 K；K 階以內的鏡面留在鏡像法",
        json_schema_extra=facts("反射階數", "1", ORDER_K, NOT_MEASURED),
    )
    eyring_t60_by_band_s: dict[str, float] = Field(
        json_schema_extra=facts("時間", "s", "絕對值：Eyring 公式、面積加權平均吸音率算出的秒數")
    )
    room_volume_m3: float = Field(
        json_schema_extra=facts("體積", "m^3", "絕對值：房三軸長相乘的立方公尺數")
    )
    schroeder_band_count: int = Field(
        ge=0,
        json_schema_extra=facts("計數", "1", NO_BASIS_COUNT, NOT_MEASURED),
    )
    low_frequency_axis: LowFrequencyAxis = Field(
        description="這份完整報表使用的低頻軸身分",
        json_schema_extra=facts("頻率軸身分", "1", NO_BASIS_TEXT, NOT_MEASURED),
    )


class BandRow(_FactsModel):
    """頻帶表一列。**宣告順序不等於印出來的順序**（這一張表是凍結的，順序不會自己漂，
    但三條路各有自己的順序）：

    * ``--format json`` 印的是 ``json.dumps(…, sort_keys=True)`` 的結果
      （``three_lane_report_cli.py``），所以鍵照**字母**排，不是宣告順序。
    * 人看的文字表由 ``_band_table`` 自己排，欄位集合與這一張**不同**：文字那一路沒有
      ``f_s_hz``、``capped_by_upper_limit`` 與兩個原因欄，而且印的是
      ``direct_energy_all_points`` 這一類欄名。
    * 兩族「空」在這一張表上分得開：``fem_energy`` 的 ``None`` 代表這個帶內一個有限元素
    頻點都沒有（``fem_point_count`` 是 0），意思是「這一帶沒有有限元素頻點」，**不必**
    帶原因；``t20_s``／``t30_s`` 的 ``None`` 是「值算不出來」，**必須**帶原因
    （``t20_unavailable_reason``／``t30_unavailable_reason``）。
    """

    center_frequency_hz: float = Field(json_schema_extra=facts("頻率", "Hz", "絕對值：八度帶中心頻率"))
    fem_energy: float | None = Field(
        json_schema_extra=facts("能量", "1", FEM, EMPTY_WHEN)
    )
    fem_point_count: int = Field(
        ge=0, json_schema_extra=facts("計數", "1", NO_BASIS_COUNT, NOT_MEASURED)
    )
    direct_energy: float = Field(json_schema_extra=facts("能量", "1", RELATIVE))
    reflected_energy: float = Field(json_schema_extra=facts("能量", "1", REFLECTED))
    interference_energy: float = Field(
        json_schema_extra=facts("能量", "1", INTERFERENCE)
    )
    late_energy: float = Field(json_schema_extra=facts("能量", "1", LATE))
    geometric_energy: float = Field(json_schema_extra=facts("能量", "1", MIXED))
    fem_contribution: float = Field(json_schema_extra=facts("能量", "1", FEM))
    geometric_contribution: float = Field(json_schema_extra=facts("能量", "1", MIXED))
    total_energy: float = Field(json_schema_extra=facts("能量", "1", MIXED_TOTAL))
    w_fem: float = Field(json_schema_extra=facts("權重", "1", FRACTION))
    w_geo: float = Field(json_schema_extra=facts("權重", "1", FRACTION))
    f_s_hz: float = Field(json_schema_extra=facts("頻率", "Hz", "絕對值：整個報表共用同一個交接頻率"))
    capped_by_upper_limit: bool = Field(
        json_schema_extra=facts("狀態", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )
    t20_s: float | None = Field(
        json_schema_extra=facts(
            "時間", "s", "絕對值：由同一條晚期衰減曲線的 −5～−25 dB 視窗擬合", EMPTY_UNLESS
        )
    )
    t20_unavailable_reason: str | None = Field(
        json_schema_extra=facts("不可估原因", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )
    t30_s: float | None = Field(
        json_schema_extra=facts(
            "時間", "s", "絕對值：由同一條晚期衰減曲線的 −5～−35 dB 視窗擬合", EMPTY_UNLESS
        )
    )
    t30_unavailable_reason: str | None = Field(
        json_schema_extra=facts("不可估原因", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )

    @model_validator(mode="after")
    def _empty_decays_carry_a_real_reason(self) -> Self:
        """值空要帶原因、有值不准帶原因——這條掛在**列**上，不只掛在整份輸出上。

        只掛在 :class:`ReportOutput` 的話，誰單獨造一列（前端的假資料、別支程式的中間值）
        就繞得過去；一份說不出為什麼的空值，讀的人分不出是「算不出來」還是「忘了填」。
        """
        _reason_or_value(self.t20_s, self.t20_unavailable_reason, where="t20_s")
        _reason_or_value(self.t30_s, self.t30_unavailable_reason, where="t30_s")
        return self


class PointRow(_FactsModel):
    """細軸逐點表一列。宣告順序**不等於**印出來的順序：``--format json`` 那一路照字母排鍵，
    文字那一路（``_point_table``）自己排。

    ``fem_energy`` 的 ``None`` 意思跟頻帶表那一欄同一族：這個頻點沒有有限元素值
    （``w_fem`` 是 0），不必帶原因——它不是「算不出來」，這一張表也沒有原因欄。
    """

    frequency_hz: float = Field(json_schema_extra=facts("頻率", "Hz", "絕對值：當次報表逐點軸上的頻點"))
    fem_energy: float | None = Field(
        json_schema_extra=facts("能量", "1", FEM, EMPTY_WHEN)
    )
    direct_energy: float = Field(json_schema_extra=facts("能量", "1", RELATIVE))
    reflected_energy: float = Field(json_schema_extra=facts("能量", "1", REFLECTED))
    interference_energy: float = Field(
        json_schema_extra=facts("能量", "1", INTERFERENCE)
    )
    late_energy: float = Field(json_schema_extra=facts("能量", "1", LATE))
    scattering: float = Field(
        json_schema_extra=facts("散射係數", "1", "相對於入射功率的比例（0 到 1）")
    )
    geometric_energy: float = Field(json_schema_extra=facts("能量", "1", MIXED))
    w_fem: float = Field(json_schema_extra=facts("權重", "1", FRACTION))
    w_geo: float = Field(json_schema_extra=facts("權重", "1", FRACTION))
    total_energy: float = Field(json_schema_extra=facts("能量", "1", MIXED_TOTAL))


class ReportOutput(_FactsModel):
    """三路接合報表的場景一節、三張表與 capability 那一行，收成一個可驗的結構。

    這一層只負責形狀與三條規則：頻帶列的中心頻率要遞增不重複、交接下端不准超過上端、
    以及不可估的欄位一定要帶原因（值空必須有原因、有值又不准給原因）。
    """

    scene: SceneSection = Field(
        description="這份報表的共享場景身分與逐份聲源／接收點座標",
        json_schema_extra=facts(
            "場景", "1", "見底下每一欄自己的基準", NOT_MEASURED
        ),
    )
    capability: CapabilitySection = Field(
        description="能力表那一條本人",
        json_schema_extra=facts(
            "能力", "1", "沒有基準（是能力表整條記錄，不是量測值）", NOT_MEASURED
        ),
    )
    top: TopFields = Field(
        description="頂層那幾格",
        json_schema_extra=facts(
            "頂層總量", "1", "見底下每一欄自己的基準", NOT_MEASURED
        ),
    )
    bands: tuple[BandRow, ...] = Field(
        description="頻帶表；七個八度帶各一列",
        json_schema_extra=facts("頻帶列", "1", "見底下每一欄自己的基準", NOT_MEASURED),
    )
    points: tuple[PointRow, ...] | None = Field(
        default=None,
        description="細軸逐點表；沒有 --points 時整塊省略",
        json_schema_extra=facts("逐點列", "1", "見底下每一欄自己的基準", NOT_MEASURED),
    )
    path_table: PathTableSection | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="逐路徑表；沒有要求時整塊省略",
        json_schema_extra=facts("路徑表", "1", "見底下每一欄自己的基準", NOT_MEASURED),
    )

    construction_notes_by_wall: dict[str, Annotated[tuple[ConstructionNote, ...], Field(min_length=1)]] | None = Field(
        default=None, exclude_if=lambda value: value is None,
        description="三路當次求值的逐牆構造註記；同面同層同代號合併外推頻點",
        json_schema_extra=facts("構造註記", "1", "沒有基準（材料模型適用範圍說明，不是量測值）", NOT_MEASURED) | {
            "propertyNames": {"enum": list(Wall.wall_names())}, "minProperties": 1,
        },
    )

    measured_seams_by_wall: dict[str, Annotated[tuple[MeasuredSeam, ...], Field(min_length=2, max_length=2)]] | None = Field(
        default=None, exclude_if=lambda value: value is None,
        description="實測牆表格兩端的接縫報告；實測與底構造各自保存，不存差值",
        json_schema_extra=facts("接縫資料", "1", "沒有基準（兩組端點資料的容器，不是量測值）", NOT_MEASURED) | {
            "propertyNames": {"enum": list(Wall.wall_names())}, "minProperties": 1},
    )

    @model_validator(mode="after")
    def _measured_seams_are_canonical(self) -> Self:
        seams = self.measured_seams_by_wall
        if seams is not None:
            if not seams or not set(seams) <= set(Wall.wall_names()):
                raise ValueError("接縫報告必須使用六面牆名，且不可為空物件")
            if any(rows[0].frequency_hz >= rows[-1].frequency_hz for rows in seams.values()):
                raise ValueError("接縫報告須依低、高端點排列")
        return self

    @model_validator(mode="after")
    def _construction_notes_are_canonical(self) -> Self:
        """只收已合併的真牆註記；缺省表示沒有註記，空物件不混作另一種形狀。"""
        notes = self.construction_notes_by_wall
        if notes is None:
            return self
        if not notes or not set(notes) <= {wall.wall_name() for wall in Wall.all()}:
            raise ValueError("構造註記必須使用六面牆名，且不可為空物件")
        for wall, items in notes.items():
            keys = [(note.layer_index, note.code) for note in items]
            if not items or len(set(keys)) != len(keys):
                raise ValueError(f"{wall} 構造註記不可為空或重複同層同代號")
            if any(note.frequencies_hz != tuple(sorted(set(note.frequencies_hz))) for note in items):
                raise ValueError(f"{wall} 外推頻點必須遞增且不重複")
        return self

    @model_validator(mode="after")
    def _decay_values_carry_reasons(self) -> Self:
        """T20／T30：值空必須有原因，有值又不准再給原因（兩個方向都報錯）。"""
        for index, band in enumerate(self.bands):
            _reason_or_value(
                band.t20_s, band.t20_unavailable_reason, where=f"bands[{index}].t20_s"
            )
            _reason_or_value(
                band.t30_s, band.t30_unavailable_reason, where=f"bands[{index}].t30_s"
            )
        return self

    @model_validator(mode="after")
    def _bands_are_the_frozen_octave_centers(self) -> Self:
        """頻帶列的中心頻率遞增、不重複、而且不是空的。"""
        centers = tuple(band.center_frequency_hz for band in self.bands)
        if not centers:
            raise ValueError("bands 不可為空：報表一定有七個八度帶")
        if tuple(sorted(centers)) != centers or len(set(centers)) != len(centers):
            raise ValueError("bands 的中心頻率必須遞增且不重複")
        return self

    @model_validator(mode="after")
    def _crossover_range_ascends(self) -> Self:
        """交接下端不准超過上端。"""
        if self.top.crossover_lower_hz > self.top.crossover_upper_hz:
            raise ValueError("crossover_lower_hz 不准大於 crossover_upper_hz")
        return self


def load_input_document(
    document: object, table: CapabilityTable, directivity: DirectivityDefaults,
) -> ReportInput:
    """把一份已經讀進來的 JSON 文件驗成 :class:`ReportInput`。

    能力表由呼叫端必給：拒收訊息的那一句 hint 要跟命令列 ``--capabilities`` 指定的
    **同一張**表算，這裡不再自己去載第二張（同一句話不准有兩個來源）。
    """
    context = {"table": table, "directivity": directivity}
    try:
        return ReportInput.model_validate(document, context=context)
    except ValidationError as exc:
        raise _rejected(exc) from exc


def load_input(path: Path, table: CapabilityTable, directivity: DirectivityDefaults) -> ReportInput:
    """從路徑讀一份輸入 JSON 並驗成 :class:`ReportInput`；能力表由呼叫端必給。"""
    with path.open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    return load_input_document(loaded, table, directivity)


class SolverInputs(NamedTuple):
    """已驗輸入與六面材料；所有物理下游共用這份材料。"""

    room: Room
    source: Point
    source_model: SourceModelSpec
    receiver: Point
    sound_speed_m_s: float
    density_kg_m3: float
    impedance_by_wall: dict[Wall, float] | None
    scattering_by_wall: dict[Wall, float] | None
    reflection_order_k: int
    low_frequency_axis: LowFrequencyAxis
    materials_by_wall: dict[Wall, SurfaceMaterial]
    furniture: tuple[AbsoluteFurniture, ...] | None = None


def solver_inputs(inputs: ReportInput) -> SolverInputs:
    """把 :class:`ReportInput` 攤成求解輸入，在此唯一建立固定阻抗材料；構造直接帶出已驗物件。

    牆名那兩格在這裡翻成 :class:`~aosr.geometry.shoebox.Wall`
    （``three_lane_report._wall_impedances`` 收的是 ``Mapping[Wall, …]``）。
    ``reflection_order_k`` 原樣帶過去：輸入檔沒給那一格時模型本來就填了產品設定
    ``REFLECTION_ORDER_K``，所以這裡不必再判一次「有沒有給」。
    ``low_frequency_axis`` 同理原樣帶過去（#435）：漏帶的話，從這裡算的報表會悄悄落回搜尋軸。
    ``furniture`` 原樣帶出；家具接觸界線由各報表入口另讀並明給物理核心。
    方案、搜尋與物理命令列各自守其入口的家具範圍。
    """
    zones, fixed = inputs.surface_zones_by_wall, inputs.impedance_pa_s_per_m_by_wall
    if zones is not None:
        materials = {wall: zones[wall.wall_name()][0].material for wall in Wall.all()}
    else:
        assert fixed is not None
        materials = {wall: ConstantImpedance(impedance_pa_s_m=fixed[wall.wall_name()]) for wall in Wall.all()}
    return SolverInputs(
        room=inputs.room_m,
        source=inputs.source_m,
        source_model=source_model_spec(inputs.source_model),
        receiver=inputs.receiver_m,
        sound_speed_m_s=inputs.sound_speed_m_s,
        density_kg_m3=inputs.density_kg_m3,
        impedance_by_wall=None if inputs.impedance_pa_s_per_m_by_wall is None else {
            wall: inputs.impedance_pa_s_per_m_by_wall[wall.wall_name()]
            for wall in Wall.all()
        },
        scattering_by_wall=(
            None
            if inputs.scattering_by_wall is None
            else {
                wall: inputs.scattering_by_wall[wall.wall_name()]
                for wall in Wall.all()
            }
        ),
        reflection_order_k=inputs.reflection_order_k,
        low_frequency_axis=inputs.low_frequency_axis,
        materials_by_wall=materials,
        furniture=inputs.furniture,
    )


# Pydantic 出廠的 JSON Schema 就是 draft 2020-12；宣告出來，讀 schema 檔的工具才知道
# 要按哪一版解（``$ref`` 旁邊那四件事是兄弟鍵，只有 2020-12 認得，見檔頭）。
JSON_SCHEMA_DRAFT: Final[str] = "https://json-schema.org/draft/2020-12/schema"


def _declared_schema(schema: dict[str, object]) -> dict[str, object]:
    """在匯出的 schema 最前面補上 draft 宣告；模型現算的內容一格不動。"""
    return {"$schema": JSON_SCHEMA_DRAFT, **schema}


def input_schema() -> dict[str, object]:
    """輸入模型現算出來的 JSON schema（``blueprint/schemas`` 那一份就是它）。"""
    return _declared_schema(ReportInput.model_json_schema())


def output_schema() -> dict[str, object]:
    """輸出模型現算出來的 JSON schema（``blueprint/schemas`` 那一份就是它）。"""
    return _declared_schema(ReportOutput.model_json_schema())


def quantity_table() -> dict[str, FieldFacts]:
    """回傳「欄名 → 四件事」的對照表（輸入、輸出與兩種列都攤平進來）。

    列的欄名用 ``bands.``／``points.`` 當前綴；``top`` 的欄位直接用 ``top.`` 前綴。
    兩種列與 ``top`` 跟其他模型一樣是 Pydantic 模型，所以走同一條 ``model_fields``
    的路——不再有第二種讀欄位的方式（NamedTuple 的 ``_fields`` 那一套）。
    """
    table: dict[str, FieldFacts] = {}
    for model in (ReportInput, CapabilitySection, ReportOutput):
        table.update(model.quantity_table())
    table.update(_prefixed_facts("bands", BandRow))
    table.update(_prefixed_facts("points", PointRow))
    table.update(_prefixed_facts("top", TopFields))
    table.update(_prefixed_facts("scene", SceneSection))
    table.update(_prefixed_facts("furniture", AbsoluteFurniture))
    table.update(_prefixed_facts("scene.source_model", SourceModelSection))
    table.update(_prefixed_facts("scene.source_model.parameters", SourceCurveParameters))
    table.update(_prefixed_facts("path_table", PathTableSection))
    table.update(_prefixed_facts("path_table.rows", PathRow))
    table.update(_prefixed_facts("path_table.furniture_materials", PathFurnitureMaterial))
    table.update(_prefixed_facts("path_table.rows.direction_angles", PathDirectionAngles))
    return table


def _prefixed_facts(prefix: str, model: type[_FactsModel]) -> dict[str, FieldFacts]:
    """把一個列模型的欄位四件事攤成 ``prefix.欄名``。"""
    return {
        f"{prefix}.{name}": facts for name, facts in model.quantity_table().items()
    }


# ── 重匯 schema 檔 ────────────────────────────────────────────────────────────
# ``blueprint/schemas/`` 那兩份檔是**產品**（前端與別的 repo 照它寫），模型改了沒重匯，
# 考卷 ``test_schema_files_match_the_models`` 就紅。這一區是那題紅掉之後的出口。
#
# 檔案放哪裡：要寫哪個目錄由呼叫端明示（`directory` 必給）。進版控的那一份位址
# （``blueprint/schemas``）由這裡算，不受 cwd 影響，考卷拿它對「版控那份」在哪。
_SCHEMA_DIR: Final[Path] = (
    Path(__file__).resolve().parents[3] / "blueprint" / "schemas"
)
SCHEMA_FILES: Final[dict[str, str]] = {
    "three_lane_report_input.schema.json": "input",
    "three_lane_report_output.schema.json": "output",
}


def regenerate_schema_files(directory: Path) -> tuple[Path, ...]:
    """把兩份 JSON schema 檔重寫成模型現算出來的樣子；回傳寫了哪幾個檔。

    ``directory`` **必給**（票 #316 第四刀非必修第 4 條）：先前省略時會寫進版控裡的
    ``blueprint/schemas/``，而那一條路是產品程式代寫版控檔、``tests-isolated-from-real-env``
    的靜態掃描看不見，誰在考卷裡喊一聲就能把契約洗掉。要覆寫版控那兩份，呼叫端得自己把
    那個路徑寫出來；不帶目錄的呼叫在型別那一關就進不來。

    寫法是逐位元組決定性的（同一份模型跑兩次得到同一個檔），縮排與鍵序跟原本手上那份一致；
    「跑兩次」這件事由考卷 ``test_regenerating_schema_files_reproduces_them_byte_for_byte``
    真的跑兩次比出來（第七刀非必修 9 補的，先前只跑一次、只跟版控那一份比）。

    **這一支不對人說話**（不印任何東西）：對人報告寫了哪幾個檔是命令列那一層的事
    （``three_lane_report_cli --regenerate-schemas <目錄>``，那是 style-guard 的輸出層），
    物理層這一支只把檔寫對。
    """
    target = directory
    target.mkdir(parents=True, exist_ok=True)
    computed: dict[str, dict[str, object]] = {
        "input": input_schema(),
        "output": output_schema(),
    }
    written: list[Path] = []
    for file_name, which in SCHEMA_FILES.items():
        path = target / file_name
        path.write_text(
            json.dumps(computed[which], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return tuple(written)
