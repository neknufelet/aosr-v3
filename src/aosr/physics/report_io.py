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
  這種本身不是量測值的容器欄）各自寫實話「不是估出來的量測值」。兩族「空」另外分開寫：
  ``fem_energy`` 空＝這一帶沒有有限元素頻點，不必帶原因；``t20_s``／``t30_s`` 空＝值算不出來，
  必須帶原因。說明與 validity 寫在同一句裡，不讓兩處各說各話）

**四件事的詞彙與界限住 :mod:`aosr.physics.report_facts`**（搬出去的直接原因是這一支頂到
寫法警衛的行數上限；分工是那一支放詞彙與界限、這一支放欄位形狀與驗證規則）（每一條的出處、界限常數
與寫進 schema 的小工具都在那一支；這裡只放欄位形狀與驗證規則）。
"""

from __future__ import annotations


import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Final, Literal, NamedTuple, Self

from pydantic import (
    BaseModel,
    WithJsonSchema,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from aosr.config.capabilities import CapabilityTable
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER, SUPPORTED_MIN_ORDER
from aosr.physics.report_facts import (
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
    wall_object_facts,
)


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

# 房與座標那幾格的鍵名：驗證器與匯出的格式檔都讀這兩份，不各寫一次。
ROOM_LENGTHS: Final[tuple[str, ...]] = ("Lx", "Ly", "Lz")
POINT_COORDINATES: Final[tuple[str, ...]] = ("x", "y", "z")

# 複數與逐頻阻抗這兩條材料形式的能力表代號；拒收訊息從那兩條 unsupported 的 note 讀。
UNSUPPORTED_MATERIALS: Final[tuple[str, ...]] = (
    "complex_impedance_by_wall",
    "frequency_dependent_impedance",
)
CAPABILITY_ENTRY: Final[str] = "three_lane_report"


def _declared_schema_of(field: object) -> object:
    """一欄的 schema 宣告：多數欄寫在 ``json_schema_extra``，換掉 ``$ref`` 的那幾欄寫在
    :class:`WithJsonSchema` 裡（房與兩個座標點）。兩種都是同一份四件事，讀的地方只有這一個。
    """
    extra = getattr(field, "json_schema_extra", None)
    if isinstance(extra, dict):
        return extra
    for item in getattr(field, "metadata", ()):
        declared = getattr(item, "json_schema", None)
        if isinstance(declared, dict):
            return declared
    return None


class _FactsModel(BaseModel):
    """把「欄名 → 四件事」從 schema 額外欄位收成對照表的共用實作。"""

    model_config = FROZEN

    @classmethod
    def quantity_table(cls) -> dict[str, FieldFacts]:
        """回傳這一層每一欄的四件事。

        四件事寫在欄位宣告上（``json_schema_extra``），不是另外抄一份表；schema 檔與這張
        對照表因此永遠說著同一件事，而改了一邊忘了另一邊的那種漂不可能發生。
        """
        table: dict[str, FieldFacts] = {}
        for name, field in cls.model_fields.items():
            extra = _declared_schema_of(field)
            if not isinstance(extra, dict):
                raise ValueError(
                    f"{cls.__name__}.{name} 沒有宣告四件事（json_schema_extra 或 WithJsonSchema）"
                )
            table[name] = FieldFacts(
                quantity=str(extra["quantity"]),
                unit=str(extra["unit"]),
                reference=str(extra["reference"]),
                validity=str(extra["validity"]),
            )
        return table


class ReportInput(_FactsModel):
    """三路接合報表輸入 JSON 的迷你契約（房、聲源、接收點、介質、六面材料）。

    驗證規則照命令列原本那一套：六個牆名固定、數字要有限、阻抗為正、散射落在 ``[0,1]``；
    複數或逐頻阻抗（物件或陣列）當場拒收，訊息從能力表那幾條 unsupported 的 ``note`` 讀
    （:func:`unsupported_materials_hint`），不在這裡再抄一次。
    """

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
    impedance_pa_s_per_m_by_wall: dict[str, float] = Field(
        description="六面牆各一個與頻率無關的正實數阻抗（帕·秒／公尺）；六個牆名都必填",
        json_schema_extra=wall_object_facts(
            "表面阻抗", "Pa*s/m", "絕對值（不是無因次）", allow_zero=False
        ),
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
        description="低頻報表軸；省略時用正式 1/24 八度搜尋軸，驗證用 20–300 Hz 每 1 Hz",
        json_schema_extra=facts("頻率軸身分", "1", NO_BASIS_TEXT, NOT_MEASURED),
    )

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
                raise ValueError(f"{where} 必須是有限正數")
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
                raise ValueError(f"room_m.{name} 必須是有限正數")
        return self


# 報表輸入的每一格只准屬於下面兩張清單之一；考卷守「兩張加起來等於全部欄位」，
# 所以替 ``ReportInput`` 新增一格的人一定得回答：它是整個場景共用的，還是每一份報表自己的。
SCENE_FINGERPRINT_FIELDS: Final[tuple[str, ...]] = (
    "room_m",
    "sound_speed_m_s",
    "density_kg_m3",
    "impedance_pa_s_per_m_by_wall",
    "scattering_by_wall",
    "reflection_order_k",
    "low_frequency_axis",
)
PER_REPORT_INPUT_FIELDS: Final[tuple[str, ...]] = ("source_m", "receiver_m")


def scene_fingerprint(inputs: ReportInput) -> str:
    """回傳跨報表共用場景輸入的 SHA-256 十六進位指紋。

    納入 ``room_m``、``sound_speed_m_s``、``density_kg_m3``、
    ``impedance_pa_s_per_m_by_wall``、``scattering_by_wall`` 與
    ``reflection_order_k``；也就是 :class:`ReportInput` 除座標外的每一格。
    ``low_frequency_axis`` 也納入；省略與明寫搜尋軸經模型預設值正規化後同指紋。
    不納入 ``source_m`` 與 ``receiver_m``：同一候選的各份報表可有不同聲源／接收點，
    兩座標由 :class:`SceneSection` 逐份另帶，不能拆散共享場景的身分。
    """
    shared = inputs.model_dump(mode="json", include=set(SCENE_FINGERPRINT_FIELDS))
    canonical = json.dumps(
        shared, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SceneSection(_FactsModel):
    """共享場景指紋，以及這一份報表自己的聲源與接收點座標。"""

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
    receiver_m: Annotated[
        Point,
        WithJsonSchema(
            coordinate_object_facts(
                "長度", "m", "房間角落為原點", names=POINT_COORDINATES, positive=False
            )
        ),
    ] = Field(description="這一份報表的接收點座標（公尺）")


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


class PathDirectionAngles(_FactsModel):
    """未折算聆聽軸的原始幾何水平角與仰角。"""

    azimuth_deg: float = Field(
        json_schema_extra=facts("角度", "deg", "房間座標的 +x 軸起算")
    )
    elevation_deg: float = Field(
        json_schema_extra=facts("角度", "deg", "房間座標的水平面起算")
    )


class PathRow(_FactsModel):
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
        json_schema_extra=facts("牆名", "1", NO_BASIS_NAMES, NOT_MEASURED)
    )
    delay_s: float = Field(json_schema_extra=facts("時間", "s", "相對於聲源發聲時刻"))
    direction_vector: tuple[float, float, float] = Field(
        json_schema_extra=facts("方向", "1", "房間座標中的接收點指向鏡像源單位向量")
    )
    direction_angles: PathDirectionAngles = Field(
        json_schema_extra=facts("方向角", "deg", "未折算聆聽軸的房間座標原始角度", NOT_MEASURED)
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


class PathTableSection(_FactsModel):
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
    includes_speaker_directivity: Literal[False] = Field(
        json_schema_extra=facts(
            "狀態",
            "1",
            "這一版一律未含喇叭指向性；日後若要包含，必須改輸出契約",
            NOT_MEASURED,
        )
    )
    rows: tuple[PathRow, ...] = Field(
        json_schema_extra=facts("路徑列", "1", "見底下每一欄自己的基準", NOT_MEASURED)
    )


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


def _number_is_finite(value: object) -> bool:
    """有限數字；布林不算數字（布林是 int 的子類，會靜靜變成 1.0）。"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def _checked_number(value: object, where: str) -> float:
    """照命令列原本的寫法驗一格數字，訊息一字不變。"""
    if not _number_is_finite(value):
        raise ValueError(f"{where} 必須是有限數字")
    assert isinstance(value, int | float)
    return float(value)


class WiringError(RuntimeError):
    """呼叫端把這一層接錯了（不是使用者的輸入不好）。

    這一種錯誤不准被 :func:`load_input_document` 包成「輸入不合 ReportInput」——
    那會把「程式接線壞了」報成「你的輸入不好」，看訊息的人會去改 JSON 而不是改呼叫端。
    """


def _table_of(info: ValidationInfo) -> CapabilityTable:
    """從驗證脈絡拿這一次要用的能力表；沒帶就大聲炸，不偷偷載第二張。

    :func:`load_input_document` 在 ``model_validate(context=…)`` 裡把表放進來，
    所以命令列 ``--capabilities`` 指定的那一張就是拒收訊息讀的那一張。

    拿不到表是**接線錯誤**不是壞輸入，所以丟 :class:`WiringError`
    （``load_input_document`` 只把 :class:`~pydantic.ValidationError` 收成人話，
    這一種照原樣往上冒）。
    """
    table = (info.context or {}).get("table")
    if not isinstance(table, CapabilityTable):
        raise WiringError(
            "輸入驗證沒有帶著能力表：load_input_document 要收呼叫端指定的那一張"
        )
    return table


def _checked_mapping(value: object, where: str) -> dict[str, object]:
    """照命令列原本的寫法驗一層 JSON 物件，訊息一字不變。"""
    if not isinstance(value, dict):
        raise ValueError(f"{where} 必須是 JSON 物件")
    return {str(key): item for key, item in value.items()}


def _wall_numbers(
    value: object,
    *,
    where: str,
    allow_zero: bool,
    table: CapabilityTable,
) -> dict[str, float]:
    """六面牆各一格數字；阻抗要正、散射落在 [0,1]，複數或逐頻形式當場拒收。

    拒收訊息的那一句 hint 只從呼叫端傳進來的**那一張**表算（命令列走
    ``--capabilities``，考卷走它自己指定的檔）——同一個地方算一次，不在這裡再拼第二份。
    """
    fields = _checked_mapping(value, where)
    names = [wall.wall_name() for wall in Wall.all()]
    unknown = sorted(key for key in fields if key not in names)
    if unknown:
        raise ValueError(
            f"{where} 多了不認識的牆名 {unknown}；這一格只收 {names} 這六面。"
            "多打的那一個被靜靜忽略，改的人會以為改生效了，所以這裡直接擋"
        )
    result: dict[str, float] = {}
    for wall in Wall.all():
        name = wall.wall_name()
        cell_where = f"{where}.{name}"
        cell = fields.get(name)
        if isinstance(cell, dict | list | tuple):
            hint = (
                unsupported_materials_hint(table)
                if where == "impedance_pa_s_per_m_by_wall"
                else "散射係數只收一個實數"
            )
            raise ValueError(f"{cell_where}：{hint}")
        number = _checked_number(cell, cell_where)
        if not allow_zero:
            if number <= POSITIVE_EXCLUSIVE_MINIMUM:
                raise ValueError(
                    f"{cell_where} 必須是正實數阻抗；{unsupported_materials_hint(table)}"
                )
        elif not SCATTERING_MINIMUM <= number <= SCATTERING_MAXIMUM:
            raise ValueError(f"{cell_where} 必須落在 [0,1]（散射係數）")
        result[name] = number
    return result


def _reason_or_value(value: float | None, reason: str | None, *, where: str) -> None:
    """值空要帶原因；有值不准帶原因。

    「帶原因」是去掉前後空白之後還有字：空字串與整串空白讀起來跟沒有原因一樣，
    契約放它過等於允許一份說不出為什麼的空值。
    """
    if value is None and (reason is None or not reason.strip()):
        raise ValueError(f"{where} 是空的就必須帶不可估原因（空字串與空白不算原因）")
    if value is not None and reason is not None:
        raise ValueError(f"{where} 有值就不准再給不可估原因（兩個只准出現一個）")


def unsupported_materials_hint(table: CapabilityTable) -> str:
    """複數與逐頻阻抗的拒收訊息來源：**呼叫端那一張**能力表上兩條 unsupported 的 ``note``。

    字串本身不寫死在程式裡，命令列與輸入模型走同一條路：命令列把 ``--capabilities``
    那一張表傳進來，這裡只讀它，不自己去載第二張（兩個來源就會對不上）。
    """
    notes = [
        item.note
        for item in table.for_entry(CAPABILITY_ENTRY).capability
        if item.materials in UNSUPPORTED_MATERIALS and item.status == "unsupported"
    ]
    if not notes:
        raise ValueError(
            f"能力表 {CAPABILITY_ENTRY} 沒有複數或逐頻阻抗的 unsupported 條目"
        )
    return " ".join(dict.fromkeys(notes))


def _hint_rejection(exc: ValidationError) -> str:
    """把 Pydantic 的多行 dump 收成一句人話：哪一欄錯、錯在哪。

    只留 Pydantic 自己指名的 ``loc`` 與 ``msg``；官網網址那一行是錯誤物件的說明文字
    （``str(exc)`` 的尾巴），不是給人看的判決，命令列那一層不收。訊息裡的換行也壓成
    一個空格——那一層印的是**一行**。

    欄名只說一次：模型的驗證器自己就在訊息開頭寫了一次欄位路徑
    （``impedance_pa_s_per_m_by_wall.floor 必須是正實數阻抗``、``sound_speed_m_s 必須是
    有限正數``），而 ``loc`` 指的是同一格。兩者指的是同一個前綴時，``loc`` 那一份收掉
    ——``loc`` 比訊息更細（``impedance_pa_s_per_m_by_wall`` 對
    ``impedance_pa_s_per_m_by_wall.floor``）時留訊息那一份，因為它多說了是哪一面牆。
    """
    lines: list[str] = []
    for error in exc.errors():
        message = _one_line(str(error["msg"]).removeprefix("Value error, "))
        cells = ".".join(str(cell) for cell in error["loc"])
        if cells and (message == cells or message.startswith(f"{cells}.")):
            # 訊息自己就從欄位路徑講起：它比 loc 更細（多說了是哪一面牆），留它。
            lines.append(message)
        elif cells and message.startswith(f"{cells} "):
            # loc 與訊息開頭同名（``room_m：room_m 必須是 JSON 物件``）：說一次就夠。
            lines.append(message)
        elif cells:
            lines.append(f"{cells}：{message}")
        else:
            lines.append(f"輸入：{message}")
    return "；".join(lines) if lines else "輸入不合契約但沒有可讀的欄位路徑"


def _one_line(text: str) -> str:
    """把一段話壓成一行（換行與連續空白收成一個空格）。"""
    return " ".join(text.split())


def _rejected(exc: ValidationError) -> ValueError:
    """模型的驗證失敗 → 一句人話的 :class:`ValueError`（命令列只看得見這一種形狀）。"""
    return ValueError(f"輸入不合 ReportInput：{_hint_rejection(exc)}")


def load_input_document(document: object, table: CapabilityTable) -> ReportInput:
    """把一份已經讀進來的 JSON 文件驗成 :class:`ReportInput`。

    能力表由呼叫端必給：拒收訊息的那一句 hint 要跟命令列 ``--capabilities`` 指定的
    **同一張**表算，這裡不再自己去載第二張（同一句話不准有兩個來源）。
    """
    context = {"table": table}
    try:
        return ReportInput.model_validate(document, context=context)
    except ValidationError as exc:
        raise _rejected(exc) from exc


def load_input(path: Path, table: CapabilityTable) -> ReportInput:
    """從路徑讀一份輸入 JSON 並驗成 :class:`ReportInput`；能力表由呼叫端必給。"""
    with path.open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    return load_input_document(loaded, table)


class SolverInputs(NamedTuple):
    """``solve_three_lane_report`` 吃的那九格，型別就是那九格的型別。"""

    room: Room
    source: Point
    receiver: Point
    sound_speed_m_s: float
    density_kg_m3: float
    impedance_by_wall: dict[Wall, float]
    scattering_by_wall: dict[Wall, float] | None
    reflection_order_k: int
    low_frequency_axis: LowFrequencyAxis


def solver_inputs(inputs: ReportInput) -> SolverInputs:
    """把 :class:`ReportInput` 攤成 ``solve_three_lane_report`` 吃的那九格。

    牆名那兩格在這裡翻成 :class:`~aosr.geometry.shoebox.Wall`
    （``three_lane_report._wall_impedances`` 收的是 ``Mapping[Wall, …]``）。
    ``reflection_order_k`` 原樣帶過去：輸入檔沒給那一格時模型本來就填了產品設定
    ``REFLECTION_ORDER_K``，所以這裡不必再判一次「有沒有給」。
    ``low_frequency_axis`` 同理原樣帶過去（#435）：漏帶的話，從這裡算的報表會悄悄落回搜尋軸。
    """
    return SolverInputs(
        room=inputs.room_m,
        source=inputs.source_m,
        receiver=inputs.receiver_m,
        sound_speed_m_s=inputs.sound_speed_m_s,
        density_kg_m3=inputs.density_kg_m3,
        impedance_by_wall={
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
    table.update(_prefixed_facts("path_table", PathTableSection))
    table.update(_prefixed_facts("path_table.rows", PathRow))
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
