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
* ``reference`` 參考基準（下一段）
* ``validity`` 有效狀態（``estimable`` 可估／``unestimable`` 不可估，後者帶原因欄名）

**參考基準怎麼判的（每一條都回得到程式或決策紙；沒把握的照實寫在值裡）。**

* 幾何路所有能量欄都是相對量：``docs/decisions/stage-five-ism-totals-and-energy.md`` 定
  「直達能量是直達那一條壓力的模平方」，而 ``path_pressure`` 沒有音源強度因子
  （``src/aosr/physics/amplitude.py``：``(1/dist)·refl·exp(−iωτ)``），距離單位是公尺
  －－所以是「單位振幅點源、距離 1 公尺處壓力為 1」的相對基準，不是帕（Pa）。
* 反射／干涉是同一條基準上的**分項**（反射為一到三階壓力的同調和、干涉為直達與反射的交叉項），
  可以為負；不是各自的絕對值。
* 有限元素欄是 ``4π`` 點音源的同一個 ``|p(1 m)| = 1`` 家族：``blueprint/fem_fenics_problem.json``
  的 ``physics.source_strength = "4*pi"``，與 ``src/aosr/physics/fem_helmholtz.py`` 的
  ``POINT_SOURCE_STRENGTH`` 同值。**它與幾何路的逐位對齊沒有機器在守、這一版沒有對過**
  （交接檔「要查」那一條記著同一件事），所以欄上寫的是「未對過」而不是斷言同一把尺。
* 晚期能量有音源功率因子 ``4π``：``src/aosr/physics/late_energy.py`` 的 ``_exact_raw_energy``
  回傳 ``DIFFUSE_MONOPOLE_4PI · 4 · mean_reflected``。
* 權重、散射、計數無因次；T20／T30 是秒；頻率是 Hz；``f_s_hz`` 是
  ``2000·sqrt(mean(T60 at 500/1000 Hz)/V)``（``src/aosr/physics/crossover.py``）。
* ``room_volume_m3`` 是房三軸長相乘；``schroeder_band_count`` 是餵進 Eyring 的中頻帶數
  （``SCHROEDER_T60_BANDS_HZ``）——這一格是「用了幾個帶」，不是哪兩個帶，帶的頻率值要看得從
  產品設定讀。
* capability 那四格照 ``capability_report.capability_line`` 的四格原樣收：頻率範圍兩端、
  輸出欄清單、狀態、收據。``status`` 是表上那一條本人，不是抽出來的字串。
* ``description`` 這一類文字格只是把「上面那些數字在什麼基準上」寫成人話帶出去，
  不是另一套數字；四件事本身住在 ``json_schema_extra``。

**失敗分三種（票上的原話；這一版做到前兩種）。**

1. 求解失敗整份往上冒：這一層不攔任何求解例外，命令列照樣回 2。
2. 某個指標算不出來就保留其他結果並帶原因：T20／T30 今天已經是這個形狀，輸出模型把它寫成
   規則－－值空必須有原因、有值又不准給原因（:meth:`ReportOutput._decay_values_carry_reasons`）。
3. 最佳化目標缺值：最佳化還沒進來，這一版不做。**缺值不准變成分數**：目標欄缺值時不准當 0 分、
   也不准沿用上一個值，形狀等最佳化真的進來再定。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, NamedTuple, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from aosr.geometry.shoebox import Point, Room, Wall
from pydantic.config import JsonDict


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

# 複數與逐頻阻抗這兩條材料形式的能力表代號；拒收訊息從那兩條 unsupported 的 note 讀。
UNSUPPORTED_MATERIALS: Final[tuple[str, ...]] = (
    "complex_impedance_by_wall",
    "frequency_dependent_impedance",
)
CAPABILITY_ENTRY: Final[str] = "three_lane_report"


class FieldFacts(NamedTuple):
    """一欄的四件事：物理量、單位、參考基準、有效狀態。"""

    quantity: str
    unit: str
    reference: str
    validity: str


def _facts(
    quantity: str,
    unit: str,
    reference: str,
    validity: str = "estimable",
) -> JsonDict:
    """組出 ``json_schema_extra``；四格一個都不能少。"""
    return JsonDict(
        {
            "quantity": quantity,
            "unit": unit,
            "reference": reference,
            "validity": validity,
        }
    )


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
            extra = field.json_schema_extra
            if not isinstance(extra, dict):
                raise ValueError(
                    f"{cls.__name__}.{name} 沒有 json_schema_extra，四件事不完整"
                )
            table[name] = FieldFacts(
                quantity=str(extra["quantity"]),
                unit=str(extra["unit"]),
                reference=str(extra["reference"]),
                validity=str(extra["validity"]),
            )
        return table


# ── 參考基準的文字；每一條的出處寫在檔頭，沒把握的照實寫 ────────────────────────
_RELATIVE: Final[str] = (
    "單位振幅點源、距離 1 公尺處壓力為 1 的相對能量（|p(1 m)|=1；路徑壓力不含音源強度"
    "因子，欄位不是帕）"
)
_FEM: Final[str] = (
    "4π 點音源、|p(1 m)|=1 的同一家族（題目檔 source_strength=\"4*pi\"）；與幾何路的逐位"
    "對齊沒有機器在守，這一版沒有對過"
)
_LATE: Final[str] = "以 4π 點音源功率為因子的晚期混響能量（raw_energy × 4π × 4 × 平均反射場）"
_DIMENSIONLESS: Final[str] = "無因次"
_FRACTION: Final[str] = "無因次；功率互補權重，兩欄相加為 1"
_SECTION: Final[str] = "欄名清單，不是量測值"


class ReportInput(_FactsModel):
    """三路接合報表輸入 JSON 的迷你契約（房、聲源、接收點、介質、六面材料）。

    驗證規則照命令列原本那一套：六個牆名固定、數字要有限、阻抗為正、散射落在 ``[0,1]``；
    複數或逐頻阻抗（物件或陣列）當場拒收，訊息從能力表那幾條 unsupported 的 ``note`` 讀
    （:func:`unsupported_materials_hint`），不在這裡再抄一次。
    """

    room_m: Room = Field(
        description="鞋盒房間三軸長度（公尺）",
        json_schema_extra=_facts("長度", "m", "房間角落為原點"),
    )
    source_m: Point = Field(
        description="點聲源座標（公尺）",
        json_schema_extra=_facts("長度", "m", "房間角落為原點"),
    )
    receiver_m: Point = Field(
        description="接收點座標（公尺）",
        json_schema_extra=_facts("長度", "m", "房間角落為原點"),
    )
    sound_speed_m_s: float = Field(
        description="聲速（公尺／秒）",
        json_schema_extra=_facts("聲速", "m/s", _DIMENSIONLESS),
    )
    density_kg_m3: float = Field(
        description="空氣密度（公斤／立方公尺）",
        json_schema_extra=_facts("密度", "kg/m^3", _DIMENSIONLESS),
    )
    impedance_pa_s_per_m_by_wall: dict[str, float] = Field(
        description="六面牆各一個與頻率無關的正實數阻抗（帕·秒／公尺）",
        json_schema_extra=_facts("表面阻抗", "Pa*s/m", _DIMENSIONLESS),
    )
    scattering_by_wall: dict[str, float] | None = Field(
        default=None,
        description="六面牆各一個落在 [0,1] 的散射係數；整格可省略",
        json_schema_extra=_facts("散射係數", "1", _DIMENSIONLESS),
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
        coordinates = ("Lx", "Ly", "Lz") if where == "room_m" else ("x", "y", "z")
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
    def _impedance_cells(cls, value: object) -> object:
        """阻抗六面：物件、六個固定牆名、複數或逐頻形式當場拒收。"""
        return _wall_numbers(
            value,
            where="impedance_pa_s_per_m_by_wall",
            allow_zero=False,
        )

    @field_validator("scattering_by_wall", mode="before")
    @classmethod
    def _scattering_cells(cls, value: object) -> object:
        """散射係數是另一種材料形式：逐面一個落在 [0,1] 的係數，可以等於 0。"""
        if value is None:
            return None
        return _wall_numbers(value, where="scattering_by_wall", allow_zero=True)

    @model_validator(mode="after")
    def _medium_must_be_physical(self) -> Self:
        """聲速與密度都必須是有限正數。"""
        for where, value in (
            ("sound_speed_m_s", self.sound_speed_m_s),
            ("density_kg_m3", self.density_kg_m3),
        ):
            if value <= 0.0:
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
            if not _number_is_finite(length) or length <= 0.0:
                raise ValueError(f"room_m.{name} 必須是有限正數")
        return self


class CapabilitySection(_FactsModel):
    """能力表那一條本人：範圍、輸出欄、狀態與收據（``capability_line`` 的四格）。"""

    frequency_hz: tuple[float, float] = Field(
        description="這一條宣告的頻率範圍兩個端點",
        json_schema_extra=_facts("頻率", "Hz", "報告頻率軸上的兩個端點"),
    )
    outputs: tuple[str, ...] = Field(
        description="這一條宣告的輸出欄名",
        json_schema_extra=_facts("欄名", "1", _SECTION),
    )
    status: str = Field(
        min_length=1,
        description="能力表三個狀態之一；unchecked 代表這一跑沒查表",
        json_schema_extra=_facts("能力狀態", "1", _DIMENSIONLESS),
    )
    evidence: tuple[str, ...] = Field(
        description="這一條指名的收據；沒查表時是空的",
        json_schema_extra=_facts("收據", "1", _DIMENSIONLESS),
    )


class TopFields(NamedTuple):
    """頂層那幾格；欄位順序照 ``_top_table`` 印出來的順序。"""

    f_s_hz: float = Field(
        json_schema_extra=_facts("頻率", "Hz", "由 500／1000 Hz 的 Eyring T60 與房間體積算出的交接頻率")
    )
    crossover_lower_hz: float = Field(
        json_schema_extra=_facts("頻率", "Hz", "交接區間下端")
    )
    crossover_upper_hz: float = Field(
        json_schema_extra=_facts("頻率", "Hz", "交接區間上端，等於有限元素上限")
    )
    capped_by_upper_limit: bool = Field(
        json_schema_extra=_facts("狀態", "1", _DIMENSIONLESS)
    )
    eyring_t60_by_band_s: dict[str, float] = Field(
        json_schema_extra=_facts("時間", "s", "Eyring 公式、面積加權平均吸音率")
    )
    room_volume_m3: float = Field(
        json_schema_extra=_facts("體積", "m^3", _DIMENSIONLESS)
    )
    schroeder_band_count: int = Field(
        ge=0,
        json_schema_extra=_facts("計數", "1", "算出上部那個時間所用的中頻帶數（帶的頻率值從產品設定讀）"),
    )


class BandRow(NamedTuple):
    """頻帶表一列；欄位順序**就是**印出來的欄位順序，這張表是凍結的。

    ``fem_energy`` 的 ``None`` 代表這個帶內一個有限元素頻點都沒有（``fem_point_count`` 是 0），
    不是「值算不出來」；``t20_s``／``t30_s`` 的 ``None`` 才帶不可估原因。
    """

    center_frequency_hz: float = Field(json_schema_extra=_facts("頻率", "Hz", "八度帶中心頻率"))
    fem_energy: float | None = Field(
        json_schema_extra=_facts("能量列", "1", _FEM, "unestimable")
    )
    fem_point_count: int = Field(
        ge=0, json_schema_extra=_facts("計數", "1", _DIMENSIONLESS)
    )
    direct_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    reflected_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    interference_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    late_energy: float = Field(json_schema_extra=_facts("能量", "1", _LATE))
    geometric_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    fem_contribution: float = Field(json_schema_extra=_facts("能量", "1", _FEM))
    geometric_contribution: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    total_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    w_fem: float = Field(json_schema_extra=_facts("權重", "1", _FRACTION))
    w_geo: float = Field(json_schema_extra=_facts("權重", "1", _FRACTION))
    f_s_hz: float = Field(json_schema_extra=_facts("頻率", "Hz", "整個報表共用同一個交接頻率"))
    capped_by_upper_limit: bool = Field(
        json_schema_extra=_facts("狀態", "1", _DIMENSIONLESS)
    )
    t20_s: float | None = Field(
        json_schema_extra=_facts("時間", "s", "由同一條晚期衰減曲線的 −5～−25 dB 視窗擬合")
    )
    t20_unavailable_reason: str | None = Field(
        json_schema_extra=_facts("不可估原因", "1", _DIMENSIONLESS)
    )
    t30_s: float | None = Field(
        json_schema_extra=_facts("時間", "s", "由同一條晚期衰減曲線的 −5～−35 dB 視窗擬合")
    )
    t30_unavailable_reason: str | None = Field(
        json_schema_extra=_facts("不可估原因", "1", _DIMENSIONLESS)
    )


class PointRow(NamedTuple):
    """細軸逐點表一列；欄位順序**就是**印出來的欄位順序。"""

    frequency_hz: float = Field(json_schema_extra=_facts("頻率", "Hz", "1/24 八度細軸上的頻點"))
    fem_energy: float | None = Field(
        json_schema_extra=_facts(
            "能量列", "1", _FEM, "unestimable（頻點高於有限元素上限時沒有這一格）"
        )
    )
    direct_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    reflected_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    interference_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    late_energy: float = Field(json_schema_extra=_facts("能量", "1", _LATE))
    scattering: float = Field(json_schema_extra=_facts("散射係數", "1", _DIMENSIONLESS))
    geometric_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    w_fem: float = Field(json_schema_extra=_facts("權重", "1", _FRACTION))
    w_geo: float = Field(json_schema_extra=_facts("權重", "1", _FRACTION))
    total_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))


class ReportOutput(_FactsModel):
    """三路接合報表的三張表與 capability 那一行，收成一個可驗的結構。

    這一層只負責形狀與三條規則：頻帶列的中心頻率要遞增不重複、交接下端不准超過上端、
    以及不可估的欄位一定要帶原因（值空必須有原因、有值又不准給原因）。
    """

    capability: CapabilitySection = Field(
        json_schema_extra=_facts("能力", "1", _SECTION)
    )
    top: TopFields = Field(
        json_schema_extra=_facts("頂層總量", "1", "見底下每一欄自己的基準")
    )
    bands: tuple[BandRow, ...] = Field(
        json_schema_extra=_facts("頻帶列", "1", "見底下每一欄自己的基準")
    )
    points: tuple[PointRow, ...] | None = Field(
        default=None,
        description="細軸逐點表；沒有 --points 時整塊省略",
        json_schema_extra=_facts("逐點列", "1", "見底下每一欄自己的基準"),
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
            raise ValueError("bands 不可為空：報表一定有六個八度帶")
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
    """有限數字；布林不算數字。"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    as_float = float(value)
    return as_float == as_float and as_float not in (float("inf"), float("-inf"))


def _checked_number(value: object, where: str) -> float:
    """照命令列原本的寫法驗一格數字，訊息一字不變。"""
    if not _number_is_finite(value):
        raise ValueError(f"{where} 必須是有限數字")
    assert isinstance(value, int | float)
    return float(value)


def _checked_mapping(value: object, where: str) -> dict[str, object]:
    """照命令列原本的寫法驗一層 JSON 物件，訊息一字不變。"""
    if not isinstance(value, dict):
        raise ValueError(f"{where} 必須是 JSON 物件")
    return {str(key): item for key, item in value.items()}


def _wall_numbers(value: object, *, where: str, allow_zero: bool) -> dict[str, float]:
    """六面牆各一格數字；阻抗要正、散射落在 [0,1]，複數或逐頻形式當場拒收。"""
    fields = _checked_mapping(value, where)
    result: dict[str, float] = {}
    for wall in Wall.all():
        name = wall.wall_name()
        cell_where = f"{where}.{name}"
        cell = fields.get(name)
        if isinstance(cell, dict | list | tuple):
            hint = (
                unsupported_materials_hint()
                if where == "impedance_pa_s_per_m_by_wall"
                else "散射係數只收一個實數"
            )
            raise ValueError(f"{cell_where}：{hint}")
        number = _checked_number(cell, cell_where)
        if not allow_zero:
            if number <= 0.0:
                raise ValueError(
                    f"{cell_where} 必須是正實數阻抗；{unsupported_materials_hint()}"
                )
        elif not 0.0 <= number <= 1.0:
            raise ValueError(f"{cell_where} 必須落在 [0,1]（散射係數）")
        result[name] = number
    return result


def _reason_or_value(value: float | None, reason: str | None, *, where: str) -> None:
    """值空要帶原因；有值不准帶原因。"""
    if value is None and reason is None:
        raise ValueError(f"{where} 是空的就必須帶不可估原因")
    if value is not None and reason is not None:
        raise ValueError(f"{where} 有值就不准再給不可估原因（兩個只准出現一個）")


def unsupported_materials_hint() -> str:
    """複數與逐頻阻抗的拒收訊息來源：能力表那兩條 unsupported 的 ``note``。

    字串本身不寫死在程式裡，跟命令列原本走同一條路；表改了訊息跟著改。
    """
    from aosr.config.capabilities import load_capabilities
    from aosr.config.paths import config_path

    table = load_capabilities(config_path("capabilities.toml"))
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


def load_input_document(document: object) -> ReportInput:
    """把一份已經讀進來的 JSON 文件驗成 :class:`ReportInput`。

    模型的驗證失敗一律轉成 ``ValueError``（訊息帶 Pydantic 指名的欄位路徑），
    命令列那一層因此只看得見一種例外形狀。
    """
    try:
        return ReportInput.model_validate(document)
    except ValidationError as exc:
        raise ValueError(f"輸入不合 {ReportInput.__name__}：{exc}") from exc


def load_input(path: Path) -> ReportInput:
    """從路徑讀一份輸入 JSON 並驗成 :class:`ReportInput`。"""
    with path.open(encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    return load_input_document(loaded)


def _capability_section(report: object) -> CapabilitySection:
    """報表那一格能力：表上那一條本人；沒查表時狀態是 unchecked、範圍與收據是空的。"""
    record = report.capability.record  # type: ignore[attr-defined]  # expires=2026-12-08 reason=呼叫端已驗過型別，這一支只收報告物件的那一格
    return CapabilitySection(
        frequency_hz=record.frequency_hz if record is not None else (0.0, 0.0),
        outputs=record.outputs if record is not None else ("none",),
        status=record.status if record is not None else "unchecked",
        evidence=record.evidence if record is not None else (),
    )


def _top_fields(report: object, room: Room) -> TopFields:
    """頂層那幾格；Schroeder 帶數從產品設定讀，不寫死第二份。"""
    from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ

    return TopFields(
        f_s_hz=report.f_s_hz,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        crossover_lower_hz=report.crossover_lower_hz,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        crossover_upper_hz=report.crossover_upper_hz,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        capped_by_upper_limit=report.capped_by_upper_limit,  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        eyring_t60_by_band_s={
            str(frequency): value
            for frequency, value in report.eyring_t60_by_band_s.items()  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
        },
        room_volume_m3=room.Lx * room.Ly * room.Lz,
        schroeder_band_count=len(SCHROEDER_T60_BANDS_HZ),
    )


def _band_rows(report: object) -> tuple[BandRow, ...]:
    """頻帶列；欄位順序就是印出來的欄位順序。"""
    return tuple(
        BandRow(
            center_frequency_hz=band.center_frequency_hz,
            fem_energy=band.fem_energy,
            fem_point_count=band.fem_point_count,
            direct_energy=band.direct_energy,
            reflected_energy=band.reflected_energy,
            interference_energy=band.interference_energy,
            late_energy=band.late_energy,
            geometric_energy=band.geometric_energy,
            fem_contribution=band.fem_contribution,
            geometric_contribution=band.geometric_contribution,
            total_energy=band.total_energy,
            w_fem=band.w_fem,
            w_geo=band.w_geo,
            f_s_hz=band.f_s_hz,
            capped_by_upper_limit=band.capped_by_upper_limit,
            t20_s=band.t20_s,
            t20_unavailable_reason=band.t20_unavailable_reason,
            t30_s=band.t30_s,
            t30_unavailable_reason=band.t30_unavailable_reason,
        )
        for band in report.bands  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
    )


def _point_rows(report: object) -> tuple[PointRow, ...]:
    """細軸逐點列；欄位順序就是印出來的欄位順序。"""
    return tuple(
        PointRow(
            frequency_hz=point.frequency_hz,
            fem_energy=point.fem_energy,
            direct_energy=point.direct_energy,
            reflected_energy=point.reflected_energy,
            interference_energy=point.interference_energy,
            late_energy=point.late_energy,
            scattering=point.scattering,
            geometric_energy=point.geometric_energy,
            w_fem=point.w_fem,
            w_geo=point.w_geo,
            total_energy=point.total_energy,
        )
        for point in report.points  # type: ignore[attr-defined]  # expires=2026-12-08 reason=同上
    )


def output_from_report(
    report: object,
    *,
    room: Room,
    with_points: bool,
) -> ReportOutput:
    """把 :class:`~aosr.physics.three_lane_report.ThreeLaneReport` 收成 :class:`ReportOutput`。

    ``with_points`` 決定帶不帶細軸逐點表。
    """
    from aosr.physics.three_lane_report import ThreeLaneReport

    if not isinstance(report, ThreeLaneReport):
        raise ValueError(f"report 不是 ThreeLaneReport：{type(report).__name__}")
    return ReportOutput(
        capability=_capability_section(report),
        top=_top_fields(report, room),
        bands=_band_rows(report),
        points=_point_rows(report) if with_points else None,
    )


class SolverInputs(NamedTuple):
    """``solve_three_lane_report`` 吃的那七格，型別就是那七格的型別。"""

    room: Room
    source: Point
    receiver: Point
    sound_speed_m_s: float
    density_kg_m3: float
    impedance_by_wall: dict[Wall, float]
    scattering_by_wall: dict[Wall, float] | None


def solver_inputs(inputs: ReportInput) -> SolverInputs:
    """把 :class:`ReportInput` 攤成 ``solve_three_lane_report`` 吃的那七格。

    牆名那兩格在這裡翻成 :class:`~aosr.geometry.shoebox.Wall`
    （``three_lane_report._wall_impedances`` 收的是 ``Mapping[Wall, …]``）。
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
    )


def input_schema() -> dict[str, object]:
    """輸入模型現算出來的 JSON schema（``blueprint/schemas`` 那一份就是它）。"""
    return ReportInput.model_json_schema()


def output_schema() -> dict[str, object]:
    """輸出模型現算出來的 JSON schema（``blueprint/schemas`` 那一份就是它）。"""
    return ReportOutput.model_json_schema()


def quantity_table() -> dict[str, FieldFacts]:
    """回傳「欄名 → 四件事」的對照表（輸入、輸出與兩種列都攤平進來）。

    列的欄名用 ``bands.``／``points.`` 當前綴；``top`` 的欄位直接用 ``top.`` 前綴。
    """
    table: dict[str, FieldFacts] = {}
    for model in (ReportInput, CapabilitySection, ReportOutput):
        table.update(model.quantity_table())
    table.update(_row_facts("bands", BandRow))
    table.update(_row_facts("points", PointRow))
    table.update(_row_facts("top", TopFields))
    return table


def _row_facts(prefix: str, row: type[NamedTuple]) -> dict[str, FieldFacts]:
    """把一列 NamedTuple 的欄位四件事攤成 ``prefix.欄名``。"""
    table: dict[str, FieldFacts] = {}
    for name in row._fields:  # noqa: SLF001  # expires=2026-12-08 reason=NamedTuple 的欄名清單只有這一個地方拿得到
        default = row._field_defaults.get(name)  # noqa: SLF001  # expires=2026-12-08 reason=同上
        extra = getattr(default, "json_schema_extra", None)
        if not isinstance(extra, dict):
            raise ValueError(f"{prefix}.{name} 沒有 json_schema_extra，四件事不完整")
        table[f"{prefix}.{name}"] = FieldFacts(
            quantity=str(extra["quantity"]),
            unit=str(extra["unit"]),
            reference=str(extra["reference"]),
            validity=str(extra["validity"]),
        )
    return table
