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
* ``reference`` 參考基準（相對於什麼，或「絕對值」；**不是**單位，字串欄與計數欄寫
  「沒有基準（不是量測值）」——每一條回得到程式或決策紙）
* ``validity`` 有效狀態。**只有真的「估出來的數值欄」才寫「可估」**；不是數值的那些格子
  （文字、狀態旗標、計數、收據、欄名、以及 ``capability``／``top``／``bands``／``points``
  這種本身不是量測值的容器欄）各自寫實話「不是估出來的量測值」。兩族「空」另外分開寫：
  ``fem_energy`` 空＝這一帶沒有有限元素頻點，不必帶原因；``t20_s``／``t30_s`` 空＝值算不出來，
  必須帶原因。說明與 validity 寫在同一句裡，不讓兩處各說各話）

**參考基準怎麼判的（每一條都回得到程式或決策紙；沒把握的照實寫在值裡）。**

* 幾何路的**早期**能量欄是相對量：``docs/decisions/stage-five-ism-totals-and-energy.md`` 定
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
  **回傳的是** ``DIFFUSE_MONOPOLE_4PI · 4 · mean_reflected``；但欄上**不是那個值本人**——
  ``:438`` 那條路再乘一次 ``_eyring_ratio(alpha_bar)`` 才得到 ``late_reverberant_energy``
  （``guarded / -log1p(-guarded)``，只有 ``alpha_bar`` 趨近 0 時才等於 1），
  ``geometric_lane.py:389`` 拿的就是這一格，頻帶表再對帶內細軸點取平均
  （``geometric_lane.py:226``）。所以欄上是「4π 因子的一次項再乘 Eyring 比值」。
* ``geometric_energy``／``geometric_contribution``／``total_energy`` 是**混合基準**：
  ``geometric_lane._geometric_energy`` 是
  ``direct + (1−s)(reflected+interference) + s·late``——含帶 4π 的晚期項；
  ``three_lane_report._band_contributions`` 的 ``geometric_contribution`` 同樣把
  ``w_geo·late`` 加進去；``total_energy``（``three_lane_report.py``：
  ``fem_contribution + geometric_contribution``）再把有限元素那一路加進來。**兩路的基準
  對齊沒有機器在守、這一版沒有對過**，所以這三欄寫的是混合基準，不是單一幾何相對基準。
* 權重、散射無因次；聲速、密度、阻抗、體積、時間與頻率是絕對值（有量綱）；計數與欄名、
  狀態、原因這些字串格寫「沒有基準（不是量測值）」。
* ``f_s_hz`` 是 ``2000·sqrt(mean(T60 at 500/1000 Hz)/V)``（``src/aosr/physics/crossover.py``）；
  那兩個中頻帶的頻率值從產品設定 ``config/three_lane_crossover.SCHROEDER_T60_BANDS_HZ`` 讀，
  這個檔不抄一份（會漂進 schema 檔）。
* ``room_volume_m3`` 是房三軸長相乘；``schroeder_band_count`` 是餵進 Eyring 的中頻帶數
  （``SCHROEDER_T60_BANDS_HZ``）——這一格是「用了幾個帶」，不是哪兩個帶。
* capability 那四格照 ``capability_report.capability_line`` 的四格原樣收：頻率範圍兩端、
  輸出欄清單、狀態、收據。``status`` 是表上那一條本人，不是抽出來的字串。
* ``description`` 這一類文字格只是把「上面那些數字在什麼基準上」寫成人話帶出去，
  不是另一套數字；四件事本身住在 ``json_schema_extra``。

**列類別是有欄名的物件（票 #316 第二刀）。** ``TopFields``／``BandRow``／``PointRow``
是凍結的 Pydantic 模型，不是 ``NamedTuple``：``--format json`` 印出來是有欄名的物件，
加一欄不會靜靜改掉既有消費者讀到的意思；四件事也跟其他模型走同一條 ``model_fields``
的路。

**schema 檔的 draft。** 兩份匯出的 schema 檔都宣告
``"$schema": "https://json-schema.org/draft/2020-12/schema"``（Pydantic 出廠的就是這一版）。
``$ref`` 與它旁邊的四件事（``quantity``／``unit``／``reference``／``validity``）是**兄弟鍵**：
只有 draft 2020-12 認得兄弟鍵，舊版（draft-07）工具會整段忽略它們。這一版不改寫法，
前端真的踩到再改——改法會是「把四件事收進一個被 ``$ref`` 指到的定義」，不是拆掉兄弟鍵。

**改了模型，schema 檔要重匯。** ``blueprint/schemas/`` 那兩份檔是模型現算結果的存檔，
考卷 ``test_schema_files_match_the_models`` 逐格比對，忘了重匯就紅。重匯的入口是既有那支
命令列（物理層這一支只寫檔、不對人說話）：

.. code-block:: shell

   uv run python -m aosr.physics.three_lane_report_cli --regenerate-schemas blueprint/schemas

它會把兩份檔寫成模型現算出來的內容（目錄必給：覆寫版控那兩份要自己把
``blueprint/schemas`` 寫出來，手滑不帶目錄只會被 argparse 擋下來），並印出寫了哪幾個檔。
考卷紅掉的那一句訊息裡也寫著同一行命令。

**失敗分三種（票上的原話；這一版做到前兩種）。**

1. 求解失敗整份往上冒：這一層不攔任何求解例外，命令列照樣回 2。
2. 某個指標算不出來就保留其他結果並帶原因：T20／T30 今天已經是這個形狀，輸出模型把它寫成
   規則－－值空必須有原因、有值又不准給原因（:meth:`ReportOutput._decay_values_carry_reasons`）。
3. 最佳化目標缺值：最佳化還沒進來，這一版不做。**缺值不准變成分數**：目標欄缺值時不准當 0 分、
   也不准沿用上一個值，形狀等最佳化真的進來再定。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final, NamedTuple, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from aosr.config.capabilities import CapabilityTable
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
    validity: str = "可估",
) -> JsonDict:
    """組出 ``json_schema_extra``；四格一個都不能少。

    預設那一格是「可估」（這一欄有值就是量到的東西）；不是數值的那些格子自己指名
    ``_NOT_MEASURED``（那一格根本不是估出來的東西）；會出現空值的兩族另外指名
    ``_EMPTY_UNLESS``（值算不出來）或 ``_EMPTY_WHEN``（這一格沒有有限元素頻點）。
    預設值必須等於 :data:`_ESTIMABLE`（同一件事不准有兩個字面）；
    ``test_facts_default_validity_is_the_estimable_constant`` 咬住「兩處同一個字串」。
    """
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
_LATE: Final[str] = (
    "晚期混響能量：late_energy._exact_raw_energy 的 raw_energy（＝4π × 4 × 平均反射場）"
    "再乘一次 _eyring_ratio(alpha_bar) 比值（late_energy.py 的 late_reverberant_energy），"
    "頻帶欄再對帶內細軸點取平均——不是 raw_energy 本人"
)
# 混合基準：這兩格不是單一幾何基準，而是把兩路各自基準的東西加在一起。
_MIXED: Final[str] = (
    "混合基準：含 4π 音源功率因子的晚期項（late_energy）與幾何路的早期項相加，"
    "兩路基準不是同一把尺"
)
_MIXED_TOTAL: Final[str] = (
    "混合基準：有限元素那一路與幾何那一路（含 4π 的晚期項）相加；兩路的基準對齊"
    "沒有機器在守、這一版沒有對過"
)
_NO_BASIS_TEXT: Final[str] = "沒有基準（只是文字或狀態，不是量測值）"
_NO_BASIS_COUNT: Final[str] = "沒有基準（只是計數，不是量測值）"
_NO_BASIS_NAMES: Final[str] = "沒有基準（只是欄名清單，不是量測值）"
_NO_BASIS_RANGE: Final[str] = "沒有基準（只是能力表上宣告的頻率範圍兩個端點，不是量測值）"
# 可估的那一格與不是量測的那一格各自實話；值空時誰帶原因跟在後面。
# ``_ESTIMABLE`` 就是 ``_facts`` 的預設值那一個字面，而 ``_facts`` 定義在上面那一區、
# 拿不到這裡的常數，所以預設值寫字面、兩處靠考卷
# ``test_facts_default_validity_is_the_estimable_constant`` 咬住。
_ESTIMABLE: Final[str] = "可估"
# 不是量測值的那些格子：文字、狀態旗標、計數、收據、欄名、表上宣告的值（頻率範圍），
# 以及本身只是容器的那幾欄（``capability``／``top``／``bands``／``points``）。
# 它們的「有效狀態」不是「估不估」，寫「可估」等於說這些格子是量出來的。
_NOT_MEASURED: Final[str] = (
    "不是估出來的量測值（這一格是文字、狀態、容器、表上宣告的值或計數，不是估出來的量）"
)
# 兩族「空」：fem_energy 空＝這一帶沒有有限元素頻點（不必帶原因）；
# T20／T30 空＝值算不出來（必須帶原因）。說明與 validity 同一句話。
_EMPTY_WHEN: Final[str] = (
    "這一格可能是空的：空＝這一格沒有有限元素頻點，不必帶原因"
)
_EMPTY_UNLESS: Final[str] = "可估（空＝值算不出來，必須帶原因；有值就不准再給原因）"
_FRACTION: Final[str] = "無因次；功率互補權重，兩欄相加為 1"
# 交接頻率那一格的說明要把「哪兩個中頻帶」從產品設定讀，不把那些數字抄進這個檔。
_F_S_REFERENCE: Final[str] = (
    "絕對值：由產品設定 ``three_lane_crossover.SCHROEDER_T60_BANDS_HZ`` 那兩個中頻帶的"
    "Eyring T60 與房間體積算出的交接頻率"
)


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
        json_schema_extra=_facts("聲速", "m/s", "絕對值（帕·秒／公尺那條阻抗換算用）"),
    )
    density_kg_m3: float = Field(
        description="空氣密度（公斤／立方公尺）",
        json_schema_extra=_facts("密度", "kg/m^3", "絕對值（與聲速相乘得 ρc）"),
    )
    impedance_pa_s_per_m_by_wall: dict[str, float] = Field(
        description="六面牆各一個與頻率無關的正實數阻抗（帕·秒／公尺）",
        json_schema_extra=_facts("表面阻抗", "Pa*s/m", "絕對值（不是無因次）"),
    )
    scattering_by_wall: dict[str, float] | None = Field(
        default=None,
        description="六面牆各一個落在 [0,1] 的散射係數；整格可省略",
        json_schema_extra=_facts("散射係數", "1", "相對於入射功率的比例（0 到 1）"),
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

    frequency_hz: tuple[float, float] | tuple[()] = Field(
        description="這一條宣告的頻率範圍：空（沒查表）或剛好兩個端點",
        json_schema_extra=_facts("頻率", "Hz", _NO_BASIS_RANGE, _NOT_MEASURED),
    )
    outputs: tuple[str, ...] = Field(
        description="這一條宣告的輸出欄名",
        json_schema_extra=_facts("欄名", "1", _NO_BASIS_NAMES, _NOT_MEASURED),
    )
    status: str = Field(
        min_length=1,
        description="能力表三個狀態之一；unchecked 代表這一跑沒查表",
        json_schema_extra=_facts("能力狀態", "1", _NO_BASIS_TEXT, _NOT_MEASURED),
    )
    evidence: tuple[str, ...] = Field(
        description="這一條指名的收據；沒查表時是空的",
        json_schema_extra=_facts("收據", "1", _NO_BASIS_TEXT, _NOT_MEASURED),
    )


class TopFields(_FactsModel):
    """頂層那幾格；宣告順序不是印出來的順序（JSON 照字母排，文字表由 ``_top_table`` 自己排）。

    這三個列類別跟其他模型一樣是凍結的 Pydantic 模型，**不是** NamedTuple：JSON 出去
    的是有欄名的物件（``{"f_s_hz": …}``），不是位置陣列——前端不必數到第幾格才知道
    那一格是什麼，加一欄也不會靜靜改掉既有消費者讀到的意思。
    """

    f_s_hz: float = Field(json_schema_extra=_facts("頻率", "Hz", _F_S_REFERENCE))
    crossover_lower_hz: float = Field(
        json_schema_extra=_facts("頻率", "Hz", "絕對值：交接區間下端的頻率")
    )
    crossover_upper_hz: float = Field(
        json_schema_extra=_facts("頻率", "Hz", "絕對值：交接區間上端，等於有限元素上限")
    )
    capped_by_upper_limit: bool = Field(
        json_schema_extra=_facts("狀態", "1", _NO_BASIS_TEXT, _NOT_MEASURED)
    )
    eyring_t60_by_band_s: dict[str, float] = Field(
        json_schema_extra=_facts("時間", "s", "絕對值：Eyring 公式、面積加權平均吸音率算出的秒數")
    )
    room_volume_m3: float = Field(
        json_schema_extra=_facts("體積", "m^3", "絕對值：房三軸長相乘的立方公尺數")
    )
    schroeder_band_count: int = Field(
        ge=0,
        json_schema_extra=_facts("計數", "1", _NO_BASIS_COUNT, _NOT_MEASURED),
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

    center_frequency_hz: float = Field(json_schema_extra=_facts("頻率", "Hz", "絕對值：八度帶中心頻率"))
    fem_energy: float | None = Field(
        json_schema_extra=_facts("能量", "1", _FEM, _EMPTY_WHEN)
    )
    fem_point_count: int = Field(
        ge=0, json_schema_extra=_facts("計數", "1", _NO_BASIS_COUNT, _NOT_MEASURED)
    )
    direct_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    reflected_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    interference_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    late_energy: float = Field(json_schema_extra=_facts("能量", "1", _LATE))
    geometric_energy: float = Field(json_schema_extra=_facts("能量", "1", _MIXED))
    fem_contribution: float = Field(json_schema_extra=_facts("能量", "1", _FEM))
    geometric_contribution: float = Field(json_schema_extra=_facts("能量", "1", _MIXED))
    total_energy: float = Field(json_schema_extra=_facts("能量", "1", _MIXED_TOTAL))
    w_fem: float = Field(json_schema_extra=_facts("權重", "1", _FRACTION))
    w_geo: float = Field(json_schema_extra=_facts("權重", "1", _FRACTION))
    f_s_hz: float = Field(json_schema_extra=_facts("頻率", "Hz", "絕對值：整個報表共用同一個交接頻率"))
    capped_by_upper_limit: bool = Field(
        json_schema_extra=_facts("狀態", "1", _NO_BASIS_TEXT, _NOT_MEASURED)
    )
    t20_s: float | None = Field(
        json_schema_extra=_facts(
            "時間", "s", "絕對值：由同一條晚期衰減曲線的 −5～−25 dB 視窗擬合", _EMPTY_UNLESS
        )
    )
    t20_unavailable_reason: str | None = Field(
        json_schema_extra=_facts("不可估原因", "1", _NO_BASIS_TEXT, _NOT_MEASURED)
    )
    t30_s: float | None = Field(
        json_schema_extra=_facts(
            "時間", "s", "絕對值：由同一條晚期衰減曲線的 −5～−35 dB 視窗擬合", _EMPTY_UNLESS
        )
    )
    t30_unavailable_reason: str | None = Field(
        json_schema_extra=_facts("不可估原因", "1", _NO_BASIS_TEXT, _NOT_MEASURED)
    )


class PointRow(_FactsModel):
    """細軸逐點表一列。宣告順序**不等於**印出來的順序：``--format json`` 那一路照字母排鍵，
    文字那一路（``_point_table``）自己排。

    ``fem_energy`` 的 ``None`` 意思跟頻帶表那一欄同一族：這個頻點沒有有限元素值
    （``w_fem`` 是 0），不必帶原因——它不是「算不出來」，這一張表也沒有原因欄。
    """

    frequency_hz: float = Field(json_schema_extra=_facts("頻率", "Hz", "絕對值：1/24 八度細軸上的頻點"))
    fem_energy: float | None = Field(
        json_schema_extra=_facts("能量", "1", _FEM, _EMPTY_WHEN)
    )
    direct_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    reflected_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    interference_energy: float = Field(json_schema_extra=_facts("能量", "1", _RELATIVE))
    late_energy: float = Field(json_schema_extra=_facts("能量", "1", _LATE))
    scattering: float = Field(
        json_schema_extra=_facts("散射係數", "1", "相對於入射功率的比例（0 到 1）")
    )
    geometric_energy: float = Field(json_schema_extra=_facts("能量", "1", _MIXED))
    w_fem: float = Field(json_schema_extra=_facts("權重", "1", _FRACTION))
    w_geo: float = Field(json_schema_extra=_facts("權重", "1", _FRACTION))
    total_energy: float = Field(json_schema_extra=_facts("能量", "1", _MIXED_TOTAL))


class ReportOutput(_FactsModel):
    """三路接合報表的三張表與 capability 那一行，收成一個可驗的結構。

    這一層只負責形狀與三條規則：頻帶列的中心頻率要遞增不重複、交接下端不准超過上端、
    以及不可估的欄位一定要帶原因（值空必須有原因、有值又不准給原因）。
    """

    capability: CapabilitySection = Field(
        description="能力表那一條本人",
        json_schema_extra=_facts(
            "能力", "1", "沒有基準（是能力表整條記錄，不是量測值）", _NOT_MEASURED
        ),
    )
    top: TopFields = Field(
        description="頂層那幾格",
        json_schema_extra=_facts(
            "頂層總量", "1", "見底下每一欄自己的基準", _NOT_MEASURED
        ),
    )
    bands: tuple[BandRow, ...] = Field(
        description="頻帶表；六個八度帶各一列",
        json_schema_extra=_facts("頻帶列", "1", "見底下每一欄自己的基準", _NOT_MEASURED),
    )
    points: tuple[PointRow, ...] | None = Field(
        default=None,
        description="細軸逐點表；沒有 --points 時整塊省略",
        json_schema_extra=_facts("逐點列", "1", "見底下每一欄自己的基準", _NOT_MEASURED),
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
            if number <= 0.0:
                raise ValueError(
                    f"{cell_where} 必須是正實數阻抗；{unsupported_materials_hint(table)}"
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


def _capability_section(report: object) -> CapabilitySection:
    """報表那一格能力：表上那一條本人；沒查表時狀態是 unchecked、範圍與收據是空的。"""
    record = report.capability.record  # type: ignore[attr-defined]  # expires=2026-12-08 reason=呼叫端已驗過型別，這一支只收報告物件的那一格
    return CapabilitySection(
        frequency_hz=record.frequency_hz if record is not None else (),
        outputs=record.outputs if record is not None else (),
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
    """頻帶列；宣告順序不是印出來的順序（見 :class:`BandRow`）。"""
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
    """細軸逐點列；宣告順序不是印出來的順序（見 :class:`PointRow`）。"""
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

    寫法是逐位元組決定性的（同一份模型跑兩次得到同一個檔），縮排與鍵序跟原本手上那份一致。

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
