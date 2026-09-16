"""報表契約每一欄的四件事：詞彙、界限，與把它們寫進 schema 的小工具（票 #316）。

**為什麼獨立一支。** 契約模型那一支（:mod:`aosr.physics.report_io`）長到頂著寫法警衛的
行數上限；這一段是**詞彙**——參考基準怎麼說、哪一族算「不是量測值」、數值界限是哪幾個
數字——跟欄位形狀是兩件事，分開放讀的人才找得到。

**界限只有一份數字。** 下面那三個常數同時餵驗證函式與 ``model_json_schema()`` 匯出的
格式檔：格式檔說得出的允許範圍，就是後端真的擋的那一份。兩邊各寫一次，就是「前端說
合法、送到後端被拒收」的開始。

**參考基準怎麼判的（每一條都回得到程式或決策紙；沒把握的照實寫在值裡）。**

* 幾何路的**早期**能量欄是相對量：``docs/decisions/stage-five-ism-totals-and-energy.md`` 定
  「直達能量是直達那一條壓力的模平方」，而 ``path_pressure`` 沒有音源強度因子
  （``src/aosr/physics/amplitude.py``：``(1/dist)·refl·exp(−iωτ)``），距離單位是公尺
  －－所以是「單位振幅點源、距離 1 公尺處壓力為 1」的相對基準，不是帕（Pa）。
* 反射／干涉是同一條基準上的**分項**（反射為一到三階壓力的同調和、干涉為直達與反射的交叉項），
  可以為負；不是各自的絕對值。
* 有限元素欄是 ``4π`` 點音源的同一個 ``|p(1 m)| = 1`` 家族：``blueprint/fem_fenics_problem.json``
  的 ``physics.source_strength = "4*pi"``，與 ``src/aosr/physics/fem_helmholtz.py`` 的
  ``POINT_SOURCE_STRENGTH`` 同值。整包的共用基準寫在 ``src/aosr/config/source_reference.py``
  的檔頭（單位振幅點源、``|p(1 m)| = 1``）。**它與幾何路的逐位對齊沒有機器在守、這一版
  沒有對過**（交接檔「要查」那一條記著同一件事），所以欄上寫的是「對齊未驗」——**不是**
  說兩路的基準定義不同。
* 晚期能量帶 ``4π``，而那個 4π 是**換算到共用基準**的因子（``source_reference`` 檔頭：
  手冊的擴散場密度 ``4/R`` 以單位聲源功率為基準，換到單位振幅點源要乘 4π），不是另一把尺：``src/aosr/physics/late_energy.py`` 的 ``_exact_raw_energy``
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
  ``fem_contribution + geometric_contribution``）再把有限元素那一路加進來。三路名義上
  共用同一個基準，**但跨方法的數值對齊沒有機器在守、這一版沒有對過**，所以這三欄寫的是
  「相加而來、對齊未驗」，也不是已校準的絕對聲壓級。
* 權重、散射無因次；聲速、密度、阻抗、體積、時間與頻率是絕對值（有量綱）；計數與欄名、
  狀態、原因這些格子走上面那一族「沒有基準（只是⋯⋯，不是量測值）」。
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

from typing import Final, NamedTuple

from aosr.geometry.shoebox import Wall
from pydantic.config import JsonDict


# 後端會擋什麼、格式檔說得出什麼，共用這一份界限數字；考卷
# ``test_schema_bounds_are_the_same_numbers_the_validators_use`` 咬住兩邊同一個數。
POSITIVE_EXCLUSIVE_MINIMUM: Final[float] = 0.0
SCATTERING_MINIMUM: Final[float] = 0.0
SCATTERING_MAXIMUM: Final[float] = 1.0


class FieldFacts(NamedTuple):
    """一欄的四件事：物理量、單位、參考基準、有效狀態。"""

    quantity: str
    unit: str
    reference: str
    validity: str


def facts(
    quantity: str,
    unit: str,
    reference: str,
    validity: str = "可估",
) -> JsonDict:
    """組出 ``json_schema_extra``；四格一個都不能少。

    預設那一格是「可估」（這一欄有值就是量到的東西）；不是數值的那些格子自己指名
    ``NOT_MEASURED``（那一格根本不是估出來的東西）；會出現空值的兩族另外指名
    ``EMPTY_UNLESS``（值算不出來）或 ``EMPTY_WHEN``（這一格沒有有限元素頻點）。
    預設值必須等於 :data:`ESTIMABLE`（同一件事不准有兩個字面）；
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


# ── 參考基準的文字；每一條的出處寫在檔頭，沒把握的照實寫 ────────────────────────
RELATIVE: Final[str] = (
    "單位振幅點源、距離 1 公尺處壓力為 1 的相對能量（|p(1 m)|=1；路徑壓力不含音源強度"
    "因子，欄位不是帕）"
)
FEM: Final[str] = (
    "4π 點音源、|p(1 m)|=1 的同一家族（題目檔 source_strength=\"4*pi\"）；與幾何路的逐位"
    "對齊沒有機器在守，這一版沒有對過"
)
LATE: Final[str] = (
    "晚期混響能量：late_energy._exact_raw_energy 的 raw_energy（＝4π × 4 × 平均反射場）"
    "再乘一次 _eyring_ratio(alpha_bar) 比值（late_energy.py 的 late_reverberant_energy），"
    "頻帶欄再對帶內細軸點取平均——不是 raw_energy 本人"
)
# 混合基準：掛這一格的三個欄位（頻帶表的 geometric_energy 與 geometric_contribution、
# 逐點表的 geometric_energy）都不是單一幾何基準，而是把兩路各自基準的東西加在一起。
# 各路名義上按同一個基準正規化（單位振幅點源、|p(1 m)|=1，見 src/aosr/config/source_reference.py）：
# 晚期擴散場本來以單位聲源功率表示，乘 4π 正是換算到那個共用基準的因子，不是另一把尺。
# 真正還沒有的是跨方法的數值對齊，所以下面兩句寫的是「意圖一致、對齊未驗」。
MIXED: Final[str] = (
    "幾何路早期項與晚期項相加；兩者名義上同一個單位振幅點源基準（晚期那一路的 4π 是"
    "換算到該基準的因子），跨方法的數值對齊沒有機器在守、這一版沒有對過"
)
MIXED_TOTAL: Final[str] = (
    "有限元素那一路與幾何那一路相加；三路名義上同一個單位振幅點源基準，跨方法的數值"
    "對齊沒有機器在守、這一版沒有對過。這一欄不是已校準的絕對聲壓級"
)
NO_BASIS_TEXT: Final[str] = "沒有基準（只是文字或狀態，不是量測值）"
NO_BASIS_COUNT: Final[str] = "沒有基準（只是計數，不是量測值）"
NO_BASIS_NAMES: Final[str] = "沒有基準（只是欄名清單，不是量測值）"
NO_BASIS_RANGE: Final[str] = "沒有基準（只是能力表上宣告的頻率範圍兩個端點，不是量測值）"
# 可估的那一格與不是量測的那一格各自實話；值空時誰帶原因跟在後面。
def positive_facts(quantity: str, unit: str, reference: str) -> JsonDict:
    """四件事加上「必須大於零」——那一格在格式檔裡就說得出來，不必讀後端才知道。"""
    extra = facts(quantity, unit, reference)
    extra["exclusiveMinimum"] = POSITIVE_EXCLUSIVE_MINIMUM
    return extra


def wall_object_facts(
    quantity: str, unit: str, reference: str, *, allow_zero: bool
) -> JsonDict:
    """六面牆那一格的四件事加上形狀：六個牆名都必填、值有上下界、不認識的牆名不收。

    牆名從 :meth:`Wall.all` 來、界限從上面那幾個常數來，所以格式檔說的跟
    :func:`_wall_numbers` 擋的是同一份東西，不是另外手寫的第二份規格。
    """
    cell: JsonDict = {"type": "number"}
    if allow_zero:
        cell["minimum"] = SCATTERING_MINIMUM
        cell["maximum"] = SCATTERING_MAXIMUM
    else:
        cell["exclusiveMinimum"] = POSITIVE_EXCLUSIVE_MINIMUM
    names = [wall.wall_name() for wall in Wall.all()]
    extra = facts(quantity, unit, reference)
    extra["properties"] = {name: dict(cell) for name in names}
    extra["required"] = list(names)
    extra["additionalProperties"] = False
    return extra


# ``ESTIMABLE`` 就是 ``facts`` 的預設值那一個字面，而 ``facts`` 定義在上面那一區、
# 拿不到這裡的常數，所以預設值寫字面、兩處靠考卷
# ``test_facts_default_validity_is_the_estimable_constant`` 咬住。
ESTIMABLE: Final[str] = "可估"
# 不是量測值的那些格子：文字、狀態旗標、計數、收據、欄名、表上宣告的值（頻率範圍），
# 以及本身只是容器的那幾欄（``capability``／``top``／``bands``／``points``）。
# 它們的「有效狀態」不是「估不估」，寫「可估」等於說這些格子是量出來的。
NOT_MEASURED: Final[str] = (
    "不是估出來的量測值（這一格是文字、狀態、容器、表上宣告的值或計數，不是估出來的量）"
)
# 兩族「空」：fem_energy 空＝這一帶沒有有限元素頻點（不必帶原因）；
# T20／T30 空＝值算不出來（必須帶原因）。說明與 validity 同一句話。
EMPTY_WHEN: Final[str] = (
    "這一格可能是空的：空＝這一格沒有有限元素頻點，不必帶原因"
)
EMPTY_UNLESS: Final[str] = "可估（空＝值算不出來，必須帶原因；有值就不准再給原因）"
FRACTION: Final[str] = "無因次；功率互補權重，兩欄相加為 1"
# 交接頻率那一格的說明要把「哪兩個中頻帶」從產品設定讀，不把那些數字抄進這個檔。
F_S_REFERENCE: Final[str] = (
    "絕對值：由產品設定 ``three_lane_crossover.SCHROEDER_T60_BANDS_HZ`` 那兩個中頻帶的"
    "Eyring T60 與房間體積算出的交接頻率"
)
