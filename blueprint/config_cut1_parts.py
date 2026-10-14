"""設定 case 表的零件：型別、值與操作的小工廠、參數還原及路徑正規化。

``blueprint/config_cut1_cases.py`` 留第 1 刀清單、合併的 MODULES 與存取函式，並再匯出零件；
``blueprint/config_cut2_cases.py`` 放第 2 刀讀檔那 9 支的清單與模組名單。
三支分工是為了避開 style-guard（寫法警衛）第⑤條的單檔 1000 行上限，case 內容與順序不變。
這些零件也由上一代的 Python 載入，只能依賴標準庫與 blueprint，不載入新家引擎。
"""
from __future__ import annotations

import re
from typing import Final, TypedDict

# 一個參數值就是一個普通 dict（不是自訂類別：答案檔用 json 存，寫成普通 dict 才不必
# 為序列化再補一層 representer）。
Tagged = dict[str, object]


class CaseEntry(TypedDict, total=False):
    """一筆 case：穩定的 id ＋ 那一筆的參數（參數值是 :data:`Tagged` 或純量）。

    ``op`` 只有這一刀（讀檔那 9 支）用得到：第 1 刀那些 case 是「拿一組參數去呼叫一支
    函式」，這一刀多了「餵哪一個檔案、要不要先突變它」。沒寫 ``op`` 就是第 1 刀那種形狀。
    """

    id: str
    args: dict[str, Tagged | int | str]
    op: CaseOp


class ModuleSpec(TypedDict):
    """一支模組在這一張表裡的三件事：對應的檔名、要凍結的常數、要跑的 case。"""

    file: str
    constants: list[str]
    cases: list[CaseEntry]


def flt(value: float) -> Tagged:
    """一個浮點參數。"""
    return {"k": "float", "v": value}


def nan() -> Tagged:
    """``float("nan")``（非有限值的守門要驗它）。"""
    return {"k": "nan"}


def plus_inf() -> Tagged:
    """``float("inf")``。"""
    return {"k": "plus_inf"}


def minus_inf() -> Tagged:
    """``float("-inf")``。"""
    return {"k": "minus_inf"}


def true() -> Tagged:
    """布林 ``True``（跟整數 1 是兩件事：有的參數會特別擋布林）。"""
    return {"k": "bool", "v": True}


def false() -> Tagged:
    """布林 ``False``。"""
    return {"k": "bool", "v": False}


def items(*values: Tagged | int | str) -> Tagged:
    """一串值，還原成 ``list``（目前只有「要列出哪幾個預設」那一筆在用）。

    上游有兩處收一串數字，但**容器不一樣**（`SplOutputConfig` 的欄位宣告是
    ``tuple[float, ...]``），而 pydantic 的訊息會把 ``input_value=[nan]`` 與
    ``input_value=(nan,)`` 講成兩句不同的話——所以「是哪一種容器」在這一張表裡是
    看得見的：數字那一串用 :func:`floats`（tuple），其餘用這一支（list）。
    """
    return {"k": "items", "v": list(values)}


def floats(*values: Tagged | float | int) -> Tagged:
    """一串數字，還原成 ``tuple``（`SplOutputConfig.sensitivity_db` 那一格）。"""
    return {"k": "float_tuple", "v": list(values)}


def strs(value: str) -> Tagged:
    """一個字串參數，**明寫成記號**。

    第 1 刀只有純量與幾個小幫手；這一刀有幾筆（`profile="critical"`、
    `mode="production"`）要說清楚「這是一個字串參數」，所以多這一支。
    """
    return {"k": "str", "v": value}


# ── 這一刀（票 #127 後半，讀檔那 9 支）加的小幫手 ─────────────────────────────


class Mutation(TypedDict):
    """一筆 TOML 突變：把真的設定檔讀進來、把其中一段換掉，餵給載入器。

    兩段都是**原文**，不是行號也不是正則：「原文在不在」本身就是一條下限——原文被改掉
    或搬家，這一筆就產生不出來（不是靜靜地跳過）。
    """

    before: str
    after: str


class CaseOp(TypedDict, total=False):
    """這一刀每一筆 case 的執行方式（第 1 刀那些 case 沒有這一格）。

    * ``fn``：要呼叫哪一支載入器（不給就用 case id 的第二段）。
    * ``module``：要載入哪一支新家模組（不給就用 case id 第一段的 ``cut2_`` 後面那個名字）。
    * ``mutations``：依序套用的突變（見 :class:`Mutation`）；真的設定檔讀進來、改過再餵。
    * ``path``：餵給載入器的路徑（``"default"`` ＝ 不傳路徑、走載入器自己的預設；
      ``"missing"`` ＝ 一個不存在的路徑，驗「檔案不存在」那條分支）。
    * ``kwargs``：其他關鍵字參數（例如 ``profile=``、``mode=``）。
    """

    fn: str
    module: str
    mutations: list[Mutation]
    path: str
    kwargs: dict[str, Tagged | int | str]


def engine_module_name(table_name: str) -> str:
    """case 表裡的名字對應到新家的模組名（去掉這一刀加的 ``cut2_`` 前綴）。

    「有哪幾支模組」只寫在這一張表裡，產生器與考卷都走這一支，不各留一份名單。
    """
    return table_name.removeprefix("cut2_")


def missing_path() -> str:
    """「檔案不存在」那幾筆餵進去的路徑。

    刻意是一個**看起來像路徑、但一定不存在**的固定字串（不開暫存檔、不碰真目錄）：
    載入器要炸在開檔那一刻，這條路徑只要不存在就驗得到。產生器與考卷共用這一支，
    所以兩邊餵的是同一個字串；比對訊息時只要把這一段正規化掉。
    """
    return "/tmp/aosr-cut2-missing/nope.toml"


def tmp_root() -> str:
    """這一刀的探針檔寫在哪。

    產生器與考卷都把突變過的 TOML 寫在這裡、都比到同一個固定位置；錯誤訊息裡那一段
    路徑不逐位元比（暫存目錄每一跑都不一樣），兩邊都把它換成 ``<tmp>`` 再比。
    """
    return "/tmp/aosr-cut2-probe"


# 錯誤訊息裡的**絕對路徑**一律換成這個記號再比。理由：暫存目錄每一跑都不一樣，
# 而真的設定檔那一份的絕對路徑**在每一棵 checkout 上也不一樣**（CI 的 runner 跟本機
# 不同）。把「repo 在哪、暫存在哪」抽掉之後，留在訊息裡的就是契約那一段：哪一支載入器、
# 哪一個欄位、哪一種錯。產生器與考卷兩邊共用這一支，不各寫一份。
#
# 這一支換的是**已知的兩個根**（repo 根與暫存根），不是用正則去猜「哪裡像路徑」：
# 猜的那一種會把訊息裡本來就有的相對路徑（`src/aosr/config/data/x.toml`）留下來，
# 於是答案檔在另一棵 checkout 上比不過——那是這一條第一版真的踩到的坑。
NORMALISED_PATH: Final[str] = "<path>"

# 一個**絕對路徑**：從 `/` 開始（前面不可以是斜線或英數字——`file:///x` 那種網址
# 例外，見檔尾），一路吃到最後一個斜線（含）。故意貪心——多收一段只是把「在哪一棵樹、
# 哪一個暫存目錄」收掉，留下來的尾段（`scoring_v2.toml`）才是契約。
_ABSOLUTE_PREFIX = re.compile(r"(?<![\w/])/(?:[^\s'\":,()\[\]]*/)+")


def normalise_paths(message: str) -> str:
    """把訊息裡的**絕對路徑前綴**換成 ``<path>/``。

    這是訊息本身的純函式，不吃「這一跑在哪一棵樹跑」這種外部狀態——產生器與考卷在
    不同的 checkout、不同的暫存目錄上跑，套完這一支之後留下的字串逐位元相同。
    相對路徑（`src/aosr/config/data/x.toml`）刻意不動：那一種不是「在哪一棵樹」，
    它是載入器訊息本身的一部分。

    **它連 pydantic 訊息裡的網址前綴一起收掉**（`https://errors.pydantic.dev/2.13/v/finite_number`
    變成 `https:<path>/finite_number`）。那是刻意的：那個前綴帶著庫的版本號，換一個
    pydantic 版本就漂一次，而「哪一種錯」已經由 `[type=…]` 那一格比到了（見
    `_config_answers.validate_message`）。**不要把這一段當成漏抓去修 regex**——修了要
    重生答案檔，而現在的行為才是對的。
    """
    return _ABSOLUTE_PREFIX.sub(NORMALISED_PATH + "/", message)


# 這一刀的載入器有兩種共同的入口形狀，收成兩支小工廠（產生器與考卷都照同一組旗標跑）。
def perimeter_default(
    fn: str,
    *,
    mutations: list[Mutation] | None = None,
    kwargs: dict[str, Tagged | int | str] | None = None,
) -> CaseOp:
    """一筆「餵真的設定檔、不傳路徑（走載入器自己的預設）」的 case。"""
    op: CaseOp = {"fn": fn, "path": "default"}
    if mutations is not None:
        op["mutations"] = mutations
    if kwargs is not None:
        op["kwargs"] = kwargs
    return op


def perimeter_missing(fn: str) -> CaseOp:
    """一筆「餵一個不存在的路徑」的 case（驗各支載入器的開檔失敗分支）。"""
    return {"fn": fn, "path": "missing"}


def perimeter_mutation(fn: str, before: str, after: str) -> CaseOp:
    """一筆「把真的設定檔改一格再餵」的 case。"""
    return {"fn": fn, "path": "default", "mutations": [Mutation(before=before, after=after)]}


def resolve(value: object) -> object:
    """把一個參數值還原成 Python 值（產生器餵給上一代、考卷餵給新家用同一支）。

    不是記號（例如 ``0``、``"y0"``）就原樣回傳。記號的種類看不懂就當場炸——
    看不懂的參數靜靜地餵下去，比炸掉糟。
    """
    if not isinstance(value, dict) or "k" not in value:
        return value
    kind = value["k"]
    if kind == "float":
        return _as_number(value.get("v"), "float")
    if kind == "bool":
        return bool(value.get("v"))
    if kind == "str":
        raw_text = value.get("v")
        if not isinstance(raw_text, str):
            raise AssertionError(f"str 記號裡不是字串：{raw_text!r}")
        return raw_text
    if kind == "nan":
        return float("nan")
    if kind == "plus_inf":
        return float("inf")
    if kind == "minus_inf":
        return float("-inf")
    if kind in ("items", "float_tuple"):
        raw = value["v"]
        if not isinstance(raw, list):
            raise AssertionError(f"{kind} 記號的內容不是一串東西：{raw!r}")
        resolved = [resolve(item) for item in raw]
        return tuple(resolved) if kind == "float_tuple" else resolved
    raise AssertionError(f"不認識的參數記號：{kind!r}")


def _as_number(raw: object, where: str) -> float:
    """把一個記號裡的數字收窄成 ``float``（不是數字就當場炸）。"""
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise AssertionError(f"{where} 記號裡不是數字：{raw!r}")
    return float(raw)
