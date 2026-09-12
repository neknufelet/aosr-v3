"""票 #134 第一候選（材料那一塊）case 表的**零件**：型別、值的記號、步驟的小工廠。

這一支只放零件；「有哪些 case」住 :mod:`blueprint.materials_cut1_cases`（那一支是單一
來源，產生器與考卷都照它跑），清單本身住兩支 ``*_cases.py``。拆成幾支檔是
``style-guard`` 第⑤條的單檔行數上限，拆法照模組走，不是照行數硬切。

**跟 config 那一張表不一樣的地方。** 這一塊的東西是 JAX 的陣列與 pytree（會被
``jax.tree_util`` 拆開再組回去的資料結構），而且 ``MaterialRegistry`` 是**有狀態**的：
「註冊兩次會怎樣」不是一次呼叫講得完的。所以這裡的一筆 case 是一小段**步驟**
（:class:`Step`），在一個只活這一筆的環境裡依序跑；步驟中途炸掉就記「第幾步、炸什麼」。

**這一支不准 import donor 或新家的任何東西，也不准 import JAX。** 它會被
``style-guard``／``type-guard``／``uv-single-entrypoint`` 掃，而且兩邊（donor 的直譯器與
新家的直譯器）都要 import 得到它。真正碰模組的是 :mod:`blueprint.materials_cut1_probe`。

"""
from __future__ import annotations

from typing import Final, TypedDict

# 一個值就是一個普通 dict（不是自訂類別：答案檔用 json 存，寫成普通 dict 才不必為序列化
# 再補一層 representer）。
Tagged = dict[str, object]


class Step(TypedDict, total=False):
    """一筆 case 裡的一個步驟。

    * ``do``：這一步做什麼（``call``／``method``／``read``／``set``／``flatten``／
      ``map``／``jit``／``sig``）。
    * ``fn``：``call``／``sig`` 要用的東西，寫成 ``<表上的模組>.<名字>[.<名字>]``
      （例如 ``response.FrequencyAxis.from_hz``）。跨模組是刻意的：註冊表那幾筆要先用
      ``response`` 的頻率軸。
    * ``on``：要對環境裡哪一個名字動手（``method``／``read``／``set``／``flatten``／
      ``map``／``jit``）。
    * ``name``：方法名或屬性名。
    * ``args``：關鍵字參數（值是 :data:`Tagged` 或純量）。這一張表**只用關鍵字**——
      「哪幾個參數只能用位置傳」由 ``sig`` 那一種步驟自己釘住，不靠呼叫方式間接驗。
    * ``value``：``set`` 要寫進去的值（驗 frozen：寫得進去就是契約鬆了）。
    * ``to``：把這一步的結果存進環境的哪一個名字（不給就丟掉）。
    """

    do: str
    fn: str
    on: str
    name: str
    args: dict[str, Tagged | int | str]
    value: Tagged | int | str
    to: str


class CaseEntry(TypedDict):
    """一筆 case：穩定的 id ＋ 要跑的步驟 ＋ 要記進答案檔的那幾個名字。

    ``report`` 是環境裡的名字清單。步驟全部跑完就記那幾個名字的值；中途炸掉就記
    「第幾步、什麼例外、訊息是什麼」——**例外也是行為**，不是失敗。
    """

    id: str
    steps: list[Step]
    report: list[str]


class ModuleSpec(TypedDict):
    """一支模組在這張表裡的幾件事。

    * ``donor``：上一代那一支的 import 路徑（產生器在唯讀 donor 樹上載入它）。
    * ``engine``：新家那一支的 import 路徑（考卷載入它）。
    * ``file``：donor 那一支的相對路徑（對得回 ``core-public-inventory.json``）。
    * ``constants``：要逐值凍結的公開常數。
    * ``aliases``：要凍結的公開型別別名（``Literal[...]``，記 ``typing.get_args`` 的結果）。
    * ``coverage``：公開符號 → 判它的那幾筆（指不出任何一筆就是沒有裁判）。
    * ``cases``：要跑的 case。
    """

    donor: str
    engine: str
    file: str
    constants: list[str]
    coverage: dict[str, list[str]]
    aliases: list[str]
    cases: list[CaseEntry]


# ── 值的記號（產生器與考卷共用同一支 :func:`resolve`）───────────────────────
def flt(value: float) -> Tagged:
    """一個浮點值。"""
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
    """布林 ``True``（跟整數 1 是兩件事）。"""
    return {"k": "bool", "v": True}


def false() -> Tagged:
    """布林 ``False``。"""
    return {"k": "bool", "v": False}


def strs(value: str) -> Tagged:
    """一個字串值。"""
    return {"k": "str", "v": value}


def none() -> Tagged:
    """``None``（「不給散射係數」那幾筆要餵它，跟「沒傳這個參數」是兩件事）。"""
    return {"k": "none"}


def cplx(real: float, imag: float) -> Tagged:
    """一個複數（材料的表面阻抗是複數，這一格是它的契約）。"""
    return {"k": "complex", "re": real, "im": imag}


def seq(*values: Tagged | float | int | str) -> Tagged:
    """一串值，還原成 ``list``。"""
    return {"k": "seq", "v": list(values)}


def ref(name: str) -> Tagged:
    """環境裡某一個名字（前面某一步 ``to`` 存進去的東西）。"""
    return {"k": "ref", "v": name}


# ── 步驟的小工廠（讓下面那張表讀起來是人話）─────────────────────────────────
def call(fn: str, to: str, **args: Tagged | int | str) -> Step:
    """呼叫 ``fn``（全部走關鍵字），結果存進 ``to``。"""
    return {"do": "call", "fn": fn, "args": dict(args), "to": to}


def method(on: str, name: str, to: str, **args: Tagged | int | str) -> Step:
    """對環境裡的 ``on`` 叫 ``name`` 這個方法，結果存進 ``to``。"""
    return {"do": "method", "on": on, "name": name, "args": dict(args), "to": to}


def read(on: str, name: str, to: str) -> Step:
    """讀 ``on`` 的 ``name`` 屬性（屬性本身可能就是守門，炸掉也記下來）。"""
    return {"do": "read", "on": on, "name": name, "to": to}


def write(on: str, name: str, value: Tagged | int | str) -> Step:
    """把 ``value`` 寫進 ``on`` 的 ``name``（驗 frozen：**寫得進去就是契約鬆了**）。"""
    return {"do": "set", "on": on, "name": name, "value": value}


def flatten(on: str, to: str) -> Step:
    """把 ``on`` 拆成 pytree 的葉子與結構（記葉子個數與每一片葉子的值）。"""
    return {"do": "flatten", "on": on, "to": to}


def tree_double(on: str, to: str) -> Step:
    """``jax.tree_util.tree_map(lambda x: x * 2, on)``——葉子要動、靜態中介資料不准動。"""
    return {"do": "map", "on": on, "to": to}


def through_jit(on: str, to: str) -> Step:
    """把 ``on`` 當參數穿過一次 ``jax.jit``（pytree 進得去、出得來）。"""
    return {"do": "jit", "on": on, "to": to}


def size_of(on: str, to: str) -> Step:
    """``len(on)``——``__len__`` 是公開行為，要真的用 ``len()`` 走一遍，不是去叫那個名字。"""
    return {"do": "len", "on": on, "to": to}


def contains(on: str, value: Tagged | int | str, to: str) -> Step:
    """``value in on``——``__contains__`` 同上，用 ``in`` 走一遍。"""
    return {"do": "has", "on": on, "value": value, "to": to}


def signature(fn: str, to: str) -> Step:
    """記下 ``fn`` 的簽章（參數名、位置／關鍵字、有沒有預設、預設是什麼）。

    刻意**不記標註文字**：型別名字可以搬家（決策紙說結構隨便改），但「哪幾個參數必填、
    預設值是什麼、能不能用位置傳」是公開契約，一放寬就是行為變了。
    """
    return {"do": "sig", "fn": fn, "to": to}


# ── 幾個常用的前置步驟 ──────────────────────────────────────────────────────
def axis3(to: str = "axis") -> list[Step]:
    """三點頻率軸（100／200／400 Hz）——舊考卷 ``test_response.py`` 用的就是這一組。"""
    return [
        call(
            "response.FrequencyAxis.from_hz",
            to,
            freqs_hz=seq(flt(100.0), flt(200.0), flt(400.0)),
            resolution=strs("custom"),
        )
    ]


def impedance3(to: str = "mat", axis_name: str = "axis", material_id: str = "m1") -> list[Step]:
    """舊考卷 ``test_response.py`` 的 ``_make()``：三點複數阻抗建出來的材料回應。"""
    return [
        call(
            "response.MaterialResponse.from_impedance",
            to,
            Z_surface=seq(cplx(1.0, 0.5), cplx(2.0, -1.0), cplx(3.0, 0.0)),
            freq_axis=ref(axis_name),
            material_id=strs(material_id),
            boundary_model=strs("local_impedance"),
        )
    ]


RESPONSE_META: Final[list[str]] = [
    "material_id",
    "boundary_model",
    "is_locally_reacting",
    "Z_surface",
    "scattering_coeff",
]


def reads(on: str, names: list[str], prefix: str) -> list[Step]:
    """一口氣讀好幾個屬性（每一個各存一格，炸掉的那一個就是這一筆的結果）。"""
    return [read(on, name, f"{prefix}_{name}") for name in names]


def report_names(prefix: str, names: list[str]) -> list[str]:
    """:func:`reads` 那一組存進環境的名字。"""
    return [f"{prefix}_{name}" for name in names]


def resolve(value: object) -> object:
    """把表上的一個記號還原成 Python 值（產生器與考卷共用這一支）。

    ``ref`` 不在這裡還原——它要看那一筆 case 的環境，由
    :mod:`blueprint.materials_cut1_probe` 處理。
    """
    if not isinstance(value, dict):
        return value
    kind = value.get("k")
    if kind == "float":
        return float(_number(value.get("v")))
    if kind == "nan":
        return float("nan")
    if kind == "plus_inf":
        return float("inf")
    if kind == "minus_inf":
        return float("-inf")
    if kind in ("bool", "str"):
        return value.get("v")
    if kind == "none":
        return None
    if kind == "complex":
        return complex(_number(value.get("re")), _number(value.get("im")))
    if kind == "seq":
        items = value.get("v")
        if not isinstance(items, list):
            raise ValueError(f"seq 那一格不是一串東西：{value!r}")
        return [resolve(item) for item in items]
    raise ValueError(f"case 表裡有不認識的記號：{value!r}")


def _number(raw: object) -> float:
    """把一格收窄成數字（不是數字就當場炸——我沒看懂就不出結論）。"""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"這一格不是數字：{raw!r}")
    return float(raw)
