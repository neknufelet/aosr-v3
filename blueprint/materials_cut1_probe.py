"""票 #134 第一候選的探針：**把一筆 case 真的跑一遍**，兩邊共用同一支。

產生器（在唯讀 donor 樹上跑上一代那三支）與考卷（在新家跑同名的三支）走的是**這一支**，
不是各寫一份。票 #127 那一刀的教訓就在這裡：產生器有自己的 ``loader_block``、考卷有自己的
``loader_case_run``，兩份形狀要靠人維持一致——一邊改了、另一邊沒改，比出來的差異是「兩支
跑法不一樣」，不是「新家跟上一代不一樣」。這一支把「怎麼跑」收成一份，兩邊只差一件事：
**去哪一個套件拿模組**（一個 ``lookup`` 函式）。

**這一支不准靜態 import JAX 或 NumPy。** 它會被 ``type-guard`` 的 mypy 嚴格模式掃，而
新家今天還沒有那兩個依賴；而且 donor 那一邊的直譯器與新家這一邊不是同一個環境。真的要用
的時候走 :func:`_module`（執行期才載入），陣列一律**看形狀不看型別**（有 ``dtype`` 與
``shape`` 就當陣列）。

**記下來的東西。** 一筆 case 跑完記 ``values``（那幾個名字各自的值），中途炸掉記
``raised``（第幾步、什麼例外、訊息）——**例外是行為，不是失敗**。值走 :func:`encode`
編成帶型別的記號：浮點存 ``float.hex()``（精確、可逐位元還原），複數分實部虛部，陣列連
``dtype`` 與 ``shape`` 一起記（``float32`` 變 ``float64`` 是行為變了），pytree 記葉子個數、
每一片葉子、以及組回去之後的整個物件（靜態中介資料掉了就看得出來）。
"""
from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Container, Sized
from dataclasses import fields, is_dataclass
from types import ModuleType
from typing import Final, get_args

from blueprint import materials_cut1_steps as steps

# 答案檔的形狀版本。讀的人與寫的人都要對得上，對不上就當場炸（不是「大概能讀」）。
ANSWER_SCHEMA: Final[int] = 1

# 一個模組解析器：表上的名字（``response``）換成真的模組物件。
Lookup = Callable[[str], ModuleType]


def _module(name: str) -> ModuleType:
    """執行期才載入一個套件（``jax``／``jax.tree_util``）。"""
    return importlib.import_module(name)


# ── 編碼：把一個值變成可以 JSON 化、又能逐位元比對的記號 ─────────────────────
def encode(value: object) -> dict[str, object]:
    """把一個值編成帶型別的記號。

    ``bool`` 要在 ``int`` 之前判（Python 的 ``True`` 是一個 ``int``）；陣列要在
    dataclass 之前判（有些陣列型別也被登記成 dataclass）。認不出來的東西**當場炸**，
    不留一個「大概是這樣」的記號——那種記號會讓兩邊比出假的相同。

    **容器的種類是行為，不是寫法（第一版實測到的三個碰撞）。**
    ``encode([1]) == encode((1,))``、``encode({1}) == encode(frozenset({1}))``、
    ``encode({1: "x"}) == encode({"1": "x"})`` 第一版全部為真——也就是說
    ``MaterialRegistry.ids()`` 從 ``list`` 變 ``tuple``、``SOURCE_KINDS`` 從
    ``frozenset`` 變 ``set``、字典的鍵從整數變字串，裁判一律看不見。公開容器的種類是
    呼叫端看得到的行為（``ids()`` 的回傳值能不能 ``.append``、``SOURCE_KINDS`` 能不能被
    改），所以這裡各記一格 ``container``；字典改記**成對的清單**，鍵自己也走一次
    :func:`encode`（鍵的型別一起釘住）。控制組在
    ``tests/engine/test_materials_judge_control.py``：先證明舊寫法擋不住，再證明現在擋得住。
    """
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "v": value}
    if isinstance(value, int):
        return {"kind": "int", "v": value}
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, complex):
        return {"kind": "complex", "re": value.real.hex(), "im": value.imag.hex()}
    if isinstance(value, str):
        return {"kind": "str", "v": value}
    if isinstance(value, (frozenset, set)):
        return {
            "kind": "set",
            "container": type(value).__name__,
            "items": [encode(item) for item in sorted(value, key=repr)],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "seq",
            "container": type(value).__name__,
            "items": [encode(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "map",
            "pairs": [{"key": encode(key), "value": encode(item)} for key, item in value.items()],
        }
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return _encode_array(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "object",
            "type": type(value).__name__,
            "fields": {item.name: encode(getattr(value, item.name)) for item in fields(value)},
        }
    if isinstance(value, type):
        return {"kind": "type", "name": value.__name__}
    raise TypeError(f"這個值編不出來：{type(value).__name__}")


def _encode_array(value: object) -> dict[str, object]:
    """一個陣列：``dtype``、``shape``、以及**逐格**的值。

    ``dtype`` 與 ``shape`` 跟值一樣是契約：``complex64`` 變 ``float32`` 是行為變了
    （相位掉了），``(3,)`` 變 ``(1,)`` 也是。這一格不比記憶體位址、不比哪一家的陣列型別
    ——新家用哪一個函式庫的陣列不是這張票要釘的東西。
    """
    dtype = getattr(value, "dtype")
    shape = getattr(value, "shape")
    listed = getattr(value, "tolist")()
    return {
        "kind": "array",
        "dtype": str(dtype),
        "shape": [int(item) for item in shape],
        "items": [encode(item) for item in _flat(listed)],
    }


def _flat(value: object) -> list[object]:
    """把巢狀的 list 攤平成一串純量（``shape`` 已經另外記了，這裡只記值）。"""
    if not isinstance(value, list):
        return [value]
    out: list[object] = []
    for item in value:
        out.extend(_flat(item))
    return out


# ── 跑一筆 case ─────────────────────────────────────────────────────────────
def run_case(lookup: Lookup, case: steps.CaseEntry) -> dict[str, object]:
    """把一筆 case 的步驟依序跑完，回傳那一筆的結果格。

    跑完 → ``{"values": {名字: 記號}}``；中途炸掉 → ``{"raised": {"step": 第幾步,
    "type": 例外種類, "message": 訊息}}``。**第幾步也要記**：同一個例外在不同步炸，
    是兩件不同的事（守門提早了或晚了）。
    """
    env: dict[str, object] = {}
    for index, step in enumerate(case["steps"]):
        try:
            _run_step(lookup, env, step)
        except Exception as exc:  # noqa: BLE001  # expires=2026-12-08 reason=這一格要記錄「炸什麼」而不是讓它往外炸，例外的種類正是被比的東西
            return {"raised": {"step": index, "type": type(exc).__name__, "message": str(exc)}}
    return {"values": {name: encode(_fetch(env, name)) for name in case["report"]}}


def _fetch(env: dict[str, object], name: str) -> object:
    """從環境裡拿一個名字（沒有就當場炸——report 寫了卻沒存進去是 case 表壞了）。"""
    if name not in env:
        raise KeyError(f"這一筆 case 的 report 要 {name!r}，但沒有任何一步存過它")
    return env[name]


def _value(env: dict[str, object], raw: object) -> object:
    """把一個參數還原成 Python 值；``ref`` 從環境裡拿。"""
    if isinstance(raw, dict) and raw.get("k") == "ref":
        name = raw.get("v")
        if not isinstance(name, str):
            raise ValueError(f"ref 那一格不是名字：{raw!r}")
        return _fetch(env, name)
    return steps.resolve(raw)


def _args(env: dict[str, object], step: steps.Step) -> dict[str, object]:
    """一個步驟的關鍵字參數，整格還原。"""
    return {name: _value(env, raw) for name, raw in (step.get("args") or {}).items()}


def _target(lookup: Lookup, path: str) -> object:
    """``<表上的模組>.<名字>[.<名字>]`` 解析成真的東西。"""
    head, _, rest = path.partition(".")
    obj: object = lookup(head)
    if not rest:
        return obj
    for part in rest.split("."):
        obj = getattr(obj, part)
    return obj


def _run_step(lookup: Lookup, env: dict[str, object], step: steps.Step) -> None:
    """跑一個步驟，把結果（如果有 ``to``）存進環境。"""
    verb = step["do"]
    if verb == "call":
        _store(env, step, _invoke(_target(lookup, step["fn"]), _args(env, step)))
        return
    if verb == "method":
        bound = getattr(_fetch(env, step["on"]), step["name"])
        _store(env, step, _invoke(bound, _args(env, step)))
        return
    if verb == "read":
        _store(env, step, getattr(_fetch(env, step["on"]), step["name"]))
        return
    if verb == "set":
        setattr(_fetch(env, step["on"]), step["name"], _value(env, step.get("value")))
        return
    if verb == "len":
        _store(env, step, len(_sized(_fetch(env, step["on"]))))
        return
    if verb == "has":
        _store(env, step, _value(env, step.get("value")) in _container(_fetch(env, step["on"])))
        return
    if verb == "flatten":
        _store(env, step, _flatten(_fetch(env, step["on"])))
        return
    if verb == "map":
        _store(env, step, _tree_double(_fetch(env, step["on"])))
        return
    if verb == "jit":
        _store(env, step, _through_jit(_fetch(env, step["on"])))
        return
    if verb == "sig":
        _store(env, step, _signature(_target(lookup, step["fn"])))
        return
    raise ValueError(f"case 表裡有不認識的步驟：{verb!r}")


def _store(env: dict[str, object], step: steps.Step, value: object) -> None:
    """把一步的結果存進環境（沒寫 ``to`` 就丟掉）。"""
    name = step.get("to")
    if name is not None:
        env[name] = value


def _invoke(target: object, args: dict[str, object]) -> object:
    """呼叫一個東西（全部走關鍵字；不可呼叫就當場炸）。"""
    if not callable(target):
        raise TypeError(f"這一步要呼叫的東西不可呼叫：{target!r}")
    return target(**args)


def _sized(value: object) -> Sized:
    """收窄成「量得出長度的東西」（量不出來就當場炸）。"""
    if not isinstance(value, Sized):
        raise TypeError(f"這一步要量長度，但它沒有 __len__：{type(value).__name__}")
    return value


def _container(value: object) -> Container[object]:
    """收窄成「問得出 in 的東西」（問不出來就當場炸）。"""
    if not isinstance(value, Container):
        raise TypeError(f"這一步要問 in，但它沒有 __contains__：{type(value).__name__}")
    return value


def _flatten(value: object) -> dict[str, object]:
    """把一個 pytree 拆開再組回去：記葉子個數、每一片葉子、組回去的整個物件。

    三格缺一不可：只記葉子數，葉子的值換掉看不出來；只記葉子，靜態中介資料（材料 id、
    邊界模型）掉了看不出來；只記組回去的物件，葉子被拆成幾片看不出來。
    """
    tree_util = _module("jax.tree_util")
    leaves, treedef = tree_util.tree_flatten(value)
    return {
        "leaf_count": len(leaves),
        "leaves": list(leaves),
        "rebuilt": tree_util.tree_unflatten(treedef, leaves),
    }


def _tree_double(value: object) -> object:
    """``tree_map(lambda x: x * 2, value)``——葉子要變兩倍，靜態中介資料不准動。"""
    tree_util = _module("jax.tree_util")
    return tree_util.tree_map(_doubled, value)


def _doubled(leaf: object) -> object:
    """一片葉子乘二（寫成具名函式，``type-guard`` 才看得到它的標註）。"""
    return leaf * 2  # type: ignore[operator]  # expires=2026-12-08 reason=葉子是執行期才知道的陣列型別，這裡要的就是「它乘得起來」


def _through_jit(value: object) -> object:
    """把一個 pytree 當參數穿過一次 ``jax.jit``（進得去、出得來、內容不變）。"""
    jax = _module("jax")
    return jax.jit(_identity)(value)


def _identity(value: object) -> object:
    """原封不動回傳（``jit`` 的被編譯函式）。"""
    return value


def _signature(target: object) -> dict[str, object]:
    """一支函式的簽章：參數名、位置／關鍵字、必填還是有預設、預設是什麼。

    刻意**不記標註文字**：型別的名字可以搬家（決策紙說結構隨便改），但「哪幾個參數必填、
    預設是什麼、能不能用位置傳」是公開契約——放寬就是行為變了（載入器那一題的決策紙
    ``config-loaders-keep-path-required`` 拍的就是這件事）。
    """
    if not callable(target):
        raise TypeError(f"這一步要看簽章，但它不可呼叫：{target!r}")
    params: list[dict[str, object]] = []
    for param in inspect.signature(target).parameters.values():
        entry: dict[str, object] = {"name": param.name, "kind": param.kind.name}
        if param.default is inspect.Parameter.empty:
            entry["required"] = True
        else:
            entry["required"] = False
            entry["default"] = param.default
        params.append(entry)
    return {"params": params}


def alias_values(module: ModuleType, name: str) -> tuple[object, ...]:
    """一個公開型別別名（``Literal[...]``）宣告的那幾個值。"""
    return get_args(getattr(module, name))
