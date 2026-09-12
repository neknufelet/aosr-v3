"""票 #134 第二刀的探針：**把一筆 case 真的跑一遍**，產生器與考卷共用這一支。

**codec（值怎麼編成記號）不重寫。** 第一刀那一支 :mod:`blueprint.materials_cut1_probe` 的
:func:`~blueprint.materials_cut1_probe.encode` 已經被一份控制組驗過（
``tests/engine/test_materials_judge_control.py``：先證明舊寫法擋不住容器換種類，再證明現在
擋得住），這一支直接借它，不另寫一份——兩份 codec 會慢慢漂開，漂開之後比出來的差異是「兩支
編法不一樣」，不是「新家跟上一代不一樣」。簽章那一格同理，借第一刀那一支。

**跑法為什麼要自己一份。** 這一刀有第一刀沒有的三件事：巢狀的一層表（pydantic 的模型）、
**真的檔案**（載入器要 glob 一個目錄）、以及讀模組層級的名字。把它們塞回第一刀那支
``_run_step`` 等於改動已經綠過的裁判；這一支照著同一個形狀另開一份，只多那三種動詞。

**暫存目錄。** 每一筆 case 現開一個（``tempfile.mkdtemp``），跑完刪掉。case 表只寫相對
名字；錯誤訊息裡那一段絕對路徑在記下來之前換成 ``<TMP>``——**只換那一段**，其他一個字都
不放寬（``FileNotFoundError`` 的 errno 文字、pydantic 的欄位路徑、yaml 的行號都照留）。

**donor 與新家跑的是同一份程式**，只差一個 ``lookup``（去哪一個套件拿模組）。
"""
from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable, Container, Sized
from pathlib import Path
from types import ModuleType
from typing import Final

from blueprint import materials_cut1_probe as cut1
from blueprint import materials_cut2_steps as steps

# 答案檔的形狀版本。讀的人與寫的人要對得上，對不上就當場炸。
ANSWER_SCHEMA: Final[int] = 1

# 暫存根在記號裡的替身。訊息裡出現真路徑就換成它——路徑每一跑都不一樣，是環境不是行為。
TMP_TOKEN: Final[str] = "<TMP>"

# 一個模組解析器：表上的名字（``freq_axis``）換成真的模組物件。
Lookup = Callable[[str], ModuleType]

# codec 借第一刀那一份（公開的那兩支）。
encode = cut1.encode
alias_values = cut1.alias_values


class CaseTableError(RuntimeError):
    """**case 表寫壞了**，不是產品的行為。

    這一種例外**不准**被收成 ``raised``：``raised`` 那一格的意思是「上一代的產品在這一筆
    輸入下炸了什麼」，是要跟新家逐字比的答案。框架自己失效（例如 case 表叫探針把檔寫到
    暫存根外面）如果也走那一格，產生器就會把一筆「我沒跑成」寫成一筆看起來很正常的標準
    答案——那是這個 repo 的頭號死因（一個只會回綠的裁判）。所以它一路往外炸，產生器當場
    非零退出、考卷當場紅。
    """


def confined(root: Path, name: str) -> Path:
    """把表上那個相對名字接到暫存根底下，**接不進去就當場炸**。

    三種寫法一律拒絕：絕對路徑（``Path("/a") / "/etc/x"`` 會把左邊整個丟掉，這是實測
    重現過的逃出）、``..`` 爬出去、以及經由符號連結解析之後落在根外面的。判準不是「字串
    看起來像什麼」而是**解析完的位置在不在根底下**——只有解析得出來的位置才算數。

    ``name`` 是空字串的時候接出來就是根自己（``load_experiment`` 那一筆要餵一個目錄給它，
    看它炸 ``IsADirectoryError``），那還在根底下，准。
    """
    candidate = Path(name)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise CaseTableError(
            f"case 表給了一個絕對路徑 {name!r}——這一刀的檔只准寫在這一筆自己的暫存根底下"
        )
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise CaseTableError(
            f"case 表給的 {name!r} 解析之後落在暫存根外面（{target}）"
            "——這一刀的檔只准寫在這一筆自己的暫存根底下"
        )
    return target


def run_case(
    lookup: Lookup,
    case: steps.CaseEntry,
    module_paths: dict[str, str],
) -> dict[str, object]:
    """把一筆 case 的步驟依序跑完，回傳那一筆的結果格。

    跑完 → ``{"values": {名字: 記號}}``；中途炸掉 → ``{"raised": {"step": 第幾步,
    "type": 例外種類, "message": 訊息}}``。**例外是行為，不是失敗**；**第幾步也要記**，
    同一個例外在不同步炸是兩件事（守門提早了或晚了）。

    ``module_paths`` 是「表上的名字 → 這一邊真的 import 路徑」，換的範圍由
    :func:`normalise_result` 說了算：暫存目錄整格換，模組路徑**只在** CPython 那一句
    ``<類別的完整名字>() argument after ** must be a mapping, not …`` 的開頭換。``values``
    整棵不做模組替換——那裡面的字串是資料，不是「模組住在哪裡」。
    """
    root = Path(tempfile.mkdtemp(prefix="aosr-cut2-")).resolve()
    try:
        return normalise_result(_run(lookup, root, case), str(root), module_paths)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run(lookup: Lookup, root: Path, case: steps.CaseEntry) -> dict[str, object]:
    """在一個只活這一筆的環境裡跑完所有步驟。"""
    env: dict[str, object] = {}
    for index, step in enumerate(case["steps"]):
        try:
            _run_step(lookup, root, env, step)
        except CaseTableError:
            # 框架自己失效：一路往外炸，**不准**變成一筆標準答案（見 CaseTableError 的說明）。
            raise
        except Exception as exc:  # noqa: BLE001  # expires=2026-12-08 reason=這一格要記錄「炸什麼」而不是讓它往外炸，例外的種類正是被比的東西
            return {"raised": {"step": index, "type": type(exc).__name__, "message": str(exc)}}
    return {"values": {name: encode(_fetch(env, name)) for name in case["report"]}}


# CPython 在「``F(**data)`` 但 data 不是一層表」時寫的那一句。前面接的是**類別的完整
# 名字**（模組路徑 ＋ 類別名），所以只有這一句的開頭那一段是「模組住在哪裡」而不是資料。
STAR_ARGS_TAIL: Final[str] = r"\(\) argument after \*\* must be a mapping, not [A-Za-z_][\w.]*"

# 那一句前面那一段長什麼樣：``<模組路徑>.<類別名>``。模組路徑不是用猜的——由
# :func:`normalise_result` 拿呼叫端給的那幾條真路徑去比，比不中就一個字都不換。
STAR_ARGS_HEAD: Final[re.Pattern[str]] = re.compile(rf"^([A-Za-z_]\w*){STAR_ARGS_TAIL}$")


def normalise_result(
    result: dict[str, object],
    root: str,
    module_paths: dict[str, str],
) -> dict[str, object]:
    """把結果格裡**兩種位置**換成替身，其他一個字都不動。

    ① **暫存目錄**（整格都換）：每一跑都不一樣，是環境不是行為。

    ② **模組的 import 路徑**，而且只換一個地方：``raised`` 是 ``TypeError``、訊息又剛好
    是 CPython 那一句「``<類別的完整名字>() argument after ** must be a mapping, not …``」
    的時候，把開頭那一段模組路徑換掉。那一段是**直譯器印出來的類別住在哪裡**，依決策紙
    本來就要搬家；不換的話那三筆等於要求新家把模組取名叫 ``lib.config.*``。

    **以前是整棵結果無差別換，那是一個洞**（獨立審查實測）：``material_id`` 這種**資料**
    剛好寫成 ``lib.config.material_loader`` 與 ``aosr.materials.material_loader`` 的時候，
    兩個不同的值會被換成同一個記號，裁判就再也看不見那一格變了。所以現在：``values``
    整棵**不做**模組替換，其他訊息（ValidationError 裡的使用者輸入、yaml 的位置）也不做。
    """
    swapped = _swap(result, [(root, TMP_TOKEN)])
    if not isinstance(swapped, dict):
        raise TypeError("結果格換完之後不是一層表——這不可能，除非上面那一支被改壞了")
    raised = swapped.get("raised")
    if isinstance(raised, dict) and raised.get("type") == "TypeError":
        message = raised.get("message")
        if isinstance(message, str):
            raised["message"] = _swap_callable_module(message, module_paths)
    return swapped


def _swap_callable_module(message: str, module_paths: dict[str, str]) -> str:
    """只有 CPython 那一句的**開頭那一段**模組路徑換得掉，其餘原字奉還。

    比對的方式是「這條訊息是不是**正好**長那個樣子」：開頭必須是呼叫端給的某一條真路徑，
    接下來必須是一個類別名，再接下來必須一字不差是那一句。差一點就整條原字回去——寧可
    多留一段會漂的路徑，也不要把一格資料悄悄換掉。
    """
    # 長的先試：``lib.config.experiment_schema`` 要在 ``lib.config`` 之前比中。
    for name, path in sorted(module_paths.items(), key=lambda pair: -len(pair[1])):
        prefix = f"{path}."
        if not message.startswith(prefix):
            continue
        rest = message[len(prefix) :]
        if STAR_ARGS_HEAD.match(rest):
            return f"<MODULE:{name}>.{rest}"
    return message


def _swap(value: object, swaps: list[tuple[str, str]]) -> object:
    """逐格走過記號，字串裡那幾段換成替身。"""
    if isinstance(value, str):
        for old, new in swaps:
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {key: _swap(item, swaps) for key, item in value.items()}
    if isinstance(value, list):
        return [_swap(item, swaps) for item in value]
    return value


def _fetch(env: dict[str, object], name: str) -> object:
    """從環境裡拿一個名字（沒有就當場炸——report 寫了卻沒存進去是 case 表壞了）。"""
    if name not in env:
        raise KeyError(f"這一筆 case 的 report 要 {name!r}，但沒有任何一步存過它")
    return env[name]


def _value(env: dict[str, object], root: Path, raw: object) -> object:
    """把表上的一格還原成 Python 值。

    ``ref`` 從環境裡拿、``obj``／``seq`` 往下遞迴（巢狀的表裡面也可以有 ``under``）、
    ``under`` 接到暫存根底下；其餘的記號交給第一刀那一支
    :func:`~blueprint.materials_cut1_steps.resolve`（同一套寫法，不另立一份）。
    """
    if not isinstance(raw, dict):
        return steps.resolve(raw)
    kind = raw.get("k")
    if kind == "ref":
        return _fetch(env, _text(raw.get("v"), "ref"))
    if kind == "obj":
        return {name: _value(env, root, item) for name, item in _table(raw.get("v")).items()}
    if kind == "seq":
        return [_value(env, root, item) for item in _listed(raw.get("v"))]
    if kind == "under":
        return confined(root, _text(raw.get("v"), "under"))
    if kind == "under_str":
        return str(confined(root, _text(raw.get("v"), "under_str")))
    return steps.resolve(raw)


def _text(raw: object, where: str) -> str:
    """收窄成字串（不是字串就當場炸——表壞了不要猜）。"""
    if not isinstance(raw, str):
        raise ValueError(f"{where} 那一格不是字串：{raw!r}")
    return raw


def _table(raw: object) -> dict[str, object]:
    """收窄成一層表。"""
    if not isinstance(raw, dict):
        raise ValueError(f"obj 那一格不是一層表：{raw!r}")
    return {str(key): item for key, item in raw.items()}


def _listed(raw: object) -> list[object]:
    """收窄成一串東西。"""
    if not isinstance(raw, list):
        raise ValueError(f"seq 那一格不是一串東西：{raw!r}")
    return raw


def _args(env: dict[str, object], root: Path, step: steps.Step) -> dict[str, object]:
    """一個步驟的關鍵字參數，整格還原。"""
    return {name: _value(env, root, raw) for name, raw in (step.get("args") or {}).items()}


def _target(lookup: Lookup, path: str) -> object:
    """``<表上的模組>.<名字>[.<名字>]`` 解析成真的東西。"""
    head, _, rest = path.partition(".")
    obj: object = lookup(head)
    if not rest:
        return obj
    for part in rest.split("."):
        obj = getattr(obj, part)
    return obj


def _run_step(lookup: Lookup, root: Path, env: dict[str, object], step: steps.Step) -> None:
    """跑一個步驟，把結果（如果有 ``to``）存進環境。"""
    verb = step["do"]
    if verb == "call":
        _store(env, step, _invoke(_target(lookup, step["fn"]), _args(env, root, step)))
        return
    if verb == "method":
        bound = getattr(_fetch(env, step["on"]), step["name"])
        _store(env, step, _invoke(bound, _args(env, root, step)))
        return
    if verb == "get":
        _store(env, step, _target(lookup, step["fn"]))
        return
    if verb == "read":
        _store(env, step, getattr(_fetch(env, step["on"]), step["name"]))
        return
    if verb == "set":
        setattr(_fetch(env, step["on"]), step["name"], _value(env, root, step.get("value")))
        return
    if verb == "len":
        _store(env, step, len(_sized(_fetch(env, step["on"]))))
        return
    if verb == "has":
        _store(env, step, _value(env, root, step.get("value")) in _container(_fetch(env, step["on"])))
        return
    if verb == "sig":
        _store(env, step, cut1._signature(_target(lookup, step["fn"])))  # noqa: SLF001  # expires=2026-12-08 reason=簽章那一格刻意跟第一刀共用同一支讀法，兩份讀法會漂開，屆時比出來的差異會是「兩邊讀法不同」而不是行為不同
        return
    _run_file_step(root, step)


def _run_file_step(root: Path, step: steps.Step) -> None:
    """檔案系統那兩種動詞（這一刀才有的：載入器要真的讀檔）。"""
    verb = step["do"]
    if verb == "mkdir":
        confined(root, step["name"]).mkdir(parents=True, exist_ok=True)
        return
    if verb == "file":
        target = confined(root, step["name"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_text(_table(step["value"]).get("v"), "file"), encoding="utf-8")
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
    """收窄成「量得出長度的東西」。"""
    if not isinstance(value, Sized):
        raise TypeError(f"這一步要量長度，但它沒有 __len__：{type(value).__name__}")
    return value


def _container(value: object) -> Container[object]:
    """收窄成「問得出 in 的東西」。"""
    if not isinstance(value, Container):
        raise TypeError(f"這一步要問 in，但它沒有 __contains__：{type(value).__name__}")
    return value
