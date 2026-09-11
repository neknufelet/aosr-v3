#!/usr/bin/env python3
"""新家的 import 只准往下流：下層不准 import 上層、不准成環、只有最底一層准碰機器設定。

掃描面是卡上登記的套件根（今天是 ``src/aosr/``）底下每一支 ``.py``，加上所有規矩卡
（層次表住在卡上，這支檢查要打開每一張卡去找自己那一張）。真的用 :mod:`ast` 解析，
走全樹——**函式裡的延遲 import 一樣算**，那是繞開分層最省事的一種寫法。

**① 每一支 .py 都要對得回層次表裡的一個模組**

模組＝套件根底下的第一段：住在 ``physics`` 那個目錄裡的檔算 ``physics``，直接放在套件根
底下的 ``src/aosr/runtime.py`` 算 ``runtime``。套件自己的 ``__init__.py`` 是門面，不屬於
任何一組，它站在最上面：它可以 import 每一層，任何一層 import 它就是往上。
沒登記在層次表裡的模組一律紅——一支沒有層的檔，這張卡的另外兩條就無從判起。

**② 下層不准 import 上層**

同一組之間互相 import 不算逆向（那是同一層的事），往下引當然可以，只有往上引是違規。
判的是子套件這一級，不是檔這一級（模組內部兩支檔互相引，這一條看不到，理由寫在卡面）。

相對 import 的點數爬出套件根（套件根底下第二層的一支檔裡寫 ``from ...physics import y``）
一律紅，判在 :func:`_escapes_root`：那是**靜態看得一清二楚**的一條往上邊，只是解析器
上一版把它跟「引到套件外面」混為一談、靜默丟掉（找碴席實測 B1，同一條邊寫成三個點就回 0）。

**③ 不准成環**

模組之間的依賴圖不准有環。第②條已經把跨層的環擋掉一半（環一定含一條往上的邊），
這一條真正多咬到的是**同一組裡面**繞回自己的環——那種環第②條放行，但它一樣是
「誰先載入誰」說不清楚的形狀。

**④ 只有最底那一個模組准碰機器設定**

卡上登記 ``machine_module``（今天是 ``runtime``）與 ``machine_origins``（JAX 那幾個會改
全域行為的入口，加上 ``os.environ``／``os.putenv``／``os.unsetenv``）。判準不是比字面，
是問「這個運算式回指哪一個 ``(模組, 名字)``」（共用零件 :mod:`governance.names`），
再把出處接成點號串做前綴比對：``from jax import config`` 之後的 ``config.update(...)``、
``import jax.config as c`` 之後的 ``c.update(...)``、``from os import environ`` 之後的
``environ[...] = ...``，三種都咬。另外咬字串字面值：以卡上 ``machine_env_prefixes``
（``JAX_``／``XLA_``）開頭的環境變數名出現在別的模組裡就紅——名字準備好交給別人去設
（回傳一個 dict、塞進子程序的 ``env=``）繞得過前一格，而名字本身看得見。

**⑤ 只有最底那一個模組准動態取模組**

卡上登記 ``dynamic_import_origins``（``importlib.import_module``、``importlib.__import__``、
``builtins.__import__`` 含裸寫的 ``__import__``、``sys.modules``）。判準跟④同一套
（名字解析＋前綴比對，別名照咬），紅的理由不一樣：動態取模組的目標是執行期才算出來的
字串，靜態上看不出它引到哪一層，前三條全部繞得過去（找碴席實測 B2／B3／B4）。
上一版把這一族寫成「已知的洞」，這一版改成**禁**——新家不需要它。

血債：``core-layer-imports-orchestration-layer``（核心層反過來 import 編排層）與
``import-time-global-flip-poisons-suite``（模組載入時偷改全域設定污染整套測試）。
第三筆 ``same-concern-many-implementations-no-owner`` 只記關聯，理由在卡上。

**沒量到東西就回 2**

讀不到卡的 ``[settings]``、``[settings]`` 形狀壞掉（層次表把同一個模組登記兩層、
``machine_module`` 不在最底那一組）、套件根底下一支 ``.py`` 都沒有、某支 ``.py`` 剖不開，
一律 raise :class:`ToolBroken`。這一跑沒量到東西，「沒問題」這句話就不算數。
"""
from __future__ import annotations

import ast
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

from governance import names
from governance.exit_codes import ToolBroken, note, run
from governance.loader import RULES_DIR, setting_strings, setting_text

# 這張卡的 id。層次表只從「id 是這個」的那張卡讀。為什麼靠 id 認卡而不靠 check 欄：
# 必紅樣本是一棵棵獨立的迷你掃描根，每棵樹裡都得放一張卡（層次表在卡上），而樣本樹裡
# 沒有 governance/checks/ 底下那支程式——寫了 check 欄，refs-and-links-resolve 會把它
# 當引用去解析。（跟 style-guard 同一個做法。）
CARD_ID = "layers-import-downward-only"

PYTHON_SUFFIX = ".py"
CARD_SUFFIX = ".toml"
INIT_NAME = "__init__.py"

# 卡上 [settings] 的形狀。打錯字的層次表等於沒有層次表，所以多一個鍵、少一個鍵、
# 型別不對，一律回 2。
TEXT_KEYS = ("package_root", "package_name", "machine_module")
LIST_KEYS = ("machine_origins", "machine_env_prefixes", "dynamic_import_origins")
LAYERS_KEY = "layers"
SETTINGS_KEYS = (*TEXT_KEYS, *LIST_KEYS, LAYERS_KEY)

DOT = "."


# ── 卡上的層次表 ────────────────────────────────────────────────────────────


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（層次表住在其中一張上）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == CARD_SUFFIX)


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這張卡的層次表與名單。

    找不到、找到多張、或形狀不對，一律 raise ToolBroken——沒有尺就不出結論。
    """
    mine: list[tuple[str, dict[str, object]]] = []
    for path in _card_files(scan_root, files):
        rel = path.relative_to(scan_root).as_posix()
        try:
            data = tomllib.loads(_read_text(path, rel))
        except tomllib.TOMLDecodeError as exc:
            raise ToolBroken(f"{rel} 解不開（{exc}）——我沒看懂就不出結論") from exc
        if data.get("id") == CARD_ID:
            mine.append((rel, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——層次表只寫在卡上，讀不到這一跑就不算數"
        )
    rel, data = mine[0]
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（層次表、套件根與機器設定名單都寫在那裡）")
    _assert_settings(settings, rel)
    return settings


def _text_problems(settings: dict[str, object]) -> list[str]:
    bad: list[str] = []
    for key in TEXT_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}")
        elif not isinstance(value, str) or not value.strip():
            bad.append(f"{key} 必須是非空字串，實際是 {value!r}")
    root = settings.get("package_root")
    if isinstance(root, str) and root.strip():
        parts = root.strip("/").split("/")
        if root.startswith("/") or ".." in parts or "." in parts:
            bad.append(f"package_root={root!r} 必須是掃描根底下的相對路徑，不准往外指")
    return bad


def _list_problems(settings: dict[str, object]) -> list[str]:
    bad: list[str] = []
    for key in LIST_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}")
        elif not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif not value:
            bad.append(f"{key} 不准是空 list——空的名單等於那一條沒在管")
    return bad


def _layer_problems(settings: dict[str, object]) -> list[str]:
    """層次表的形狀：非空的組、每組非空、模組名不准重複出現在兩組。"""
    layers = settings.get(LAYERS_KEY)
    if LAYERS_KEY not in settings:
        return [f"缺 {LAYERS_KEY}（層次表，由下往上一層一組）"]
    if not isinstance(layers, list) or not layers:
        return [f"{LAYERS_KEY} 必須是非空的組清單，實際是 {layers!r}"]
    bad: list[str] = []
    seen: dict[str, int] = {}
    for index, group in enumerate(layers):
        where = f"{LAYERS_KEY} 第 {index + 1} 組"
        if not isinstance(group, list) or not group:
            bad.append(f"{where} 必須是非空的字串 list，實際是 {group!r}")
            continue
        for module in group:
            if not isinstance(module, str) or not module.strip():
                bad.append(f"{where} 裡有一項不是非空字串（{module!r}）")
            elif module in seen:
                bad.append(
                    f"模組 {module!r} 同時登記在第 {seen[module] + 1} 組與第 {index + 1} 組"
                    "——一個模組只准有一層，兩層等於沒有層"
                )
            else:
                seen[module] = index
    machine = settings.get("machine_module")
    if isinstance(machine, str) and machine.strip() and not bad:
        if machine not in seen:
            bad.append(f"machine_module={machine!r} 沒有登記在層次表裡")
        elif seen[machine] != 0:
            bad.append(
                f"machine_module={machine!r} 登記在第 {seen[machine] + 1} 組，不是最底那一組"
                "——唯一准碰機器設定的那一層必須是最底層，不然它上面那些層就沒有人擋得住"
            )
    return bad


def _assert_settings(settings: dict[str, object], rel: str) -> None:
    """[settings] 的形狀。任何一格不對就 raise ToolBroken——尺壞掉的時候不出結論。"""
    bad: list[str] = []
    extra = sorted(k for k in settings if k not in SETTINGS_KEYS)
    if extra:
        bad.append(f"多了不認識的鍵 {extra}，只認 {list(SETTINGS_KEYS)}")
    bad += _text_problems(settings)
    bad += _list_problems(settings)
    bad += _layer_problems(settings)
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _layer_index(settings: dict[str, object]) -> dict[str, int]:
    """模組名 -> 它在第幾層（由下往上，最底是 0）。形狀已經由 :func:`_assert_settings` 驗過。"""
    out: dict[str, int] = {}
    raw = settings.get(LAYERS_KEY)
    groups = raw if isinstance(raw, list) else []
    for index, group in enumerate(groups):
        for module in setting_strings({"g": group}, "g"):
            out[module] = index
    return out


# ── 掃描面 ─────────────────────────────────────────────────────────────────


def _package_files(scan_root: Path, files: list[Path], settings: dict[str, object]) -> list[Path]:
    """套件根底下每一支 ``.py``。"""
    prefix = setting_text(settings, "package_root").strip("/") + "/"
    return sorted(
        f
        for f in files
        if f.suffix == PYTHON_SUFFIX and f.relative_to(scan_root).as_posix().startswith(prefix)
    )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀／會判的檔：所有規矩卡 ＋ 套件根底下每一支 .py。

    ``check()`` 自己也是叫這一支拿掃描面，所以 ``--list-files`` 印出來的清單就是真的
    被掃的那一組，不是第二份會各自漂的宣告。
    """
    settings = _card_settings(scan_root, files)
    return sorted({*_card_files(scan_root, files), *_package_files(scan_root, files, settings)})


def _module_of(rel: str, settings: dict[str, object]) -> str:
    """這支檔屬於哪一個模組。套件自己的 ``__init__.py``（門面）回空字串。"""
    prefix = setting_text(settings, "package_root").strip("/") + "/"
    inside = rel[len(prefix):]
    parts = inside.split("/")
    if len(parts) == 1:
        return "" if parts[0] == INIT_NAME else parts[0].removesuffix(PYTHON_SUFFIX)
    return parts[0]


# ── ①②③ import 的方向 ─────────────────────────────────────────────────────


def _here(rel: str, settings: dict[str, object]) -> list[str]:
    """這支檔所在的那個套件，拆成一段一段（``config`` 目錄裡的檔是 ``["aosr", "config"]``）。"""
    package = setting_text(settings, "package_name")
    prefix = setting_text(settings, "package_root").strip("/") + "/"
    return [package, *rel[len(prefix):].split("/")[:-1]]


def _absolute_module(node: ast.ImportFrom, rel: str, settings: dict[str, object]) -> str:
    """把 ``from . import x``／``from ..physics import y`` 的模組名解析成絕對名。

    相對層級從這支檔所在的套件往上數：``physics`` 那個目錄裡的一支檔，一個點是
    ``aosr.physics``，兩個點是 ``aosr``。往上數超過套件根就回空字串，那一種由
    :func:`_escapes_root` 單獨判成違規——**不是「看不到」，是「不准」**（見那支函式）。
    """
    here = _here(rel, settings)
    up = (node.level or 1) - 1
    base = here[: len(here) - up] if up <= len(here) else []
    if not base:
        return ""
    return DOT.join([*base, node.module]) if node.module else DOT.join(base)


def _escapes_root(node: ast.stmt, rel: str, settings: dict[str, object]) -> bool:
    """這一行相對 import 的點數有沒有爬出套件根。

    套件根底下 ``config`` 那個目錄裡的一支檔寫 ``from ...physics import solver`` 就是這一種：
    兩個點已經是套件根，第三個點爬到套件外面去了。Python 會把它解析回這棵樹裡的
    ``aosr.physics``（套件根再上一層就是模組搜尋路徑），所以它跟寫
    ``from aosr.physics import solver`` 是同一件事——**一條靜態、AST 看得一清二楚的往上邊**。

    上一版把它跟「引到套件外面的第三方套件」混為一談、兩種都靜默丟掉，於是這一條路
    整條繞過分層（找碴席實測 B1：同一條往上邊寫成三個點就回 0）。這一版判紅：
    這種寫法沒有任何正當用途，而它算出來的目標取決於套件被裝在哪裡——
    那正是「同一份程式在兩台機器上行為不同」的來源。
    """
    if not isinstance(node, ast.ImportFrom) or not node.level:
        return False
    return node.level - 1 >= len(_here(rel, settings))


def _imported_modules(
    node: ast.stmt, rel: str, settings: dict[str, object], index: dict[str, int]
) -> Iterator[str]:
    """這一行 import 引到套件裡的哪幾個模組。空字串＝引到套件根自己（門面）。

    ``from aosr import physics`` 引到的是子模組 physics，不是門面——只有 ``import aosr``
    與 ``from aosr import <不是模組的名字>`` 才算引到門面（後者拿的是 ``__init__.py``
    自己定義或轉手出去的東西，那正是門面不准放 re-export 的理由）。
    """
    package = setting_text(settings, "package_name")
    if isinstance(node, ast.Import):
        for alias in node.names:
            parts = alias.name.split(DOT)
            if parts[0] == package:
                yield parts[1] if len(parts) > 1 else ""
        return
    if not isinstance(node, ast.ImportFrom):
        return
    full = _absolute_module(node, rel, settings) if node.level else (node.module or "")
    parts = full.split(DOT)
    if not full or parts[0] != package:
        return
    if len(parts) > 1:
        yield parts[1]
        return
    wanted = [alias.name for alias in node.names if alias.name != names.STAR_IMPORT]
    if not wanted:
        # ``from aosr import *``：拿到的是門面攤開之後的東西，靜態上看不出是哪幾個。
        yield ""
        return
    for alias_name in wanted:
        yield alias_name if alias_name in index else ""


def _import_hits(
    tree: ast.Module,
    rel: str,
    module: str,
    settings: dict[str, object],
    index: dict[str, int],
) -> tuple[list[str], set[str]]:
    """①②：模組登記了沒、這一支檔引到的每一個模組在不在它下面。順便回這支檔的出邊。"""
    bad: list[str] = []
    edges: set[str] = set()
    top = len(set(index.values()))
    mine = top if module == "" else index[module]
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if _escapes_root(node, rel, settings):
            bad.append(
                f"{rel}:{node.lineno} 相對 import 爬出了套件根"
                f"（`{ast.unparse(node)}`）——點數比這支檔所在的那幾層還多，"
                "算出來的目標取決於這個套件被裝在哪裡，而且它繞過了分層："
                "同一條往上引的邊寫成絕對名會紅，寫成多幾個點就沒人看得到。"
                "要引就寫絕對名，讓它被判一次方向"
            )
            continue
        for other in _imported_modules(node, rel, settings, index):
            if other == module:
                continue
            if other != "" and other not in index:
                bad.append(
                    f"{rel}:{node.lineno} 引到模組 {other!r}，但它沒有登記在卡 {CARD_ID} 的層次表裡"
                    "——沒有層的模組判不出方向，先把它登記到某一層（改卡要走 PR）"
                )
                continue
            theirs = top if other == "" else index[other]
            shown = "套件根的門面" if other == "" else f"模組 {other!r}"
            if theirs > mine:
                bad.append(
                    f"{rel}:{node.lineno} 第 {mine + 1} 層的 {module or '門面'} 引到"
                    f"第 {theirs + 1} 層的 {shown}——import 只准往下流。"
                    "依賴倒過來之後，下面那一層就再也不能單獨測、單獨換；"
                    "要嘛把需要的東西下沉到下層，要嘛把方向倒過來（改層次表要走 PR）"
                )
            elif other != "":
                edges.add(other)
    return bad, edges


def _cycle_hits(graph: dict[str, set[str]]) -> list[str]:
    """③：模組之間不准成環。回傳每一個找到的環（同一個環只報一次）。"""
    found: set[tuple[str, ...]] = set()
    stack: list[str] = []
    on_stack: set[str] = set()
    done: set[str] = set()

    def walk(node: str) -> None:
        stack.append(node)
        on_stack.add(node)
        for other in sorted(graph.get(node, set())):
            if other in on_stack:
                ring = stack[stack.index(other):]
                start = ring.index(min(ring))
                found.add(tuple(ring[start:] + ring[:start]))
            elif other not in done:
                walk(other)
        on_stack.discard(node)
        done.add(node)
        stack.pop()

    for node in sorted(graph):
        if node not in done:
            walk(node)
    return [
        "模組之間成環：" + " -> ".join([*ring, ring[0]])
        + "——環裡沒有一個是「下面那一層」，誰先載入誰說不清楚，"
        "拆不出可以單獨測的一塊；把環上某一條邊反過來，或把共用的東西抽成更下面一層"
        for ring in sorted(found)
    ]


# ── ④⑤ 只有最底那一個模組准碰機器設定與動態取模組 ─────────────────────────


def _origin_dotted(found: names.Origin) -> str:
    """把解析出來的出處接成點號串（``Origin("jax", "config.update")`` -> ``jax.config.update``）。"""
    return f"{found.module}{DOT}{found.name}" if found.name else found.module


def _matched_origin(node: ast.expr, resolved: names.Names, wanted: list[str]) -> str:
    """這個運算式回指的出處在不在名單上（前綴比對）。是就回命中的那一條，不是回空字串。"""
    found = resolved.origin(node)
    if found is None:
        return ""
    dotted = _origin_dotted(found)
    for prefix in wanted:
        if dotted == prefix or dotted.startswith(prefix + DOT):
            return prefix
    return ""


def _assumed_names(*groups: list[str]) -> dict[str, names.Origin]:
    """名字解析的打底：登記簿上每一條出處的第一段（``os``、``jax``、``sys``…）。

    ``builtins.`` 開頭的那幾條另外打底裸名字（``__import__`` 寫出來就是內建那一個），
    不然 ``__import__("aosr.physics")`` 這種不必先 import 的寫法追不到。
    清單從登記簿現算，不另外登記一份會漂的名單。
    """
    assumed: dict[str, names.Origin] = {}
    for prefix in [item for group in groups for item in group]:
        parts = prefix.split(DOT)
        assumed.setdefault(parts[0], names.Origin(parts[0], ""))
        if parts[0] == names.BUILTINS and len(parts) > 1:
            bare = DOT.join(parts[1:])
            assumed[bare] = names.Origin(names.BUILTINS, bare)
    return assumed


def _env_name_hits(node: ast.Constant, rel: str, machine: str, prefixes: list[str]) -> list[str]:
    """④的第二格：那一族環境變數的名字寫成字串字面值。"""
    if not isinstance(node.value, str):
        return []
    return [
        f"{rel}:{node.lineno} 出現環境變數名 {node.value!r}"
        f"——那一族機器設定只准 {machine!r} 那個模組碰。"
        "別處提到它的名字，就是準備把設定交給別人去改，"
        "而全域設定被誰改掉，從測試的紅綠上看不出來"
        for prefix in prefixes
        if node.value.startswith(prefix)
    ]


def _privilege_hits(
    tree: ast.Module, rel: str, module: str, settings: dict[str, object]
) -> list[str]:
    """④⑤：機器設定、那幾族環境變數名、以及動態取模組，只准出現在 machine_module 裡。"""
    machine = setting_text(settings, "machine_module")
    if module == machine:
        return []
    machines = setting_strings(settings, "machine_origins")
    dynamics = setting_strings(settings, "dynamic_import_origins")
    resolved = names.resolve(tree, _assumed_names(machines, dynamics))
    # 兩族各一句人話：紅的理由不一樣，訊息就不該長一樣。
    groups = [
        (
            machines,
            f"只准 {machine!r} 那個模組碰。模組載入的時候偷改一次全域設定，"
            "整套測試從此跑在另一組設定上，而紅綠上只看得到「昨天會過今天不過」",
        ),
        (
            dynamics,
            "動態取模組就是分層的逃生門：目標是執行期才算出來的字串，"
            "靜態上看不出它引到哪一層，這張卡的前三條全部繞得過去。"
            f"新家不需要它；真的需要就走 {machine!r} 那一層、開一張票談",
        ),
    ]
    bad: list[str] = []
    seen: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            bad += _env_name_hits(
                node, rel, machine, setting_strings(settings, "machine_env_prefixes")
            )
            continue
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        for wanted, why in groups:
            prefix = _matched_origin(node, resolved, wanted)
            if not prefix or (node.lineno, prefix) in seen:
                continue
            seen.add((node.lineno, prefix))
            bad.append(f"{rel}:{node.lineno} 碰到 {prefix}（`{ast.unparse(node)}`）——{why}")
    return bad


# ── 主流程 ─────────────────────────────────────────────────────────────────


def _parse(path: Path, rel: str) -> ast.Module:
    try:
        return ast.parse(_read_text(path, rel), filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行剖不開（{exc.msg}）——我沒看懂就不出結論") from exc


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    root = setting_text(settings, "package_root")
    picked = _package_files(scan_root, files, settings)
    if not picked:
        raise ToolBroken(
            f"{scan_root}/{root} 底下一支版控裡的 .py 都沒有"
            "——這一跑沒讀到任何新家的程式，「沒問題」這句話不算數"
        )
    index = _layer_index(settings)
    note(f"層次表這一跑登記了 {sorted(index)}，套件根底下量到 {len(picked)} 支 .py")

    graph: dict[str, set[str]] = {}
    bad: list[str] = []
    for path in picked:
        rel = path.relative_to(scan_root).as_posix()
        if not path.is_file():
            # 版控認得、檔案系統上不在（剛被刪掉還沒 commit）。不猜內容，跳過。
            continue
        module = _module_of(rel, settings)
        tree = _parse(path, rel)
        bad += _privilege_hits(tree, rel, module, settings)
        if module != "" and module not in index:
            bad.append(
                f"{rel} 屬於模組 {module!r}，但它沒有登記在卡 {CARD_ID} 的層次表裡"
                "——沒有層就判不出它引的東西是往上還是往下，先把它登記到某一層（改卡要走 PR）"
            )
            continue
        hits, edges = _import_hits(tree, rel, module, settings, index)
        bad += hits
        if module != "":
            graph.setdefault(module, set()).update(edges)
    bad += _cycle_hits(graph)
    return sorted(bad)


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="新家的 import 只准往下流：下層不准引上層、不准成環、只有最底一層准碰機器設定",
            targets=targets,
        )
    )
