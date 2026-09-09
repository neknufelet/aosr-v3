#!/usr/bin/env python3
"""寫法警衛：print 只准在輸出層、字串不准拼路徑、開檔一律 with、函式不准超過門檻。

掃描面是版控裡的每一支 ``.py``（扣掉卡上登記的前綴：必紅樣本樹與本機 agent 工具目錄），
加上所有規矩卡（門檻與白名單住在卡上，這支檢查要打開每一張卡去找自己那一張）。
真的用 :mod:`ast` 解析，不用正則——正則分不出文件字串裡舉的例子與真的會跑的那一行。

**① print 只准出現在輸出層**

輸出層是卡的 ``[[settings.allow]]`` 具名列出的那幾個檔，一筆一個檔，各自帶 ``reason``
與 ``expires``（那兩格的形狀定義在 :mod:`governance.loader`，「到期了沒」由規矩卡
``exemptions-need-expiry`` 判）。不在名單上的檔出現 ``print(...)`` 就是一筆違規。
治理層要對人說話的正路是 :func:`governance.exit_codes.note`：判決收據、HIT、NOTE
全部從同一層出去，才看得出哪一行是這一跑的產品、哪一行是誰某天為了除錯加的。

**② 字串拼路徑（只認流進路徑水槽的那一種）**

判準刻意收窄：``+`` 拼出來的字串（拼接鏈裡要有字串字面值）**流進路徑水槽**才算違規。
水槽有三種，名單在卡上：開檔／路徑建構那幾支呼叫、``os.path.`` 底下每一支、以及子程序的
argv 那一格（第一個位置參數或關鍵字 ``args``）。資料流走一手：拼接結果直接當參數，
或先指給同一個作用域裡的一個名字再當參數。

「看到字串相加就紅」是不能做的：乾淨樹裡 ``blueprint/remap_cards.py`` 有一處
``json.dumps(...) + 換行`` 再寫成檔案內容，樸素實作會當場誤咬。誤咬機比漏抓更糟。

**③ 開檔一律 with**

卡上登記的那幾支開檔呼叫，必須出現在某個 ``with`` 的頭上（``with open(...) as fh``、
``with closing(open(...))`` 都算）。``Path.read_text``／``write_text`` 合法，它們自己關。

**④ 函式行數與分支數**

門檻登記在卡的 ``[settings]``，這支程式沒有預設值。行數＝函式的起訖行數扣掉它自己的
文件字串（把說明算進行數等於在罰寫說明）；分支數＝函式身體裡 ``if``／``for``／``while``／
``except``／``with``／三元／推導式／布林運算／``assert``／``match`` case 的節點數。
巢狀的 ``def`` 各自算，外層不把內層算進去。

**ruff 是這張卡的另一半**

這支程式自己不跑 ruff：ruff 的規則集與它自己的數字住在 ``pyproject.toml`` 的
``[tool.ruff]``（只有一份、走 PR 看得到），由 CI 上獨立一步跑。這裡只確認 ruff 裝得起來
——裝不起來就 raise :class:`ToolBroken` 讓外殼回 2，因為那時這張卡的結論不完整。
為什麼不在這裡跑 ruff：那要嘛得把規則集抄一份到卡上（同一份名單兩個家），要嘛得讓每一份
必紅樣本樹各帶一份 ruff 設定，兩條路都比現在糟。

**沒掃到東西就回 2**

讀不到卡的 ``[settings]``、``[settings]`` 形狀壞掉、掃描面上一支 ``.py`` 都沒有、
某支 ``.py`` 剖不開，一律 raise :class:`ToolBroken`。這一跑沒量到東西，
「沒問題」這句話就不算數。

血債：沒有。這張卡是好習慣卡，理由寫在 ``governance/rules/style-guard.toml`` 的檔頭，
它刻意沒管的八件事也在那張卡的檔尾。
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.loader import (
    EXEMPTION_KEYS,
    RULES_DIR,
    setting_int,
    setting_strings,
    setting_tables,
)

# 這張卡的 id。門檻與白名單只從「id 是這個」的那張卡讀。為什麼靠 id 認卡而不靠 check 欄：
# 必紅樣本是一棵棵獨立的迷你掃描根，每棵樹裡都得放一張卡（門檻在卡上），而樣本樹裡沒有
# governance/checks/ 底下那支程式——寫了 check 欄，refs-and-links-resolve 會把它當引用去解析。
CARD_ID = "style-guard"

PYTHON_SUFFIX = ".py"
CARD_SUFFIX = ".toml"

# ① 只認這一個呼叫。sys.stdout.write 與 logging 繞得過去，理由寫在卡面。
PRINT_NAME = "print"

# ruff 是這張卡的另一半：裝不起來就回 2。刻意只問版本，不在這裡跑 lint（理由見模組說明）。
RUFF_VERSION_ARGV = ("ruff", "--version")

# 卡上 [settings] 的形狀。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
INT_KEYS = ("max_function_lines", "max_function_branches")
LIST_KEYS = (
    "path_sink_calls",
    "path_sink_prefixes",
    "subprocess_argv_calls",
    "with_required_calls",
    "scan_exempt_prefixes",
)
NONEMPTY_LIST_KEYS = ("path_sink_calls", "subprocess_argv_calls", "with_required_calls")
ALLOW_KEY = "allow"
# 一筆放行三格：放行哪個檔（path），加上放行條目共用的兩格（reason ＋ expires，形狀定義在
# governance/loader.py 的 EXEMPTION_KEYS，由規矩卡 exemptions-need-expiry 統一）。
ALLOW_ENTRY_KEYS = ("path", *EXEMPTION_KEYS)
SETTINGS_KEYS = (*INT_KEYS, *LIST_KEYS, ALLOW_KEY)

# 自己一個作用域的節點。走到它就換一層算，外層不把內層的行數與分支算進來。
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

# ④ 算分支的節點。清單寫在這裡而不是卡上：這是「分支」這個詞的定義，不是可以調寬調窄的門檻
# ——調它不會把紅變綠，會變成在算另一件事。門檻（幾個算太多）才在卡上。
BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.IfExp,
    ast.Assert,
    ast.comprehension,
    ast.BoolOp,
    ast.match_case,
)

# ② 子程序的 argv 住在哪一格。位置參數的第一格，或這個名字的關鍵字參數。
ARGV_KEYWORD = "args"


# ── 卡上的門檻 ──────────────────────────────────────────────────────────────


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（門檻與白名單住在其中一張上）。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == CARD_SUFFIX)


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這張卡的門檻與白名單。

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
            "——門檻與白名單只寫在卡上，讀不到這一跑就不算數"
        )
    rel, data = mine[0]
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（門檻、路徑水槽名單與輸出層白名單都寫在那裡）")
    _assert_settings(settings, rel)
    return settings


def _int_problems(settings: dict[str, object]) -> list[str]:
    bad: list[str] = []
    for key in INT_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}（函式的上限，正整數）")
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            bad.append(f"{key} 必須是 1 以上的整數，實際是 {value!r}——門檻寫 0 等於沒有門檻")
    return bad


def _list_problems(settings: dict[str, object]) -> list[str]:
    bad: list[str] = []
    for key in LIST_KEYS:
        value = settings.get(key)
        if key not in settings:
            bad.append(f"缺 {key}（沒有也要明寫 {key} = []）")
        elif not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif key in NONEMPTY_LIST_KEYS and not value:
            bad.append(f"{key} 不准是空 list——空的名單等於那一條沒在管")
    return bad


def _allow_problems(settings: dict[str, object]) -> list[str]:
    if ALLOW_KEY not in settings:
        return [
            f"缺 {ALLOW_KEY}（輸出層白名單，沒有也要明寫 {ALLOW_KEY} = []）"
            "——省略跟「空的」不是同一件事：省略讀起來像忘了寫，這一條就不知道自己有沒有尺"
        ]
    allow = settings[ALLOW_KEY]
    if not isinstance(allow, list) or not all(isinstance(x, dict) for x in allow):
        return [
            f"{ALLOW_KEY} 必須是 [[settings.{ALLOW_KEY}]] 表陣列"
            f"（沒有也要明寫 {ALLOW_KEY} = []），實際是 {allow!r}"
        ]
    bad: list[str] = []
    for index, entry in enumerate(allow):
        where = f"[[settings.{ALLOW_KEY}]] 第 {index + 1} 條"
        missing = [k for k in ALLOW_ENTRY_KEYS if k not in entry]
        if missing:
            bad.append(f"{where} 缺 {missing}——輸出層放行要說清楚是哪個檔、為什麼、到什麼時候")
        odd = [k for k in entry if k not in ALLOW_ENTRY_KEYS]
        if odd:
            bad.append(f"{where} 多了不認識的鍵 {sorted(odd)}，只認 {list(ALLOW_ENTRY_KEYS)}")
        for key in ALLOW_ENTRY_KEYS:
            value = entry.get(key)
            if key in entry and (not isinstance(value, str) or not value.strip()):
                bad.append(f"{where} 的 {key} 必須是非空字串，實際是 {value!r}")
    return bad


def _assert_settings(settings: dict[str, object], rel: str) -> None:
    """[settings] 的形狀。任何一格不對就 raise ToolBroken——尺壞掉的時候不出結論。"""
    bad: list[str] = []
    extra = sorted(k for k in settings if k not in SETTINGS_KEYS)
    if extra:
        bad.append(f"多了不認識的鍵 {extra}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒有門檻）")
    bad += _int_problems(settings)
    bad += _list_problems(settings)
    bad += _allow_problems(settings)
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _names(settings: dict[str, object], key: str) -> list[str]:
    """一張名單。形狀已經由 :func:`_assert_settings` 驗過，收窄走載入器那一支。"""
    return setting_strings(settings, key)


def _threshold(settings: dict[str, object], key: str) -> int:
    return setting_int(settings, key)


def _allow_paths(settings: dict[str, object]) -> list[str]:
    """輸出層白名單上的那幾個檔（相對掃描根）。"""
    return sorted({str(entry["path"]) for entry in setting_tables(settings, ALLOW_KEY)})


# ── ruff 在不在 ────────────────────────────────────────────────────────────


def _ruff_version() -> str:
    """確認 ruff 裝得起來。裝不起來就 raise ToolBroken（回 2，不回 0）。

    刻意只問版本：ruff 的規則集住在 pyproject.toml，由 CI 上獨立一步跑，
    這裡不重跑也不重抄那份名單（理由見模組說明）。
    沒有看門狗秒數：問版本不會卡住，真的卡住有 job 那一層的分鐘上限接著。
    """
    try:
        proc = subprocess.run(list(RUFF_VERSION_ARGV), capture_output=True, text=True)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise ToolBroken(
            f"ruff 跑不起來（{exc}）——它是這張卡的另一半（CI 上那一步 uv run ruff check），"
            "它不在的時候這張卡只判得到自己那四條、判不到 ruff 那半，「沒問題」這句話不算數"
        ) from exc
    if proc.returncode != 0:
        raise ToolBroken(
            f"ruff 回 {proc.returncode}（{proc.stderr.strip()[:200]}）——問版本都失敗，這一跑不算數"
        )
    return proc.stdout.strip() or proc.stderr.strip()


# ── 共用的 AST 小工具 ──────────────────────────────────────────────────────


def _dotted(node: ast.expr) -> str:
    """把 ``os.path.join`` 這種呼叫對象攤成 ``"os.path.join"``。認不出來就回空字串。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _parse(path: Path, rel: str) -> ast.Module:
    try:
        return ast.parse(_read_text(path, rel), filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行剖不開（{exc.msg}）——我沒看懂就不出結論") from exc


def _own_statements(scope: ast.AST) -> Iterator[ast.stmt]:
    """這一層自己的每一個 statement（走進 if／for／try，但不走進巢狀的 def／class）。"""
    stack: list[ast.stmt] = list(getattr(scope, "body", []))
    while stack:
        stmt = stack.pop()
        yield stmt
        if isinstance(stmt, SCOPE_NODES):
            continue
        for field in ("body", "orelse", "finalbody"):
            stack.extend(getattr(stmt, field, None) or [])
        for handler in getattr(stmt, "handlers", None) or []:
            stack.extend(handler.body)


def _scopes(tree: ast.Module) -> Iterator[ast.AST]:
    """一份檔裡的每一個作用域：模組自己，以及每一個 def／class。"""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, SCOPE_NODES):
            yield node


def _calls_in_scope(scope: ast.AST) -> Iterator[ast.Call]:
    """這一層自己的每一個呼叫（不含巢狀 def／class 身體裡的——那些各自算一層）。"""
    for stmt in _own_statements(scope):
        if isinstance(stmt, SCOPE_NODES):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                yield node


# ── ① print 只准出現在輸出層 ───────────────────────────────────────────────


def _print_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted(node.func) == PRINT_NAME
    ]


def _print_hits(tree: ast.Module, rel: str) -> list[str]:
    return [
        f"{rel}:{call.lineno} 用 {PRINT_NAME}(...) 說話，但這個檔不在輸出層白名單上"
        "——要對人說話就走輸出層那支 note()（governance/exit_codes.py）；"
        "散在各支程式裡的 print 混在判決輸出裡，之後看不出哪一行是這一跑的收據、"
        f"哪一行是誰某天為了除錯加的。真的是這支程式的產品，就在卡 {CARD_ID} 的"
        f" [[settings.{ALLOW_KEY}]] 具名放行，寫理由與到期日"
        for call in _print_calls(tree)
    ]


# ── ② 字串拼路徑 ──────────────────────────────────────────────────────────


def _concat_literals(node: ast.AST) -> list[str]:
    """這個運算式是不是「``+`` 拼出來的字串」。是就回拼接鏈裡的字串字面值。"""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return []
    found: list[str] = []
    for side in (node.left, node.right):
        if isinstance(side, ast.Constant) and isinstance(side.value, str):
            found.append(side.value)
        else:
            found += _concat_literals(side)
    return found


def _first_concat(expr: ast.expr) -> ast.BinOp | None:
    """這段運算式裡最外面那一個字串拼接。找不到回 None。"""
    for node in ast.walk(expr):
        if isinstance(node, ast.BinOp) and _concat_literals(node):
            return node
    return None


def _sink_kind(call: ast.Call, settings: dict[str, object]) -> str:
    """這個呼叫是不是路徑水槽。是就回一句人話（水槽的名字），不是回空字串。"""
    name = _dotted(call.func)
    if not name:
        return ""
    if name in _names(settings, "path_sink_calls"):
        return f"{name}()"
    for prefix in _names(settings, "path_sink_prefixes"):
        if name.startswith(prefix):
            return f"{name}()"
    if name in _names(settings, "subprocess_argv_calls"):
        return f"{name}() 的 argv"
    return ""


def _sink_args(call: ast.Call, settings: dict[str, object]) -> list[ast.expr]:
    """這個水槽的哪幾格算「路徑」。子程序只看 argv 那一格，其餘看每一個位置參數。"""
    if _dotted(call.func) in _names(settings, "subprocess_argv_calls"):
        argv = [kw.value for kw in call.keywords if kw.arg == ARGV_KEYWORD]
        return argv or call.args[:1]
    return list(call.args)


def _bound_concats(scope: ast.AST) -> dict[str, ast.BinOp]:
    """這一層裡「被指定成一個字串拼接」的名字（一手資料流）。"""
    bound: dict[str, ast.BinOp] = {}
    for stmt in _own_statements(scope):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        # 明寫 isinstance(…, ast.BinOp)：_concat_literals 非空就一定是 BinOp，但那是
        # 那支函式內部的事，型別上看不出來。寫出來比掛一個抑制註解誠實，也少一個抑制。
        if not isinstance(target, ast.Name) or not isinstance(stmt.value, ast.BinOp):
            continue
        if _concat_literals(stmt.value):
            bound[target.id] = stmt.value
    return bound


def _concat_message(rel: str, lineno: int, where: str, shown: str) -> str:
    return (
        f"{rel}:{lineno} 字串拼出來的路徑流進 {where}（{shown}）"
        "——自己組分隔符就得自己想到重複斜線、上一層、名字裡本來就有斜線這些事，"
        "平台一換再想一次。路徑用 Path 接（`base / name`），不要用 + 拼字串"
    )


def _concat_hits(tree: ast.Module, rel: str, settings: dict[str, object]) -> list[str]:
    """② 拼接結果直接當水槽的參數，或先指給同一層的一個名字再當參數。"""
    bad: list[str] = []
    for scope in _scopes(tree):
        bound = _bound_concats(scope)
        for call in _calls_in_scope(scope):
            where = _sink_kind(call, settings)
            if not where:
                continue
            for arg in _sink_args(call, settings):
                direct = _first_concat(arg)
                if direct is not None:
                    bad.append(_concat_message(rel, direct.lineno, where, ast.unparse(direct)))
                    continue
                for node in ast.walk(arg):
                    if isinstance(node, ast.Name) and node.id in bound:
                        hit = bound[node.id]
                        shown = f"{node.id} = {ast.unparse(hit)}"
                        bad.append(_concat_message(rel, call.lineno, where, shown))
    return bad


# ── ③ 開檔一律 with ───────────────────────────────────────────────────────


def _open_hits(tree: ast.Module, rel: str, settings: dict[str, object]) -> list[str]:
    wanted = _names(settings, "with_required_calls")
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            for inner in ast.walk(item.context_expr):
                if isinstance(inner, ast.Call) and _dotted(inner.func) in wanted:
                    guarded.add(id(inner))
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _dotted(node.func) not in wanted:
            continue
        if id(node) in guarded:
            continue
        bad.append(
            f"{rel}:{node.lineno} {_dotted(node.func)}(...) 不在 with 的頭上"
            "——就算記得 close，中間任何一個例外都會讓那個檔案描述子留著；"
            "with 是唯一保證關得掉的寫法（只要讀寫整份就用 Path 的 read_text／write_text，"
            "它們自己關）"
        )
    return bad


# ── ④ 函式行數與分支數 ────────────────────────────────────────────────────


def _docstring_span(node: ast.AST) -> int:
    """這個函式自己的文件字串佔幾行。沒有文件字串就是 0。"""
    body = getattr(node, "body", [])
    first = body[0] if body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return int(first.value.end_lineno or first.value.lineno) - first.value.lineno + 1
    return 0


def _branch_count(node: ast.AST) -> int:
    """這個函式身體裡的分支節點數。巢狀的 def／class 不算進來（各自算一層）。"""
    total = 0
    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, SCOPE_NODES):
            continue
        if isinstance(child, BRANCH_NODES):
            total += 1
        stack.extend(ast.iter_child_nodes(child))
    return total


def _size_hits(tree: ast.Module, rel: str, settings: dict[str, object]) -> list[str]:
    max_lines = _threshold(settings, "max_function_lines")
    max_branches = _threshold(settings, "max_function_branches")
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_NODES):
            continue
        lines = int(node.end_lineno or node.lineno) - node.lineno + 1 - _docstring_span(node)
        if lines > max_lines:
            bad.append(
                f"{rel}:{node.lineno} 函式 {node.name}() 有 {lines} 行（不含文件字串），"
                f"超過卡 {CARD_ID} 登記的上限 {max_lines}"
                "——一支函式長到這樣，改它的人沒辦法一眼看完它在做什麼；切開它，"
                "或把門檻連著實測分布一起改（改門檻要走 PR）"
            )
        branches = _branch_count(node)
        if branches > max_branches:
            bad.append(
                f"{rel}:{node.lineno} 函式 {node.name}() 有 {branches} 個分支，"
                f"超過卡 {CARD_ID} 登記的上限 {max_branches}"
                "——分支多到這樣，測試不可能走完每一條路；切開它，"
                "或把門檻連著實測分布一起改（改門檻要走 PR）"
            )
    return bad


# ── 掃描面與主流程 ─────────────────────────────────────────────────────────


def _scanned_python(scan_root: Path, files: list[Path], exempt: list[str]) -> list[Path]:
    return sorted(
        f
        for f in files
        if f.suffix == PYTHON_SUFFIX
        and not any(f.relative_to(scan_root).as_posix().startswith(prefix) for prefix in exempt)
    )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀／會判的檔：所有規矩卡 ＋ 卡上沒被扣掉的每一支 .py。

    ``check()`` 自己也是叫這一支拿掃描面，所以 ``--list-files`` 印出來的清單就是真的被掃的
    那一組——不是第二份會各自漂的宣告。
    """
    settings = _card_settings(scan_root, files)
    python = _scanned_python(scan_root, files, _names(settings, "scan_exempt_prefixes"))
    return sorted({*_card_files(scan_root, files), *python})


def _note_output_layer(counts: dict[str, int], allowed: list[str]) -> None:
    """輸出層白名單這一跑的實況：誰在 print、幾處，以及哪幾筆今天沒有對象。"""
    live = sorted(f"{rel}（{count} 處）" for rel, count in counts.items() if count)
    if live:
        note(f"輸出層白名單這一跑放過 {live}——清單長大這件事要看得見")
    stale = sorted(set(allowed) - {rel for rel, count in counts.items() if count})
    if stale:
        note(f"卡上這幾筆輸出層放行這一跑沒有對象，可能該從卡上刪掉：{stale}")


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    note(f"ruff 在崗：{_ruff_version()}")
    picked = _scanned_python(scan_root, files, _names(settings, "scan_exempt_prefixes"))
    if not picked:
        raise ToolBroken(
            f"{scan_root} 底下一支要判的 .py 都沒有（扣掉卡上登記的前綴之後是空集合）"
            "——這一跑沒讀到任何程式，「沒問題」這句話不算數"
        )

    allowed = _allow_paths(settings)
    counts: dict[str, int] = {rel: 0 for rel in allowed}
    bad: list[str] = []
    for path in picked:
        rel = path.relative_to(scan_root).as_posix()
        if not path.is_file():
            # 版控認得、檔案系統上不在（剛被刪掉還沒 commit）。不猜內容，跳過。
            continue
        tree = _parse(path, rel)
        if rel in counts:
            counts[rel] = len(_print_calls(tree))
        else:
            bad += _print_hits(tree, rel)
        bad += _concat_hits(tree, rel, settings)
        bad += _open_hits(tree, rel, settings)
        bad += _size_hits(tree, rel, settings)

    _note_output_layer(counts, allowed)
    return sorted(bad)


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="寫法警衛：print 只准在輸出層、字串不准拼路徑、開檔一律 with、函式不准超過門檻",
            targets=targets,
        )
    )
