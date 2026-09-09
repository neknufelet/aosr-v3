#!/usr/bin/env python3
"""測試不准依賴環境現況，也不准寫進真的 repo（靜態那半）。

掃描面是 ``<scan_root>/tests/`` 底下所有進得了版控的 ``.py``——含 ``conftest.py``、
含子目錄、含測試用的輔助模組。真的用 :mod:`ast` 解析，不用正則。

**判準是「有沒有經過那支 fixture」，不是字串比對。** 草稿版寫的是「偵測參數裡含
``"git status"`` 的子程序呼叫」，實測抓不到：``subprocess.run(["git", "status",
"--porcelain"])`` 的字面值是 ``['git', 'status', '--porcelain']``，沒有任何一個含
``"git status"``。所以這裡改成「這個函式（或包著它的那一層）有沒有要唯一那支 sandbox
fixture」，fixture 的名字由卡宣告。

**什麼會紅**

1. **不經 fixture spawn 版控工具**——呼叫的第一個參數是「以 git 開頭的參數陣列」
   （``["git", ...]``、``"git status"``、``["/usr/bin/git", ...]``），而包著它的函式
   沒有要那支 fixture。參數陣列先指給同一個作用域裡的名字再餵下去（``ARGV = ["git",
   ...]`` 之後 ``subprocess.run(ARGV)``）一樣咬——那是零成本的繞法。
2. **不經 fixture 往真樹寫檔**——寫入類的呼叫（``write_text``／``mkdir``／``unlink``／
   ``open(..., "w")``／``shutil.copy``／``os.remove`` 這一類）的目標是從 ``__file__``
   或 ``Path.cwd()`` 算出來的真樹路徑（``REPO / "leftover.txt"``）。例外只能開在卡的
   ``[[settings.allow]]``，見下。
3. **同名 fixture 被定義第二次**——「唯一」是這條規矩的地基。有第二支同名 fixture，
   測試要到的就可能是把真樹交出來的那支，而判準看到的仍是「有用 fixture」。
4. **要了一支這棵樹裡不存在的 fixture**——形式上有沙盒，實際上沒有。
5. **拿家目錄當答案來源**——``Path.home()``、``os.path.expanduser``、``os.path.expandvars``。
   答案取決於「跑在誰的機器上」。這一條不看有沒有 fixture：家目錄從來不是這棵樹的一部分。
6. **放行本身壞掉**——放行指向的檔不在掃描面裡（過期的放行）、放行的路徑沒有被掃描根的
   ``.gitignore`` 蓋住（等於准你在版控樹裡留殘留）、或放行從頭到尾沒有對象（該拿掉了）。

**門檻只寫在卡上**

卡（宣告 ``check = "governance/checks/tests_isolated_from_real_env.py"`` 的那一張）的
``[settings]`` 給兩樣：``sandbox_fixture``（唯一那支 fixture 的名字）與選填的
``[[settings.allow]]``（寫入真樹的放行，一條一格：``file``／``function``／``path``／``why``）。
讀不到、找到多張、或形狀不對，一律 raise :class:`ToolBroken` 讓外殼回 2——這支檢查沒有
預設值，「沒設定就當乾淨」正是 v2 假綠的病根。

**動態那半不在這裡**

「跑完真 repo 的 ``git status --porcelain`` 必須跟跑前一樣」由這張卡帶進
``tests/conftest.py`` 的 session 守衛做（量測的呼叫收在 ``governance/repo_residue.py``），
接線由 ``tests/test_repo_residue_guard.py`` 咬。靜態這半咬不到的東西——子程序間接寫進
真樹、經過中間模組洗一手的 git 呼叫——由那半接住。

**沒掃到東西就回 2**

``tests/`` 底下一支 ``.py`` 都沒有、或有一支解不開，一律回 2。零命中回 0 的前提是真的
讀過檔；一支都沒讀到，「沒問題」這句話不算數。
"""
from __future__ import annotations

import ast
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from governance.exit_codes import ToolBroken, run
from governance.loader import RULES_DIR

# 這支檢查在卡裡的名字。門檻只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/tests_isolated_from_real_env.py"

TESTS_DIR = "tests"
GITIGNORE = ".gitignore"

SETTINGS_KEYS = ("sandbox_fixture", "allow")
ALLOW_KEY = "allow"
ALLOW_KEYS = ("file", "function", "path", "why")

# 版控工具的名字。參數陣列的第一格（去掉目錄）等於它，就是在 spawn 它。
GIT_TOOL = "git"

SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

# 寫入類的呼叫。只比對最後一段名字（``p.write_text``／``shutil.copy``／``os.remove``），
# 因為「目標是不是真樹路徑」才是判準——名字放寬不會誤咬，路徑不是真樹的一律不看。
WRITE_NAMES = frozenset(
    {
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "touch",
        "unlink",
        "remove",
        "rmdir",
        "removedirs",
        "rmtree",
        "rename",
        "replace",
        "symlink",
        "symlink_to",
        "hardlink_to",
        "link",
        "truncate",
        "chmod",
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "copyfileobj",
        "move",
        "make_archive",
        "unpack_archive",
        "writelines",
    }
)

# 兩格的那幾個：目標是最後一格（``shutil.copy(src, dst)`` 的來源是讀，不算違規）。
COPY_NAMES = frozenset(
    {"copy", "copy2", "copyfile", "copytree", "copyfileobj", "move", "make_archive", "unpack_archive"}
)
# 內容型的方法：參數是要寫的東西、不是路徑，所以只看接收者。
CONTENT_NAMES = frozenset({"write_text", "write_bytes", "writelines"})

# ``open`` 另外處理：要看模式字串才知道是讀還是寫。
OPEN_NAME = "open"
WRITE_MODES = "wax+"

# 家目錄類的呼叫。這三個沒有正當用法，一律咬（有沒有 fixture 都一樣）。
HOME_NAMES = frozenset({"home", "expanduser", "expandvars"})

# 真樹的種子：從這兩種東西算出來的路徑就是真樹裡的路徑。
FILE_SEED = "__file__"
CWD_NAMES = frozenset({"cwd", "getcwd"})

USEFIXTURES = "usefixtures"


@dataclass(frozen=True)
class Allow:
    """卡上的一條放行：哪支檔、哪個函式、寫到哪、為什麼。"""

    file: str
    function: str
    path: str
    why: str


@dataclass(frozen=True)
class Hit:
    """一筆違規（或一筆被放行擋掉的寫入）。"""

    kind: str
    rel: str
    lineno: int
    chain: tuple[str, ...]
    prefix: str | None
    message: str

    @property
    def where(self) -> str:
        return f"{self.rel}:{self.lineno}" + (f" 的 {'.'.join(self.chain)}()" if self.chain else " 模組層")


# ── 門檻：只從卡上讀 ────────────────────────────────────────────────────────
def _settings(scan_root: Path, files: list[Path]) -> tuple[str, tuple[Allow, ...]]:
    """從掃描根自己的 ``governance/rules/`` 讀這支檢查的門檻。

    找不到、找到多張、或形狀不對，一律 raise ToolBroken——沒有門檻就不出結論。
    """
    rules_dir = scan_root / RULES_DIR
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——門檻（唯一那支 fixture 的名字、寫入的放行清單）只寫在卡上，讀不到就不算數"
        )
    path, data = mine[0]
    rel = str(path.relative_to(scan_root))
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表——那支 fixture 的名字與放行清單都寫在那裡")
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        raise ToolBroken(f"{rel} 的 [settings] 多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}")
    name = settings.get("sandbox_fixture")
    if not isinstance(name, str) or not name.isidentifier():
        raise ToolBroken(f"{rel} 的 [settings] sandbox_fixture 必須是一個合法的 Python 名字，實際是 {name!r}")
    return name, _allow_list(settings, rel)


def _allow_list(settings: dict[str, object], rel: str) -> tuple[Allow, ...]:
    """讀 ``[[settings.allow]]``。沒寫就是沒有放行；寫了形狀就要對。"""
    raw = settings.get(ALLOW_KEY)
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw or not all(isinstance(x, dict) for x in raw):
        raise ToolBroken(f"{rel} 的 [[settings.allow]] 寫了就必須是非空的表陣列，實際是 {raw!r}")
    out: list[Allow] = []
    for index, entry in enumerate(raw):
        where = f"{rel} 的 [[settings.allow]] 第 {index + 1} 條"
        missing = [k for k in ALLOW_KEYS if k not in entry]
        if missing:
            raise ToolBroken(f"{where} 缺 {missing}——四格要寫滿：哪支檔、哪個函式、寫到哪、為什麼")
        extra = [k for k in entry if k not in ALLOW_KEYS]
        if extra:
            raise ToolBroken(f"{where} 多了不認識的鍵 {sorted(extra)}，只認 {list(ALLOW_KEYS)}")
        bad = [k for k in ALLOW_KEYS if not isinstance(entry[k], str) or not str(entry[k]).strip()]
        if bad:
            raise ToolBroken(f"{where} 的 {bad} 必須是非空字串")
        path = str(entry["path"])
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ToolBroken(f"{where} 的 path={path!r} 必須是掃描根底下的相對路徑，不准絕對路徑或 ..")
        out.append(
            Allow(
                file=str(entry["file"]).strip(),
                function=str(entry["function"]).strip(),
                path=path.strip("/"),
                why=str(entry["why"]).strip(),
            )
        )
    return tuple(out)


# ── .gitignore：放行的路徑必須被蓋住 ────────────────────────────────────────
def _ignored_prefixes(scan_root: Path) -> tuple[str, ...]:
    """掃描根那份 ``.gitignore`` 裡的路徑樣式（去掉註解、否定、前後斜線）。"""
    path = scan_root / GITIGNORE
    if not path.is_file():
        return ()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不開 {GITIGNORE}：{exc}") from exc
    out: list[str] = []
    for line in text.splitlines():
        row = line.strip()
        if not row or row.startswith("#") or row.startswith("!"):
            continue
        out.append(row.strip("/"))
    return tuple(out)


def _is_ignored(path: str, prefixes: tuple[str, ...]) -> bool:
    """這個路徑有沒有被那些樣式蓋住。只認整段相等或整段前綴，不解 glob。"""
    return any(path == p or path.startswith(p + "/") for p in prefixes)


# ── AST 小工具 ──────────────────────────────────────────────────────────────
def _callee(func: ast.expr) -> str:
    """把 ``x.write_text`` 攤成 ``"x.write_text"``。認不出來就回空字串。"""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _callee(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _last(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _own_statements(scope: ast.AST) -> list[ast.stmt]:
    """這一層自己的每一個 statement（走進 if／for／try／with，不走進巢狀的 def／class）。

    刻意照原始順序回傳：底下要靠這個順序把 ``REPO = …`` 先看到，再看到
    ``NOTES = REPO / "docs"``。順序倒過來的話後者就認不出是真樹路徑
    （實測踩過：樣本 case-allow-path-not-gitignored 的兩筆寫入曾經整個漏掉）。
    """
    out: list[ast.stmt] = []
    for stmt in getattr(scope, "body", []):
        out.append(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            for inner in getattr(stmt, field, None) or []:
                out += _own_statements(ast.Module(body=[inner], type_ignores=[]))
        for handler in getattr(stmt, "handlers", None) or []:
            out += _own_statements(ast.Module(body=handler.body, type_ignores=[]))
    return out


def _own_expressions(stmt: ast.stmt):
    """這一句自己的表達式（``if`` 的條件、``with`` 的 items、``for`` 的 iter、呼叫本身）。

    刻意不走進子句：子句裡的每一句已經由 :func:`_own_statements` 各自列出來一次，
    再走進去會把同一個呼叫算兩次（``hits`` 翻倍，控制樣本「正好 1 筆」的約定就破了），
    而且會拿外層的 guarded 去判包在 ``if`` 底下的巢狀 def——那支 def 明明要了 fixture，
    卻會被判違規。
    """
    for child in ast.iter_child_nodes(stmt):
        if isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)):
            continue
        yield from ast.walk(child)


def _str_const(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


def _names_git(expr: ast.expr, seqs: dict[str, list[ast.expr]]) -> bool:
    """這個表達式是不是「以版控工具開頭的參數」？"""
    if isinstance(expr, (ast.List, ast.Tuple)):
        if not expr.elts:
            return False
        first = expr.elts[0]
        if isinstance(first, ast.Starred):
            return _names_git(first.value, seqs)
        return _names_git(first, seqs)
    if isinstance(expr, ast.Name):
        return _names_git_seq(seqs[expr.id], seqs) if expr.id in seqs else False
    if isinstance(expr, ast.JoinedStr):
        return bool(expr.values) and _names_git(expr.values[0], seqs)
    text = _str_const(expr)
    if text is None:
        return False
    words = text.split()
    return bool(words) and Path(words[0]).name == GIT_TOOL


def _names_git_seq(elts: list[ast.expr], seqs: dict[str, list[ast.expr]]) -> bool:
    if not elts:
        return False
    first = elts[0]
    return _names_git(first.value if isinstance(first, ast.Starred) else first, seqs)


def _is_tree_expr(expr: ast.expr, anchors: dict[str, str | None]) -> bool:
    """這個表達式算不算「真樹裡的路徑」——從 ``__file__``、``Path.cwd()`` 或已知的錨算出來。"""
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and (node.id == FILE_SEED or node.id in anchors):
            return True
        if isinstance(node, ast.Call) and _last(_callee(node.func)) in CWD_NAMES:
            return True
    return False


def _prefix(expr: ast.expr, anchors: dict[str, str | None]) -> str | None:
    """算得出來就回這個路徑相對掃描根的前綴（``""`` 是樹根），算不出來回 ``None``。"""
    if isinstance(expr, ast.Name):
        return anchors.get(expr.id) if expr.id in anchors else ("" if expr.id == FILE_SEED else None)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        base = _prefix(expr.left, anchors)
        if base is None:
            return None
        text = _str_const(expr.right)
        return f"{base}/{text}".strip("/") if text else None
    if isinstance(expr, ast.Subscript):
        return _prefix(expr.value, anchors)
    if isinstance(expr, ast.Attribute):
        base = _prefix(expr.value, anchors)
        if expr.attr in ("parent", "parents"):
            # 算不出來的路徑，它的上一層一樣算不出來——不准回 ""（那是樹根，會把放行判成不適用）。
            if base is None:
                return None
            return base.rsplit("/", 1)[0] if "/" in base else ""
        return base
    if isinstance(expr, ast.Call):
        func = expr.func
        if isinstance(func, ast.Attribute):
            if _last(_callee(func)) in CWD_NAMES:
                return ""
            return _prefix(func.value, anchors)
        if any(isinstance(n, ast.Name) and n.id == FILE_SEED for n in ast.walk(expr)):
            return ""
        return None
    return None


def _write_targets(call: ast.Call, anchors: dict[str, str | None]) -> list[ast.expr]:
    """這個呼叫在寫檔的話，回它可能寫到的那幾個路徑表達式；不是寫檔就回空 list。

    分三種形狀，因為「目標在哪一格」不一樣：

    * ``p.write_text(...)``／``p.mkdir()``——目標是接收者。
    * ``shutil.copy(src, dst)``／``shutil.move(...)``——目標是**最後**那一格（來源是讀，
      咬它就是誤咬）。
    * ``os.remove(p)``／``shutil.rmtree(p)``／``os.mkdir(p)``——目標在參數裡。接收者
      （``os``／``shutil``）不是路徑，所以接收者不是真樹路徑的時候要往參數看。
      內容型的方法（``write_text`` 這一類）刻意不看參數：``(tmp / "f").write_text(str(REPO))``
      寫的是暫存檔，內容裡有真樹路徑不算違規。
    """
    name = _last(_callee(call.func))
    if name == OPEN_NAME:
        if isinstance(call.func, ast.Attribute):
            mode = call.args[0] if call.args else _keyword(call, "mode")
            target: ast.expr | None = call.func.value
        else:
            mode = call.args[1] if len(call.args) > 1 else _keyword(call, "mode")
            target = call.args[0] if call.args else None
        text = _str_const(mode) if mode is not None else None
        if text is None or not any(ch in text for ch in WRITE_MODES):
            return []
        return [target] if target is not None else []
    if name not in WRITE_NAMES:
        return []
    if isinstance(call.func, ast.Attribute) and _is_tree_expr(call.func.value, anchors):
        return [call.func.value]
    if name in CONTENT_NAMES:
        return []
    args = [a for a in call.args if not isinstance(a, ast.Starred)]
    if name in COPY_NAMES:
        return args[-1:]
    return args


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = node.args
    got = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return got


def _asks_for(node: ast.FunctionDef | ast.AsyncFunctionDef, fixture: str) -> bool:
    """這個函式有沒有要那支 fixture——參數列，或 ``@pytest.mark.usefixtures("…")``。"""
    if fixture in _params(node):
        return True
    for deco in node.decorator_list:
        if isinstance(deco, ast.Call) and _last(_callee(deco.func)) == USEFIXTURES:
            if any(_str_const(a) == fixture for a in deco.args):
                return True
    return False


# ── 一支檔 ──────────────────────────────────────────────────────────────────
def _parse(path: Path, rel: str) -> ast.Module:
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc
    try:
        return ast.parse(src, filename=rel)
    except SyntaxError as exc:
        raise ToolBroken(f"{rel} 第 {exc.lineno} 行解不開（{exc.msg}）——我沒看懂就不出結論") from exc


@dataclass
class FileFacts:
    """一支檔看完之後手上有的東西。"""

    hits: list[Hit]
    definitions: list[tuple[str, int]]
    requesters: list[tuple[str, int]]


def _scan_scope(
    scope: ast.AST,
    rel: str,
    chain: tuple[str, ...],
    guarded: bool,
    anchors: dict[str, str | None],
    seqs: dict[str, list[ast.expr]],
    fixture: str,
    facts: FileFacts,
) -> None:
    """一個作用域：先把錨與參數陣列的名字收出來，再看每一個呼叫。"""
    anchors = dict(anchors)
    seqs = dict(seqs)
    statements = _own_statements(scope)
    # 跑到穩定為止：``X = REPO / "a"`` 要等 ``REPO`` 先進錨點清單才認得出來，
    # 而指定的順序不保證（先用後定義的寫法也存在）。
    for _round in range(len(statements) + 1):
        before = (len(anchors), len(seqs))
        for stmt in statements:
            targets: list[ast.Name] = []
            value: ast.expr | None = None
            if isinstance(stmt, ast.Assign):
                targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
                value = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                targets = [stmt.target]
                value = stmt.value
            if value is None:
                continue
            for target in targets:
                if isinstance(value, (ast.List, ast.Tuple)):
                    seqs[target.id] = list(value.elts)
                if _is_tree_expr(value, anchors):
                    anchors[target.id] = _prefix(value, anchors)
        if (len(anchors), len(seqs)) == before:
            break

    for stmt in statements:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name == fixture:
                facts.definitions.append((rel, stmt.lineno))
            if fixture in _params(stmt):
                facts.requesters.append((rel, stmt.lineno))
            _scan_scope(
                stmt,
                rel,
                (*chain, stmt.name),
                guarded or stmt.name == fixture or _asks_for(stmt, fixture),
                anchors,
                seqs,
                fixture,
                facts,
            )
            continue
        if isinstance(stmt, ast.ClassDef):
            _scan_scope(stmt, rel, (*chain, stmt.name), guarded, anchors, seqs, fixture, facts)
            continue
        for node in _own_expressions(stmt):
            if isinstance(node, ast.Call):
                _look_at_call(node, rel, chain, guarded, anchors, seqs, fixture, facts)


def _look_at_call(
    call: ast.Call,
    rel: str,
    chain: tuple[str, ...],
    guarded: bool,
    anchors: dict[str, str | None],
    seqs: dict[str, list[ast.expr]],
    fixture: str,
    facts: FileFacts,
) -> None:
    name = _last(_callee(call.func))
    if name in HOME_NAMES:
        facts.hits.append(
            Hit(
                kind="home",
                rel=rel,
                lineno=call.lineno,
                chain=chain,
                prefix=None,
                message=(
                    f"拿家目錄當答案來源：{ast.unparse(call)[:80]}"
                    "——答案取決於跑在誰的機器上，換一台就換一個結果。要什麼路徑就自己造，"
                    f"或跟 {fixture} 要暫存樹"
                ),
            )
        )
    first = call.args[0] if call.args else _keyword(call, "args")
    if first is not None and _names_git(first, seqs) and not guarded:
        facts.hits.append(
            Hit(
                kind="spawn",
                rel=rel,
                lineno=call.lineno,
                chain=chain,
                prefix=None,
                message=(
                    f"不經 {fixture} 就 spawn 版控工具：{ast.unparse(call)[:100]}"
                    f"——會執行版控工具的測試只能經過唯一那支 {fixture} fixture（它會把 GIT_DIR／"
                    "GIT_WORK_TREE／GIT_INDEX_FILE 指到暫存樹，並在寫入前做負控制）。"
                    "v2 事故 worktree-hook-writes-into-real-repo：hook 底下的測試就是這樣把兩顆"
                    "fixture commit 寫回真 repo 的"
                ),
            )
        )
    target = next((t for t in _write_targets(call, anchors) if _is_tree_expr(t, anchors)), None)
    if target is not None and not guarded:
        prefix = _prefix(target, anchors)
        shown = prefix if prefix else "（算不出字面路徑）"
        facts.hits.append(
            Hit(
                kind="write",
                rel=rel,
                lineno=call.lineno,
                chain=chain,
                prefix=prefix,
                message=(
                    f"不經 {fixture} 就往真樹寫檔：{ast.unparse(call)[:100]}（寫到 {shown}）"
                    f"——測試跑完不准在版控樹裡留東西。要寫就寫在 {fixture} 給的暫存樹，"
                    "真的非寫真樹不可就在卡的 [[settings.allow]] 開一條，說明寫到哪、為什麼，"
                    "而且那個路徑要被 .gitignore 蓋住"
                ),
            )
        )


# ── 主流程 ──────────────────────────────────────────────────────────────────
def _allow_problems(
    allow: tuple[Allow, ...], scanned: set[str], ignored: tuple[str, ...]
) -> tuple[list[str], list[Allow]]:
    """放行本身有沒有壞。回（問題清單，還能用的放行）。"""
    bad: list[str] = []
    usable: list[Allow] = []
    for entry in allow:
        problems: list[str] = []
        if entry.file not in scanned:
            problems.append(
                f"放行指向的 {entry.file} 不在掃描面裡（{TESTS_DIR}/ 底下進得了版控的 .py）"
                "——過期的放行就是一張沒人看的豁免，該拿掉"
            )
        if not _is_ignored(entry.path, ignored):
            problems.append(
                f"放行的路徑 {entry.path!r} 沒有被掃描根的 {GITIGNORE} 蓋住"
                "——寫入真樹的放行只准開給跑完不會留在版控裡的路徑，"
                "不然這條放行等於「准你在版控樹裡留殘留」"
            )
        if problems:
            bad += [f"卡上的放行（{entry.file} 的 {entry.function}()）：{p}" for p in problems]
            continue
        usable.append(entry)
    return bad, usable


def _matches(entry: Allow, hit: Hit) -> bool:
    """這條放行擋不擋得住這一筆寫入。只擋寫入，不擋 spawn。"""
    if hit.kind != "write" or hit.rel != entry.file or entry.function not in hit.chain:
        return False
    return hit.prefix is None or hit.prefix == entry.path or hit.prefix.startswith(entry.path + "/")


def check(scan_root: Path, files: list[Path]) -> list[str]:
    fixture, allow = _settings(scan_root, files)
    base = scan_root / TESTS_DIR
    targets = sorted(f for f in files if f.suffix == ".py" and f.is_relative_to(base))
    if not targets:
        raise ToolBroken(
            f"{scan_root}/{TESTS_DIR} 底下一支版控裡的 .py 都沒有"
            "——這一跑沒讀到任何測試檔，「沒問題」這句話不算數"
        )

    facts = FileFacts(hits=[], definitions=[], requesters=[])
    scanned: set[str] = set()
    for path in targets:
        rel = str(path.relative_to(scan_root).as_posix())
        scanned.add(rel)
        _scan_scope(_parse(path, rel), rel, (), False, {}, {}, fixture, facts)

    bad, usable = _allow_problems(allow, scanned, _ignored_prefixes(scan_root))
    used: set[int] = set()
    for hit in facts.hits:
        keep = [i for i, entry in enumerate(usable) if _matches(entry, hit)]
        if keep:
            used.update(keep)
            continue
        bad.append(f"{hit.where} {hit.message}")
    for index, entry in enumerate(usable):
        if index not in used:
            bad.append(
                f"卡上的放行（{entry.file} 的 {entry.function}()，寫到 {entry.path}）從頭到尾沒有對象"
                "——那個函式裡沒有任何一筆會被擋掉的寫入，這條放行該拿掉了"
            )

    if len(facts.definitions) > 1:
        where = "、".join(f"{rel}:{line}" for rel, line in facts.definitions)
        bad.append(
            f"{fixture} 被定義了不只一次（{where}）——「唯一」是這條規矩的地基：有第二支同名"
            "fixture，測試要到的就可能是把真樹交出來的那支，而判準看到的仍是「有用 fixture」，"
            "綠燈就掛在正在碰真樹的測試上"
        )
    if facts.requesters and not facts.definitions:
        where = "、".join(f"{rel}:{line}" for rel, line in facts.requesters)
        bad.append(
            f"有函式要了 {fixture}（{where}），但這棵樹底下沒有人定義它"
            "——形式上經過 fixture，實際上沒有沙盒"
        )
    return sorted(bad)


if __name__ == "__main__":
    sys.exit(run(check, description="測試不准依賴環境現況，也不准寫進真的 repo"))
